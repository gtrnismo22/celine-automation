import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from pedalboard import load_plugin
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStyleFactory,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


CHUNK_SIZE = 65536
SETUP_FILENAME = "celine_setup.json"


def find_celine_vst3():
    if platform.system() == "Windows":
        bundles = [
            Path(os.environ.get("CommonProgramFiles", r"C:\Program Files\Common Files")) / "VST3" / "Celine.vst3",
            Path(os.environ.get("CommonProgramW6432", r"C:\Program Files")) / "Common Files" / "VST3" / "Celine.vst3",
        ]
        candidates = [
            bundle / "Contents" / "x86_64-win" / "Celine.vst3"
            for bundle in bundles
        ] + bundles
    elif platform.system() == "Darwin":
        candidates = [
            Path("/Library/Audio/Plug-Ins/VST3/Celine.vst3"),
            Path.home() / "Library/Audio/Plug-Ins/VST3/Celine.vst3",
        ]
    else:
        candidates = []
    return next((candidate for candidate in candidates if candidate.is_file() or candidate.is_dir()), None)


OUTPUT_SAMPLE_RATE = 48000
NAM_KNOB_MIN = 0.0
NAM_KNOB_MAX = 10.0
CELINE_INPUT_DB_MIN = -20.0
CELINE_INPUT_DB_MAX = 20.0


def center_celine_editor_window(stop_event, excluded_window=None):
    if platform.system() != "Windows":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    moved_window = False
    baseline_windows = set()

    def collect_window(window_handle, _):
        baseline_windows.add(int(window_handle))
        return True

    user32.EnumWindows(
        ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(
            collect_window
        ),
        0,
    )

    while not stop_event.is_set() and not moved_window:
        candidate_windows = []

        def find_candidate(window_handle, _):
            if (
                user32.IsWindowVisible(window_handle)
                and window_handle != excluded_window
                and int(window_handle) not in baseline_windows
            ):
                candidate_windows.append(window_handle)
            return True

        user32.EnumWindows(
            ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(
                find_candidate
            ),
            0,
        )
        foreground_window = user32.GetForegroundWindow()
        if (
            foreground_window
            and foreground_window != excluded_window
            and user32.IsWindowVisible(foreground_window)
        ):
            candidate_windows.insert(0, foreground_window)

        for window_handle in candidate_windows:
            if window_handle == excluded_window:
                continue
            rect = wintypes.RECT()
            if user32.GetWindowRect(window_handle, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                screen_width = user32.GetSystemMetrics(0)
                screen_height = user32.GetSystemMetrics(1)
                x = max(0, (screen_width - width) // 2)
                y = max(0, (screen_height - height) // 2)
                user32.MoveWindow(window_handle, x, y, width, height, True)
                moved_window = True
                break
        if not moved_window:
            time.sleep(0.05)


def load_audio(path):
    audio_data, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(
            f"{path} is {sample_rate} Hz; capture inputs must be {OUTPUT_SAMPLE_RATE} Hz."
        )
    if audio_data.ndim == 1:
        audio_data = np.vstack([audio_data, audio_data])
    elif audio_data.shape[0] != 2 and audio_data.shape[1] == 2:
        audio_data = audio_data.T
    if audio_data.ndim != 2:
        raise ValueError(f"{path} must contain mono or stereo audio.")
    return audio_data, sample_rate


def prepare_capture_output(output):
    if output.ndim == 1:
        return output
    if output.ndim != 2:
        raise ValueError("Celine returned audio with an unsupported channel layout.")
    return np.mean(output, axis=0, dtype=np.float32)


def is_valid_capture(path):
    try:
        info = sf.info(path)
    except RuntimeError:
        return False
    return (
        info.samplerate == OUTPUT_SAMPLE_RATE
        and info.channels == 1
        and info.subtype == "PCM_24"
    )


def nam_value_to_celine_input_db(value):
    normalized = (value - NAM_KNOB_MIN) / (NAM_KNOB_MAX - NAM_KNOB_MIN)
    return CELINE_INPUT_DB_MIN + normalized * (
        CELINE_INPUT_DB_MAX - CELINE_INPUT_DB_MIN
    )


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def set_celine_parameter(parameter_name, parameter, value, is_nam_value=False):
    if parameter_name.casefold() == "input_db":
        actual_value = (
            nam_value_to_celine_input_db(value) if is_nam_value else value
        )
        parameter.value = actual_value
    else:
        normalized_value = (
            (value - NAM_KNOB_MIN) / (NAM_KNOB_MAX - NAM_KNOB_MIN)
            if is_nam_value
            else value
        )
        parameter.raw_value = normalized_value


def process_audio_in_chunks(plugin, audio, sample_rate, progress_callback, stop_event):
    output_chunks = []
    total_chunks = (audio.shape[1] + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_number, start in enumerate(range(0, audio.shape[1], CHUNK_SIZE), start=1):
        if stop_event.is_set():
            return None
        end = min(start + CHUNK_SIZE, audio.shape[1])
        output_chunks.append(
            plugin(audio[:, start:end], sample_rate=sample_rate, reset=False)
        )
        progress_callback(chunk_number, total_chunks)
    return np.hstack(output_chunks)


def write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=4)
        file.write("\n")
    temporary_path.replace(path)


def create_data_json(project):
    data = {
        "type": "parametric",
        "common": {"delay": 0},
        "train": [],
        "validation": [],
    }
    for entry in project["entries"]:
        split = entry["split"]
        window = project["train_window"] if split == "train" else project["validation_window"]
        data[split].append(
            {
                "x_path": project["train_input"] if split == "train" else project["validation_input"],
                "y_path": entry["y_path"],
                "delay": 0,
                "params": entry["params"],
                "start_seconds": window["start_seconds"],
                "stop_seconds": window["stop_seconds"],
                "ny": window["ny"],
            }
        )
    return data


def capture_metadata(output):
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    peak_db = float(20 * np.log10(peak)) if peak > 0 else None
    return {
        "peak": peak_db,
        "clipping": bool(peak >= 1.0),
        "impulse_detected": None,
        "delay_disagreement": False,
        "loopback_disagreement": False,
        "loopback_delay": None,
        "amp_return_delay": None,
        "subsample_shift": None,
        "peak_delay": None,
        "blip_delays": [],
        "messages": ["Rendered by Celine pedalboard automation; no hardware QA was performed."],
    }


class WorkerSignals(QObject):
    status = Signal(str)
    eta = Signal(str)
    capture_progress = Signal(int, int, int, str)
    capture_finished = Signal(int)
    stopped = Signal(str)
    finished = Signal(str)
    error = Signal(str)


class CelineWorker(QObject):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.signals = WorkerSignals()
        self.stop_event = threading.Event()

    def run(self):
        try:
            self._render()
        except Exception as error:
            self.signals.error.emit(str(error))

    def _render(self):
        plugin, project_dir, project, assignments = self.settings
        project["sample_rate"] = OUTPUT_SAMPLE_RATE
        entries = project["entries"]
        total = len(entries)
        completed = 0
        rendered_completed = 0
        pending_entries = [
            entry
            for entry in entries
            if not (
                (project_dir / entry["y_path"].replace("/", os.sep)).is_file()
                and is_valid_capture(project_dir / entry["y_path"].replace("/", os.sep))
            )
        ]
        pending_total = len(pending_entries)
        render_durations = []
        self.signals.eta.emit(
            "Estimated time remaining: calculating..."
            if pending_total
            else "Estimated time remaining: 0:00"
        )
        project_path = project_dir / "capture_project.json"
        data_path = project_dir / "data.json"
        data = create_data_json(project)
        for row, entry in enumerate(entries):
            if self.stop_event.is_set():
                self.signals.stopped.emit("Rendering stopped. Start again to resume.")
                return
            output_path = project_dir / entry["y_path"].replace("/", os.sep)
            if output_path.is_file() and is_valid_capture(output_path):
                entry["status"] = "captured"
                completed += 1
                self.signals.capture_finished.emit(row)
                continue
            split = entry["split"]
            input_name = (
                project["train_input"]
                if split == "train"
                else project["validation_input"]
            )
            input_path = project_dir / input_name
            audio, sample_rate = load_audio(input_path)
            for assignment in assignments:
                if assignment["mode"] == "capture":
                    knob = next(
                        knob
                        for knob in project["knobs"]
                        if knob["name"] == assignment["project_knob"]
                    )
                    value = entry["params"][knob["name"]]
                    project_range = knob["max"] - knob["min"]
                    nam_value = (
                        (value - knob["min"]) / project_range * NAM_KNOB_MAX
                    )
                    set_celine_parameter(
                        assignment["parameter"],
                        plugin.parameters[assignment["parameter"]],
                        nam_value,
                        is_nam_value=True,
                    )
                else:
                    set_celine_parameter(
                        assignment["parameter"],
                        plugin.parameters[assignment["parameter"]],
                        assignment["value"],
                    )
            capture_started = time.monotonic()
            completed += 1
            rendered_completed += 1

            def update_progress(done, chunks, completed=completed):
                self.signals.status.emit(
                    f"Rendering {completed}/{total} ({split})... chunk {done}/{chunks}"
                )
                if render_durations:
                    current_fraction = done / chunks if chunks else 0
                    current_elapsed = time.monotonic() - capture_started
                    current_estimate = (
                        current_elapsed / current_fraction
                        if current_fraction > 0
                        else 0
                    )
                    average_duration = sum(render_durations) / len(render_durations)
                    remaining = max(0, pending_total - rendered_completed)
                    seconds = round(remaining * average_duration + max(
                        0, current_estimate - current_elapsed
                    ))
                    self.signals.eta.emit(
                        f"Estimated time remaining: {format_duration(seconds)}"
                    )
                self.signals.capture_progress.emit(
                    row, done, chunks, f"{split.title()} {done}/{chunks}"
                )

            output = process_audio_in_chunks(
                plugin, audio, sample_rate, update_progress, self.stop_event
            )
            if output is None:
                self.signals.stopped.emit("Rendering stopped. Start again to resume.")
                return
            output = prepare_capture_output(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                output,
                OUTPUT_SAMPLE_RATE,
                subtype="PCM_24",
            )
            render_durations.append(time.monotonic() - capture_started)
            remaining = max(0, pending_total - rendered_completed)
            average_duration = sum(render_durations) / len(render_durations)
            self.signals.eta.emit(
                f"Estimated time remaining: {format_duration(round(remaining * average_duration))}"
            )
            entry["status"] = "captured"
            entry["captured_at"] = datetime.now(timezone.utc).isoformat()
            entry["delay"] = 0
            entry["qa"] = capture_metadata(output)
            entry["stream_config"] = "Celine VST3 via pedalboard"
            write_json(project_path, project)
            data_entry = next(
                item for item in data[split] if item["y_path"] == entry["y_path"]
            )
            data_entry["captured_at"] = entry["captured_at"]
            data_entry["status"] = entry["status"]
            data_entry["qa"] = entry["qa"]
            write_json(data_path, data)
            self.signals.capture_finished.emit(row)
        self.signals.finished.emit(f"Finished {total} capture(s).")

class CelineAutomationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Celine Capture Automation")
        self.resize(980, 700)
        self.setMinimumSize(820, 560)
        self.threads = []
        self.workers = []
        self.active_worker = None
        self.project_dir = None
        self.project = None
        self.plugin = None
        self.plugin_parameters = []
        self.saved_setup = None
        self._updating_setup_table = False
        detected_vst = find_celine_vst3()
        self.vst_path = str(detected_vst) if detected_vst else None
        self.project_summary = QLabel("No project open.")
        self.input_status = QLabel("inputTrunc.wav: no project open")
        self.validation_status = QLabel("validation.wav: no project open")
        self.status = QLabel("Open a project directory to begin.")
        self.eta_status = QLabel("Estimated time remaining: --")
        self.capture_table = QTableWidget()
        self.capture_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.capture_table.setAlternatingRowColors(True)
        self.capture_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.capture_table.setSortingEnabled(True)
        self.capture_table.horizontalHeader().setStretchLastSection(True)
        self._set_normal_header_font(self.capture_table)
        self.knob_table = QTableWidget()
        self.knob_table.setColumnCount(4)
        self.knob_table.setHorizontalHeaderLabels(
            ["Celine parameter", "Captured", "Capture knob", "Static value (0–10)"]
        )
        self.knob_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.knob_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._set_normal_header_font(self.knob_table)
        self._build_ui()

    @staticmethod
    def _set_normal_header_font(table):
        header_font = table.horizontalHeader().font()
        header_font.setBold(False)
        table.horizontalHeader().setFont(header_font)
        table.horizontalHeader().setStyleSheet(
            "QHeaderView::section, QTableWidget QHeaderView::section "
            "{ font-weight: 400; }"
        )

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        title = QLabel("Celine Capture Automation")
        layout.addWidget(title)
        layout.addWidget(QLabel("Create parameter sweeps from Celine and render WAV captures."))

        project_actions = QHBoxLayout()
        open_project_button = QPushButton("Open Project")
        open_project_button.setDefault(True)
        open_project_button.clicked.connect(self._open_project)
        project_actions.addWidget(open_project_button)
        project_actions.addWidget(self.project_summary, 1)
        layout.addLayout(project_actions)

        source = QGroupBox("Project audio")
        source_layout = QFormLayout(source)
        source_layout.addRow("Training input", self.input_status)
        source_layout.addRow("Validation input", self.validation_status)
        layout.addWidget(source)

        setup = QGroupBox("Celine setup")
        setup_layout = QVBoxLayout(setup)
        setup_actions = QHBoxLayout()
        self.setup_button = QPushButton("Set Up Celine")
        self.setup_button.clicked.connect(self._setup_celine)
        setup_actions.addWidget(self.setup_button)
        self.reset_setup_button = QPushButton("Reset Celine Setup")
        self.reset_setup_button.clicked.connect(self._reset_celine_setup)
        setup_actions.addWidget(self.reset_setup_button)
        setup_actions.addWidget(QLabel("Available plugin knobs"))
        self.knob_count = QSpinBox()
        self.knob_count.setRange(0, 64)
        self.knob_count.setValue(0)
        self.knob_count.valueChanged.connect(self._limit_knob_rows)
        setup_actions.addWidget(self.knob_count)
        setup_actions.addStretch()
        setup_layout.addLayout(setup_actions)
        setup_layout.addWidget(
            QLabel(
                "Set Up Celine opens the native editor first. After closing it, "
                "choose which parameters are captured or held static."
            )
        )
        setup_layout.addWidget(self.knob_table)
        layout.addWidget(setup)

        parameters = QGroupBox("Celine parameters")
        parameters_layout = QVBoxLayout(parameters)
        parameters_layout.addWidget(
            QLabel("Each row is a planned capture read from capture_project.json.")
        )
        parameters_layout.addWidget(self.capture_table)
        layout.addWidget(parameters, 1)

        actions = QHBoxLayout()
        actions.addStretch()
        self.render_button = QPushButton("Render Captures")
        self.render_button.setDefault(True)
        self.render_button.clicked.connect(self._start_render)
        self.render_button.setEnabled(False)
        actions.addWidget(self.render_button)
        self.reset_button = QPushButton("Reset Captures")
        self.reset_button.clicked.connect(self._reset_captures)
        actions.addWidget(self.reset_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_render)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        layout.addWidget(self.eta_status)
        self.setCentralWidget(central)

    def _path_widget(self, line_edit, callback):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit)
        button = QPushButton("Browse...")
        button.clicked.connect(callback)
        row.addWidget(button)
        return widget

    def _open_project(self):
        path = QFileDialog.getExistingDirectory(self, "Open capture project directory")
        if not path:
            return
        project_dir = Path(path)
        project_path = project_dir / "capture_project.json"
        if not project_path.is_file():
            QMessageBox.critical(
                self,
                "Invalid project",
                "The selected directory is missing capture_project.json.",
            )
            return
        try:
            with project_path.open(encoding="utf-8") as file:
                project = json.load(file)
            self._validate_project(project)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            QMessageBox.critical(self, "Invalid project", str(error))
            return
        self.project_dir = project_dir
        self.project = project
        self.saved_setup = None
        self._update_audio_status()
        setup_path = project_dir / SETUP_FILENAME
        if setup_path.is_file():
            try:
                self.saved_setup = json.loads(setup_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "Setup file unreadable", str(error))
        self._show_capture_table(project)
        if self.plugin_parameters:
            if self.saved_setup:
                self._populate_knob_table()
                self._apply_saved_setup()
            else:
                previous = (
                    {
                        item.get("parameter"): {
                            "captured": bool(item.get("captured", False)),
                            "capture_knob": item.get("capture_knob", ""),
                            "static_value": float(item.get("static_value", 5.0)),
                        }
                        for item in self.saved_setup.get("parameters", [])
                    }
                    if self.saved_setup
                    else self._collect_setup_parameters()
                )
                self._populate_knob_table(self.knob_count.value(), previous)
        self.render_button.setEnabled(self.plugin is not None and self._inputs_available())
        data_path = project_dir / "data.json"
        if not data_path.exists():
            write_json(data_path, create_data_json(project))
        train_count = sum(entry["split"] == "train" for entry in project["entries"])
        validation_count = sum(entry["split"] == "validation" for entry in project["entries"])
        self.project_summary.setText(
            f"{project.get('name', project_dir.name)}  |  "
            f"{len(project['knobs'])} knobs  |  "
            f"{train_count} train + {validation_count} validation captures"
        )
        self.status.setText(f"Opened project: {project_dir}")

    def _update_audio_status(self):
        if self.project_dir is None:
            return
        train_path = self.project_dir / "inputTrunc.wav"
        validation_path = self.project_dir / "validation.wav"
        self.input_status.setText(
            "inputTrunc.wav: found" if train_path.is_file() else "inputTrunc.wav: missing"
        )
        self.validation_status.setText(
            "validation.wav: found" if validation_path.is_file() else "validation.wav: missing"
        )

    def _inputs_available(self):
        return (
            self.project_dir is not None
            and (self.project_dir / "inputTrunc.wav").is_file()
            and (self.project_dir / "validation.wav").is_file()
        )

    @staticmethod
    def _validate_project(project):
        required = ("knobs", "entries", "train_input", "validation_input")
        missing = [name for name in required if name not in project]
        if missing:
            raise ValueError("capture_project.json is missing: " + ", ".join(missing))
        if not project["knobs"]:
            raise ValueError("capture_project.json does not define any knobs.")
        knob_names = []
        for knob in project["knobs"]:
            for key in ("name", "min", "max", "step"):
                if key not in knob:
                    raise ValueError(f"Knob definition is missing '{key}'.")
            if knob["max"] <= knob["min"]:
                raise ValueError(f"Knob '{knob['name']}' has an invalid range.")
            knob_names.append(knob["name"])
        for entry in project["entries"]:
            if entry.get("split") not in ("train", "validation"):
                raise ValueError("Each capture entry must use train or validation split.")
            if "y_path" not in entry or not isinstance(entry.get("params"), dict):
                raise ValueError("Each capture entry needs y_path and params.")
            missing = [name for name in knob_names if name not in entry["params"]]
            if missing:
                raise ValueError(
                    f"Capture entry {entry.get('index', '?')} is missing: {', '.join(missing)}"
                )

    def _show_capture_table(self, project):
        knob_names = [knob["name"] for knob in project["knobs"]]
        headers = ["Index", "Split", *knob_names, "Status", "Output"]
        self.capture_table.setSortingEnabled(False)
        self.capture_table.clear()
        self.capture_table.setColumnCount(len(headers))
        self.capture_table.setHorizontalHeaderLabels(headers)
        self._set_normal_header_font(self.capture_table)
        self.capture_table.setRowCount(len(project["entries"]))
        for row, entry in enumerate(project["entries"]):
            values = [
                str(entry["index"]),
                entry["split"],
                *[f"{entry['params'][name]:g}" for name in knob_names],
                entry.get("status", "pending"),
                entry["y_path"],
            ]
            for column, value in enumerate(values):
                if column == len(headers) - 2:
                    progress = QProgressBar()
                    progress.setRange(0, 100)
                    progress.setValue(100 if entry.get("status") == "captured" else 0)
                    progress.setFormat(entry.get("status", "Pending").title())
                    self.capture_table.setCellWidget(row, column, progress)
                else:
                    self.capture_table.setItem(row, column, QTableWidgetItem(value))
        self.capture_table.resizeColumnsToContents()
        self.capture_table.setSortingEnabled(True)

    def _setup_celine(self):
        if not self.vst_path:
            QMessageBox.warning(self, "Celine not found", "Install Celine.vst3 in the default VST3 directory.")
            return
        if not Path(self.vst_path).exists():
            QMessageBox.warning(self, "Celine not found", "Install Celine.vst3 in the default VST3 directory.")
            return
        previous = self._collect_setup_parameters()
        self.setup_button.setEnabled(False)
        self.status.setText("Loading Celine and opening its editor...")
        try:
            self.plugin = load_plugin(self.vst_path)
            editor_position_stop = threading.Event()
            editor_position_thread = threading.Thread(
                target=center_celine_editor_window,
                args=(editor_position_stop, int(self.winId())),
                daemon=True,
            )
            editor_position_thread.start()
            try:
                self.plugin.show_editor(threading.Event())
            finally:
                editor_position_stop.set()
            self.plugin_parameters = list(self.plugin.parameters)
            plugin_defaults = (
                self._read_plugin_defaults() if self.saved_setup is None else {}
            )
            self.knob_count.blockSignals(True)
            self.knob_count.setValue(min(len(self.plugin_parameters), self.knob_count.value() or len(self.plugin_parameters)))
            self.knob_count.blockSignals(False)
            if self.saved_setup:
                self._apply_saved_setup()
            elif previous:
                project_knobs = [
                    knob["name"] for knob in (self.project or {}).get("knobs", [])
                ]
                for parameter, static_value in plugin_defaults.items():
                    if parameter in previous:
                        previous[parameter]["static_value"] = static_value
                    else:
                        matching_knob = next(
                            (
                                knob_name
                                for knob_name in project_knobs
                                if knob_name.casefold() == parameter.casefold()
                            ),
                            None,
                        )
                        previous[parameter] = {
                            "captured": matching_knob is not None,
                            "capture_knob": matching_knob or "",
                            "static_value": static_value,
                        }
                self._populate_knob_table(self.knob_count.value(), previous)
            else:
                defaults = {}
                project_knobs = [
                    knob["name"] for knob in (self.project or {}).get("knobs", [])
                ]
                for parameter in self.plugin_parameters:
                    matching_knob = next(
                        (
                            knob_name
                            for knob_name in project_knobs
                            if knob_name.casefold() == parameter.casefold()
                        ),
                        None,
                    )
                    defaults[parameter] = {
                        "captured": matching_knob is not None,
                        "capture_knob": matching_knob or "",
                        "static_value": plugin_defaults.get(parameter, 5.0),
                    }
                self._populate_knob_table(self.knob_count.value(), defaults)
            self.render_button.setEnabled(self._inputs_available())
            self._save_setup()
            self.status.setText(
                "Celine setup loaded. Assign capture knobs and static values, then render."
            )
        except Exception as error:
            self.plugin = None
            QMessageBox.critical(self, "Celine setup failed", str(error))
            self.status.setText("Celine setup failed.")
        finally:
            self.setup_button.setEnabled(True)

    def _read_plugin_defaults(self):
        defaults = {}
        for parameter_name in self.plugin_parameters:
            parameter = self.plugin.parameters[parameter_name]
            raw_value = float(parameter.raw_value)
            if parameter_name.casefold() == "input_db":
                minimum, maximum, *_ = parameter.range
                actual_value = float(minimum + raw_value * (maximum - minimum))
                static_value = (
                    actual_value - CELINE_INPUT_DB_MIN
                ) / (CELINE_INPUT_DB_MAX - CELINE_INPUT_DB_MIN) * NAM_KNOB_MAX
            else:
                static_value = raw_value * NAM_KNOB_MAX
            defaults[parameter_name] = max(
                NAM_KNOB_MIN, min(NAM_KNOB_MAX, round(static_value, 1))
            )
        return defaults

    def _limit_knob_rows(self, count):
        if self.plugin_parameters:
            previous = self._collect_setup_parameters()
            self._populate_knob_table(count, previous)
            self._save_setup()

    def _collect_setup_parameters(self):
        values = {}
        for row in range(self.knob_table.rowCount()):
            item = self.knob_table.item(row, 0)
            captured = self.knob_table.cellWidget(row, 1)
            capture_box = self.knob_table.cellWidget(row, 2)
            value = self.knob_table.cellWidget(row, 3)
            if item is None or captured is None or capture_box is None or value is None:
                continue
            values[item.text()] = {
                "captured": captured.isChecked(),
                "capture_knob": capture_box.currentText(),
                "static_value": value.value(),
            }
        return values

    def _populate_knob_table(self, count=None, previous=None):
        count = len(self.plugin_parameters) if count is None else min(
            count, len(self.plugin_parameters)
        )
        project_knobs = [knob["name"] for knob in (self.project or {}).get("knobs", [])]
        previous = previous or {}
        self._updating_setup_table = True
        try:
            self.knob_table.setRowCount(count)
            for row in range(count):
                parameter = self.plugin_parameters[row]
                self.knob_table.setItem(row, 0, QTableWidgetItem(parameter))

                captured = QCheckBox("Captured")
                captured.stateChanged.connect(
                    lambda state, row=row: self._set_knob_captured(row, state)
                )
                self.knob_table.setCellWidget(row, 1, captured)

                capture_box = QComboBox()
                for knob_name in project_knobs:
                    capture_box.addItem(knob_name)
                capture_box.setEnabled(False)
                capture_box.currentTextChanged.connect(lambda _value: self._save_setup())
                self.knob_table.setCellWidget(row, 2, capture_box)
                matching_knob = next(
                    (
                        knob_name
                        for knob_name in project_knobs
                        if knob_name.casefold() == parameter.casefold()
                    ),
                    None,
                )
                if matching_knob is not None:
                    capture_box.setCurrentText(matching_knob)
                    captured.setChecked(True)

                value = QDoubleSpinBox()
                value.setRange(0.0, 10.0)
                value.setDecimals(1)
                value.setSingleStep(0.1)
                value.setValue(5.0)
                value.valueChanged.connect(lambda _value: self._save_setup())
                self.knob_table.setCellWidget(row, 3, value)
                saved = previous.get(parameter)
                if saved:
                    capture_box.setCurrentText(saved.get("capture_knob", ""))
                    value.setValue(float(saved.get("static_value", 5.0)))
                    captured.setChecked(bool(saved.get("captured", False)))
        finally:
            self._updating_setup_table = False
        self.knob_table.resizeColumnsToContents()

    def _apply_saved_setup(self):
        if not self.saved_setup:
            return
        self._updating_setup_table = True
        try:
            count = min(
                int(self.saved_setup.get("parameter_count", self.knob_table.rowCount())),
                len(self.plugin_parameters),
            )
            self.knob_count.blockSignals(True)
            self.knob_count.setValue(count)
            self.knob_count.blockSignals(False)
            self._populate_knob_table(count)
            saved = {
                item.get("parameter"): item
                for item in self.saved_setup.get("parameters", [])
            }
            for row in range(count):
                parameter = self.plugin_parameters[row]
                item = saved.get(parameter)
                if not item:
                    continue
                captured = self.knob_table.cellWidget(row, 1)
                capture_box = self.knob_table.cellWidget(row, 2)
                value = self.knob_table.cellWidget(row, 3)
                capture_box.setCurrentText(item.get("capture_knob", ""))
                value.setValue(float(item.get("static_value", 5.0)))
                captured.setChecked(bool(item.get("captured", False)))
        finally:
            self._updating_setup_table = False

    def _save_setup(self):
        if self.project_dir is None or self._updating_setup_table:
            return
        setup = {
            "version": 1,
            "plugin_path": self.vst_path,
            "parameter_count": self.knob_table.rowCount(),
            "parameters": [],
        }
        for row in range(self.knob_table.rowCount()):
            parameter = self.knob_table.item(row, 0)
            captured = self.knob_table.cellWidget(row, 1)
            capture_box = self.knob_table.cellWidget(row, 2)
            value = self.knob_table.cellWidget(row, 3)
            if parameter is None or captured is None or capture_box is None or value is None:
                continue
            setup["parameters"].append({
                "parameter": parameter.text(),
                "captured": captured.isChecked(),
                "capture_knob": capture_box.currentText(),
                "static_value": value.value(),
            })
        write_json(self.project_dir / SETUP_FILENAME, setup)
        self.saved_setup = setup

    def _reset_celine_setup(self):
        if self.project_dir is None:
            QMessageBox.warning(self, "No project open", "Open a capture project first.")
            return
        setup_path = self.project_dir / SETUP_FILENAME
        if setup_path.exists():
            answer = QMessageBox.question(
                self,
                "Reset Celine setup",
                "Delete this project's saved Celine setup and use the current "
                "Celine knob values the next time setup is opened?",
            )
            if answer != QMessageBox.Yes:
                return
            try:
                setup_path.unlink()
            except OSError as error:
                QMessageBox.critical(self, "Reset Celine setup failed", str(error))
                return
        self.saved_setup = None
        self.plugin_parameters = []
        self.plugin = None
        self.knob_count.blockSignals(True)
        self.knob_count.setValue(0)
        self.knob_count.blockSignals(False)
        self.knob_table.setRowCount(0)
        self.render_button.setEnabled(False)
        self.status.setText(
            "Celine setup reset. Set Up Celine to import current knob values."
        )

    def _set_knob_captured(self, row, state):
        capture_box = self.knob_table.cellWidget(row, 2)
        value = self.knob_table.cellWidget(row, 3)
        if capture_box is None or value is None:
            return
        is_captured = state == 2
        capture_box.setEnabled(is_captured)
        value.setEnabled(not is_captured)
        self._save_setup()

    def _get_assignments(self):
        assignments = []
        project_knobs = [knob["name"] for knob in self.project["knobs"]]
        for row in range(self.knob_table.rowCount()):
            parameter = self.knob_table.item(row, 0).text()
            is_captured = self.knob_table.cellWidget(row, 1).isChecked()
            capture_name = self.knob_table.cellWidget(row, 2).currentText()
            value = self.knob_table.cellWidget(row, 3).value()
            if is_captured:
                if capture_name not in project_knobs:
                    raise ValueError(f"Assign a JSON knob to Celine parameter {parameter}.")
                assignments.append(
                    {"parameter": parameter, "mode": "capture", "project_knob": capture_name}
                )
            else:
                assignments.append(
                    {
                        "parameter": parameter,
                        "mode": "static",
                        "value": value,
                    }
                )
        if not assignments:
            raise ValueError("Set up at least one Celine parameter.")
        return assignments

    def _run_worker(self, settings):
        thread = QThread(self)
        worker = CelineWorker(settings)
        self.active_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.status.connect(self.status.setText)
        worker.signals.eta.connect(self.eta_status.setText)
        worker.signals.capture_progress.connect(self._update_capture_progress)
        worker.signals.capture_finished.connect(self._capture_finished)
        worker.signals.finished.connect(self._render_finished)
        worker.signals.stopped.connect(self._render_stopped)
        worker.signals.error.connect(self._worker_error)
        worker.signals.finished.connect(thread.quit)
        worker.signals.stopped.connect(thread.quit)
        worker.signals.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.threads.append(thread)
        self.workers.append(worker)
        thread.finished.connect(lambda: self.threads.remove(thread))
        thread.finished.connect(lambda: self.workers.remove(worker))
        thread.finished.connect(lambda: self._clear_active_worker(worker))
        thread.start()
        return worker

    def _clear_active_worker(self, worker):
        if self.active_worker is worker:
            self.active_worker = None

    def _start_render(self):
        if self.project_dir is None or self.project is None:
            QMessageBox.warning(self, "No project open", "Open a capture project first.")
            return
        if not self._inputs_available():
            QMessageBox.warning(
                self,
                "Missing input audio",
                "The project must contain both inputTrunc.wav and validation.wav before rendering.",
            )
            return
        if self.plugin is None:
            QMessageBox.warning(self, "Celine not set up", "Click Set Up Celine first.")
            return
        try:
            assignments = self._get_assignments()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid Celine setup", str(error))
            return
        self.render_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._save_setup()
        self.status.setText("Rendering captures...")
        self.eta_status.setText("Estimated time remaining: calculating...")
        self._run_worker((self.plugin, self.project_dir, self.project, assignments))

    def _update_capture_progress(self, row, done, chunks, label):
        progress = self.capture_table.cellWidget(row, len(self.project["knobs"]) + 2)
        if progress is not None:
            progress.setValue(int(done * 100 / chunks))
            progress.setFormat(label)

    def _capture_finished(self, row):
        progress = self.capture_table.cellWidget(row, len(self.project["knobs"]) + 2)
        if progress is not None:
            progress.setValue(100)
            progress.setFormat("Captured")

    def _render_finished(self, message):
        self.render_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.project is not None:
            self._show_capture_table(self.project)
        self.status.setText(message)
        self.eta_status.setText("Estimated time remaining: 0:00")
        QMessageBox.information(self, "Render complete", message)

    def _stop_render(self):
        if self.active_worker is not None:
            self.active_worker.stop_event.set()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the current audio block...")

    def _reset_captures(self):
        if self.project_dir is None or self.project is None:
            QMessageBox.warning(self, "No project open", "Open a capture project first.")
            return
        answer = QMessageBox.question(
            self,
            "Reset captures",
            "Reset every capture to pending and delete the rendered files in the "
            "captures directory?",
        )
        if answer != QMessageBox.Yes:
            return
        try:
            for entry in self.project["entries"]:
                output_path = self.project_dir / entry["y_path"].replace("/", os.sep)
                if output_path.is_file():
                    output_path.unlink()
                entry["status"] = "pending"
                entry["captured_at"] = None
                entry["delay"] = None
                entry["qa"] = None
                entry["stream_config"] = None
            write_json(self.project_dir / "capture_project.json", self.project)
            write_json(self.project_dir / "data.json", create_data_json(self.project))
            self._show_capture_table(self.project)
            self.status.setText("All captures reset to pending.")
        except OSError as error:
            QMessageBox.critical(self, "Reset failed", str(error))

    def _render_stopped(self, message):
        self.render_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText(message)
        self.eta_status.setText("Estimated time remaining: stopped")

    def _worker_error(self, error):
        self.render_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText("An error occurred.")
        self.eta_status.setText("Estimated time remaining: unavailable")
        QMessageBox.critical(self, "Celine automation", error)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    available_styles = [name.lower() for name in QStyleFactory.keys()]
    if "windows11" in available_styles:
        app.setStyle("Windows11")
    elif "windowsvista" in available_styles:
        app.setStyle("WindowsVista")
    window = CelineAutomationApp()
    window.show()
    sys.exit(app.exec())
