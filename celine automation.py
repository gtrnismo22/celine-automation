import json
import os
from pathlib import Path
import sys
import threading
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


DEFAULT_VST_PATH = (
    r"C:\Program Files\Common Files\VST3\Celine.vst3"
    r"\Contents\x86_64-win\Celine.vst3"
)
CHUNK_SIZE = 65536
OUTPUT_SAMPLE_RATE = 48000
NAM_KNOB_MIN = 0.0
NAM_KNOB_MAX = 10.0
CELINE_INPUT_DB_MIN = -20.0
CELINE_INPUT_DB_MAX = 20.0


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
            completed += 1

            def update_progress(done, chunks, completed=completed):
                self.signals.status.emit(
                    f"Rendering {completed}/{total} ({split})... chunk {done}/{chunks}"
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
        self.vst_path = QLineEdit(DEFAULT_VST_PATH)
        self.project_summary = QLabel("No project open.")
        self.status = QLabel("Open a project directory to begin.")
        self.capture_table = QTableWidget()
        self.capture_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.capture_table.setAlternatingRowColors(True)
        self.capture_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.capture_table.setSortingEnabled(True)
        self.capture_table.horizontalHeader().setStretchLastSection(True)
        self.knob_table = QTableWidget()
        self.knob_table.setColumnCount(4)
        self.knob_table.setHorizontalHeaderLabels(
            ["Celine parameter", "Captured", "Capture knob", "Static value (0–10)"]
        )
        self.knob_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.knob_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._build_ui()

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

        source = QGroupBox("Sources and output")
        source_layout = QFormLayout(source)
        source_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        source_layout.addRow("Celine VST3", self._path_widget(self.vst_path, self._choose_vst))
        layout.addWidget(source)

        setup = QGroupBox("Celine setup")
        setup_layout = QVBoxLayout(setup)
        setup_actions = QHBoxLayout()
        self.setup_button = QPushButton("Set Up Celine")
        self.setup_button.clicked.connect(self._setup_celine)
        setup_actions.addWidget(self.setup_button)
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
        required_files = [
            project_path,
            project_dir / "inputTrunc.wav",
            project_dir / "validation.wav",
        ]
        missing_files = [file.name for file in required_files if not file.is_file()]
        if missing_files:
            QMessageBox.critical(
                self,
                "Invalid project",
                "The selected directory is missing:\n" + "\n".join(missing_files),
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
        self._show_capture_table(project)
        if self.plugin_parameters:
            self._populate_knob_table(self.knob_count.value())
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

    def _choose_vst(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Celine VST3", "", "VST3 plugin (*.vst3)")
        if path:
            self.vst_path.setText(path)

    def _setup_celine(self):
        if not self.vst_path.text().strip():
            QMessageBox.warning(self, "Missing plugin", "Select the Celine VST3 first.")
            return
        self.setup_button.setEnabled(False)
        self.status.setText("Loading Celine and opening its editor...")
        try:
            self.plugin = load_plugin(self.vst_path.text().strip())
            self.plugin.show_editor(threading.Event())
            self.plugin_parameters = list(self.plugin.parameters)
            self.knob_count.setValue(len(self.plugin_parameters))
            self._populate_knob_table()
            self.render_button.setEnabled(True)
            self.status.setText(
                "Celine setup loaded. Assign capture knobs and static values, then render."
            )
        except Exception as error:
            self.plugin = None
            QMessageBox.critical(self, "Celine setup failed", str(error))
            self.status.setText("Celine setup failed.")
        finally:
            self.setup_button.setEnabled(True)

    def _limit_knob_rows(self, count):
        if self.plugin_parameters:
            self._populate_knob_table(count)

    def _populate_knob_table(self, count=None):
        count = len(self.plugin_parameters) if count is None else min(
            count, len(self.plugin_parameters)
        )
        project_knobs = [knob["name"] for knob in (self.project or {}).get("knobs", [])]
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
            self.knob_table.setCellWidget(row, 3, value)
        self.knob_table.resizeColumnsToContents()

    def _set_knob_captured(self, row, state):
        capture_box = self.knob_table.cellWidget(row, 2)
        value = self.knob_table.cellWidget(row, 3)
        is_captured = state == 2
        capture_box.setEnabled(is_captured)
        value.setEnabled(not is_captured)

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
        self.status.setText("Rendering captures...")
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

    def _worker_error(self, error):
        self.render_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText("An error occurred.")
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
