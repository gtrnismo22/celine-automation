import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import soundfile as sf

# Plugin UI knob order (you can reorder this list to match your desired plugin faceplate layout)
KNOBS = ["input", "gain", "bass", "mid", "treble", "presence", "master"]

# Map filename token prefixes -> normalized knob names
TOKEN_TO_KNOB = {
    "I": "input",       # 0..10 -> 0..+40 dB front-end boost
    "Ma": "master",     # 0..10
    "P": "presence",    # 0..10
    "B": "bass",        # 0..10
    "Mi": "mid",        # 0..10
    "T": "treble",      # 0..10
    "G": "gain",        # 0..10
}

# Reasonable default knob positions (normalized 0.0 - 1.0) stored in .param.nam metadata
DEFAULTS = {
    "input": 0.50,       # 0 dB boost
    "gain": 0.50,
    "bass": 0.50,
    "mid": 0.50,
    "treble": 0.50,
    "presence": 0.50,
    "master": 0.50,
}

# Matches:
#   lhs_000_I0_Ma1.3_P7.8_B1.1_Mi3.6_T5.8_G7.6.wav  (or Ihs_000_...)
#   corner_127_I10_Ma10_P10_B10_Mi10_T10_G10.wav
#   val_lhs_000_I0_Ma9.9_P6.9_B2.5_Mi4.5_T6.3_G6.5.wav (or val_Ihs_000_...)
FILENAME_RE = re.compile(
    r"^(?P<split>val_(?:lhs|ihs)|(?:lhs|ihs)|corner)_(?P<idx>\d+)_"
    r"(?P<params>.+)\.wav$",
    re.IGNORECASE,
)

DEFAULT_TOKEN_TO_KNOB = {
    "I": "input",
    "Ma": "master",
    "P": "presence",
    "B": "bass",
    "Mi": "mid",
    "T": "treble",
    "G": "gain",
    "M": "mid",
    "V": "volume",
}


def choose_project_dir() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(title="Select Celine project folder")
    root.destroy()

    if not selected_dir:
        raise SystemExit("No project folder selected.")
    return Path(selected_dir).resolve()


def mode_name_from_project_dir(project_dir: Path) -> str:
    mode_part = project_dir.name.rsplit(" - ", 1)[-1]
    mode_name = re.sub(r"[^a-zA-Z0-9]+", "_", mode_part).strip("_").lower()
    if not mode_name:
        raise ValueError(f"Could not derive a mode name from project folder: {project_dir}")
    return mode_name


def extract_capture_tokens(wav_path: Path) -> list[str]:
    match = FILENAME_RE.match(wav_path.name)
    if not match:
        return []

    tokens = []
    for part in match.group("params").split("_"):
        token_match = re.fullmatch(r"(?P<token>[A-Za-z]+)\d+(?:\.\d+)?", part)
        if token_match:
            tokens.append(token_match.group("token"))
    return tokens


def detect_capture_tokens(captures_dir: Path) -> list[str]:
    tokens = []
    seen = set()
    for wav_path in sorted(captures_dir.glob("*.wav")):
        for token in extract_capture_tokens(wav_path):
            token_key = token.casefold()
            if token_key not in seen:
                seen.add(token_key)
                tokens.append(token)
    if not tokens:
        raise ValueError(f"Could not detect knob tokens in WAV filenames under {captures_dir}")
    return tokens


def choose_knob_mapping(tokens: list[str]) -> dict[str, str]:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Configure capture knobs")
    root.resizable(False, False)

    tk.Label(
        root,
        text="Assign each filename token to a knob name.\n"
        "The names will be used in params.csv and config.json.",
        justify="left",
        padx=12,
        pady=12,
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    tk.Label(root, text="Filename token", padx=12).grid(row=1, column=0, sticky="w")
    tk.Label(root, text="Knob name", padx=12).grid(row=1, column=1, sticky="w")

    entries = []
    for row, token in enumerate(tokens, start=2):
        tk.Label(root, text=token, padx=12).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(root, width=24)
        entry.insert(0, DEFAULT_TOKEN_TO_KNOB.get(token, token.lower()))
        entry.grid(row=row, column=1, padx=12, pady=2)
        entries.append((token, entry))

    result = {}

    def submit():
        normalized = {}
        for token, entry in entries:
            knob_name = re.sub(r"[^a-zA-Z0-9]+", "_", entry.get().strip()).strip("_").lower()
            if not knob_name:
                messagebox.showerror("Invalid knob name", f"Enter a name for token {token}.", parent=root)
                return
            if knob_name in normalized.values():
                messagebox.showerror("Duplicate knob name", f"'{knob_name}' is used more than once.", parent=root)
                return
            normalized[token.casefold()] = knob_name
        result.update(normalized)
        root.destroy()

    def cancel():
        root.destroy()

    buttons = tk.Frame(root, pady=12)
    buttons.grid(row=len(tokens) + 2, column=0, columnspan=2)
    tk.Button(buttons, text="Use these knobs", command=submit).pack(side="left", padx=4)
    tk.Button(buttons, text="Cancel", command=cancel).pack(side="left", padx=4)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if not result:
        raise SystemExit("Knob configuration cancelled.")
    return result


def parse_capture_filename(wav_path: Path, token_to_knob: dict[str, str]):
    m = FILENAME_RE.match(wav_path.name)
    if not m:
        return None

    raw_split = m.group("split").lower()
    if raw_split.startswith("val_"):
        split = "val"
        sort_group = 0
    elif raw_split in ("lhs", "ihs"):
        split = "train"
        sort_group = 0  # LHS first (0..549)
    else:
        split = "train"
        sort_group = 1  # Corners second (0..127)

    idx_num = int(m.group("idx"))
    params = {}
    parsed_tokens = set()
    for part in m.group("params").split("_"):
        token_match = re.fullmatch(r"(?P<token>[A-Za-z]+)(?P<value>\d+(?:\.\d+)?)", part)
        if not token_match:
            return None
        token = token_match.group("token").casefold()
        knob_name = token_to_knob.get(token)
        if knob_name is None:
            return None
        parsed_tokens.add(token)
        val_0_to_10 = float(token_match.group("value"))
        if not (0.0 <= val_0_to_10 <= 10.0):
            raise ValueError(f"Out-of-range value in {wav_path.name}: {part}")
        params[knob_name] = round(val_0_to_10 / 10.0, 6)
    if parsed_tokens != set(token_to_knob):
        return None

    return {
        "split": split,
        "sort_key": (sort_group, idx_num),
        "wav_path": wav_path,
        "params": params,
    }


def write_parametric_dataset(
    dry_wav_path: Path,
    records: list[dict],
    out_dir: Path,
    gear_make: str,
    gear_model: str,
    knobs: list[str],
):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty directory: {out_dir}. "
            "Choose a fresh output folder so existing files are never overwritten."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dry excitation (mono float32)
    dry, sr = sf.read(dry_wav_path, dtype="float32")
    if dry.ndim > 1:
        dry = dry[:, 0]
    sf.write(out_dir / "sweep.wav", dry, sr, subtype="FLOAT")

    n_combos = len(records)
    n_samples = len(dry)
    outputs = np.zeros((n_combos, n_samples), dtype=np.float32)

    csv_rows = []
    clipped_files = []
    max_peak_overall = 0.0

    for idx, rec in enumerate(records):
        y, y_sr = sf.read(rec["wav_path"], dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]

        if y_sr != sr:
            raise ValueError(f"Sample rate mismatch on {rec['wav_path'].name}: {y_sr} vs {sr}")
        if len(y) != n_samples:
            raise ValueError(
                f"Length mismatch on {rec['wav_path'].name}: {len(y)} samples vs {n_samples} in {dry_wav_path.name}"
            )

        outputs[idx] = y
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        peak = float(np.max(np.abs(y)))
        max_peak_overall = max(max_peak_overall, peak)

        if peak >= 0.9999:
            clipped_files.append((rec["wav_path"].name, peak))

        row = {"idx": idx}
        for k in knobs:
            row[k] = rec["params"][k]
        row.update({"rms": rms, "peak": peak, "ok": 1, "error": ""})
        csv_rows.append(row)

    np.save(out_dir / "outputs.npy", outputs)

    fieldnames = ["idx"] + knobs + ["rms", "peak", "ok", "error"]
    with open(out_dir / "params.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    config = {
        "knobs": knobs,
        "bounds": {k: [0.0, 1.0] for k in knobs},
        "defaults": {k: DEFAULTS.get(k, 0.50) for k in knobs},
        "param_map": {k: k for k in knobs},
        "sample_rate": int(sr),
        "gear_type": "amp",
        "gear_make": gear_make,
        "gear_model": gear_model,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] {out_dir}")
    print(f"     Combos: {n_combos} | Samples/combo: {n_samples} ({n_samples / sr:.2f}s @ {sr} Hz)")
    print(f"     outputs.npy size: {outputs.nbytes / (1024**2):.1f} MB | Max peak: {max_peak_overall:.4f}")
    if clipped_files:
        print(
            f"     [WARN] {len(clipped_files)} files hit >= 0.9999 peak (possible digital clipping on +40 dB corners). "
            f"First example: {clipped_files[0][0]} (peak={clipped_files[0][1]:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Celine/phillipmself capture project into mrgeneko/parametric-nam train & val datasets."
    )
    parser.add_argument(
        "--project-dir",
        help="Path to existing channel/mode project folder (opens a folder picker if omitted)",
    )
    parser.add_argument(
        "--output-root",
        help="Destination directory (defaults to sibling Updated/<project-folder-name>)",
    )
    parser.add_argument(
        "--mode-name",
        help="Mode identifier (defaults to the final part of the project folder name)",
    )
    parser.add_argument("--gear-make", default="Mesa Boogie", help="Amp manufacturer metadata")
    parser.add_argument(
        "--gear-model",
        default=None,
        help="Amp model metadata (defaults to 'Dual Rectifier Rev G - <Mode Name>')",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve() if args.project_dir else choose_project_dir()
    mode_name = args.mode_name or mode_name_from_project_dir(project_dir)
    captures_dir = project_dir / "captures"
    train_dry = project_dir / "inputTrunc.wav"
    val_dry = project_dir / "validation.wav"

    for required in (captures_dir, train_dry, val_dry):
        if not required.exists():
            raise FileNotFoundError(f"Missing required path: {required}")

    capture_tokens = detect_capture_tokens(captures_dir)
    token_to_knob = choose_knob_mapping(capture_tokens)
    knobs = list(token_to_knob.values())

    train_records = []
    val_records = []
    unmatched = []

    for wav_path in sorted(captures_dir.glob("*.wav")):
        parsed = parse_capture_filename(wav_path, token_to_knob)
        if parsed is None:
            unmatched.append(wav_path.name)
            continue
        if parsed["split"] == "train":
            train_records.append(parsed)
        else:
            val_records.append(parsed)

    train_records.sort(key=lambda r: r["sort_key"])
    val_records.sort(key=lambda r: r["sort_key"])

    if unmatched:
        print(f"[WARN] Skipped {len(unmatched)} unrecognized WAV files (e.g. {unmatched[:3]})")

    print(f"[INFO] Found {len(train_records)} train captures and {len(val_records)} validation captures")

    pretty_mode = mode_name.replace("_", " ").title()
    gear_model = args.gear_model or f"Dual Rectifier Rev G - {pretty_mode}"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else project_dir.parent / "Updated" / project_dir.name
    )

    write_parametric_dataset(
        dry_wav_path=train_dry,
        records=train_records,
        out_dir=output_root / f"{mode_name}_train",
        gear_make=args.gear_make,
        gear_model=gear_model,
        knobs=knobs,
    )
    write_parametric_dataset(
        dry_wav_path=val_dry,
        records=val_records,
        out_dir=output_root / f"{mode_name}_val",
        gear_make=args.gear_make,
        gear_model=gear_model,
        knobs=knobs,
    )


if __name__ == "__main__":
    main()