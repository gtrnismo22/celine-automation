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
    "input": 0.0,       # 0 dB boost
    "gain": 0.65,
    "bass": 0.50,
    "mid": 0.40,
    "treble": 0.60,
    "presence": 0.50,
    "master": 0.50,
}

# Matches:
#   lhs_000_I0_Ma1.3_P7.8_B1.1_Mi3.6_T5.8_G7.6.wav  (or Ihs_000_...)
#   corner_127_I10_Ma10_P10_B10_Mi10_T10_G10.wav
#   val_lhs_000_I0_Ma9.9_P6.9_B2.5_Mi4.5_T6.3_G6.5.wav (or val_Ihs_000_...)
FILENAME_RE = re.compile(
    r"^(?P<split>val_(?:lhs|ihs)|(?:lhs|ihs)|corner)_(?P<idx>\d+)_"
    r"I(?P<I>\d+(?:\.\d+)?)_"
    r"Ma(?P<Ma>\d+(?:\.\d+)?)_"
    r"P(?P<P>\d+(?:\.\d+)?)_"
    r"B(?P<B>\d+(?:\.\d+)?)_"
    r"Mi(?P<Mi>\d+(?:\.\d+)?)_"
    r"T(?P<T>\d+(?:\.\d+)?)_"
    r"G(?P<G>\d+(?:\.\d+)?)\.wav$",
    re.IGNORECASE,
)


def parse_capture_filename(wav_path: Path):
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
    for token, knob_name in TOKEN_TO_KNOB.items():
        val_0_to_10 = float(m.group(token))
        if not (0.0 <= val_0_to_10 <= 10.0):
            raise ValueError(f"Out-of-range value {token}={val_0_to_10} in {wav_path.name}")
        params[knob_name] = round(val_0_to_10 / 10.0, 6)

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
        for k in KNOBS:
            row[k] = rec["params"][k]
        row.update({"rms": rms, "peak": peak, "ok": 1, "error": ""})
        csv_rows.append(row)

    np.save(out_dir / "outputs.npy", outputs)

    fieldnames = ["idx"] + KNOBS + ["rms", "peak", "ok", "error"]
    with open(out_dir / "params.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    config = {
        "knobs": KNOBS,
        "bounds": {k: [0.0, 1.0] for k in KNOBS},
        "defaults": DEFAULTS,
        "param_map": {k: k for k in KNOBS},
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
    parser.add_argument("--project-dir", required=True, help="Path to existing channel/mode project folder")
    parser.add_argument("--output-root", required=True, help="New destination directory for converted datasets")
    parser.add_argument(
        "--mode-name",
        required=True,
        help="Mode identifier, e.g. red_modern, red_vintage, orange_modern, orange_vintage, green_clean",
    )
    parser.add_argument("--gear-make", default="Mesa Boogie", help="Amp manufacturer metadata")
    parser.add_argument(
        "--gear-model",
        default=None,
        help="Amp model metadata (defaults to 'Dual Rectifier Rev G - <Mode Name>')",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    captures_dir = project_dir / "captures"
    train_dry = project_dir / "inputTrunc.wav"
    val_dry = project_dir / "validation.wav"

    for required in (captures_dir, train_dry, val_dry):
        if not required.exists():
            raise FileNotFoundError(f"Missing required path: {required}")

    train_records = []
    val_records = []
    unmatched = []

    for wav_path in sorted(captures_dir.glob("*.wav")):
        parsed = parse_capture_filename(wav_path)
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

    if len(train_records) != 678:
        print(f"[WARN] Expected 678 train captures (550 LHS + 128 corner), found {len(train_records)}")
    if len(val_records) != 50:
        print(f"[WARN] Expected 50 validation captures, found {len(val_records)}")

    pretty_mode = args.mode_name.replace("_", " ").title()
    gear_model = args.gear_model or f"Dual Rectifier Rev G - {pretty_mode}"
    output_root = Path(args.output_root).resolve()

    write_parametric_dataset(
        dry_wav_path=train_dry,
        records=train_records,
        out_dir=output_root / f"{args.mode_name}_train",
        gear_make=args.gear_make,
        gear_model=gear_model,
    )
    write_parametric_dataset(
        dry_wav_path=val_dry,
        records=val_records,
        out_dir=output_root / f"{args.mode_name}_val",
        gear_make=args.gear_make,
        gear_model=gear_model,
    )


if __name__ == "__main__":
    main()