import os
import soundfile as sf
import numpy as np
from pedalboard import load_plugin
from tqdm import tqdm

# Paths
VST_PATH = r"C:\Program Files\Common Files\VST3\Celine.vst3\Contents\x86_64-win\Celine.vst3"
INPUT_SWEEP = "input.wav"
OUTPUT_DIR = "./test_renders"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Audio
print(f"Loading {INPUT_SWEEP}...")
audio_data, sr = sf.read(INPUT_SWEEP, dtype="float32")

if audio_data.ndim == 1:
    audio_data = np.vstack([audio_data, audio_data])
elif audio_data.shape[0] != 2 and audio_data.shape[1] == 2:
    audio_data = audio_data.T

print(f"Audio loaded: Sample Rate = {sr}Hz, Shape = {audio_data.shape}")

# 2. Load Plugin
plugin = load_plugin(VST_PATH)

try:
    plugin.show_editor()
except Exception as e:
    pass

print("\nEditor closed! Starting chunked renders...\n")

def apply_marshall_specs(p):
    p["knob_1"].raw_value = 1.0  # NFB ON
    p["knob_2"].raw_value = 0.0  # Headroom OFF
    p["knob_3"].raw_value = 0.7  # Presence
    p["knob_5"].raw_value = 0.5  # Mid
    p["knob_7"].raw_value = 0.0  # Normal Vol OFF
    p["knob_8"].raw_value = 0.75 # Treble Vol
    p["knob_9"].raw_value = 0.0  # Treble/Normal OFF

def process_audio_in_chunks_with_pbar(plugin_obj, audio, sample_rate, file_label, chunk_size=65536):
    """Processes audio in small blocks and updates a tqdm progress bar for the current file."""
    num_samples = audio.shape[1]
    output_chunks = []
    
    # Calculate total chunk iterations
    total_chunks = (num_samples + chunk_size - 1) // chunk_size
    
    with tqdm(total=total_chunks, desc=file_label, unit="chunk", leave=True) as pbar:
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk = audio[:, start:end]
            
            # Process block (reset=False keeps filter & oversampling state continuous)
            processed_chunk = plugin_obj(chunk, sample_rate=sample_rate, reset=False)
            output_chunks.append(processed_chunk)
            
            pbar.update(1)
            
    return np.hstack(output_chunks)

test_levels = [0.2, 0.5, 0.8]

# Render loop
for idx, level in enumerate(test_levels, start=1):
    plugin.reset() # Reset state before starting new file
    apply_marshall_specs(plugin.parameters)
    plugin.parameters["input_db"].raw_value = level
    
    file_label = f"File {idx}/{len(test_levels)} (Input @ {int(level*100)}%)"
    
    # Process audio with inner progress bar
    output_audio = process_audio_in_chunks_with_pbar(plugin, audio_data, sr, file_label)
    
    out_name = f"test_marshall_clean_input_{int(level*100)}pct.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, output_audio.T, sr, subtype='PCM_16')
    print(f"✓ Saved: {out_name}\n")

print("Render test completed successfully!")