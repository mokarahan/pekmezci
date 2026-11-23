#!/usr/bin/env python3
"""
Microphone -> 1 s chunks (no overlap) -> 24-bit FLAC files.

Only the compressed audio is stored. The 20-band matrix (.npy) and the stacked
band plot (.png) are NOT saved; rebuild them from a FLAC file with
flac_to_bands.py when needed.

    chunk_<start time>.flac    mono, 48 kHz, 24-bit, lossless FLAC compression
                               (float32 samples are clipped to +-1.0 and stored as 24-bit)

The filename carries the chunk's start time (local time).

Install:  pip install sounddevice soundfile numpy
Run:      python mic_band_recorder.py [--out recordings] [--device N]
Stop:     Ctrl+C
"""

import argparse
import queue
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 48000                  # Hz
CHUNK_SEC = 1.0
OVERLAP = 0

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
HOP_SAMPLES = int(CHUNK_SAMPLES * (1 - OVERLAP))


def make_stem(start_time: datetime) -> str:
    # No colons in the name so it is valid on Windows too.
    return f"{start_time:%Y%m%dT%H%M%S}.{start_time.microsecond // 1000:03d}"


def save_chunk(out_dir: Path, stem: str, chunk: np.ndarray) -> Path:
    flac_path = out_dir / f"chunk_{stem}.flac"
    sf.write(flac_path, chunk, SAMPLE_RATE, format="FLAC", subtype="PCM_24")
    return flac_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="recordings", help="output directory")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}")
        q.put(indata[:, 0].copy())   # mono, float32

    buffer = np.empty(0, dtype=np.float32)
    samples_consumed = 0             # samples dropped from the buffer front so far
    t0 = None                        # wall-clock time of the first sample

    print(f"Recording at {SAMPLE_RATE} Hz, saving 24-bit FLAC. Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            device=args.device, callback=callback):
            t0 = datetime.now()
            while True:
                buffer = np.concatenate([buffer, q.get()])

                while len(buffer) >= CHUNK_SAMPLES:
                    chunk = buffer[:CHUNK_SAMPLES]
                    start_time = t0 + timedelta(seconds=samples_consumed / SAMPLE_RATE)

                    flac_path = save_chunk(out_dir, make_stem(start_time), chunk)
                    print(f"saved {flac_path.name}")

                    buffer = buffer[HOP_SAMPLES:]
                    samples_consumed += HOP_SAMPLES
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
