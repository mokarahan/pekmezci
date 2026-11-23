#!/usr/bin/env python3
"""
FLAC chunk(s) -> 20 x 1 kHz band matrix (.npy) and/or stacked-band plot (.png).

Rebuilds, from the saved FLAC chunks of mic_band_recorder.py, the files the
recorder no longer stores:
    bands_<start time>.npy   float32 array of shape (20, n_samples)
                             row b = content between b*1000 Hz and (b+1)*1000 Hz
    bands_<start time>.png   the 20 bands plotted under each other
                             (band 0 on top, each with its own amplitude scale)
The input chunk_<start time>.flac is not modified. The sample rate is read from
the FLAC file itself.

Install:  pip install soundfile numpy matplotlib
Run:      python flac_to_bands.py recordings/chunk_20261121T101500.000.flac
          python flac_to_bands.py recordings/            # every .flac in the folder
          python flac_to_bands.py a.flac b.flac --out bands_out --no-png
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                # file output only, never opens a window
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

N_BANDS = 20
BAND_WIDTH_HZ = 1000

PLOT_COLUMNS = 1000                  # min/max pairs drawn per band (keeps plotting fast)
PNG_DPI = 100                        # 12 x 10 in figure -> 1200 x 1000 px


def split_bands(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    """Split a 1D chunk into N_BANDS equal-width bands via FFT masking.

    Returns float32 array of shape (N_BANDS, len(chunk)).
    The bands are disjoint, so their sum reconstructs the 0-20 kHz content.
    (Identical to the split done by mic_band_recorder.py.)
    """
    n = len(chunk)
    spectrum = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    bands = np.empty((N_BANDS, n), dtype=np.float32)
    for b in range(N_BANDS):
        lo, hi = b * BAND_WIDTH_HZ, (b + 1) * BAND_WIDTH_HZ
        mask = (freqs >= lo) & (freqs < hi)
        bands[b] = np.fft.irfft(spectrum * mask, n=n).astype(np.float32)
    return bands


class BandPlot:
    """20 stacked subplots (one per band) sharing the time axis; saved to PNG."""

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.fig, self.axes = plt.subplots(N_BANDS, 1, sharex=True, figsize=(12, 10))
        self.fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, hspace=0.15)
        self.lines = []
        self.x = None

    def _setup(self, n_points: int) -> None:
        self.x = np.linspace(0, self.duration, n_points)
        self.lines = []
        for b, ax in enumerate(self.axes):
            (line,) = ax.plot(self.x, np.zeros_like(self.x), lw=0.6)
            ax.set_ylabel(f"{b}-{b + 1} kHz", rotation=0, ha="right", va="center", fontsize=7)
            ax.set_yticks([])
            ax.margins(x=0)
            self.lines.append(line)
        self.axes[-1].set_xlim(0, self.duration)
        self.axes[-1].set_xlabel("time within chunk (s)")

    @staticmethod
    def _envelope(bands: np.ndarray) -> np.ndarray:
        """Shrink each band to (min, max) pairs so drawing stays fast."""
        n = bands.shape[1]
        cols = min(PLOT_COLUMNS, n)
        usable = (n // cols) * cols
        blocks = bands[:, :usable].reshape(N_BANDS, cols, -1)
        env = np.empty((N_BANDS, 2 * cols), dtype=np.float32)
        env[:, 0::2] = blocks.min(axis=2)
        env[:, 1::2] = blocks.max(axis=2)
        return env

    def save(self, path: Path, bands: np.ndarray, title: str, duration: float) -> None:
        env = self._envelope(bands)
        if self.x is None or len(self.x) != env.shape[1] or duration != self.duration:
            self.duration = duration
            for ax in self.axes:
                ax.clear()
            self._setup(env.shape[1])
        for b, (ax, line) in enumerate(zip(self.axes, self.lines)):
            line.set_ydata(env[b])
            peak = max(float(np.abs(env[b]).max()) * 1.1, 1e-9)
            ax.set_ylim(-peak, peak)
        self.fig.suptitle(title)
        self.fig.savefig(path, dpi=PNG_DPI)


def collect_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.flac")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"[warn] not found: {p}")
    return files


def output_stem(flac: Path) -> str:
    stem = flac.stem                       # chunk_20261121T101500.000
    return "bands_" + stem[len("chunk_"):] if stem.startswith("chunk_") else "bands_" + stem


def main() -> None:
    parser = argparse.ArgumentParser(description="FLAC chunk -> 20-band .npy and .png")
    parser.add_argument("inputs", nargs="+", help="FLAC file(s) or folder(s) containing them")
    parser.add_argument("--out", default=None,
                        help="output directory (default: next to each FLAC file)")
    parser.add_argument("--no-npy", action="store_true", help="do not write the .npy matrix")
    parser.add_argument("--no-png", action="store_true", help="do not write the .png plot")
    args = parser.parse_args()

    if args.no_npy and args.no_png:
        parser.error("nothing to do: both --no-npy and --no-png given")

    files = collect_inputs(args.inputs)
    if not files:
        parser.error("no FLAC files found")

    out_override = Path(args.out) if args.out else None
    if out_override:
        out_override.mkdir(parents=True, exist_ok=True)

    plot = None if args.no_png else BandPlot(duration=1.0)

    for flac in files:
        data, sample_rate = sf.read(flac, dtype="float32", always_2d=True)
        chunk = data[:, 0]                 # recorder writes mono
        out_dir = out_override or flac.parent
        stem = output_stem(flac)

        bands = split_bands(chunk, sample_rate)
        names = []
        if not args.no_npy:
            np.save(out_dir / f"{stem}.npy", bands)
            names.append(f"{stem}.npy")
        if plot is not None:
            plot.save(out_dir / f"{stem}.png", bands,
                      title=f"20 bands - {flac.name}", duration=len(chunk) / sample_rate)
            names.append(f"{stem}.png")
        print(f"{flac.name} -> " + " + ".join(names))


if __name__ == "__main__":
    main()
