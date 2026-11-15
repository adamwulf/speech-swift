"""Noise floor of rendered WAVs: RMS of the lead-in and of the quietest audible windows.

Usage: python noise_floor.py <wav> [<wav> ...]

Prints, per file, in dBFS:
- lead-in: RMS of the first 150 ms. On a graph that emits a start-of-pass
  burst this measures the burst, not a floor: the "-55 dBFS fp16 floor" first
  reported in docs/research/2026-11-13-kokoro-long-window-and-exporters.md was
  this figure. Check seam_bursts.py / first_frame.py before reading it as noise.
- p5 audible: 5th percentile of 10 ms window RMS over windows above 16-bit
  digital silence (-90 dBFS), so inserted silence between pieces does not
  hide the floor. This is the level of the quietest real audio (pauses,
  lead-in noise).
- silent: share of 10 ms windows that are digital silence.
"""
import sys

import numpy as np
import soundfile as sf


def db(v):
    return 20 * np.log10(max(float(v), 1e-10))


for path in sys.argv[1:]:
    x, sr = sf.read(path, dtype="float32")
    win = sr // 100
    n = len(x) // win
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    lead = np.sqrt(np.mean(x[: int(0.15 * sr)] ** 2))
    audible = rms[rms > 10 ** (-90 / 20)]
    p5 = np.percentile(audible, 5) if len(audible) else 0.0
    silent = 100 * (1 - len(audible) / max(n, 1))
    print(f"{path.split('/')[-1]:45s} lead-in {db(lead):7.1f} dBFS | p5 audible {db(p5):7.1f} dBFS | silent {silent:4.0f}%")
