"""Noise floor of rendered WAVs: RMS of the quietest 10 ms windows.

Usage: python noise_floor.py <wav> [<wav> ...]

Prints, per file, the RMS of the lead-in (first 150 ms) and the 5th
percentile of 10 ms window RMS over the whole file, both in dBFS.
"""
import sys

import numpy as np
import soundfile as sf

for path in sys.argv[1:]:
    x, sr = sf.read(path, dtype="float32")
    win = sr // 100
    n = len(x) // win
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-20)
    lead = np.sqrt(np.mean(x[: int(0.15 * sr)] ** 2) + 1e-20)
    db = lambda v: 20 * np.log10(v)
    print(f"{path.split('/')[-1]:45s} lead-in {db(lead):7.1f} dBFS   p5 window {db(np.percentile(rms, 5)):7.1f} dBFS")
