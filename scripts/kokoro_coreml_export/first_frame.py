"""RMS of the first 25 ms of each render, and of 25-30 ms, grouped by configuration.

Usage: python first_frame.py <dir> [<dir> ...]

Reads <config>_<text>.wav files (the listening harness's naming) and prints,
per folder and configuration, the loudest first-25 ms RMS and 25-30 ms RMS
over its files, in dBFS. Kokoro clamps every duration to at least one 25 ms
frame, so the first 25 ms of a pass is the start token's lead-in, never
speech: every graph should read -85 dBFS or less there. Higher values are the
fp16 start-of-pass burst that KokoroLeadInBurst clears. 25-30 ms is the fade
that follows; a short reply's speech may begin inside it.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def db(v):
    return 20 * np.log10(max(float(v), 1e-10))


worst = defaultdict(lambda: [-200.0, -200.0, 0, ""])
for folder in sys.argv[1:]:
    for path in sorted(Path(folder).glob("*.wav")):
        if "_" not in path.stem:
            continue
        config, text = path.stem.rsplit("_", 1)
        x, sr = sf.read(str(path), dtype="float32")
        lead = int(0.025 * sr)
        fade = int(0.030 * sr)
        first = db(np.sqrt(np.mean(x[:lead] ** 2)))
        after = db(np.sqrt(np.mean(x[lead:fade] ** 2)))
        row = worst[f"{Path(folder).name}/{config}"]
        if first > row[0]:
            row[0], row[3] = first, text
        row[1] = max(row[1], after)
        row[2] += 1
for name, (first, after, count, text) in sorted(worst.items()):
    print(f"{name:55s} n={count:2d} first 25 ms max {first:6.1f} ({text}) | 25-30 ms max {after:6.1f}")
