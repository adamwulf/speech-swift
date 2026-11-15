"""Find short bursts of sound surrounded by digital silence in rendered WAVs.

Usage: python seam_bursts.py <wav> [<wav> ...]

Splits each file at runs of at least 20 ms of digital silence (exact zeros in
a 16-bit WAV, i.e. under about -90 dBFS in the float audio) and lists every
non-silent segment shorter than 300 ms: start time, length, peak and RMS in
dBFS. Speech never forms such a segment. A burst at 0.00 s or right after a
seam is the start-of-pass noise some fp16 graphs emit (see
docs/research/2026-11-13-kokoro-long-window-and-exporters.md, section 5).
With KokoroLeadInBurst in place, expect "no short bursts" or only a faint
remnant of a few ms under -70 dBFS RMS just after 0.025 s (the faded tail of
a burst longer than 25 ms); before the fix the bursts measured -50 to -59.
"""
import sys

import numpy as np
import soundfile as sf


def db(v):
    return 20 * np.log10(max(float(v), 1e-10))


for path in sys.argv[1:]:
    x, sr = sf.read(path, dtype="float32")
    min_zero = sr // 50
    nonzero = x != 0
    # Mark every sample that lies inside a run of >= min_zero exact zeros.
    silent = np.zeros(len(x), dtype=bool)
    i = 0
    while i < len(x):
        if nonzero[i]:
            i += 1
            continue
        j = i
        while j < len(x) and not nonzero[j]:
            j += 1
        if j - i >= min_zero:
            silent[i:j] = True
        i = j
    bursts = []
    i = 0
    while i < len(x):
        if silent[i]:
            i += 1
            continue
        j = i
        while j < len(x) and not silent[j]:
            j += 1
        if j - i < 0.3 * sr:
            seg = x[i:j]
            bursts.append(f"{i / sr:5.2f}s {1000 * (j - i) / sr:4.0f} ms peak {db(np.abs(seg).max()):6.1f} "
                          f"rms {db(np.sqrt(np.mean(seg ** 2))):6.1f}")
        i = j
    name = path.split("/")[-1]
    print(f"{name:44s} {len(x) / sr:6.2f}s | " + (" ; ".join(bursts) if bursts else "no short bursts"))
