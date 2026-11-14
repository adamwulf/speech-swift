"""Copy A/B WAV pairs to shuffled names for blind listening.

Usage: python make_blind_set.py <listen_dir> <blind_dir> <key_csv> [label:configA:configB ...]

For each text and each pair, writes the two configurations' WAVs as
<text>_<label>_1.wav / _2.wav in random order, and records which is which
in key_csv (keep it closed while listening). Without pair arguments it
pairs A with X1 (label ids12) and A0 with X2 (label ids00).
"""
import csv
import random
import shutil
import sys
from pathlib import Path

listen, blind, key_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
blind.mkdir(parents=True, exist_ok=True)
texts = ["short", "long", "yes", "hello", "question", "numbers", "commas", "paragraph"]
pairs = [tuple(spec.split(":")) for spec in sys.argv[4:]] or [
    ("ids12", "A_aufklarer-5s", "X1_eliza-5s-fp32"),
    ("ids00", "A0_aufklarer-5s_pad0", "X2_eliza-5s-fp32_pad0"),
]
rng = random.Random()
rows = []
for text in texts:
    for pair, a, x in pairs:
        order = [a, x]
        rng.shuffle(order)
        for slot, config in enumerate(order, start=1):
            src = listen / f"{config}_{text}.wav"
            dst = blind / f"{text}_{pair}_{slot}.wav"
            shutil.copyfile(src, dst)
            rows.append({"file": dst.name, "config": config})
with open(key_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "config"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} files in {blind}; key in {key_path}")
