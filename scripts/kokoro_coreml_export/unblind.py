"""Join a blind-set key with the listener's tally and count wins per comparison.

Usage: python unblind.py <key_csv> <tally_csv>

key_csv is what make_blind_set.py wrote (columns file, config). tally_csv has
one row per pair (columns pair, preferred, strength): pair is the blind file
stem without the _1/_2 suffix (e.g. "airport_c15"), preferred is 1, 2 or
"same", and strength is free text ("way better", "slight", ...). Prints each
pair unblinded, then the wins per comparison label (the part after the last
"_" in the pair, e.g. "c15").
"""
import csv
import sys
from collections import defaultdict

key_path, tally_path = sys.argv[1], sys.argv[2]
key = {row["file"]: row["config"] for row in csv.DictReader(open(key_path))}
counts = defaultdict(lambda: defaultdict(int))
for row in csv.DictReader(open(tally_path)):
    pair = row["pair"]
    label = pair.rsplit("_", 1)[1]
    one, two = key[f"{pair}_1.wav"], key[f"{pair}_2.wav"]
    preferred = row["preferred"].strip()
    winner = {"1": one, "2": two}.get(preferred, "tie")
    counts[label][winner] += 1
    print(f"{pair:18} _1={one:30} _2={two:30} -> {winner} ({row.get('strength', '')})")
print()
for label, wins in counts.items():
    print(label, dict(wins))
