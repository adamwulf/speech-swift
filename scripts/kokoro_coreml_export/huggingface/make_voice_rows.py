"""Add Kokoro's per-length style rows to voice JSON files.

Usage: python make_voice_rows.py <hexgrad voices dir> <voice json dir> <out dir>

<hexgrad voices dir> holds hexgrad/Kokoro-82M's voices/<id>.pt tensors,
[510, 1, 256] each: row i is the style for an utterance of i + 1 phonemes
(upstream Kokoro picks pack[len(phonemes) - 1]). <voice json dir> holds the
mean-only voice files (aufklarer/Kokoro-82M-CoreML's voices/<id>.json, whose
"embedding" is the mean of those rows). For every voice JSON with a matching
tensor, writes <out dir>/<id>.json: the original keys unchanged, plus the rows
under "1" ... "510", the keys KokoroVoice reads. Fails if a voice has no
tensor, or if its embedding is not the mean of the rows.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

pt_dir, json_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
problems = []
for voice_path in sorted(json_dir.glob("*.json")):
    name = voice_path.stem
    pt_path = pt_dir / f"{name}.pt"
    if not pt_path.is_file():
        problems.append(f"{name}: no {pt_path.name}")
        continue
    pack = torch.load(pt_path, map_location="cpu", weights_only=True).float().numpy()
    if pack.shape != (510, 1, 256):
        problems.append(f"{name}: tensor shape {pack.shape}")
        continue
    rows = pack[:, 0, :]
    voice = json.load(open(voice_path))
    embedding = np.array(voice["embedding"][:256], dtype=np.float32)
    diff = float(np.abs(embedding - rows.mean(axis=0)).max())
    if diff > 1e-4:
        problems.append(f"{name}: embedding differs from the mean of the rows by {diff:.6f}")
        continue
    for i in range(510):
        voice[str(i + 1)] = [float(v) for v in rows[i]]
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(voice, f)
    print(f"{name}: 510 rows, embedding matches the mean (max diff {diff:.2e})")
if problems:
    print("\n".join(problems))
    raise SystemExit(1)
