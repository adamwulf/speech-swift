"""Op count and Core ML load time of a compiled graph.

Usage: python load_time.py <model.mlmodelc> [units=cpu|ane|all]

Counts the ops in the compiled program (model.mil), all and without `const`,
plus the most common op types. Then copies the graph to a new temporary
folder, so Core ML has no build of it cached under that path, and times the
first load there (cold) and a second load of the same copy (warm).
"""
import collections
import pathlib
import re
import shutil
import sys
import tempfile
import time

import coremltools as ct

src = pathlib.Path(sys.argv[1])
units = {"cpu": ct.ComputeUnit.CPU_ONLY, "ane": ct.ComputeUnit.CPU_AND_NE,
         "all": ct.ComputeUnit.ALL}[sys.argv[2] if len(sys.argv) > 2 else "cpu"]

op = re.compile(r"=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
types = collections.Counter(m.group(1) for line in (src / "model.mil").read_text().splitlines()
                            if (m := op.search(line)))
total = sum(types.values())
print(f"{src.name}: ops={total} non_const={total - types['const']} "
      f"top={dict(types.most_common(10))}")

with tempfile.TemporaryDirectory() as tmp:
    dst = pathlib.Path(tmp) / src.name
    shutil.copytree(src, dst)
    for label in ("cold", "warm"):
        t0 = time.time()
        model = ct.models.CompiledMLModel(str(dst), compute_units=units)
        print(f"{src.name}: {label} load {time.time() - t0:.1f} s")
        del model
