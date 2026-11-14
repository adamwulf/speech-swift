# Kokoro end-to-end CoreML export (masked LSTMs)

Recipe for exporting Kokoro-82M as one end-to-end CoreML graph that `KokoroTTSModel` loads through `fromPretrained(modelFile:)`. Its bidirectional LSTMs hold their state on padded steps, so predicted durations match upstream Kokoro. The step-by-step plan is [docs/plans/kokoro-e2e-5s-export-ab.md](../../docs/plans/kokoro-e2e-5s-export-ab.md), and the findings are in [docs/research/2026-11-13-kokoro-long-window-and-exporters.md](../../docs/research/2026-11-13-kokoro-long-window-and-exporters.md).

The exporter itself is not vendored here. It is elizaOS/eliza's `packages/training/scripts/kokoro/coreml/export_e2e_coreml.py` (MIT, Copyright (c) 2026 Shaw Walters and elizaOS Contributors) at commit `b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3`, run against mattmireles/kokoro-coreml (Apache-2.0) at `2a75c780664b2ea0430eef6829c04f4704555623`.

## Files

| File | |
|---|---|
| `export_e2e_coreml.local.patch` | Our changes to eliza's exporter; apply with `patch -p1` in the folder that holds the exporter |
| `custom_stft.coreml.patch` | eliza's patch to mattmireles' `kokoro/custom_stft.py` |
| `parity_coreml.py` | CoreML vs torch: durations, waveform and spectral parity, on `CPU_ONLY` and `CPU_AND_NE` |
| `diag_coreml.py` | Where CoreML diverges: F0, harmonic source, audio |
| `diag_sinegen.py` | Inside the harmonic source: rad, resampled and cumulative phase, sin, sine waves, voiced mask |
| `check_nearest.py` | Confirms the constant-index nearest resample equals `F.interpolate(mode="nearest")` |
| `make_blind_set.py` | Shuffles A/B WAV pairs under neutral names, with a separate key |

## The local changes

All of them are needed with coremltools 8.3 on macOS 27:

1. **The harmonic source.** `(f0 / sr) % 1` converts wrongly and zeroes it; the patch uses `x - floor(x)`, the same value.
2. **The masked AdaIN's frame mask.** It is stretched with `F.interpolate(mode="nearest")`, which converts to an integer-scale upsample one sample short; the patch uses a constant-index `index_select`.
3. **Loading during conversion.** `ct.convert(compute_units=ALL)` loads the model for the GPU, which aborts in MPSGraph; the patch loads it for `CPU_AND_NE`.
4. **`--precision fp16mix`.** fp16 everywhere except the harmonic source's phase path, meaning every op with a rank-3 output that has the 9-harmonic axis. The accumulated phase needs more than fp16's 11 bits.

## Run

Python 3.12 with torch 2.5.0, coremltools 8.3.0, numpy 1.26.4, transformers 4.44.2, `misaki[en]` and soundfile:

```bash
export KOKORO_COREML_REF=<mattmireles checkout, with checkpoints/config.json and checkpoints/kokoro-v1_0.pth>
export KOKORO_COREML_HEXGRAD=<hexgrad/Kokoro-82M download, with voices/af_heart.pt>
python export_e2e_coreml.py --stage torch --seconds 5 --tokens 128               # torch parity only
python export_e2e_coreml.py --stage full --seconds 5 --tokens 128 --precision fp32 --out out/fp32
python parity_coreml.py export_e2e_coreml.py out/fp32/kokoro_5s.mlpackage
xcrun coremlcompiler compile out/fp32/kokoro_5s.mlpackage out/fp32/
```

The compiled `.mlmodelc` loads next to the aufklarer vocabulary, G2P and voice files, with `fromPretrained(modelFile:cacheDir:offlineMode: true)`. `E2EKokoroListeningSetTests` renders an A/B listening set from such a folder (set `KOKORO_LISTENING_DIR` and `KOKORO_AB_BUNDLE_DIR`).
