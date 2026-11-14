# Kokoro end-to-end CoreML export (masked LSTMs)

Recipe for exporting Kokoro-82M as one end-to-end CoreML graph that `KokoroTTSModel` loads through `fromPretrained(modelFile:)`. Its bidirectional LSTMs hold their state on padded steps, so predicted durations match upstream Kokoro. The step-by-step plan is [docs/plans/kokoro-e2e-5s-export-ab.md](../../docs/plans/kokoro-e2e-5s-export-ab.md), and the findings are in [docs/research/2026-11-13-kokoro-long-window-and-exporters.md](../../docs/research/2026-11-13-kokoro-long-window-and-exporters.md).

`export_e2e_coreml.py` is elizaOS/eliza's `packages/training/scripts/kokoro/coreml/export_e2e_coreml.py` (MIT, Copyright (c) 2026 Shaw Walters and elizaOS Contributors; see `LICENSE-elizaOS-eliza`) at commit `b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3`, with our changes applied. It runs against mattmireles/kokoro-coreml (Apache-2.0) at `2a75c780664b2ea0430eef6829c04f4704555623`.

## Files

| File | |
|---|---|
| `export_e2e_coreml.py` | The exporter as run: eliza's, with the changes below |
| `LICENSE-elizaOS-eliza` | eliza's MIT license, which covers `export_e2e_coreml.py` |
| `export_e2e_coreml.local.patch` | The same changes as a diff against eliza's file (`patch -p1`) |
| `custom_stft.coreml.patch` | eliza's patch to mattmireles' `kokoro/custom_stft.py` |
| `parity_coreml.py` | CoreML vs torch: durations, waveform and spectral parity, on `CPU_ONLY` and `CPU_AND_NE` |
| `diag_coreml.py` | Where CoreML diverges: F0, harmonic source, audio |
| `diag_decoder.py` | Per decoder / vocoder stage under a precision: NaN, inf, magnitude, correlation with torch |
| `diag_sinegen.py` | Inside the harmonic source: rad, resampled and cumulative phase, sin, sine waves, voiced mask |
| `check_nearest.py` | Confirms the constant-index nearest resample equals `F.interpolate(mode="nearest")` |
| `make_blind_set.py` | Shuffles A/B WAV pairs under neutral names, with a separate key |

## The local changes

All of them are needed with coremltools 8.3 on macOS 27:

1. **The harmonic source.** `(f0 / sr) % 1` converts wrongly and zeroes it; the patch uses `x - floor(x)`, the same value.
2. **The masked AdaIN's frame mask.** It is stretched with `F.interpolate(mode="nearest")`, which converts to an integer-scale upsample one sample short; the patch uses a constant-index `index_select`.
3. **Loading during conversion.** `ct.convert(compute_units=ALL)` loads the model for the GPU, which aborts in MPSGraph; the patch loads it for `CPU_AND_NE`.
4. **`--precision fp16mix`.** fp16 everywhere except the harmonic source's phase path: every op with a rank-3 input or output on a kept axis (`KOKORO_FP32_AXES`, default 9, the harmonic axis). The accumulated phase needs more than fp16's 11 bits. This alone is not enough: in fp16 the text side gives NaN durations on CPU and wrong ones on the Neural Engine.
5. **`--precision fp16dec`.** Everything up to the decoder input (the `asr` alignment matmul: BERT, every masked LSTM, the duration head, the F0/N predictor, the text encoder) stays fp32. The decoder and vocoder run in fp16, except the kept axes. Use `KOKORO_FP32_AXES=9,11` to also keep the vocoder's STFT bins (n_fft 20) in fp32.
6. **The masked AdaIN statistics are made fp16-safe**, with identical math. They are means divided by the valid fraction rather than sums divided by a count, since a sum over 24,001 samples overflows fp16. The variance is taken on `(x - mean) / 16`.
7. **`--skip-load`** saves without loading. The first Neural Engine compile of an fp16 graph can take hours, so parity runs with `parity_coreml.py ... cpu`.

## Run

Python 3.12 with torch 2.5.0, coremltools 8.3.0, numpy 1.26.4, transformers 4.44.2, `misaki[en]` and soundfile:

```bash
export KOKORO_COREML_REF=<mattmireles checkout, with checkpoints/config.json and checkpoints/kokoro-v1_0.pth>
export KOKORO_COREML_HEXGRAD=<hexgrad/Kokoro-82M download, with voices/af_heart.pt>
python export_e2e_coreml.py --stage torch --seconds 5 --tokens 128               # torch parity only
python export_e2e_coreml.py --stage full --seconds 5 --tokens 128 --precision fp32 --out out/fp32
python parity_coreml.py export_e2e_coreml.py out/fp32/kokoro_5s.mlpackage
xcrun coremlcompiler compile out/fp32/kokoro_5s.mlpackage out/fp32/

# fp16 decoder (fp32 text side, phase path and STFT):
KOKORO_FP32_AXES=9,11 python export_e2e_coreml.py --stage full --seconds 5 --tokens 128 \
    --precision fp16dec --skip-load --out out/fp16dec
python parity_coreml.py export_e2e_coreml.py out/fp16dec/kokoro_5s.mlpackage 128 200 cpu
```

The compiled `.mlmodelc` loads next to the aufklarer vocabulary, G2P and voice files, with `fromPretrained(modelFile:cacheDir:offlineMode: true)`. `E2EKokoroListeningSetTests` renders an A/B listening set from such a folder (set `KOKORO_LISTENING_DIR` and `KOKORO_AB_BUNDLE_DIR`).
