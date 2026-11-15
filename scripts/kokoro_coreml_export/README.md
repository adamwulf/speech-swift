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
| `unblind.py` | Joins a blind key with the listener's tally and counts wins per comparison |
| `noise_floor.py` | Lead-in RMS, quietest audible 10 ms windows (digital silence excluded) and share of digital silence, in dBFS |
| `seam_bursts.py` | Short bursts of sound between runs of digital silence: the fp16 start-of-pass burst check |
| `first_frame.py` | First 25 ms (and 25-30 ms) RMS per configuration: should be silence on every graph |
| `huggingface/` | Canonical model card and publishing steps for `milestonemade/Kokoro-82M-CoreML-Masked` (see `huggingface/PUBLISHING.md`) |

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

# 15 s window (10 s: --seconds 10 --tokens 168, parity 168 400):
KOKORO_FP32_AXES=9,11 python export_e2e_coreml.py --stage full --seconds 15 --tokens 249 \
    --precision fp16dec --skip-load --out out/fp16dec-15s
python parity_coreml.py export_e2e_coreml.py out/fp16dec-15s/kokoro_15s.mlpackage 249 600 cpu
```

On the CPU, the 5 s and 10 s fp16 graphs open each pass with a faint noise burst (12-27 ms); `KokoroTTSModel` silences it (`KokoroLeadInBurst`, see [kokoro-tts.md](../../docs/models/kokoro-tts.md#long-text)).

The compiled `.mlmodelc` loads next to the aufklarer vocabulary, G2P and voice files, with `fromPretrained(modelFile:cacheDir:offlineMode: true)`.

## Regenerating the listening sets

`E2EKokoroListeningSetTests` renders every listening set used in the research notes. It writes `<config>_<text>.wav` (24 kHz, 16-bit) and prints, per configuration and text, the pieces, audio length, render time and pitch stability. It skips unless `KOKORO_LISTENING_DIR` is set.

The bundle folder (`KOKORO_AB_BUNDLE_DIR`) holds the compiled graphs under the names the configurations use: `kokoro_5s.mlmodelc` (soniqo) and `kokoro_eliza_{5,10,15}s_{fp32,fp16}.mlmodelc` (these exports). Next to them go the aufklarer `vocab_index.json`, `g2p_vocab.json`, `us_gold.json`, `us_silver.json`, `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc` and `voices/`. A configuration whose graph is missing is skipped. The FluidAudio configurations download `FluidInference/kokoro-82m-coreml` instead.

| Set | Environment | Texts | Configurations |
|---|---|---|---|
| Warble | `KOKORO_LISTENING_DIR` only | short, long | soniqo `kokoro_5s` vs FluidAudio 5 s / 15 s, with zero phases, CPU only, BOS/EOS 0, averaged voice |
| A/B | + `KOKORO_AB_BUNDLE_DIR` | 8 texts, "Yes." to three sentences | soniqo, these exports (fp32/fp16, CPU/Neural Engine, BOS/EOS 0), FluidAudio 5 s / 15 s, 10 s / 15 s fp16 |
| Long | + `KOKORO_LISTENING_SET=long` | 6 single sentences of 120-200 tokens | soniqo and fp16 5 s, fp32/fp16 10 s and 15 s |
| Parentheses | + `KOKORO_LISTENING_SET=parens` | 4 asides, each also with a comma before the `(` | as Long |

`KOKORO_LISTENING_CONFIGS` (comma-separated configuration names, such as `L15h_eliza-15s-fp16_cpu`) limits a run to those configurations. Example, run from the repository root:

```bash
KOKORO_LISTENING_DIR=/tmp/listen-long KOKORO_AB_BUNDLE_DIR=/tmp/kokoro-bundle KOKORO_LISTENING_SET=long \
KOKORO_LISTENING_CONFIGS=X3c_eliza-5s-fp16_cpu,L15h_eliza-15s-fp16_cpu \
    swift test --filter E2EKokoroListeningSetTests --disable-sandbox
```

Each pass draws new random phases, so renders differ slightly from run to run; durations and seams do not.

The blind loop:

```bash
python make_blind_set.py /tmp/listen-long /tmp/listen-long/blind /tmp/blind-key-long.csv \
    texts=build,basement,airport,next-step,saturday,dentist \
    c15:X3c_eliza-5s-fp16_cpu:L15h_eliza-15s-fp16_cpu
# listen to blind/<text>_c15_1.wav and _2.wav; record pair,preferred,strength rows in a tally CSV
python unblind.py /tmp/blind-key-long.csv /tmp/tally-long.csv
```

Checks after rendering: `seam_bursts.py` and `first_frame.py` for the start-of-pass burst, and `noise_floor.py` for floors. `E2EKokoroComputePlanTests` (`KOKORO_PLAN_DIR`, `KOKORO_PLAN_FILES`) reports where Core ML places each op of a compiled graph.
