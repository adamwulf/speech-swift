# Kokoro split Core ML pipeline on this Mac

2026-11-14. Evaluation only; no changes to `Sources/` or the package manifest.

**A split export is worth prototyping.** The tested pipeline genuinely places its large stages on the Neural Engine, also runs on the GPU without the fused graphs' MPSGraph abort, and delivers much lower warm latency. **Sound-quality parity remains unproven:** the blind sets are ready, but several short replies have materially shorter speech spans. Keep the published models as the quality reference until the blind pairs have been rated and a matched-input comparison has isolated that difference.

## Scope and provenance

- Source: `mattmireles/kokoro-coreml`, Apache-2.0, commit **`b23caa862f0585c2c65187d0fc97e0f8311041dc`**.
- Hugging Face snapshot: **`f50e74cd9465e3cdcfbeba95a0df67530c312b34`**, last modified 2026-10-25. Its full SDK manifest identifies artifact revision `0c1d78b47d19f39be6bc0cf0d07316822ae879d4` and SDK construction commit `e8bf3d4e80d7dd30ed58e343c1527fad09cc9d1f`; these are manifest provenance, not the source/snapshot revisions tested.[^1][^2]
- Host: Apple Silicon Mac, **macOS 27.3, build 26C5035r**. Chip model and RAM were not recorded because the host-information commands were unavailable. These are the same machine's reference timings supplied for our exports: about 200 ms per 5 s fp16 CPU pass and about 1 s per 15 s fp16 CPU pass.
- Separate release-mode Swift package: `/tmp/kokoro-mattmireles/`, depending on the isolated checkout at `/tmp/kokoro-mattmireles-upstream/`. Downloads remain in `/tmp/kokoro-mattmireles-models/`; evaluation scripts and compiled caches are separate. No upstream exporter or download script was executed.
- Text preparation uses the upstream SDK and pinned MisakiSwift `3a27756a780fc138e328a96e533fb440a3419d5b`. Scratch resolution selected MLX Swift 0.31.4, MLXUtilsLibrary 0.0.6, ZIPFoundation 0.9.20 and swift-numerics 1.1.1.
- All model jobs and builds ran sequentially, with the host slot coordinated before starting. Each probe/configuration had a 1,200-second timeout; none reached it. No full test suite, uploads, pushes or GitHub comments were run.

An initial sandboxed duration-front probe logged `sandbox_extension_issue_file: Operation not permitted`, reported all 961 ops on CPU and supplied no costs. **That result is invalid for device placement.** The same binary run on the host reported 958 ANE ops with complete costs. All placement, inference and timing results below come from the host-side runs.

## Published bundles and precision

The snapshot contains **94 actual `.mlpackage` paths**, excluding Hugging Face download-cache metadata. Of these, 55 are under `coreml/`; the other directories publish additional shapes and copies. The current full SDK profile supplies five audio buckets, while the starter supplies only the 15 s bucket and `af_heart`. The full profile contains 28 English voice tables, vocabulary, harmonic-source weights, and optional legacy Core ML G2P assets.[^1][^2]

Let `B = 3,7,10,15,30` seconds and `F = 120,280,400,600,1200` duration frames respectively.

| Published paths, relative to snapshot | Shapes / buckets | Precision and role |
|---|---|---|
| `coreml/kokoro_duration_ane_front_t128.mlpackage` | 128 token slots | fp16 floating-point internals; int32 token/mask API |
| `coreml/kokoro_duration_back_t128.mlpackage` | 128 token slots | fp16 recurrent/linear math; integer duration/index operations |
| `coreml/kokoro_f0ntrain_front_t{F}.mlpackage` | All five frame buckets | fp16 masked recurrent stage |
| `coreml/kokoro_f0ntrain_ane_back_t{F}.mlpackage` | All five frame buckets | fp16 convolutional pitch/noise branches |
| `coreml/kokoro_decoder_pre_{B}s.mlpackage` | All five audio buckets | Predominantly fp16 internals, float32 input/output API; one non-constant float32 tensor output in the serialized program |
| `coreml/kokoro_generator_fold_poly_{B}s.mlpackage` | All five audio buckets | fp16 folded generator; polynomial Snake; selected on this Mac |
| `coreml/kokoro_generator_fold_{B}s.mlpackage` | All five audio buckets | fp16 alternative using native sine; inventoried, not selected on this Mac |
| `coreml/hybrid/hybrid_generator_r{R}.mlpackage`, `plan.json`, `adain.bin` | R = 2,3,4,6,10,11 rows | fp16 Core ML convolution functions plus custom Metal normalization/activation work |
| `coreml/kokoro_duration_t{T}.mlpackage`, `kokoro_duration.mlpackage`, `kokoro_duration_multifunction.mlpackage` | T = 32,64,128,256,320,384,512; legacy alias; multifunction | Older whole-stage alternatives, mixed fp16/fp32 |
| `coreml/kokoro_f0ntrain_t{F}.mlpackage`, `kokoro_f0ntrain_multifunction.mlpackage` | All five frame buckets | Older whole-stage alternatives, mixed fp16/fp32 |
| `coreml/kokoro_decoder_pre_multifunction.mlpackage` | Five functions | Older mixed fp16/fp32 alternative |
| `coreml/kokoro_decoder_har_post_{B}s.mlpackage`, `kokoro_decoder_har_post_range.mlpackage` | Five fixed buckets; flexible time axis | Older unfolded GPU alternatives, predominantly fp16 with some fp32 |

Additional directories are `coreml-dpre-ane/` (five decoder-pre packages), `coreml-f0n-split/` (ten pitch/noise packages), `coreml-fold/` (ten generator packages), `coreml-hybrid/` (six hybrid packages), and `coreml-dur-ane/` (front/back pairs at **64,128,256,512** tokens). Only the t128 split pair is in the current SDK profile. The measured stage set below is the **28 packages used by the Mac's full-profile fast path**, not every older alternative.

Precision was checked from serialized MIL tensor types without loading models. “fp16” does not mean every index, mask, API boundary or Swift calculation is fp16. The generator and split pitch branches are fp16; the harmonic phase accumulation and inverse STFT are outside those graphs.[^3][^4][^6][^7]

## Pipeline and alignment

For a selected bucket, F is its 40 Hz frame capacity. Floating-point types below describe the published interfaces.

| Stage | Inputs | Outputs | Default execution |
|---|---|---|---|
| Duration front | `input_ids`, `attention_mask`: int32 [1,128] | `d_en`, `t_cnn`: fp16 [1,512,128] | ANE, with three small CPU ops |
| Duration back | Those two features, fp16 `ref_s` [1,256], `speed` [1], int32 mask | int32 `pred_dur` [1,128]; fp16 `d` [1,128,640], `t_en` [1,512,128], `s` [1,128], `ref_s_out` [1,256] | CPU |
| **Alignment in Swift** | Valid predicted durations, `d`, `t_en` | `en` [1,640,validFrames], `asr` [1,512,validFrames]; then padding and valid-frame masks | CPU |
| F0/noise front | fp16 `en` [1,640,F], `mask` [1,1,F] | fp16 `x` [1,512,F] | CPU |
| F0/noise back | fp16 `x`, `s` [1,128], mask | fp16 `F0_pred`, `N_pred` [1,2F] | ANE |
| Decoder-pre | float32 `asr` [1,512,F], `f0` and `n_input` [1,1,2F], `ref_s` [1,256], mask | float32 `x_pre` [1,512,2F] | ANE |
| Harmonic source | Predicted F0, learned nine-weight projection/bias, seeded noise | 22-channel harmonic spectral features | Swift/Accelerate; double-precision phase; overlaps decoder-pre |
| Folded generator | fp16 folded `x_pre` [R,512,1,W], `har` [R,22,1,60W+5], `ref_s`, statistics/zero masks at both resolutions | fp16 `logits` [R,22,1,60W+5] | ANE below 7 s and as fallback |
| Hybrid generator | Same logical features, row geometry and normalization parameters; per-convolution tensors passed between 53 Core ML functions and Metal kernels | Spectral logits, followed by the audio reconstruction path | Core ML convolutions on ANE, custom Metal work on GPU |
| Reconstruction and joining | Logits and valid length | Trimmed 24 kHz float mono PCM; joined chunks | Swift/Accelerate inverse STFT and stream joiner |

Alignment is **not in a Core ML graph**. The hot path repeats each token's feature vector according to its duration; it only constructs an explicit one-hot alignment matrix for diagnostic dumps. The waveform length is derived from valid duration frames, at 600 samples per frame.[^6]

The important masking difference from the problematic FluidAudio export is explicit in source: `MaskedBidirectionalLSTM` reverses only the valid prefix with gathers, uses two native one-direction LSTMs, and masks outputs. It preserves packed-sequence semantics without unrolling every recurrent step. Pitch/decoder normalization also excludes padded frames. Folding bounds the generator's axes and uses overlapping rows with separate masks for valid samples and ownership of normalization statistics.[^3][^4][^5][^7]

The raw-text SDK uses Misaki, BOS/EOS **0**, and an utterance-length voice row. Our supplied references use the package's phonemizer and BOS/EOS **1/2**. Those preparation differences, voice selection, chunk boundaries and joining are part of this pipeline A/B; it is **not a matched-token export-parity test**. The SDK's 128-token ceiling causes two or three chunks in every long fixture here despite the larger acoustic buckets.[^8][^9][^12]

## Placement and first load

Each row is a fresh probe process. Counts exclude constants and include nested blocks. Percentages are **CPU / GPU / ANE**. Estimated cost is Core ML's normalized prediction, not a measured device-time or power share. Each plan was requested under both `.cpuAndNeuralEngine` (“NE”) and `.all`.

“Load” is the first `MLModel(contentsOf:configuration:)` call in that probe; “compile” is `MLModel.compileModel(at:)` into the new scratch cache. NE was tested first, then `.all` reused the compiled package. System-wide Core ML caches were not cleared, so these are observed first loads, not guaranteed cache-free compiler benchmarks. Compute-plan acquisition followed loading and took additional time (up to 9.65 s).

| Package (without extension) | Ops | NE op % C/G/N | NE cost % C/G/N | .all op % C/G/N | .all cost % C/G/N | Load s, NE / .all | Package compile s |
|---|---:|---|---|---|---|---|---:|
| `kokoro_decoder_pre_10s` | 284 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.730 / 0.696 | 0.041 |
| `kokoro_decoder_pre_15s` | 284 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.764 / 0.202 | 0.042 |
| `kokoro_decoder_pre_30s` | 284 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.709 / 0.186 | 0.042 |
| `kokoro_decoder_pre_3s` | 284 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.594 / 0.574 | 0.044 |
| `kokoro_decoder_pre_7s` | 284 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.722 / 0.696 | 0.042 |
| `kokoro_duration_ane_front_t128` | 961 | 0.31/0.00/99.69 | 0.16/0.00/99.84 | 0.31/0.00/99.69 | 0.16/0.00/99.84 | 3.018 / 2.968 | 0.093* |
| `kokoro_duration_back_t128` | 146 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.118 / 0.103 | 0.033 |
| `kokoro_f0ntrain_ane_back_t120` | 314 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.559 / 0.555 | 0.045 |
| `kokoro_f0ntrain_ane_back_t1200` | 314 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.641 / 0.647 | 0.043 |
| `kokoro_f0ntrain_ane_back_t280` | 314 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.585 / 0.587 | 0.044 |
| `kokoro_f0ntrain_ane_back_t400` | 314 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.600 / 0.564 | 0.042 |
| `kokoro_f0ntrain_ane_back_t600` | 314 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.00/0.00/100.00 | 0.619 / 0.613 | 0.044 |
| `kokoro_f0ntrain_front_t120` | 26 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.050 / 0.039 | 0.026 |
| `kokoro_f0ntrain_front_t1200` | 26 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.041 / 0.038 | 0.023 |
| `kokoro_f0ntrain_front_t280` | 26 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.042 / 0.040 | 0.025 |
| `kokoro_f0ntrain_front_t400` | 26 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.040 / 0.038 | 0.024 |
| `kokoro_f0ntrain_front_t600` | 26 | 100.00/0.00/0.00 | 100.00/0.00/0.00 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 0.040 / 0.038 | 0.024 |
| `kokoro_generator_fold_poly_10s` | 1928 | 0.05/0.00/99.95 | 0.09/0.00/99.91 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 5.301 / 0.724 | 0.131 |
| `kokoro_generator_fold_poly_15s` | 1928 | 0.05/0.00/99.95 | 0.09/0.00/99.91 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 5.508 / 0.709 | 0.138 |
| `kokoro_generator_fold_poly_30s` | 1928 | 0.05/0.00/99.95 | 0.09/0.00/99.91 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 9.289 / 0.699 | 0.136 |
| `kokoro_generator_fold_poly_3s` | 1928 | 0.05/0.00/99.95 | 0.08/0.00/99.92 | 0.05/0.00/99.95 | 0.08/0.00/99.92 | 5.486 / 5.350 | 0.137 |
| `kokoro_generator_fold_poly_7s` | 1928 | 0.05/0.00/99.95 | 0.08/0.00/99.92 | 0.00/100.00/0.00 | 0.00/100.00/0.00 | 5.270 / 0.721 | 0.140 |

*The duration-front package was first compiled in the discarded sandbox probe (0.093 s); host-side loading reused it. Its valid first host load was 3.018 s.

The hybrid package contains **53 functions and 83 non-constant operations**. Loading its default function and counting every function initially left 81 operations without placement. The corrected measurement selected each function explicitly. All 636 named-function probes (6 row sizes × 2 configurations × 53 functions) completed, with no missing device assignments.

| Hybrid rows | NE ops C/G/N % | .all ops C/G/N % | .all cost, per function | Later sum of 53 function loads s, NE / .all |
|---|---|---|---|---|
| 2 | 0 / 0 / 100 | 0 / 45.78 / 54.22 | 31 functions 100% ANE; 22 functions 100% GPU | 3.922 / 3.742 |
| 3 | 0 / 0 / 100 | 0 / 69.88 / 30.12 | 21 ANE; 32 GPU, each 100% | 3.738 / 3.252 |
| 4 | 0 / 0 / 100 | 0 / 89.16 / 10.84 | 7 ANE; 46 GPU, each 100% | 3.769 / 3.175 |
| 6 | 0 / 0 / 100 | 0 / 89.16 / 10.84 | 7 ANE; 46 GPU, each 100% | 3.963 / 3.299 |
| 10 | 0 / 0 / 100 | 0 / 98.80 / 1.20 | 1 ANE; 52 GPU, each 100% | 4.365 / 3.243 |
| 11 | 0 / 0 / 100 | 0 / 98.80 / 1.20 | 1 ANE; 52 GPU, each 100% | 4.060 / 3.170 |

Under NE, every hybrid function's estimated cost is **100% ANE**, 0% CPU/GPU. Costs normalize independently per function, so an aggregate hybrid cost percentage cannot be inferred by summing them; the table gives the actual per-function allocations. Custom Metal kernels and Swift DSP are outside `MLComputePlan`. The stock hybrid explicitly loads its convolution functions for ANE and executes its other work through Metal.

Initial default-function package loads, before the named-function pass:

| Rows | Package compile s | First default-function load s, NE / .all |
|---|---:|---|
| 2 | 0.041 | 0.058 / 0.072 |
| 3 | 0.041 | 0.059 / 0.080 |
| 4 | 0.042 | 0.059 / 0.075 |
| 6 | 0.044 | 0.060 / 0.074 |
| 10 | 0.042 | 0.057 / 0.054 |
| 11 | 0.041 | 0.061 / 0.056 |

**No tested fixed split model aborted under `.all`, and actual GPU-enabled synthesis completed.** The older flexible `decoder_har_post_range` was not exercised: upstream already gates that path off on macOS 27 because of a documented MPSGraph abort. This result establishes that the fixed split path works here, not that every published alternative is safe.[^11]

## Warm latency and time to first audio

Each text/configuration had one complete discarded run, followed by three timed runs. Medians below include text preparation, model inference, joining, and the first nonempty streaming callback; WAV writing is excluded. The short fixture is “Sure, I can do that.” The long fixture is “basement,” yielding three chunks (7/7/3 s buckets) and 12.865 s of audio. The 3 s short bucket yields 1.800 s.

The SDK deliberately preserves measured placement for split/folded fast paths under every policy except CPU-only. A superficial `.all` policy override would therefore be misleading. The two uniform configurations use a scratch implementation of the public model-provider interface, loading **every Core ML stage with the requested units**, retaining the upstream text preparation, executor, masks and stream joiner, and using the folded generator. No upstream or package source was patched.[^10]

| Configuration | Short total / first audio ms | Short total range ms | Basement total / first audio ms | Basement total range ms |
|---|---:|---:|---:|---:|
| Stock SDK, hybrid enabled | 45.27 / 45.24 | 44.79–48.63 | 393.64 / 181.62 | 383.52–398.63 |
| Stock SDK, hybrid disabled; folded ANE generator | 45.99 / 45.97 | 45.94–50.71 | 247.96 / 107.77 | 245.66–248.25 |
| Stock SDK, CPU-only | 137.56 / 137.53 | 135.11–140.63 | 747.56 / 313.05 | 742.97–753.91 |
| Uniform CPU + Neural Engine, folded | 45.05 / 45.04 | 44.71–45.89 | 245.91 / 107.48 | 244.76–246.84 |
| Uniform .all, folded | 57.88 / 57.87 | 57.15–66.17 | **134.53 / 55.29** | 129.35–138.62 |

The sub-millisecond short-text differences among the ANE configurations are too small to justify calling one decisively faster. The blind A/B set uses the stock configuration. Uniform `.all` is the fastest tested setting on the long benchmark and is used for the long blind set.

The stock hybrid's “basement” cost persists after warm-up: its two 7 s chunks use different row geometries, and the SDK retains only one hybrid generator. It reloads those functions as geometries alternate; the logged padding/load span is about 123–132 ms per affected chunk. Its generator computation itself is faster (about 34–44 ms) than folded ANE (about 94 ms), but the reload cost reverses that advantage. This is a runtime residency/shape issue, not slow ANE convolution.[^8]

The subsequent six-text `.all` listening run produced:

| Long label | Total ms | First audio ms | Bucket sequence |
|---|---:|---:|---|
| build | 141.10 | 57.39 | 7 / 3 / 3 |
| basement | 127.13 | 51.92 | 7 / 7 / 3 |
| airport | 123.86 | 49.05 | 7 / 7 / 3 |
| next-step | 82.11 | 48.57 | 7 / 7 |
| saturday | 82.04 | 48.59 | 7 / 7 |
| dentist | 82.12 | 48.69 | 7 / 7 |

This is a separate warm run, not additional samples silently pooled into the benchmark table. The stock hybrid is faster on some of these individual sentences (about 69–71 ms for the last three); `.all` is a consistent long-set configuration, not the winner for every text.

Against the supplied reference timings, roughly **45 ms versus 200 ms** for a short reply is about 4.4× faster, and **82–141 ms versus roughly 1 s per 15 s pass** is a substantial long-text improvement. These are pipeline comparisons with different buckets and chunking, not isolated identical-graph speedups. First audio on the long set arrives in about 49–57 ms, well before our one-pass 15 s reference can return.

The discarded first short runs took 10.43 s for the stock SDK and 11.60 s for uniform `.all`, including lazy loading/compilation and first prediction. The longest individual stage load was 9.29 s. Nothing approached the 20-minute cap or our fused fp16 graphs' 70+ minute ANE compile.

## Duration and numerical sanity

All **112 synthesis runs** completed with finite float samples, **zero samples at or beyond ±1**, and maximum absolute amplitude **0.724048**. Saved candidates are 24 kHz, 16-bit PCM, mono WAVs, written without gain normalization. The selected 14 candidates also have no saturated PCM samples.

The first eight rows compare `M_default` against the supplied burst-fixed `X3c` renders; the last six compare `M_all` against burst-fixed `L15h`.

| Text label | Split duration s | Our duration s | Split / ours | Split peak | Saturated PCM samples |
|---|---:|---:|---:|---:|---:|
| short | 1.800 | 1.925 | 0.935 | 0.425 | 0 |
| long | 8.940 | 8.120 | 1.101 | 0.378 | 0 |
| yes | 1.350 | 0.850 | 1.588 | 0.380 | 0 |
| hello | 2.175 | 2.525 | 0.861 | 0.421 | 0 |
| question | 3.750 | 4.375 | 0.857 | 0.546 | 0 |
| numbers | 7.025 | 7.085 | 0.992 | 0.419 | 0 |
| commas | 5.200 | 5.835 | 0.891 | 0.435 | 0 |
| paragraph | 10.815 | 11.345 | 0.953 | 0.377 | 0 |
| build | 8.940 | 7.950 | 1.125 | 0.378 | 0 |
| basement | 12.865 | 11.395 | 1.129 | 0.720 | 0 |
| airport | 11.590 | 10.100 | 1.148 | 0.503 | 0 |
| next-step | 10.420 | 9.750 | 1.069 | 0.405 | 0 |
| saturday | 10.020 | 9.350 | 1.072 | 0.632 | 0 |
| dentist | 9.970 | 9.025 | 1.105 | 0.473 | 0 |

Total duration alone hides an important difference. Measuring 10 ms RMS windows at 0.002 amplitude, and checking again at 0.01:

- **Short replies have shorter speech envelopes, not merely trimmed outer silence.** Excluding outer silence, “short” spans 1.06 s versus 1.17 s, “hello” 1.42 versus 1.81 s, “question” 2.93 versus 3.57 s, and “commas” 4.42 versus 5.04 s. Their span ratios are 0.91, 0.78, 0.82 and 0.88. The higher RMS threshold gives the same direction. These are energy spans, not phonetic alignments: they suggest faster delivery, but different pronunciations or omissions have not been excluded. Listen for rushed or incomplete speech.
- **“Yes.” is longer because of silence.** Its speech span is 0.47 versus 0.50 s, while its lead-in is 0.40 versus 0.03 s and tail 0.48 versus 0.32 s.
- **Long texts contain more internal silence.** Their speech-envelope spans are 1.08–1.18× ours, while the number of active 10 ms windows is roughly comparable: basement 9.11 versus 9.16 s, airport 8.01 versus 7.95 s. With two or three upstream chunks, extra pauses are a plausible explanation; this is not evidence that all speech is slower.

Time to first audio above means **buffer availability**, not audible speech onset. If playback starts immediately, the stock “Yes.” buffer's roughly 400 ms lead-in would put its speech onset near 445 ms, versus roughly 230 ms using our 200 ms reference pass and 30 ms lead-in. That is an estimate from the waveform and supplied pass timing, not a measured playback test. Faster inference does not automatically make every reply sound sooner.

These differences do **not** establish the FluidAudio unmasked-LSTM failure: this exporter explicitly implements valid-prefix recurrent masking. They also do **not** establish equal quality. Frontend phonemes, BOS/EOS, length-dependent voice rows, precision and chunking remain confounded. A matched-token/style/boundary comparison against our export or a packed PyTorch reference is needed before assigning the faster short speech to a particular cause.[^5][^8][^9]

The selected `.all` “saturday” render is one 25 ms duration frame longer than stock; the benchmark short/basement durations match across all five settings.

## Blind listening deliverables

All 14 stock candidates, six selected `.all` long candidates, diagnostic alternatives and the 14 copied reference WAVs are under:

`/tmp/kokoro-export/listen-mattmireles/`

- **A/B set:** `blind/` — 16 WAVs, eight pairs, `<text>_split_1.wav` and `_2.wav`. Stock `M_default` against `X3c_eliza-5s-fp16_cpu`.
- **Long set:** `blind-long/` — 12 WAVs, six pairs with the same neutral suffixes. Uniform `M_all` against `L15h_eliza-15s-fp16_cpu`.
- Keys: `/tmp/kokoro-export/blind-key-mattmireles.csv` and `/tmp/kokoro-export/blind-key-mattmireles-long.csv`. **Neither key was opened or summarized.**

The existing `make_blind_set.py` was run separately for the default eight labels and for `texts=build,basement,airport,next-step,saturday,dentist`. References were copied from exactly `listen-ab-burstfix3/` and `listen-long-burstfix2/`; they were not regenerated. The text labels and sentences came from the existing listening fixture.[^12]

### Blind results

| Set | Split pipeline preferred | Ours preferred | Tie |
|---|---|---|---|
| Eight A/B texts: stock `M_default` vs our 5 s fp16 | 5 (hello, paragraph, question, short, "Yes.") | 2 (commas, long) | 1 (numbers) |
| Six long sentences: `.all` `M_all` vs our 15 s fp16 | 0 | 6 (saturday only "slightly", "probably equally good") | 0 |

- **Numbers pair:** the tie is not a fair comparison. Our side read the digits raw, because this package's English front end does not spell out numbers. The split SDK's Misaki front end does.
- **Paragraph:** the split render was preferred "even though it has more pauses".
- **Long sentences:** our one-pass 15 s render was preferred every time. That fits the split SDK's 128-token ceiling, which cuts these sentences into two or three chunks with pauses between them.

### Per-length voice rows and BOS/EOS

The aufklarer `af_heart` embedding this package loads is exactly the mean of hexgrad's 510 per-length rows (max difference 0.000000). For long utterances the matching row is close to that mean, at cosine 0.999 for 120 phonemes. For short ones it is not: 0.94 at 60 phonemes, 0.74 at 20, 0.49 at 5. A voice file with all 510 rows was built from hexgrad's `af_heart.pt`, keeping the mean as `embedding`; `KokoroVoice` already selects `rows[phonemes - 1]` when a file has them. The test then re-rendered the eight A/B texts with our 5 s fp16 export.

- **Rows make short replies faster.** On the 15 s graph (one pass, no chunking): hello 2.52 s to 1.95 s, short 1.93 s to 1.62 s, question 4.00 s to 3.55 s; long texts are unchanged. Chunking is unchanged.
- **Blind, our 5 s fp16 with rows vs stock split `M_default`:** split preferred 3 (yes, question "barely", paragraph "close"), ours 2 (long, numbers), 3 ties (short, hello, commas). Without rows the split pipeline had won 5 of 8.
- **"Yes." with BOS/EOS 1/2 sounded like "ess".** The predicted durations are normal (`j` 50-75 ms), but the "y" starts from near silence. BOS/EOS 0, the split SDK's convention, lengthens the start token from 3-5 frames to 17-19, giving about 450 ms of lead-in. With rows and BOS/EOS 0, single-chunk renders match the split pipeline's lengths exactly: yes 1.35 s, short 1.80 s, hello 2.17 s.
- **Blind, rows with BOS/EOS 0 vs rows with 1/2 (both ours):** 0 preferred 3 (yes "still better", numbers, question "slightly"), 1/2 preferred 1 (paragraph, "barely"), 4 ties.

So most of the split pipeline's short-text advantage comes from the voice's per-length rows and the BOS/EOS convention, not the split export. The cost of BOS/EOS 0 is about 400 ms more silence before a short reply's speech. Trimming the first piece's lead-in would remove it.

The short-text preference is not attributable to the split export alone. The two sides also differ in front end (Misaki vs this package's phonemizer), BOS/EOS (0 vs 1/2) and voice selection. The split SDK picks the voice's per-length style row. This package's aufklarer voice files hold only the mean embedding, so every chunk uses the same style. Per-length rows are the cheapest of these to test: `KokoroVoice` already uses rows when a voice file has them, and hexgrad's voice tensors hold all 510.

## Recommendation

Build a split-export prototype while retaining our published models as the reference. The evidence supports three specific architectural choices:

1. Keep masked LSTMs as native forward operations with valid-prefix reversal on CPU; put ALBERT/CNN work in ANE-friendly layouts. Avoid thousands of unrolled recurrent operations.
2. Split pitch prediction, decoder preparation, harmonic phase generation and waveform reconstruction at explicit tensor boundaries. Keep the phase/inverse-STFT work in suitable precision outside the fp16 convolution graphs.
3. Start with the folded generator and explicit per-stage placement. Benchmark GPU placement on this Mac. Add the hybrid generator only if its additional complexity and model-residency policy improve end-to-end latency on representative multi-chunk workloads.

Preserve our phonemizer, boundary convention, voice selection and long-window chunk policy in that prototype so quality comparisons isolate the export. The current upstream SDK's 128-token limit and pause behavior are not reasons to discard our long-window quality gains.

Blind listening (above) preferred the split pipeline on five of eight short texts and our 15 s export on all six long sentences. Before attributing the short-text result to the export, test our graphs with per-length voice rows. Adoption should require matched-input duration/parity tests and blind listening preferences. This experiment clears the **hardware placement, compilation and speed** questions. It leaves the **short-speech rate and perceptual quality** questions open.

## Reproduction evidence

Raw material remains outside the repository: `/tmp/kokoro-mattmireles/Sources/{Plan,Evaluate}/main.swift`, `Package.resolved`, `plans/`, `hybrid-plans/`, `renders/`, `all-long.log`, `spec-inventory.json`, and the WAV directory above. The speech-span check is `speech_sanity.py`. Model descriptions and non-constant MIL types were inspected directly. All five runtime configurations and the supplemental long run exited successfully; 56 initial model probes and 636 corrected named-function probes succeeded. Only this summary report is committed.

[^1]: [Pinned upstream SDK profiles and published path lists](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/scripts/sdk_profiles.json)
[^2]: [Pinned Hugging Face full SDK manifest](https://huggingface.co/mattmireles/kokoro-coreml/blob/f50e74cd9465e3cdcfbeba95a0df67530c312b34/sdk/full/KokoroRuntimeManifest.json)
[^3]: [Duration export: AneFront and fp16 split export](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/scripts/export_duration_ane.py)
[^4]: [F0/noise export: masked Front and ANE-layout Back](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/scripts/export_f0ntrain_split.py)
[^5]: [MaskedBidirectionalLSTM: native recurrence and valid-prefix reversal](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/coreml_export_lstm.py)
[^6]: [executeKokoroSynthesis: alignment, stage interfaces and Swift harmonic source](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift/Sources/KokoroPipeline/KokoroSynthesisExecutor.swift)
[^7]: [FoldGeometry, foldedGeneratorInputs and foldedGeneratorAudio](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift/Sources/KokoroPipeline/FoldedGenerator.swift)
[^8]: [KokoroTTS.prepare and synthesizeStreaming: SDK preparation and 128-token limit](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift-tts/Sources/KokoroTTS/KokoroTTS.swift); [KokoroSDKModelProvider.hybridGenerator: one resident geometry](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift-tts/Sources/KokoroTTS/KokoroSDKModelProvider.swift)
[^9]: [VoiceTable.refS: phoneme-length voice-row selection](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift-tts/Sources/KokoroTTS/VoiceTable.swift); [KokoroPreparedInput: boundary convention](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift/Sources/KokoroPipeline/KokoroPreparedInput.swift)
[^10]: [KokoroComputePolicy: fast-path override semantics](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift-tts/Sources/KokoroTTS/KokoroComputePolicy.swift)
[^11]: [PipelineConstants: bucket selection, polynomial generator preference and macOS 27 flexible-generator gate](https://github.com/mattmireles/kokoro-coreml/blob/b23caa862f0585c2c65187d0fc97e0f8311041dc/swift/Sources/KokoroPipeline/KokoroPipeline.swift)
[^12]: [Listening texts and reference configurations](../../Tests/KokoroTTSTests/KokoroListeningSetE2ETests.swift:E2EKokoroListeningSetTests); [Blind pair generator](../../scripts/kokoro_coreml_export/make_blind_set.py)
