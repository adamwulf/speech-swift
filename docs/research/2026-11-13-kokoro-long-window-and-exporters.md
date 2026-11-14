# Kokoro: longer windows, the "warble", and where the exporters are

Research notes from 2026-11-13. They cover three questions:

1. Can `KokoroTTSModel` run a Kokoro graph with a window longer than 5 s, so a whole sentence renders in one pass?
2. Why do FluidAudio's Kokoro graphs sound "warbly" next to the `aufklarer/Kokoro-82M-CoreML` `kokoro_5s` graph?
3. Is the exporter behind `kokoro_5s` public, so it can be run at 10 s and 15 s?

## Summary

- `KokoroTTSModel` now loads any end-to-end Kokoro bundle through `fromPretrained(modelFile:)`. It reads the token slots, the output window and whether a `speed` input exists from the model. The chunk budget follows from the window: 56/64 tokens for `kokoro_5s`, 201/230 for FluidAudio's `kokoro_21_15s`. See [kokoro-tts.md](../models/kokoro-tts.md#other-end-to-end-bundles).
- In listening tests, every configuration of the FluidAudio graphs sounded warbly or odd next to `kokoro_5s`. The likely cause is that FluidAudio's export runs its bidirectional LSTMs over the padding, which `kokoro_5s` does not. Random phases, fp16 on the Neural Engine, BOS/EOS ids and the voice row are not the cause.
- The exporter behind `kokoro_5s` lives in soniqo's private `speech-models` repository and is not public. The closest public match is elizaOS's fused end-to-end CoreML exporter. It masks all six LSTMs and takes the window and token count as flags. The next step, an A/B of a 5 s export against `kokoro_5s`, is planned in [kokoro-e2e-5s-export-ab.md](../plans/kokoro-e2e-5s-export-ab.md).

## 1. Running a 15 s graph

### The bundles compared

| | `kokoro_5s` (aufklarer) | `kokoro_21_15s` (FluidInference) | `kokoro_21_5s` (FluidInference) |
|---|---|---|---|
| `input_ids` | [1, 128] | [1, 249] | [1, 124] |
| `audio` | [1, 1, 120000] (5.0 s) | [1, 1, 372600] (15.5 s) | [1, 1, 175800] (7.3 s) |
| `speed` input | yes | no | no |
| Voice files | `embedding` only | `embedding` + rows `"1"`…`"510"` | same as 15 s |

`vocab_index.json`, `g2p_vocab.json`, `us_gold.json`, `us_silver.json` and the G2P bundles are byte-identical across both repositories (the same git blobs), so token ids agree. In both repositories a voice's `embedding` is the mean of its 510 rows (max difference 4e-8), and aufklarer's `af_heart` embedding equals FluidInference's. FluidAudio's own code picks row `clamp(n - 1, 0, 509)` for `n` phonemes, BOS and EOS not counted, as upstream picks `pack[len(ps) - 1]`.

### Speech rate

Measured with `af_heart` at speed 1 from `audio_length_samples`:

- `kokoro_5s`: 14.5-15.5 tokens per second of output for typical sentences, including about 1 s of lead-in and trailing silence per pass.
- `kokoro_21_15s`: 18.4-19.9 tokens per second, with little silence. A 200-token sentence takes 10.2 s of the 15.3 s trusted window.

Neither graph matches the 10.5 tokens/s that the 5 s budget was described with. The 15 s budget therefore scales speech-core's validated 5 s profile to the window's speech time instead of assuming a rate.

### Latency and loading

Warm `synthesize` times with `.cpuAndNeuralEngine` on one Apple Silicon Mac:

| Text | `kokoro_5s` | `kokoro_21_15s` |
|---|---|---|
| "Yes." (7 tokens) | 380 ms, 1 chunk | 1,200 ms, 1 chunk |
| 60 tokens | 770 ms, 2 chunks | 1,210 ms, 1 chunk |
| 200 tokens | 1,570 ms, 4 chunks | 1,240 ms, 1 chunk |

- A pass costs about the same whatever the text's length, because the decoder renders the whole window.
- Both FluidInference graphs run as fast or faster with `.cpuOnly` (15 s: 1.09-1.13 s; 5 s: about 0.5 s). With `.cpuAndNeuralEngine` they evidently do not run on the Neural Engine.
- With `offlineMode: true`, either bundle loads in about 0.6 s. A voice file with rows takes about 70 ms to parse.
- With `offlineMode: false`, the shared downloader's up-to-date check against the FluidInference repository takes 7.5-8.4 s even when every file is cached, against about 0.4 s for aufklarer. That cost is in `HuggingFaceDownloader`, not in Kokoro.
- Compiling these graphs for the GPU (the default `.all`) aborts inside MPSGraph on macOS 27; use `.cpuAndNeuralEngine`.

## 2. The warble

### Listening set

The same two texts were rendered with `af_heart` at speed 1 in each configuration. The short text was "Sure, I can do that."; the long one was a 25-word sentence.

| | Configuration |
|---|---|
| A | `kokoro_5s`, as shipped |
| B | `kokoro_21_15s`: random phases, BOS 1 / EOS 2, length-matched voice row |
| C | B with `random_phases` all zero |
| D | B on `.cpuOnly` |
| E | C on `.cpuOnly` |
| F | B with id 0 at both ends instead of 1 / 2 |
| G | B with the averaged `embedding` instead of the row |
| H | `kokoro_21_5s` with B's settings, plus zero phases and `.cpuOnly` variants |

In listening, A was clearly the best. Every FluidAudio configuration sounded warbly or odd, though their long sentences carried better intonation because they render in one pass.

### Pitch stability

The set is rendered by `E2EKokoroListeningSetTests` (`Tests/KokoroTTSTests/KokoroListeningSetE2ETests.swift`), with `KOKORO_LISTENING_DIR` set to the output folder. The pitch tracker was autocorrelation-based: 40 ms frames with a 10 ms hop, 70-400 Hz, and parabolic peak refinement. Jitter is the mean frame-to-frame F0 change in cents. Wobble is the RMS deviation from a 50 ms moving average. Each value is the mean of 3 renders.

| | Short: jitter / wobble / voiced frames | Long: jitter / wobble / voiced frames |
|---|---|---|
| A | 18.0 / 12.1 / 64 | 26.7 / 14.9 / 415 |
| B | 33.1 / 15.2 / 43 | 24.0 / 17.1 / 349 |
| C | 33.9 / 16.1 / 42 | 23.9 / 18.0 / 359 |
| D | 33.3 / 14.9 / 45 | 24.6 / 18.2 / 348 |
| F | 28.0 / 17.6 / 53 | 23.3 / 15.9 / 361 |
| G | 26.5 / 13.9 / 44 | 23.7 / 17.3 / 359 |
| H | 37.8 / 19.8 / 47 | 25.7 / 17.8 / 332 |

Faster speech raises per-frame jitter on its own, so these numbers support the listening result but do not settle it.

### What the graphs show

Both graphs' MIL programs were inspected (`model.mil` inside each `.mlmodelc`):

- **LSTMs.** `kokoro_5s` runs each of the five token-level bidirectional LSTMs as a forward LSTM plus a reverse LSTM over the reversed *valid* length. It also unrolls the frame-level F0/N "shared" LSTM over its 200 frames, with the backward half reversed by valid length (a `gather` with valid-length indices, then a mask multiply). `kokoro_21_15s` runs all six bidirectional LSTMs over the full padded length: 249 token slots, and 621 frames for the shared LSTM. For a short reply, about 90% of what the backward direction reads is padding.
- **Phase.** `kokoro_5s` sums the sine source's phase in 1000-sample segments (119 `cumsum` ops in all), carries each segment's end forward, and wraps with `phase - floor(phase)` before `sin`. `kokoro_21_15s` uses a single fp32 `cumsum` over 372,600 samples, then `floor`.

### Conclusion

The likely cause is the unpacked LSTMs:

1. Zero phases (C, E) and CPU-only execution (D) change nothing measurable. FluidAudio's phase sum is already fp32, and the graph does not run on the Neural Engine anyway, so fp16 phase drift is not the cause.
2. speech-core notes that Kokoro graphs exported before the packed-LSTM fix "spoke ~1.5x faster" (soniqo/speech-core#160). FluidAudio speaks 1.49x faster than A on the short text (64 vs 43 voiced frames) and 1.19x faster on the long one, which has less padding. The F0/N predictor reads the same polluted context, which fits an unsteady pitch that is worst on short replies.
3. Pad id 0 (F) and the averaged voice (G) lower short-text jitter only partly (33 to 28 and 26.5) and do not reach A.

No runtime option fixes this; it needs a re-export with packed or masked LSTMs. soniqo's re-export notes (speech-swift PR #518) give predicted durations in 25 ms frames for upstream / the earlier export / the packed re-export: "Done." 45 / 34 / 45, "Hello world." 64 / 59 / 64, a 62-token sentence 160 / 147 / 160.

Two other differences surfaced but are not the cause:

- speech-swift pads with BOS = 1 (`;`) and EOS = 2 (`:`), while upstream Kokoro and FluidAudio use 0.
- speech-core uses style row 128 for every voice (soniqo/speech-core#161).

## 3. Where the exporters are

### soniqo's exporter: private

- speech-swift commit `6d2a1b280` (2026-10-24, "Describe exports without naming internal repositories") removed references to `speech-models/models/<model>/export/convert.py` and `/Users/ivan/repos/speech-models/...`.
- speech-core PR #110 lists "speech-models export tests: 9 passed" in its test plan. PR #162 validates against a "PyTorch packed-sequence reference".
- `soniqo/speech-models` returns 404 to an authenticated API call and is not among soniqo's public repositories. The Wayback Machine has no captures of it.

### What the public history holds

- **speech-swift `scripts/convert_kokoro_coreml.py`.** Added `1cf36f4` (2026-03-09), updated `d3f159c` (2026-03-31), deleted `a2ccbf7` (2026-04-02). The last version is the 3-stage exporter: duration, prosody, and decoder buckets of 5, 10 and 15 s, with unpacked LSTMs and constants rather than flags. The end-to-end `kokoro_5s` reached Hugging Face on 2026-04-03 (`d21c9228`, "segmented cumsum"), after the script was removed, so its exporter was never committed here.
- **speech-swift PR #128.** Closed unmerged on 2026-03-16; its head commit `3af7165eb93b401b142c4a56f8655e8ac2b78049` is still on GitHub. It rewrites the script as an end-to-end exporter with FluidAudio-style buckets (`kokoro_21_5s`/`10s`/`15s`, `kokoro_24_10s`/`15s`). It uses unpacked LSTMs, bakes speed in as a constant, ignores `random_phases`, and uses a single phase `cumsum`.
- **Everything else in speech-swift.** All 46 branches, 29 tags and 436 pull requests were searched. No other Kokoro exporter turned up. PRs #144, #173 and #518 carry none.
- **speech-core.** No Kokoro exporter on `main` or in the history of `scripts/`. PRs #160-#163 pin and document the packed graphs only.
- **Hugging Face.** The full commit histories of the four `aufklarer/Kokoro-82M-*` CoreML repositories and of `soniqo/Kokoro-82M-ONNX` (branches `main`, `packed-lstm`, `fp32`) and `soniqo/Kokoro-82M-LiteRT` hold only model files and `README.md`. There are no discussions. The ONNX README says the graphs use "the E2E pipeline with FluidInference-aligned SineGen (segmented cumsum + phase wrapping)".
- **Wayback Machine.** Two snapshots of speech-swift's `scripts/` (2026-03-21 and 2026-04-12) show nothing beyond the above.

### Public alternatives

**elizaOS/eliza, `packages/training/scripts/kokoro/coreml/export_e2e_coreml.py`** (MIT; folder state at `b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3`, the exporter last changed in `a4d1faac1a`):

- Exports one fused graph with the same interface as `kokoro_5s`:
  - Inputs: `input_ids`, `attention_mask`, `ref_s`, `random_phases`, `speed`.
  - Outputs: `audio`, `audio_length_samples` (int32), `pred_dur`.
- `MaskedBidirectionalLSTM` replaces all six bidirectional LSTMs, including the frame-level shared LSTM. The recurrence holds its state on padded steps, which matches packing for right-padded input.
- AdaIN statistics are taken over valid frames only. `random_phases` is injected, with the fundamental's phase fixed at 0.
- Flags: `--stage torch|full`, `--seconds`, `--frames`, `--tokens`, `--precision fp16|fp32` (default fp32), `--out`. The torch stage checks parity against upstream `forward_with_tokens`; the full stage also checks CoreML against torch.
- The phase is upstream's: a frame-rate `cumsum` and a linear resample, with no wrap. That is fine at fp32; at fp16 over a long window the phase loses precision.
- `custom_stft.coreml.patch` in the same folder replaces a boolean scatter in `CustomSTFT` with a float blend. `validate_e2e_coreml.py` checks a real sentence against torch.
- It depends on `mattmireles/kokoro-coreml`, which supplies `kokoro/_export_utils.py` (`load_kokoro_for_export`) and its patched `kokoro` package.

**mattmireles/kokoro-coreml** (Apache-2.0; `main` at `b23caa862f0585c2c65187d0fc97e0f8311041dc`, with bundles on Hugging Face as `mattmireles/kokoro-coreml`):

- A Swift package (product `KokoroTTS`) plus published bundles with 3, 7, 10, 15 and 30 s buckets and masked padding.
- A multi-stage pipeline split across the Neural Engine, GPU and CPU rather than one graph, measured on iPhone 12 Pro and iPhone 15 Pro Max.
- Its exporters pin Python 3.12, torch 2.5.0, coremltools 8.3.0, numpy 1.26.4 and transformers 4.44.2 (`requirements-export.txt`).

Other paths:

- Ask soniqo for the Kokoro exporter in `speech-models`. That is the shortest route to a graph identical to `kokoro_5s` at 10 s and 15 s.
- Port soniqo's two graph changes, the valid-length reversed LSTMs and the segmented phase sum, into the eliza exporter. The shipped `kokoro_5s` MIL is a complete reference for both.

## 4. First 5 s export with the eliza exporter (2026-11-14)

Steps 1-5 of [kokoro-e2e-5s-export-ab.md](../plans/kokoro-e2e-5s-export-ab.md) were run. The setup: Python 3.12, torch 2.5.0, coremltools 8.3.0, eliza `b2e30700`, mattmireles `2a75c780`, and hexgrad `kokoro-v1_0.pth`.

### Fixes the exporter needed

Three changes, all in the local copy of `export_e2e_coreml.py`:

1. **The sine source converted to all zeros.** coremltools 8.3 converts `(f0 / sr) % 1` wrongly. A stage-by-stage comparison showed CoreML's `rad_values` with a mean of 1.6e-6 against torch's 0.748, and everything after it zero. With the sine source dead, the audio's spectral correlation with torch was 0.658. `x - torch.floor(x)` gives the same values, Python modulo semantics included, and converts correctly.
2. **The masked AdaIN's frame mask came out one sample short.** It is stretched with `F.interpolate(mode="nearest")`, which converts to an integer-scale upsample: 24,000 samples instead of 24,001, and the conversion fails. A constant-index `index_select` reproduces PyTorch's nearest indices exactly; this was checked for 200, 400 and 600-frame masks at every decoder length.
3. **Loading for the GPU aborts.** `ct.convert(compute_units=ALL)` loads the converted model for the GPU, which aborts inside MPSGraph on macOS 27 before the package is saved. `CPU_AND_NE` avoids it; this affects only loading in Python.

### Parity

- **Torch, masked against upstream** (24 random tokens): rounded durations identical (86 frames), STFT-magnitude correlation 0.979.
- **CoreML fp32 against torch:** waveform correlation 0.998, MAE 0.00135. Durations match. The results are identical on `CPU_ONLY` and `CPU_AND_NE`.

### A/B against `kokoro_5s`

Both graphs ran through `KokoroTTSModel` with `af_heart` at speed 1, using the same chunking (56/64) and the same voice file. Each value is length / jitter / wobble, jitter and wobble in cents, averaged over 3 renders:

| Text | `kokoro_5s` | eliza 5 s fp32 |
|---|---|---|
| "Sure, I can do that." | 1.93 s / 18.4 / 13.7 | 1.93 s / 21.3 / 15.6 |
| 25-word sentence (4 chunks) | 8.12 s / 27.2 / 15.0 | 8.12 s / 27.2 / 16.8 |
| "Yes." | 0.85 s / 33.6 / 13.1 | 0.85 s / 28.6 / 11.2 |
| "Hello there, how are you today?" | 2.52 s / 27.2 / 11.7 | 2.52 s / 29.1 / 12.4 |
| Question (2 chunks) | 4.38 s / 22.6 / 11.1 | 4.38 s / 22.3 / 10.8 |
| Numbers and a time (3 chunks) | 7.08 s / 24.9 / 12.6 | 7.09 s / 24.7 / 12.0 |
| Commas (2 chunks) | 5.86 s / 28.9 / 17.3 | 5.83 s / 29.7 / 17.4 |
| Three sentences (5 chunks) | 11.34 s / 23.5 / 13.0 | 11.35 s / 23.9 / 12.5 |

- **Durations** match `kokoro_5s` within one 25 ms frame per chunk, and so does the pitch behaviour. This is unlike FluidAudio's graphs, whose short-text jitter was 33 against 18.
- **BOS / EOS = 0 instead of 1 / 2** lengthens speech by 5-40% on both graphs (for example "Yes." 0.85 to 1.20 s), and the two graphs again match each other.
- **Speed:** about 385 ms per 5 s pass with `.cpuAndNeuralEngine`, the same with `.cpuOnly`, against 375 ms for `kokoro_5s`. The fp32 graph runs on the CPU.

### Still open

- Blind listening.
- An fp16 build. The unwrapped phase reaches about 220,000 rad, because each unvoiced sample with slightly negative F0 adds about one cycle. That needs soniqo's per-sample wrap, or the phase path kept in fp32.
- An ASR round trip.
