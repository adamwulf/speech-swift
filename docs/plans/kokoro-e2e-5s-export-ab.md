# Plan: export a 5 s Kokoro graph and A/B it against `kokoro_5s`

## Goal

Build a 5 s end-to-end Kokoro CoreML graph with the public elizaOS exporter, load it in `KokoroTTSModel`, and compare it blind and by measurement with the graph that sounds best today: `kokoro_5s` from `aufklarer/Kokoro-82M-CoreML`.

The 5 s window is deliberate. It matches `kokoro_5s` exactly (128 token slots, 120,000 samples), so the chunking, the trimming and the voice are identical on both sides, and only the graph differs. If the export matches `kokoro_5s`, the same exporter can then be run at 10 s and 15 s.

Background and the findings this builds on: [the 2026-11-13 research notes](../research/2026-11-13-kokoro-long-window-and-exporters.md).

## Success criteria

The exported graph, called X below, passes if:

1. **Torch parity.** The masked torch model predicts exactly the same rounded durations as upstream `forward_with_tokens` for the same token ids. Its audio matches upstream with an STFT-magnitude correlation of at least 0.99.
2. **CoreML parity.** The converted model matches the torch model with a waveform correlation of at least 0.99.
3. **Loads unchanged.** `KokoroTTSModel` loads X through `modelFile` and reports 128 slots, 120,000 output samples, a `speed` input, and a 56 / 64 chunk budget.
4. **Durations.** For every test text, X's predicted frames are within 1 frame per text of `kokoro_5s` (same BOS / EOS ids) and of upstream torch (BOS / EOS 0).
5. **Pitch.** Jitter and wobble, measured with the tracker from the research notes, are within about 10% of `kokoro_5s`.
6. **Intelligibility.** The Qwen3-ASR round trip gets the same word accuracy as `kokoro_5s`.
7. **Listening.** Blind, Adam rates X at least equal to `kokoro_5s` and hears no warble.

## Step 0: the listening harness

The harness from the warble investigation is committed. It has two parts:

- `E2EKokoroListeningSetTests` in `Tests/KokoroTTSTests/KokoroListeningSetE2ETests.swift`. It renders WAVs and prints render times and pitch stability for each configuration and text.
- Three internal switches used only by listening tests: `KokoroNetwork.experimentZeroPhases`, `KokoroTTSModel.experimentBoundaryId` (since made public as `boundaryTokenId`) and `KokoroTTSModel.experimentAveragedVoice`.

The test skips unless `KOKORO_LISTENING_DIR` is set. With `KOKORO_AB_BUNDLE_DIR` also set, it renders this plan's A/B set from that folder (Step 5) instead of the warble set. If the A/B shows that BOS / EOS = 0 matters, replace the switch with a real per-bundle option in a separate change.

## Step 1: environment (about 1 hour)

Work in a folder outside the repository, for example `~/kokoro-export/`, with `src/`, `weights/`, `out/` and `listen/`. Nothing from it is committed.

1. Create a Python 3.12 environment. This Mac's `python3` is 3.14 with no ML packages, and coremltools needs 3.11 or 3.12.
   ```bash
   uv venv --python 3.12 ~/kokoro-export/.venv
   source ~/kokoro-export/.venv/bin/activate
   uv pip install torch==2.5.0 torchaudio==2.5.0 coremltools==8.3.0 numpy==1.26.4 \
       transformers==4.44.2 huggingface_hub loguru "misaki[en]>=0.9.4" soundfile scipy
   python -c "import torch, coremltools as ct; print(torch.__version__, ct.__version__)"
   ```
   These are the versions pinned by `mattmireles/kokoro-coreml`'s `requirements-export.txt`.
2. Get mattmireles' Kokoro checkout (the exporter's `KOKORO_COREML_REF`) at `2a75c780664b2ea0430eef6829c04f4704555623` (2026-06-14), the version eliza's exporter was written against. Later commits make `AdaIN1d` take a mask argument (from `6b043523c8`, 2026-09-14), which eliza's AdaIN patch does not accept.
   ```bash
   gh api repos/mattmireles/kokoro-coreml/tarball/2a75c780664b2ea0430eef6829c04f4704555623 > kokoro-coreml-2a75c78.tar.gz
   mkdir -p ~/kokoro-export/src/kokoro-coreml-2a75c78
   tar -xzf kokoro-coreml-2a75c78.tar.gz -C ~/kokoro-export/src/kokoro-coreml-2a75c78 --strip-components 1
   ```
   At that commit, `checkpoints/config.json` and `checkpoints/kokoro-v1_0.pth` are symlinks into the author's home folder; delete them before step 5 copies the real files in.
3. Get the eliza exporter files at a fixed commit (folder `packages/training/scripts/kokoro/coreml/` at `b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3`):
   - `export_e2e_coreml.py`
   - `validate_e2e_coreml.py`
   - `diag_stages.py`
   - `custom_stft.coreml.patch`

   For example: `curl -L -o export_e2e_coreml.py https://raw.githubusercontent.com/elizaOS/eliza/b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3/packages/training/scripts/kokoro/coreml/export_e2e_coreml.py`.
4. Apply `custom_stft.coreml.patch` to the checkout's `kokoro/custom_stft.py` (it replaces the `correction_mask` boolean scatter with a float blend; its paths are written as `ref/kokoro/custom_stft.py`). Then make three changes to the local copy of `export_e2e_coreml.py`, needed with coremltools 8.3 on macOS 27 (see the research notes, section 4):
   - In `_f02sine_injected`, replace `(f0_values / self.sampling_rate) % 1` with `x - torch.floor(x)`. The `%` converts wrongly, and the sine source comes out all zero.
   - In `_adain_forward_masked`, replace `F.interpolate(fm, size=T, mode="nearest")` with a nearest resample by constant `index_select` indices. The interpolate converts to an integer-scale upsample, 24,000 samples long instead of 24,001.
   - Pass `compute_units=ct.ComputeUnit.CPU_AND_NE` to `ct.convert`. `ALL` makes coremltools load the model for the GPU, which aborts on macOS 27.
5. Download the upstream weights and place them where the exporter looks:
   ```bash
   huggingface-cli download hexgrad/Kokoro-82M --local-dir ~/kokoro-export/weights/kokoro-82m
   mkdir -p ~/kokoro-export/src/kokoro-coreml/checkpoints
   cp ~/kokoro-export/weights/kokoro-82m/{config.json,kokoro-v1_0.pth} ~/kokoro-export/src/kokoro-coreml/checkpoints/
   export KOKORO_COREML_REF=~/kokoro-export/src/kokoro-coreml
   export KOKORO_COREML_HEXGRAD=~/kokoro-export/weights/kokoro-82m
   mkdir -p /tmp/kokoro-coreml-work   # the script writes its parity WAVs here
   ```

Done when the exporter imports without errors and `load_model()` loads the weights.

## Step 2: torch parity (about 30 minutes)

```bash
python export_e2e_coreml.py --stage torch --seconds 5 --tokens 128
```

- Check the printed `pdur_e2e` against `pdur_ref`. They must be identical: this proves the masked LSTMs ignore the padding.
- Check `[spectral parity] stft_mag_corr` (at least 0.99) and `wav_corr`.
- Listen to `/tmp/kokoro-coreml-work/out_fused.wav` against `out_truth.wav`.

**Gate 1:** if the durations differ, stop. The masking does not work on this checkout, and the rest of the plan would measure the wrong thing.

## Step 3: convert and compile (1-2 hours)

1. Convert at fp32 and at fp16. Time each run and record its peak memory, for example with `/usr/bin/time -l`.
   ```bash
   python export_e2e_coreml.py --stage full --seconds 5 --tokens 128 --precision fp32 --out ~/kokoro-export/out/fp32
   python export_e2e_coreml.py --stage full --seconds 5 --tokens 128 --precision fp16 --out ~/kokoro-export/out/fp16
   ```
   - Each run prints `[parity coreml-vs-torch]`; record it.
   - Expect fp16 to struggle. The exporter keeps upstream's unwrapped phase, and at 5 s the 9th harmonic's phase approaches fp16's maximum of 65,504, where fp16 resolves only steps of 32. fp32 has no such problem.
2. Compile both packages and give each bundle a distinct name:
   ```bash
   xcrun coremlcompiler compile ~/kokoro-export/out/fp32/kokoro_5s.mlpackage ~/kokoro-export/out/fp32/
   mv ~/kokoro-export/out/fp32/kokoro_5s.mlmodelc ~/kokoro-export/out/kokoro_eliza_5s_fp32.mlmodelc
   # repeat for fp16 -> kokoro_eliza_5s_fp16.mlmodelc
   ```
3. Check each bundle's `metadata.json`:
   - `input_ids` [1, 128]
   - `speed` present
   - `audio` with 120,000 samples (`[1, 120000]` is fine)
   - `audio_length_samples` int32

## Step 4: load in speech-swift (about 30 minutes)

1. Build one local bundle folder, for example `~/kokoro-export/bundle/`, that serves both sides of the A/B:
   - Copy from the aufklarer cache, `~/Library/Caches/qwen3-speech/models/aufklarer/Kokoro-82M-CoreML/`:
     - `vocab_index.json`, `g2p_vocab.json`, `us_gold.json`, `us_silver.json`
     - `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`
     - `voices/`
     - `kokoro_5s.mlmodelc`
   - Add `kokoro_eliza_5s_fp32.mlmodelc` and `kokoro_eliza_5s_fp16.mlmodelc`.

   All configurations then share the same vocabulary, G2P and voice files (the aufklarer `af_heart` embedding), so only the graph differs.
2. Load each graph with `.cpuAndNeuralEngine`. Never use `.all` or `.cpuAndGPU` here: compiling these graphs for the GPU aborts on macOS 27.
   ```swift
   let x = try await KokoroTTSModel.fromPretrained(
       modelId: "local/kokoro-ab", modelFile: "kokoro_eliza_5s_fp32",
       cacheDir: bundleURL, offlineMode: true, computeUnits: .cpuAndNeuralEngine)
   ```
   With `offlineMode: true`, the loader only checks that the folder holds an `.mlmodelc`.
3. Check `network.inputSlots == 128`, `network.outputSamples == 120_000`, `network.acceptsSpeed`, and `chunkBudget == KokoroChunkBudget(budget: 56, hardCap: 64)`.
4. Smoke test: "Hello world." renders audible speech, and Qwen3-ASR transcribes it as "hello world".

## Step 5: A/B render set (1-2 hours)

Configurations, all with `af_heart` at speed 1, `.cpuAndNeuralEngine`, and the same chunking, trimming and clamp:

| Name | Graph | BOS / EOS |
|---|---|---|
| A | `kokoro_5s` (aufklarer) | 1 / 2, as speech-swift sends today |
| A0 | `kokoro_5s` | 0 / 0 (optional: does the id matter for soniqo's graph?) |
| X1 | `kokoro_eliza_5s_fp32` | 1 / 2 |
| X2 | `kokoro_eliza_5s_fp32` | 0 / 0, as upstream and eliza's harness use |
| X3 | `kokoro_eliza_5s_fp16` | 1 / 2 |
| X4 | `kokoro_eliza_5s_fp16` | 0 / 0 |

Also render X1 with `.cpuOnly`, to compare speed and to see where the graph runs.

Texts, 8 to 10 of them:

- Short replies: "Yes.", "Sure, I can do that.", "Hello there, how are you today?"
- A question.
- A sentence with numbers and a time.
- A sentence heavy with commas.
- The 25-word build-server sentence from the warble set, which splits into chunks.
- A three-sentence paragraph, to hear the seams.

Steps:

1. Render the set. `E2EKokoroListeningSetTests` already holds these configurations and texts; the boundary switch covers A0, X2 and X4.
   ```bash
   KOKORO_LISTENING_DIR=~/kokoro-export/listen KOKORO_AB_BUNDLE_DIR=~/kokoro-export/bundle \
       swift test --filter E2EKokoroListeningSetTests --disable-sandbox
   ```
   It writes `<config>_<text>.wav` and prints one `LISTEN` line per configuration and text: chunks, audio length, render times, jitter, wobble and voiced frames.
2. For blind listening, copy the WAVs to shuffled names (`01.wav`, `02.wav`, …) and keep the key in a separate `key.csv`. For each text, Adam compares A with each X, records a preference, and notes any warble.

## Step 6: measurements

For each configuration and text:

1. **Durations.** Sum `pred_dur`.
   - X1 and X3 are compared with A.
   - X2 and X4 are compared with upstream torch `forward_with_tokens` on the same ids, from the Step 1 environment.
   - Reference points from soniqo's packed re-export: "Done." 45 frames, "Hello world." 64 frames.
2. **Pitch.** Jitter and wobble from the `LISTEN` lines, averaged over 3 renders.
3. **Intelligibility.** The Qwen3-ASR round trip, as in `E2EKokoroTests`, scored as word accuracy.
4. **Level.** Peak absolute sample before the clamp (should be about 1.05 or below), and no NaN or infinite samples.
5. **Speed and placement.** Warm per-pass time with `.cpuAndNeuralEngine` and with `.cpuOnly`. Where the ops run, read from `MLComputePlan` (macOS 14.4+), or else inferred from the two timings.
6. **Seams** (optional): run `E2EKokoroSeamTests` against X through `modelFile`.

## Step 7: decide

**Gate 2:**

- **X matches A** (blind listening and Step 6): write the 10 s / 15 s plan. It uses the same exporter with `--seconds 10` / `--seconds 15` and `--tokens 249`, revisits the chunk budget and latency, and has to keep the unrolled frame LSTM convertible at 400 and 600 frames.
- **fp32 is good, fp16 warbles:** port soniqo's phase handling into `_f02sine_injected`, then repeat Steps 3-6 at fp16. That means summing the phase per sample in 1000-sample segments, carrying each segment's end forward, and wrapping with `x - floor(x)` before `sin`; the shipped `kokoro_5s` MIL shows the exact structure. This is needed only if running on the Neural Engine matters.
- **X warbles at fp32 too:** compare stage by stage against A with `diag_stages.py` (predicted F0, harmonic source). Suspect the phase path or the resampling first, and reconsider asking soniqo for their exporter.

Record the outcome in the research notes. Raw WAVs and per-run data stay in `~/kokoro-export/`; at most, one summary goes into `docs/benchmarks/`.

## Risks

- **Toolchain.** torch 2.5.0 and coremltools 8.3.0 on macOS 27 with Python 3.12 are untested here. Fall back to Python 3.11 if wheels or the converter misbehave.
- **Graph size.** The masked LSTMs unroll in the trace: 128 token steps × 2 directions × 5 LSTMs, plus 200 × 2 frame steps. That is about the size of `kokoro_5s` itself, so conversion should work. It may still take tens of minutes and several GB of memory.
- **fp16 phase precision**, explained in Step 3. The port in Gate 2 removes it.
- **Placement.** An fp32 graph runs on the CPU and GPU only. The FluidInference graphs ran on the CPU at about 1.1 s per 15 s pass, so this is acceptable for an A/B, but not necessarily for shipping.
- **Licenses.** eliza is MIT, mattmireles' code Apache-2.0, Kokoro's weights Apache-2.0. Local experiments are fine; keep attribution if a bundle is ever published.

## Out of scope

- Publishing a bundle to Hugging Face.
- A per-bundle BOS / EOS option in the runtime.
- The 10 s and 15 s exports.
- Any change to how the library chooses models.

Each gets its own plan after Gate 2.

## Estimate

About a day: Step 1, 1 hour; Step 2, 30 minutes; Step 3, 1-2 hours; Step 4, 30 minutes; Steps 5-6, 2-3 hours; Step 7, as the result dictates.
