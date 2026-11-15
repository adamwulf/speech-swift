---
license: apache-2.0
language:
- en
- zh
- es
- fr
- hi
- it
- ja
- pt
tags:
- text-to-speech
- tts
- coreml
- kokoro
base_model: hexgrad/Kokoro-82M
library_name: coreml
pipeline_tag: text-to-speech
---

# Kokoro-82M CoreML (masked LSTMs)

End-to-end CoreML exports of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) whose LSTMs ignore the input padding. Each model runs the whole pipeline (BERT, duration prediction, alignment, prosody, decoder) in one CoreML call. There are four, which differ in window length and precision:

| Model | Window | Precision | Size | One pass on the CPU |
|---|---|---|---|---|
| `kokoro_masked_5s.mlmodelc` | 5 s, 128 phoneme tokens | fp32 | 327 MB | ~270 ms |
| `kokoro_masked_5s_fp16.mlmodelc` | 5 s, 128 phoneme tokens | fp16 decoder | 233 MB | ~200 ms |
| `kokoro_masked_10s_fp16.mlmodelc` | 10 s, 168 phoneme tokens | fp16 decoder | 234 MB | ~0.4 s |
| `kokoro_masked_15s_fp16.mlmodelc` | 15 s, 249 phoneme tokens | fp16 decoder | 236 MB | ~0.65 s |

All render 24 kHz audio and require iOS 18+ or macOS 15+. Times are from one Apple Silicon Mac. Each model has about 5,000 Core ML operations, and its first load with `.cpuOnly` (Core ML's one-time on-device build) takes about 0.5 s on that Mac. Files published before 2026-11-15 unrolled their LSTMs into 50,000-105,000 operations and took 10-54 s to load the first time on the Mac, and minutes on an iPhone. Download again if you have those cached.

## Why masked LSTMs

An end-to-end graph has fixed input shapes, so short text arrives padded. Kokoro's bidirectional LSTMs (the text encoder, the duration encoder and predictor, and the frame-level F0/N predictor) read that padding in their backward direction unless the export prevents it. When they do, predicted durations come out short and the pitch contour unsteady: speech that is rushed and "warbly", worst on short replies.

In these exports every bidirectional LSTM ignores the padding, which matches PyTorch's packed sequences for right-padded input. Each direction is one Core ML `lstm` operation: the forward one runs over the whole input, where the padding comes last, and the backward one runs over the valid tokens reversed in place. The decoder's AdaIN statistics are taken over valid frames only. Predicted durations match upstream Kokoro exactly.

## Which model

- **`kokoro_masked_5s_fp16`** for most use. It is the fastest, and in blind listening it was preferred to the best 5 s graph we had found in 6 of 8 pairs on the CPU. Longer text is split into 5 s chunks, and the first chunk's audio is ready after one ~200 ms pass.
- **`kokoro_masked_10s_fp16`** as a middle ground. It renders up to about 125 phoneme tokens (a typical long sentence) in one pass, at about half the 15 s model's cost per pass. In the blind long-sentence test it was preferred to the 5 s fp16 model in 5 of 6 pairs.
- **`kokoro_masked_15s_fp16`** for long sentences. It renders a sentence of up to about 194 phoneme tokens (about 10 s of speech) in one pass, with one intonation contour and no seams. It won every pair in a blind long-sentence test. A pass costs the full window whatever the text's length, and no audio is ready until the pass ends.
- **`kokoro_masked_5s`**, fp32, when fp16 is not wanted.

A runtime can load more than one fp16 model and pick by input length: 5 s for text that fits one 5 s chunk, 10 s or 15 s for longer text.

## Model I/O

| Name | Shape (5 s / 10 s / 15 s) | Type | |
|---|---|---|---|
| `input_ids` | [1, 128] / [1, 168] / [1, 249] | Int32 | Phoneme ids with a boundary token at each end (0 recommended, see "Voices and boundary tokens"), padded |
| `attention_mask` | [1, 128] / [1, 168] / [1, 249] | Int32 | 1 for real tokens, 0 for padding |
| `ref_s` | [1, 256] | Float32 | Voice style vector |
| `random_phases` | [1, 9] | Float32 | Initial phases of the harmonic source (the fundamental's is fixed at 0) |
| `speed` | [1] | Float32 | Speed multiplier, 1.0 = normal |
| `audio` | [1, 120000] / [1, 240000] / [1, 360000] | Float32 (fp32 model), Float16 (fp16 models) | 24 kHz waveform |
| `audio_length_samples` | [1] | Int32 | Valid samples in `audio` |
| `pred_dur` | [1, 128] / [1, 168] / [1, 249] | Int32 | Predicted frames per token (25 ms each) |

## Files

| File | Description |
|---|---|
| `kokoro_masked_5s.mlmodelc` | 5 s end-to-end model, fp32, pre-compiled |
| `kokoro_masked_5s_fp16.mlmodelc` | 5 s end-to-end model, fp16 decoder, pre-compiled |
| `kokoro_masked_10s_fp16.mlmodelc` | 10 s end-to-end model, fp16 decoder, pre-compiled |
| `kokoro_masked_15s_fp16.mlmodelc` | 15 s end-to-end model, fp16 decoder, pre-compiled |
| `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc` | Grapheme-to-phoneme model |
| `voices/*.json` | 54 voice presets in 10 languages, about 2.7 MB each: `embedding` (the voice's mean style vector) and the 510 per-length style rows under `"1"` … `"510"` |
| `vocab_index.json`, `g2p_vocab.json` | Phoneme and G2P vocabularies (hexgrad's token table) |
| `us_gold.json`, `us_silver.json` | English pronunciation dictionaries |
| `export/` | How the models were exported (see below) |

The vocabulary, dictionaries and G2P model are the same files as in [aufklarer/Kokoro-82M-CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML). The voice files are aufklarer's with the per-length rows added from [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)'s `voices/*.pt`. Their `embedding` is unchanged and is the mean of the rows.

## Voices and boundary tokens

Kokoro's style vector depends on the utterance's length: row `n` of a voice is the style for `n` phonemes, and upstream Kokoro uses `rows[phonemes - 1]`. The mean `embedding` is close to the long-utterance rows (cosine 0.999 at 120 phonemes), but far from the short ones (0.74 at 20, 0.49 at 5). Short replies rendered with the mean sound noticeably different.

Upstream Kokoro also puts token 0 at both ends of the phoneme ids. With 0, the start token gets a longer predicted lead-in (about 450 ms before "Yes." against 75-125 ms with ids 1/2), and a short reply's first sound comes out cleaner.

speech-swift does all three for these models: it picks the row by phoneme count, uses boundary token 0 (`boundaryTokenId`, set automatically for `kokoro_masked_*`), and cuts the utterance's lead-in to about 100 ms. Other runtimes should do the same.

In blind listening on eight short texts, our 5 s fp16 model with rows and boundary 0 was preferred to rows with ids 1/2 three times, the reverse once, with four ties. Against a split-pipeline Kokoro that had been preferred on five of eight short texts, rows alone brought it roughly level: three to two, with three ties.

## Usage

With [speech-swift](https://github.com/soniqo/speech-swift) (a version whose `KokoroTTSModel.fromPretrained` takes `modelFile`):

```swift
import KokoroTTS

let tts = try await KokoroTTSModel.fromPretrained(
    modelId: "milestonemade/Kokoro-82M-CoreML-Masked",
    modelFile: "kokoro_masked_5s_fp16",   // or "kokoro_masked_10s_fp16", "kokoro_masked_15s_fp16", "kokoro_masked_5s"
    computeUnits: .cpuOnly)
let audio = try tts.synthesize(text: "Hello world, this is a Kokoro test.", voice: "af_heart")
```

Pass `voiceFilter` (for example `["af_heart"]`) to download and parse only the voices you use: each voice file is about 2.7 MB and takes tens of milliseconds to parse.

Load the fp16 models with `.cpuOnly`. With `.cpuAndNeuralEngine`, the first load runs a Neural Engine compile of about 70 minutes, and each pass is then about 5 times slower than on the CPU: the graph keeps its text side and phase path in fp32, and the handoffs between CPU and Neural Engine cost more than the Neural Engine saves. The fp32 model runs entirely on the CPU under either setting.

## Quality

All blind tests used voice `af_heart` at speed 1, the same text chunking, voice file and post-processing on both sides, and shuffled, unlabeled pairs. They were run on the files published before 2026-11-15. The current files compute the same thing with native LSTM operations: on identical inputs they give the same durations, and their audio correlates with the earlier files' at 0.9987-0.9999.

Against `kokoro_5s` from aufklarer/Kokoro-82M-CoreML, the most natural-sounding end-to-end Kokoro CoreML graph we had found, on eight English texts from "Yes." to three sentences:

| Model | Preferred | Tied | `kokoro_5s` preferred |
|---|---|---|---|
| `kokoro_masked_5s` (fp32) | 9 of 16 pairs | 7 | 0 |
| `kokoro_masked_5s_fp16`, CPU | 6 of 8 | 1 | 1 |
| `kokoro_masked_5s_fp16`, Neural Engine | 6 of 8 | 2 | 0 |

On six long single sentences (8-11 s of speech each), rendered on the CPU:

| Pair | Result |
|---|---|
| `kokoro_masked_15s_fp16` vs `kokoro_masked_5s_fp16` (4-5 chunks per sentence) | 15 s preferred 6 of 6 |
| `kokoro_masked_15s_fp16` vs `kokoro_5s` | 15 s preferred 6 of 6 |
| `kokoro_masked_10s_fp16` (2-3 chunks per sentence) vs `kokoro_masked_5s_fp16` | 10 s preferred 5 of 6 |

The 10 s renders in that test came before the start-of-pass burst was cleared (see Limitations), and the listener noted faint static at their seams.

The listener's reasons were intonation and seams: chunked 5 s renders "sound stitched".

Against PyTorch, every model's predicted durations match exactly. The audio correlates with the masked PyTorch model at 0.998 for the fp32 waveform; the fp16 models reach a spectral correlation of 0.996 (5 s), 0.997 (10 s) and 0.9985 (15 s).

## Limitations

- **fp16 start-of-pass burst.** On the CPU, `kokoro_masked_5s_fp16` opens each pass with a faint burst of noise, about 12 ms at -39 to -50 dBFS, and `kokoro_masked_10s_fp16` with about 27 ms at -42 to -60 dBFS. It repeats at every chunk seam, where the 10 s burst is heard as static, and on a short reply it lands just before the speech onset. speech-swift clears it. Another runtime should do the same: the start token always lasts at least one 25 ms frame, so zero the first 25 ms of each pass when it is quieter than -30 dBFS RMS, and fade in over the next 5 ms. `kokoro_masked_15s_fp16` and the fp32 model start in silence. With the burst cleared, the 5 s fp16 lead-in sits at about -70 to -74 dBFS and the 10 s fp16 lead-in at about -99 dBFS, against -81 dBFS or less for fp32.
- **Fixed cost per pass.** A pass renders the whole window, so a one-word reply costs as much as a full window: about 200 ms for 5 s fp16, about 0.4 s for 10 s fp16, about 0.65 s for 15 s fp16.
- Speed, battery use and heat on iPhone have not been measured yet.

## How they were exported

- **Exporter:** [elizaOS/eliza](https://github.com/elizaOS/eliza) `packages/training/scripts/kokoro/coreml/export_e2e_coreml.py` (MIT) at commit `b2e30700d0b9a63e5e478505b6ddd28cd4fa8bc3`, with local changes.
- **Kokoro sources:** [mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml) (Apache-2.0) at commit `2a75c780664b2ea0430eef6829c04f4704555623`.
- **Weights:** hexgrad's `kokoro-v1_0.pth`.
- **Toolchain:** Python 3.12, torch 2.5.0, coremltools 8.3.0.
- **Commands**, each followed by `xcrun coremlcompiler compile`:
  - `kokoro_masked_5s`: `--stage full --seconds 5 --tokens 128 --precision fp32`
  - `kokoro_masked_5s_fp16`: `KOKORO_FP32_AXES=9,11`, `--stage full --seconds 5 --tokens 128 --precision fp16dec --skip-load`
  - `kokoro_masked_10s_fp16`: `KOKORO_FP32_AXES=9,11`, `--stage full --seconds 10 --tokens 168 --precision fp16dec --skip-load`
  - `kokoro_masked_15s_fp16`: `KOKORO_FP32_AXES=9,11`, `--stage full --seconds 15 --tokens 249 --precision fp16dec --skip-load`

The `export/` folder holds:

- `export_e2e_coreml.py`: the exporter as run, with eliza's MIT notice (`LICENSE-elizaOS-eliza`).
- `export_e2e_coreml.local.patch`: the same changes as a diff against eliza's file.
- `custom_stft.coreml.patch`: eliza's patch to mattmireles' `kokoro/custom_stft.py`.
- `parity_coreml.py`: compares a converted model with PyTorch on the CPU and on CPU + Neural Engine.
- `check_native_lstm.py`: checks the exporter's native LSTMs against PyTorch's packed sequences.
- `load_time.py`: counts a compiled model's operations and times its first and second load.

To re-export, set `KOKORO_COREML_REF` to the mattmireles checkout (with `checkpoints/config.json` and `checkpoints/kokoro-v1_0.pth`) and `KOKORO_COREML_HEXGRAD` to a hexgrad/Kokoro-82M download, apply the STFT patch, and run the exporter with the commands above.

The local changes to eliza's exporter (1-7 are needed with coremltools 8.3; 8 makes the models load quickly):

1. **The harmonic source.** `(f0 / sr) % 1` converts wrongly and zeroes the harmonic source; it is replaced by `x - floor(x)`, the same value.
2. **The AdaIN frame mask.** It is stretched with `F.interpolate(mode="nearest")`, which converts one sample short; it is replaced by a constant-index gather with PyTorch's exact nearest indices.
3. **Loading during conversion.** The model is loaded for CPU + Neural Engine instead of all units, since loading for the GPU aborts on macOS 27.
4. **`--precision fp16dec`.** Everything up to the decoder input (BERT, every masked LSTM, the duration head, the F0/N predictor, the text encoder) stays fp32. The decoder and vocoder run in fp16, except the axes in `KOKORO_FP32_AXES`: 9, the harmonic source's phase path, and 11, the vocoder's STFT bins.
5. **fp16-safe AdaIN statistics**, with identical math: means divided by the valid fraction rather than sums divided by a count, since a sum over 24,001 samples overflows fp16, and the variance taken on `(x - mean) / 16`.
6. **`--skip-load`** saves without loading, to avoid the long Neural Engine compile during conversion.
7. **`--precision fp16mix`**, an earlier attempt not used for these files.
8. **Native LSTMs.** eliza's exporter unrolls every bidirectional LSTM one time step at a time, which turned into 50,000-105,000 Core ML operations and a slow first load. Each direction is now one native LSTM: the forward one over the whole padded input, the backward one over each sequence's valid steps reversed in place with a gather and put back in order. The outputs match PyTorch's packed sequences within 2e-6.

## License

- Model weights: Apache-2.0, from hexgrad/Kokoro-82M.
- Voices, vocabulary, dictionaries and G2P: Apache-2.0.
- `export/export_e2e_coreml.py` and its patch: modified MIT-licensed code from elizaOS/eliza, Copyright (c) 2026 Shaw Walters and elizaOS Contributors (see `export/LICENSE-elizaOS-eliza`).
- `export/custom_stft.coreml.patch`: a patch to Apache-2.0 code from mattmireles/kokoro-coreml.
