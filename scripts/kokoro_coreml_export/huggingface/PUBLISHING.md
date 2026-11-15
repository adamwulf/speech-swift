# Publishing the masked Kokoro models to Hugging Face

`Kokoro-82M-CoreML-Masked/README.md` here is the canonical source of the model card at
[milestonemade/Kokoro-82M-CoreML-Masked](https://huggingface.co/milestonemade/Kokoro-82M-CoreML-Masked).
Edit it here, then upload it. Do not edit it on the Hub.

## Published revisions

| Hub commit | Change |
|---|---|
| `55224889022bfbf517bd8e9d86dc829575f0933b` | First upload: `kokoro_masked_5s`, `kokoro_masked_5s_fp16`, `kokoro_masked_15s_fp16`, shared files, `export/` |
| `4f34ae2a4f2a5985114eef59fcd07d7c71e02b11` | Added `kokoro_masked_10s_fp16`; model card updated to match `Kokoro-82M-CoreML-Masked/README.md` |
| `d8b3a084a4f3945a76f085605fd2e29a795181d7` | Voices gained the 510 per-length style rows (`make_voice_rows.py`); card documents rows, boundary token 0 and the lead-in cut |

## Repository layout

| Published path | Source |
|---|---|
| `README.md` | `Kokoro-82M-CoreML-Masked/README.md` in this folder |
| `kokoro_masked_5s.mlmodelc` | 5 s fp32 export: `--stage full --seconds 5 --tokens 128 --precision fp32`, then `xcrun coremlcompiler compile` |
| `kokoro_masked_5s_fp16.mlmodelc` | 5 s fp16dec export: `KOKORO_FP32_AXES=9,11`, `--seconds 5 --tokens 128 --precision fp16dec --skip-load` |
| `kokoro_masked_10s_fp16.mlmodelc` | 10 s fp16dec export: same, `--seconds 10 --tokens 168` |
| `kokoro_masked_15s_fp16.mlmodelc` | 15 s fp16dec export: same, `--seconds 15 --tokens 249` |
| `G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `vocab_index.json`, `g2p_vocab.json`, `us_gold.json`, `us_silver.json` | Unchanged copies from [aufklarer/Kokoro-82M-CoreML](https://huggingface.co/aufklarer/Kokoro-82M-CoreML) |
| `voices/*.json` | aufklarer's voice files with hexgrad/Kokoro-82M's 510 per-length rows added: `python make_voice_rows.py <hexgrad voices/> <aufklarer voices/> <out>` (fails unless every voice has a tensor whose row mean equals its `embedding`) |
| `export/export_e2e_coreml.py`, `export/export_e2e_coreml.local.patch`, `export/custom_stft.coreml.patch`, `export/parity_coreml.py`, `export/LICENSE-elizaOS-eliza` | Copies of the same files in `scripts/kokoro_coreml_export/` |

Each export must pass `parity_coreml.py` (durations match torch, no NaN) before it is compiled and staged. The export commands and their parity arguments are in `../README.md`.

## Staging and upload

1. Make an empty staging folder laid out as in the table above. Copy each compiled export in under its published name, add the aufklarer shared files and the `export/` copies, and copy in `Kokoro-82M-CoreML-Masked/README.md`.
2. Check the staged text files for local paths or personal details, and for hidden or cache files (`.DS_Store`, `__pycache__`).
3. Upload. This publishes the files; the repository is public.

   ```bash
   hf auth login
   hf upload milestonemade/Kokoro-82M-CoreML-Masked <staging folder> . --type model --commit-message "<what changed>"
   ```

   The upload only sends files whose content changed, and it can be re-run to resume. The first upload also needs `hf repos create milestonemade/Kokoro-82M-CoreML-Masked --type model --public`.
4. Verify every model by loading it from the Hub as the card shows: `KokoroTTSModel.fromPretrained(modelId: "milestonemade/Kokoro-82M-CoreML-Masked", modelFile: "<name>", computeUnits: .cpuOnly)` (`.cpuAndNeuralEngine` for the fp32 model). Then synthesize a sentence and check its slots, output window and speed input.
5. Add a row to "Published revisions" above, with the new Hub commit (`hf models info milestonemade/Kokoro-82M-CoreML-Masked --expand=sha`).

To change only the card, upload just that file: `hf upload milestonemade/Kokoro-82M-CoreML-Masked Kokoro-82M-CoreML-Masked/README.md README.md --type model`.
