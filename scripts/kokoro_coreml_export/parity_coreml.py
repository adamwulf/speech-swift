"""CoreML vs torch parity for an exported Kokoro E2E package, per compute unit.

Usage: python parity_coreml.py <export_e2e_coreml.py> <model.mlpackage|.mlmodelc> [tokens=128] [frames=200]

Feeds the exporter's own deterministic test inputs (24 tokens, seed 3,
af_heart averaged over the first 64 rows) to the torch fused model and to
the CoreML model on CPU_ONLY and CPU_AND_NE, and prints waveform and
phase-insensitive spectral parity for each, plus predicted durations.
"""
import importlib.util
import sys
import time

import numpy as np
import torch
import coremltools as ct

exporter_path, model_path = sys.argv[1], sys.argv[2]
tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128
frames = int(sys.argv[4]) if len(sys.argv) > 4 else 200

spec = importlib.util.spec_from_file_location("kokoro_e2e_export", exporter_path)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)  # applies the SineGen / AdaIN patches

m = E.load_model()
e2e = E.KokoroE2E(m, frames).eval()
for mod in e2e.modules():
    mod.eval()
ref_vec = E.rep_voice_embedding(E.HEX / "voices/af_heart.pt")
ids, mask, ref_s, phases, speed = E.make_inputs(tokens, 24, ref_vec, seed=3)
with torch.no_grad():
    a_t, alen_t, pdur_t = e2e(ids, mask, ref_s, phases, speed)
valid = int(alen_t.item())
a_t = a_t[0, :valid].numpy().astype(np.float64)
print(f"torch: valid={valid} pred_sum={int(pdur_t.sum())} rms={np.sqrt(np.mean(a_t ** 2)):.4f}")

inputs = {"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "ref_s": ref_s.numpy(),
          "random_phases": phases.numpy(), "speed": speed.numpy()}
for name, units in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
    if model_path.endswith(".mlmodelc"):
        model = ct.models.CompiledMLModel(model_path, compute_units=units)
    else:
        model = ct.models.MLModel(model_path, compute_units=units)
    model.predict(inputs)  # warm-up
    t0 = time.time()
    out = model.predict(inputs)
    ms = (time.time() - t0) * 1000
    alen = int(np.asarray(out["audio_length_samples"]).reshape(-1)[0])
    a_c = np.asarray(out["audio"]).reshape(-1)[:alen].astype(np.float64)
    n = min(len(a_c), len(a_t))
    corr = float(np.corrcoef(a_c[:n], a_t[:n])[0, 1])
    mae = float(np.mean(np.abs(a_c[:n] - a_t[:n])))
    sc, ld = E.spectral_parity(a_c[:n], a_t[:n])
    pdur_c = np.asarray(out["pred_dur"]).reshape(-1)
    same_dur = bool(np.array_equal(pdur_c[:24].astype(np.int64), pdur_t[0, :24].numpy().astype(np.int64)))
    peak = float(np.max(np.abs(a_c[:n])))
    print(f"{name}: {ms:.0f} ms alen={alen} durations_match={same_dur} wav_corr={corr:.4f} MAE={mae:.5f} "
          f"stft_mag_corr={sc:.5f} logmel_L1={ld:.5f} peak={peak:.3f} nan={bool(np.isnan(a_c).any())}")
