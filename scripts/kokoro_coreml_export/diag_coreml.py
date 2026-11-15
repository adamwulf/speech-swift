"""Localize where the CoreML Kokoro graph diverges from torch.

Usage: python diag_coreml.py <export_e2e_coreml.py> [frames=200] [tokens=128]

Converts the fused graph with two extra outputs, the predicted F0 and the
harmonic source, then compares CoreML (CPU_ONLY, fp32) against torch on the
exporter's own test inputs:
  F0 diverges                 -> F0Ntrain (masked LSTM / AdaIN)
  F0 matches, har diverges    -> SineGen (phase / resample)
  both match, audio diverges  -> decoder / generator / iSTFT
Adapted from eliza's diag_stages.py (local paths, CPU_ONLY load).
"""
import importlib.util
import sys

import numpy as np
import torch
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY

exporter_path = sys.argv[1]
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

spec = importlib.util.spec_from_file_location("kokoro_e2e_export", exporter_path)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

if "new_ones" not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
    @register_torch_op
    def new_ones(context, node):
        inputs = _get_inputs(context, node)
        shape = inputs[1]
        if isinstance(shape, list):
            shape = mb.concat(values=shape, axis=0)
        shape = mb.cast(x=shape, dtype="int32")
        context.add(mb.fill(shape=shape, value=1.0, name=node.name))


@register_torch_op(torch_alias=["and"], override=True)
def bitwise_and(context, node):
    a, b = _get_inputs(context, node, expected=2)
    context.add(mb.logical_and(x=mb.cast(x=a, dtype="bool"), y=mb.cast(x=b, dtype="bool"), name=node.name))


m = E.load_model()
e2e = E.KokoroE2E(m, frames).eval()
for mod in e2e.modules():
    mod.eval()
e2e._diag = True
ref_vec = E.rep_voice_embedding(E.HEX / "voices/af_heart.pt")
ids, mask, ref_s, phases, speed = E.make_inputs(tokens, 24, ref_vec, seed=3)
with torch.no_grad():
    a_t, alen_t, pdur_t, f0_t, har_t = e2e(ids, mask, ref_s, phases, speed)
    traced = torch.jit.trace(e2e, (ids, mask, ref_s, phases, speed), strict=False, check_trace=False)
print(f"[torch] audio={tuple(a_t.shape)} F0={tuple(f0_t.shape)} har={tuple(har_t.shape)} valid={int(alen_t)}")

ml = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
        ct.TensorType(name="random_phases", shape=(1, 9), dtype=np.float32),
        ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
    ],
    outputs=[ct.TensorType(name=k) for k in ("audio", "audio_length_samples", "pred_dur", "F0_pred", "har_source")],
    convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
)
pred = ml.predict({"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "ref_s": ref_s.numpy(),
                   "random_phases": phases.numpy(), "speed": speed.numpy()})


def stats(name, c, t, valid=None):
    c = np.asarray(c).reshape(-1).astype(np.float64)
    t = np.asarray(t).reshape(-1).astype(np.float64)
    k = min(len(c), len(t)) if valid is None else valid
    c, t = c[:k], t[:k]
    corr = float(np.corrcoef(c, t)[0, 1]) if k > 1 else 0.0
    print(f"[DIAG] {name}: n={k} corr={corr:.5f} max_abs_diff={np.max(np.abs(c - t)):.5f} "
          f"rms_c={np.sqrt(np.mean(c ** 2)):.5f} rms_t={np.sqrt(np.mean(t ** 2)):.5f}")
    return corr


valid = int(alen_t)
f0_valid = 2 * int(pdur_t.sum())
stats("F0_pred (valid frames)", pred["F0_pred"], f0_t.numpy(), f0_valid)
har_c = np.asarray(pred["har_source"])
print(f"[DIAG] har_source shapes coreml={har_c.shape} torch={tuple(har_t.shape)}")
stats("har_source (valid samples)", har_c, har_t.numpy(), valid)
stats("audio (valid samples)", pred["audio"], a_t.numpy(), valid)
sc, ld = E.spectral_parity(np.asarray(pred["audio"]).reshape(-1)[:valid], a_t.reshape(-1)[:valid].numpy())
print(f"[DIAG] audio spectral: stft_mag_corr={sc:.5f} logmel_L1={ld:.5f}")
