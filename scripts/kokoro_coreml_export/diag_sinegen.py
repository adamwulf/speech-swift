"""Drill into SineGen: compare CoreML vs torch for its internal tensors.

Usage: python diag_sinegen.py <export_e2e_coreml.py> [frames=200] [tokens=128]

Re-implements the exporter's injected-phase _f02sine with the same ops but
stashes rad_values, the downsampled cumulative phase and the upsampled
phase; also stashes SineGen's sine waves and voiced mask. The fused graph is
converted with those as outputs (fp32, CPU_ONLY) and each is compared with
torch.
"""
import importlib.util
import sys

import numpy as np
import torch
import torch.nn as nn
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


def f02sine_stash(self, f0_values):
    x = f0_values / self.sampling_rate
    rad_values = x - torch.floor(x)
    phase = getattr(self, "_inj_phase")
    mask = torch.ones(1, phase.shape[1], dtype=phase.dtype, device=phase.device)
    mask = torch.cat([torch.zeros(1, 1, dtype=phase.dtype), mask[:, 1:]], dim=1)
    rand_ini = phase * mask
    rad_values = torch.cat([rad_values[:, :1, :] + rand_ini.unsqueeze(1), rad_values[:, 1:, :]], dim=1)
    B, L, D = rad_values.shape
    down_len = max(1, int((L + self.upsample_scale - 1) // self.upsample_scale))
    rad_values_ds = E._linear_resample(rad_values.transpose(1, 2), down_len).transpose(1, 2)
    phase_c = torch.cumsum(rad_values_ds, dim=1) * 2 * torch.pi
    up_len = down_len * self.upsample_scale
    phase_up = E._linear_resample(phase_c.transpose(1, 2) * self.upsample_scale, up_len).transpose(1, 2)
    out = torch.sin(phase_up)
    self._dbg = {"f0_in": f0_values, "rad": rad_values, "rad_ds": rad_values_ds, "phase_c": phase_c,
                 "phase_up": phase_up, "sin": out}
    return out


orig_forward = E.SineGen.forward


def sinegen_forward_stash(self, f0):
    sine_waves, uv, noise = orig_forward(self, f0)
    self._dbg["sine_waves"] = sine_waves
    self._dbg["uv"] = uv
    return sine_waves, uv, noise


E.SineGen._f02sine = f02sine_stash
E.SineGen.forward = sinegen_forward_stash
KEYS = ["f0_in", "rad", "rad_ds", "phase_c", "phase_up", "sin", "sine_waves", "uv"]


class Diag(nn.Module):
    def __init__(self, e2e):
        super().__init__()
        self.e2e = e2e

    def forward(self, ids, mask, ref_s, phases, speed):
        audio, alen, pdur = self.e2e(ids, mask, ref_s, phases, speed)
        d = self.e2e.lsin._dbg
        return (audio, alen) + tuple(d[k] for k in KEYS)


m = E.load_model()
e2e = E.KokoroE2E(m, frames).eval()
for mod in e2e.modules():
    mod.eval()
diag = Diag(e2e).eval()
ref_vec = E.rep_voice_embedding(E.HEX / "voices/af_heart.pt")
ids, mask, ref_s, phases, speed = E.make_inputs(tokens, 24, ref_vec, seed=3)
with torch.no_grad():
    outs_t = diag(ids, mask, ref_s, phases, speed)
    traced = torch.jit.trace(diag, (ids, mask, ref_s, phases, speed), strict=False, check_trace=False)
names = ["audio", "alen"] + KEYS
for n, t in zip(names, outs_t):
    print(f"[torch] {n}: {tuple(t.shape)}")

ml = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
        ct.TensorType(name="random_phases", shape=(1, 9), dtype=np.float32),
        ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
    ],
    outputs=[ct.TensorType(name=n) for n in names],
    convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
)
pred = ml.predict({"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "ref_s": ref_s.numpy(),
                   "random_phases": phases.numpy(), "speed": speed.numpy()})
for n, t in zip(names, outs_t):
    if n in ("alen",):
        continue
    c = np.asarray(pred[n]).astype(np.float64)
    tt = t.numpy().astype(np.float64)
    if c.shape != tt.shape:
        print(f"[DIAG] {n}: SHAPE coreml={c.shape} torch={tt.shape}")
        k = min(c.size, tt.size)
        c, tt = c.reshape(-1)[:k], tt.reshape(-1)[:k]
    cf, tf = c.reshape(-1), tt.reshape(-1)
    var_c, var_t = float(np.var(cf)), float(np.var(tf))
    corr = float(np.corrcoef(cf, tf)[0, 1]) if var_c > 0 and var_t > 0 else float("nan")
    print(f"[DIAG] {n}: corr={corr:.5f} max_abs_diff={np.max(np.abs(cf - tf)):.5g} "
          f"mean_c={cf.mean():.5g} mean_t={tf.mean():.5g} std_c={np.sqrt(var_c):.5g} std_t={np.sqrt(var_t):.5g}")
sys.stdout.flush()
