"""Find the first decoder stage that goes NaN / diverges under a precision.

Usage: python diag_decoder.py <export_e2e_coreml.py> <precision> [frames=200] [tokens=128]

Hooks the decoder's and the generator's child modules (and the generator's
ups / resblocks / noise_convs / noise_res lists element by element), converts
the fused graph with those outputs added (CPU_ONLY), and prints for each
stage, in execution order: NaN / inf, max |x| in torch and CoreML, and the
CoreML-vs-torch correlation. Precision-selector env vars of the exporter
(KOKORO_FP32_AXES, KOKORO_DEC_FP32_OPTYPES) apply.
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

exporter_path, precision = sys.argv[1], sys.argv[2]
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 200
tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 128

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

captured = {}
order = []


def hook(name):
    def fn(module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor):
            captured[name] = out
            if name not in order:
                order.append(name)
    return fn


dec = e2e.k.decoder
for name, child in dec.named_children():
    if name != "generator":
        child.register_forward_hook(hook(f"decoder.{name}"))
    if isinstance(child, nn.ModuleList):
        for i, sub in enumerate(child):
            sub.register_forward_hook(hook(f"decoder.{name}[{i}]"))
gen = dec.generator
for name, child in gen.named_children():
    if isinstance(child, nn.ModuleList):
        for i, sub in enumerate(child):
            sub.register_forward_hook(hook(f"generator.{name}[{i}]"))
    else:
        child.register_forward_hook(hook(f"generator.{name}"))
gen.register_forward_hook(hook("generator(out)"))

ref_vec = E.rep_voice_embedding(E.HEX / "voices/af_heart.pt")
ids, mask, ref_s, phases, speed = E.make_inputs(tokens, 24, ref_vec, seed=3)
with torch.no_grad():
    e2e(ids, mask, ref_s, phases, speed)
names = list(order)
print("[stages]", len(names), names)


class Diag(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, ids, mask, ref_s, phases, speed):
        captured.clear()
        audio, alen, pdur = self.inner(ids, mask, ref_s, phases, speed)
        return (audio,) + tuple(captured[n] for n in names)


diag = Diag(e2e).eval()
with torch.no_grad():
    outs_t = diag(ids, mask, ref_s, phases, speed)
    traced = torch.jit.trace(diag, (ids, mask, ref_s, phases, speed), strict=False, check_trace=False)

E.DEC_STATE["frames"] = frames
E.DEC_STATE["in_decoder"] = False
out_names = ["audio"] + [f"s{i}" for i in range(len(names))]
ml = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, tokens), dtype=np.int32),
        ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
        ct.TensorType(name="random_phases", shape=(1, 9), dtype=np.float32),
        ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
    ],
    outputs=[ct.TensorType(name=n) for n in out_names],
    convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
    compute_precision=E._compute_precision(ct, precision), compute_units=ct.ComputeUnit.CPU_ONLY,
)
pred = ml.predict({"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "ref_s": ref_s.numpy(),
                   "random_phases": phases.numpy(), "speed": speed.numpy()})
for label, n, t in zip(["audio"] + names, out_names, outs_t):
    c = np.asarray(pred[n]).astype(np.float64).reshape(-1)
    tt = t.detach().numpy().astype(np.float64).reshape(-1)
    k = min(c.size, tt.size)
    c, tt = c[:k], tt[:k]
    nan_c, inf_c = int(np.isnan(c).sum()), int(np.isinf(c).sum())
    finite = np.isfinite(c)
    corr = float(np.corrcoef(c[finite], tt[finite])[0, 1]) if finite.sum() > 1 and np.std(c[finite]) > 0 else float("nan")
    print(f"[DIAG] {label:28s} shape={tuple(t.shape)} nan={nan_c} inf={inf_c} "
          f"max_t={np.max(np.abs(tt)):.4g} max_c={np.max(np.abs(c[finite])) if finite.any() else float('nan'):.4g} corr={corr:.5f}")
sys.stdout.flush()
