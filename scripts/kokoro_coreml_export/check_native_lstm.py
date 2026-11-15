"""The exporter's MaskedBidirectionalLSTM (native LSTM ops) must equal PyTorch's
packed sequences, and eliza's step-by-step version it replaced, at every valid
step, and be zero at every padded step. Runs each of Kokoro's six bidirectional
LSTMs on random right-padded input over the lengths the graphs use.

`unrolled` is eliza's MaskedBidirectionalLSTM.forward from
export_e2e_coreml.py (MIT, Copyright (c) 2026 Shaw Walters and elizaOS
Contributors; see LICENSE-elizaOS-eliza), rewritten to read an nn.LSTM's
weights directly.

Usage: python check_native_lstm.py <export_e2e_coreml.py>
"""
import importlib.util
import sys

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("kokoro_e2e_export", sys.argv[1])
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def unrolled(lstm, x, attention_mask):
    """eliza's MaskedBidirectionalLSTM.forward: one cell per step, state held on
    padded steps."""
    def cell(x_t, h, c, sfx):
        g = (F.linear(x_t, getattr(lstm, "weight_ih_l0" + sfx), getattr(lstm, "bias_ih_l0" + sfx))
             + F.linear(h, getattr(lstm, "weight_hh_l0" + sfx), getattr(lstm, "bias_hh_l0" + sfx)))
        i, f, gg, o = g.chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        return torch.sigmoid(o) * torch.tanh(c_new), c_new

    b, steps, _ = x.shape
    m = attention_mask.to(dtype=x.dtype)
    out = {}
    for sfx, order in (("", range(steps)), ("_reverse", range(steps - 1, -1, -1))):
        h = x.new_zeros((b, lstm.hidden_size))
        c = x.new_zeros((b, lstm.hidden_size))
        seq = [None] * steps
        for t in order:
            a = m[:, t].unsqueeze(1)
            h_new, c_new = cell(x[:, t, :], h, c, sfx)
            h = h_new * a + h * (1 - a)
            c = c_new * a + c * (1 - a)
            seq[t] = h * a
        out[sfx] = torch.stack(seq, 1)
    return torch.cat([out[""], out["_reverse"]], dim=2)


def packed(lstm, x, length):
    """PyTorch's packed-sequence result, zero-padded back to the full length."""
    p = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor([length]), batch_first=True)
    y, _ = lstm(p)
    y, _ = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
    return y


k = E.load_model()
lstms = {
    "text_encoder.lstm": k.text_encoder.lstm,
    **{f"predictor.text_encoder.lstms.{i}": b for i, b in enumerate(k.predictor.text_encoder.lstms)
       if isinstance(b, torch.nn.LSTM)},
    "predictor.lstm": k.predictor.lstm,
    "predictor.shared": k.predictor.shared,
}
assert len(lstms) == 6, list(lstms)

g = torch.Generator().manual_seed(0)
worst = {"packed": 0.0, "unrolled": 0.0, "pad": 0.0}
cases = 0
with torch.no_grad():
    for name, lstm in lstms.items():
        native = E.MaskedBidirectionalLSTM(lstm).eval()
        # token windows 128 / 168 / 249; frame windows 200 / 400 / 600
        for steps in (128, 168, 200, 249, 400, 600):
            for length in sorted({1, 2, 24, steps // 2, steps - 1, steps,
                                  int(torch.randint(3, steps, (1,), generator=g))}):
                x = torch.randn(1, steps, lstm.input_size, generator=g)
                mask = torch.zeros(1, steps, dtype=torch.int32)
                mask[0, :length] = 1
                got = native(x, mask)
                worst["packed"] = max(worst["packed"], float((got - packed(lstm, x, length)).abs().max()))
                worst["unrolled"] = max(worst["unrolled"], float((got - unrolled(lstm, x, mask)).abs().max()))
                worst["pad"] = max(worst["pad"], float(got[:, length:].abs().max()) if length < steps else 0.0)
                cases += 1
        print(f"{name}: input {lstm.input_size}, hidden {lstm.hidden_size}, ok so far: {worst}")

print(f"cases={cases} max_abs_diff vs packed={worst['packed']:.2e} vs unrolled={worst['unrolled']:.2e} "
      f"padded max={worst['pad']:.2e}")
ok = worst["packed"] < 1e-5 and worst["unrolled"] < 1e-5 and worst["pad"] == 0.0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
