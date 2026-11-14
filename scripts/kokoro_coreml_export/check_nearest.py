import numpy as np
import torch
import torch.nn.functional as F

# Constant-index nearest resample must equal F.interpolate(mode="nearest")
# for every length the masked AdaIN sees: the 200-frame mask stretched to
# each decoder/generator length (multiples of frames, frames*k + 1, ...).


def nearest_indices(in_len, out_len):
    scale = np.float32(in_len) / np.float32(out_len)
    idx = np.floor(np.arange(out_len, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(idx, in_len - 1)


bad = []
for in_len in (200, 400, 600):
    for out_len in sorted({in_len, in_len * 2, in_len * 2 + 1, in_len * 10, in_len * 60, in_len * 60 + 1,
                           in_len * 120, in_len * 120 + 1, in_len * 300, in_len * 600, in_len * 600 + 1,
                           *range(in_len, in_len * 4 + 3)}):
        x = torch.arange(in_len, dtype=torch.float32).view(1, 1, in_len)
        ref = F.interpolate(x, size=out_len, mode="nearest").view(-1).long().numpy()
        got = nearest_indices(in_len, out_len)
        if not np.array_equal(ref, got):
            bad.append((in_len, out_len, int((ref != got).sum())))
print("mismatches:", bad[:10], "total", len(bad))
