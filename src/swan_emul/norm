import json, numpy as np

def load_norm_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(x, lo, hi):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype="float32")

def denorm(x, lo, hi):
    return x * (hi - lo) + lo

def depth_grad_mag(depth2d):
    a = depth2d.astype("float32")
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    v = g[np.isfinite(g)]
    if v.size == 0: return np.zeros_like(a, "float32")
    p1, p99 = np.percentile(v, [1, 99])
    g = (g - p1) / max(p99 - p1, 1e-6)
    return np.clip(g, 0, 1).astype("float32")
