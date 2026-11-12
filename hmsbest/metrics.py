"""Competition metric: mean KL divergence KL(target || prediction), rows renormalised."""
import numpy as np


def kl_divergence(target, pred, eps: float = 1e-15) -> float:
    t = np.asarray(target, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    t = t / t.sum(axis=1, keepdims=True)
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(t > 0, t * np.log(t / p), 0.0)
    return float(z.sum(axis=1).mean())
