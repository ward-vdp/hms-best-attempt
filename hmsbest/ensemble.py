"""Blend OOF predictions of several experiments: non-negative weights summing to 1 minimising KL."""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import CLASSES, TARGET_COLS
from .metrics import kl_divergence


def blend_weights(preds, y):
    k = len(preds)
    f = lambda w: kl_divergence(y, sum(wi * p for wi, p in zip(np.abs(w) / np.abs(w).sum(), preds)))
    r = minimize(f, np.ones(k) / k, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
    w = np.abs(r.x) / np.abs(r.x).sum()
    return w, r.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oofs", nargs="+", help="oof.csv paths")
    a = ap.parse_args()
    dfs = [pd.read_csv(p).sort_values("eeg_id").reset_index(drop=True) for p in a.oofs]
    ids = set.intersection(*[set(d.eeg_id) for d in dfs])
    dfs = [d[d.eeg_id.isin(ids)].reset_index(drop=True) for d in dfs]
    y = dfs[0][TARGET_COLS].to_numpy()
    preds = [d[[f"pred_{c}" for c in CLASSES]].to_numpy() for d in dfs]
    for p, d in zip(a.oofs, preds):
        print(f"{p}: KL {kl_divergence(y, d):.4f}")
    w, kl = blend_weights(preds, y)
    print("weights", dict(zip(a.oofs, w.round(3))), f"blend KL {kl:.4f}")


if __name__ == "__main__":
    main()
