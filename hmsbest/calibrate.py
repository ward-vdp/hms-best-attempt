"""Post-hoc calibration of OOF predictions, evaluated honestly by cross-fitting over folds.

p_cal = softmax(log(p) / T + b): one temperature T plus a per-class bias b (7 parameters), fitted on the other folds
and applied to the held-out fold. KL heavily punishes over/under-confident predictions, and the training targets
(heavily smoothed low-vote labels) differ from the >=10-vote evaluation targets, so the optimal sharpness differs too.
Usage: python -m hmsbest.calibrate oof.csv [--min_votes 10]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constants import CLASSES, TARGET_COLS
from .metrics import kl_divergence


def apply(p, prm):
    z = np.log(np.clip(p, 1e-8, 1)) / prm[0] + prm[1:]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit(p, y):
    f = lambda prm: kl_divergence(y, apply(p, np.concatenate([[abs(prm[0]) + 1e-3], prm[1:]])))
    r = minimize(f, np.r_[1.0, np.zeros(p.shape[1])], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
    return np.concatenate([[abs(r.x[0]) + 1e-3], r.x[1:]])


def crossfit(oof, min_votes=0):
    pc = [f"pred_{c}" for c in CLASSES]
    out = oof[pc].to_numpy().copy()
    for f in sorted(oof.fold.unique()):
        tr = (oof.fold != f) & (oof.n_votes >= min_votes)
        prm = fit(oof.loc[tr, pc].to_numpy(), oof.loc[tr, TARGET_COLS].to_numpy())
        te = (oof.fold == f).to_numpy()
        out[te] = apply(oof.loc[te, pc].to_numpy(), prm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--min_votes", type=int, default=10, help="fit calibration on windows with >= this many votes")
    a = ap.parse_args()
    o = pd.read_csv(a.oof)
    pc = [f"pred_{c}" for c in CLASSES]
    cal = crossfit(o, a.min_votes)
    for name, m in (("all", o.n_votes >= 0), (">=6", o.n_votes >= 6), (">=10", o.n_votes >= 10)):
        m = m.to_numpy()
        y = o.loc[m, TARGET_COLS].to_numpy()
        print(f"{name:5s} raw {kl_divergence(y, o.loc[m, pc].to_numpy()):.4f}  calibrated {kl_divergence(y, cal[m]):.4f}")
    print("params on all folds:", fit(o.loc[o.n_votes >= a.min_votes, pc].to_numpy(),
                                      o.loc[o.n_votes >= a.min_votes, TARGET_COLS].to_numpy()).round(3))


if __name__ == "__main__":
    main()
