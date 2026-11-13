"""Cross-fitted blend of several OOF files (same folds): non-negative weights summing to 1, fitted on the >=10-vote rows
of the other folds and applied to the held-out fold, followed by cross-fitted calibration. Honest ensemble estimate.
Usage: python -m hmsbest.blend a_oof.csv b_oof.csv ... [--out blend_oof.csv]"""
import argparse

import numpy as np
import pandas as pd

from .calibrate import crossfit
from .constants import CLASSES, TARGET_COLS
from .ensemble import blend_weights
from .metrics import kl_divergence

PC = [f"pred_{c}" for c in CLASSES]


def score(o, p):
    y, n = o[TARGET_COLS].to_numpy(), o.n_votes.to_numpy()
    return " / ".join(f"{kl_divergence(y[m], p[m]):.4f}" for m in (n >= 0, n >= 6, n >= 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oofs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dfs = [pd.read_csv(f).sort_values("eeg_id").reset_index(drop=True) for f in a.oofs]
    ids = set.intersection(*[set(d.eeg_id) for d in dfs])
    dfs = [d[d.eeg_id.isin(ids)].reset_index(drop=True) for d in dfs]
    o = dfs[0]
    assert all((d.fold.to_numpy() == o.fold.to_numpy()).all() for d in dfs), "OOF files use different folds"
    P = [d[PC].to_numpy() for d in dfs]
    y, n, fold = o[TARGET_COLS].to_numpy(), o.n_votes.to_numpy(), o.fold.to_numpy()
    print("KL all / >=6 / >=10")
    for f, p in zip(a.oofs, P):
        print(f"  {score(o, p)}  {f}")
    print(f"  {score(o, np.mean(P, 0))}  equal-weight mean")
    geo = np.exp(np.mean([np.log(np.clip(p, 1e-8, 1)) for p in P], 0)); geo /= geo.sum(1, keepdims=True)
    print(f"  {score(o, geo)}  geometric mean")
    out = np.zeros_like(P[0])
    for f in np.unique(fold):
        tr, te = (fold != f) & (n >= 10), fold == f
        w, _ = blend_weights([p[tr] for p in P], y[tr])
        out[te] = sum(wi * p[te] for wi, p in zip(w, P))
        print(f"  fold {f} weights {np.round(w, 2)}")
    print(f"  {score(o, out)}  cross-fitted weighted blend")
    b = o.copy(); b[PC] = out
    cal = crossfit(b, 10)
    print(f"  {score(o, cal)}  + cross-fitted calibration")
    hv = n >= 10
    print("  per fold (>=10):", " ".join(f"f{f}={kl_divergence(y[hv & (fold == f)], out[hv & (fold == f)]):.4f}" for f in np.unique(fold)))
    if a.out:
        b.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
