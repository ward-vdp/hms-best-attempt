"""Error analysis of an OOF file: where does the KL come from? Usage: python -m hmsbest.analyze oof.csv [more.csv ...]"""
import sys

import numpy as np
import pandas as pd

from .calibrate import apply, crossfit, fit
from .constants import CLASSES, TARGET_COLS
from .metrics import kl_divergence

PC = [f"pred_{c}" for c in CLASSES]


def kl_rows(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return np.where(y > 0, y * np.log(np.clip(y, eps, 1) / p), 0.0)          # (N, 6) per-class contributions


def report(o):
    y, p, n = o[TARGET_COLS].to_numpy(), o[PC].to_numpy(), o.n_votes.to_numpy()
    c = kl_rows(y, p)
    hv = n >= 10
    print(f"rows {len(o)}  KL all {c.sum(1).mean():.4f}  >=6 {c[n >= 6].sum(1).mean():.4f}  >=10 {c[hv].sum(1).mean():.4f}")
    if "fold" in o and o.fold.nunique() > 1:
        print("per fold (>=10):", " ".join(f"f{f}={c[(o.fold == f).to_numpy() & hv].sum(1).mean():.4f}" for f in sorted(o.fold.unique())))
    print("by votes:", " | ".join(f"{a}-{b}: n={m.sum()} KL={c[m].sum(1).mean():.3f}"
                                  for a, b in ((1, 2), (3, 5), (6, 9), (10, 14), (15, 99)) for m in [(n >= a) & (n <= b)] if m.any()))
    print("-- on >=10 votes --")
    print("class      share_of_KL  mean_y  mean_p   KL_when_dominant (n)")
    dom = y.argmax(1)
    for k, name in enumerate(CLASSES):
        m = hv & (dom == k)
        print(f"{name:9s}  {c[hv][:, k].mean() / c[hv].sum(1).mean():10.1%}  {y[hv][:, k].mean():.3f}  {p[hv][:, k].mean():.3f}"
              f"   {c[m].sum(1).mean() if m.any() else float('nan'):.3f} ({m.sum()})")
    ent = lambda q: -(np.where(q > 0, q * np.log(np.clip(q, 1e-12, 1)), 0)).sum(1)
    print(f"sharpness: mean max y {y[hv].max(1).mean():.3f} vs max p {p[hv].max(1).mean():.3f}; "
          f"entropy y {ent(y[hv]).mean():.3f} vs p {ent(p[hv]).mean():.3f}")
    he = ent(y[hv])
    for lo, hi, lab in ((-1, 0.01, "unanimous"), (0.01, 0.7, "mild disagreement"), (0.7, 9, "strong disagreement")):
        m = (he > lo) & (he <= hi)
        if m.any():
            print(f"  target {lab:19s} n={m.sum():5d} KL={c[hv][m].sum(1).mean():.3f}  mean max p={p[hv][m].max(1).mean():.3f}")
    prm = fit(p[hv], y[hv])
    print(f"in-sample calibration on >=10: T={prm[0]:.3f} bias={np.round(prm[1:], 2)} -> KL {kl_divergence(y[hv], apply(p[hv], prm)):.4f}")
    if "fold" in o and o.fold.nunique() > 1:
        cal = crossfit(o, 10)
        print(f"cross-fitted calibration: >=10 {kl_divergence(y[hv], cal[hv]):.4f}  all {kl_divergence(y, cal):.4f}")
    print("confusion on >=10 (rows: dominant true class, cols: mean predicted mass)")
    print(pd.DataFrame([p[hv & (dom == k)].mean(0) if (hv & (dom == k)).any() else np.full(6, np.nan) for k in range(6)],
                       index=CLASSES, columns=CLASSES).round(2).to_string())


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print("==", f)
        report(pd.read_csv(f))
