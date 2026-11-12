"""Teacher-refined labels.

Model: the n votes of a window are a Multinomial(n, q) sample of the unknown consensus distribution q. An out-of-fold
teacher prediction p is used as a Dirichlet prior q ~ Dir(tau * p); the posterior mean is

    y' = (n * y + tau * p) / (n + tau),

so tau is the teacher's worth in "pseudo-annotators": windows with few votes are pulled strongly towards the teacher,
windows with many votes hardly move. tau is not hand-picked: it is the maximum-likelihood concentration of the
Dirichlet-multinomial on votes the teacher did not see (estimate_tau).

Leakage rules (see docs/PLAN.md):
  1. A row's teacher prediction must come from a model that did not train on that row's patient (out-of-fold),
     otherwise the teacher just echoes the memorised noisy label.
  2. For an unbiased validation score of a student validated on fold f, the teacher may not have seen fold f either
     (its soft targets would carry information about the student's validation labels). Hence `exclude_folds`:
     teachers for student f are trained on the remaining folds only (model T_{f,g}: trained without f and g, predicts g).
"""
import glob

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .constants import CLASSES, VOTE_COLS

PRED_COLS = [f"pred_{c}" for c in CLASSES]


def refine(y, n, p, tau):
    """Posterior-mean label. y: (.., 6) vote fractions, n: (..,) vote counts, p: (.., 6) teacher, tau: scalar."""
    n = np.asarray(n, np.float64)[..., None]
    return ((n * y + tau * p) / (n + tau)).astype(np.float32)


def dm_loglik(votes, p, tau):
    """Mean Dirichlet-multinomial log-likelihood of integer vote counts under prior Dir(tau * p) (constant dropped)."""
    a = tau * np.clip(p, 1e-4, 1.0)
    a = a / a.sum(1, keepdims=True) * tau
    n = votes.sum(1)
    ll = gammaln(tau) - gammaln(n + tau) + (gammaln(votes + a) - gammaln(a)).sum(1)
    return float(ll.mean())


def estimate_tau(votes, p, grid=None):
    grid = np.geomspace(0.25, 64, 33) if grid is None else grid
    ll = [dm_loglik(votes, p, t) for t in grid]
    return float(grid[int(np.argmax(ll))])


def load_teacher(pattern):
    """Read row-prediction csv(s) (label_id + pred_*), averaging duplicates (several teachers/seeds)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return t.groupby("label_id")[PRED_COLS].mean()


def attach(df, teacher, tau="auto"):
    """Add teacher columns t_<class> to the meta frame (NaN where no prediction) and resolve tau."""
    out = df.merge(teacher.rename(columns=dict(zip(PRED_COLS, [f"t_{c}" for c in CLASSES]))),
                   left_on="label_id", right_index=True, how="left")
    has = out[f"t_{CLASSES[0]}"].notna().to_numpy()
    if tau == "auto":
        tau = estimate_tau(out.loc[has, VOTE_COLS].to_numpy(np.float64),
                           out.loc[has, [f"t_{c}" for c in CLASSES]].to_numpy(np.float64))
    return out, float(tau), float(has.mean())
