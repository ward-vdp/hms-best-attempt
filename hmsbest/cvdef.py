"""How much does the CV definition move the number? Same out-of-fold row predictions, three ways to score them.

  ours   : per eeg_id the window with the most votes; its votes vs its prediction (what RESULTS.md reports).
  sony   : per eeg_id all windows' votes summed and normalised; prediction = mean over the eeg_id's windows
           (closest to the 1st-place CV, whose CV ~0.22-0.23 went with private 0.272).
  mixed  : summed votes per eeg_id, but the prediction of the most-voted window only.
All on the eeg_ids whose most-voted window has >= min_votes votes (the test-like population).

    python -m hmsbest.cvdef --train /kaggle/input/.../train.csv --rows '/kaggle/input/**/runs/pair_*/rows_for*.csv'
"""
import argparse
import glob

import numpy as np
import pandas as pd

from .constants import CLASSES, VOTE_COLS
from .metrics import kl_divergence

PC = [f"pred_{c}" for c in CLASSES]


def scores(train, preds, min_votes=10):
    df = train.merge(preds, on="label_id", how="inner")
    df["n"] = df[VOTE_COLS].sum(1)
    top = df.sort_values(["eeg_id", "n"], ascending=[True, False], kind="stable").drop_duplicates("eeg_id")
    top = top[top.n >= min_votes]
    sub = df[df.eeg_id.isin(top.eeg_id)]
    votes = sub.groupby("eeg_id")[VOTE_COLS].sum().loc[top.eeg_id].to_numpy(np.float64)
    pmean = sub.groupby("eeg_id")[PC].mean().loc[top.eeg_id].to_numpy(np.float64)
    ptop = top[PC].to_numpy(np.float64)
    return dict(n_eeg=len(top),
                ours=kl_divergence(top[VOTE_COLS].to_numpy(np.float64), ptop),
                sony=kl_divergence(votes, pmean),
                mixed=kl_divergence(votes, ptop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--rows", required=True, help="glob of row-prediction csv files (label_id + pred_*); duplicates averaged")
    ap.add_argument("--min_votes", type=int, default=10)
    a = ap.parse_args()
    files = sorted(glob.glob(a.rows, recursive=True))
    preds = pd.concat([pd.read_csv(f) for f in files]).groupby("label_id")[PC].mean().reset_index()
    print(f"{len(files)} files, {len(preds)} rows with a prediction")
    for mv in (a.min_votes, 0):
        print(f"min_votes={mv}:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores(pd.read_csv(a.train), preds, mv).items()})


if __name__ == "__main__":
    main()
