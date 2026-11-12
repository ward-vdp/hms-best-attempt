"""Metadata handling: soft targets, patient-grouped stratified folds."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import TARGET_COLS, VOTE_COLS


def load_train_meta(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    v = df[VOTE_COLS].to_numpy(np.float32)
    n = v.sum(axis=1)
    df["n_votes"] = n
    df[TARGET_COLS] = v / n[:, None]
    return df


def assign_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Patient-grouped, class-stratified folds. Fold is constant per eeg_id (and per patient)."""
    agg = {c: (c, "sum") for c in VOTE_COLS}
    e = df.groupby("eeg_id").agg(patient_id=("patient_id", "first"), **agg).reset_index()
    y = e[VOTE_COLS].to_numpy().argmax(axis=1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(e), -1)
    for k, (_, va) in enumerate(sgkf.split(e, y, groups=e["patient_id"])):
        fold[va] = k
    e["fold"] = fold
    out = df.drop(columns=["fold"], errors="ignore").merge(e[["eeg_id", "fold"]], on="eeg_id", how="left")
    return out
