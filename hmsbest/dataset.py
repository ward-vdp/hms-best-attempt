"""Datasets. Unit = eeg_id. Train: a random labelled window of the recording each epoch."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import TARGET_COLS
from .signal import eeg_window, spec_window


class HMSDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache_dir: str, mode: str = "train",
                 min_votes: int = 0, spec_rows: int = 300, filt: bool = True, eeg_hi: float = 40.0,
                 target_mean_w: float = 0.0):
        assert mode in ("train", "eval")
        self.cache, self.mode, self.spec_rows, self.filt = cache_dir, mode, spec_rows, filt
        self.eeg_hi = eeg_hi
        self.tw = target_mean_w if mode == "train" else 0.0   # train only: blend window label with the eeg_id-mean label
        r = rows[rows.n_votes >= min_votes] if mode == "train" else rows
        self.groups = []
        for eeg_id, g in r.groupby("eeg_id", sort=True):
            self.groups.append(dict(
                eeg_id=int(eeg_id), spec_id=int(g.spectrogram_id.iloc[0]),
                eeg_off=g.eeg_label_offset_seconds.to_numpy(np.float32),
                spec_off=g.spectrogram_label_offset_seconds.to_numpy(np.float32),
                y=g[TARGET_COLS].to_numpy(np.float32), n=g.n_votes.to_numpy(np.float32),
                y_mean=g[TARGET_COLS].to_numpy(np.float32).mean(0), n_mean=float(g.n_votes.mean())))
        # eval: one fixed window per eeg_id = the one with the most annotators (first on ties)
        self.eval_pick = [int(np.argmax(g["n"])) for g in self.groups]
        self.eval_meta = pd.DataFrame({
            "eeg_id": [g["eeg_id"] for g in self.groups],
            "n_votes": [g["n"][k] for g, k in zip(self.groups, self.eval_pick)]})

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]
        k = self.eval_pick[i] if self.mode == "eval" else int(torch.randint(len(g["n"]), (1,)))
        eeg = np.load(f"{self.cache}/eeg/{g['eeg_id']}.npy", mmap_mode="r")
        spec = np.load(f"{self.cache}/spec/{g['spec_id']}.npy", mmap_mode="r")
        return dict(
            eeg=torch.from_numpy(eeg_window(eeg, g["eeg_off"][k], lo=0.5 if self.filt else None, hi=self.eeg_hi)),
            spec=torch.from_numpy(spec_window(spec, g["spec_off"][k], self.spec_rows)),
            y=torch.from_numpy((1 - self.tw) * g["y"][k] + self.tw * g["y_mean"]),
            n=torch.tensor((1 - self.tw) * g["n"][k] + self.tw * g["n_mean"]),
            eeg_id=g["eeg_id"])
