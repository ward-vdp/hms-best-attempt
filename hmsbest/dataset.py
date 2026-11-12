"""Datasets. Unit = eeg_id. Train: a random labelled window of the recording each epoch."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import TARGET_COLS
from .signal import eeg_window, spec_window


class HMSDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, cache_dir: str, mode: str = "train",
                 min_votes: int = 0, spec_rows: int = 300, filt: bool = True):
        assert mode in ("train", "eval")
        self.cache, self.mode, self.spec_rows, self.filt = cache_dir, mode, spec_rows, filt
        r = rows[rows.n_votes >= min_votes] if mode == "train" else rows
        self.groups = []
        for eeg_id, g in r.groupby("eeg_id", sort=True):
            self.groups.append(dict(
                eeg_id=int(eeg_id), spec_id=int(g.spectrogram_id.iloc[0]),
                eeg_off=g.eeg_label_offset_seconds.to_numpy(np.float32),
                spec_off=g.spectrogram_label_offset_seconds.to_numpy(np.float32),
                y=g[TARGET_COLS].to_numpy(np.float32), n=g.n_votes.to_numpy(np.float32)))
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
            eeg=torch.from_numpy(eeg_window(eeg, g["eeg_off"][k], lo=0.5 if self.filt else None)),
            spec=torch.from_numpy(spec_window(spec, g["spec_off"][k], self.spec_rows)),
            y=torch.from_numpy(g["y"][k]), n=torch.tensor(g["n"][k]),
            eeg_id=g["eeg_id"])
