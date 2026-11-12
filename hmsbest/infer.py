"""Offline inference for the Kaggle code competition (no internet needed, no cache needed).

    python -m hmsbest.infer --data_root /kaggle/input/hms-... --exp /kaggle/input/mymodels/specb0:0.5 \
        --exp /kaggle/input/mymodels/eeg1d:0.5 --out submission.csv
Each --exp dir holds config.yaml + fold*.pt written by train.py; ':w' is the blend weight.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import CLASSES, MIRROR_CHANNELS, MIRROR_REGIONS
from .models import build_model
from .signal import eeg_bipolar, eeg_window, spec_log, spec_window


def load_models(exp_dir, dev):
    cfg = yaml.safe_load(open(os.path.join(exp_dir, "config.yaml")))
    mcfg = copy_no_pretrained(cfg["model"])
    models = []
    for p in sorted(glob.glob(os.path.join(exp_dir, "fold*.pt"))):
        m = build_model(mcfg)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.to(dev).eval())
    return cfg, models


def copy_no_pretrained(m):
    m = dict(m); m["pretrained"] = False
    if "parts" in m:
        m["parts"] = [copy_no_pretrained(p) for p in m["parts"]]
    return m


@torch.no_grad()
def predict_batch(models, batch, tta=True):
    ps = []
    for m in models:
        ps.append(m(batch).float().softmax(1))
        if tta:
            fb = dict(batch); fb["eeg"] = batch["eeg"][:, MIRROR_CHANNELS]; fb["spec"] = batch["spec"][:, MIRROR_REGIONS]
            ps.append(m(fb).float().softmax(1))
    return torch.stack(ps).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--exp", action="append", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(os.path.join(a.data_root, "test.csv"))
    exps = []
    for e in a.exp:
        d, _, w = e.partition(":")
        exps.append((*load_models(d, dev), float(w or 1.0)))
    wsum = sum(w for _, _, w in exps)
    out = np.zeros((len(test), 6), np.float64)
    for s in range(0, len(test), a.bs):
        rows = test.iloc[s:s + a.bs]
        raw_e, spec = [], []
        for _, r in rows.iterrows():
            raw_e.append(eeg_bipolar(pd.read_parquet(f"{a.data_root}/test_eegs/{r.eeg_id}.parquet")))
            sp = spec_log(pd.read_parquet(f"{a.data_root}/test_spectrograms/{r.spectrogram_id}.parquet"))
            spec.append(spec_window(sp, 0.0))
        spec_t = torch.from_numpy(np.stack(spec)).to(dev)
        eeg_by_hi = {}          # experiments may use different EEG low-pass cut-offs
        for cfg, models, w in exps:
            hi = float(cfg.get("eeg_hi", 40.0))
            if hi not in eeg_by_hi:
                eeg_by_hi[hi] = torch.from_numpy(np.stack([eeg_window(e, 0.0, hi=hi) for e in raw_e])).to(dev)
            batch = dict(eeg=eeg_by_hi[hi], spec=spec_t)
            out[s:s + len(rows)] += w / wsum * predict_batch(models, batch, cfg.get("tta", True)).cpu().numpy()
    out = out / out.sum(1, keepdims=True)
    sub = pd.DataFrame(out, columns=[f"{c}_vote" for c in CLASSES]); sub.insert(0, "eeg_id", test.eeg_id.values)
    sub.to_csv(a.out, index=False)
    print(sub.head())


if __name__ == "__main__":
    main()
