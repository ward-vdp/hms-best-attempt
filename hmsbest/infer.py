"""Offline inference for the Kaggle code competition (no internet needed, no cache needed).

    python -m hmsbest.infer --data_root /kaggle/input/hms-... --exp /kaggle/input/mymodels/FB:0.5 \
        --exp /kaggle/input/mymodels/FA:0.5 --out submission.csv
Each --exp dir holds config.yaml + fold*.pt written by train.py; ':w' is the blend weight.

--check N runs the exact same code path on N training windows (one per eeg_id, >= 10 votes, read from the raw train
parquets at their label offsets) and prints the KL against the expert votes: a guard against train/test preprocessing
mismatches. With full-fit models these windows were seen in training, so the number is optimistic; it only has to be
in the same range as the training loss, not ~1.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import CLASSES, MIRROR_CHANNELS, MIRROR_REGIONS
from .metrics import kl_divergence
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
    if not models:
        raise FileNotFoundError(f"no fold*.pt in {exp_dir}")
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
        with torch.autocast(batch["eeg"].device.type, enabled=batch["eeg"].device.type == "cuda"):
            ps.append(m(batch).float().softmax(1))
            if tta:
                fb = dict(batch); fb["eeg"] = batch["eeg"][:, MIRROR_CHANNELS]; fb["spec"] = batch["spec"][:, MIRROR_REGIONS]
                ps.append(m(fb).float().softmax(1))
    return torch.stack(ps).mean(0)


def predict(n, load, exps, dev, bs=32):
    """load(i) -> (raw_eeg (16,T), eeg_offset_s, log_spec (rows,400), spec_offset_s), read lazily per batch.
    Returns (n, 6) probabilities."""
    wsum = sum(w for _, _, w in exps)
    out = np.zeros((n, 6), np.float64)
    for s in range(0, n, bs):
        chunk = [load(i) for i in range(s, min(n, s + bs))]
        cache = {}              # experiments may use different EEG filters / spectrogram crops
        for cfg, models, w in exps:
            hi, lo = float(cfg.get("eeg_hi", 40.0)), (0.5 if cfg.get("filt", True) else None)
            nr = int(cfg.get("spec_rows", 300))
            if (hi, lo) not in cache:
                cache[(hi, lo)] = torch.from_numpy(np.stack([eeg_window(e, eo, lo=lo, hi=hi) for e, eo, _, _ in chunk])).to(dev)
            if nr not in cache:
                cache[nr] = torch.from_numpy(np.stack([spec_window(sp, so, nr) for _, _, sp, so in chunk])).to(dev)
            batch = dict(eeg=cache[(hi, lo)], spec=cache[nr])
            out[s:s + len(chunk)] += w / wsum * predict_batch(models, batch, cfg.get("tta", True)).cpu().numpy()
    return out / out.sum(1, keepdims=True)


def check(data_root, exps, dev, n, bs):
    from .constants import VOTE_COLS
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df["n"] = df[VOTE_COLS].sum(1)
    df = df[df.n >= 10].sort_values("n", ascending=False).drop_duplicates("eeg_id")
    df = df.sample(min(n, len(df)), random_state=0)
    rows = df.to_dict("records")

    def load(i):
        r = rows[i]
        e = eeg_bipolar(pd.read_parquet(f"{data_root}/train_eegs/{r['eeg_id']}.parquet"))
        sp = spec_log(pd.read_parquet(f"{data_root}/train_spectrograms/{r['spectrogram_id']}.parquet"))
        return e, float(r["eeg_label_offset_seconds"]), sp, float(r["spectrogram_label_offset_seconds"])
    p = predict(len(rows), load, exps, dev, bs)
    y = df[VOTE_COLS].to_numpy(np.float64); y = y / y.sum(1, keepdims=True)
    print(f"check on {len(df)} train windows (>= 10 votes, in-sample for full-fit models): KL {kl_divergence(y, p):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--exp", action="append", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exps = []
    for e in a.exp:
        d, _, w = e.partition(":")
        exps.append((*load_models(d, dev), float(w or 1.0)))
        print(f"{d}: {len(exps[-1][1])} model(s), weight {exps[-1][2]}")
    if a.check:
        check(a.data_root, exps, dev, a.check, a.bs)
        return
    test = pd.read_csv(os.path.join(a.data_root, "test.csv"))
    rows = test.to_dict("records")

    def load(i):                 # test EEG = the 50 s window, test spectrogram = the 10 min window -> offsets 0
        r = rows[i]
        e = eeg_bipolar(pd.read_parquet(f"{a.data_root}/test_eegs/{r['eeg_id']}.parquet"))
        sp = spec_log(pd.read_parquet(f"{a.data_root}/test_spectrograms/{r['spectrogram_id']}.parquet"))
        return e, 0.0, sp, 0.0
    out = predict(len(rows), load, exps, dev, a.bs)
    sub = pd.DataFrame(out, columns=[f"{c}_vote" for c in CLASSES]); sub.insert(0, "eeg_id", test.eeg_id.values)
    sub.to_csv(a.out, index=False)
    print(sub.head())


if __name__ == "__main__":
    main()
