"""One-off caching of the raw parquet files as float16 .npy (fast random access for training).

    python -m hmsbest.prep --data_root /kaggle/input/hms-harmful-brain-activity-classification \
        --cache_dir /kaggle/working/cache --meta train.csv
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .signal import eeg_bipolar, spec_log


def _eeg(job):
    src, dst = job
    if not os.path.exists(dst):
        np.save(dst, np.clip(eeg_bipolar(pd.read_parquet(src)), -2000, 2000).astype(np.float16))
    return dst


def _spec(job):
    src, dst = job
    if not os.path.exists(dst):
        np.save(dst, spec_log(pd.read_parquet(src)).astype(np.float16))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--meta", default="train.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N ids")
    a = ap.parse_args()
    meta = pd.read_csv(os.path.join(a.data_root, a.meta), usecols=["eeg_id", "spectrogram_id"])
    if a.limit:   # debug subset: the first N eeg_ids (same rule as `limit_eegs` in train.py)
        meta = meta[meta.eeg_id.isin(sorted(meta.eeg_id.unique())[:a.limit])]
    eegs = sorted(meta.eeg_id.unique())
    specs = sorted(meta.spectrogram_id.unique())
    for sub in ("eeg", "spec"):
        os.makedirs(os.path.join(a.cache_dir, sub), exist_ok=True)
    tag = "train" if "train" in a.meta else "test"
    ej = [(f"{a.data_root}/{tag}_eegs/{i}.parquet", f"{a.cache_dir}/eeg/{i}.npy") for i in eegs]
    sj = [(f"{a.data_root}/{tag}_spectrograms/{i}.parquet", f"{a.cache_dir}/spec/{i}.npy") for i in specs]
    with Pool(a.workers) as p:
        for name, fn, jobs in (("eeg", _eeg, ej), ("spec", _spec, sj)):
            for k, _ in enumerate(p.imap_unordered(fn, jobs, chunksize=16)):
                if k % 1000 == 0:
                    print(f"{name} {k}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
