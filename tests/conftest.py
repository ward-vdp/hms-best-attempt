import os

import numpy as np
import pandas as pd
import pytest

from hmsbest.constants import EEG_CHANNELS, REGIONS, VOTE_COLS


def _spec_cols():
    freqs = [f"{0.59 + 0.1953125 * i:.2f}" for i in range(100)]
    # real files order the regions LL, RL, LP, RP
    return ["time"] + [f"{r}_{f}" for r in ["LL", "RL", "LP", "RP"] for f in freqs]


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("hms")
    rng = np.random.default_rng(0)
    for d in ("train_eegs", "train_spectrograms", "test_eegs", "test_spectrograms"):
        os.makedirs(root / d)
    rows = []
    for k in range(30):
        eeg_id, spec_id, patient = 1000 + k, 5000 + k, k // 3
        secs = int(rng.integers(60, 100))
        t = np.arange(secs * 200) / 200
        eeg = rng.normal(0, 20, (len(t), 20)).astype(np.float32) + 30 * np.sin(2 * np.pi * 2 * t)[:, None]
        eeg[rng.integers(0, len(t), 50), 3] = np.nan
        pd.DataFrame(eeg, columns=EEG_CHANNELS).to_parquet(root / "train_eegs" / f"{eeg_id}.parquet")
        nrow = (secs + 600) // 2
        sp = np.exp(rng.normal(1, 1, (nrow, 400))).astype(np.float32)
        spdf = pd.DataFrame(sp, columns=_spec_cols()[1:]); spdf.insert(0, "time", np.arange(nrow) * 2)
        spdf.to_parquet(root / "train_spectrograms" / f"{spec_id}.parquet")
        for j in range(int(rng.integers(1, 4))):
            off = float(rng.integers(0, secs - 50))
            v = rng.multinomial(int(rng.choice([3, 8, 15])), np.ones(6) / 6)
            rows.append(dict(eeg_id=eeg_id, eeg_sub_id=j, eeg_label_offset_seconds=off, spectrogram_id=spec_id,
                             spectrogram_sub_id=j, spectrogram_label_offset_seconds=off, label_id=len(rows),
                             patient_id=patient, expert_consensus="x", **dict(zip(VOTE_COLS, v))))
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    tests = []
    for k in range(5):
        eeg_id, spec_id = 9000 + k, 9500 + k
        eeg = rng.normal(0, 20, (10000, 20)).astype(np.float32)
        pd.DataFrame(eeg, columns=EEG_CHANNELS).to_parquet(root / "test_eegs" / f"{eeg_id}.parquet")
        sp = np.exp(rng.normal(1, 1, (300, 400))).astype(np.float32)
        spdf = pd.DataFrame(sp, columns=_spec_cols()[1:]); spdf.insert(0, "time", np.arange(300) * 2)
        spdf.to_parquet(root / "test_spectrograms" / f"{spec_id}.parquet")
        tests.append(dict(eeg_id=eeg_id, spectrogram_id=spec_id, patient_id=999))
    pd.DataFrame(tests).to_csv(root / "test.csv", index=False)
    return root


@pytest.fixture(scope="session")
def cache(fake_root, tmp_path_factory):
    import subprocess, sys
    c = tmp_path_factory.mktemp("cache")
    subprocess.run([sys.executable, "-m", "hmsbest.prep", "--data_root", str(fake_root), "--cache_dir", str(c),
                    "--workers", "2"], check=True)
    return c
