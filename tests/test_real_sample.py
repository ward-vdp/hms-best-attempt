"""Runs on the single real EEG/spectrogram pair that ships locally (skipped if absent)."""
import os

import numpy as np
import pandas as pd
import pytest

from hmsbest.signal import eeg_bipolar, eeg_window, spec_log, spec_window

REAL = os.environ.get("HMS_REAL_SAMPLE", "/tmp/hmsdata")


@pytest.mark.skipif(not os.path.exists(f"{REAL}/1628180742.parquet"), reason="no real sample")
def test_real_eeg_and_spec():
    e = eeg_bipolar(pd.read_parquet(f"{REAL}/1628180742.parquet"))
    assert e.shape == (16, 18000)
    for off in (0.0, 6.0, 8.0):
        w = eeg_window(e, off)
        assert np.isfinite(w).all() and w.std() > 0
    s = spec_log(pd.read_parquet(f"{REAL}/353733.parquet"))
    assert s.shape == (320, 400) and np.isfinite(s).all()
    assert spec_window(s, 6.0).shape == (4, 100, 300)
