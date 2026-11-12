"""Signal processing shared by training, caching and inference (so train/test inputs match)."""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .constants import BIPOLAR, EEG_CHANNELS, EEG_LEN, FS, REGIONS

_SOS = {}


def _sos(lo, hi, order=4):
    key = (lo, hi, order)
    if key not in _SOS:
        _SOS[key] = butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return _SOS[key]


def eeg_bipolar(df: pd.DataFrame) -> np.ndarray:
    """Parquet frame -> (16, T) float32 double-banana bipolar signal in uV. NaN -> 0."""
    idx = {c: i for i, c in enumerate(EEG_CHANNELS)}
    a = df[EEG_CHANNELS].to_numpy(np.float32)
    out = np.stack([a[:, idx[x]] - a[:, idx[y]] for x, y in BIPOLAR], axis=0)
    return np.nan_to_num(out, nan=0.0)


def eeg_window(arr: np.ndarray, offset_s: float, lo=0.5, hi=40.0, clip=500.0, scale=50.0) -> np.ndarray:
    """Cut the 50 s window starting at offset_s, band-pass (zero-phase) per window, clip, scale.

    Filtering is done per window (not on the full recording) so that training windows and the
    50 s test recordings undergo exactly the same edge effects.
    """
    s = int(round(offset_s * FS))
    w = np.asarray(arr[:, s:s + EEG_LEN], dtype=np.float32)
    # float16 caches can hold +-inf where the raw signal overflowed; filtering would turn that into NaN
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=2000.0, neginf=-2000.0), -2000.0, 2000.0)
    if w.shape[1] < EEG_LEN:
        w = np.pad(w, ((0, 0), (0, EEG_LEN - w.shape[1])))
    if lo is not None:
        w = sosfiltfilt(_sos(lo, hi), w, axis=-1)
    return (np.clip(w, -clip, clip) / scale).astype(np.float32)


def spec_log(df: pd.DataFrame) -> np.ndarray:
    """Kaggle spectrogram parquet -> (rows, 400) float32 log-power, columns ordered by REGIONS."""
    cols = []
    for r in REGIONS:
        cols += sorted([c for c in df.columns if c.startswith(r + "_")],
                       key=lambda c: float(c.split("_")[1]))
    x = df[cols].to_numpy(np.float32)
    x = np.log(np.clip(x, np.exp(-4), np.exp(8)))
    return np.nan_to_num(x, nan=0.0)


def spec_window(arr: np.ndarray, offset_s: float, n_rows: int = 300) -> np.ndarray:
    """(rows, 400) cached log-spectrogram -> (4, 100, n_rows) per-region standardised crop.

    The 10-min spectrogram window has 300 rows (2 s each) with the labelled 10 s in the middle;
    n_rows < 300 takes a central sub-window.
    """
    start = int(offset_s // 2) + (300 - n_rows) // 2
    x = np.asarray(arr[start:start + n_rows], dtype=np.float32)
    if x.shape[0] < n_rows:
        x = np.pad(x, ((0, n_rows - x.shape[0]), (0, 0)))
    x = x.T.reshape(4, 100, n_rows)
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True)
    return ((x - m) / (s + 1e-4)).astype(np.float32)
