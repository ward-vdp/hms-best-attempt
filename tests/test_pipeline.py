import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hmsbest.constants import MIRROR_CHANNELS, MIRROR_REGIONS, TARGET_COLS
from hmsbest.data import assign_folds, load_train_meta
from hmsbest.dataset import HMSDataset
from hmsbest.metrics import kl_divergence
from hmsbest.models import build_model
from hmsbest.signal import eeg_bipolar, eeg_window, spec_log, spec_window
from hmsbest.specs import EEGSpec


def test_metric():
    t = np.array([[0.5, 0.5, 0, 0, 0, 0.0]])
    assert kl_divergence(t, t) == pytest.approx(0, abs=1e-9)
    assert kl_divergence(t, np.full((1, 6), 1 / 6)) == pytest.approx(np.log(3), abs=1e-6)


def test_folds_group_by_patient(fake_root):
    df = assign_folds(load_train_meta(fake_root / "train.csv"), 5, 42)
    assert df.fold.notna().all() and set(df.fold) == set(range(5))
    assert (df.groupby("patient_id").fold.nunique() == 1).all()
    assert np.allclose(df[TARGET_COLS].sum(axis=1), 1)


def test_mirror_is_involution():
    assert [MIRROR_CHANNELS[i] for i in MIRROR_CHANNELS] == list(range(16))
    assert [MIRROR_REGIONS[i] for i in MIRROR_REGIONS] == list(range(4))


def test_signal_shapes(fake_root):
    e = eeg_bipolar(pd.read_parquet(fake_root / "train_eegs" / "1000.parquet"))
    assert e.shape[0] == 16 and np.isfinite(e).all()
    w = eeg_window(e, 3.0)
    assert w.shape == (16, 10000) and np.abs(w).max() <= 10.0
    s = spec_log(pd.read_parquet(fake_root / "train_spectrograms" / "5000.parquet"))
    assert s.shape[1] == 400
    x = spec_window(s, 4.0)
    assert x.shape == (4, 100, 300) and abs(x.mean()) < 1e-3
    assert spec_window(s, 4.0, 120).shape == (4, 100, 120)


def test_region_order_in_spec(fake_root):
    """LL,LP,RP,RL: block 1 (LP) must come from LP_* columns, although files list LL,RL,LP,RP."""
    df = pd.read_parquet(fake_root / "train_spectrograms" / "5000.parquet")
    s = spec_log(df)
    lp = np.log(np.clip(df[[c for c in df.columns if c.startswith("LP_")]].to_numpy(), np.exp(-4), np.exp(8)))
    assert np.allclose(s[:, 100:200], lp, atol=1e-5)


def test_dataset_and_models(fake_root, cache):
    df = assign_folds(load_train_meta(fake_root / "train.csv"), 5, 42)
    ds = HMSDataset(df, str(cache), "train")
    item = ds[0]
    assert item["eeg"].shape == (16, 10000) and item["spec"].shape == (4, 100, 300)
    ev = HMSDataset(df, str(cache), "eval")
    assert len(ev) == df.eeg_id.nunique()
    batch = torch.utils.data.default_collate([ev[i] for i in range(2)])
    f = EEGSpec()(batch["eeg"])
    assert f.shape[:2] == (2, 4) and torch.isfinite(f).all()
    cfgs = [dict(kind="spec", backbone="efficientnet_b0", pretrained=False),
            dict(kind="eegspec", backbone="efficientnet_b0", pretrained=False),
            dict(kind="eeg1d"),
            dict(kind="fusion", parts=[dict(kind="spec", backbone="efficientnet_b0", pretrained=False),
                                       dict(kind="eeg1d")])]
    for c in cfgs:
        out = build_model(c).eval()(batch)
        assert out.shape == (2, 6) and torch.isfinite(out).all(), c


def test_train_and_infer_end_to_end(fake_root, cache, tmp_path):
    cfg = dict(name="t", seed=1, folds=3, run_folds=[0], data_root=str(fake_root), cache_dir=str(cache),
               out_dir=str(tmp_path / "runs"), results_csv=str(tmp_path / "results.csv"), batch_size=4,
               num_workers=0, limit_eegs=8, max_steps=2, ema=0.9,
               model=dict(kind="eeg1d"), aug=dict(mirror=True, polarity=True),
               stages=[dict(epochs=1, lr=1e-3, min_votes=0, weight="votes"),
                       dict(epochs=1, lr=1e-4, min_votes=6, weight="none")])
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "-m", "hmsbest.train", "--config", str(p)], check=True)
    res = pd.read_csv(tmp_path / "results.csv")
    assert np.isfinite(res.kl_all.iloc[0])
    out = tmp_path / "sub.csv"
    subprocess.run([sys.executable, "-m", "hmsbest.infer", "--data_root", str(fake_root), "--exp",
                    str(tmp_path / "runs" / "t") + ":1.0", "--out", str(out)], check=True)
    sub = pd.read_csv(out)
    assert len(sub) == 5 and np.allclose(sub.iloc[:, 1:].sum(axis=1), 1)


def test_eeg_window_survives_inf():
    x = np.random.default_rng(0).normal(0, 20, (16, 12000)).astype(np.float16)
    x[3, 5000] = np.inf
    x[5, 6000] = -np.inf
    w = eeg_window(x, 0.0)
    assert np.isfinite(w).all()


def test_mask_and_mixup():
    from hmsbest.specs import spec_mask
    from hmsbest.train import mixup
    x = torch.ones(3, 4, 100, 50)
    m = spec_mask(x)
    assert m.shape == x.shape and (m == 0).any() and (m == 1).any()
    b = dict(eeg=torch.randn(4, 16, 100), spec=torch.randn(4, 4, 10, 10), y=torch.eye(6)[:4], n=torch.ones(4) * 5)
    torch.manual_seed(0); np.random.seed(0)
    out = mixup(dict(b), 0.4, 1.0)
    assert torch.allclose(out["y"].sum(1), torch.ones(4), atol=1e-5)
    m = build_model(dict(kind="spec", backbone="efficientnet_b0", pretrained=False, mask=2)).train()
    assert m(dict(spec=torch.randn(2, 4, 100, 300))).shape == (2, 6)


def test_eegimg_model_and_target_mean(fake_root, cache):
    df = assign_folds(load_train_meta(fake_root / "train.csv"), 5, 42)
    ds = HMSDataset(df, str(cache), "train", target_mean_w=0.5, eeg_hi=20.0)
    it = ds[0]
    assert it["y"].shape == (6,) and abs(float(it["y"].sum()) - 1) < 1e-5
    ev = HMSDataset(df, str(cache), "eval", target_mean_w=0.5)          # eval labels must stay the window labels
    assert ev.tw == 0.0
    b = torch.utils.data.default_collate([ds[0], ds[1]])
    for c in (dict(kind="eegimg", backbone="efficientnet_b0", pretrained=False),
              dict(kind="fusion", parts=[dict(kind="eegimg", backbone="efficientnet_b0", pretrained=False),
                                         dict(kind="eegspec", backbone="efficientnet_b0", pretrained=False, mask=2)])):
        m = build_model(c).train()
        assert m(b).shape == (2, 6)


def test_label_smoothing_and_shift(fake_root, cache):
    df = assign_folds(load_train_meta(fake_root / "train.csv"), 5, 42)
    base = HMSDataset(df, str(cache), "train")
    ds = HMSDataset(df, str(cache), "train", label_smooth=0.02, shift_s=5.0)
    for i in range(3):
        it = ds[i]
        assert abs(float(it["y"].sum()) - 1) < 1e-5 and it["eeg"].shape == (16, 10000)
        assert torch.isfinite(it["eeg"]).all()
    # smoothing pulls a pure label towards uniform, more so for few votes
    y, n = np.array([1, 0, 0, 0, 0, 0], np.float32), 3.0
    ys = (y * n + 0.02) / (n + 6 * 0.02)
    assert ys[0] < 1 and ys[1] > 0
    ev = HMSDataset(df, str(cache), "eval", label_smooth=0.02, shift_s=5.0)
    assert ev.ls == 0.0 and ev.shift == 0.0
