"""Cross-validated training. Usage:
    python -m hmsbest.train --config configs/spec_b0.yaml [--set key=value ...]
Every run is logged (config + CV KL) in outputs/results.csv; OOF predictions are saved per run.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .constants import CLASSES, MIRROR_CHANNELS, MIRROR_REGIONS, TARGET_COLS
from .data import assign_folds, load_train_meta
from .dataset import HMSDataset, RowDataset
from .metrics import kl_divergence
from .models import build_model
from .teacher import PRED_COLS, attach, load_teacher


def load_config(path, overrides=()):
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d[kk]
        d[ks[-1]] = yaml.safe_load(v)
    return cfg


def kl_loss(logits, y, n=None, weight="none"):
    lp = F.log_softmax(logits.float(), dim=1)
    per = (y * (torch.log(y.clamp_min(1e-8)) - lp)).sum(1)
    if weight == "votes":                       # trust windows with more annotators more
        w = n.clamp(max=20) / 20.0
        return (per * w).sum() / w.sum()
    return per.mean()


def augment(batch, cfg):
    b = batch["y"].shape[0]
    dev = batch["y"].device
    if cfg.get("mirror", True):
        m = torch.rand(b, device=dev) < 0.5
        if m.any():
            batch["eeg"][m] = batch["eeg"][m][:, MIRROR_CHANNELS]
            batch["spec"][m] = batch["spec"][m][:, MIRROR_REGIONS]
    if cfg.get("polarity", True):
        batch["eeg"] = batch["eeg"] * (torch.randint(0, 2, (b, 1, 1), device=dev) * 2 - 1)
    return batch


def mixup(batch, alpha, p):
    """Mix inputs and soft targets of random pairs (one lambda per batch), with probability p."""
    if alpha <= 0 or np.random.rand() > p:
        return batch
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(batch["y"].shape[0], device=batch["y"].device)
    for k in ("eeg", "spec", "y", "n"):
        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]
    return batch


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, dev, tta=True):
    model.eval()
    P, Y, N, ID = [], [], [], []
    for batch in loader:
        ID.append(batch["eeg_id"].numpy())
        batch = to_dev(batch, dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            lo = model(batch).float().softmax(1)
            if tta:
                fb = dict(batch)
                fb["eeg"] = batch["eeg"][:, MIRROR_CHANNELS]
                fb["spec"] = batch["spec"][:, MIRROR_REGIONS]
                lo = 0.5 * (lo + model(fb).float().softmax(1))
        P.append(lo.cpu().numpy()); Y.append(batch["y"].cpu().numpy())
        N.append(batch["n"].cpu().numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(N), np.concatenate(ID)


@torch.no_grad()
def predict_rows(model, rows, cfg, dkw, dev):
    """Mirror-TTA predictions for every row (window) of `rows` -> DataFrame(label_id, pred_*)."""
    model.eval()
    loader = DataLoader(RowDataset(rows, **dkw), batch_size=cfg["batch_size"] * 2, num_workers=cfg.get("num_workers", 4))
    P, L = [], []
    for batch in loader:
        L.append(batch["label_id"].numpy())
        batch = to_dev(batch, dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = model(batch).float().softmax(1)
            fb = dict(batch); fb["eeg"] = batch["eeg"][:, MIRROR_CHANNELS]; fb["spec"] = batch["spec"][:, MIRROR_REGIONS]
            p = 0.5 * (p + model(fb).float().softmax(1))
        P.append(p.cpu().numpy())
    out = pd.DataFrame(np.concatenate(P), columns=PRED_COLS)
    out.insert(0, "label_id", np.concatenate(L))
    return out


def run_fold(cfg, df, fold, dev, out_dir):
    tr, va = df[df.fold != fold], df[df.fold == fold]
    dkw = dict(cache_dir=cfg["cache_dir"], spec_rows=cfg.get("spec_rows", 300), filt=cfg.get("filt", True),
               eeg_hi=cfg.get("eeg_hi", 40.0))
    val_ds = HMSDataset(va, mode="eval", **dkw)
    nw = cfg.get("num_workers", 4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, num_workers=nw)

    torch.manual_seed(cfg["seed"] + fold)
    model = build_model(cfg["model"]).to(dev)
    d = cfg.get("ema", 0.998)
    # EMA with warm-up: decay grows from ~0.1 to d, so short runs are not stuck near the initial weights
    ema = AveragedModel(model, use_buffers=True, avg_fn=lambda a, p, n: a + (p - a) * (1 - min(d, (1 + n) / (10 + n))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    t0 = time.time()
    for si, st in enumerate(cfg["stages"]):
        ds = HMSDataset(tr, mode="train", min_votes=st.get("min_votes", 0),
                        target_mean_w=cfg.get("target_mean_w", 0.0),
                        label_smooth=cfg.get("label_smooth", 0.0), shift_s=cfg.get("shift_s", 0.0),
                        label_prior=cfg.get("label_prior", "uniform"),
                        teacher_tau=st.get("teacher_tau", cfg.get("teacher_tau", 0.0)), **dkw)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                            drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=nw > 0)
        steps = st["epochs"] * len(loader)
        if cfg.get("max_steps"):
            steps = min(steps, cfg["max_steps"])
        opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], weight_decay=cfg.get("wd", 1e-2))
        warm = max(1, int(0.05 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        step = 0
        for ep in range(st["epochs"]):
            model.train()
            run_loss = 0.0
            for batch in loader:
                batch = augment(to_dev(batch, dev), cfg.get("aug", {}))
                batch = mixup(batch, cfg.get("aug", {}).get("mixup", 0.0), cfg.get("aug", {}).get("mixup_p", 0.5))
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    logits = model(batch)
                loss = kl_loss(logits, batch["y"], batch["n"], st.get("weight", "none"))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step(); ema.update_parameters(model)
                run_loss += loss.item(); step += 1
                if step >= steps:
                    break
            p, y, n, _ = predict(ema, val_loader, dev, tta=False)
            hv = n >= 6
            print(f"[f{fold} s{si} ep{ep}] train {run_loss / max(1, len(loader)):.4f} "
                  f"val {kl_divergence(y, p):.4f} val>=6 {kl_divergence(y[hv], p[hv]) if hv.any() else float('nan'):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if step >= steps:
                break
    p, y, n, ids = predict(ema, val_loader, dev, tta=cfg.get("tta", True))
    torch.save(ema.module.state_dict(), os.path.join(out_dir, f"fold{fold}.pt"))
    if cfg.get("save_row_preds"):               # out-of-fold predictions for ALL windows of this fold (teacher labels)
        predict_rows(ema, va, cfg, dkw, dev).to_csv(os.path.join(out_dir, f"rows_fold{fold}.csv"), index=False)
    oof = pd.DataFrame(p, columns=[f"pred_{c}" for c in CLASSES])
    oof[TARGET_COLS] = y; oof["n_votes"] = n; oof["eeg_id"] = ids; oof["fold"] = fold
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out_dir = os.path.join(cfg["out_dir"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "config.yaml"), "w"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_train_meta(os.path.join(cfg["data_root"], "train.csv"))
    if cfg.get("limit_eegs"):                  # debug subset: first N eeg_ids (matches prep --limit)
        df = df[df.eeg_id.isin(sorted(df.eeg_id.unique())[:cfg["limit_eegs"]])]
    df = assign_folds(df, cfg["folds"], cfg["seed"])
    if cfg.get("exclude_folds"):               # nested CV: these folds are invisible to this run (see hmsbest/teacher.py)
        df = df[~df.fold.isin(cfg["exclude_folds"])]
    if cfg.get("teacher_csv"):
        df, tau, cover = attach(df, load_teacher(cfg["teacher_csv"]), cfg.get("teacher_tau", "auto"))
        cfg["teacher_tau"] = tau
        for st in cfg["stages"]:
            if st.get("teacher_tau") == "auto":
                st["teacher_tau"] = tau
        print(f"teacher: tau={tau:.2f} pseudo-annotators, coverage {cover:.1%} of rows", flush=True)
    oofs = [run_fold(cfg, df, f, dev, out_dir) for f in cfg.get("run_folds", range(cfg["folds"]))]
    oof = pd.concat(oofs, ignore_index=True)
    oof.to_csv(os.path.join(out_dir, "oof.csv"), index=False)
    pc = [f"pred_{c}" for c in CLASSES]
    hv, hv10 = oof.n_votes >= 6, oof.n_votes >= 10
    res = dict(ts=time.strftime("%Y-%m-%d %H:%M"), exp=cfg["name"], folds=len(oofs),
               kl_all=kl_divergence(oof[TARGET_COLS], oof[pc]),
               kl_hv=kl_divergence(oof.loc[hv, TARGET_COLS], oof.loc[hv, pc]) if hv.any() else np.nan,
               kl_hv10=kl_divergence(oof.loc[hv10, TARGET_COLS], oof.loc[hv10, pc]) if hv10.any() else np.nan,
               cfg=json.dumps(cfg, sort_keys=True))
    path = cfg.get("results_csv", "outputs/results.csv")
    pd.DataFrame([res]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print({k: v for k, v in res.items() if k != "cfg"})


if __name__ == "__main__":
    main()
