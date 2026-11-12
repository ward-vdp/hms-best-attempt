"""Models. Every model takes the batch dict (keys 'eeg', 'spec') and returns 6 logits."""
import timm
import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from .specs import EEGSpec, spec_mask


def _stack_regions(x):
    """(B, 4, F, T) -> (B, 1, 4F, T): regions stacked along frequency into one image."""
    return torch.cat(list(x.unbind(1)), dim=1).unsqueeze(1)


def eeg_image(eeg):
    """(B, 16, 10000) raw bipolar EEG -> (B, 1, 160, 1000): each channel cut into 10 rows of 1000 samples (idea: 2nd-place
    HMS solution, see docs/PLAN.md), standardised per sample."""
    x = eeg.reshape(eeg.shape[0], 1, 160, 1000)
    return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-4)


class LearnableScalogram(nn.Module):
    """Differentiable wavelet front-end: a bank of complex Morlet filters (shared over the 16 bipolar channels) whose
    weights are initialised as a CWT and then trained end-to-end. Output: log-power scalogram per channel, channels
    stacked along the frequency axis -> (B, 1, 16 * n_freq, T / stride).
    Motivation: CWT/superlet images were the strongest inputs of the 1st place (Team Sony) and a learnable EEG-to-image
    conv front-end drove VIPEEGNet (arXiv 2507.08874); this combines both (fixed-wavelet init, free to adapt)."""

    def __init__(self, n_freq=24, f_lo=0.75, f_hi=40.0, ksize=401, stride=16, fs=200.0, learnable=True):
        super().__init__()
        f = torch.logspace(math.log10(f_lo), math.log10(f_hi), n_freq)
        cyc = (f * 1.5).clamp(2.0, 7.0)                           # fewer cycles at low f so the kernel fits in 2 s
        t = (torch.arange(ksize) - ksize // 2) / fs
        sig = cyc / (2 * math.pi * f)
        env = torch.exp(-0.5 * (t[None] / sig[:, None]) ** 2)
        env = env / env.sum(1, keepdim=True)
        w = torch.cat([env * torch.cos(2 * math.pi * f[:, None] * t[None]),
                       env * torch.sin(2 * math.pi * f[:, None] * t[None])], 0)  # (2F, K)
        self.weight = nn.Parameter(w[:, None, :].clone(), requires_grad=learnable)
        self.n_freq, self.stride, self.pad = n_freq, stride, ksize // 2

    def forward(self, eeg):
        b, c, t = eeg.shape
        z = F.conv1d(eeg.reshape(b * c, 1, t).float(), self.weight.float(), stride=self.stride, padding=self.pad)
        pw = z[:, :self.n_freq] ** 2 + z[:, self.n_freq:] ** 2
        x = torch.log(pw + 1e-6).reshape(b, 1, c * self.n_freq, -1)
        return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-4)


class ImageNet(nn.Module):
    """2D CNN on an image. source='spec' (Kaggle spectrogram), 'eegspec' (STFT computed from the EEG), 'eegimg' (raw EEG)
    or 'eegcwt' (learnable Morlet scalogram).

    siamese=True: hemisphere-symmetric variant. The image always consists of 4 equal bands along its height, ordered
    LL, LP, RP, RL (left/right lateral and parasagittal chains). The left half [LL, LP] and the mirrored right half
    [RL, RP] go through the SAME backbone; the features are combined as [fL + fR, |fL - fR|]. This is exactly invariant
    to left-right mirroring and gives the head an explicit asymmetry feature: 4 of the 6 classes differ only in being
    lateralised (LPD, LRDA) or generalised (GPD, GRDA)."""

    def __init__(self, source="spec", backbone="efficientnet_b0", pretrained=True, drop=0.3, drop_path=0.1, mask=0,
                 siamese=False):
        super().__init__()
        self.source, self.mask, self.siamese = source, mask, siamese
        self.espec = EEGSpec() if source == "eegspec" else None
        self.scal = LearnableScalogram() if source == "eegcwt" else None
        self.net = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=6,
                                     drop_rate=drop, drop_path_rate=drop_path)
        self.feat_dim = self.net.num_features * (2 if siamese else 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.feat_dim, 6)) if siamese else None

    def image(self, batch):
        if self.source == "eegimg":
            return eeg_image(batch["eeg"])
        if self.source == "eegcwt":
            x = self.scal(batch["eeg"])
            return spec_mask(x, self.mask) if (self.training and self.mask) else x
        x = self.espec(batch["eeg"]) if self.source == "eegspec" else batch["spec"]
        if self.training and self.mask:
            x = spec_mask(x, self.mask)
        return _stack_regions(x)

    def _f(self, x):
        return self.net.forward_head(self.net.forward_features(x), pre_logits=True)

    def feat(self, batch):
        x = self.image(batch)
        if not self.siamese:
            return self._f(x)
        h = x.shape[2] // 4
        q = [x[:, :, i * h:(i + 1) * h] for i in range(4)]
        left, right = torch.cat([q[0], q[1]], 2), torch.cat([q[3], q[2]], 2)
        fl, fr = self._f(torch.cat([left, right], 0)).chunk(2)
        return torch.cat([fl + fr, (fl - fr).abs()], 1)

    def forward(self, batch):
        if self.siamese:
            return self.head(self.feat(batch))
        return self.net(self.image(batch))


class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) \
            if (cin != cout or stride != 1) else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.act(y + self.sc(x))


class EEG1D(nn.Module):
    """1D residual CNN over the raw 16-channel EEG followed by a BiGRU (time-resolved pooling)."""

    def __init__(self, width=48, drop=0.3, gru=128):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(16, w, 15, 2, 7, bias=False), nn.BatchNorm1d(w), nn.SiLU())
        chans = [(w, w, 2), (w, 2 * w, 2), (2 * w, 2 * w, 2), (2 * w, 4 * w, 2), (4 * w, 4 * w, 2)]
        self.blocks = nn.Sequential(*[ResBlock1d(a, b, s) for a, b, s in chans])
        self.gru = nn.GRU(4 * w, gru, batch_first=True, bidirectional=True)
        self.feat_dim = 4 * gru
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * gru, 6))

    def feat(self, batch):
        x = self.blocks(self.stem(batch["eeg"]))          # (B, C, T/64)
        x, _ = self.gru(x.transpose(1, 2))
        return torch.cat([x.mean(1), x.amax(1)], dim=1)

    def forward(self, batch):
        return self.head(self.feat(batch))


class Fusion(nn.Module):
    """Late fusion of feature vectors from several sub-models (spec CNN + EEG CNN, ...).

    vote_cond=True: annotator-count conditioning. The fused feature vector is modulated (FiLM) by the number of
    annotators n of the training label. Low-n and high-n windows are different populations in this data set (few votes:
    mostly clear-cut, near one-hot labels; many votes: harder, mixed labels), and the test set contains only many-vote
    windows. Conditioning lets one network learn both label 'styles' from ALL data; at inference n is unknown, so it is
    fixed to vote_ref, i.e. we ask for the many-annotator answer."""

    def __init__(self, parts, drop=0.3, vote_cond=False, vote_ref=15.0):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        d = sum(p.feat_dim for p in self.parts)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 6))
        self.vote_ref = vote_ref
        self.cond = None
        if vote_cond:
            self.cond = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 2 * d))
            nn.init.zeros_(self.cond[2].weight); nn.init.zeros_(self.cond[2].bias)

    def forward(self, batch):
        f = torch.cat([p.feat(batch) for p in self.parts], dim=1)
        if self.cond is not None:
            n = batch["n"].float() if self.training else torch.full((f.shape[0],), self.vote_ref, device=f.device)
            s = (torch.log(n.clamp(1.0, 30.0)) / math.log(30.0)).unsqueeze(1)
            g, b = self.cond(s).float().chunk(2, dim=1)
            f = f * (1 + torch.tanh(g)) + b
        return self.head(f)


def build_model(cfg: dict) -> nn.Module:
    kind = cfg["kind"]
    if kind in ("spec", "eegspec", "eegimg", "eegcwt"):
        return ImageNet(kind, cfg.get("backbone", "efficientnet_b0"), cfg.get("pretrained", True),
                        cfg.get("drop", 0.3), cfg.get("drop_path", 0.1), cfg.get("mask", 0),
                        cfg.get("siamese", False))
    if kind == "eeg1d":
        return EEG1D(cfg.get("width", 48), cfg.get("drop", 0.3))
    if kind == "fusion":
        parts = [build_model(p) for p in cfg["parts"]]
        return Fusion(parts, cfg.get("drop", 0.3), cfg.get("vote_cond", False), cfg.get("vote_ref", 15.0))
    raise ValueError(kind)
