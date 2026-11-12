"""Models. Every model takes the batch dict (keys 'eeg', 'spec') and returns 6 logits."""
import timm
import torch
import torch.nn as nn

from .specs import EEGSpec, spec_mask


def _stack_regions(x):
    """(B, 4, F, T) -> (B, 1, 4F, T): regions stacked along frequency into one image."""
    return torch.cat(list(x.unbind(1)), dim=1).unsqueeze(1)


class ImageNet(nn.Module):
    """2D CNN on a spectrogram image. source='spec' (Kaggle) or 'eegspec' (computed from the EEG)."""

    def __init__(self, source="spec", backbone="efficientnet_b0", pretrained=True, drop=0.3, drop_path=0.1, mask=0):
        super().__init__()
        self.source, self.mask = source, mask
        self.espec = EEGSpec() if source == "eegspec" else None
        self.net = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=6,
                                     drop_rate=drop, drop_path_rate=drop_path)

    def forward(self, batch):
        x = self.espec(batch["eeg"]) if self.source == "eegspec" else batch["spec"]
        if self.training and self.mask:
            x = spec_mask(x, self.mask)
        return self.net(_stack_regions(x))


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
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * gru, 6))

    def forward(self, batch):
        x = self.blocks(self.stem(batch["eeg"]))          # (B, C, T/64)
        x, _ = self.gru(x.transpose(1, 2))
        return self.head(torch.cat([x.mean(1), x.amax(1)], dim=1))


class Fusion(nn.Module):
    """Late fusion of feature vectors from several sub-models (spec CNN + EEG CNN, ...)."""

    def __init__(self, parts, drop=0.3):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        dims = []
        for p in self.parts:
            if isinstance(p, ImageNet):
                dims.append(p.net.num_features)
            else:
                dims.append(p.head[1].in_features)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(sum(dims), 6))

    @staticmethod
    def _feat(p, batch):
        if isinstance(p, ImageNet):
            x = p.espec(batch["eeg"]) if p.source == "eegspec" else batch["spec"]
            if p.training and p.mask:
                x = spec_mask(x, p.mask)
            return p.net.forward_head(p.net.forward_features(_stack_regions(x)), pre_logits=True)
        x = p.blocks(p.stem(batch["eeg"]))
        x, _ = p.gru(x.transpose(1, 2))
        return torch.cat([x.mean(1), x.amax(1)], dim=1)

    def forward(self, batch):
        return self.head(torch.cat([self._feat(p, batch) for p in self.parts], dim=1))


def build_model(cfg: dict) -> nn.Module:
    kind = cfg["kind"]
    if kind in ("spec", "eegspec"):
        return ImageNet(kind, cfg.get("backbone", "efficientnet_b0"), cfg.get("pretrained", True),
                        cfg.get("drop", 0.3), cfg.get("drop_path", 0.1), cfg.get("mask", 0))
    if kind == "eeg1d":
        return EEG1D(cfg.get("width", 48), cfg.get("drop", 0.3))
    if kind == "fusion":
        parts = [build_model(p) for p in cfg["parts"]]
        return Fusion(parts, cfg.get("drop", 0.3))
    raise ValueError(kind)
