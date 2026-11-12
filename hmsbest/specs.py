"""EEG -> log-power spectrogram on the (GPU) device, per brain region."""
import torch
import torch.nn as nn

from .constants import FS


class EEGSpec(nn.Module):
    """(B, 16, T) bipolar EEG -> (B, 4, F, frames) standardised log power (mean over each chain)."""

    def __init__(self, n_fft=512, win=128, hop=50, fmin=0.5, fmax=40.0):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        f = torch.fft.rfftfreq(n_fft, 1.0 / FS)
        self.register_buffer("sel", torch.nonzero((f >= fmin) & (f <= fmax)).squeeze(1), persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        x = x.float()
        b, c, t = x.shape
        s = torch.stft(x.reshape(b * c, t), self.n_fft, self.hop, self.win, self.window,
                       center=True, return_complex=True).abs().pow(2)
        s = s[:, self.sel]
        s = s.reshape(b, 4, c // 4, s.shape[1], s.shape[2]).mean(2)
        s = torch.log(s + 1e-3)
        m = s.mean(dim=(2, 3), keepdim=True)
        sd = s.std(dim=(2, 3), keepdim=True)
        return (s - m) / (sd + 1e-4)
