"""Athena temporal model: compatible CNN/BiLSTM trunk + multi-scale residual context."""
import torch
import torch.nn as nn


class TemporalResidual(nn.Module):
    def __init__(self, channels, dilation, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ActivityNet(nn.Module):
    """Multi-scale temporal classifier for complete activity/fall transition windows."""
    def __init__(self, n_feat=64, n_cls=7, drop=0.3):
        super().__init__()
        # Keep these layer names/shapes stable so compatible old checkpoint weights can warm-start.
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop * 0.3),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(drop * 0.3),
        )
        self.temporal = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.temporal_context = nn.Sequential(
            TemporalResidual(128, 1, drop * 0.5),
            TemporalResidual(128, 2, drop * 0.5),
        )
        self.norm = nn.LayerNorm(128)
        self.attention = nn.Sequential(nn.Linear(128, 48), nn.Tanh(), nn.Linear(48, 1))
        self.head = nn.Sequential(
            nn.Linear(128 * 2, 96), nn.GELU(), nn.Dropout(drop),
            nn.Linear(96, 64), nn.GELU(), nn.Dropout(drop * 0.7), nn.Linear(64, n_cls)
        )

    def forward(self, x):  # x: (batch, time, features)
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.temporal(h)
        h = self.temporal_context(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        a = torch.softmax(self.attention(h), dim=1)
        attended = (a * h).sum(dim=1)
        # Mean context helps distinguish a short impact-like transient from sustained movement.
        pooled = h.mean(dim=1)
        return self.head(torch.cat([attended, pooled], dim=1))



class LegacyActivityNet(nn.Module):
    """Exact architecture used by the original Athena checkpoints."""
    def __init__(self, n_feat=64, n_cls=7, drop=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop * 0.3),
            nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(drop * 0.3),
        )
        self.temporal = nn.LSTM(96, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_cls))

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.temporal(h)
        a = torch.softmax(self.attention(h), dim=1)
        return self.head((a * h).sum(dim=1))


def load_pretrained(model, path):
    """Warm-start only tensors that genuinely match; never pretend the old head is compatible."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    src = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    dst = model.state_dict()
    ok = {k: v for k, v in src.items()
          if k in dst and getattr(v, "shape", None) == dst[k].shape and not k.startswith("head.")}
    model.load_state_dict(ok, strict=False)
    return len(ok)
