"""Live activity inference with calibrated probabilities and a latched fall-pattern event."""
from collections import deque
import time

import numpy as np

from .data import normalize_window, resample


class Predictor:
    def __init__(self, path):
        import torch
        from .model import ActivityNet, LegacyActivityNet
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.torch = torch
        self.labels = list(ck["labels"])
        self.win = int(ck["window_length"])
        self.fs = float(ck["fs"])
        self.scale = max(float(ck.get("scale", 1.0)), 1e-6)
        self.threshold = float(ck.get("confidence_threshold", 0.50))
        self.fall_threshold = float(ck.get("fall_threshold", 0.72))
        self.temperature = max(float(ck.get("temperature", 1.0)), 0.05)
        self.fall_confirm_windows = max(1, int(ck.get("fall_confirm_windows", 2)))
        self.version = ck.get("version", "legacy")
        self.normalize = bool(ck.get("normalize", True))
        legacy = ck.get("version") == "athena-own-1" or "temporal_context.0.net.0.weight" not in ck["state_dict"]
        model_cls = LegacyActivityNet if legacy else ActivityNet
        self.model = model_cls(int(ck["feature_count"]), len(self.labels))
        self.model.load_state_dict(ck["state_dict"], strict=True)
        self.model.eval()
        self._hist = deque(maxlen=5)
        self._fall_hits = 0
        self._fall_clear_hits = 0
        self._fall_latched = False
        self._fall_active_until = 0.0
        self._last_alert = 0.0
        self._last_result = None

    @property
    def seconds_needed(self):
        return self.win / self.fs

    def acknowledge_fall(self):
        self._fall_active_until = 0.0
        self._fall_hits = 0
        # Keep the event latched until several clear windows avoid instant re-alerting.
        self._fall_latched = True
        self._fall_clear_hits = 0
        self._last_result = None

    def predict(self, ts, amp):
        series = resample(ts, amp, self.fs)
        if len(series) < self.win:
            return None
        w = series[-self.win:]
        if not np.isfinite(w).all():
            return None
        x = (normalize_window(w) if self.normalize else w) / self.scale
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x[None]).float())
            p = self.torch.softmax(logits / self.temperature, dim=1)[0].numpy()
        self._hist.append(p)
        avg = np.mean(self._hist, axis=0)
        i = int(avg.argmax())
        conf = float(avg[i])
        now = time.monotonic()
        fall_probability = float(avg[self.labels.index("fall")]) if "fall" in self.labels else 0.0
        fall_candidate = "fall" in self.labels and fall_probability >= self.fall_threshold
        if fall_candidate:
            self._fall_hits += 1
            self._fall_clear_hits = 0
        else:
            self._fall_hits = 0
            self._fall_clear_hits += 1
            if self._fall_clear_hits >= 4:
                self._fall_latched = False
        if (self._fall_hits >= self.fall_confirm_windows and not self._fall_latched
                and now - self._last_alert >= 8.0):
            self._fall_active_until = now + 18.0
            self._last_alert = now
            self._fall_latched = True
        alert = now < self._fall_active_until
        label = self.labels[i] if conf >= self.threshold else "uncertain"
        result = {
            "label": label,
            "best": self.labels[i],
            "confidence": conf,
            "probs": {lab: float(v) for lab, v in zip(self.labels, avg)},
            "fall_probability": fall_probability,
            "fall_candidate": bool(fall_candidate),
            "fall_alert": bool(alert),
            "fall_threshold": self.fall_threshold,
            "fall_confirmed_at": self._last_alert if alert else None,
        }
        self._last_result = result
        return result


def motion_energy(amp):
    """Signal variability index for display/quality monitoring, not a fall classifier."""
    a = np.asarray(amp, dtype=np.float32)
    if a.ndim != 2 or len(a) < 5 or not np.isfinite(a).all():
        return 0.0
    return float(np.mean(np.std(a, axis=0)))
