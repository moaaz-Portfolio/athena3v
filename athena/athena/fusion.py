"""Shared-CSI motion/presence estimators used alongside the learned fall classifier.

These are signal-state estimators, not medical or safety-grade detectors. Presence
requires an explicit empty-room calibration and must be validated per deployment.
"""
import json
import os
import time
from collections import deque

import numpy as np


class MotionPresenceFusion:
    def __init__(self, calibration_path):
        self.path = calibration_path
        self.empty_mean = None
        self.empty_std = None
        self.motion_noise = None
        self.motion_history = deque(maxlen=12)
        self.presence_hits = 0
        self.empty_hits = 0
        self.presence = "unknown"
        self.last = {"motion": "unknown", "presence": "unknown", "ready": False}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.empty_mean = np.asarray(d["empty_mean"], dtype=np.float32)
            self.empty_std = np.asarray(d["empty_std"], dtype=np.float32)
            self.motion_noise = float(d["motion_noise"])
            if self.empty_mean.shape != (64,) or self.empty_std.shape != (64,):
                raise ValueError("invalid calibration shape")
        except (OSError, ValueError, KeyError, TypeError):
            self.empty_mean = self.empty_std = self.motion_noise = None

    def calibrate_empty(self, amp):
        a = np.asarray(amp, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 64 or len(a) < 100 or not np.isfinite(a).all():
            raise ValueError("Need at least 100 valid CSI frames for empty-room calibration")
        mean = np.mean(a, axis=0)
        std = np.std(a, axis=0) + 1.0
        delta = np.mean(np.abs(np.diff(a, axis=0)), axis=1)
        noise = float(np.percentile(delta, 90)) if len(delta) else 1.0
        payload = {"empty_mean": mean.tolist(), "empty_std": std.tolist(),
                   "motion_noise": max(noise, 0.25), "created": time.time()}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)
        self.empty_mean, self.empty_std, self.motion_noise = mean, std, payload["motion_noise"]
        self.presence_hits = self.empty_hits = 0
        self.presence = "empty"
        return {"frames": int(len(a)), "motion_noise": self.motion_noise}

    def update(self, ts, amp):
        a = np.asarray(amp, dtype=np.float32)
        result = {"motion": "unknown", "motion_score": None,
                  "presence": self.presence, "presence_score": None,
                  "ready": self.empty_mean is not None, "calibrated": self.empty_mean is not None}
        if a.ndim != 2 or a.shape[1] != 64 or len(a) < 20 or not np.isfinite(a).all():
            self.last = result
            return result

        # Dynamic CSI variation: independent of the fall classifier.
        delta = np.mean(np.abs(np.diff(a, axis=0)), axis=1)
        recent_delta = float(np.mean(delta[-min(len(delta), 50):]))
        self.motion_history.append(recent_delta)
        noise = self.motion_noise if self.motion_noise is not None else max(float(np.median(delta)), 0.5)
        motion_score = recent_delta / max(noise, 0.25)
        motion_active = motion_score >= 2.2
        result["motion"] = "moving" if motion_active else "still"
        result["motion_score"] = round(float(motion_score), 3)

        # Occupancy estimate compares a rolling CSI profile with a user-calibrated empty room.
        if self.empty_mean is not None and len(a) >= 100:
            profile = np.median(a[-min(len(a), 250):], axis=0)
            z = np.abs(profile - self.empty_mean) / np.maximum(self.empty_std, 1.0)
            presence_score = float(np.median(z))
            result["presence_score"] = round(presence_score, 3)
            if presence_score >= 2.5:
                self.presence_hits += 1
                self.empty_hits = 0
            elif presence_score <= 1.5:
                self.empty_hits += 1
                self.presence_hits = 0
            else:
                self.presence_hits = max(0, self.presence_hits - 1)
                self.empty_hits = max(0, self.empty_hits - 1)
            if self.presence_hits >= 3:
                self.presence = "occupied"
            elif self.empty_hits >= 6:
                self.presence = "empty"
            result["presence"] = self.presence
        else:
            result["presence"] = "needs_calibration"
        self.last = result
        return result
