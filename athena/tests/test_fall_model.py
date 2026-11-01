"""Fall-path smoke test: gap handling, model training, checkpoint and live inference."""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from athena import data  # noqa: E402
from athena.live import Predictor  # noqa: E402
from athena.train import train  # noqa: E402
import fake_esp  # noqa: E402


def make_realistic_shape_only_demo(root):
    """Synthetic data tests code paths only; it is not real fall CSI or accuracy evidence."""
    rng = np.random.default_rng(21)
    base = 65 + 18 * np.sin(np.linspace(0, 5, 64))
    for label in ("empty", "walking", "fall"):
        for take in range(4):
            n = 9 * 90
            ts = (np.arange(n) * (1000.0 / 90)).astype(np.uint32)
            local_rng = np.random.default_rng(1000 + take + len(label))
            amp = np.stack([fake_esp.frame(i / 90.0 + take * 0.07, label, base, local_rng)
                            for i in range(n)]).astype(np.uint8)
            data.save_take(root, label, f"person{take % 2}", f"room{take % 2}", ts,
                           np.full(n, -48, np.int8), amp)


def main():
    tmp = tempfile.mkdtemp(prefix="athena-fall-test-")
    try:
        # A long packet gap must not be interpolated into artificial movement.
        ts = np.r_[np.arange(100) * 20, 10000 + np.arange(100) * 20]
        amp = np.full((200, 64), 60, dtype=np.float32)
        res = data.resample(ts, amp, 50)
        assert np.isnan(res).any(), "long gaps should be represented as invalid samples"
        windows = data.make_windows(res, 64, 16)
        assert len(windows) > 0 and np.isfinite(windows).all()
        # ESP timer rollover must remain monotonic after unwrapping.
        wrap_ts = np.array([2**32 - 40, 2**32 - 20, 0, 20, 40], dtype=np.uint64)
        wrap = data.resample(wrap_ts, np.full((5, 64), 60, np.float32), 50)
        assert len(wrap) >= 3 and np.isfinite(wrap).all()
        print("gap-aware resampling and timestamp rollover ok")

        root = os.path.join(tmp, "data")
        make_realistic_shape_only_demo(root)
        metrics = train(root, os.path.join(tmp, "models"), epochs=2, fs=50, win=160, hop=25,
                        log=lambda line: print(line), patience=3, seed=21)
        assert metrics["model_version"] == "athena-fall-2.0"
        assert "fall" in metrics["labels"] and metrics["test"]["fall"] is not None
        model_path = os.path.join(tmp, "models", "activity_model.pt")
        predictor = Predictor(model_path)
        n = 500
        t = np.arange(n) * (1000.0 / 90)
        stream_rng = np.random.default_rng(9)
        stream = np.stack([fake_esp.frame(i / 90, "fall", 65 + 18*np.sin(np.linspace(0,5,64)), stream_rng)
                           for i in range(n)]).astype(np.float32)
        pred = predictor.predict(t, stream)
        assert pred and "fall_probability" in pred and "fall_alert" in pred
        print("fall training/checkpoint/live inference smoke test ok")
        print("NOTE: synthetic smoke data is not valid fall-detection performance evidence")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
