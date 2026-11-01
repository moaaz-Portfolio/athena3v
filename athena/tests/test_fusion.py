import tempfile
import unittest
import numpy as np
from athena.fusion import MotionPresenceFusion


class FusionTests(unittest.TestCase):
    def test_calibration_and_state(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as d:
            f = MotionPresenceFusion(d + "/empty.json")
            empty = np.clip(100 + rng.normal(0, 1, (300, 64)), 0, 255)
            f.calibrate_empty(empty)
            same = f.update(np.arange(300), empty)
            self.assertTrue(same["calibrated"])
            self.assertEqual(same["presence"], "empty")
            moving = np.clip(100 + rng.normal(0, 8, (300, 64)), 0, 255)
            result = f.update(np.arange(300), moving)
            self.assertIn(result["motion"], ("moving", "still"))
            self.assertIsNotNone(result["motion_score"])

    def test_requires_enough_frames(self):
        with tempfile.TemporaryDirectory() as d:
            f = MotionPresenceFusion(d + "/empty.json")
            with self.assertRaises(ValueError):
                f.calibrate_empty(np.ones((20, 64), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
