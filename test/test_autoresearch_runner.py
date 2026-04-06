from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "research" / "v2e_imu" / "autoresearch_runner.py"
)
spec = importlib.util.spec_from_file_location("autoresearch_runner", MODULE_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class AutoresearchRunnerTests(unittest.TestCase):
    def test_restore_baseline_uses_best_batch4_defaults(self) -> None:
        restored = runner.restore_baseline(runner.TRAIN_FILE.read_text(encoding="utf-8"))

        self.assertIn("TOTAL_BATCH_SIZE = 4", restored)
        self.assertIn("LEARNING_RATE = 2e-3", restored)
        self.assertIn("WEIGHT_DECAY = 0.0", restored)


if __name__ == "__main__":
    unittest.main()
