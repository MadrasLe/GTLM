from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gtlm.config import TrainingConfig, load_config
from gtlm.deepspeed_config import build_deepspeed_config


class ConfigTests(unittest.TestCase):
    def test_base_config_loads(self):
        config = load_config(ROOT / "configs" / "train_base.json")
        self.assertIsInstance(config, TrainingConfig)
        self.assertEqual(config.model.top_k, 2)

    def test_deepspeed_batch_size_is_derived(self):
        config = load_config(ROOT / "configs" / "train_base.json")
        ds_config = build_deepspeed_config(config, world_size=2)
        expected = (
            config.deepspeed.train_micro_batch_size_per_gpu
            * config.deepspeed.gradient_accumulation_steps
            * 2
        )
        self.assertEqual(ds_config["train_batch_size"], expected)


if __name__ == "__main__":
    unittest.main()
