import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "utils"
    / "logger"
    / "aggregate_logger.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_logger_module", MODULE_PATH)
AGGREGATE_LOGGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE_LOGGER)


class AggregateLoggerTest(unittest.TestCase):
    def test_eitr_drift_uses_high_precision_scientific_notation(self):
        output = AGGREGATE_LOGGER.concat_dict_to_str(
            {
                "actor/eitr_env_drift_pre": 1.234567890123e-6,
                "actor/eitr_env_drift_delta": -1.25e-11,
                "actor/pg_loss": 0.123456,
            },
            step=1,
        )
        self.assertIn("actor/eitr_env_drift_pre:1.2345678901e-06", output)
        self.assertIn("actor/eitr_env_drift_delta:-1.2500000000e-11", output)
        self.assertIn("actor/pg_loss:0.123", output)


if __name__ == "__main__":
    unittest.main()
