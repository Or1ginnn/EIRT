import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "console_trace.py"
)
SPEC = importlib.util.spec_from_file_location("console_trace_module", MODULE_PATH)
CONSOLE_TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSOLE_TRACE)


class _Tokenizer:
    TOKENS = {
        1: "<PROMPT>",
        2: "question",
        3: "<think>x</think>",
        4: "<search>q</search>",
        5: "<information>doc</information>",
        6: "<answer>Paris</answer>",
        99: "<PAD>",
    }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.TOKENS[token_id] for token_id in ids)


class _Batch:
    def __init__(self):
        self.batch = {
            "prompts": torch.tensor([[99, 1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 6, 99]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]]),
            "token_level_scores": torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]),
        }
        self.non_tensor_batch = {
            "data_source": np.array(["nq"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": {"target": ["Paris"]}}], dtype=object
            ),
        }

    def __len__(self):
        return 1


class ConsoleTraceTest(unittest.TestCase):
    def test_prints_masked_prompt_full_trajectory_answer_and_reward(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            CONSOLE_TRACE.print_training_trace_samples(
                batch=_Batch(),
                tokenizer=_Tokenizer(),
                outer_update_step=7,
                sample_count=1,
            )

        text = output.getvalue()
        self.assertIn("outer_update=7", text)
        self.assertIn("data_source=nq reward=1.000000", text)
        self.assertIn("<PROMPT>question", text)
        self.assertNotIn("<PAD>", text)
        self.assertIn("<information>doc</information>", text)
        self.assertIn("[EXTRACTED ANSWER]\nParis", text)
        self.assertIn("'target': ['Paris']", text)

    def test_zero_samples_is_silent(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            CONSOLE_TRACE.print_training_trace_samples(
                batch=_Batch(),
                tokenizer=_Tokenizer(),
                outer_update_step=1,
                sample_count=0,
            )
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
