import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "step_plan.py"
)
SPEC = importlib.util.spec_from_file_location("step_plan_module", MODULE_PATH)
STEP_PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP_PLAN)


class TrainingStepPlanTest(unittest.TestCase):
    def test_epochs_define_budget_when_max_steps_is_unset(self):
        plan = STEP_PLAN.resolve_training_step_plan(
            steps_per_epoch=3,
            total_epochs=10,
            total_training_steps=None,
        )
        self.assertEqual(plan.target_outer_updates, 30)
        self.assertEqual(plan.required_epochs, 10)
        self.assertFalse(plan.explicit_step_budget)

    def test_explicit_steps_cycle_dataloader_until_exact_budget(self):
        plan = STEP_PLAN.resolve_training_step_plan(
            steps_per_epoch=1,
            total_epochs=10,
            total_training_steps=20,
        )
        self.assertEqual(plan.target_outer_updates, 20)
        self.assertEqual(plan.required_epochs, 20)
        self.assertTrue(plan.explicit_step_budget)

    def test_partial_last_epoch_is_allowed(self):
        plan = STEP_PLAN.resolve_training_step_plan(
            steps_per_epoch=4,
            total_epochs=2,
            total_training_steps=9,
        )
        self.assertEqual(plan.target_outer_updates, 9)
        self.assertEqual(plan.required_epochs, 3)

    def test_non_positive_budgets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "total_training_steps"):
            STEP_PLAN.resolve_training_step_plan(
                steps_per_epoch=1,
                total_epochs=1,
                total_training_steps=0,
            )
        with self.assertRaisesRegex(ValueError, "total_epochs"):
            STEP_PLAN.resolve_training_step_plan(
                steps_per_epoch=1,
                total_epochs=0,
                total_training_steps=None,
            )


if __name__ == "__main__":
    unittest.main()
