import ast
import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_PATH = REPO_ROOT / 'verl/utils/tracking.py'
TRACKING_SPEC = importlib.util.spec_from_file_location('standalone_tracking', TRACKING_PATH)
TRACKING_MODULE = importlib.util.module_from_spec(TRACKING_SPEC)
TRACKING_SPEC.loader.exec_module(TRACKING_MODULE)
Tracking = TRACKING_MODULE.Tracking


class _FakeWandb:
    def __init__(self):
        self.exit_codes = []

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class _FakeMlflow:
    def __init__(self):
        self.exit_codes = []

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class TrackingFinishTest(unittest.TestCase):

    def _tracking_with(self, loggers):
        tracker = Tracking.__new__(Tracking)
        tracker.logger = loggers
        tracker._finished = False
        return tracker

    def test_finish_flushes_wandb_once_with_success_code(self):
        wandb = _FakeWandb()
        tracker = self._tracking_with({'wandb': wandb})

        tracker.finish(exit_code=0)
        tracker.finish(exit_code=1)

        self.assertEqual(wandb.exit_codes, [0])

    def test_finish_propagates_failure_to_all_terminal_backends(self):
        wandb = _FakeWandb()
        mlflow = _FakeMlflow()
        tracker = self._tracking_with({'wandb': wandb, 'mlflow': mlflow})

        tracker.finish(exit_code=1)

        self.assertEqual(wandb.exit_codes, [1])
        self.assertEqual(mlflow.exit_codes, [1])


class TrainingEntrypointTrackingTest(unittest.TestCase):

    def test_both_entrypoints_finish_tracking_in_finally(self):
        for relative_path in ('verl/trainer/main_ppo.py', 'verl/trainer/main_ppo_format.py'):
            module = ast.parse((REPO_ROOT / relative_path).read_text(encoding='utf-8'))
            main_task = next(
                node for node in ast.walk(module)
                if isinstance(node, ast.FunctionDef) and node.name == 'main_task'
            )
            finalizers = [node.finalbody for node in ast.walk(main_task) if isinstance(node, ast.Try)]
            self.assertTrue(
                any(
                    any(
                        isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)
                        and statement.value.func.attr == 'finish'
                        for statement in finalbody
                    )
                    for finalbody in finalizers
                ),
                relative_path,
            )


class OuterStepSemanticsTest(unittest.TestCase):

    def test_final_validation_is_merged_into_terminal_outer_update(self):
        source = (REPO_ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8')
        terminal_validation = source.index('The final validation belongs to the update')
        common_log = source.index('logger.log(data=metrics, step=self.global_steps)', terminal_validation)
        terminal_return = source.index('if self.global_steps >= self.total_training_steps:', common_log)
        self.assertLess(terminal_validation, common_log)
        self.assertLess(common_log, terminal_return)
        self.assertIn("'trainer/outer_update_step': float(self.global_steps)", source)


if __name__ == '__main__':
    unittest.main()
