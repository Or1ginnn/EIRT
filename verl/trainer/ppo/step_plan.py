"""Training-step semantics shared by Search-R1 PPO/GRPO runs.

An outer update is one consumed dataloader batch followed by the rollout,
reward/advantage computation, and the configured actor update.  It is not an
individual AdamW ``optimizer.step()``; those are reported separately by the
actor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingStepPlan:
    steps_per_epoch: int
    target_outer_updates: int
    required_epochs: int
    explicit_step_budget: bool


def resolve_training_step_plan(
    *,
    steps_per_epoch: int,
    total_epochs: int,
    total_training_steps=None,
) -> TrainingStepPlan:
    """Resolve an exact outer-update budget.

    ``total_training_steps`` follows the usual max-steps contract: when it is
    set, it is authoritative and the dataloader is re-iterated for as many
    epochs as needed.  Without it, ``total_epochs`` remains authoritative.
    """

    steps_per_epoch = int(steps_per_epoch)
    total_epochs = int(total_epochs)
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if total_epochs <= 0:
        raise ValueError(f"trainer.total_epochs must be positive, got {total_epochs}")

    explicit_step_budget = total_training_steps is not None
    if explicit_step_budget:
        target_outer_updates = int(total_training_steps)
        if target_outer_updates <= 0:
            raise ValueError(
                "trainer.total_training_steps must be positive when set, "
                f"got {target_outer_updates}"
            )
        required_epochs = (
            target_outer_updates + steps_per_epoch - 1
        ) // steps_per_epoch
    else:
        required_epochs = total_epochs
        target_outer_updates = steps_per_epoch * required_epochs

    return TrainingStepPlan(
        steps_per_epoch=steps_per_epoch,
        target_outer_updates=target_outer_updates,
        required_epochs=required_epochs,
        explicit_step_budget=explicit_step_budget,
    )
