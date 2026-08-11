"""Runtime guards for memory-safe, deterministic EITR probe scoring."""

from contextlib import contextmanager
from numbers import Real
from typing import List, Tuple

from torch import nn


_DROPOUT_PROBABILITY_ATTRIBUTES = (
    "attention_dropout",
    "activation_dropout",
    "classifier_dropout",
    "dropout",
    "embd_pdrop",
    "hidden_dropout",
    "hidden_dropout_prob",
    "resid_pdrop",
    "summary_first_dropout",
)


def gradient_checkpointing_enabled(module: nn.Module) -> bool:
    """Return whether any wrapped Hugging Face module has checkpointing enabled."""
    return any(
        bool(getattr(child, "gradient_checkpointing", False))
        for child in module.modules()
    )


def nonzero_dropout_settings(module: nn.Module) -> List[Tuple[str, float]]:
    """Find stochastic dropout that would make train-mode probe scores differ."""
    findings: List[Tuple[str, float]] = []
    seen = set()

    def record(owner_name: str, owner, attribute: str) -> None:
        key = (id(owner), attribute)
        if key in seen:
            return
        seen.add(key)
        value = getattr(owner, attribute, None)
        if isinstance(value, bool) or not isinstance(value, Real):
            return
        probability = float(value)
        if probability > 0.0:
            findings.append((f"{owner_name}.{attribute}", probability))

    for module_name, child in module.named_modules():
        display_name = module_name or "<root>"
        if isinstance(child, nn.modules.dropout._DropoutNd):
            record(display_name, child, "p")
        for attribute in _DROPOUT_PROBABILITY_ATTRIBUTES:
            record(display_name, child, attribute)
        config = getattr(child, "config", None)
        if config is not None:
            for attribute in _DROPOUT_PROBABILITY_ATTRIBUTES:
                record(f"{display_name}.config", config, attribute)
    return findings


def validate_deterministic_probe_checkpointing(module: nn.Module) -> None:
    """Fail early unless train mode is both checkpointed and deterministic."""
    if not gradient_checkpointing_enabled(module):
        raise ValueError(
            "EITR probe gradient checkpointing was requested, but the actor model "
            "does not report gradient_checkpointing=True"
        )
    dropout_settings = nonzero_dropout_settings(module)
    if dropout_settings:
        preview = ", ".join(
            f"{name}={probability:g}"
            for name, probability in dropout_settings[:8]
        )
        raise ValueError(
            "EITR deterministic train-mode probe scoring requires zero dropout; "
            f"found {preview}"
        )


@contextmanager
def deterministic_probe_checkpointing_mode(module: nn.Module):
    """Temporarily enable HF checkpointing without leaking module train state.

    Hugging Face decoder models normally gate layer checkpointing on
    ``module.training``. EITR's Qwen2.5 actor has zero dropout, so train mode is
    deterministic while allowing the already-enabled layer checkpoint wrappers
    to discard activations and recompute them during backward.
    """
    training_states = [(child, bool(child.training)) for child in module.modules()]
    module.train(True)
    try:
        yield
    finally:
        # Restore each nested flag exactly. Calling module.train(previous) would
        # erase any intentionally mixed nested train/eval state.
        for child, was_training in training_states:
            child.training = was_training
