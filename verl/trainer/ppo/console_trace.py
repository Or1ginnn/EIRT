"""Small, bounded terminal traces for inspecting real training trajectories."""

import re
from typing import Any, Optional


def _decode(tokenizer: Any, token_ids: Any) -> str:
    ids = token_ids.detach().cpu().tolist() if hasattr(token_ids, "detach") else list(token_ids)
    return tokenizer.decode(ids, skip_special_tokens=False)


def _last_closed_answer(trajectory: str) -> Optional[str]:
    matches = re.findall(r"<answer>(.*?)</answer>", trajectory, re.DOTALL)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def print_training_trace_samples(
    *,
    batch: Any,
    tokenizer: Any,
    outer_update_step: int,
    sample_count: int,
) -> None:
    """Print a few prompt/trajectory/reward tuples without changing scoring.

    This runs on the driver after the rule reward has been computed.  It reads
    the already materialized training batch and never calls the reward
    function, retriever, or model.
    """

    sample_count = max(0, min(int(sample_count), len(batch)))
    if sample_count == 0:
        return

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"].bool()
    scores = batch.batch["token_level_scores"].sum(dim=-1)
    prompt_width = int(prompts.shape[-1])
    response_width = int(responses.shape[-1])

    data_sources = batch.non_tensor_batch.get("data_source")
    reward_models = batch.non_tensor_batch.get("reward_model")

    for index in range(sample_count):
        prompt_mask = attention_mask[index, :prompt_width]
        response_mask = attention_mask[
            index,
            prompt_width : prompt_width + response_width,
        ]
        prompt = _decode(tokenizer, prompts[index][prompt_mask])
        trajectory = _decode(tokenizer, responses[index][response_mask])
        extracted_answer = _last_closed_answer(trajectory)
        data_source = data_sources[index] if data_sources is not None else "unknown"
        ground_truth = reward_models[index] if reward_models is not None else "unknown"
        score = float(scores[index].detach().cpu().item())

        print(
            f"\n===== TRAIN TRACE outer_update={int(outer_update_step)} "
            f"sample={index} data_source={data_source} reward={score:.6f} =====",
            flush=True,
        )
        print("[PROMPT]", flush=True)
        print(prompt, flush=True)
        print("[TRAJECTORY: model output + environment information]", flush=True)
        print(trajectory, flush=True)
        print("[EXTRACTED ANSWER]", flush=True)
        print(extracted_answer if extracted_answer is not None else "<NONE>", flush=True)
        print("[GROUND TRUTH]", flush=True)
        print(ground_truth, flush=True)
        print("===== END TRAIN TRACE =====\n", flush=True)
