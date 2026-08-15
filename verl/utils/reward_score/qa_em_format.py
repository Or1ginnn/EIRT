# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import unicodedata
import math


ASSISTANT_MARKER_PATTERN = r"<\|im_start\|>assistant\s*"
SUPPORTED_REWARD_PROFILES = {
    "pure_em",
    "official_v03",
    "evidence_shaping",
    "mandatory_search",
}


def reward_group_diagnostics(scores, group_ids, tolerance=1e-12):
    """Summarize whether GRPO sibling groups contain usable reward contrast."""

    scores = [float(value) for value in scores]
    group_ids = list(group_ids)
    if len(scores) != len(group_ids):
        raise ValueError(
            "reward scores and group ids must have equal length: "
            f"{len(scores)} != {len(group_ids)}"
        )
    if not scores:
        return {
            'nonzero_rate': 0.0,
            'score_std': 0.0,
            'unique_level_count': 0.0,
            'nonzero_advantage_group_rate': 0.0,
            'zero_advantage_group_rate': 0.0,
        }

    groups = {}
    for score_value, group_id in zip(scores, group_ids):
        groups.setdefault(str(group_id), []).append(score_value)
    varying_groups = sum(
        max(group_scores) - min(group_scores) > float(tolerance)
        for group_scores in groups.values()
    )
    mean_score = sum(scores) / len(scores)
    variance = sum((value - mean_score) ** 2 for value in scores) / len(scores)
    nonzero_advantage_rate = varying_groups / len(groups)
    return {
        'nonzero_rate': sum(value > 0.0 for value in scores) / len(scores),
        'score_std': math.sqrt(max(variance, 0.0)),
        'unique_level_count': float(len(set(scores))),
        'nonzero_advantage_group_rate': nonzero_advantage_rate,
        'zero_advantage_group_rate': 1.0 - nonzero_advantage_rate,
    }


def extract_assistant_content(text):
    """Return only model-generated content after the first assistant marker.

    Search-R1 prompts contain literal ``<answer>...</answer>`` examples.  Those
    examples are instructions, not model actions, and must never be eligible
    for answer extraction or format reward.
    """
    matches = list(re.finditer(ASSISTANT_MARKER_PATTERN, text))
    if not matches:
        return text
    # RewardManager supplies one synthetic marker before the generated
    # trajectory.  Always anchor on that first marker: accepting a later marker
    # would let the model hide malformed output by emitting a second assistant
    # turn itself.
    return text[matches[0].end():]

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the one assistant turn and validate generated content only.
    assistant_matches = list(re.finditer(ASSISTANT_MARKER_PATTERN, text))
    
    if not assistant_matches:
        return False, "Missing assistant marker"
    if len(assistant_matches) != 1:
        return False, "Unexpected assistant marker inside generated trajectory"
    
    # Extract the content after the assistant marker
    start_pos = assistant_matches[0].end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Closed but empty actions are not valid environment actions and must not
    # receive structure reward either.
    for tag in ("search", "answer"):
        for block in re.findall(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL):
            if not block.strip():
                return False, f"Empty {tag} block"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the final answer from model-generated content only."""

    answer_pattern = r'<answer>(.*?)</answer>'
    assistant_content = extract_assistant_content(solution_str)
    match = re.finditer(answer_pattern, assistant_content, re.DOTALL)
    matches = list(match)
    
    if not matches:
        return None
    
    # A trajectory should contain one final answer.  Taking the last complete
    # generated block is robust to an earlier malformed/retried answer action.
    answer = matches[-1].group(1).strip()
    return answer or None


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, extract_assistant_content(text), re.DOTALL)
    return [match.strip() for match in matches]


def extract_environment_information_blocks(text: str) -> list[str]:
    """Extract observations without applying model-turn marker semantics."""

    pattern = r"<information>(.*?)</information>"
    return [
        match.strip()
        for match in re.findall(pattern, text or "", re.DOTALL)
    ]


def decode_environment_observation(
    tokenizer,
    valid_response_ids,
    response_info_mask,
) -> str:
    """Decode only environment-owned response tokens selected by info_mask."""

    if len(valid_response_ids) != len(response_info_mask):
        raise ValueError(
            'response ids and response info mask must have equal length: '
            f'{len(valid_response_ids)} != {len(response_info_mask)}'
        )
    if hasattr(response_info_mask, 'eq'):
        environment_ids = valid_response_ids[response_info_mask.eq(0)]
    else:
        environment_ids = [
            token_id
            for token_id, mask_value in zip(
                valid_response_ids,
                response_info_mask,
            )
            if int(mask_value) == 0
        ]
    return tokenizer.decode(environment_ids)


def decode_model_generated_response(
    tokenizer,
    valid_response_ids,
    response_info_mask,
) -> str:
    """Decode only model-owned response tokens selected by ``info_mask``."""

    if len(valid_response_ids) != len(response_info_mask):
        raise ValueError(
            'response ids and response info mask must have equal length: '
            f'{len(valid_response_ids)} != {len(response_info_mask)}'
        )
    if hasattr(response_info_mask, 'ne'):
        model_ids = valid_response_ids[response_info_mask.ne(0)]
    else:
        model_ids = [
            token_id
            for token_id, mask_value in zip(
                valid_response_ids,
                response_info_mask,
            )
            if int(mask_value) != 0
        ]
    return tokenizer.decode(model_ids)


def extract_search_information_pairs(text: str) -> list[tuple[str, str]]:
    """Return non-empty search/environment-observation pairs.

    The bonus is trajectory-level and must be tied to an actual tool-shaped
    interaction.  A free-standing ``<information>`` block is not evidence that
    the model issued a usable search action.
    """

    pattern = (
        r"<search>(.*?)</search>\s*"
        r"<information>(.*?)</information>"
    )
    pairs = re.findall(pattern, extract_assistant_content(text), re.DOTALL)
    return [
        (query.strip(), information.strip())
        for query, information in pairs
        if query.strip() and information.strip()
    ]


def generated_format_components(text: str) -> tuple[bool, bool]:
    """Return independently useful think/answer format indicators.

    The complete Search-R1 protocol is still checked by ``is_valid_sequence``.
    These two indicators provide bounded shaping when a trajectory has a real
    tool execution but has not yet learned the whole protocol.
    """

    assistant_matches = list(re.finditer(ASSISTANT_MARKER_PATTERN, text))
    if len(assistant_matches) != 1:
        return False, False
    content = text[assistant_matches[0].end():]

    think_open = list(re.finditer(r"<think>", content))
    think_close = list(re.finditer(r"</think>", content))
    think_blocks = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    think_valid = bool(
        think_open
        and len(think_open) == len(think_close) == len(think_blocks)
        and all(block.strip() for block in think_blocks)
        and content.lstrip().startswith("<think>")
    )

    answer_open = list(re.finditer(r"<answer>", content))
    answer_close = list(re.finditer(r"</answer>", content))
    answer_blocks = list(
        re.finditer(r"<answer>(.*?)</answer>", content, re.DOTALL)
    )
    answer_valid = bool(
        len(answer_open) == len(answer_close) == len(answer_blocks) == 1
        and answer_blocks[0].group(1).strip()
        and not content[answer_blocks[0].end():].strip()
    )
    return think_valid, answer_valid


def _contains_normalized_answer(information: str, golden_answer: str) -> bool:
    """Match a normalized gold alias as a complete token sequence."""

    def normalize_evidence_text(text: str) -> str:
        # Convert both ASCII and Unicode punctuation into token boundaries
        # before applying the v0.3 answer normalization.  Deleting punctuation
        # would turn ``Paris—the`` into one token and miss a genuine hit.
        separated = ''.join(
            ' ' if unicodedata.category(char).startswith('P') else char
            for char in text
        )
        return normalize_answer(separated)

    normalized_information = normalize_evidence_text(information)
    normalized_answer = normalize_evidence_text(golden_answer)
    if not normalized_answer:
        return False
    return re.search(
        rf"(?<!\w){re.escape(normalized_answer)}(?!\w)",
        normalized_information,
    ) is not None


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    """Whether any executed-search observation contains a gold alias.

    This is an answer-bearing-evidence heuristic, not a claim that the whole
    retrieved passage is relevant.  Multiple hits still produce one Boolean
    trajectory-level event and therefore cannot reward repeated searches.
    """

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    for _, information in extract_search_information_pairs(text):
        for golden_answer in golden_answers:
            if _contains_normalized_answer(information, golden_answer):
                return True
    return False


def observations_contain_answer(
    observation_text: str,
    golden_answers: list[str],
) -> bool:
    """Check only environment-owned information blocks for a gold alias."""

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    for information in extract_environment_information_blocks(observation_text):
        if not information:
            continue
        for golden_answer in golden_answers:
            if _contains_normalized_answer(information, golden_answer):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0,
                     final_format_score=0, retrieval_score=0, format_score=0,
                     score=1., return_details=False,
                     reward_profile='pure_em', executed_search_count=None,
                     environment_observation_str=None,
                     model_generated_str=None,
                     think_format_score=0.05, answer_format_score=0.05,
                     evidence_score=0.2, answer_em_score=0.7,
                     joint_success_bonus=0.5):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if reward_profile not in SUPPORTED_REWARD_PROFILES:
        raise ValueError(
            f"Unknown reward_profile={reward_profile!r}; expected one of "
            f"{sorted(SUPPORTED_REWARD_PROFILES)}"
        )
    if reward_profile == 'mandatory_search':
        component_scores = {
            'think_format_score': think_format_score,
            'answer_format_score': answer_format_score,
            'evidence_score': evidence_score,
            'answer_em_score': answer_em_score,
            'joint_success_bonus': joint_success_bonus,
        }
        invalid_scores = {
            name: value
            for name, value in component_scores.items()
            if not math.isfinite(float(value)) or float(value) < 0
        }
        if invalid_scores:
            raise ValueError(
                'mandatory_search reward components must be finite and '
                f'non-negative, got {invalid_scores}'
            )
    if reward_profile == 'official_v03':
        exact_v03 = {
            'structure_format_score': (float(structure_format_score), 0.2),
            'final_format_score': (float(final_format_score), 0.1),
            'retrieval_score': (float(retrieval_score), 0.0),
            'score': (float(score), 1.0),
        }
        mismatched = {
            name: actual
            for name, (actual, expected) in exact_v03.items()
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        }
        if mismatched:
            raise ValueError(
                'official_v03 requires exact Search-R1 3B reward weights '
                f'(structure=0.2, final=0.1, retrieval=0, score=1); got '
                f'{mismatched}'
            )
    if reward_profile == 'pure_em':
        unused_scores = {
            'structure_format_score': structure_format_score,
            'final_format_score': final_format_score,
            'retrieval_score': retrieval_score,
        }
        nonzero_scores = {
            name: float(value)
            for name, value in unused_scores.items()
            if not math.isclose(
                float(value),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        if nonzero_scores:
            raise ValueError(
                'pure_em ignores format/evidence shaping and therefore '
                f'requires their scores to be zero, got {nonzero_scores}'
            )

    # The combined trajectory contains both model tokens and environment
    # observations.  Keep it for the end-to-end protocol check, but never use
    # it to award model-owned think/answer/search components: retrieved text
    # can itself contain literal action tags.
    if reward_profile == 'mandatory_search':
        model_solution_str = (
            "<|im_start|>assistant\n" + (model_generated_str or "")
        )
    else:
        model_solution_str = solution_str

    is_valid_format, _ = is_valid_sequence(solution_str)
    think_format_valid, answer_format_valid = generated_format_components(
        model_solution_str
    )
    search_information_pairs = extract_search_information_pairs(solution_str)
    # Record the legacy parsed-evidence event independently of the strict
    # format gate.  Mandatory-search scoring replaces it below with trusted
    # environment-owned evidence.
    parsed_retrieval_correct = is_retrieval_correct(
        solution_str,
        ground_truth['target'],
    )
    answer = extract_solution(solution_str=model_solution_str)
    answer_correct = bool(
        answer is not None and em_check(answer, ground_truth['target'])
    )
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
            
    parsed_search_count = len(search_information_pairs)
    if executed_search_count is None:
        actual_search_count = parsed_search_count
        execution_signal_available = False
    else:
        actual_search_count = int(executed_search_count)
        execution_signal_available = True
    if actual_search_count < 0:
        raise ValueError("executed_search_count must be non-negative")
    trusted_information_blocks = (
        extract_environment_information_blocks(environment_observation_str)
        if environment_observation_str is not None
        else []
    )
    assistant_content = extract_assistant_content(model_solution_str)
    nonempty_search_count = sum(
        bool(block.strip())
        for block in re.findall(
            r"<search>(.*?)</search>",
            assistant_content,
            re.DOTALL,
        )
    )
    tool_trace_consistent = bool(
        actual_search_count > 0
        and environment_observation_str is not None
        and len(trusted_information_blocks) == actual_search_count
        and nonempty_search_count == actual_search_count
    )
    generated_information_detected = bool(
        model_generated_str is not None
        and re.search(
            r"<\s*/?\s*information\b[^>]*>",
            model_generated_str,
            flags=re.IGNORECASE,
        )
    )
    model_ownership_signal_available = model_generated_str is not None
    trusted_retrieval_correct = bool(
        tool_trace_consistent
        and observations_contain_answer(
            environment_observation_str,
            ground_truth['target'],
        )
    )
    retrieval_correct = (
        trusted_retrieval_correct
        if reward_profile == 'mandatory_search'
        else parsed_retrieval_correct
    )

    if reward_profile == 'mandatory_search':
        strict_protocol_valid = bool(
            is_valid_format
            and think_format_valid
            and answer_format_valid
        )
        hard_reward_gate_pass = bool(
            execution_signal_available
            and tool_trace_consistent
            and model_ownership_signal_available
            and not generated_information_detected
        )
        eligible_answer_correct = bool(answer_correct)
        eligible_evidence = bool(
            retrieval_correct and tool_trace_consistent
        )
        full_success = bool(
            strict_protocol_valid
            and hard_reward_gate_pass
            and eligible_evidence
            and eligible_answer_correct
        )
        if not hard_reward_gate_pass:
            reward = 0.0
        else:
            reward = (
                float(think_format_score) * float(think_format_valid)
                + float(answer_format_score) * float(answer_format_valid)
                + float(evidence_score) * float(eligible_evidence)
                + float(answer_em_score) * float(eligible_answer_correct)
                + float(joint_success_bonus) * float(full_success)
            )
    elif reward_profile == 'pure_em':
        reward = float(score) if answer_correct else 0.0
    else:
        if reward_profile == 'official_v03' and float(retrieval_score) != 0.0:
            raise ValueError(
                "official_v03 requires retrieval_score=0; use "
                "reward_profile='evidence_shaping' for the legacy hook"
            )
        active_retrieval_score = (
            float(retrieval_score)
            if reward_profile == 'evidence_shaping'
            else 0.0
        )
        if answer is None:
            if is_valid_format:
                if retrieval_correct:
                    reward = structure_format_score + active_retrieval_score
                else:
                    reward = structure_format_score
            else:
                reward = 0
        else:
            if answer_correct:
                if is_valid_format:
                    reward = score
                else:
                    reward = score - structure_format_score
            elif is_valid_format:
                if retrieval_correct:
                    reward = structure_format_score + active_retrieval_score
                else:
                    reward = structure_format_score
            else:
                reward = final_format_score

        reward = min(float(score), float(reward))
    details = {
        'reward_profile_mandatory_search': bool(
            reward_profile == 'mandatory_search'
        ),
        'format_valid': bool(is_valid_format),
        'think_format_valid': bool(think_format_valid),
        'answer_format_valid': bool(answer_format_valid),
        'answer_em': bool(answer_correct),
        'has_executed_search': bool(actual_search_count > 0),
        'execution_signal_available': bool(execution_signal_available),
        'tool_trace_consistent': bool(tool_trace_consistent),
        'model_ownership_signal_available': bool(
            model_ownership_signal_available
        ),
        'generated_information_detected': bool(
            generated_information_detected
        ),
        'hard_reward_gate_pass': bool(
            reward_profile == 'mandatory_search'
            and hard_reward_gate_pass
        ),
        'answer_bearing_evidence': bool(retrieval_correct),
        'trusted_environment_evidence': bool(trusted_retrieval_correct),
        'evidence_bonus_applied': bool(
            (
                reward_profile == 'mandatory_search'
                and hard_reward_gate_pass
                and retrieval_correct
            )
            or (
                reward_profile == 'evidence_shaping'
                and is_valid_format
                and retrieval_correct
                and not answer_correct
                and float(retrieval_score) > 0
            )
        ),
        'joint_success_bonus_applied': bool(
            reward_profile == 'mandatory_search'
            and hard_reward_gate_pass
            and strict_protocol_valid
            and retrieval_correct
            and answer_correct
        ),
    }
    if return_details:
        return float(reward), details
    return float(reward)
