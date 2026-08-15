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


ASSISTANT_MARKER_PATTERN = r"<\|im_start\|>assistant\s*"


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


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0,
                     final_format_score=0, retrieval_score=0, format_score=0,
                     score=1., return_details=False):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    search_information_pairs = extract_search_information_pairs(solution_str)
    # Record the raw answer-bearing evidence event independently of whether the
    # final trajectory passes the strict format gate.  Only a format-valid
    # trajectory is eligible to turn this diagnostic event into reward.
    retrieval_correct = is_retrieval_correct(
        solution_str,
        ground_truth['target'],
    )
    answer = extract_solution(solution_str=solution_str)
    answer_correct = bool(
        answer is not None and em_check(answer, ground_truth['target'])
    )
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                reward = structure_format_score + retrieval_score
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
                reward = structure_format_score + retrieval_score
            else:
                reward = structure_format_score
        else:
            reward = final_format_score

    reward = min(float(score), float(reward))
    details = {
        'format_valid': bool(is_valid_format),
        'answer_em': bool(answer_correct),
        'has_executed_search': bool(search_information_pairs),
        'answer_bearing_evidence': bool(retrieval_correct),
        'evidence_bonus_applied': bool(
            is_valid_format
            and retrieval_correct
            and not answer_correct
            and float(retrieval_score) > 0
        ),
    }
    if return_details:
        return float(reward), details
    return float(reward)
