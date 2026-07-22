from __future__ import annotations

import math
from collections import Counter
from typing import Any


def get_valid_answers(
    responses: list[dict[str, Any]],
) -> list[str]:
    return [
        str(response["extracted_answer"])
        for response in responses
        if response.get("extracted_answer") is not None
    ]


def get_majority_answer(
    answers: list[str],
) -> str | None:
    if not answers:
        return None

    counts = Counter(answers)
    highest_count = max(counts.values())

    tied_answers = {
        answer for answer, count in counts.items() if count == highest_count
    }

    for answer in reversed(answers):
        if answer in tied_answers:
            return answer

    return None


def calculate_answer_entropy(
    answers: list[str],
) -> float:
    if not answers:
        return 0.0

    counts = Counter(answers)
    total = len(answers)

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_consecutive_agreement(
    answers: list[str | None],
) -> int:
    if not answers or answers[-1] is None:
        return 0

    latest_answer = answers[-1]
    agreement_count = 0

    for answer in reversed(answers):
        if answer == latest_answer:
            agreement_count += 1
        else:
            break

    return agreement_count


def calculate_majority_history(
    answers: list[str | None],
) -> list[str | None]:
    history: list[str | None] = []

    for stage in range(1, len(answers) + 1):
        valid_answers = [answer for answer in answers[:stage] if answer is not None]

        history.append(get_majority_answer(valid_answers))

    return history


def count_majority_changes(
    majority_history: list[str | None],
) -> int:
    change_count = 0

    for previous, current in zip(
        majority_history,
        majority_history[1:],
    ):
        if previous is not None and current is not None and previous != current:
            change_count += 1

    return change_count


def extract_snapshot_features(
    responses: list[dict[str, Any]],
) -> dict[str, int | float | bool]:
    if not responses:
        raise ValueError("At least one response is required")

    sorted_responses = sorted(
        responses,
        key=lambda response: int(response["attempt"]),
    )

    all_answers: list[str | None] = [
        response.get("extracted_answer") for response in sorted_responses
    ]

    valid_answers = [str(answer) for answer in all_answers if answer is not None]

    num_attempts = len(sorted_responses)
    num_valid_answers = len(valid_answers)
    missing_answer_count = num_attempts - num_valid_answers

    answer_counts = Counter(valid_answers)

    sorted_counts = sorted(
        answer_counts.values(),
        reverse=True,
    )

    top_count = sorted_counts[0] if sorted_counts else 0
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0

    top_vote_share = top_count / num_valid_answers if num_valid_answers > 0 else 0.0

    second_vote_share = (
        second_count / num_valid_answers if num_valid_answers > 0 else 0.0
    )

    vote_margin = top_vote_share - second_vote_share

    majority_answer = get_majority_answer(valid_answers)
    latest_answer = all_answers[-1]

    majority_history = calculate_majority_history(all_answers)

    previous_majority = majority_history[-2] if len(majority_history) >= 2 else None

    current_majority = majority_history[-1]

    majority_changed = (
        previous_majority is not None
        and current_majority is not None
        and previous_majority != current_majority
    )

    output_tokens = [
        int(response["output_tokens"])
        for response in sorted_responses
        if response.get("output_tokens") is not None
    ]

    average_output_tokens = (
        sum(output_tokens) / len(output_tokens) if output_tokens else 0.0
    )

    total_output_tokens = sum(output_tokens)

    return {
        "num_attempts": num_attempts,
        "num_valid_answers": num_valid_answers,
        "num_unique_answers": len(answer_counts),
        "missing_answer_count": missing_answer_count,
        "top_vote_share": top_vote_share,
        "second_vote_share": second_vote_share,
        "vote_margin": vote_margin,
        "answer_entropy": calculate_answer_entropy(valid_answers),
        "latest_matches_majority": (
            latest_answer is not None and latest_answer == majority_answer
        ),
        "unanimous": (
            num_valid_answers > 0
            and len(answer_counts) == 1
            and missing_answer_count == 0
        ),
        "majority_changed": majority_changed,
        "majority_change_count": count_majority_changes(majority_history),
        "consecutive_agreement": (calculate_consecutive_agreement(all_answers)),
        "average_output_tokens": average_output_tokens,
        "total_output_tokens": total_output_tokens,
    }
