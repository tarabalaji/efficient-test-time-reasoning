from __future__ import annotations

import pytest

from src.extract_features import (
    calculate_answer_entropy,
    calculate_consecutive_agreement,
    calculate_majority_history,
    count_majority_changes,
    extract_snapshot_features,
    get_majority_answer,
    get_valid_answers,
)


def make_response(
    attempt: int,
    answer: str | None,
    output_tokens: int | None = 100,
) -> dict:
    return {
        "attempt": attempt,
        "extracted_answer": answer,
        "output_tokens": output_tokens,
    }


def test_get_valid_answers() -> None:
    responses = [
        make_response(1, "10"),
        make_response(2, None),
        make_response(3, "12"),
    ]

    assert get_valid_answers(responses) == ["10", "12"]


def test_get_majority_answer_clear_majority() -> None:
    answers = ["10", "10", "12"]

    assert get_majority_answer(answers) == "10"


def test_get_majority_answer_empty() -> None:
    assert get_majority_answer([]) is None


def test_get_majority_answer_tie_uses_latest_answer() -> None:
    answers = ["10", "12"]

    assert get_majority_answer(answers) == "12"


def test_get_majority_answer_larger_tie_uses_latest() -> None:
    answers = ["10", "12", "10", "12"]

    assert get_majority_answer(answers) == "12"


def test_calculate_entropy_unanimous() -> None:
    entropy = calculate_answer_entropy(["10", "10", "10"])

    assert entropy == pytest.approx(0.0)


def test_calculate_entropy_even_split() -> None:
    entropy = calculate_answer_entropy(["10", "12"])

    assert entropy == pytest.approx(1.0)


def test_calculate_entropy_empty() -> None:
    assert calculate_answer_entropy([]) == 0.0


def test_consecutive_agreement() -> None:
    answers = ["10", "12", "12", "12"]

    assert calculate_consecutive_agreement(answers) == 3


def test_consecutive_agreement_single() -> None:
    assert calculate_consecutive_agreement(["10"]) == 1


def test_consecutive_agreement_latest_missing() -> None:
    assert calculate_consecutive_agreement(["10", None]) == 0


def test_consecutive_agreement_empty() -> None:
    assert calculate_consecutive_agreement([]) == 0


def test_calculate_majority_history() -> None:
    answers = ["10", "12", "10", "12", "12"]

    history = calculate_majority_history(answers)

    assert history == ["10", "12", "10", "12", "12"]


def test_calculate_majority_history_with_missing_answer() -> None:
    answers = ["10", None, "10"]

    history = calculate_majority_history(answers)

    assert history == ["10", "10", "10"]


def test_count_majority_changes() -> None:
    history = ["10", "12", "10", "10"]

    assert count_majority_changes(history) == 2


def test_count_majority_changes_with_none() -> None:
    history = [None, "10", "10"]

    assert count_majority_changes(history) == 0


def test_extract_features_unanimous() -> None:
    responses = [
        make_response(1, "24", 100),
        make_response(2, "24", 120),
        make_response(3, "24", 140),
    ]

    features = extract_snapshot_features(responses)

    assert features["num_attempts"] == 3
    assert features["num_valid_answers"] == 3
    assert features["num_unique_answers"] == 1
    assert features["missing_answer_count"] == 0
    assert features["top_vote_share"] == pytest.approx(1.0)
    assert features["second_vote_share"] == pytest.approx(0.0)
    assert features["vote_margin"] == pytest.approx(1.0)
    assert features["answer_entropy"] == pytest.approx(0.0)
    assert features["latest_matches_majority"] is True
    assert features["unanimous"] is True
    assert features["majority_changed"] is False
    assert features["majority_change_count"] == 0
    assert features["consecutive_agreement"] == 3
    assert features["average_output_tokens"] == pytest.approx(120.0)
    assert features["total_output_tokens"] == 360


def test_extract_features_split_vote() -> None:
    responses = [
        make_response(1, "10"),
        make_response(2, "12"),
        make_response(3, "10"),
    ]

    features = extract_snapshot_features(responses)

    assert features["num_unique_answers"] == 2
    assert features["top_vote_share"] == pytest.approx(2 / 3)
    assert features["second_vote_share"] == pytest.approx(1 / 3)
    assert features["vote_margin"] == pytest.approx(1 / 3)
    assert features["latest_matches_majority"] is True
    assert features["unanimous"] is False
    assert features["majority_changed"] is True
    assert features["majority_change_count"] == 2
    assert features["consecutive_agreement"] == 1


def test_extract_features_with_missing_answer() -> None:
    responses = [
        make_response(1, "10"),
        make_response(2, None),
        make_response(3, "10"),
    ]

    features = extract_snapshot_features(responses)

    assert features["num_attempts"] == 3
    assert features["num_valid_answers"] == 2
    assert features["num_unique_answers"] == 1
    assert features["missing_answer_count"] == 1
    assert features["top_vote_share"] == pytest.approx(1.0)
    assert features["unanimous"] is False
    assert features["latest_matches_majority"] is True
    assert features["consecutive_agreement"] == 1


def test_extract_features_all_missing() -> None:
    responses = [
        make_response(1, None, None),
        make_response(2, None, None),
    ]

    features = extract_snapshot_features(responses)

    assert features["num_attempts"] == 2
    assert features["num_valid_answers"] == 0
    assert features["num_unique_answers"] == 0
    assert features["missing_answer_count"] == 2
    assert features["top_vote_share"] == 0.0
    assert features["second_vote_share"] == 0.0
    assert features["vote_margin"] == 0.0
    assert features["answer_entropy"] == 0.0
    assert features["latest_matches_majority"] is False
    assert features["unanimous"] is False
    assert features["consecutive_agreement"] == 0
    assert features["average_output_tokens"] == 0.0
    assert features["total_output_tokens"] == 0


def test_extract_features_sorts_attempts() -> None:
    responses = [
        make_response(3, "12"),
        make_response(1, "10"),
        make_response(2, "12"),
    ]

    features = extract_snapshot_features(responses)

    assert features["top_vote_share"] == pytest.approx(2 / 3)
    assert features["latest_matches_majority"] is True
    assert features["consecutive_agreement"] == 2


def test_extract_features_requires_responses() -> None:
    with pytest.raises(
        ValueError,
        match="At least one response is required",
    ):
        extract_snapshot_features([])
