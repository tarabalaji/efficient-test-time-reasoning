from __future__ import annotations

import pytest

from src.stopping_policies import (
    get_default_policies,
    get_policy_by_name,
    stop_after_consecutive_agreement,
    stop_at_fixed_stage,
    stop_at_vote_margin,
    stop_at_vote_share,
    stop_below_entropy,
    stop_on_two_answer_agreement,
    stop_when_stable_majority,
    stop_when_unanimous,
    stop_with_combined_confidence,
)


def make_snapshot(
    stage: int = 2,
    unanimous: bool = False,
    top_vote_share: float = 0.5,
    vote_margin: float = 0.0,
    answer_entropy: float = 1.0,
    consecutive_agreement: int = 1,
    majority_changed: bool = False,
) -> dict[str, object]:
    return {
        "stage": stage,
        "unanimous": unanimous,
        "top_vote_share": top_vote_share,
        "vote_margin": vote_margin,
        "answer_entropy": answer_entropy,
        "consecutive_agreement": (consecutive_agreement),
        "majority_changed": majority_changed,
    }


def test_fixed_stage_stops_at_threshold() -> None:
    policy = stop_at_fixed_stage(3)

    assert not policy.should_stop(make_snapshot(stage=2))

    assert policy.should_stop(make_snapshot(stage=3))

    assert policy.should_stop(make_snapshot(stage=4))


def test_unanimous_policy_requires_unanimity() -> None:
    policy = stop_when_unanimous(minimum_stage=2)

    assert not policy.should_stop(
        make_snapshot(
            stage=2,
            unanimous=False,
        )
    )

    assert policy.should_stop(
        make_snapshot(
            stage=2,
            unanimous=True,
        )
    )


def test_vote_share_policy() -> None:
    policy = stop_at_vote_share(
        threshold=0.75,
        minimum_stage=2,
    )

    assert not policy.should_stop(make_snapshot(top_vote_share=0.74))

    assert policy.should_stop(make_snapshot(top_vote_share=0.75))


def test_vote_margin_policy() -> None:
    policy = stop_at_vote_margin(
        threshold=0.5,
        minimum_stage=2,
    )

    assert not policy.should_stop(make_snapshot(vote_margin=0.49))

    assert policy.should_stop(make_snapshot(vote_margin=0.5))


def test_entropy_policy() -> None:
    policy = stop_below_entropy(
        threshold=0.8,
        minimum_stage=2,
    )

    assert policy.should_stop(make_snapshot(answer_entropy=0.8))

    assert not policy.should_stop(make_snapshot(answer_entropy=0.81))


def test_consecutive_agreement_policy() -> None:
    policy = stop_after_consecutive_agreement(
        required_agreement=2,
        minimum_stage=2,
    )

    assert not policy.should_stop(make_snapshot(consecutive_agreement=1))

    assert policy.should_stop(make_snapshot(consecutive_agreement=2))


def test_stable_majority_policy() -> None:
    policy = stop_when_stable_majority(
        minimum_stage=3,
        required_agreement=2,
    )

    assert not policy.should_stop(
        make_snapshot(
            stage=2,
            consecutive_agreement=2,
        )
    )

    assert not policy.should_stop(
        make_snapshot(
            stage=3,
            consecutive_agreement=2,
            majority_changed=True,
        )
    )

    assert policy.should_stop(
        make_snapshot(
            stage=3,
            consecutive_agreement=2,
            majority_changed=False,
        )
    )


def test_two_answer_agreement_policy() -> None:
    policy = stop_on_two_answer_agreement()

    assert policy.should_stop(
        make_snapshot(
            stage=2,
            unanimous=True,
        )
    )

    assert not policy.should_stop(
        make_snapshot(
            stage=2,
            unanimous=False,
        )
    )

    assert policy.should_stop(
        make_snapshot(
            stage=3,
            unanimous=False,
        )
    )


def test_combined_confidence_policy() -> None:
    policy = stop_with_combined_confidence(
        vote_share_threshold=0.75,
        entropy_threshold=0.8,
        required_agreement=2,
        minimum_stage=2,
    )

    assert policy.should_stop(
        make_snapshot(
            top_vote_share=0.75,
            answer_entropy=0.8,
            consecutive_agreement=2,
        )
    )

    assert not policy.should_stop(
        make_snapshot(
            top_vote_share=0.75,
            answer_entropy=0.9,
            consecutive_agreement=2,
        )
    )


def test_default_policy_names_are_unique() -> None:
    policies = get_default_policies(maximum_stage=4)

    names = [policy.name for policy in policies]

    assert len(names) == len(set(names))


def test_get_policy_by_name() -> None:
    policy = get_policy_by_name(
        "two_answer_agreement",
        maximum_stage=4,
    )

    assert policy.name == "two_answer_agreement"


def test_unknown_policy_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown policy",
    ):
        get_policy_by_name(
            "missing_policy",
            maximum_stage=4,
        )


def test_invalid_fixed_stage_raises_error() -> None:
    with pytest.raises(ValueError):
        stop_at_fixed_stage(0)


def test_invalid_vote_share_raises_error() -> None:
    with pytest.raises(ValueError):
        stop_at_vote_share(1.1)


def test_invalid_entropy_raises_error() -> None:
    with pytest.raises(ValueError):
        stop_below_entropy(-0.1)
