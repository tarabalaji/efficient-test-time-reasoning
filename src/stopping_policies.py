from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

PolicyFunction = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class StoppingPolicy:
    name: str
    should_stop: PolicyFunction


def get_value(
    snapshot: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if hasattr(snapshot, "get"):
        return snapshot.get(key, default)

    try:
        return snapshot[key]
    except (KeyError, TypeError):
        return default


def get_float(
    snapshot: Mapping[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    value = get_value(snapshot, key, default)

    if value is None:
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    if np.isnan(converted):
        return default

    return converted


def get_int(
    snapshot: Mapping[str, Any],
    key: str,
    default: int = 0,
) -> int:
    value = get_value(snapshot, key, default)

    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_bool(
    snapshot: Mapping[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    value = get_value(snapshot, key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no"}:
            return False

    return default


def stop_at_fixed_stage(
    stage: int,
) -> StoppingPolicy:
    if stage < 1:
        raise ValueError("Fixed stopping stage must be at least 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        current_stage = get_int(
            snapshot,
            "stage",
        )

        return current_stage >= stage

    return StoppingPolicy(
        name=f"fixed_{stage}",
        should_stop=should_stop,
    )


def stop_when_unanimous(
    minimum_stage: int = 2,
) -> StoppingPolicy:
    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        return get_bool(
            snapshot,
            "unanimous",
        )

    return StoppingPolicy(
        name=f"unanimous_min_{minimum_stage}",
        should_stop=should_stop,
    )


def stop_at_vote_share(
    threshold: float,
    minimum_stage: int = 2,
) -> StoppingPolicy:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Vote-share threshold must be between 0 and 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        top_vote_share = get_float(
            snapshot,
            "top_vote_share",
        )

        return top_vote_share >= threshold

    threshold_name = str(threshold).replace(".", "_")

    return StoppingPolicy(
        name=(f"vote_share_{threshold_name}_min_{minimum_stage}"),
        should_stop=should_stop,
    )


def stop_at_vote_margin(
    threshold: float,
    minimum_stage: int = 2,
) -> StoppingPolicy:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Vote-margin threshold must be between 0 and 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        vote_margin = get_float(
            snapshot,
            "vote_margin",
        )

        return vote_margin >= threshold

    threshold_name = str(threshold).replace(".", "_")

    return StoppingPolicy(
        name=(f"vote_margin_{threshold_name}_min_{minimum_stage}"),
        should_stop=should_stop,
    )


def stop_below_entropy(
    threshold: float,
    minimum_stage: int = 2,
) -> StoppingPolicy:
    if threshold < 0.0:
        raise ValueError("Entropy threshold cannot be negative")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        answer_entropy = get_float(
            snapshot,
            "answer_entropy",
        )

        return answer_entropy <= threshold

    threshold_name = str(threshold).replace(".", "_")

    return StoppingPolicy(
        name=(f"entropy_{threshold_name}_min_{minimum_stage}"),
        should_stop=should_stop,
    )


def stop_after_consecutive_agreement(
    required_agreement: int,
    minimum_stage: int = 2,
) -> StoppingPolicy:
    if required_agreement < 1:
        raise ValueError("Required agreement must be at least 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        consecutive_agreement = get_int(
            snapshot,
            "consecutive_agreement",
        )

        return consecutive_agreement >= required_agreement

    return StoppingPolicy(
        name=(f"consecutive_{required_agreement}_min_{minimum_stage}"),
        should_stop=should_stop,
    )


def stop_when_stable_majority(
    minimum_stage: int = 3,
    required_agreement: int = 2,
) -> StoppingPolicy:
    if minimum_stage < 2:
        raise ValueError("Stable-majority minimum stage must be at least 2")

    if required_agreement < 1:
        raise ValueError("Required agreement must be at least 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        majority_changed = get_bool(
            snapshot,
            "majority_changed",
        )

        consecutive_agreement = get_int(
            snapshot,
            "consecutive_agreement",
        )

        return not majority_changed and consecutive_agreement >= required_agreement

    return StoppingPolicy(
        name=(f"stable_majority_{required_agreement}_min_{minimum_stage}"),
        should_stop=should_stop,
    )


def stop_on_two_answer_agreement() -> StoppingPolicy:
    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < 2:
            return False

        if stage > 2:
            return True

        return get_bool(
            snapshot,
            "unanimous",
        )

    return StoppingPolicy(
        name="two_answer_agreement",
        should_stop=should_stop,
    )


def stop_with_combined_confidence(
    vote_share_threshold: float = 0.75,
    entropy_threshold: float = 0.8,
    required_agreement: int = 2,
    minimum_stage: int = 2,
) -> StoppingPolicy:
    if not 0.0 <= vote_share_threshold <= 1.0:
        raise ValueError("Vote-share threshold must be between 0 and 1")

    if entropy_threshold < 0.0:
        raise ValueError("Entropy threshold cannot be negative")

    if required_agreement < 1:
        raise ValueError("Required agreement must be at least 1")

    def should_stop(
        snapshot: Mapping[str, Any],
    ) -> bool:
        stage = get_int(
            snapshot,
            "stage",
        )

        if stage < minimum_stage:
            return False

        top_vote_share = get_float(
            snapshot,
            "top_vote_share",
        )

        answer_entropy = get_float(
            snapshot,
            "answer_entropy",
        )

        consecutive_agreement = get_int(
            snapshot,
            "consecutive_agreement",
        )

        return (
            top_vote_share >= vote_share_threshold
            and answer_entropy <= entropy_threshold
            and consecutive_agreement >= required_agreement
        )

    vote_name = str(vote_share_threshold).replace(".", "_")

    entropy_name = str(entropy_threshold).replace(".", "_")

    return StoppingPolicy(
        name=(
            f"combined_vote_{vote_name}"
            f"_entropy_{entropy_name}"
            f"_agreement_{required_agreement}"
        ),
        should_stop=should_stop,
    )


def get_default_policies(
    maximum_stage: int = 5,
) -> list[StoppingPolicy]:
    if maximum_stage < 2:
        raise ValueError("Maximum stage must be at least 2")

    policies = [
        stop_at_fixed_stage(stage)
        for stage in range(
            2,
            maximum_stage + 1,
        )
    ]

    policies.extend(
        [
            stop_when_unanimous(
                minimum_stage=2,
            ),
            stop_at_vote_share(
                threshold=0.67,
                minimum_stage=2,
            ),
            stop_at_vote_share(
                threshold=0.75,
                minimum_stage=2,
            ),
            stop_at_vote_margin(
                threshold=0.50,
                minimum_stage=2,
            ),
            stop_below_entropy(
                threshold=0.0,
                minimum_stage=2,
            ),
            stop_below_entropy(
                threshold=0.8,
                minimum_stage=2,
            ),
            stop_after_consecutive_agreement(
                required_agreement=2,
                minimum_stage=2,
            ),
            stop_when_stable_majority(
                minimum_stage=3,
                required_agreement=2,
            ),
            stop_on_two_answer_agreement(),
            stop_with_combined_confidence(
                vote_share_threshold=0.75,
                entropy_threshold=0.8,
                required_agreement=2,
                minimum_stage=2,
            ),
        ]
    )

    return policies


def get_policy_by_name(
    name: str,
    maximum_stage: int = 5,
) -> StoppingPolicy:
    policies = get_default_policies(
        maximum_stage=maximum_stage,
    )

    matching_policies = [policy for policy in policies if policy.name == name]

    if not matching_policies:
        available_names = sorted(policy.name for policy in policies)

        raise ValueError(
            f"Unknown policy: {name}. Available policies: {available_names}"
        )

    return matching_policies[0]
