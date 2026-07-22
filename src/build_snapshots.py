from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from src.extract_features import (
    extract_snapshot_features,
    get_majority_answer,
)
from src.utils.config import load_config, resolve_project_path
from src.utils.io import read_jsonl


def group_responses(
    responses: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for response in responses:
        if response.get("status") != "success":
            continue

        question_id = response.get("question_id")

        if not isinstance(question_id, str):
            continue

        grouped[question_id].append(response)

    for question_responses in grouped.values():
        question_responses.sort(key=lambda response: int(response["attempt"]))

    return dict(grouped)


def get_majority_at_stage(
    responses: list[dict[str, Any]],
    stage: int,
) -> str | None:
    answers = [
        str(response["extracted_answer"])
        for response in responses
        if int(response["attempt"]) <= stage
        and response.get("extracted_answer") is not None
    ]

    return get_majority_answer(answers)


def get_future_majorities(
    responses: list[dict[str, Any]],
    stage: int,
    maximum_attempts: int,
) -> list[str | None]:
    return [
        get_majority_at_stage(responses, future_stage)
        for future_stage in range(
            stage + 1,
            maximum_attempts + 1,
        )
    ]


def determine_continue_label(
    current_majority: str | None,
    gold_answer: str,
    future_majorities: list[str | None],
) -> bool:
    current_correct = current_majority == gold_answer

    if current_correct:
        return False

    return any(future_majority == gold_answer for future_majority in future_majorities)


def determine_majority_will_change(
    current_majority: str | None,
    future_majorities: list[str | None],
) -> bool:
    return any(
        future_majority is not None and future_majority != current_majority
        for future_majority in future_majorities
    )


def create_snapshot(
    question: dict[str, Any],
    responses: list[dict[str, Any]],
    stage: int,
    maximum_attempts: int,
) -> dict[str, Any]:
    stage_responses = [
        response for response in responses if int(response["attempt"]) <= stage
    ]

    if len(stage_responses) < stage:
        raise ValueError(
            f"Question {question['question_id']} does not have "
            f"all responses through attempt {stage}"
        )

    features = extract_snapshot_features(stage_responses)

    gold_answer = str(question["gold_answer"])
    current_majority = get_majority_at_stage(
        responses=responses,
        stage=stage,
    )

    future_majorities = get_future_majorities(
        responses=responses,
        stage=stage,
        maximum_attempts=maximum_attempts,
    )

    final_majority = get_majority_at_stage(
        responses=responses,
        stage=maximum_attempts,
    )

    continue_label = determine_continue_label(
        current_majority=current_majority,
        gold_answer=gold_answer,
        future_majorities=future_majorities,
    )

    snapshot = {
        "question_id": question["question_id"],
        "split": question["split"],
        "stage": stage,
        "gold_answer": gold_answer,
        "current_majority_answer": current_majority,
        "final_majority_answer": final_majority,
        "current_majority_correct": (current_majority == gold_answer),
        "final_majority_correct": (final_majority == gold_answer),
        "majority_will_change": determine_majority_will_change(
            current_majority=current_majority,
            future_majorities=future_majorities,
        ),
        "continue_label": continue_label,
        **features,
    }

    return snapshot


def validate_question_responses(
    question_id: str,
    responses: list[dict[str, Any]],
    maximum_attempts: int,
) -> bool:
    attempts = {int(response["attempt"]) for response in responses}

    required_attempts = set(range(1, maximum_attempts + 1))

    if not required_attempts.issubset(attempts):
        missing_attempts = sorted(required_attempts - attempts)

        print(f"Skipping {question_id}: missing attempts {missing_attempts}")

        return False

    return True


def main() -> None:
    config = load_config()

    controller_config = config["controller"]
    paths = config["paths"]

    decision_stages = [int(stage) for stage in controller_config["decision_stages"]]

    maximum_attempts = int(controller_config["maximum_attempts"])

    questions = read_jsonl(paths["questions"])
    responses = read_jsonl(paths["generations"])

    grouped_responses = group_responses(responses)

    snapshots: list[dict[str, Any]] = []
    skipped_questions = 0

    for question in questions:
        question_id = str(question["question_id"])
        question_responses = grouped_responses.get(
            question_id,
            [],
        )

        if not validate_question_responses(
            question_id=question_id,
            responses=question_responses,
            maximum_attempts=maximum_attempts,
        ):
            skipped_questions += 1
            continue

        for stage in decision_stages:
            snapshot = create_snapshot(
                question=question,
                responses=question_responses,
                stage=stage,
                maximum_attempts=maximum_attempts,
            )

            snapshots.append(snapshot)

    if not snapshots:
        raise ValueError(
            "No snapshots were created. Make sure response generation is complete."
        )

    snapshots_dataframe = pd.DataFrame(snapshots)

    output_path = resolve_project_path(paths["snapshots"])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Snapshot construction complete.")
    print(f"Questions available: {len(questions)}")
    print(f"Questions skipped: {skipped_questions}")
    print(f"Snapshots created: {len(snapshots_dataframe)}")
    print(f"Output file: {output_path}")
    print()
    print("Snapshots by split:")
    print(snapshots_dataframe["split"].value_counts().sort_index().to_string())
    print()
    print("Snapshots by stage:")
    print(snapshots_dataframe["stage"].value_counts().sort_index().to_string())
    print()
    print("Continue-label distribution:")
    print(snapshots_dataframe["continue_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
