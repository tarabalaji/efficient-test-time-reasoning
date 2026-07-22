from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.stopping_policies import StoppingPolicy, get_default_policies
from src.utils.config import load_config, resolve_project_path

BOOLEAN_COLUMNS = [
    "current_majority_correct",
    "final_majority_correct",
    "majority_will_change",
    "continue_label",
    "latest_matches_majority",
    "unanimous",
    "majority_changed",
]


def convert_boolean_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    boolean_mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    for column in BOOLEAN_COLUMNS:
        if column not in dataframe.columns:
            continue

        if pd.api.types.is_bool_dtype(dataframe[column]):
            continue

        converted = (
            dataframe[column].astype(str).str.strip().str.lower().map(boolean_mapping)
        )

        if converted.isna().any():
            invalid_values = sorted(
                dataframe.loc[
                    converted.isna(),
                    column,
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(f"Invalid boolean values in {column}: {invalid_values}")

        dataframe[column] = converted

    return dataframe


def validate_snapshots(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = {
        "question_id",
        "split",
        "stage",
        "gold_answer",
        "current_majority_answer",
        "current_majority_correct",
        "final_majority_answer",
        "final_majority_correct",
        "top_vote_share",
        "vote_margin",
        "answer_entropy",
        "unanimous",
        "majority_changed",
        "consecutive_agreement",
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Snapshots file is missing columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("Snapshots file is empty")

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "question_id",
                "split",
                "stage",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate question-stage rows")


def get_output_tokens(
    snapshot: pd.Series,
) -> float:
    if "total_output_tokens" in snapshot.index:
        value = snapshot["total_output_tokens"]

        if pd.notna(value):
            return float(value)

    if "average_output_tokens" in snapshot.index:
        value = snapshot["average_output_tokens"]

        if pd.notna(value):
            return float(value) * int(snapshot["stage"])

    return 0.0


def select_stopping_snapshot(
    question_snapshots: pd.DataFrame,
    policy: StoppingPolicy,
) -> tuple[pd.Series, bool]:
    ordered_snapshots = question_snapshots.sort_values("stage")

    for _, snapshot in ordered_snapshots.iterrows():
        if policy.should_stop(snapshot):
            return snapshot, True

    return ordered_snapshots.iloc[-1], False


def simulate_policy(
    dataframe: pd.DataFrame,
    policy: StoppingPolicy,
) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []

    group_columns = [
        "split",
        "question_id",
    ]

    for (
        split_name,
        question_id,
    ), question_snapshots in dataframe.groupby(
        group_columns,
        sort=False,
    ):
        question_snapshots = question_snapshots.sort_values("stage").reset_index(
            drop=True
        )

        selected_snapshot, triggered = select_stopping_snapshot(
            question_snapshots,
            policy,
        )

        final_snapshot = question_snapshots.iloc[-1]

        stopping_stage = int(selected_snapshot["stage"])

        maximum_stage = int(final_snapshot["stage"])

        predicted_answer = selected_snapshot["current_majority_answer"]

        gold_answer = selected_snapshot["gold_answer"]

        correct = bool(selected_snapshot["current_majority_correct"])

        final_correct = bool(final_snapshot["final_majority_correct"])

        stopped_early = stopping_stage < maximum_stage

        incorrect_early_stop = stopped_early and not correct and final_correct

        protected_from_regression = stopped_early and correct and not final_correct

        tokens_used = get_output_tokens(selected_snapshot)

        maximum_tokens = get_output_tokens(final_snapshot)

        tokens_saved = max(
            maximum_tokens - tokens_used,
            0.0,
        )

        token_savings_rate = (
            tokens_saved / maximum_tokens if maximum_tokens > 0 else 0.0
        )

        attempts_saved = maximum_stage - stopping_stage

        result_rows.append(
            {
                "policy": policy.name,
                "split": split_name,
                "question_id": question_id,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "correct": correct,
                "final_majority_answer": final_snapshot["final_majority_answer"],
                "final_correct": final_correct,
                "stopping_stage": stopping_stage,
                "maximum_stage": maximum_stage,
                "attempts_used": stopping_stage,
                "attempts_saved": attempts_saved,
                "stopped_early": stopped_early,
                "policy_triggered": triggered,
                "incorrect_early_stop": (incorrect_early_stop),
                "protected_from_regression": (protected_from_regression),
                "top_vote_share": selected_snapshot["top_vote_share"],
                "vote_margin": selected_snapshot["vote_margin"],
                "answer_entropy": selected_snapshot["answer_entropy"],
                "unanimous": selected_snapshot["unanimous"],
                "consecutive_agreement": (selected_snapshot["consecutive_agreement"]),
                "tokens_used": tokens_used,
                "maximum_tokens": maximum_tokens,
                "tokens_saved": tokens_saved,
                "token_savings_rate": (token_savings_rate),
            }
        )

    return pd.DataFrame(result_rows)


def summarize_simulations(
    simulation_results: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    grouping_columns = [
        "policy",
        "split",
    ]

    for (
        policy_name,
        split_name,
    ), group in simulation_results.groupby(
        grouping_columns,
        sort=False,
    ):
        question_count = len(group)

        accuracy = group["correct"].mean()
        final_accuracy = group["final_correct"].mean()

        accuracy_difference = accuracy - final_accuracy

        average_attempts = group["attempts_used"].mean()

        average_maximum_attempts = group["maximum_stage"].mean()

        average_attempts_saved = group["attempts_saved"].mean()

        attempt_savings_rate = (
            average_attempts_saved / average_maximum_attempts
            if average_maximum_attempts > 0
            else 0.0
        )

        total_tokens_used = group["tokens_used"].sum()

        total_maximum_tokens = group["maximum_tokens"].sum()

        total_tokens_saved = group["tokens_saved"].sum()

        token_savings_rate = (
            total_tokens_saved / total_maximum_tokens
            if total_maximum_tokens > 0
            else 0.0
        )

        stopped_early_count = int(group["stopped_early"].sum())

        incorrect_early_stops = int(group["incorrect_early_stop"].sum())

        protected_from_regression = int(group["protected_from_regression"].sum())

        triggered_count = int(group["policy_triggered"].sum())

        summary_rows.append(
            {
                "policy": policy_name,
                "split": split_name,
                "questions": question_count,
                "accuracy": accuracy,
                "final_accuracy": final_accuracy,
                "accuracy_difference": (accuracy_difference),
                "average_attempts": average_attempts,
                "average_maximum_attempts": (average_maximum_attempts),
                "average_attempts_saved": (average_attempts_saved),
                "attempt_savings_rate": (attempt_savings_rate),
                "total_tokens_used": (total_tokens_used),
                "total_maximum_tokens": (total_maximum_tokens),
                "total_tokens_saved": (total_tokens_saved),
                "token_savings_rate": (token_savings_rate),
                "stopped_early_count": (stopped_early_count),
                "stopped_early_rate": (stopped_early_count / question_count),
                "incorrect_early_stops": (incorrect_early_stops),
                "protected_from_regression": (protected_from_regression),
                "policy_triggered_count": (triggered_count),
                "policy_triggered_rate": (triggered_count / question_count),
            }
        )

    return pd.DataFrame(summary_rows)


def create_output_paths(
    config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    paths_config = config.get("paths", {})

    processed_directory = resolve_project_path(
        paths_config.get(
            "processed_dir",
            "data/processed",
        )
    )

    results_directory = resolve_project_path(
        paths_config.get(
            "results_dir",
            "results",
        )
    )

    processed_directory = Path(processed_directory)

    results_directory = Path(results_directory)

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = processed_directory / ("policy_simulations.csv")

    summary_path = results_directory / ("policy_summary.csv")

    json_path = results_directory / ("policy_summary.json")

    return (
        detailed_path,
        summary_path,
        json_path,
    )


def save_results(
    simulation_results: pd.DataFrame,
    summary: pd.DataFrame,
    detailed_path: Path,
    summary_path: Path,
    json_path: Path,
) -> None:
    simulation_results.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    json_records = summary.to_dict(orient="records")

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_records,
            file,
            indent=2,
        )


def print_summary(
    summary: pd.DataFrame,
) -> None:
    display_columns = [
        "policy",
        "split",
        "accuracy",
        "accuracy_difference",
        "average_attempts",
        "attempt_savings_rate",
        "token_savings_rate",
        "incorrect_early_stops",
    ]

    display_dataframe = summary[display_columns].copy()

    percentage_columns = [
        "accuracy",
        "accuracy_difference",
        "attempt_savings_rate",
        "token_savings_rate",
    ]

    for column in percentage_columns:
        display_dataframe[column] = display_dataframe[column].map(
            lambda value: f"{value:.2%}"
        )

    display_dataframe["average_attempts"] = display_dataframe["average_attempts"].map(
        lambda value: f"{value:.2f}"
    )

    print()
    print("Policy Simulation Summary")
    print("=========================")
    print(display_dataframe.to_string(index=False))


def main() -> None:
    config = load_config()

    snapshots_path = resolve_project_path(config["paths"]["snapshots"])

    snapshots = pd.read_csv(snapshots_path)

    snapshots = convert_boolean_columns(snapshots)

    validate_snapshots(snapshots)

    maximum_stage = int(snapshots["stage"].max())

    policies = get_default_policies(
        maximum_stage=maximum_stage,
    )

    simulation_frames = [
        simulate_policy(
            snapshots,
            policy,
        )
        for policy in policies
    ]

    simulation_results = pd.concat(
        simulation_frames,
        ignore_index=True,
    )

    summary = summarize_simulations(simulation_results)

    summary = summary.sort_values(
        [
            "split",
            "accuracy",
            "average_attempts",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    (
        detailed_path,
        summary_path,
        json_path,
    ) = create_output_paths(config)

    save_results(
        simulation_results,
        summary,
        detailed_path,
        summary_path,
        json_path,
    )

    print(f"Snapshots file: {snapshots_path}")
    print(f"Maximum available stage: {maximum_stage}")
    print(f"Policies simulated: {len(policies)}")
    print(f"Detailed results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"JSON summary saved to: {json_path}")

    print_summary(summary)


if __name__ == "__main__":
    main()
