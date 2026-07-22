from __future__ import annotations

import pandas as pd

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


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = {
        "question_id",
        "split",
        "stage",
        "current_majority_correct",
        "final_majority_correct",
        "majority_will_change",
        "continue_label",
        "top_vote_share",
        "vote_margin",
        "answer_entropy",
        "unanimous",
        "majority_changed",
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Snapshots file is missing columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("Snapshots file is empty")


def print_heading(
    title: str,
) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_basic_summary(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Dataset Summary")

    print(f"Snapshots: {len(dataframe)}")
    print(f"Questions: {dataframe['question_id'].nunique()}")
    print(f"Splits: {sorted(dataframe['split'].unique().tolist())}")
    print(f"Stages: {sorted(dataframe['stage'].unique().tolist())}")

    print()
    print("Snapshots by split:")
    print(dataframe["split"].value_counts().sort_index().to_string())

    print()
    print("Questions by split:")
    print(dataframe.groupby("split")["question_id"].nunique().sort_index().to_string())


def print_label_distribution(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Continue Label Distribution")

    overall_counts = (
        dataframe["continue_label"].value_counts().reindex([False, True], fill_value=0)
    )

    total = len(dataframe)

    print(f"Stop labels: {overall_counts[False]} ({overall_counts[False] / total:.2%})")
    print(
        f"Continue labels: {overall_counts[True]} ({overall_counts[True] / total:.2%})"
    )

    print()
    print("Labels by split:")

    by_split = pd.crosstab(
        dataframe["split"],
        dataframe["continue_label"],
    ).reindex(
        columns=[False, True],
        fill_value=0,
    )

    by_split.columns = ["stop", "continue"]

    by_split["continue_rate"] = by_split["continue"] / by_split.sum(axis=1)

    print(by_split.to_string())

    print()
    print("Labels by split and stage:")

    by_split_stage = pd.crosstab(
        [
            dataframe["split"],
            dataframe["stage"],
        ],
        dataframe["continue_label"],
    ).reindex(
        columns=[False, True],
        fill_value=0,
    )

    by_split_stage.columns = [
        "stop",
        "continue",
    ]

    by_split_stage["continue_rate"] = by_split_stage["continue"] / by_split_stage.sum(
        axis=1
    )

    print(by_split_stage.to_string())


def print_accuracy_summary(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Accuracy Summary")

    overall_current_accuracy = dataframe["current_majority_correct"].mean()

    final_question_rows = (
        dataframe.sort_values(["question_id", "stage"]).groupby("question_id").tail(1)
    )

    overall_final_accuracy = final_question_rows["final_majority_correct"].mean()

    print(f"Current snapshot accuracy: {overall_current_accuracy:.2%}")

    print(f"Final question accuracy: {overall_final_accuracy:.2%}")

    print()
    print("Current accuracy by split and stage:")

    current_accuracy = (
        dataframe.groupby(["split", "stage"])["current_majority_correct"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "snapshots",
                "mean": "accuracy",
            }
        )
    )

    print(current_accuracy.to_string())

    print()
    print("Final accuracy by split:")

    final_accuracy = (
        final_question_rows.groupby("split")["final_majority_correct"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "questions",
                "mean": "accuracy",
            }
        )
    )

    print(final_accuracy.to_string())


def print_transition_summary(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Current-to-Final Transitions")

    transition_counts = pd.crosstab(
        dataframe["current_majority_correct"],
        dataframe["final_majority_correct"],
    ).reindex(
        index=[False, True],
        columns=[False, True],
        fill_value=0,
    )

    transition_counts.index = [
        "current_wrong",
        "current_correct",
    ]

    transition_counts.columns = [
        "final_wrong",
        "final_correct",
    ]

    print(transition_counts.to_string())

    wrong_to_correct = int(
        (
            ~dataframe["current_majority_correct"] & dataframe["final_majority_correct"]
        ).sum()
    )

    correct_to_wrong = int(
        (
            dataframe["current_majority_correct"] & ~dataframe["final_majority_correct"]
        ).sum()
    )

    unchanged_correct = int(
        (
            dataframe["current_majority_correct"] & dataframe["final_majority_correct"]
        ).sum()
    )

    unchanged_wrong = int(
        (
            ~dataframe["current_majority_correct"]
            & ~dataframe["final_majority_correct"]
        ).sum()
    )

    print()
    print(f"Wrong to correct: {wrong_to_correct}")
    print(f"Correct to wrong: {correct_to_wrong}")
    print(f"Correct to correct: {unchanged_correct}")
    print(f"Wrong to wrong: {unchanged_wrong}")


def print_majority_change_summary(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Majority Change Summary")

    print(
        f"Snapshots where future majority changes: "
        f"{dataframe['majority_will_change'].sum()} "
        f"({dataframe['majority_will_change'].mean():.2%})"
    )

    print(
        f"Snapshots where majority already changed: "
        f"{dataframe['majority_changed'].sum()} "
        f"({dataframe['majority_changed'].mean():.2%})"
    )

    change_outcomes = dataframe[dataframe["majority_will_change"]].copy()

    if change_outcomes.empty:
        print()
        print("No future majority changes were found.")
        return

    improvements = int(
        (
            ~change_outcomes["current_majority_correct"]
            & change_outcomes["final_majority_correct"]
        ).sum()
    )

    harms = int(
        (
            change_outcomes["current_majority_correct"]
            & ~change_outcomes["final_majority_correct"]
        ).sum()
    )

    unchanged = len(change_outcomes) - improvements - harms

    print()
    print(f"Majority changes that improve accuracy: {improvements}")
    print(f"Majority changes that harm accuracy: {harms}")
    print(f"Majority changes with no accuracy effect: {unchanged}")

    print()
    print("Future majority changes by split and stage:")

    change_by_stage = (
        dataframe.groupby(["split", "stage"])["majority_will_change"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "snapshots",
                "sum": "changes",
                "mean": "change_rate",
            }
        )
    )

    print(change_by_stage.to_string())


def print_feature_summary(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Uncertainty Feature Summary")

    feature_columns = [
        "top_vote_share",
        "vote_margin",
        "answer_entropy",
    ]

    available_features = [
        column for column in feature_columns if column in dataframe.columns
    ]

    feature_summary = dataframe[available_features].describe().transpose()

    print(feature_summary.to_string())

    print()
    print("Mean features by continue label:")

    by_label = dataframe.groupby("continue_label")[available_features].mean()

    by_label.index = [
        "stop" if value is False else "continue" for value in by_label.index
    ]

    print(by_label.to_string())

    print()
    print("Unanimous snapshots by label:")

    unanimous_table = pd.crosstab(
        dataframe["continue_label"],
        dataframe["unanimous"],
    ).reindex(
        index=[False, True],
        columns=[False, True],
        fill_value=0,
    )

    unanimous_table.index = [
        "stop",
        "continue",
    ]

    unanimous_table.columns = [
        "not_unanimous",
        "unanimous",
    ]

    print(unanimous_table.to_string())


def print_positive_examples(
    dataframe: pd.DataFrame,
    limit: int = 20,
) -> None:
    print_heading("Positive Continue Examples")

    positives = dataframe[dataframe["continue_label"]].copy()

    if positives.empty:
        print("No positive continue-label examples found.")
        return

    columns = [
        "question_id",
        "split",
        "stage",
        "gold_answer",
        "current_majority_answer",
        "final_majority_answer",
        "top_vote_share",
        "vote_margin",
        "answer_entropy",
        "majority_change_count",
        "consecutive_agreement",
    ]

    columns = [column for column in columns if column in positives.columns]

    print(positives[columns].head(limit).to_string(index=False))

    if len(positives) > limit:
        print()
        print(f"Showing {limit} of {len(positives)} positive examples.")


def print_warnings(
    dataframe: pd.DataFrame,
) -> None:
    print_heading("Warnings")

    warnings: list[str] = []

    for split_name, split_dataframe in dataframe.groupby("split"):
        positive_count = int(split_dataframe["continue_label"].sum())

        negative_count = int((~split_dataframe["continue_label"]).sum())

        if positive_count == 0:
            warnings.append(f"{split_name} has no positive continue labels.")

        if negative_count == 0:
            warnings.append(f"{split_name} has no negative stop labels.")

        if positive_count > 0 and positive_count < 20:
            warnings.append(
                f"{split_name} has only {positive_count} positive examples."
            )

    positive_rate = dataframe["continue_label"].mean()

    if positive_rate < 0.05:
        warnings.append(f"Overall positive-label rate is only {positive_rate:.2%}.")

    if dataframe["continue_label"].sum() > 0:
        positive_stages = dataframe.loc[
            dataframe["continue_label"],
            "stage",
        ].value_counts(normalize=True)

        if positive_stages.max() > 0.90:
            dominant_stage = int(positive_stages.idxmax())

            warnings.append(
                f"More than 90% of positive labels occur at stage {dominant_stage}."
            )

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
        warnings.append(f"Found {duplicate_count} duplicate question-stage rows.")

    if not warnings:
        print("No major dataset warnings detected.")
        return

    for warning in warnings:
        print(f"- {warning}")


def main() -> None:
    config = load_config()

    snapshots_path = resolve_project_path(config["paths"]["snapshots"])

    dataframe = pd.read_csv(snapshots_path)

    dataframe = convert_boolean_columns(dataframe)

    validate_dataframe(dataframe)

    print(f"Snapshots file: {snapshots_path}")

    print_basic_summary(dataframe)
    print_label_distribution(dataframe)
    print_accuracy_summary(dataframe)
    print_transition_summary(dataframe)
    print_majority_change_summary(dataframe)
    print_feature_summary(dataframe)
    print_positive_examples(dataframe)
    print_warnings(dataframe)


if __name__ == "__main__":
    main()
