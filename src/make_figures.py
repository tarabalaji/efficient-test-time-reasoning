from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config, resolve_project_path

REQUIRED_COLUMNS = {
    "policy",
    "split",
    "accuracy",
    "final_accuracy",
    "average_attempts",
    "token_savings_rate",
    "incorrect_early_stops",
}


POLICY_LABELS = {
    "fixed_2": "Fixed 2",
    "fixed_3": "Fixed 3",
    "fixed_4": "Fixed 4",
    "two_answer_agreement": "Two-answer agreement",
    "consecutive_2_min_2": "Consecutive agreement",
    "unanimous_min_2": "Unanimity",
    "vote_share_0_67_min_2": "Vote share ≥ 0.67",
    "vote_share_0_75_min_2": "Vote share ≥ 0.75",
    "vote_margin_0_5_min_2": "Vote margin ≥ 0.50",
    "entropy_0_0_min_2": "Entropy = 0",
    "entropy_0_8_min_2": "Entropy ≤ 0.8",
    "stable_majority_2_min_3": "Stable majority",
    "combined_vote_0_75_entropy_0_8_agreement_2": "Combined confidence",
}


def validate_summary(
    dataframe: pd.DataFrame,
) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Policy summary is missing columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("Policy summary is empty")


def get_results_directory(
    config: dict[str, Any],
) -> Path:
    results_directory = Path(
        resolve_project_path(
            config.get(
                "paths",
                {},
            ).get(
                "results_dir",
                "results",
            )
        )
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return results_directory


def prepare_validation_results(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    validation = summary[summary["split"] == "validation"].copy()

    if validation.empty:
        raise ValueError("No validation policy results were found")

    validation["display_name"] = (
        validation["policy"].map(POLICY_LABELS).fillna(validation["policy"])
    )

    return validation


def select_main_policies(
    validation: pd.DataFrame,
) -> pd.DataFrame:
    selected_names = [
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "two_answer_agreement",
        "consecutive_2_min_2",
        "unanimous_min_2",
    ]

    selected = validation[validation["policy"].isin(selected_names)].copy()

    policy_order = {name: index for index, name in enumerate(selected_names)}

    selected["policy_order"] = selected["policy"].map(policy_order)

    return selected.sort_values("policy_order")


def create_accuracy_compute_plot(
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))

    axis.scatter(
        validation["average_attempts"],
        validation["accuracy"] * 100,
        s=90,
    )

    for _, row in validation.iterrows():
        axis.annotate(
            row["display_name"],
            (
                row["average_attempts"],
                row["accuracy"] * 100,
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    baseline_accuracy = validation["final_accuracy"].max() * 100

    axis.axhline(
        baseline_accuracy,
        linestyle="--",
        linewidth=1.3,
        label="Maximum-stage accuracy",
    )

    axis.set_xlabel("Average Number of Attempts")

    axis.set_ylabel("Validation Accuracy (%)")

    axis.set_title("Accuracy–Compute Tradeoff")

    axis.legend()

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_main_policy_plot(
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = select_main_policies(validation)

    figure, axis = plt.subplots(figsize=(10, 6))

    positions = range(len(selected))

    axis.bar(
        positions,
        selected["token_savings_rate"] * 100,
    )

    axis.set_xticks(list(positions))

    axis.set_xticklabels(
        selected["display_name"],
        rotation=25,
        ha="right",
    )

    axis.set_ylabel("Token Savings (%)")

    axis.set_title("Token Savings of Main Stopping Policies")

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_split_comparison_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_names = [
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "two_answer_agreement",
        "consecutive_2_min_2",
        "unanimous_min_2",
    ]

    selected = summary[summary["policy"].isin(selected_names)].copy()

    selected["display_name"] = (
        selected["policy"].map(POLICY_LABELS).fillna(selected["policy"])
    )

    pivot = selected.pivot(
        index="display_name",
        columns="split",
        values="average_attempts",
    )

    ordered_labels = [
        POLICY_LABELS[name]
        for name in selected_names
        if POLICY_LABELS[name] in pivot.index
    ]

    pivot = pivot.reindex(ordered_labels)

    figure, axis = plt.subplots(figsize=(10, 7))

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_xlabel("Policy")

    axis.set_ylabel("Average Attempts")

    axis.set_title("Average Attempts Across Dataset Splits")

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_error_savings_plot(
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = validation.sort_values(
        [
            "incorrect_early_stops",
            "token_savings_rate",
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.scatter(
        plot_data["token_savings_rate"] * 100,
        plot_data["incorrect_early_stops"],
        s=90,
    )

    for _, row in plot_data.iterrows():
        axis.annotate(
            row["display_name"],
            (
                row["token_savings_rate"] * 100,
                row["incorrect_early_stops"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("Token Savings (%)")

    axis.set_ylabel("Incorrect Early Stops")

    axis.set_title("Efficiency Versus Early-Stopping Risk")

    axis.grid(alpha=0.25)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    config = load_config()

    results_directory = get_results_directory(config)

    summary_path = results_directory / "policy_summary.csv"

    figures_directory = results_directory / "figures"

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(summary_path)

    validate_summary(summary)

    validation = prepare_validation_results(summary)

    create_accuracy_compute_plot(
        validation,
        figures_directory / "accuracy_compute_tradeoff.png",
    )

    create_main_policy_plot(
        validation,
        figures_directory / "main_policy_token_savings.png",
    )

    create_split_comparison_plot(
        summary,
        figures_directory / "train_validation_attempts.png",
    )

    create_error_savings_plot(
        validation,
        figures_directory / "error_savings_tradeoff.png",
    )

    print(f"Policy summary loaded from: {summary_path}")

    print(f"Figures saved to: {figures_directory}")


if __name__ == "__main__":
    main()
