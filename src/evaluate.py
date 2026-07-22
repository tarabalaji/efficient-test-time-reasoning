from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config, resolve_project_path

REQUIRED_COLUMNS = {
    "policy",
    "split",
    "questions",
    "accuracy",
    "final_accuracy",
    "accuracy_difference",
    "average_attempts",
    "average_maximum_attempts",
    "attempt_savings_rate",
    "token_savings_rate",
    "incorrect_early_stops",
    "protected_from_regression",
}


def validate_summary(
    dataframe: pd.DataFrame,
) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Policy summary is missing columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("Policy summary file is empty")


def create_output_directory(
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

    figures_directory = results_directory / "figures"

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return figures_directory


def select_recommended_policy(
    validation_results: pd.DataFrame,
) -> pd.Series:
    baseline_accuracy = float(validation_results["final_accuracy"].max())

    eligible = validation_results[
        validation_results["accuracy"] >= baseline_accuracy
    ].copy()

    if eligible.empty:
        eligible = validation_results.copy()

    eligible = eligible.sort_values(
        [
            "accuracy",
            "incorrect_early_stops",
            "average_attempts",
            "token_savings_rate",
        ],
        ascending=[
            False,
            True,
            True,
            False,
        ],
    )

    return eligible.iloc[0]


def create_ranked_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    ranked = summary.copy()

    ranked["accuracy_retention"] = ranked["accuracy"] / ranked["final_accuracy"]

    ranked["efficiency_score"] = ranked["accuracy"] * (
        1.0 + ranked["token_savings_rate"]
    )

    ranked = ranked.sort_values(
        [
            "split",
            "accuracy",
            "incorrect_early_stops",
            "average_attempts",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    ranked["rank"] = ranked.groupby("split").cumcount() + 1

    columns = [
        "rank",
        "policy",
        "split",
        "accuracy",
        "final_accuracy",
        "accuracy_difference",
        "accuracy_retention",
        "average_attempts",
        "average_attempts_saved",
        "attempt_savings_rate",
        "token_savings_rate",
        "incorrect_early_stops",
        "protected_from_regression",
        "efficiency_score",
    ]

    available_columns = [column for column in columns if column in ranked.columns]

    return ranked[available_columns]


def create_accuracy_attempts_plot(
    validation_results: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))

    axis.scatter(
        validation_results["average_attempts"],
        validation_results["accuracy"],
        s=80,
    )

    for _, row in validation_results.iterrows():
        axis.annotate(
            row["policy"],
            (
                row["average_attempts"],
                row["accuracy"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("Average Attempts")

    axis.set_ylabel("Accuracy")

    axis.set_title("Validation Accuracy vs. Test-Time Attempts")

    axis.grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_token_savings_plot(
    validation_results: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = validation_results.sort_values(
        "token_savings_rate",
        ascending=True,
    ).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(10, 8))

    axis.barh(
        plot_data["policy"],
        plot_data["token_savings_rate"] * 100,
    )

    axis.set_xlabel("Token Savings (%)")

    axis.set_ylabel("Policy")

    axis.set_title("Validation Token Savings by Policy")

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_accuracy_plot(
    validation_results: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = validation_results.sort_values(
        [
            "accuracy",
            "average_attempts",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(10, 8))

    axis.barh(
        plot_data["policy"],
        plot_data["accuracy"] * 100,
    )

    baseline_accuracy = validation_results["final_accuracy"].max() * 100

    axis.axvline(
        baseline_accuracy,
        linestyle="--",
        linewidth=1.5,
        label="Maximum-stage accuracy",
    )

    axis.set_xlabel("Accuracy (%)")

    axis.set_ylabel("Policy")

    axis.set_title("Validation Accuracy by Policy")

    axis.legend()

    axis.grid(
        axis="x",
        alpha=0.3,
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
    pivot = summary.pivot(
        index="policy",
        columns="split",
        values="average_attempts",
    )

    pivot = pivot.sort_values(
        by=list(pivot.columns),
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(11, 8))

    pivot.plot(
        kind="barh",
        ax=axis,
    )

    axis.set_xlabel("Average Attempts")

    axis.set_ylabel("Policy")

    axis.set_title("Average Attempts by Dataset Split")

    axis.grid(
        axis="x",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_recommendation_report(
    recommended: pd.Series,
    validation_results: pd.DataFrame,
) -> dict[str, Any]:
    baseline_accuracy = float(validation_results["final_accuracy"].max())

    fixed_four = validation_results[validation_results["policy"] == "fixed_4"]

    fixed_two = validation_results[validation_results["policy"] == "fixed_2"]

    report: dict[str, Any] = {
        "recommended_policy": (recommended["policy"]),
        "validation_accuracy": float(recommended["accuracy"]),
        "baseline_accuracy": (baseline_accuracy),
        "accuracy_difference": float(recommended["accuracy_difference"]),
        "average_attempts": float(recommended["average_attempts"]),
        "attempt_savings_rate": float(recommended["attempt_savings_rate"]),
        "token_savings_rate": float(recommended["token_savings_rate"]),
        "incorrect_early_stops": int(recommended["incorrect_early_stops"]),
        "protected_from_regression": int(recommended["protected_from_regression"]),
    }

    if not fixed_four.empty:
        report["fixed_4_average_attempts"] = float(
            fixed_four.iloc[0]["average_attempts"]
        )

    if not fixed_two.empty:
        report["fixed_2_accuracy"] = float(fixed_two.iloc[0]["accuracy"])

    return report


def print_recommendation(
    report: dict[str, Any],
) -> None:
    print()
    print("Recommended Policy")
    print("==================")

    print(f"Policy: {report['recommended_policy']}")

    print(f"Validation accuracy: {report['validation_accuracy']:.2%}")

    print(f"Maximum-stage accuracy: {report['baseline_accuracy']:.2%}")

    print(f"Accuracy difference: {report['accuracy_difference']:.2%}")

    print(f"Average attempts: {report['average_attempts']:.2f}")

    print(f"Attempt savings: {report['attempt_savings_rate']:.2%}")

    print(f"Token savings: {report['token_savings_rate']:.2%}")

    print(f"Incorrect early stops: {report['incorrect_early_stops']}")


def print_validation_table(
    validation_results: pd.DataFrame,
) -> None:
    display_columns = [
        "policy",
        "accuracy",
        "accuracy_difference",
        "average_attempts",
        "token_savings_rate",
        "incorrect_early_stops",
    ]

    display = (
        validation_results[display_columns]
        .sort_values(
            [
                "accuracy",
                "average_attempts",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .copy()
    )

    for column in [
        "accuracy",
        "accuracy_difference",
        "token_savings_rate",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    display["average_attempts"] = display["average_attempts"].map(
        lambda value: f"{value:.2f}"
    )

    print()
    print("Validation Policy Ranking")
    print("=========================")

    print(display.to_string(index=False))


def main() -> None:
    config = load_config()

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

    summary_path = results_directory / "policy_summary.csv"

    summary = pd.read_csv(summary_path)

    validate_summary(summary)

    validation_results = (
        summary[summary["split"] == "validation"].copy().reset_index(drop=True)
    )

    if validation_results.empty:
        raise ValueError("No validation results were found")

    ranked_table = create_ranked_table(summary)

    recommended = select_recommended_policy(validation_results)

    report = create_recommendation_report(
        recommended,
        validation_results,
    )

    figures_directory = create_output_directory(config)

    ranked_path = results_directory / "policy_rankings.csv"

    report_path = results_directory / "evaluation_summary.json"

    ranked_table.to_csv(
        ranked_path,
        index=False,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    create_accuracy_attempts_plot(
        validation_results,
        figures_directory / "accuracy_vs_attempts.png",
    )

    create_token_savings_plot(
        validation_results,
        figures_directory / "token_savings.png",
    )

    create_accuracy_plot(
        validation_results,
        figures_directory / "policy_accuracy.png",
    )

    create_split_comparison_plot(
        summary,
        figures_directory / "split_attempt_comparison.png",
    )

    print(f"Policy summary loaded from: {summary_path}")

    print(f"Ranked results saved to: {ranked_path}")

    print(f"Evaluation summary saved to: {report_path}")

    print(f"Figures saved to: {figures_directory}")

    print_validation_table(validation_results)

    print_recommendation(report)


if __name__ == "__main__":
    main()
