from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.config import load_config, resolve_project_path

FEATURE_COLUMNS = [
    "stage",
    "num_attempts",
    "num_valid_answers",
    "num_unique_answers",
    "missing_answer_count",
    "top_vote_share",
    "second_vote_share",
    "vote_margin",
    "answer_entropy",
    "latest_matches_majority",
    "unanimous",
    "majority_changed",
    "majority_change_count",
    "consecutive_agreement",
    "average_output_tokens",
    "total_output_tokens",
]

TARGET_COLUMN = "continue_label"


def convert_boolean_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    boolean_columns = [
        "latest_matches_majority",
        "unanimous",
        "majority_changed",
        TARGET_COLUMN,
    ]

    for column in boolean_columns:
        if column not in dataframe.columns:
            continue

        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )

    return dataframe


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = {
        "split",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Snapshots file is missing columns: {missing_columns}")

    if dataframe.empty:
        raise ValueError("Snapshots file is empty")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("continue_label contains missing or invalid values")

    missing_feature_values = dataframe[FEATURE_COLUMNS].isna().sum()

    invalid_features = missing_feature_values[missing_feature_values > 0]

    if not invalid_features.empty:
        raise ValueError(
            f"Feature columns contain missing values:\n{invalid_features.to_string()}"
        )


def split_data(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    train_dataframe = dataframe[dataframe["split"] == "train"].copy()

    validation_dataframe = dataframe[dataframe["split"] == "validation"].copy()

    if train_dataframe.empty:
        raise ValueError("No training snapshots were found")

    if validation_dataframe.empty:
        raise ValueError(
            "No validation snapshots were found. "
            "Generate validation responses before training."
        )

    x_train = train_dataframe[FEATURE_COLUMNS].astype(float)
    y_train = train_dataframe[TARGET_COLUMN].astype(int)

    x_validation = validation_dataframe[FEATURE_COLUMNS].astype(float)

    y_validation = validation_dataframe[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training labels contain only one class. "
            "Generate more responses before training."
        )

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def calculate_scale_pos_weight(
    labels: pd.Series,
) -> float:
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())

    if positive_count == 0:
        raise ValueError("No positive continue_label examples were found")

    return negative_count / positive_count


def create_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def create_xgboost_model(
    scale_pos_weight: float,
    seed: int,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_model(
    model: Any,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    predictions = model.predict(x_validation)

    probabilities = model.predict_proba(x_validation)[:, 1]

    metrics: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": (
            classification_report(
                y_validation,
                predictions,
                labels=[0, 1],
                target_names=["stop", "continue"],
                zero_division=0,
                output_dict=True,
            )
        ),
    }

    if y_validation.nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_validation,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def save_model(
    model: Any,
    path_value: str,
) -> Path:
    output_path = resolve_project_path(path_value)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        output_path,
    )

    return output_path


def save_metrics(
    metrics: dict[str, Any],
    path_value: str,
) -> Path:
    output_path = resolve_project_path(path_value)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    return output_path


def print_metrics(
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    print()
    print(model_name)
    print("-" * len(model_name))
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    else:
        print("ROC AUC:   unavailable")

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    config = load_config()

    seed = int(config["seed"])
    paths = config["paths"]

    snapshots_path = resolve_project_path(paths["snapshots"])

    dataframe = pd.read_csv(snapshots_path)
    dataframe = convert_boolean_columns(dataframe)

    validate_dataframe(dataframe)

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = split_data(dataframe)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    print(f"Training snapshots: {len(x_train)}")
    print(f"Validation snapshots: {len(x_validation)}")
    print(f"Training stop labels: {(y_train == 0).sum()}")
    print(f"Training continue labels: {(y_train == 1).sum()}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")

    logistic_model = create_logistic_model()

    logistic_model.fit(
        x_train,
        y_train,
    )

    logistic_metrics = evaluate_model(
        model=logistic_model,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    xgboost_model = create_xgboost_model(
        scale_pos_weight=scale_pos_weight,
        seed=seed,
    )

    xgboost_model.fit(
        x_train,
        y_train,
    )

    xgboost_metrics = evaluate_model(
        model=xgboost_model,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    logistic_path = save_model(
        model=logistic_model,
        path_value=paths["logistic_model"],
    )

    xgboost_path = save_model(
        model=xgboost_model,
        path_value=paths["xgboost_model"],
    )

    all_metrics = {
        "dataset": {
            "training_snapshots": len(x_train),
            "validation_snapshots": len(x_validation),
            "training_stop_labels": int((y_train == 0).sum()),
            "training_continue_labels": int((y_train == 1).sum()),
            "validation_stop_labels": int((y_validation == 0).sum()),
            "validation_continue_labels": int((y_validation == 1).sum()),
        },
        "logistic_regression": logistic_metrics,
        "xgboost": xgboost_metrics,
    }

    metrics_path = save_metrics(
        metrics=all_metrics,
        path_value=paths["metrics"],
    )

    print_metrics(
        "Logistic Regression",
        logistic_metrics,
    )

    print_metrics(
        "XGBoost",
        xgboost_metrics,
    )

    print()
    print("Training complete.")
    print(f"Logistic model: {logistic_path}")
    print(f"XGBoost model: {xgboost_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
