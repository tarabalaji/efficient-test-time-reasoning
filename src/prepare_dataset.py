from __future__ import annotations

import random
from typing import Any

from datasets import Dataset, load_dataset

from src.extract_answers import extract_gsm8k_gold_answer
from src.utils.config import load_config
from src.utils.io import write_jsonl


def prepare_example(
    example: dict[str, Any],
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    gold_answer = extract_gsm8k_gold_answer(example.get("answer"))

    if gold_answer is None:
        raise ValueError(
            f"Could not extract the gold answer from "
            f"{source_split} example {source_index}"
        )

    question = str(example.get("question", "")).strip()

    if not question:
        raise ValueError(
            f"Question text is missing from {source_split} example {source_index}"
        )

    return {
        "source_split": source_split,
        "source_index": source_index,
        "question": question,
        "gold_solution": str(example["answer"]).strip(),
        "gold_answer": gold_answer,
    }


def convert_split(
    dataset: Dataset,
    source_split: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for source_index, example in enumerate(dataset):
        records.append(
            prepare_example(
                example=dict(example),
                source_split=source_split,
                source_index=source_index,
            )
        )

    return records


def assign_project_split(
    records: list[dict[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    assigned_records: list[dict[str, Any]] = []

    for split_index, record in enumerate(records):
        assigned_records.append(
            {
                "question_id": f"{split_name}_{split_index:04d}",
                "split": split_name,
                **record,
            }
        )

    return assigned_records


def validate_requested_sizes(
    train_size: int,
    validation_size: int,
    test_size: int,
    available_train: int,
    available_test: int,
) -> None:
    if train_size <= 0:
        raise ValueError("train_size must be greater than zero")

    if validation_size <= 0:
        raise ValueError("validation_size must be greater than zero")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero")

    requested_training_total = train_size + validation_size

    if requested_training_total > available_train:
        raise ValueError(
            f"Requested {requested_training_total} examples from the official "
            f"training split, but only {available_train} are available"
        )

    if test_size > available_test:
        raise ValueError(
            f"Requested {test_size} examples from the official test split, "
            f"but only {available_test} are available"
        )


def main() -> None:
    config = load_config()

    seed = int(config["seed"])
    dataset_config = config["dataset"]
    paths = config["paths"]

    dataset_name = str(dataset_config["name"])
    dataset_subset = str(dataset_config["subset"])

    train_size = int(dataset_config["train_size"])
    validation_size = int(dataset_config["validation_size"])
    test_size = int(dataset_config["test_size"])

    print(f"Loading {dataset_name} ({dataset_subset})...")

    dataset = load_dataset(
        dataset_name,
        dataset_subset,
    )

    if "train" not in dataset or "test" not in dataset:
        raise ValueError("The dataset must contain official train and test splits")

    official_train = convert_split(
        dataset=dataset["train"],
        source_split="train",
    )
    official_test = convert_split(
        dataset=dataset["test"],
        source_split="test",
    )

    validate_requested_sizes(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        available_train=len(official_train),
        available_test=len(official_test),
    )

    write_jsonl(official_train, paths["raw_train"])
    write_jsonl(official_test, paths["raw_test"])

    random_generator = random.Random(seed)
    shuffled_train = official_train.copy()
    shuffled_test = official_test.copy()

    random_generator.shuffle(shuffled_train)
    random_generator.shuffle(shuffled_test)

    train_records = shuffled_train[:train_size]

    validation_start = train_size
    validation_end = train_size + validation_size
    validation_records = shuffled_train[validation_start:validation_end]

    test_records = shuffled_test[:test_size]

    processed_records = [
        *assign_project_split(train_records, "train"),
        *assign_project_split(validation_records, "validation"),
        *assign_project_split(test_records, "test"),
    ]

    write_jsonl(processed_records, paths["questions"])

    print()
    print("Dataset preparation complete.")
    print(f"Official training examples: {len(official_train)}")
    print(f"Official test examples: {len(official_test)}")
    print(f"Project training examples: {len(train_records)}")
    print(f"Project validation examples: {len(validation_records)}")
    print(f"Project test examples: {len(test_records)}")
    print(f"Total project questions: {len(processed_records)}")
    print()
    print(f"Raw training data: {paths['raw_train']}")
    print(f"Raw test data: {paths['raw_test']}")
    print(f"Processed questions: {paths['questions']}")


if __name__ == "__main__":
    main()
