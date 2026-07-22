from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.utils.config import create_parent_directory, resolve_project_path


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    output_path = create_parent_directory(path)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def append_jsonl(
    record: dict[str, Any],
    path: str | Path,
) -> Path:
    output_path = create_parent_directory(path)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = resolve_project_path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} in {input_path} must contain a JSON object"
                )

            records.append(record)

    return records


def write_json(data: Any, path: str | Path) -> Path:
    output_path = create_parent_directory(path)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return output_path


def read_json(path: str | Path) -> Any:
    input_path = resolve_project_path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)
