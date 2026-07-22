from __future__ import annotations

import re

from src.normalize_answers import normalize_answer

ANSWER_PATTERNS = [
    re.compile(
        r"\\boxed\{([^{}]+)\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"####\s*([^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"final\s+answer\s*(?:is|:|=)\s*([^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"answer\s*(?:is|:|=)\s*([^\n]+)",
        re.IGNORECASE,
    ),
]


NUMBER_PATTERN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?(?:\s*/\s*-?\d+)?")


def extract_answer(response_text: str | None) -> str | None:
    if response_text is None:
        return None

    response_text = response_text.strip()

    if not response_text:
        return None

    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(response_text)

        if not matches:
            continue

        for match in reversed(matches):
            candidate = extract_numeric_candidate(match)

            if candidate is not None:
                return candidate

    numeric_matches = NUMBER_PATTERN.findall(response_text)

    for match in reversed(numeric_matches):
        normalized = normalize_answer(match)

        if normalized is not None:
            return normalized

    return None


def extract_numeric_candidate(text: str) -> str | None:
    cleaned = text.strip()

    normalized = normalize_answer(cleaned)

    if normalized is not None:
        return normalized

    numeric_matches = NUMBER_PATTERN.findall(cleaned)

    for match in reversed(numeric_matches):
        normalized = normalize_answer(match)

        if normalized is not None:
            return normalized

    return None


def extract_gsm8k_gold_answer(solution_text: str | None) -> str | None:
    if solution_text is None:
        return None

    matches = re.findall(
        r"####\s*([^\n]+)",
        solution_text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return extract_numeric_candidate(matches[-1])
