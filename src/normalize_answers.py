from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None

    cleaned = answer.strip().lower()

    if not cleaned:
        return None

    cleaned = cleaned.replace("\\$", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.strip()

    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", cleaned)

    if boxed_match:
        cleaned = boxed_match.group(1).strip()

    fraction_match = re.fullmatch(r"-?\d+\s*/\s*-?\d+", cleaned)

    if fraction_match:
        numerator_text, denominator_text = re.split(r"\s*/\s*", cleaned)

        numerator = int(numerator_text)
        denominator = int(denominator_text)

        if denominator == 0:
            return None

        value = Decimal(Fraction(numerator, denominator).numerator) / Decimal(
            Fraction(numerator, denominator).denominator
        )

        return format_decimal(value)

    percentage_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", cleaned)

    if percentage_match:
        try:
            value = Decimal(percentage_match.group(1))
        except InvalidOperation:
            return None

        return format_decimal(value)

    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned)

    if not number_match:
        return None

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    return format_decimal(value)


def format_decimal(value: Decimal) -> str:
    formatted = format(value.normalize(), "f")

    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")

    if formatted == "-0":
        return "0"

    return formatted
