from src.normalize_answers import normalize_answer


def test_normalize_integer() -> None:
    assert normalize_answer("42") == "42"


def test_normalize_decimal_integer() -> None:
    assert normalize_answer("42.0") == "42"


def test_normalize_decimal() -> None:
    assert normalize_answer("12.50") == "12.5"


def test_normalize_currency() -> None:
    assert normalize_answer("$1,250") == "1250"


def test_normalize_latex_currency() -> None:
    assert normalize_answer(r"\$75") == "75"


def test_normalize_fraction() -> None:
    assert normalize_answer("1/2") == "0.5"


def test_normalize_negative_fraction() -> None:
    assert normalize_answer("-3/4") == "-0.75"


def test_normalize_percentage() -> None:
    assert normalize_answer("25%") == "25"


def test_normalize_boxed_answer() -> None:
    assert normalize_answer(r"\boxed{7}") == "7"


def test_normalize_negative_zero() -> None:
    assert normalize_answer("-0.0") == "0"


def test_normalize_none() -> None:
    assert normalize_answer(None) is None


def test_normalize_empty_string() -> None:
    assert normalize_answer("") is None


def test_reject_non_numeric_text() -> None:
    assert normalize_answer("forty-two") is None


def test_reject_zero_denominator() -> None:
    assert normalize_answer("1/0") is None
