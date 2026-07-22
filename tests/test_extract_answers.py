from src.extract_answers import extract_answer, extract_gsm8k_gold_answer


def test_extract_boxed_answer() -> None:
    response = r"The calculation gives 7, so the final answer is \boxed{7}."
    assert extract_answer(response) == "7"


def test_extract_final_answer_colon() -> None:
    response = "After completing the calculation, final answer: 42"
    assert extract_answer(response) == "42"


def test_extract_final_answer_is() -> None:
    response = "Therefore, the final answer is $1,250."
    assert extract_answer(response) == "1250"


def test_extract_answer_marker() -> None:
    response = "The result follows from the equation.\nAnswer: 18"
    assert extract_answer(response) == "18"


def test_extract_gsm8k_marker() -> None:
    response = "The student has 24 pencils.\n#### 24"
    assert extract_answer(response) == "24"


def test_extract_decimal_answer() -> None:
    response = "Final answer = 12.5"
    assert extract_answer(response) == "12.5"


def test_extract_negative_answer() -> None:
    response = "After subtracting, the final answer is -7."
    assert extract_answer(response) == "-7"


def test_extract_fraction_answer() -> None:
    response = "Final answer: 1/2"
    assert extract_answer(response) == "0.5"


def test_extract_percentage_answer() -> None:
    response = "The final answer is 25%."
    assert extract_answer(response) == "25"


def test_fallback_to_last_number() -> None:
    response = "There are 3 groups with 4 objects each, giving 12."
    assert extract_answer(response) == "12"


def test_fallback_uses_last_number() -> None:
    response = "First calculate 6 times 7. The result is 42."
    assert extract_answer(response) == "42"


def test_empty_response() -> None:
    assert extract_answer("") is None


def test_none_response() -> None:
    assert extract_answer(None) is None


def test_response_without_number() -> None:
    assert extract_answer("I cannot determine the answer.") is None


def test_extract_gsm8k_gold_answer() -> None:
    solution = (
        "Janet sells 16 eggs per day. She eats 3 and uses 4 for baking, "
        "leaving 9 eggs. She earns $2 per egg, so she earns 9 * 2 = 18.\n"
        "#### 18"
    )

    assert extract_gsm8k_gold_answer(solution) == "18"


def test_extract_gsm8k_gold_with_commas() -> None:
    solution = "The final total is 1,200.\n#### 1,200"
    assert extract_gsm8k_gold_answer(solution) == "1200"


def test_missing_gsm8k_marker() -> None:
    solution = "The final answer is 42."
    assert extract_gsm8k_gold_answer(solution) is None
