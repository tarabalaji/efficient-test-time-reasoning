from __future__ import annotations

import argparse
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.extract_answers import extract_answer
from src.utils.config import load_config
from src.utils.io import append_jsonl, read_jsonl


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to a .env file.")

    return OpenAI(api_key=api_key)


def build_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.format(question=question)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def generate_single_response(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    latency_seconds = time.perf_counter() - start_time
    response_text = response.output_text.strip()

    input_tokens = None
    output_tokens = None
    total_tokens = None

    if response.usage is not None:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens

    return {
        "response_id": response.id,
        "response_text": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency_seconds, 4),
    }


def load_completed_attempts(
    generations_path: str,
) -> set[tuple[str, int]]:
    try:
        existing_records = read_jsonl(generations_path)
    except FileNotFoundError:
        return set()

    completed_attempts: set[tuple[str, int]] = set()

    for record in existing_records:
        question_id = record.get("question_id")
        attempt = record.get("attempt")

        if isinstance(question_id, str) and isinstance(attempt, int):
            completed_attempts.add((question_id, attempt))

    return completed_attempts


def select_questions(
    questions: list[dict[str, Any]],
    split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if split != "all":
        questions = [
            question for question in questions if question.get("split") == split
        ]

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")

        questions = questions[:limit]

    return questions


def main() -> None:
    arguments = parse_arguments()
    config = load_config()

    generation_config = config["generation"]
    paths = config["paths"]

    model = str(generation_config["model"])
    attempts_per_question = int(generation_config["attempts_per_question"])
    temperature = float(generation_config["temperature"])
    max_output_tokens = int(generation_config["max_output_tokens"])
    prompt_template = str(generation_config["prompt"])

    questions_path = str(paths["questions"])
    generations_path = str(paths["generations"])

    questions = read_jsonl(questions_path)
    questions = select_questions(
        questions=questions,
        split=arguments.split,
        limit=arguments.limit,
    )

    if not questions:
        raise ValueError("No questions matched the requested selection")

    client = create_client()
    completed_attempts = load_completed_attempts(generations_path)

    pending_jobs = [
        (question, attempt)
        for question in questions
        for attempt in range(1, attempts_per_question + 1)
        if (question["question_id"], attempt) not in completed_attempts
    ]

    print(f"Model: {model}")
    print(f"Questions selected: {len(questions)}")
    print(f"Attempts per question: {attempts_per_question}")
    print(f"Completed attempts found: {len(completed_attempts)}")
    print(f"Pending generations: {len(pending_jobs)}")

    if not pending_jobs:
        print("All requested generations are already complete.")
        return

    for question, attempt in tqdm(
        pending_jobs,
        desc="Generating responses",
    ):
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        gold_answer = str(question["gold_answer"])

        prompt = build_prompt(
            prompt_template=prompt_template,
            question=question_text,
        )

        try:
            generated = generate_single_response(
                client=client,
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            extracted_answer = extract_answer(generated["response_text"])

            record = {
                "question_id": question_id,
                "split": question["split"],
                "attempt": attempt,
                "model": model,
                "question": question_text,
                "gold_answer": gold_answer,
                "response_id": generated["response_id"],
                "response_text": generated["response_text"],
                "extracted_answer": extracted_answer,
                "correct": extracted_answer == gold_answer,
                "input_tokens": generated["input_tokens"],
                "output_tokens": generated["output_tokens"],
                "total_tokens": generated["total_tokens"],
                "latency_seconds": generated["latency_seconds"],
                "status": "success",
            }

        except Exception as error:
            record = {
                "question_id": question_id,
                "split": question["split"],
                "attempt": attempt,
                "model": model,
                "question": question_text,
                "gold_answer": gold_answer,
                "response_id": None,
                "response_text": None,
                "extracted_answer": None,
                "correct": False,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_seconds": None,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        append_jsonl(record, generations_path)

    print()
    print(f"Generation complete: {generations_path}")


if __name__ == "__main__":
    main()
