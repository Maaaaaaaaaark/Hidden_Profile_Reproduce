from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.models import ModelClient
from hiddenbench.prompts import load_prompts
from hiddenbench.simulator import Profile, collect_initial_votes


TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "shared_information": {"type": "array", "items": {"type": "string"}},
        "hidden_information": {"type": "array", "items": {"type": "string"}},
        "possible_answers": {"type": "array", "items": {"type": "string"}},
        "correct_answer": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "name",
        "description",
        "shared_information",
        "hidden_information",
        "possible_answers",
        "correct_answer",
    ],
    "additionalProperties": True,
}


@dataclass(frozen=True)
class ValidationResult:
    task: BenchmarkTask
    passed: bool
    full_accuracy_before: float
    hidden_accuracy_before: float

    def to_dict(self) -> dict[str, Any]:
        data = self.task.to_dict()
        data.update(
            {
                "validation_passed": self.passed,
                "full_accuracy_before": self.full_accuracy_before,
                "hidden_accuracy_before": self.hidden_accuracy_before,
            }
        )
        return data


def generate_task(model_client: ModelClient, *, task_id: int | str, prompt: str | None = None) -> BenchmarkTask:
    prompts = load_prompts()
    generation_prompt = prompt or prompts.generation_prompt
    if not generation_prompt:
        raise ValueError("No generation prompt available")
    response = model_client.chat_json([{"role": "system", "content": generation_prompt}], TASK_SCHEMA)
    response["id"] = task_id
    return BenchmarkTask.from_mapping(response)


def validate_task_candidate(
    task: BenchmarkTask,
    model_client: ModelClient,
    *,
    duplications: int = 10,
    full_min_accuracy: float = 0.8,
    hidden_max_accuracy: float = 0.2,
    seed: int | None = None,
) -> ValidationResult:
    prompts = load_prompts()
    full_scores: list[float] = []
    hidden_scores: list[float] = []
    for duplication in range(duplications):
        run_seed = None if seed is None else seed + duplication
        full_votes, _ = collect_initial_votes(task, model_client, Profile.FULL, prompts, seed=run_seed)
        hidden_votes, _ = collect_initial_votes(task, model_client, Profile.HIDDEN, prompts, seed=run_seed)
        full_scores.append(sum(vote["vote"] == task.correct_answer for vote in full_votes) / len(full_votes))
        hidden_scores.append(sum(vote["vote"] == task.correct_answer for vote in hidden_votes) / len(hidden_votes))

    full_accuracy = mean(full_scores)
    hidden_accuracy = mean(hidden_scores)
    return ValidationResult(
        task=task,
        passed=full_accuracy >= full_min_accuracy and hidden_accuracy <= hidden_max_accuracy,
        full_accuracy_before=full_accuracy,
        hidden_accuracy_before=hidden_accuracy,
    )


def generate_and_validate_tasks(
    model_client: ModelClient,
    *,
    num_candidates: int,
    output: str | Path,
    full_min_accuracy: float = 0.8,
    hidden_max_accuracy: float = 0.2,
    validation_duplications: int = 10,
    seed: int | None = None,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for index in range(num_candidates):
        task = generate_task(model_client, task_id=index + 1)
        result = validate_task_candidate(
            task,
            model_client,
            duplications=validation_duplications,
            full_min_accuracy=full_min_accuracy,
            hidden_max_accuracy=hidden_max_accuracy,
            seed=None if seed is None else seed + index * 1000,
        )
        results.append(result)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results
