from __future__ import annotations

import pytest

from hiddenbench.benchmark import BenchmarkTask, load_benchmark, validate_task_mapping


def test_load_short_benchmark():
    tasks = load_benchmark(path="data/benchmark_short.json")
    assert len(tasks) == 3
    assert all(isinstance(task, BenchmarkTask) for task in tasks)
    assert tasks[0].name == "evacuation_west_city"
    assert tasks[0].correct_answer in tasks[0].possible_answers


def test_rejects_invalid_correct_answer():
    item = {
        "id": 1,
        "name": "bad",
        "description": "Bad task",
        "shared_information": ["Shared"],
        "hidden_information": ["Hidden"],
        "possible_answers": ["A", "B", "C"],
        "correct_answer": "D",
    }
    with pytest.raises(ValueError, match="correct_answer"):
        validate_task_mapping(item)
