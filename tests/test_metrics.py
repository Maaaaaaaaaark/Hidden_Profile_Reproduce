from __future__ import annotations

from hiddenbench.benchmark import load_benchmark
from hiddenbench.metrics import score_results


def test_score_results_average_and_majority():
    task = load_benchmark(path="data/benchmark_short.json")[0]
    results = {
        "metadata": {"profile": "hidden", "model": "fake"},
        "runs": [
            {
                "task_id": task.id,
                "scenario": task.name,
                "initial_votes": [
                    {"vote": task.correct_answer},
                    {"vote": "East Town"},
                    {"vote": "East Town"},
                    {"vote": "East Town"},
                ],
                "final_votes": [
                    {"vote": task.correct_answer},
                    {"vote": task.correct_answer},
                    {"vote": task.correct_answer},
                    {"vote": "East Town"},
                ],
            }
        ],
    }
    scores = score_results(results, [task])
    assert scores["average_accuracy"]["pre"] == 0.25
    assert scores["average_accuracy"]["post"] == 0.75
    assert scores["majority_accuracy"]["pre"] == 0.0
    assert scores["majority_accuracy"]["post"] == 1.0
