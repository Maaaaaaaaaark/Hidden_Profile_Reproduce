from __future__ import annotations

from hiddenbench import load_benchmark, run_evaluation, score_results


class AlwaysFirstAnswerModel:
    """Tiny demo model for integration tests and API exploration."""

    model_name = "always-first-answer"

    def chat(self, messages):
        return "I will share the most relevant evidence I have."

    def chat_json(self, messages, schema):
        first_answer = schema["properties"]["vote"]["enum"][0]
        return {"vote": first_answer, "rationale": "Demo client always selects the first valid answer."}


if __name__ == "__main__":
    tasks = load_benchmark(path="data/benchmark_short.json")
    results = run_evaluation(tasks, AlwaysFirstAnswerModel(), profile="hidden", rounds=1, seed=7)
    print(score_results(results, tasks)["average_accuracy"])
