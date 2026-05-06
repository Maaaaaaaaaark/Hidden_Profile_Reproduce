from __future__ import annotations

from hiddenbench.benchmark import load_benchmark
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, instantiate_agents, run_scenario


class FakeModel:
    model_name = "fake"

    def chat(self, messages):
        return "I am sharing my relevant facts."

    def chat_json(self, messages, schema):
        answer = schema["properties"]["vote"]["enum"][0]
        return {"vote": answer, "rationale": "The fake model picks a valid answer."}


def test_prompt_rendering():
    rendered = render_prompt("Hello %name%", {"name": "HiddenBench"})
    assert rendered == "Hello HiddenBench"


def test_hidden_profile_assigns_one_hidden_fact_per_agent():
    task = load_benchmark(path="data/benchmark_short.json")[0]
    agents, assignments = instantiate_agents(task, FakeModel(), Profile.HIDDEN, load_prompts(), __import__("random").Random(1))
    assert len(agents) == len(task.hidden_information)
    for assignment in assignments:
        visible_hidden = [fact for fact in task.hidden_information if fact in assignment["visible_facts"]]
        assert len(visible_hidden) == 1


def test_full_profile_assigns_all_hidden_facts():
    task = load_benchmark(path="data/benchmark_short.json")[0]
    _, assignments = instantiate_agents(task, FakeModel(), Profile.FULL, load_prompts(), __import__("random").Random(1))
    for assignment in assignments:
        for hidden_fact in task.hidden_information:
            assert hidden_fact in assignment["visible_facts"]


def test_run_scenario_returns_valid_structure():
    task = load_benchmark(path="data/benchmark_short.json")[0]
    result = run_scenario(task, FakeModel(), Profile.HIDDEN, rounds=1, seed=42)
    assert result["scenario"] == task.name
    assert len(result["initial_votes"]) == len(task.hidden_information)
    assert len(result["final_votes"]) == len(task.hidden_information)
    assert all(vote["vote"] in task.possible_answers for vote in result["final_votes"])
