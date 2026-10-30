from __future__ import annotations

import pytest

from hiddenbench.benchmark import load_benchmark
from hiddenbench.models import ModelResponseParseError
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import AgentState, Profile, instantiate_agents, normalize_vote, run_evaluation, run_scenario


class FakeModel:
    model_name = "fake"

    def chat(self, messages):
        return "I am sharing my relevant facts."

    def chat_json(self, messages, schema):
        answer = schema["properties"]["vote"]["enum"][0]
        return {"vote": answer, "rationale": "The fake model picks a valid answer."}


@pytest.mark.parametrize("vote,options,expected", [
    ("B", ["Candidate A", "Candidate B"], "Candidate B"),
    ("b", ["Candidate A", "Candidate B"], "Candidate B"),
    ("B", ["West City", "East Town"], "B"),
    ("B", ["Candidate B", "Candidate b"], "B"),
    ("I choose B", ["Candidate A", "Candidate B"], "I choose B"),
    ("B or A", ["Candidate A", "Candidate B"], "B or A"),
    ("Candidate A", ["Candidate A", "Candidate B"], "Candidate A"),
    ("B", ["Warehouse A", "Warehouse B", "Warehouse C"], "Warehouse B"),
    ("B", ["Hospital A", "Hospital B", "Hospital C"], "Hospital B"),
    ("A", ["Site C", "Site A", "Site B"], "Site A"),
    ("B", ["Route A (Mountain Pass)", "Route B (Frozen Lake)"], "Route B (Frozen Lake)"),
    ("B", ["Option A: Biomedical startup", "Option B: AI hardware startup"], "Option B: AI hardware startup"),
    ("B", ["Deep Jungle Site (A)", "Riverbank Clearing (B)"], "Riverbank Clearing (B)"),
    ("B", ["Cold Storage B", "Lab 2 Bench", "Decontamination Room"], "B"),
    ("A", ["Alpha", "Bravo", "Charlie"], "A"),
    ("B", ["Warehouse B", "Hospital A"], "B"),
    ("B", ["Warehouse B", "Warehouse b"], "B"),
    ("D", ["Warehouse A", "Warehouse B"], "D"),
    ("Route C", ["Route A (Mountain Pass)", "Route B (Frozen Lake)", "Route C (Old Supply Road)"], "Route C (Old Supply Road)"),
    ("University Science Library", ["University Science Library (USL)", "Hilltop Chapel", "Town Hall"], "University Science Library (USL)"),
    ("USL", ["University Science Library (USL)", "Hilltop Chapel", "Town Hall"], "University Science Library (USL)"),
    ("town hall", ["University Science Library (USL)", "Hilltop Chapel", "Town Hall"], "Town Hall"),
    ("Route A", ["Route A (North)", "Route A (South)"], "Route A"),
    ("USL", ["University Science Library (USL)", "Urban Storage Lab (USL)"], "USL"),
    ("Library", ["University Science Library (USL)", "Hilltop Chapel", "Town Hall"], "Library"),
    ("Mountain Pass", ["Route A (Mountain Pass)", "Route B (Frozen Lake)"], "Mountain Pass"),
    ("I choose Route C", ["Route C (Old Supply Road)", "Route B (Frozen Lake)"], "I choose Route C"),
    ("Option C", ["Option A: Biomedical startup", "Option B: AI hardware startup", "Option C: Logistics software company"], "Option C: Logistics software company"),
    ("Option A", ["Option A: North", "Option A: South"], "Option A"),
    ("Option", ["Option A: Biomedical startup", "Option B: AI hardware startup"], "Option"),
    ("Logistics software company", ["Option A: Biomedical startup", "Option C: Logistics software company"], "Logistics software company"),
    ("Beta", ["Conference Room Alpha", "Conference Room Beta", "Conference Room Gamma"], "Conference Room Beta"),
    ("beta", ["Lab Alpha", "Lab Beta", "Lab Gamma"], "Lab Beta"),
    ("Bravo", ["Station Charlie", "Station Alpha", "Station Bravo"], "Station Bravo"),
    ("Bravo", ["Alpha Storage", "Bravo Storage", "Charlie Storage"], "Bravo Storage"),
    ("Beta", ["Lab Beta", "Shelter Alpha"], "Beta"),
    ("Beta", ["Lab Beta", "Lab beta"], "Beta"),
    ("B", ["Conference Room Alpha", "Conference Room Beta"], "B"),
    ("I vote Beta", ["Conference Room Alpha", "Conference Room Beta"], "I vote Beta"),
])
def test_vote_normalization_is_unambiguous(vote, options, expected):
    assert normalize_vote(vote, options) == expected


@pytest.mark.parametrize("raw_vote,options,expected", [
    ("B", ["Candidate A", "Candidate B"], "Candidate B"),
    ("Route C", ["Route A (Mountain Pass)", "Route C (Old Supply Road)"], "Route C (Old Supply Road)"),
    ("University Science Library", ["University Science Library (USL)", "Town Hall"], "University Science Library (USL)"),
    ("Option C", ["Option A: Biomedical startup", "Option C: Logistics software company"], "Option C: Logistics software company"),
    ("Beta", ["Conference Room Alpha", "Conference Room Beta", "Conference Room Gamma"], "Conference Room Beta"),
])
def test_vote_keeps_raw_alias_and_does_not_retry_or_change_history(raw_vote, options, expected):
    class LetterModel:
        def chat_json(self, messages, schema):
            return {"vote": raw_vote, "rationale": "Evidence supports this option."}

    history = [{"role": "system", "content": "Facts."}]
    agent = AgentState("Person 1", LetterModel(), history)
    result = agent.vote("Vote now", options)
    assert result["vote"] == expected
    assert result["vote_attempts"][0]["raw_vote"] == raw_vote
    assert len(result["vote_attempts"]) == 1
    assert history == [{"role": "system", "content": "Facts."}]


def test_real_benchmark_explicit_labels_are_supported():
    tasks = load_benchmark(path="data/benchmark.json")
    # Expected aliases are specified independently of the parser's matching rules.
    conventions = {
        "Candidate": "{prefix} {letter}",
        "Hospital": "{prefix} {letter}",
        "Warehouse": "{prefix} {letter}",
        "Restaurant": "{prefix} {letter}",
        "Site": "{prefix} {letter}",
    }
    for prefix, template in conventions.items():
        options = next(task.possible_answers for task in tasks if task.possible_answers[0] == f"{prefix} A")
        for letter in "ABC":
            assert normalize_vote(letter, options) == template.format(prefix=prefix, letter=letter)


def test_real_benchmark_named_aliases_cover_described_options():
    tasks = load_benchmark(path="data/benchmark.json")
    expected_aliases = {
        "University Science Library (USL)": ["University Science Library", "USL"],
        "Deep Jungle Site (A)": ["Deep Jungle Site", "A"],
        "Riverbank Clearing (B)": ["Riverbank Clearing", "B"],
        "Hilltop Overlook (C)": ["Hilltop Overlook", "C"],
        "Plateau Camp (D)": ["Plateau Camp", "D"],
        "Route A (Mountain Pass)": ["Route A", "A"],
        "Route B (Frozen Lake)": ["Route B", "B"],
        "Route C (Old Supply Road)": ["Route C", "C"],
        "Site A (Open field with tent)": ["Site A", "A"],
        "Site B (Community sports center)": ["Site B", "B"],
        "Site C (Hilltop park)": ["Site C", "C"],
        "Site D (Library)": ["Site D", "D"],
        "Area A (the Oak Woods)": ["Area A", "A"],
        "Area B (the Wetland)": ["Area B", "B"],
        "Area C (the Meadow)": ["Area C", "C"],
        "Option A: Biomedical startup": ["Option A", "A"],
        "Option B: AI hardware startup": ["Option B", "B"],
        "Option C: Logistics software company": ["Option C", "C"],
    }
    seen = set()
    assert len(tasks) == 65
    for task in tasks:
        for answer in task.possible_answers:
            assert normalize_vote(answer, task.possible_answers) == answer
            for alias in expected_aliases.get(answer, []):
                assert normalize_vote(alias, task.possible_answers) == answer
                seen.add(answer)
    assert seen == set(expected_aliases)


@pytest.mark.parametrize("options", [
    ["Lab Alpha", "Lab Beta", "Lab Gamma"],
    ["Shelter Alpha", "Shelter Beta", "Shelter Gamma"],
    ["Datacenter Alpha", "Datacenter Beta", "Datacenter Gamma"],
    ["Conference Room Alpha", "Conference Room Beta", "Conference Room Gamma"],
    ["Station Alpha", "Station Bravo", "Station Charlie", "Station Delta"],
    ["LZ Alpha", "LZ Bravo", "LZ Charlie"],
    ["Alpha Storage", "Bravo Storage", "Charlie Storage"],
])
def test_real_benchmark_uniform_named_labels(options):
    tasks = load_benchmark(path="data/benchmark.json")
    assert any(task.possible_answers == options for task in tasks)
    names = {"Alpha", "Beta", "Gamma", "Delta", "Bravo", "Charlie"}
    for answer in options:
        label = next(word for word in answer.split() if word in names)
        assert normalize_vote(label, options) == answer


def test_hidden_scenario_completes_with_short_room_votes():
    task = next(task for task in load_benchmark(path="data/benchmark.json")
                if task.name == "secure_meeting_room_decision")

    class ShortRoomModel(FakeModel):
        def chat_json(self, messages, schema):
            return {"vote": "Beta", "rationale": "I choose Beta."}

    result = run_scenario(task, ShortRoomModel(), Profile.HIDDEN, rounds=1, seed=2026)
    for vote in result["initial_votes"] + result["final_votes"]:
        assert vote["vote"] == "Conference Room Beta"
        assert vote["vote_attempts"][0]["raw_vote"] == "Beta"


def test_malformed_json_retries_with_original_text_and_is_auditable():
    class RecoveringModel:
        calls = 0

        def chat_json(self, messages, schema):
            self.calls += 1
            if self.calls == 1:
                raise ModelResponseParseError("Unbalanced JSON", '{"vote":')
            assert messages[-2] == {"role": "assistant", "content": '{"vote":'}
            return {"vote": "Candidate A", "rationale": "Evidence."}

    result = AgentState("Person 1", RecoveringModel(), []).vote("Vote", ["Candidate A"])
    assert result["vote"] == "Candidate A"
    assert len(result["vote_attempts"]) == 2
    assert result["vote_attempts"][0]["parse_error"] == "Unbalanced JSON"


def test_unmappable_vote_still_fails_after_three_attempts_and_logs_raw(caplog):
    class InvalidModel:
        calls = 0

        def chat_json(self, messages, schema):
            self.calls += 1
            return {"vote": "B", "rationale": "Evidence."}

    model = InvalidModel()
    with pytest.raises(ValueError, match="invalid vote 'B'"):
        AgentState("Person 1", model, []).vote("Vote", ["West City", "East Town"])
    assert model.calls == 3
    assert "raw_response=" in caplog.text
    assert "West City" in caplog.text


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


def test_run_evaluation_checkpoints_and_resumes_completed_runs():
    tasks = load_benchmark(path="data/benchmark_short.json")
    checkpoints = []
    first = run_evaluation(
        tasks,
        FakeModel(),
        Profile.HIDDEN,
        rounds=1,
        duplications=1,
        seed=2026,
        checkpoint_callback=lambda result: checkpoints.append(len(result["runs"])),
    )
    assert checkpoints == [1, 2, 3]
    assert [run["duplication_index"] for run in first["runs"]] == [0, 0, 0]

    class ExplodingModel:
        model_name = "fake"

        def chat(self, messages):
            raise AssertionError("A completed run should not execute again")

        def chat_json(self, messages, schema):
            raise AssertionError("A completed run should not execute again")

    resumed = run_evaluation(
        tasks,
        ExplodingModel(),
        Profile.HIDDEN,
        rounds=1,
        duplications=1,
        seed=2026,
        existing_runs=first["runs"],
    )
    assert resumed["runs"] == first["runs"]
