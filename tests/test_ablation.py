from __future__ import annotations

import hashlib
import json
import random

import pytest

import hiddenbench.ablation as ablation
from hiddenbench.ablation import (
    DECIDE,
    EXCHANGE_BOTH,
    EXCHANGE_DISSENT,
    EXCHANGE_SHARE,
    fork_agents,
    run_trajectory,
)
from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.prompts import load_prompts, render_prompt
from hiddenbench.simulator import Profile, collect_final_votes, instantiate_agents, speak_round


class FakeModel:
    model_name = "fake"

    def __init__(self):
        self.chat_calls = []
        self.json_calls = []

    def chat(self, messages):
        self.chat_calls.append(list(messages))
        digest = hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        return f"response-{digest}"

    def chat_json(self, messages, schema):
        self.json_calls.append(list(messages))
        digest = hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        answers = schema["properties"]["vote"]["enum"]
        return {"vote": answers[int(digest, 16) % len(answers)], "rationale": "fixed rationale"}


@pytest.fixture
def task():
    return BenchmarkTask(
        id=1,
        name="tiny",
        description="Choose an option.",
        shared_information=["Shared fact."],
        hidden_information=["Hidden A.", "Hidden B."],
        possible_answers=["A", "B"],
        correct_answer="A",
    )


def _initial_votes(agents):
    return [
        {"agent": agent.name, "vote": "A", "rationale": "fixed rationale"}
        for agent in agents
    ]


def test_prompt_formats(task):
    prompts = load_prompts()
    agents, _ = instantiate_agents(task, FakeModel(), Profile.HIDDEN, prompts, random.Random(1))
    votes = _initial_votes(agents)
    first = speak_round(agents, 0, votes, prompts, "instruction")
    later = speak_round(agents, 1, votes, prompts, "instruction")
    expected_first = (
        "Your initial vote was: A\nYour initial rationale was: fixed rationale\n\nYou are the first to speak."
    )
    expected_second = (
        "Your initial vote was: A\nYour initial rationale was: fixed rationale\n\n"
        f"Previous messages from other people:\nPerson 1: {first[0]['response']}\n"
        "It's your turn to speak. instruction"
    )
    expected_later = render_prompt(
        "Previous messages from other people:\n%messages%\nIt's your turn to speak. %extra%",
        {"messages": f"Person 2: {first[1]['response']}", "extra": "instruction"},
    )
    assert [first[0]["prompt"], first[1]["prompt"], later[0]["prompt"]] == [
        expected_first,
        expected_second,
        expected_later,
    ]


def test_a0_votes_on_round_15(task):
    model = FakeModel()
    result = run_trajectory(task, model, 0, 7, "free")
    round_15 = next(row for row in result["rounds"] if row["round"] == 15)
    transcript = "\n".join(f"{message['agent']}: {message['response']}" for message in round_15["messages"])
    assert any(transcript in call[-1]["content"] for call in model.json_calls)


@pytest.mark.parametrize(
    ("trajectory", "instruction", "condition"),
    [("share", EXCHANGE_SHARE, "A1"), ("dissent", EXCHANGE_DISSENT, "A2"), ("both", EXCHANGE_BOTH, "A3")],
)
def test_a1_a3_exchange_round(task, trajectory, instruction, condition):
    model = FakeModel()
    result = run_trajectory(task, model, 0, 7, trajectory)
    round_1, round_2 = result["rounds"][:2]
    transcript = "\n".join(f"{message['agent']}: {message['response']}" for message in round_2["messages"])
    assert (
        condition in result["votes"]
        and instruction in model.chat_calls[0][0]["content"]
        and instruction not in round_1["messages"][0]["prompt"]
        and instruction in round_1["messages"][1]["prompt"]
        and any(transcript in call[-1]["content"] for call in model.json_calls)
    )


@pytest.mark.parametrize("trajectory", ["free", "share", "dissent", "both"])
def test_a4_a7_decide_round(task, trajectory):
    model = FakeModel()
    result = run_trajectory(task, model, 0, 7, trajectory)
    decide = next(row for row in result["rounds"] if row.get("stage") == "decide")
    condition = "A4" if trajectory == "free" else {"share": "A5", "dissent": "A6", "both": "A7"}[trajectory]
    exchange = "" if trajectory == "free" else ablation.EXCHANGE_INSTRUCTIONS[trajectory]
    transcript = "\n".join(f"{message['agent']}: {message['response']}" for message in decide["messages"])
    assert (
        condition in result["votes"]
        and all(
            DECIDE in message["prompt"] and (not exchange or exchange not in message["prompt"])
            for message in decide["messages"]
        )
        and (trajectory != "free" or DECIDE not in model.chat_calls[0][0]["content"])
        and any(transcript in call[-1]["content"] for call in model.json_calls)
    )


def test_fork_isolation(task):
    prompts = load_prompts()
    model = FakeModel()
    agents, _ = instantiate_agents(task, model, Profile.HIDDEN, prompts, random.Random(7))
    votes = _initial_votes(agents)
    for round_index in range(15):
        speak_round(agents, round_index, votes, prompts, "")
    histories = [list(agent.history) for agent in agents]
    fork = fork_agents(agents)
    speak_round(fork, 15, votes, prompts, DECIDE)
    continued = [speak_round(agents, index, votes, prompts, "") for index in range(15, 20)]

    baseline, _ = instantiate_agents(task, FakeModel(), Profile.HIDDEN, prompts, random.Random(7))
    baseline_rounds = [speak_round(baseline, index, votes, prompts, "") for index in range(20)]
    unchanged = [agent.history[: len(history)] for agent, history in zip(agents, histories, strict=True)]
    assert unchanged == histories and continued == baseline_rounds[15:]


def test_checkpoint_vote_does_not_change_history(task):
    prompts = load_prompts()
    agents, _ = instantiate_agents(task, FakeModel(), Profile.HIDDEN, prompts, random.Random(7))
    messages = speak_round(agents, 0, _initial_votes(agents), prompts, "")
    before = [list(agent.history) for agent in agents]
    collect_final_votes(agents, prompts, task.possible_answers, messages)
    assert [agent.history for agent in agents] == before


def test_same_seed_hidden_fact_assignments(task):
    hidden_assignments = [
        run_trajectory(task, FakeModel(), 0, 7, trajectory)["fact_assignments"]
        for trajectory in ("free", "share", "dissent", "both")
    ]
    hidden_order = [
        [entry["assigned_hidden_information"] for entry in assignment]
        for assignment in hidden_assignments
    ]
    full = run_trajectory(task, FakeModel(), 0, 7, "full")["fact_assignments"]
    full_order = [[item] for item in full[0]["assigned_hidden_information"]]
    assert hidden_order.count(hidden_order[0]) == len(hidden_order) and hidden_order[0] == full_order


def test_resume_skips_completed_key(task, tmp_path, monkeypatch):
    output = tmp_path / "fake.jsonl"
    called = []

    def fake_run(*args, **kwargs):
        called.append(args[4])
        return {"task_id": 1, "dup": 0, "trajectory": args[4]}

    monkeypatch.setattr(ablation, "run_trajectory", fake_run)
    ablation.run_ablation([task], FakeModel(), tmp_path, duplications=1)
    assert len(called) == 5
    called.clear()
    ablation.run_ablation([task], FakeModel(), tmp_path, duplications=1)
    assert called == []


def test_ablation_refuses_changed_settings(task, tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, 'run_trajectory', lambda *a: {'task_id': 1, 'dup': 0, 'trajectory': a[4]})
    ablation.run_ablation([task], FakeModel(), tmp_path, duplications=1, seed=1)
    with pytest.raises(ValueError, match='configuration changed'):
        ablation.run_ablation([task], FakeModel(), tmp_path, duplications=1, seed=2)


def test_ablation_failed_jobs_are_not_reported_as_success(task, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('synthetic model failure')
    monkeypatch.setattr(ablation, 'run_trajectory', fail)
    with pytest.raises(RuntimeError, match='5 trajectories failed'):
        ablation.run_ablation([task], FakeModel(), tmp_path, duplications=1)
    assert len((tmp_path / 'failures.jsonl').read_text().splitlines()) == 5


@pytest.mark.parametrize('model_name', [
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'Qwen/Qwen3.5-397B-A17B',
    'provider/organization/model',
])
def test_slash_model_names_write_flat_results_and_resume(task, tmp_path, monkeypatch, model_name):
    model = FakeModel()
    model.model_name = model_name
    called = []

    def fake_run(task, client, dup, seed, trajectory, prompts):
        assert client.model_name == model_name  # The API name is never sanitized.
        called.append(trajectory)
        return {'task_id': task.id, 'dup': dup, 'trajectory': trajectory, 'model': client.model_name}

    monkeypatch.setattr(ablation, 'run_trajectory', fake_run)
    ablation.run_ablation([task], model, tmp_path, duplications=1)
    files = list(tmp_path.glob('*.jsonl'))
    assert len(files) == 1
    assert files[0].name == model_name.replace('/', '_') + '.jsonl'
    assert len(files[0].read_text().splitlines()) == 5
    assert all(json.loads(line)['model'] == model_name for line in files[0].read_text().splitlines())
    assert not any(path.is_dir() for path in tmp_path.iterdir())
    called.clear()
    ablation.run_ablation([task], model, tmp_path, duplications=1)
    assert called == []


def test_filename_collision_is_rejected_before_any_model_work(task, tmp_path, monkeypatch):
    model = FakeModel()
    model.model_name = 'provider/model'
    called = []
    def fake_run(task, client, dup, seed, trajectory, prompts):
        called.append(trajectory)
        return {'task_id': task.id, 'dup': dup, 'trajectory': trajectory}
    monkeypatch.setattr(ablation, 'run_trajectory', fake_run)
    ablation.run_ablation([task], model, tmp_path, duplications=1)
    original = (tmp_path / 'provider_model.jsonl').read_bytes()
    called.clear()
    model.model_name = 'provider_model'
    with pytest.raises(ValueError, match='configuration changed'):
        ablation.run_ablation([task], model, tmp_path, duplications=1)
    assert called == []
    assert (tmp_path / 'provider_model.jsonl').read_bytes() == original
