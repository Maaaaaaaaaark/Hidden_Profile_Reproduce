from __future__ import annotations

import json
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.models import ModelClient
from hiddenbench.prompts import PromptSet, load_prompts, render_prompt
from hiddenbench.simulator import (
    AgentState,
    Profile,
    _scenario_seed,
    collect_final_votes,
    collect_initial_votes,
    instantiate_agents,
    speak_round,
)


EXCHANGE_SHARE = "Share 1-2 facts that you think are relevant to the decision."
EXCHANGE_DISSENT = "Give one reason why the current front-runner option may be incorrect."
EXCHANGE_BOTH = (
    "Share 1-2 facts that you think are relevant to the decision, and give one reason why the current "
    "front-runner option may be incorrect."
)
DECIDE = "Summarize the strongest evidence and the remaining uncertainty."

TRAJECTORIES = ("free", "share", "dissent", "both", "full")
EXCHANGE_INSTRUCTIONS = {
    "share": EXCHANGE_SHARE,
    "dissent": EXCHANGE_DISSENT,
    "both": EXCHANGE_BOTH,
}
CONDITIONS = {
    "share": ("A1", "A5"),
    "dissent": ("A2", "A6"),
    "both": ("A3", "A7"),
}


def fork_agents(agents: list[AgentState]) -> list[AgentState]:
    return [
        AgentState(
            name=agent.name,
            model_client=agent.model_client,
            history=list(agent.history),
            is_special=agent.is_special,
        )
        for agent in agents
    ]


def _initial_votes(
    agents: list[AgentState], task: BenchmarkTask, prompts: PromptSet
) -> list[dict[str, str]]:
    prompt = render_prompt(
        prompts.first_vote_prompt,
        {
            "group_discussion": (
                "No discussion has occurred yet. This is your initial vote based solely on the information available "
                "to you."
            ),
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    votes = []
    for agent in agents:
        response = agent.vote(prompt, task.possible_answers)
        votes.append({"agent": agent.name, **response})
    return votes


def _base_result(
    task: BenchmarkTask,
    model_client: ModelClient,
    duplication_index: int,
    seed: int | None,
    trajectory: str,
    assignments: list[dict[str, Any]],
    initial_votes: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "model": model_client.model_name,
        "task_id": task.id,
        "scenario": task.name,
        "dup": duplication_index,
        "seed": seed,
        "trajectory": trajectory,
        "num_agents": len(task.hidden_information),
        "correct_answer": task.correct_answer,
        "possible_answers": list(task.possible_answers),
        "fact_assignments": assignments,
        "initial_votes": initial_votes,
        "rounds": [],
        "votes": {},
    }


def run_trajectory(
    task: BenchmarkTask,
    model_client: ModelClient,
    duplication_index: int,
    seed: int | None,
    trajectory: str,
    prompts: PromptSet | None = None,
) -> dict[str, Any]:
    prompt_set = prompts or load_prompts()
    if trajectory == "full":
        initial_votes, assignments = collect_initial_votes(
            task, model_client, Profile.FULL, prompt_set, seed=seed
        )
        result = _base_result(
            task, model_client, duplication_index, seed, trajectory, assignments, initial_votes
        )
        result["votes"]["Y_full"] = initial_votes
        return result

    system_extra = "" if trajectory == "free" else EXCHANGE_INSTRUCTIONS[trajectory]
    agents, assignments = instantiate_agents(
        task,
        model_client,
        Profile.HIDDEN,
        prompt_set,
        random.Random(seed),
        extra_prompt=system_extra,
        special_agent_ratio=1.0,
    )
    initial_votes = _initial_votes(agents, task, prompt_set)
    result = _base_result(
        task, model_client, duplication_index, seed, trajectory, assignments, initial_votes
    )

    if trajectory == "free":
        checkpoints = {3, 5, 10, 15, 20}
        for round_index in range(20):
            messages = speak_round(agents, round_index, initial_votes, prompt_set, "")
            result["rounds"].append({"round": round_index + 1, "messages": messages})
            if round_index + 1 in checkpoints:
                result["votes"][f"ckpt_{round_index + 1}"] = collect_final_votes(
                    agents, prompt_set, task.possible_answers, messages
                )
            if round_index + 1 == 15:
                decide_agents = fork_agents(agents)
                decide_messages = speak_round(decide_agents, 15, initial_votes, prompt_set, DECIDE)
                result["rounds"].append(
                    {"round": 16, "stage": "decide", "messages": decide_messages}
                )
                result["votes"]["A4"] = collect_final_votes(
                    decide_agents, prompt_set, task.possible_answers, decide_messages
                )
        return result

    for round_index in range(2):
        messages = speak_round(agents, round_index, initial_votes, prompt_set, system_extra)
        result["rounds"].append({"round": round_index + 1, "messages": messages})
    direct_condition, decide_condition = CONDITIONS[trajectory]
    result["votes"][direct_condition] = collect_final_votes(
        agents, prompt_set, task.possible_answers, messages
    )
    decide_agents = fork_agents(agents)
    decide_messages = speak_round(decide_agents, 2, initial_votes, prompt_set, DECIDE)
    result["rounds"].append({"round": 3, "stage": "decide", "messages": decide_messages})
    result["votes"][decide_condition] = collect_final_votes(
        decide_agents, prompt_set, task.possible_answers, decide_messages
    )
    return result


def _completed_keys(output_path: Path) -> set[tuple[str, int, str]]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            completed.add((str(row["task_id"]), row["dup"], row["trajectory"]))
    return completed


def run_ablation(
    benchmark: list[BenchmarkTask],
    model_client: ModelClient,
    output_dir: str | Path,
    *,
    duplications: int = 10,
    seed: int | None = None,
    max_workers: int = 1,
    task_ids: set[str] | None = None,
    prompts: PromptSet | None = None,
    progress_callback: Callable[[int, int, tuple[str, int, str]], None] | None = None,
) -> None:
    prompt_set = prompts or load_prompts()
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{model_client.model_name}.jsonl"
    failure_path = output_directory / "failures.jsonl"
    completed_keys = _completed_keys(output_path)
    jobs = [
        (task_index, duplication_index, task, trajectory)
        for task_index, task in enumerate(benchmark)
        if task_ids is None or str(task.id) in task_ids
        for duplication_index in range(duplications)
        for trajectory in TRAJECTORIES
        if (str(task.id), duplication_index, trajectory) not in completed_keys
    ]

    def run_job(job: tuple[int, int, BenchmarkTask, str]):
        task_index, duplication_index, task, trajectory = job
        job_seed = _scenario_seed(seed, task_index, duplication_index)
        key = (str(task.id), duplication_index, trajectory)
        try:
            return key, run_trajectory(
                task, model_client, duplication_index, job_seed, trajectory, prompt_set
            ), None
        except Exception:
            return key, None, traceback.format_exc()

    def write_result(key, result, failure) -> None:
        if failure is None:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "traceback": failure}, ensure_ascii=False) + "\n")

    if max_workers <= 1:
        for index, job in enumerate(jobs, start=1):
            key, result, failure = run_job(job)
            write_result(key, result, failure)
            if progress_callback is not None:
                progress_callback(index, len(jobs), key)
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            key, result, failure = future.result()
            write_result(key, result, failure)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(jobs), key)
