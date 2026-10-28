from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from hiddenbench.benchmark import BenchmarkTask
from hiddenbench.models import Message, ModelClient, ModelResponseParseError
from hiddenbench.prompts import PromptSet, load_prompts, render_prompt

logger = logging.getLogger(__name__)


def normalize_vote(vote: str, possible_answers: list[str]) -> str:
    """Resolve explicit letter labels, never an option's position or initial."""
    if vote in possible_answers or re.fullmatch(r"[A-Za-z]", vote) is None:
        return vote
    # All options must use the same label convention. A lone suffix (e.g.
    # "Cold Storage B" next to "Lab 2 Bench") is not a multiple-choice label.
    patterns = (
        r"(?P<prefix>.+?)\s+(?P<label>[A-Za-z])(?:\s*\([^()]*\)|:\s*.+)?",
        r".+\s+\((?P<label>[A-Za-z])\)",
    )
    for pattern in patterns:
        labels = [re.fullmatch(pattern, answer) for answer in possible_answers]
        if not labels or any(label is None for label in labels):
            continue
        if "prefix" in labels[0].re.groupindex:
            prefixes = {label.group("prefix").casefold() for label in labels}
            if len(prefixes) != 1:
                continue
        letters = [label.group("label").upper() for label in labels]
        if len(set(letters)) != len(letters):
            continue
        if vote.upper() in letters:
            return possible_answers[letters.index(vote.upper())]
    return vote


class Profile(StrEnum):
    HIDDEN = "hidden"
    FULL = "full"


@dataclass
class AgentState:
    name: str
    model_client: ModelClient
    history: list[Message]
    is_special: bool = False

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        response = self.model_client.chat(self.history)
        self.history.append({"role": "assistant", "content": str(response)})
        return str(response)

    def vote(self, message: str, possible_answers: list[str], retries: int = 3) -> dict[str, Any]:
        if retries < 1:
            raise ValueError("Vote retries must be at least one")
        schema = vote_json_schema(possible_answers)
        local_history = list(self.history)
        local_history.append({"role": "user", "content": message})
        attempts: list[dict[str, Any]] = []
        for attempt in range(retries):
            try:
                response = self.model_client.chat_json(local_history, schema)
            except ModelResponseParseError as exc:
                raw_response = exc.raw_response
                attempts.append({"raw_response": raw_response, "parse_error": str(exc)})
                vote = ""
            else:
                raw_response = response.get("raw_response", json.dumps(response, ensure_ascii=False))
                raw_vote = response.get("vote")
                vote = normalize_vote(raw_vote.strip(), possible_answers) if isinstance(raw_vote, str) else ""
                rationale = response.get("rationale")
                rationale = rationale.strip() if isinstance(rationale, str) else ""
                attempts.append({"raw_response": raw_response, "raw_vote": raw_vote, "normalized_vote": vote})
                if vote in possible_answers and rationale:
                    return {"vote": vote, "rationale": rationale, "vote_attempts": attempts}
            logger.warning(
                "Invalid vote: agent=%s attempt=%s/%s options=%r messages=%r raw_response=%r",
                self.name, attempt + 1, retries, possible_answers, local_history, raw_response,
            )
            local_history.append({"role": "assistant", "content": raw_response})
            local_history.append(
                {
                    "role": "user",
                    "content": f"The vote must be exactly one of {possible_answers}. Return valid JSON again.",
                }
            )
        raise ValueError(f"Model returned invalid vote {vote!r}; expected one of {possible_answers}")


def vote_json_schema(possible_answers: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "vote": {"type": "string", "enum": possible_answers},
            "rationale": {"type": "string"},
        },
        "required": ["vote", "rationale"],
        "additionalProperties": False,
    }


def format_facts(facts: list[str]) -> str:
    return "".join(f"Fact {index}: {fact}\n" for index, fact in enumerate(facts, start=1))


def format_discussion_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "No discussion has occurred."
    return "\n".join(f"{entry['agent']}: {entry['response']}" for entry in messages)


def instantiate_agents(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet,
    rng: random.Random,
    *,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
) -> tuple[list[AgentState], list[dict[str, Any]]]:
    profile = Profile(profile)
    hidden_info = list(task.hidden_information)
    rng.shuffle(hidden_info)
    num_agents = len(hidden_info)
    num_special_agents = min(num_agents, math.ceil(num_agents * special_agent_ratio))
    special_indices = set(rng.sample(range(num_agents), num_special_agents)) if num_special_agents else set()

    agents: list[AgentState] = []
    fact_assignments: list[dict[str, Any]] = []
    for index in range(num_agents):
        facts = list(task.shared_information)
        if profile == Profile.HIDDEN:
            facts.append(hidden_info[index])
            assigned_hidden = [hidden_info[index]]
        else:
            facts.extend(hidden_info)
            assigned_hidden = list(hidden_info)
        rng.shuffle(facts)
        system_prompt = render_prompt(
            prompts.system_prompt,
            {
                "description": task.description,
                "information": format_facts(facts),
                "extra": extra_prompt if index in special_indices else "",
            },
        )
        agent = AgentState(
            name=f"Person {index + 1}",
            model_client=model_client,
            history=[{"role": "system", "content": system_prompt}],
            is_special=index in special_indices,
        )
        agents.append(agent)
        fact_assignments.append(
            {
                "agent": agent.name,
                "is_special": agent.is_special,
                "visible_facts": list(facts),
                "assigned_hidden_information": assigned_hidden,
            }
        )
    return agents, fact_assignments


def collect_initial_votes(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet | None = None,
    *,
    seed: int | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_set = prompts or load_prompts()
    rng = random.Random(seed)
    agents, assignments = instantiate_agents(
        task,
        model_client,
        profile,
        prompt_set,
        rng,
        extra_prompt=extra_prompt,
        special_agent_ratio=special_agent_ratio,
    )
    vote_prompt = render_prompt(
        prompt_set.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    votes = []
    for agent in agents:
        response = agent.vote(vote_prompt, task.possible_answers)
        votes.append({"agent": agent.name, **response})
    return votes, assignments


def collect_final_votes(
    agents: list[AgentState],
    prompts: PromptSet,
    possible_answers: list[str],
    discussion_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_discussion = format_discussion_transcript(discussion_messages)
    vote_prompt = render_prompt(
        prompts.vote_prompt,
        {
            "group_discussion": group_discussion,
            "possible_answers": ", ".join(possible_answers),
        },
    )
    votes = []
    for agent in agents:
        response = agent.vote(vote_prompt, possible_answers)
        votes.append({"agent": agent.name, **response})
    return votes


def run_scenario(
    task: BenchmarkTask,
    model_client: ModelClient,
    profile: Profile | str,
    prompts: PromptSet | None = None,
    *,
    rounds: int = 15,
    seed: int | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
    final_vote_scope: str = "last_round",
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if final_vote_scope not in {"last_round", "full_transcript"}:
        raise ValueError("final_vote_scope must be 'last_round' or 'full_transcript'")

    profile = Profile(profile)
    prompt_set = prompts or load_prompts()
    rng = random.Random(seed)
    agents, assignments = instantiate_agents(
        task,
        model_client,
        profile,
        prompt_set,
        rng,
        extra_prompt=extra_prompt,
        special_agent_ratio=special_agent_ratio,
    )

    run_data: dict[str, Any] = {
        "task_id": task.id,
        "scenario": task.name,
        "profile": profile.value,
        "model": model_client.model_name,
        "num_agents": len(agents),
        "num_special": sum(agent.is_special for agent in agents),
        "correct_answer": task.correct_answer,
        "possible_answers": list(task.possible_answers),
        "fact_assignments": assignments,
        "initial_votes": [],
        "rounds": [],
        "discussion_transcript": [],
        "final_votes": [],
        "majority_vote": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }

    initial_vote_prompt = render_prompt(
        prompt_set.first_vote_prompt,
        {
            "group_discussion": "No discussion has occurred yet. This is your initial vote based solely on the information available to you.",
            "possible_answers": ", ".join(task.possible_answers),
        },
    )
    for agent in agents:
        response = agent.vote(initial_vote_prompt, task.possible_answers)
        run_data["initial_votes"].append({"agent": agent.name, **response})

    transcript: list[dict[str, Any]] = []
    for round_index in range(rounds):
        round_data = {"round": round_index + 1, "messages": [], "votes": []}
        if round_index == 0:
            previous_messages: list[str] = []
            for agent_index, agent in enumerate(agents):
                initial_vote = run_data["initial_votes"][agent_index]
                initial_prefix = (
                    f"Your initial vote was: {initial_vote['vote']}\n"
                    f"Your initial rationale was: {initial_vote['rationale']}\n\n"
                )
                if not previous_messages:
                    prompt = initial_prefix + prompt_set.first_user_prompt
                else:
                    prompt = initial_prefix + render_prompt(
                        prompt_set.user_prompt,
                        {
                            "messages": "\n".join(previous_messages),
                            "extra": extra_prompt if agent.is_special else "",
                        },
                    )
                response = agent.chat(prompt)
                previous_messages.append(f"{agent.name}: {response}")
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})
        else:
            for agent in agents:
                current_index = agents.index(agent)
                previous_messages = []
                for offset in range(1, len(agents)):
                    other = agents[(current_index + offset) % len(agents)]
                    previous_messages.append(f"{other.name}: {other.history[-1]['content']}")
                prompt = render_prompt(
                    prompt_set.user_prompt,
                    {
                        "messages": "\n".join(previous_messages),
                        "extra": extra_prompt if agent.is_special else "",
                    },
                )
                response = agent.chat(prompt)
                message = {"agent": agent.name, "prompt": prompt, "response": response}
                round_data["messages"].append(message)
                transcript.append({"round": round_index + 1, "agent": agent.name, "response": response})

        if round_index == rounds - 1:
            vote_messages = round_data["messages"] if final_vote_scope == "last_round" else transcript
            round_data["votes"] = collect_final_votes(agents, prompt_set, task.possible_answers, vote_messages)
        run_data["rounds"].append(round_data)

    run_data["discussion_transcript"] = transcript
    run_data["final_votes"] = run_data["rounds"][-1]["votes"]
    vote_counts = Counter(vote["vote"] for vote in run_data["final_votes"])
    if vote_counts:
        run_data["majority_vote"] = vote_counts.most_common(1)[0][0]
    return run_data


def _scenario_seed(base_seed: int | None, task_index: int, duplication_index: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + task_index * 10_000 + duplication_index


def run_evaluation(
    benchmark: list[BenchmarkTask],
    model_client: ModelClient,
    profile: Profile | str,
    *,
    rounds: int = 15,
    duplications: int = 1,
    seed: int | None = None,
    max_workers: int = 1,
    prompts: PromptSet | None = None,
    extra_prompt: str = "",
    special_agent_ratio: float = 1.0,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    existing_runs: list[dict[str, Any]] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = Profile(profile)
    prompt_set = prompts or load_prompts()
    all_jobs = [
        (task_index, duplication_index, task)
        for task_index, task in enumerate(benchmark)
        for duplication_index in range(duplications)
    ]

    resumed_runs = list(existing_runs or [])
    completed_keys = {
        (str(run.get("task_id")), int(run.get("duplication_index", -1)))
        for run in resumed_runs
        if run.get("task_id") is not None and run.get("duplication_index") is not None
    }
    jobs = [
        job
        for job in all_jobs
        if (str(job[2].id), job[1]) not in completed_keys
    ]

    metadata = {
        "benchmark": "HiddenBench",
        "profile": profile.value,
        "model": model_client.model_name,
        "rounds": rounds,
        "duplications": duplications,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata.update(metadata_extra or {})
    results: dict[str, Any] = {"metadata": metadata, "runs": resumed_runs}

    task_order = {str(task.id): index for index, task in enumerate(benchmark)}

    def sort_runs() -> None:
        results["runs"].sort(
            key=lambda run: (
                task_order.get(str(run.get("task_id")), len(task_order)),
                int(run.get("duplication_index", -1)),
            )
        )

    def record_run(run: dict[str, Any]) -> None:
        results["runs"].append(run)
        sort_runs()
        if checkpoint_callback is not None:
            checkpoint_callback(results)

    def run_job(job: tuple[int, int, BenchmarkTask]) -> dict[str, Any]:
        task_index, duplication_index, task = job
        run = run_scenario(
            task,
            model_client,
            profile,
            prompt_set,
            rounds=rounds,
            seed=_scenario_seed(seed, task_index, duplication_index),
            extra_prompt=extra_prompt,
            special_agent_ratio=special_agent_ratio,
        )
        run["duplication_index"] = duplication_index
        return run

    if max_workers <= 1:
        completed = len(resumed_runs)
        for job in jobs:
            run = run_job(job)
            record_run(run)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(all_jobs), run)
        return results

    completed = len(resumed_runs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            run = future.result()
            record_run(run)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(all_jobs), run)
    return results
