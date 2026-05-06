from __future__ import annotations

import argparse
import json
from pathlib import Path

from hiddenbench.benchmark import DEFAULT_HF_REPO, find_task, load_benchmark
from hiddenbench.generation import generate_and_validate_tasks
from hiddenbench.metrics import (
    format_score_summary,
    iter_result_runs,
    load_results,
    result_model_name,
    result_profile_name,
    score_results,
)
from hiddenbench.models import build_model_client
from hiddenbench.simulator import Profile, run_evaluation


def _load_benchmark_from_args(args: argparse.Namespace):
    if getattr(args, "hf", None):
        return load_benchmark(source="hf", hf_repo=args.hf, split=args.split)
    return load_benchmark(source="local", path=args.benchmark)


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--benchmark", default="data/benchmark.json", help="Path to a local benchmark JSON file.")
    source.add_argument("--hf", help=f"Hugging Face dataset repo, for example {DEFAULT_HF_REPO}.")
    parser.add_argument("--split", default="train", help="Hugging Face split to load.")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="openai", help="openai, google, together, qwen, openrouter, or openai-compatible.")
    parser.add_argument("--model", default="gpt-4.1", help="Model name sent to the provider.")
    parser.add_argument("--base-url", help="Base URL for OpenAI-compatible providers.")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key.")
    parser.add_argument("--api-key", help="API key value. Prefer environment variables for shared scripts.")
    parser.add_argument("--temperature", type=float, help="Optional sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=3)


def _load_and_merge_results(paths: list[str]) -> dict:
    loaded = [load_results(path) for path in paths]
    if len(loaded) == 1:
        return loaded[0]
    models = sorted({model for result in loaded if (model := result_model_name(result))})
    profiles = sorted({profile for result in loaded if (profile := result_profile_name(result))})
    return {
        "metadata": {
            "model": ",".join(models) if models else None,
            "profile": ",".join(profiles) if profiles else None,
            "merged_from": paths,
        },
        "runs": [run for result in loaded for run in iter_result_runs(result)],
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    benchmark = _load_benchmark_from_args(args)
    task = find_task(benchmark, task_id=args.task_id, name=args.task_name)
    print(json.dumps(task.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    benchmark = _load_benchmark_from_args(args)
    client = build_model_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    def report_progress(completed: int, total: int, run: dict) -> None:
        print(f"[{completed}/{total}] {run['scenario']} ({run['profile']})")

    results = run_evaluation(
        benchmark,
        client,
        Profile(args.profile),
        rounds=args.rounds,
        duplications=args.duplications,
        seed=args.seed,
        max_workers=args.max_workers,
        extra_prompt=args.extra_prompt,
        special_agent_ratio=args.special_agent_ratio,
        progress_callback=report_progress,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results['runs'])} runs to {output_path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    benchmark = _load_benchmark_from_args(args)
    results = _load_and_merge_results(args.results)
    full_results = _load_and_merge_results(args.full_results) if args.full_results else None
    scores = score_results(results, benchmark, full_profile_results=full_results)
    if args.json:
        print(json.dumps(scores, indent=2, ensure_ascii=False))
    else:
        print(format_score_summary(scores))
    return 0


def cmd_generate_tasks(args: argparse.Namespace) -> int:
    client = build_model_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    results = generate_and_validate_tasks(
        client,
        num_candidates=args.num_candidates,
        output=args.output,
        full_min_accuracy=args.full_min_accuracy,
        hidden_max_accuracy=args.hidden_max_accuracy,
        validation_duplications=args.validation_duplications,
        seed=args.seed,
    )
    passed = sum(result.passed for result in results)
    print(f"Wrote {len(results)} generated candidates to {args.output}; {passed} passed validation.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate multi-agent LLMs on HiddenBench.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print a benchmark task.")
    _add_benchmark_args(inspect_parser)
    inspect_selector = inspect_parser.add_mutually_exclusive_group(required=True)
    inspect_selector.add_argument("--task-id")
    inspect_selector.add_argument("--task-name")
    inspect_parser.set_defaults(func=cmd_inspect)

    eval_parser = subparsers.add_parser("eval", help="Run HiddenBench evaluation.")
    _add_benchmark_args(eval_parser)
    _add_model_args(eval_parser)
    eval_parser.add_argument("--profile", choices=[Profile.HIDDEN.value, Profile.FULL.value], required=True)
    eval_parser.add_argument("--rounds", type=int, default=15)
    eval_parser.add_argument("--duplications", type=int, default=1)
    eval_parser.add_argument("--seed", type=int)
    eval_parser.add_argument("--max-workers", type=int, default=1)
    eval_parser.add_argument("--extra-prompt", default="")
    eval_parser.add_argument("--special-agent-ratio", type=float, default=1.0)
    eval_parser.add_argument("--output", required=True)
    eval_parser.set_defaults(func=cmd_eval)

    score_parser = subparsers.add_parser("score", help="Score an evaluation output JSON file.")
    score_parser.add_argument("results", nargs="+")
    _add_benchmark_args(score_parser)
    score_parser.add_argument("--full-results", nargs="+", help="Optional full-profile results for gap-to-full metrics.")
    score_parser.add_argument("--json", action="store_true", help="Print machine-readable scores.")
    score_parser.set_defaults(func=cmd_score)

    generate_parser = subparsers.add_parser("generate-tasks", help="Generate and validate Hidden Profile tasks.")
    _add_model_args(generate_parser)
    generate_parser.add_argument("--num-candidates", type=int, default=200)
    generate_parser.add_argument("--full-min-accuracy", type=float, default=0.8)
    generate_parser.add_argument("--hidden-max-accuracy", type=float, default=0.2)
    generate_parser.add_argument("--validation-duplications", type=int, default=10)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--output", default="runs/generated_tasks.json")
    generate_parser.set_defaults(func=cmd_generate_tasks)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
