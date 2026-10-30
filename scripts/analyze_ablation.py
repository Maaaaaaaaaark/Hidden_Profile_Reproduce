from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from hiddenbench.metrics import _individual_accuracy, _majority_accuracy


CONDITIONS = {
    "A0": ("free", "ckpt_15"),
    "A1": ("share", "A1"),
    "A2": ("dissent", "A2"),
    "A3": ("both", "A3"),
    "A4": ("free", "A4"),
    "A5": ("share", "A5"),
    "A6": ("dissent", "A6"),
    "A7": ("both", "A7"),
}
STAGE1 = {"A0": "free", "A1": "share", "A2": "dissent", "A3": "both"}


def _read_runs(output_dir: Path) -> list[dict]:
    runs = []
    for path in output_dir.glob("*.jsonl"):
        if path.name == "failures.jsonl" or path.name.endswith('.calls.jsonl'):
            continue
        with path.open("r", encoding="utf-8") as handle:
            runs.extend(json.loads(line) for line in handle)
    return runs


def _sem(values: pd.Series) -> float:
    return values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def _score_rows(runs: list[dict]) -> pd.DataFrame:
    full = {
        (run["model"], str(run["task_id"]), run["dup"]): run["votes"]["Y_full"]
        for run in runs
        if run["trajectory"] == "full"
    }
    rows = []
    for condition, (trajectory, vote_key) in CONDITIONS.items():
        for run in runs:
            if run["trajectory"] != trajectory:
                continue
            y_full_votes = full[(run["model"], str(run["task_id"]), run["dup"])]
            for rule, metric in (
                ("average", _individual_accuracy),
                ("majority", _majority_accuracy),
            ):
                y_pre = metric(run["initial_votes"], run["correct_answer"])
                y_post = metric(run["votes"][vote_key], run["correct_answer"])
                y_full = metric(y_full_votes, run["correct_answer"])
                rows.append(
                    {
                        "model": run["model"],
                        "task_id": str(run["task_id"]),
                        "dup": run["dup"],
                        "condition": condition,
                        "rule": rule,
                        "y_pre": y_pre,
                        "y_post": y_post,
                        "change": y_post - y_pre,
                        "gap_to_full": y_post - y_full,
                    }
                )
    return pd.DataFrame(rows)


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scores.groupby(["model", "condition", "rule"], sort=True):
        task_means = group.groupby('task_id').mean(numeric_only=True)
        rows.append(
            {
                "model": keys[0],
                "condition": keys[1],
                "rule": keys[2],
                "n_runs": len(group),
                "y_pre": group["y_pre"].mean(),
                "y_pre_sem_task": _sem(task_means["y_pre"]),
                "y_pre_sem_run": _sem(group["y_pre"]),
                "y_post": group["y_post"].mean(),
                "y_post_sem_task": _sem(task_means["y_post"]),
                "y_post_sem_run": _sem(group["y_post"]),
                "change": group["change"].mean(),
                "change_sem_task": _sem(task_means["change"]),
                "change_sem_run": _sem(group["change"]),
                "gap_to_full": group["gap_to_full"].mean(),
                "gap_to_full_sem_task": _sem(task_means["gap_to_full"]),
                "gap_to_full_sem_run": _sem(group["gap_to_full"]),
            }
        )
    return pd.DataFrame(rows)


def _ypre_consistency(scores: pd.DataFrame) -> pd.DataFrame:
    direct = scores[scores["condition"].isin(STAGE1)].copy()
    direct["trajectory"] = direct["condition"].map(STAGE1)
    return (
        direct.groupby(["model", "trajectory", "rule"], as_index=False)["y_pre"]
        .mean()
        .pivot(index=["model", "rule"], columns="trajectory", values="y_pre")
        .reset_index()
    )


def _holm(p_values: list[float]) -> list[float]:
    order = sorted((i for i, p in enumerate(p_values) if np.isfinite(p)), key=lambda i: p_values[i])
    adjusted = np.full(len(p_values), np.nan)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _inference(scores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (model, rule), model_scores in scores.groupby(["model", "rule"], sort=True):
        task_means = model_scores.groupby(["condition", "task_id"], as_index=False)["y_post"].mean()
        a0 = task_means[task_means["condition"] == "A0"].set_index("task_id")["y_post"]
        model_rows = []
        p_values = []
        for condition in CONDITIONS:
            values = task_means[task_means["condition"] == condition].set_index("task_id")["y_post"]
            sample = values.to_numpy()
            boot = (np.mean(rng.choice(sample, size=(10000, len(sample)), replace=True), axis=1)
                    if len(sample) > 1 else np.array([np.nan]))
            paired = values.loc[a0.index] - a0
            p_value = (1.0 if condition == "A0" else
                       np.nan if len(paired) < 2 else
                       1.0 if np.allclose(paired, 0) else
                       wilcoxon(paired, zero_method="zsplit").pvalue)
            model_rows.append(
                {
                    "model": model,
                    "rule": rule,
                    "condition": condition,
                    "n_tasks": len(sample),
                    "y_post": sample.mean(),
                    "bootstrap_95_low": np.quantile(boot, 0.025),
                    "bootstrap_95_high": np.quantile(boot, 0.975),
                    "paired_difference_to_A0": paired.mean(),
                    "wilcoxon_p": p_value,
                }
            )
            if condition != 'A0':
                p_values.append(p_value)
        adjusted = _holm(p_values)
        adjusted = iter(adjusted)
        for row in model_rows:
            row["wilcoxon_holm_p"] = 1.0 if row['condition'] == 'A0' else next(adjusted)
            rows.append(row)
    return pd.DataFrame(rows)


def _validate_runs(output_dir, runs):
    for manifest_path in output_dir.glob('*.manifest.json'):
        manifest = json.loads(manifest_path.read_text())
        expected = {(task_id, dup, trajectory) for task_id in manifest['task_ids']
                    for dup in range(manifest['duplications'])
                    for trajectory in ('free', 'share', 'dissent', 'both', 'full')}
        keys = [(str(run['task_id']), run['dup'], run['trajectory']) for run in runs
                if run['model'] == manifest['model']]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('Results are incomplete or duplicated. Resume the run before final analysis.')


def _factorial(scores: pd.DataFrame) -> pd.DataFrame:
    means = scores.groupby(["model", "rule", "condition"], as_index=False)["y_post"].mean()
    rows = []
    for (model, rule), group in means.groupby(["model", "rule"], sort=True):
        values = group.set_index("condition")["y_post"]
        for direct, stage1 in STAGE1.items():
            decide = f"A{int(direct[1]) + 4}"
            rows.append(
                {
                    "model": model,
                    "rule": rule,
                    "stage1": stage1,
                    "direct_vote": values[direct],
                    "decide": values[decide],
                    "decide_minus_direct": values[decide] - values[direct],
                }
            )
    return pd.DataFrame(rows)


def _round_curve(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run["trajectory"] != "free":
            continue
        for checkpoint in (2, 3, 5, 10, 15, 20):
            if f"ckpt_{checkpoint}" not in run['votes']:
                continue
            votes = run["votes"][f"ckpt_{checkpoint}"]
            for rule, metric in (("average", _individual_accuracy), ("majority", _majority_accuracy)):
                rows.append(
                    {
                        "model": run["model"],
                        "task_id": str(run["task_id"]),
                        "dup": run["dup"],
                        "rule": rule,
                        "round": checkpoint,
                        "y_post": metric(votes, run["correct_answer"]),
                    }
                )
    return pd.DataFrame(rows).groupby(["model", "rule", "round"], as_index=False)["y_post"].mean()


def _to_markdown(table: pd.DataFrame) -> str:
    values = [[str(value) for value in row] for row in table.itertuples(index=False, name=None)]
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze HiddenBench ablation JSONL files.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    runs = _read_runs(output_dir)
    _validate_runs(output_dir, runs)
    scores = _score_rows(runs)
    summary = _summary(scores)
    consistency = _ypre_consistency(scores)
    inference = _inference(scores, np.random.default_rng(2026))
    factorial = _factorial(scores)
    curve = _round_curve(runs)

    tables = {
        "ablation_summary.csv": summary,
        "ypre_consistency.csv": consistency,
        "ablation_inference.csv": inference,
        "factorial_effects.csv": factorial,
        "round_curve.csv": curve,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    sections = [
        ("Ablation summary", summary),
        ("Y_pre consistency", consistency),
        ("Cluster bootstrap and paired tests", inference),
        ("Paired Decide additions (free uses 15 rounds; other prefixes use 2)", factorial),
        ("Round curve", curve),
    ]
    markdown = "# HiddenBench ablation analysis\n\n" + "\n\n".join(
        f"## {title}\n\n{_to_markdown(table)}" for title, table in sections
    )
    (output_dir / "ablation_analysis.md").write_text(markdown + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
