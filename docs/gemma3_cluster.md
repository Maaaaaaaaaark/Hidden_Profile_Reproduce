# Qwen3 4B Non-Thinking Cluster Reproduction

This workflow reproduces the paper's main 65-task protocol with one intended change: every agent uses the official Hugging Face model `Qwen/Qwen3-4B`, served locally by vLLM with thinking hard-disabled server-wide.

## Fixed experimental settings

- Benchmark: `data/benchmark.json` (65 tasks)
- Profiles: Hidden and Full
- Discussion rounds: 15 for both profiles
- Sessions per task and profile: 10
- Seed: 2026
- Workers: 1
- Prompts, fact assignment, voting, final-vote scope, and scorer: unchanged

Qwen's documented non-thinking settings are used: `enable_thinking=false`, temperature `0.7`, and top-p `0.8`. The script records these settings, the exact Hugging Face model name, vLLM version, PyTorch version, and software environment.

## Cluster requirements

- Slurm
- Violet modules `Python/3.11.11-GCCcore-13.3.0` and `CUDA/13.0.2`
- Hugging Face access to `Qwen/Qwen3-4B` (no gated Gemma license is required)
- vLLM installed in the project virtual environment
- `curl`
- One GPU; 16 GB or more VRAM is recommended for the official BF16 model
- Network access for the first Python dependency installation and model pull

If compute nodes cannot access the internet, pre-populate `.cluster_bundle/venv` and the Hugging Face cache from a network-enabled login/data-transfer node.

To prepare the Python environment on a network-enabled login node before submission:

```bash
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/13.0.2
python3 -m venv .cluster_bundle/venv
source .cluster_bundle/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install --upgrade huggingface_hub vllm
export HF_HOME="$PWD/.cluster_bundle/cache/huggingface"
hf auth login
hf auth whoami
hiddenbench --help
deactivate
```

The token is stored under `.cluster_bundle/cache/huggingface`, not in Git. The batch job activates this same environment. When dependencies are already present, it refreshes the editable project install without contacting the package index.

The batch script purges inherited modules, loads `Python/3.11.11-GCCcore-13.3.0` and `CUDA/13.0.2` (matching the vLLM-installed `cu130` PyTorch build), creates `.cluster_bundle/venv`, activates it, and runs both vLLM and HiddenBench from that environment.

## Obtain Violet account settings

On the login node, run:

```bash
myinfo
```

Export the three values reported for your account. For example only:

```bash
export VIOLET_PARTITION=researchlong
export VIOLET_ACCOUNT=is000
export VIOLET_QOS=is000qos
```

These values are user-specific and are deliberately not committed to the repository. The submit helper refuses to submit when any value is missing.

The job uses `#SBATCH --requeue`. Its per-run checkpoints make automatic restart safe on Violet's preemptible `researchlong` queue. To request a particular GPU or a non-preemptible node, optionally set a Violet constraint before submission:

```bash
export VIOLET_CONSTRAINT='nopreempt&l40s'
```

Omit this variable to let the scheduler choose any compatible GPU.

## Submit once

From the repository root:

```bash
bash scripts/submit_gemma3_cluster.sh
```

Before the full run, submit the smoke configuration once:

```bash
HIDDENBENCH_SMOKE=1 bash scripts/submit_gemma3_cluster.sh
```

This uses the three-task short benchmark, one session, and one discussion round. It verifies GPU access, Hugging Face authentication, vLLM startup, JSON voting, checkpointing, and scoring. Its files end in `_smoke` and do not collide with the full run. After it succeeds, submit the full experiment with the normal command above.

Adjust the time, memory, and GPU resource lines in `scripts/run_gemma3_cluster.slurm` if your allocation requires different limits.

The job runs Full Profile, Hidden Profile, and scoring sequentially. Both evaluations checkpoint after every completed task-session run. If the job is preempted or reaches its time limit, submit the same command again; `--resume` skips completed runs after validating that the configuration is unchanged.

### Vote-format repair (2026-10-28)

After job 341306 stopped on a bare `B` instead of `Candidate B`, the client now
accepts a single letter only when it uniquely matches an explicit `Candidate X`
option. It never maps letters to options by list position or guesses from the
rationale. Exact valid votes keep their original interpretation.

New votes store `vote_attempts`, including raw response text and the normalized
vote. Invalid JSON uses the existing maximum of three vote attempts; invalid
attempts log the request and raw response to Slurm stderr. Exhausted retries
still stop the job rather than dropping the session or inventing an answer.
These attempts are real model calls and must be included in any later cost audit.

This is an explicit parsing/retry amendment. Prompts, sampling settings, API JSON
mode, and scoring are unchanged; strict schema decoding is not newly enabled.
Completed checkpoints can be resumed. Earlier votes lack the new audit fields;
keep the original job logs and do not interpret missing fields as zero retries.

## Output

Everything created by the experiment is stored under `.cluster_bundle/`:

```text
.cluster_bundle/
  cache/
  experiment_manifest_d10.txt
  experiment_manifest_smoke.txt
  logs/
  nvidia-smi_d10.txt
  nvidia-smi_smoke.txt
  python_packages_d10.txt
  python_packages_smoke.txt
  source_snapshot/
  runs/
    qwen3_4b_full_d10.json
    qwen3_4b_full_smoke.json
    qwen3_4b_hidden_d10.json
    qwen3_4b_hidden_smoke.json
    qwen3_4b_scores_d10.json
    qwen3_4b_scores_d10.txt
    qwen3_4b_scores_smoke.json
    qwen3_4b_scores_smoke.txt
  tmp/
  venv/
```

The score file reports:

- `average_accuracy.pre`: Hidden Pre
- `average_accuracy.post`: Hidden Post
- `full_profile_pre_average_accuracy`: Full Pre
- `average_accuracy.improvement`: Hidden Post minus Hidden Pre
- `gap_to_full_profile_pre`: Hidden Post minus Full Pre

Copy the `runs/`, manifest, Python package list, GPU information, and `source_snapshot/` somewhere permanent before cleanup if the results are needed later.

## Cleanup

Deleting `.cluster_bundle/` removes the downloaded Hugging Face model, token cache, virtual environment, caches, logs, temporary data, and generated results. Deleting the cloned repository removes the source code as well. Check that the desired result files have been copied first.
