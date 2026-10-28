# Gemma 3 4B Cluster Reproduction

This workflow reproduces the paper's main 65-task protocol with one intended change: every agent uses the local Ollama model `gemma3:4b-it-qat`.

## Fixed experimental settings

- Benchmark: `data/benchmark.json` (65 tasks)
- Profiles: Hidden and Full
- Discussion rounds: 15 for both profiles
- Sessions per task and profile: 10
- Seed: 2026
- Workers: 1
- Prompts, fact assignment, voting, final-vote scope, and scorer: unchanged

The paper does not report one provider-independent temperature or an immutable model snapshot. The script therefore leaves temperature unset, matching the repository's paper-style commands, and records the exact Ollama model information and software environment.

## Cluster requirements

- Slurm
- Violet modules `Python/3.11.7` and `CUDA/12.4.0`
- Ollama available on compute nodes
- `curl`
- One GPU; 8 GB or more VRAM is recommended for the 4B quantized model
- Network access for the first Python dependency installation and model pull

If compute nodes cannot access the internet, run the same script once on a network-enabled node or pre-populate `.cluster_bundle/venv` and `.cluster_bundle/ollama_models` from the login/data-transfer node.

The batch script follows Violet's documented Python GPU setup: it purges inherited modules, loads `Python/3.11.7` and `CUDA/12.4.0`, creates `.cluster_bundle/venv`, activates it, and installs HiddenBench inside that environment. Model inference runs in Ollama, while every HiddenBench Python process runs from the activated virtual environment.

If `ollama` is not already available as a command or module, place a cluster-compatible Ollama executable at `tools/ollama`. This binary is ignored by Git and is removed when the project directory is deleted.

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

Adjust the time, memory, and GPU resource lines in `scripts/run_gemma3_cluster.slurm` if your allocation requires different limits.

The job runs Full Profile, Hidden Profile, and scoring sequentially. Both evaluations checkpoint after every completed task-session run. If the job is preempted or reaches its time limit, submit the same command again; `--resume` skips completed runs after validating that the configuration is unchanged.

## Output

Everything created by the experiment is stored under `.cluster_bundle/`:

```text
.cluster_bundle/
  cache/
  experiment_manifest.txt
  logs/
  nvidia-smi.txt
  ollama_model_info.txt
  ollama_models/
  source_snapshot/
  runs/
    gemma3_4b_full_d10.json
    gemma3_4b_hidden_d10.json
    gemma3_4b_scores.json
    gemma3_4b_scores.txt
  tmp/
  venv/
```

The score file reports:

- `average_accuracy.pre`: Hidden Pre
- `average_accuracy.post`: Hidden Post
- `full_profile_pre_average_accuracy`: Full Pre
- `average_accuracy.improvement`: Hidden Post minus Hidden Pre
- `gap_to_full_profile_pre`: Hidden Post minus Full Pre

Copy the `runs/`, manifest, model information, GPU information, and `source_snapshot/` somewhere permanent before cleanup if the results are needed later.

## Cleanup

Deleting `.cluster_bundle/` removes the downloaded model, virtual environment, caches, logs, temporary data, and generated results. Deleting the cloned repository removes the source code as well. Check that the desired result files have been copied first.
