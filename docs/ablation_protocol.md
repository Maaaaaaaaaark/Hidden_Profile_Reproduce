# Ablation protocol differences

This implementation follows the repository's existing agent, voting, fact-allocation, prompt-rendering, and retry behavior. It differs from the authors' published implementation or adds analysis in these explicit ways:

1. The Exchange and Decide instruction text was written for this reproduction because the authors did not publish it.
2. Decide is implemented as one ordinary discussion round, interpreting the paper's “1 pass” description.
3. The Exchange instruction is supplied in both the system prompt and the user turn, as in the authors' `%extra%` placement. Initial votes therefore see it in the system prompt, and in A5–A7 the Decide round's system prompt still contains it.
4. Models are called through APIs; the authors did not publish their sampling parameters.
5. The user prompt uses “It's”, matching the released run files rather than the repository's former “It is”.
6. The pipeline adds checkpoint votes at rounds 2, 3, 5, 10, 15, and 20, task-clustered statistics, and round-matched comparisons. A0/A4 use a 15-round prefix; other pairs use a 2-round prefix, so their differences are not a pure 2x4 factorial effect.

## API runtime

- Completed trajectories are appended to `<model>.jsonl`; restarting the identical command skips them, and an interrupted trajectory restarts from its beginning.
- The manifest pins model/API settings, selected tasks, duplications, seed and prompts. Changed settings require a new output directory.
- Failed jobs go to `failures.jsonl` and make the process exit nonzero. Analysis rejects incomplete or duplicate trajectory sets.
- Each API attempt appends one line (status, returned model, finish reason, token usage; no message content) to `<model>.calls.jsonl`. SDK retries are disabled; `--max-retries` is the only retry layer. No API keys are logged.
- Invalid vote JSON is retried inside `vote()`; the number of attempts is stored with the vote.
- One-task smoke results are descriptive only.

## DeepSeek cluster run

The fixed first configuration is `deepseek-flash` on the official OpenAI-compatible
endpoint, thinking explicitly disabled, temperature 0.7, max_tokens 1024, one
concurrent trajectory, seed 2026 for task/fact ordering. The seed is not sent as
a model-generation seed. After the smoke run, check `actual_model` in the `.calls.jsonl` log to confirm which model the alias served, and record it with the run date, since aliases can change. Parameters unsupported by the service are not guaranteed to take effect.

This is our configuration, not a claim about the authors' undisclosed sampling settings.
No real DeepSeek requests were made during local verification.

1. Create a replacement API key if an earlier key was posted in chat. Keep it out of Git and logs.
2. On the cluster load Python/3.11.11-GCCcore-13.3.0, activate `.venv-api`, install `.[dev,analysis]`, and run pytest.
3. Set the key in the cluster Bash terminal without echoing it:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

4. From the repository root, submit a **CPU-only** smoke job. Do not use the old vLLM/GPU script:

```bash
mkdir -p .cluster_bundle/logs
HIDDENBENCH_MODE=smoke sbatch \
  --partition=researchshort \
  --account=pradeepresearch \
  --qos=research-1-qos \
  --chdir="$PWD" \
  --output="$PWD/.cluster_bundle/logs/deepseek-%j.out" \
  --error="$PWD/.cluster_bundle/logs/deepseek-%j.err" \
  scripts/run_deepseek_api.slurm
```

The key must be exported in this same shell before submission. The script
requests 2 CPUs, 8GB RAM, 12 hours, and zero GPUs. It uses the existing venv
without reinstalling packages. Compute-node access to api.deepseek.com and
account credit must be available; these have not been verified locally.

- `smoke`: task 1, one duplication, five trajectories. With four agents and no retries this makes **192 paid API requests**, including diagnostics and branch votes. The script includes A0 and Full pre, so no second baseline job is needed.
- `pilot`: tasks 1–5, two duplications; use only after smoke review.
- `full`: all 65 tasks, ten duplications; use only after pilot and spending review.

Set `HIDDENBENCH_MODE` to change modes. Each mode has a separate output directory
under `runs/ablation-deepseek-flash-<mode>-v2`. Rerun the same mode to resume;
do not launch two jobs into the same directory simultaneously. `--max-tokens`
is a per-request bound, not a dollar-spending cap.

Run `scripts/analyze_ablation.py` on the output directory for the accuracy tables. Reconcile spending with the platform's usage page.
