#!/bin/bash
#SBATCH --job-name=hiddenbench-deepseek
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_DIR="${HIDDENBENCH_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
module purge
module load Python/3.11.11-GCCcore-13.3.0
source .venv-api/bin/activate
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before submitting}"
export PYTHONUNBUFFERED=1

MODE="${HIDDENBENCH_MODE:-smoke}"
case "$MODE" in
    smoke) DUPLICATIONS=1; TASK_ARGS=(--task-ids 1) ;;
    pilot) DUPLICATIONS=2; TASK_ARGS=(--task-ids 1,2,3,4,5) ;;
    full) DUPLICATIONS=10; TASK_ARGS=() ;;
    *) echo "Unknown HIDDENBENCH_MODE: $MODE" >&2; exit 2 ;;
esac
OUTPUT_DIR="runs/ablation-deepseek-flash-${MODE}-v2"
echo "model=deepseek-flash thinking=disabled mode=$MODE duplications=$DUPLICATIONS"
echo "git_commit=$(git rev-parse HEAD) output=$OUTPUT_DIR"
# Fixed settings for every condition. No model weights or GPU are needed.
hiddenbench ablate \
    --benchmark data/benchmark.json \
    --provider openai-compatible \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --model deepseek-flash \
    --temperature 0.7 \
    --max-tokens 1024 \
    --extra-body '{"thinking":{"type":"disabled"}}' \
    --seed 2026 \
    --duplications "$DUPLICATIONS" \
    --max-workers 1 \
    "${TASK_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR"
python scripts/analyze_ablation.py --output-dir "$OUTPUT_DIR"
echo "Completed: $OUTPUT_DIR"
