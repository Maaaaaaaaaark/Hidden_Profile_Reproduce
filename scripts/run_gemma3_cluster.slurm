#!/bin/bash
#SBATCH --job-name=hiddenbench-gemma3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --requeue

set -euo pipefail

# Submit this file through scripts/submit_gemma3_cluster.sh so Slurm logs are
# created inside the disposable experiment bundle.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BUNDLE_DIR="${PROJECT_DIR}/.cluster_bundle"
VENV_DIR="${BUNDLE_DIR}/venv"
RUN_DIR="${BUNDLE_DIR}/runs"
LOG_DIR="${BUNDLE_DIR}/logs"
CACHE_DIR="${BUNDLE_DIR}/cache"
TMP_DIR="${BUNDLE_DIR}/tmp"
MODEL_DIR="${BUNDLE_DIR}/ollama_models"

MODEL="gemma3:4b-it-qat"
ROUNDS=15
DUPLICATIONS=10
SEED=2026

# Violet's documented module combination for Python GPU jobs.
module purge
module load Python/3.11.7
module load CUDA/12.4.0
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$CACHE_DIR" "$TMP_DIR" "$MODEL_DIR"

# Keep model weights, Python packages, caches, temporary files, and outputs
# under the cloned repository. Removing the repository removes the experiment.
export OLLAMA_MODELS="$MODEL_DIR"
export XDG_CACHE_HOME="$CACHE_DIR"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export TMPDIR="$TMP_DIR"

# A per-job port avoids collisions on shared compute nodes.
OLLAMA_PORT=$((12000 + ${SLURM_JOB_ID:-1} % 20000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
BASE_URL="http://${OLLAMA_HOST}/v1"

if command -v ollama >/dev/null 2>&1; then
    OLLAMA_BIN="$(command -v ollama)"
elif [ -x "${PROJECT_DIR}/tools/ollama" ]; then
    OLLAMA_BIN="${PROJECT_DIR}/tools/ollama"
else
    echo "Ollama is not available. Load an Ollama module or place its executable at ${PROJECT_DIR}/tools/ollama." >&2
    exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -e "$PROJECT_DIR"

"$OLLAMA_BIN" serve > "${LOG_DIR}/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
    if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -fsS "${BASE_URL}/models" >/dev/null

if ! "$OLLAMA_BIN" show "$MODEL" >/dev/null 2>&1; then
    "$OLLAMA_BIN" pull "$MODEL"
fi

{
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "project_dir=$PROJECT_DIR"
    echo "model=$MODEL"
    echo "rounds=$ROUNDS"
    echo "duplications=$DUPLICATIONS"
    echo "seed=$SEED"
    echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    "$OLLAMA_BIN" --version
    echo "loaded_modules:"
    module -t list 2>&1
} > "${BUNDLE_DIR}/experiment_manifest.txt"

"$OLLAMA_BIN" show "$MODEL" > "${BUNDLE_DIR}/ollama_model_info.txt"
nvidia-smi > "${BUNDLE_DIR}/nvidia-smi.txt" 2>&1 || true

SOURCE_SNAPSHOT="${BUNDLE_DIR}/source_snapshot"
mkdir -p "$SOURCE_SNAPSHOT"
cp -R \
    "${PROJECT_DIR}/src" \
    "${PROJECT_DIR}/scripts" \
    "${PROJECT_DIR}/prompts" \
    "${PROJECT_DIR}/data" \
    "${PROJECT_DIR}/pyproject.toml" \
    "$SOURCE_SNAPSHOT/"
git -C "$PROJECT_DIR" status --short > "${BUNDLE_DIR}/git_status.txt" 2>/dev/null || true

COMMON_ARGS=(
    --benchmark "${PROJECT_DIR}/data/benchmark.json"
    --provider ollama
    --base-url "$BASE_URL"
    --model "$MODEL"
    --rounds "$ROUNDS"
    --duplications "$DUPLICATIONS"
    --seed "$SEED"
    --max-workers 1
    --resume
)

echo "Running Full Profile: 65 tasks x 10 sessions x 15 rounds"
srun --nodes=1 --ntasks=1 "${VENV_DIR}/bin/hiddenbench" eval \
    "${COMMON_ARGS[@]}" \
    --profile full \
    --output "${RUN_DIR}/gemma3_4b_full_d10.json"

echo "Running Hidden Profile: 65 tasks x 10 sessions x 15 rounds"
srun --nodes=1 --ntasks=1 "${VENV_DIR}/bin/hiddenbench" eval \
    "${COMMON_ARGS[@]}" \
    --profile hidden \
    --output "${RUN_DIR}/gemma3_4b_hidden_d10.json"

srun --nodes=1 --ntasks=1 "${VENV_DIR}/bin/hiddenbench" score \
    "${RUN_DIR}/gemma3_4b_hidden_d10.json" \
    --benchmark "${PROJECT_DIR}/data/benchmark.json" \
    --full-results "${RUN_DIR}/gemma3_4b_full_d10.json" \
    --json > "${RUN_DIR}/gemma3_4b_scores.json"

srun --nodes=1 --ntasks=1 "${VENV_DIR}/bin/hiddenbench" score \
    "${RUN_DIR}/gemma3_4b_hidden_d10.json" \
    --benchmark "${PROJECT_DIR}/data/benchmark.json" \
    --full-results "${RUN_DIR}/gemma3_4b_full_d10.json" \
    | tee "${RUN_DIR}/gemma3_4b_scores.txt"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${BUNDLE_DIR}/experiment_manifest.txt"
echo "Completed. All generated data is under ${BUNDLE_DIR}."
