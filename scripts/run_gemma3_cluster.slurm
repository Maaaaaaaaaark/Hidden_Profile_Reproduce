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

MODEL="google/gemma-3-4b-it"
ROUNDS=15
DUPLICATIONS=10
SEED=2026

# Violet's documented module combination for Python GPU jobs.
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/12.4.0
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$CACHE_DIR" "$TMP_DIR"

# Keep Hugging Face weights, Python packages, caches, temporary files, and outputs
# under the cloned repository. Removing the repository removes the experiment.
export XDG_CACHE_HOME="$CACHE_DIR"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export HF_HOME="${CACHE_DIR}/huggingface"
export TMPDIR="$TMP_DIR"

# A per-job port avoids collisions on shared compute nodes.
VLLM_PORT=$((12000 + ${SLURM_JOB_ID:-1} % 20000))
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
if python -c "import dotenv, hiddenbench, openai" >/dev/null 2>&1; then
    python -m pip install --no-deps --no-build-isolation -e "$PROJECT_DIR"
else
    python -m pip install -e "$PROJECT_DIR"
fi

if ! python -c "import huggingface_hub, vllm" >/dev/null 2>&1; then
    python -m pip install --upgrade huggingface_hub vllm
fi

if ! hf auth whoami >/dev/null 2>&1; then
    echo "Hugging Face authentication is missing under ${HF_HOME}. Activate the project venv, export HF_HOME to this path, and run 'hf auth login'." >&2
    exit 1
fi

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    > "${LOG_DIR}/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 360); do
    if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM stopped before becoming ready. See ${LOG_DIR}/vllm-${SLURM_JOB_ID:-local}.log" >&2
        exit 1
    fi
    sleep 5
done
curl -fsS "${BASE_URL}/models" >/dev/null

{
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "project_dir=$PROJECT_DIR"
    echo "model=$MODEL"
    echo "rounds=$ROUNDS"
    echo "duplications=$DUPLICATIONS"
    echo "seed=$SEED"
    echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    python -c "import torch; print(f'torch={torch.__version__}')"
    python -c "import vllm; print(f'vllm={vllm.__version__}')"
    hf auth whoami
    echo "loaded_modules:"
    module -t list 2>&1
} > "${BUNDLE_DIR}/experiment_manifest.txt"

nvidia-smi > "${BUNDLE_DIR}/nvidia-smi.txt" 2>&1 || true
python -m pip freeze > "${BUNDLE_DIR}/python_packages.txt"

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
    --provider openai-compatible
    --base-url "$BASE_URL"
    --api-key vllm
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
