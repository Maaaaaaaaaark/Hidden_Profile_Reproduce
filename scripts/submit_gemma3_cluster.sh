#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${PROJECT_DIR}/.cluster_bundle"
mkdir -p "${BUNDLE_DIR}/logs"

missing=()
for name in VIOLET_PARTITION VIOLET_ACCOUNT VIOLET_QOS; do
    if [ -z "${!name:-}" ]; then
        missing+=("$name")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "Missing Violet account settings: ${missing[*]}" >&2
    echo "Run 'myinfo', then export VIOLET_PARTITION, VIOLET_ACCOUNT, and VIOLET_QOS with the reported values." >&2
    exit 2
fi

SBATCH_ARGS=(
    --partition="$VIOLET_PARTITION"
    --account="$VIOLET_ACCOUNT"
    --qos="$VIOLET_QOS"
    --requeue
    --output="${BUNDLE_DIR}/logs/slurm-%j.out"
    --error="${BUNDLE_DIR}/logs/slurm-%j.err"
)

if [ -n "${VIOLET_CONSTRAINT:-}" ]; then
    SBATCH_ARGS+=(--constraint="$VIOLET_CONSTRAINT")
fi

cd "$PROJECT_DIR"
sbatch "${SBATCH_ARGS[@]}" scripts/run_gemma3_cluster.slurm
