#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: $0 INPUT_JSON OUTPUT_DIR N_SAMPLE" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${DAMN_MODEL_DIR:-${ROOT}/model}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f "${MODEL_DIR}/model.safetensors" || ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Model files not found; running setup..."
    "${ROOT}/setup.sh"
fi

cd "${ROOT}"
echo "Generating $3 sample(s)..."
"${PYTHON_BIN}" -m src.infer \
    --checkpoint "${MODEL_DIR}" \
    --input "$1" \
    --output-dir "$2" \
    --n-samples "$3" \
    --seed 42 \
    --temperature 1 \
    --top-p 0.95 \
    --top-k 0 \
    --max-notes 512 \
    --max-notes-per-onset 32 \
    --min-generation-end 240
echo "Done: $2"
