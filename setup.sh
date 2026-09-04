#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_ID="naivecynics/MIREX2026-DAMN"
DEFAULT_REVISION="4131195e4a9edf25da7c2f30307c106c17790164"

if [[ "$#" -gt 2 ]]; then
    echo "usage: $0 [HF_REPO_ID] [HF_REVISION]" >&2
    exit 2
fi

REPO_ID="${1:-$DEFAULT_REPO_ID}"
REVISION="${2:-$DEFAULT_REVISION}"

if ! command -v hf >/dev/null 2>&1; then
    echo "missing Hugging Face CLI; install it from https://hf.co/cli" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${DAMN_MODEL_DIR:-${ROOT}/model}"

for FILE in model.safetensors config.json; do
    hf download "${REPO_ID}" "${FILE}" \
        --revision "${REVISION}" \
        --local-dir "${MODEL_DIR}"
done

test -f "${MODEL_DIR}/model.safetensors"
test -f "${MODEL_DIR}/config.json"
echo "model ready: ${MODEL_DIR}"
