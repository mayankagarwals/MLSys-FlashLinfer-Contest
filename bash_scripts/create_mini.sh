#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash_scripts/create_mini.sh <definition_name> <op_dir> [output_dir]" >&2
  echo "Example: bash_scripts/create_mini.sh gdn_decode_qk4_v8_d128_k_last gdn" >&2
  exit 1
fi

NAME="$1"
OP="$2"
ROOT="${3:-$ROOT_DIR/mini_datasets/${NAME}_single}"
SRC="${FIB_FULL_DATASET_PATH:-/home/simon/flashinfer-competition/mlsys26-contest}"

mkdir -p "$ROOT/definitions/$OP" "$ROOT/workloads/$OP"
cp "$SRC/definitions/$OP/$NAME.json" "$ROOT/definitions/$OP/"
head -n 1 "$SRC/workloads/$OP/$NAME.jsonl" > "$ROOT/workloads/$OP/$NAME.jsonl"
ln -sfn "$SRC/blob" "$ROOT/blob"

echo "Created mini dataset: $ROOT"
