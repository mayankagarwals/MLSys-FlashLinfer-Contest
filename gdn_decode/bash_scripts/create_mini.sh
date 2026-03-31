#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash_scripts/create_mini.sh <definition_name> <op_dir> [uuid] [output_dir]" >&2
  echo "Examples:" >&2
  echo "  bash_scripts/create_mini.sh gdn_decode_qk4_v8_d128_k_last gdn" >&2
  echo "  bash_scripts/create_mini.sh gdn_decode_qk4_v8_d128_k_last gdn 901e5104-dccb-4c3f-ae13-ef4d31a4d456" >&2
  exit 1
fi

NAME="$1"
OP="$2"
UUID="${3:-}"
SRC="${FIB_FULL_DATASET_PATH:-/home/simon/flashinfer-competition/mlsys26-contest}"
WORKLOAD_FILE="$SRC/workloads/$OP/$NAME.jsonl"

if [[ -n "$UUID" ]]; then
  DEFAULT_OUT="$ROOT_DIR/mini_datasets/${NAME}_${UUID}"
else
  DEFAULT_OUT="$ROOT_DIR/mini_datasets/${NAME}_single"
fi
ROOT="${4:-$DEFAULT_OUT}"

mkdir -p "$ROOT/definitions/$OP" "$ROOT/workloads/$OP"
cp "$SRC/definitions/$OP/$NAME.json" "$ROOT/definitions/$OP/"

if [[ -n "$UUID" ]]; then
  grep "\"$UUID\"" "$WORKLOAD_FILE" > "$ROOT/workloads/$OP/$NAME.jsonl"
  if [[ ! -s "$ROOT/workloads/$OP/$NAME.jsonl" ]]; then
    echo "Error: UUID '$UUID' not found in $WORKLOAD_FILE" >&2
    rm -rf "$ROOT"
    exit 1
  fi
else
  head -n 1 "$WORKLOAD_FILE" > "$ROOT/workloads/$OP/$NAME.jsonl"
fi

ln -sfn "$SRC/blob" "$ROOT/blob"

echo "Created mini dataset: $ROOT"
