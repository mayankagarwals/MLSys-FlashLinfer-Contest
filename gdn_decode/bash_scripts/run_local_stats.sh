#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_FULL_DATASET_PATH="/home/simon/flashinfer-competition/mlsys26-contest"
FULL_DATASET_PATH="${FIB_FULL_DATASET_PATH:-$DEFAULT_FULL_DATASET_PATH}"
DEFAULT_RUNS=10
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
TMP_DIR=""

usage() {
  cat <<'USAGE'
Usage:
  bash_scripts/run_local_stats.sh full [runs]
  bash_scripts/run_local_stats.sh mini [mini_dataset_name] [runs]
  bash_scripts/run_local_stats.sh [runs]

Examples:
  bash_scripts/run_local_stats.sh full
  bash_scripts/run_local_stats.sh full 10
  bash_scripts/run_local_stats.sh mini
  bash_scripts/run_local_stats.sh mini gdn_decode_qk4_v8_d128_k_last_single 10
  bash_scripts/run_local_stats.sh mini 10
  bash_scripts/run_local_stats.sh 10

Notes:
  - Default runs: 10
  - Full dataset path defaults to /home/simon/flashinfer-competition/mlsys26-contest
    Override with FIB_FULL_DATASET_PATH.
  - The script aggregates per-workload latency_ms from new trace rows after each run.
USAGE
}

get_definition() {
  local definition
  definition=$(sed -n 's/^definition = "\(.*\)"/\1/p' "$ROOT_DIR/config.toml" | head -n 1)
  if [[ -z "$definition" ]]; then
    echo "Could not read solution definition from config.toml" >&2
    exit 1
  fi
  echo "$definition"
}

get_mini_dataset_path() {
  local def_name mini_name mini_path
  def_name=$(get_definition)
  mini_name="${1:-${def_name}_single}"
  mini_path="$ROOT_DIR/mini_datasets/$mini_name"

  if [[ ! -d "$mini_path" ]]; then
    echo "Mini dataset not found: $mini_path" >&2
    echo "Create one with: bash_scripts/create_mini.sh <definition_name> <op_dir> [output_dir]" >&2
    exit 1
  fi

  echo "$mini_path"
}

find_trace_path() {
  local dataset_path="$1"
  local definition="$2"
  find "$dataset_path/traces" -type f -name "${definition}.jsonl" 2>/dev/null | head -n 1
}

line_count_or_zero() {
  local file="$1"
  if [[ -n "$file" && -f "$file" ]]; then
    wc -l < "$file" | tr -d ' '
  else
    echo 0
  fi
}

run_once() {
  local dataset_path="$1"
  mkdir -p "$UV_CACHE_DIR"
  (
    cd "$ROOT_DIR"
    UV_CACHE_DIR="$UV_CACHE_DIR" FIB_DATASET_PATH="$dataset_path" uv run python scripts/run_local.py
  )
}

extract_run_rows() {
  local run_id="$1"
  local run_jsonl="$2"
  python3 - "$run_id" "$run_jsonl" <<'PY'
import json
import sys

run_id = int(sys.argv[1])
path = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        workload = row.get("workload") or {}
        evaluation = row.get("evaluation") or {}
        performance = evaluation.get("performance") or {}

        workload_uuid = workload.get("uuid")
        if not workload_uuid:
            continue

        out = {
            "run": run_id,
            "workload_uuid": workload_uuid,
            "status": evaluation.get("status", "UNKNOWN"),
            "latency_ms": performance.get("latency_ms"),
        }
        print(json.dumps(out, sort_keys=True))
PY
}

print_stats() {
  local aggregate_jsonl="$1"
  local runs="$2"
  python3 - "$aggregate_jsonl" "$runs" <<'PY'
import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
runs = int(sys.argv[2])

by_workload = defaultdict(lambda: {"lat": [], "statuses": defaultdict(int), "runs": set()})

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        w = row["workload_uuid"]
        status = row.get("status", "UNKNOWN")
        latency = row.get("latency_ms")
        by_workload[w]["statuses"][status] += 1
        by_workload[w]["runs"].add(int(row["run"]))
        if status == "PASSED" and isinstance(latency, (int, float)):
            by_workload[w]["lat"].append(float(latency))

print()
print(f"Aggregated {runs} runs")
print("workload_uuid                          n_pass n_seen    mean_ms    std_ms    min_ms    max_ms")
for workload_uuid in sorted(by_workload):
    item = by_workload[workload_uuid]
    vals = item["lat"]
    n_pass = len(vals)
    n_seen = len(item["runs"])

    if n_pass > 0:
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if n_pass > 1 else 0.0
        min_v = min(vals)
        max_v = max(vals)
        print(
            f"{workload_uuid:36} {n_pass:6d} {n_seen:6d} "
            f"{mean:10.6f} {std:9.6f} {min_v:9.6f} {max_v:9.6f}"
        )
    else:
        print(
            f"{workload_uuid:36} {n_pass:6d} {n_seen:6d} "
            f"{float('nan'):10.6f} {float('nan'):9.6f} "
            f"{float('nan'):9.6f} {float('nan'):9.6f}"
        )

    non_passed = {k: v for k, v in item["statuses"].items() if k != "PASSED"}
    if non_passed:
        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(non_passed.items()))
        print(f"  non_passed: {breakdown}")
PY
}

cleanup() {
  if [[ -n "${TMP_DIR:-}" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}

on_interrupt() {
  echo
  echo "Interrupted by user; stopping remaining runs."
  exit 130
}

main() {
  trap cleanup EXIT
  trap on_interrupt INT TERM

  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  # Convenience shorthand: `run_local_stats.sh 10` => `full 10`
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    set -- full "$1"
  fi

  local mode="$1"
  local dataset_path=""
  local runs="$DEFAULT_RUNS"

  case "$mode" in
    full)
      runs="${2:-$DEFAULT_RUNS}"
      dataset_path="$FULL_DATASET_PATH"
      ;;
    mini)
      if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
        runs="$2"
        dataset_path="$(get_mini_dataset_path "")"
      else
        dataset_path="$(get_mini_dataset_path "${2:-}")"
        runs="${3:-$DEFAULT_RUNS}"
      fi
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac

  if ! [[ "$runs" =~ ^[0-9]+$ ]] || [[ "$runs" -le 0 ]]; then
    echo "runs must be a positive integer, got: $runs" >&2
    exit 1
  fi

  local definition
  definition=$(get_definition)

  TMP_DIR=$(mktemp -d)

  local aggregate_jsonl="$TMP_DIR/aggregate.jsonl"
  : > "$aggregate_jsonl"

  echo "Definition: $definition"
  echo "Dataset path: $dataset_path"
  echo "Runs: $runs"

  local run
  for run in $(seq 1 "$runs"); do
    local trace_path_before trace_path_after
    local before_count after_count run_slice

    trace_path_before=$(find_trace_path "$dataset_path" "$definition")
    before_count=$(line_count_or_zero "$trace_path_before")

    echo
    echo "[Run $run/$runs] Starting benchmark..."
    run_once "$dataset_path"

    trace_path_after=$(find_trace_path "$dataset_path" "$definition")
    if [[ -z "$trace_path_after" || ! -f "$trace_path_after" ]]; then
      echo "Trace file for definition '$definition' not found after run." >&2
      exit 1
    fi

    after_count=$(line_count_or_zero "$trace_path_after")
    if [[ "$after_count" -le "$before_count" ]]; then
      echo "No new trace rows detected for run $run (before=$before_count, after=$after_count)." >&2
      exit 1
    fi

    run_slice="$TMP_DIR/run_${run}.jsonl"
    sed -n "$((before_count + 1)),\$p" "$trace_path_after" > "$run_slice"
    extract_run_rows "$run" "$run_slice" >> "$aggregate_jsonl"
    echo "[Run $run/$runs] Captured $((after_count - before_count)) new trace rows."
  done

  print_stats "$aggregate_jsonl" "$runs"
}

main "$@"
