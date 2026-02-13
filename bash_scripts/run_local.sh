#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_FULL_DATASET_PATH="/home/simon/flashinfer-competition/mlsys26-contest"
FULL_DATASET_PATH="${FIB_FULL_DATASET_PATH:-$DEFAULT_FULL_DATASET_PATH}"
LOG_OUTPUT_DIR="${FIB_RUN_LOG_DIR:-$ROOT_DIR/logs}"

usage() {
  cat <<'USAGE'
Usage:
  bash_scripts/run_local.sh full
  bash_scripts/run_local.sh mini [mini_dataset_name]
  bash_scripts/run_local.sh full-dump-trace-md
  bash_scripts/run_local.sh mini-dump-trace-md [mini_dataset_name]

Notes:
  - mini_dataset_name defaults to <definition>_single from config.toml.
  - Full dataset path defaults to /home/simon/flashinfer-competition/mlsys26-contest.
    Override with FIB_FULL_DATASET_PATH.
  - Markdown logs are written to ./logs by default.
    Override with FIB_RUN_LOG_DIR.
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

run_with_dataset() {
  local dataset_path="$1"
  (
    cd "$ROOT_DIR"
    FIB_DATASET_PATH="$dataset_path" uv run python scripts/run_local.py
  )
}

get_trace_path_for_definition() {
  local dataset_path="$1"
  local def_name trace_path

  def_name=$(get_definition)
  trace_path=$(find "$dataset_path/traces" -type f -name "${def_name}.jsonl" 2>/dev/null | head -n 1)
  echo "$trace_path"
}

get_trace_line_count() {
  local dataset_path="$1"
  local trace_path

  trace_path=$(get_trace_path_for_definition "$dataset_path")
  if [[ -n "$trace_path" && -f "$trace_path" ]]; then
    wc -l < "$trace_path" | tr -d ' '
  else
    echo 0
  fi
}

dump_current_run_trace_markdown() {
  local dataset_path="$1"
  local before_count="$2"
  local def_name trace_path
  local timestamp_file output_file
  local run_entries_file
  local entry_count run_start_ts run_end_ts status_breakdown
  local table_rows

  def_name=$(get_definition)
  trace_path=$(get_trace_path_for_definition "$dataset_path")

  if [[ -z "$trace_path" ]]; then
    echo "No trace file found for definition '$def_name' under: $dataset_path/traces" >&2
    exit 1
  fi

  run_entries_file=$(mktemp)
  if [[ "$before_count" -eq 0 ]]; then
    cat "$trace_path" > "$run_entries_file"
  else
    sed -n "$((before_count + 1)),\$p" "$trace_path" > "$run_entries_file"
  fi

  if [[ ! -s "$run_entries_file" ]]; then
    echo "No new trace rows detected for this run; falling back to latest trace row."
    tail -n 1 "$trace_path" > "$run_entries_file"
  fi

  mkdir -p "$LOG_OUTPUT_DIR"

  if ! command -v jq >/dev/null 2>&1; then
    timestamp_file=$(date +%Y-%m-%dT%H-%M-%S)
    output_file="$LOG_OUTPUT_DIR/${def_name}_${timestamp_file}.md"
    {
      echo "# Benchmark Run Log"
      echo
      echo "- Definition: \`$def_name\`"
      echo "- Dataset path: \`$dataset_path\`"
      echo "- Trace file: \`$trace_path\`"
      echo "- New entries in current run: \`$(wc -l < "$run_entries_file" | tr -d ' ')\`"
      echo "- Timestamp: \`$timestamp_file\`"
      echo
      echo "## Current Run Trace JSONL"
      echo
      echo '```json'
      cat "$run_entries_file"
      echo '```'
    } > "$output_file"
    rm -f "$run_entries_file"
    echo "Saved markdown log: $output_file"
    return
  fi

  entry_count=$(wc -l < "$run_entries_file" | tr -d ' ')
  run_start_ts=$(head -n 1 "$run_entries_file" | jq -r '.evaluation.timestamp // "UNKNOWN"')
  run_end_ts=$(tail -n 1 "$run_entries_file" | jq -r '.evaluation.timestamp // "UNKNOWN"')
  status_breakdown=$(
    jq -r '.evaluation.status // "UNKNOWN"' "$run_entries_file" \
      | sort \
      | uniq -c \
      | sed -E 's/^ +//; s/ +/ /g'
  )
  table_rows=$(
    jq -sr '
      to_entries[]
      | [
          (.key + 1),
          (.value.workload.uuid // "UNKNOWN"),
          (.value.evaluation.status // "UNKNOWN"),
          (.value.evaluation.performance.latency_ms // "N/A"),
          (.value.evaluation.performance.speedup_factor // "N/A"),
          (.value.evaluation.correctness.max_absolute_error // "N/A"),
          (.value.evaluation.correctness.max_relative_error // "N/A"),
          (.value.evaluation.timestamp // "UNKNOWN")
        ]
      | @tsv
    ' "$run_entries_file"
  )

  timestamp_file=$(printf '%s' "$run_end_ts" | sed -E 's/[^0-9A-Za-z]+/-/g; s/^-+//; s/-+$//')
  if [[ -z "$timestamp_file" || "$timestamp_file" == "UNKNOWN" ]]; then
    timestamp_file=$(date +%Y-%m-%dT%H-%M-%S)
  fi
  output_file="$LOG_OUTPUT_DIR/${def_name}_${timestamp_file}.md"

  {
    echo "# Benchmark Run Log"
    echo
    echo "- Definition: \`$def_name\`"
    echo "- Dataset path: \`$dataset_path\`"
    echo "- Trace file: \`$trace_path\`"
    echo "- New entries in current run: \`$entry_count\`"
    echo "- Run timestamp start: \`$run_start_ts\`"
    echo "- Run timestamp end: \`$run_end_ts\`"
    echo
    echo "## Status Breakdown"
    echo
    printf '%s\n' "$status_breakdown" | awk '{print "- `"$2"`: "$1}'
    echo
    echo "## Workload Summary"
    echo
    echo "| # | Workload UUID | Status | Latency ms | Speedup | Max abs err | Max rel err | Timestamp |"
    echo "| --- | --- | --- | --- | --- | --- | --- | --- |"
    printf '%s\n' "$table_rows" | while IFS=$'\t' read -r idx uuid status latency speedup max_abs max_rel ts; do
      echo "| $idx | \`$uuid\` | \`$status\` | \`$latency\` | \`$speedup\` | \`$max_abs\` | \`$max_rel\` | \`$ts\` |"
    done
    echo
    echo "## Per-Entry Details"
    echo
    jq -sc 'to_entries[]' "$run_entries_file" | while IFS= read -r row; do
      idx=$(printf '%s' "$row" | jq -r '.key + 1')
      uuid=$(printf '%s' "$row" | jq -r '.value.workload.uuid // "UNKNOWN"')
      status=$(printf '%s' "$row" | jq -r '.value.evaluation.status // "UNKNOWN"')
      ts=$(printf '%s' "$row" | jq -r '.value.evaluation.timestamp // "UNKNOWN"')
      log_text=$(printf '%s' "$row" | jq -r '.value.evaluation.log // ""')

      echo "### Entry $idx - \`$uuid\` (\`$status\`)"
      echo
      echo "- Timestamp: \`$ts\`"
      echo
      echo "<details><summary>Evaluation Log</summary>"
      echo
      echo '```text'
      if [[ -z "$log_text" ]]; then
        echo "<empty>"
      else
        printf '%s\n' "$log_text"
      fi
      echo '```'
      echo
      echo "</details>"
      echo
      echo "<details><summary>Raw Entry JSON</summary>"
      echo
      echo '```json'
      printf '%s' "$row" | jq '.value'
      echo '```'
      echo
      echo "</details>"
      echo
    done
  } > "$output_file"

  rm -f "$run_entries_file"
  echo "Saved markdown log: $output_file"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local mode="$1"

  case "$mode" in
    full)
      run_with_dataset "$FULL_DATASET_PATH"
      ;;
    full-dump-trace-md)
      local full_before_count
      full_before_count=$(get_trace_line_count "$FULL_DATASET_PATH")
      run_with_dataset "$FULL_DATASET_PATH"
      dump_current_run_trace_markdown "$FULL_DATASET_PATH" "$full_before_count"
      ;;
    mini)
      run_with_dataset "$(get_mini_dataset_path "${2:-}")"
      ;;
    mini-dump-trace-md)
      local mini_path
      local mini_before_count
      mini_path=$(get_mini_dataset_path "${2:-}")
      mini_before_count=$(get_trace_line_count "$mini_path")
      run_with_dataset "$mini_path"
      dump_current_run_trace_markdown "$mini_path" "$mini_before_count"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
