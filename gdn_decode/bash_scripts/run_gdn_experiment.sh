#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

TARGET_FILE="solution/python/gdn_decode_cutedsl.py"
LEDGER_FILE="$ROOT_DIR/LOGS.md"
DEFAULT_FULL_DATASET_PATH="/home/simon/flashinfer-competition/mlsys26-contest"
FULL_DATASET_PATH="${FIB_FULL_DATASET_PATH:-$DEFAULT_FULL_DATASET_PATH}"
DEFAULT_DEFINITION="gdn_decode_qk4_v8_d128_k_last"
ALLOW_DIRTY="${FIB_GDN_ALLOW_DIRTY:-0}"

REVERT_DONE=0

usage() {
  cat <<'USAGE'
Usage:
  bash_scripts/run_gdn_experiment.sh "<hypothesis>" "<change_summary>" [decision]

Examples:
  bash_scripts/run_gdn_experiment.sh \
    "Reduce redundant sync cost in small-batch loop" \
    "Moved cp.async issue point before compute barrier for next tile"

  bash_scripts/run_gdn_experiment.sh \
    "Lower register pressure in main loop" \
    "Reused accumulator registers and simplified reduction dataflow" \
    "Reject (no speedup)"

Behavior:
  1) Requires local edits in solution/python/gdn_decode_cutedsl.py.
  2) Runs mini preflight benchmark (must pass to continue).
  3) Runs full benchmark with markdown dump.
  4) Appends one summary entry to LOGS.md.
  5) Reverts solution/python/gdn_decode_cutedsl.py to HEAD.

Optional environment variables:
  - FIB_GDN_ALLOW_DIRTY=1: skip strict worktree guard.
  - FIB_MINI_DATASET_NAME=<name>: override mini dataset name.
USAGE
}

cleanup() {
  if [[ "$REVERT_DONE" -eq 1 ]]; then
    return
  fi
  if ! git -C "$ROOT_DIR" diff --quiet -- "$TARGET_FILE"; then
    echo "Reverting $TARGET_FILE to HEAD..."
    git -C "$ROOT_DIR" checkout -- "$TARGET_FILE" || true
  fi
  REVERT_DONE=1
}
trap cleanup EXIT

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
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

get_trace_path_for_definition() {
  local dataset_path="$1"
  local definition="$2"
  find "$dataset_path/traces" -type f -name "${definition}.jsonl" 2>/dev/null | head -n 1
}

get_trace_line_count() {
  local dataset_path="$1"
  local definition="$2"
  local trace_path
  trace_path=$(get_trace_path_for_definition "$dataset_path" "$definition")
  if [[ -n "$trace_path" && -f "$trace_path" ]]; then
    wc -l < "$trace_path" | tr -d ' '
  else
    echo 0
  fi
}

get_mini_dataset_path() {
  local definition="$1"
  local mini_name="${2:-${definition}_single}"
  local mini_path="$ROOT_DIR/mini_datasets/$mini_name"
  if [[ ! -d "$mini_path" ]]; then
    echo "Mini dataset not found: $mini_path" >&2
    echo "Create it with: bash_scripts/create_mini.sh ${definition} gdn" >&2
    exit 1
  fi
  echo "$mini_path"
}

extract_new_entries() {
  local trace_path="$1"
  local before_count="$2"
  local out_file="$3"

  if [[ ! -f "$trace_path" ]]; then
    : > "$out_file"
    return
  fi

  if [[ "$before_count" -eq 0 ]]; then
    cat "$trace_path" > "$out_file"
  else
    sed -n "$((before_count + 1)),\$p" "$trace_path" > "$out_file"
  fi

  if [[ ! -s "$out_file" ]]; then
    tail -n 1 "$trace_path" > "$out_file"
  fi
}

summarize_entries_json() {
  local entries_file="$1"
  local out_json="$2"
  jq -sr '
    def num_array(a): a | map(select(type == "number"));

    . as $rows
    | (num_array([.[].evaluation.performance.latency_ms])) as $lat
    | (num_array([.[].evaluation.performance.speedup_factor])) as $spd
    | {
        entry_count: ($rows | length),
        statuses: (
          $rows
          | map(.evaluation.status // "UNKNOWN")
          | group_by(.)
          | map({status: .[0], count: length})
        ),
        all_passed: (
          if ($rows | length) == 0 then false
          else ($rows | all((.evaluation.status // "UNKNOWN") == "PASSED"))
          end
        ),
        run_start_ts: (
          if ($rows | length) == 0 then "UNKNOWN"
          else ($rows[0].evaluation.timestamp // "UNKNOWN")
          end
        ),
        run_end_ts: (
          if ($rows | length) == 0 then "UNKNOWN"
          else ($rows[-1].evaluation.timestamp // "UNKNOWN")
          end
        ),
        latency_mean_ms: (
          if ($lat | length) > 0 then ($lat | add / length) else null end
        ),
        speedup_mean: (
          if ($spd | length) > 0 then ($spd | add / length) else null end
        ),
        speedup_max: (
          if ($spd | length) > 0 then ($spd | max) else null end
        ),
        speedup_min: (
          if ($spd | length) > 0 then ($spd | min) else null end
        )
      }
  ' "$entries_file" > "$out_json"
}

fmt_num() {
  local value="$1"
  if [[ "$value" == "null" || -z "$value" ]]; then
    echo "N/A"
  else
    printf "%.9f" "$value"
  fi
}

status_breakdown_line() {
  local summary_json="$1"
  jq -r '
    if (.statuses | length) == 0 then
      "UNKNOWN: 0"
    else
      (.statuses | map("\(.status): \(.count)") | join(", "))
    end
  ' "$summary_json"
}

next_iteration_number() {
  local n
  n=$(grep -E '^## Iteration [0-9]+' "$LEDGER_FILE" 2>/dev/null | tail -n 1 | awk '{print $3}')
  if [[ -z "$n" ]]; then
    echo 1
  else
    echo $((n + 1))
  fi
}

init_ledger_if_missing() {
  if [[ -f "$LEDGER_FILE" ]]; then
    return
  fi

  cat > "$LEDGER_FILE" <<'EOF'
# GDN Decode Experiment Log

This ledger records ephemeral experiments on `solution/python/gdn_decode_cutedsl.py`.
Each experiment should run mini preflight + full benchmark, then revert kernel code.

EOF
}

assert_expected_worktree() {
  if [[ "$ALLOW_DIRTY" == "1" ]]; then
    echo "Skipping strict worktree guard (FIB_GDN_ALLOW_DIRTY=1)."
    return
  fi

  local unexpected=()
  local line path

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line:3}"

    if [[ "$path" == "$TARGET_FILE" || "$path" == "LOGS.md" ]]; then
      continue
    fi
    if [[ "$path" == logs/* || "$path" == ncu_logs/* ]]; then
      continue
    fi

    unexpected+=("$line")
  done < <(git -C "$ROOT_DIR" status --porcelain)

  if [[ "${#unexpected[@]}" -gt 0 ]]; then
    echo "Unexpected worktree changes detected; refusing to run experiment." >&2
    printf '  %s\n' "${unexpected[@]}" >&2
    echo "Allowed paths: $TARGET_FILE, LOGS.md, logs/*, ncu_logs/*" >&2
    exit 1
  fi
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi
  if [[ $# -lt 2 ]]; then
    usage
    exit 1
  fi

  local hypothesis="$1"
  local change_summary="$2"
  local decision="${3:-pending}"
  local definition
  local mini_dataset_name mini_dataset_path
  local full_trace_path mini_trace_path
  local full_before_count mini_before_count
  local mini_entries full_entries
  local mini_summary_json full_summary_json
  local mini_exit=0 full_exit=0
  local mini_ok="false" full_ran="false"
  local mini_status_breakdown full_status_breakdown
  local mini_entry_count full_entry_count
  local mini_all_passed full_all_passed
  local full_latency_mean full_speedup_mean full_speedup_best full_speedup_worst
  local mini_latency_mean mini_speedup_mean
  local full_run_start full_run_end
  local mini_run_start mini_run_end
  local full_log_md=""
  local iteration ts_now head_sha
  local mini_output=""
  local full_output=""

  require_cmd git
  require_cmd jq
  require_cmd sed
  require_cmd uv

  definition=$(get_definition)
  if [[ "$definition" != "$DEFAULT_DEFINITION" ]]; then
    echo "Expected definition '$DEFAULT_DEFINITION' but found '$definition' in config.toml." >&2
    exit 1
  fi

  if [[ ! -d "$FULL_DATASET_PATH" ]]; then
    echo "Full dataset path not found: $FULL_DATASET_PATH" >&2
    exit 1
  fi

  assert_expected_worktree

  if git -C "$ROOT_DIR" diff --quiet -- "$TARGET_FILE"; then
    echo "No local edits detected in $TARGET_FILE. Make a kernel change before running." >&2
    exit 1
  fi

  init_ledger_if_missing

  mini_dataset_name="${FIB_MINI_DATASET_NAME:-${definition}_single}"
  mini_dataset_path=$(get_mini_dataset_path "$definition" "$mini_dataset_name")

  full_trace_path=$(get_trace_path_for_definition "$FULL_DATASET_PATH" "$definition")
  full_before_count=$(get_trace_line_count "$FULL_DATASET_PATH" "$definition")
  mini_trace_path=$(get_trace_path_for_definition "$mini_dataset_path" "$definition")
  mini_before_count=$(get_trace_line_count "$mini_dataset_path" "$definition")

  mini_entries=$(mktemp)
  full_entries=$(mktemp)
  mini_summary_json=$(mktemp)
  full_summary_json=$(mktemp)
  trap 'rm -f -- "${mini_entries:-}" "${full_entries:-}" "${mini_summary_json:-}" "${full_summary_json:-}"; cleanup' EXIT

  echo "Running mini preflight benchmark..."
  set +e
  mini_output=$(cd "$ROOT_DIR" && bash_scripts/run_local.sh mini "$mini_dataset_name" 2>&1)
  mini_exit=$?
  set -e
  printf '%s\n' "$mini_output"

  mini_trace_path=$(get_trace_path_for_definition "$mini_dataset_path" "$definition")
  if [[ -n "$mini_trace_path" && -f "$mini_trace_path" ]]; then
    extract_new_entries "$mini_trace_path" "$mini_before_count" "$mini_entries"
    summarize_entries_json "$mini_entries" "$mini_summary_json"
    mini_status_breakdown=$(status_breakdown_line "$mini_summary_json")
    mini_entry_count=$(jq -r '.entry_count' "$mini_summary_json")
    mini_all_passed=$(jq -r '.all_passed' "$mini_summary_json")
    mini_latency_mean=$(fmt_num "$(jq -r '.latency_mean_ms' "$mini_summary_json")")
    mini_speedup_mean=$(fmt_num "$(jq -r '.speedup_mean' "$mini_summary_json")")
    mini_run_start=$(jq -r '.run_start_ts' "$mini_summary_json")
    mini_run_end=$(jq -r '.run_end_ts' "$mini_summary_json")
  else
    mini_status_breakdown="UNKNOWN: 0"
    mini_entry_count=0
    mini_all_passed="false"
    mini_latency_mean="N/A"
    mini_speedup_mean="N/A"
    mini_run_start="UNKNOWN"
    mini_run_end="UNKNOWN"
  fi

  if [[ "$mini_exit" -eq 0 && "$mini_all_passed" == "true" ]]; then
    mini_ok="true"
  fi

  full_status_breakdown="SKIPPED"
  full_entry_count=0
  full_all_passed="false"
  full_latency_mean="N/A"
  full_speedup_mean="N/A"
  full_speedup_best="N/A"
  full_speedup_worst="N/A"
  full_run_start="UNKNOWN"
  full_run_end="UNKNOWN"

  if [[ "$mini_ok" == "true" ]]; then
    echo "Running full benchmark with markdown trace dump..."
    set +e
    full_output=$(cd "$ROOT_DIR" && bash_scripts/run_local.sh full-dump-trace-md 2>&1)
    full_exit=$?
    set -e
    printf '%s\n' "$full_output"

    full_log_md=$(printf '%s\n' "$full_output" | sed -n 's/^Saved markdown log: //p' | tail -n 1)
    full_ran="true"

    full_trace_path=$(get_trace_path_for_definition "$FULL_DATASET_PATH" "$definition")
    if [[ -n "$full_trace_path" && -f "$full_trace_path" ]]; then
      extract_new_entries "$full_trace_path" "$full_before_count" "$full_entries"
      summarize_entries_json "$full_entries" "$full_summary_json"
      full_status_breakdown=$(status_breakdown_line "$full_summary_json")
      full_entry_count=$(jq -r '.entry_count' "$full_summary_json")
      full_all_passed=$(jq -r '.all_passed' "$full_summary_json")
      full_latency_mean=$(fmt_num "$(jq -r '.latency_mean_ms' "$full_summary_json")")
      full_speedup_mean=$(fmt_num "$(jq -r '.speedup_mean' "$full_summary_json")")
      full_speedup_best=$(fmt_num "$(jq -r '.speedup_max' "$full_summary_json")")
      full_speedup_worst=$(fmt_num "$(jq -r '.speedup_min' "$full_summary_json")")
      full_run_start=$(jq -r '.run_start_ts' "$full_summary_json")
      full_run_end=$(jq -r '.run_end_ts' "$full_summary_json")
    else
      full_status_breakdown="UNKNOWN: 0"
    fi
  else
    full_exit=99
  fi

  iteration=$(next_iteration_number)
  ts_now=$(date -Iseconds)
  head_sha=$(git -C "$ROOT_DIR" rev-parse --short HEAD)

  {
    echo "## Iteration $iteration - $ts_now"
    echo
    echo "- Commit: \`$head_sha\`"
    echo "- Hypothesis: $hypothesis"
    echo "- Code change: $change_summary"
    echo "- Target file: \`$TARGET_FILE\`"
    echo "- Mini dataset: \`$mini_dataset_name\`"
    echo "- Mini command: \`bash_scripts/run_local.sh mini $mini_dataset_name\`"
    echo "- Mini exit code: \`$mini_exit\`"
    echo "- Mini status breakdown: \`$mini_status_breakdown\`"
    echo "- Mini entries: \`$mini_entry_count\`"
    echo "- Mini mean latency (ms): \`$mini_latency_mean\`"
    echo "- Mini mean speedup: \`$mini_speedup_mean\`"
    echo "- Mini run window: \`$mini_run_start\` -> \`$mini_run_end\`"
    echo "- Full command: \`bash_scripts/run_local.sh full-dump-trace-md\`"
    if [[ "$full_ran" == "true" ]]; then
      echo "- Full exit code: \`$full_exit\`"
      echo "- Full status breakdown: \`$full_status_breakdown\`"
      echo "- Full entries: \`$full_entry_count\`"
      echo "- Full mean latency (ms): \`$full_latency_mean\`"
      echo "- Full mean speedup: \`$full_speedup_mean\`"
      echo "- Full best speedup: \`$full_speedup_best\`"
      echo "- Full worst speedup: \`$full_speedup_worst\`"
      echo "- Full run window: \`$full_run_start\` -> \`$full_run_end\`"
      if [[ -n "$full_log_md" ]]; then
        echo "- Full markdown log: \`$full_log_md\`"
      fi
    else
      echo "- Full result: \`SKIPPED (mini preflight failed)\`"
    fi
    echo "- Decision: $decision"
    echo
  } >> "$LEDGER_FILE"

  cleanup

  echo
  echo "Appended experiment entry to: $LEDGER_FILE"
  echo "Kernel file reverted to HEAD: $TARGET_FILE"

  if [[ "$mini_ok" != "true" ]]; then
    echo "Mini preflight failed (exit=$mini_exit, all_passed=$mini_all_passed)." >&2
    exit 2
  fi
  if [[ "$full_exit" -ne 0 ]]; then
    echo "Full benchmark command failed with exit code $full_exit." >&2
    exit 3
  fi
  if [[ "$full_all_passed" != "true" ]]; then
    echo "Full benchmark completed but not all workloads PASSED." >&2
    exit 4
  fi
}

main "$@"
