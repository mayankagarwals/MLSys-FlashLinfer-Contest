#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_FULL_DATASET_PATH="/home/simon/flashinfer-competition/mlsys26-contest"
FULL_DATASET_PATH="${FIB_FULL_DATASET_PATH:-$DEFAULT_FULL_DATASET_PATH}"
NCU_OUTPUT_DIR="${FIB_NCU_OUTPUT_DIR:-$ROOT_DIR/ncu_logs}"
NCU_PATH="${FIB_NCU_PATH:-ncu}"
NCU_DEVICE="${FIB_NCU_DEVICE:-cuda:0}"
NCU_SET="${FIB_NCU_SET:-detailed}"
NCU_PAGE="${FIB_NCU_PAGE:-details}"
NCU_TARGET_PROCESSES="${FIB_NCU_TARGET_PROCESSES:-all}"
NCU_KERNEL_NAME="${FIB_NCU_KERNEL_NAME:-regex:kernel_cutlass}"
NCU_WORKLOAD_SCOPE="${FIB_NCU_WORKLOAD_SCOPE:-all}"
NCU_MAX_WORKLOADS="${FIB_NCU_MAX_WORKLOADS:-0}"
NCU_TIMEOUT="${FIB_NCU_TIMEOUT:-300}"
NCU_SECTIONS="${FIB_NCU_SECTIONS:-}"
NCU_NVTX_INCLUDE="${FIB_NCU_NVTX_INCLUDE:-flashinfer_bench_ncu_profile]}"

usage() {
  cat <<'USAGE'
Usage:
  bash_scripts/run_ncu.sh full
  bash_scripts/run_ncu.sh mini [mini_dataset_name]

Purpose:
  - Run Nsight Compute on selected workloads (single profiled forward pass each).
  - No benchmark loop.
  - Save NCU output in a dedicated folder.
  - Uses flashinfer-bench native API: flashinfer_bench_run_ncu.

Environment variables:
  - FIB_FULL_DATASET_PATH: full dataset root (default: /home/simon/flashinfer-competition/mlsys26-contest)
  - FIB_NCU_OUTPUT_DIR: output folder for NCU artifacts (default: ./ncu_logs)
  - FIB_NCU_PATH: ncu binary path (default: ncu)
  - FIB_NCU_DEVICE: device string (default: cuda:0)
  - FIB_NCU_SET: ncu set, e.g. detailed/basic (default: detailed)
  - FIB_NCU_PAGE: ncu page, e.g. details/raw/source (default: details)
  - FIB_NCU_TARGET_PROCESSES: ncu target processes (default: all)
  - FIB_NCU_KERNEL_NAME: kernel-name filter (default: regex:kernel_cutlass)
  - FIB_NCU_WORKLOAD_SCOPE: all|first workloads to profile (default: all)
  - FIB_NCU_MAX_WORKLOADS: optional cap on selected workloads, 0 means no cap (default: 0)
  - FIB_NCU_TIMEOUT: timeout in seconds per workload (default: 300)
  - FIB_NCU_SECTIONS: optional comma-separated extra NCU sections
  - FIB_NCU_NVTX_INCLUDE: NVTX include expression patch (default: flashinfer_bench_ncu_profile])
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

prepare_profile_payload() {
  local dataset_path="$1"
  local data_dir="$2"
  (
    cd "$ROOT_DIR"
    FIB_DATASET_PATH="$dataset_path" FIB_PROFILE_DATA_DIR="$data_dir" FIB_NCU_WORKLOAD_SCOPE="$NCU_WORKLOAD_SCOPE" FIB_NCU_MAX_WORKLOADS="$NCU_MAX_WORKLOADS" uv run python - <<'PY'
import os
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from flashinfer_bench import Solution, TraceSet
from scripts.pack_solution import pack_solution

dataset_path = os.environ["FIB_DATASET_PATH"]
data_dir = Path(os.environ["FIB_PROFILE_DATA_DIR"])

solution_path = pack_solution()
solution = Solution.model_validate_json(solution_path.read_text())

trace_set = TraceSet.from_path(dataset_path)
definition = trace_set.definitions.get(solution.definition)
if definition is None:
    raise SystemExit(
        f"Definition '{solution.definition}' not found in trace set: {dataset_path}"
    )

workload_entries = trace_set.workloads.get(solution.definition, [])
if not workload_entries:
    raise SystemExit(f"No workloads found for definition '{solution.definition}'")

scope = os.environ.get("FIB_NCU_WORKLOAD_SCOPE", "all").strip().lower()
if scope not in {"all", "first"}:
    raise SystemExit(f"Invalid FIB_NCU_WORKLOAD_SCOPE='{scope}', expected 'all' or 'first'")

max_workloads_raw = os.environ.get("FIB_NCU_MAX_WORKLOADS", "0").strip()
try:
    max_workloads = int(max_workloads_raw)
except ValueError as e:
    raise SystemExit(f"Invalid FIB_NCU_MAX_WORKLOADS='{max_workloads_raw}': {e}")
if max_workloads < 0:
    raise SystemExit("FIB_NCU_MAX_WORKLOADS must be >= 0")

selected_entries = workload_entries[:1] if scope == "first" else list(workload_entries)
if max_workloads > 0:
    selected_entries = selected_entries[:max_workloads]

if not selected_entries:
    raise SystemExit("No workloads selected for profiling")

(data_dir / "definition.json").write_text(definition.model_dump_json())
(data_dir / "solution.json").write_text(solution.model_dump_json())
(data_dir / "workloads").mkdir(exist_ok=True)

for idx, entry in enumerate(selected_entries):
    workload = entry.workload if hasattr(entry, "workload") else entry
    workload_path = data_dir / "workloads" / f"{idx:06d}_{workload.uuid}.json"
    workload_path.write_text(workload.model_dump_json())

print(f"DEFINITION={solution.definition}")
print(f"TOTAL_WORKLOADS={len(workload_entries)}")
print(f"SELECTED_WORKLOADS={len(selected_entries)}")
PY
  )
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found in PATH." >&2
    exit 1
  fi
  if ! command -v "$NCU_PATH" >/dev/null 2>&1; then
    echo "NCU executable not found: $NCU_PATH" >&2
    exit 1
  fi

  local mode="$1"
  local dataset_path
  case "$mode" in
    full)
      dataset_path="$FULL_DATASET_PATH"
      ;;
    mini)
      dataset_path=$(get_mini_dataset_path "${2:-}")
      ;;
    -h|--help|help)
      usage
      return
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage
      exit 1
      ;;
  esac

  if [[ ! -d "$dataset_path" ]]; then
    echo "Dataset path not found: $dataset_path" >&2
    exit 1
  fi

  local data_dir
  data_dir=$(mktemp -d)
  trap 'if [[ -n "${data_dir:-}" ]]; then rm -rf -- "$data_dir"; fi' EXIT

  local definition=""
  local total_workloads=""
  local selected_workloads=""
  local meta_lines
  mapfile -t meta_lines < <(prepare_profile_payload "$dataset_path" "$data_dir")
  for line in "${meta_lines[@]}"; do
    case "$line" in
      DEFINITION=*)
        definition="${line#DEFINITION=}"
        ;;
      TOTAL_WORKLOADS=*)
        total_workloads="${line#TOTAL_WORKLOADS=}"
        ;;
      SELECTED_WORKLOADS=*)
        selected_workloads="${line#SELECTED_WORKLOADS=}"
        ;;
    esac
  done

  if [[ -z "$definition" || -z "$selected_workloads" || -z "$total_workloads" ]]; then
    echo "Failed to collect definition/workload metadata for NCU run." >&2
    exit 1
  fi

  local run_ts
  run_ts=$(date +%Y-%m-%dT%H-%M-%S)

  local run_dir="$NCU_OUTPUT_DIR/${definition}_${run_ts}"
  mkdir -p "$run_dir"
  cp "$data_dir/definition.json" "$run_dir/definition.json"
  cp "$data_dir/solution.json" "$run_dir/solution.json"
  mkdir -p "$run_dir/workloads"

  local -a workload_files
  mapfile -t workload_files < <(find "$data_dir/workloads" -maxdepth 1 -type f -name '*.json' | sort)
  if [[ "${#workload_files[@]}" -eq 0 ]]; then
    echo "No workload files were generated for profiling." >&2
    exit 1
  fi

  echo "Profiling workloads with NCU..."
  echo "  definition:    $definition"
  echo "  workloads:     $selected_workloads selected (from $total_workloads total)"
  echo "  scope:         $NCU_WORKLOAD_SCOPE"
  if [[ "$NCU_MAX_WORKLOADS" != "0" ]]; then
    echo "  max_workloads: $NCU_MAX_WORKLOADS"
  fi
  echo "  ncu_set/page:  $NCU_SET / $NCU_PAGE"
  echo "  ncu_api:       flashinfer_bench_run_ncu"
  echo "  dataset_path:  $dataset_path"
  echo "  output_dir:    $run_dir"

  local idx=0
  for workload_file in "${workload_files[@]}"; do
    idx=$((idx + 1))
    local workload_basename workload_uuid workload_dir
    workload_basename=$(basename "$workload_file")
    workload_uuid="${workload_basename#*_}"
    workload_uuid="${workload_uuid%.json}"
    workload_dir="$run_dir/workloads/$(printf '%06d_%s' "$idx" "$workload_uuid")"
    mkdir -p "$workload_dir"

    cp "$workload_file" "$data_dir/workload.json"
    cp "$workload_file" "$workload_dir/workload.json"

    local ncu_txt="$workload_dir/ncu_output.txt"
    local ncu_export_base="$workload_dir/ncu_report"
    local cmd_file="$workload_dir/command.txt"

    {
      echo "definition=$definition"
      echo "workload_uuid=$workload_uuid"
      echo "workload_index=$idx"
      echo "selected_workloads=$selected_workloads"
      echo "total_workloads=$total_workloads"
      echo "dataset_path=$dataset_path"
      echo "device=$NCU_DEVICE"
      echo "api=flashinfer_bench_run_ncu"
      echo "target_processes=$NCU_TARGET_PROCESSES"
      echo "kernel_filter=$NCU_KERNEL_NAME"
      echo "ncu_set=$NCU_SET"
      echo "ncu_page=$NCU_PAGE"
      echo "timeout_seconds=$NCU_TIMEOUT"
      echo "nvtx_include=$NCU_NVTX_INCLUDE"
      echo "ncu_export=$ncu_export_base.ncu-rep"
      if [[ -n "$NCU_SECTIONS" ]]; then
        echo "sections=$NCU_SECTIONS"
      fi
    } > "$cmd_file"

    echo
    echo "[$idx/$selected_workloads] Profiling workload: $workload_uuid"
    (
      cd "$ROOT_DIR"
      FIB_DATASET_PATH="$dataset_path" \
      FIB_NCU_DEVICE="$NCU_DEVICE" \
      FIB_NCU_SET="$NCU_SET" \
      FIB_NCU_PAGE="$NCU_PAGE" \
      FIB_NCU_TARGET_PROCESSES="$NCU_TARGET_PROCESSES" \
      FIB_NCU_KERNEL_NAME="$NCU_KERNEL_NAME" \
      FIB_NCU_PATH="$NCU_PATH" \
      FIB_NCU_TIMEOUT="$NCU_TIMEOUT" \
      FIB_NCU_SECTIONS="$NCU_SECTIONS" \
      FIB_NCU_NVTX_INCLUDE="$NCU_NVTX_INCLUDE" \
      FIB_NCU_EXPORT_BASE="$ncu_export_base" \
      FIB_NCU_SOLUTION_JSON="$run_dir/solution.json" \
      FIB_NCU_WORKLOAD_JSON="$workload_file" \
      uv run python - <<'PY' | tee "$ncu_txt"
import os
import sys

import flashinfer_bench.agents.ncu as ncu_mod

solution_json = os.environ["FIB_NCU_SOLUTION_JSON"]
workload_json = os.environ["FIB_NCU_WORKLOAD_JSON"]
sections_env = os.environ.get("FIB_NCU_SECTIONS", "").strip()
sections = [s.strip() for s in sections_env.split(",") if s.strip()] if sections_env else None
nvtx_include = os.environ.get("FIB_NCU_NVTX_INCLUDE", "flashinfer_bench_ncu_profile]")
target_processes = os.environ.get("FIB_NCU_TARGET_PROCESSES", "all").strip()
export_base = os.environ.get("FIB_NCU_EXPORT_BASE", "").strip()

orig_build = ncu_mod._build_ncu_command


def patched_build(*args, **kwargs):
    cmd = orig_build(*args, **kwargs)
    runner_start = len(cmd)
    if "-u" in cmd:
        runner_start = max(cmd.index("-u") - 1, 0)

    for i in range(len(cmd) - 1):
        if cmd[i] == "--nvtx-include":
            cmd[i + 1] = nvtx_include
            break

    if target_processes and "--target-processes" not in cmd:
        cmd[runner_start:runner_start] = ["--target-processes", target_processes]
        runner_start += 2

    if export_base and "--export" not in cmd:
        cmd[runner_start:runner_start] = ["--export", export_base]

    return cmd


ncu_mod._build_ncu_command = patched_build

result = ncu_mod.flashinfer_bench_run_ncu(
    solution=solution_json,
    workload=workload_json,
    device=os.environ["FIB_NCU_DEVICE"],
    trace_set_path=os.environ.get("FIB_DATASET_PATH"),
    set=os.environ["FIB_NCU_SET"],
    sections=sections,
    kernel_name=os.environ.get("FIB_NCU_KERNEL_NAME") or None,
    page=os.environ["FIB_NCU_PAGE"],
    ncu_path=os.environ["FIB_NCU_PATH"],
    timeout=int(os.environ["FIB_NCU_TIMEOUT"]),
    max_lines=None,
)

print(result, end="" if result.endswith("\n") else "\n")
if result.startswith("ERROR:"):
    sys.exit(1)
PY
    )
  done

  echo
  echo "Saved NCU artifacts:"
  echo "  $run_dir"
}

main "$@"
