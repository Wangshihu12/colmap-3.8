#!/usr/bin/env bash

set -u
set -o pipefail

############################################
# User config: edit this section only.
############################################

EXECUTABLE="/home/xgrids/github/github/colmap-3.8/build/bin/test_vpgo_pipeline"
SPARSE_PATH="/home/xgrids/文档/data_need_pgo/data_1/0/sfm_result/final/triangle/"
DATABASE_PATH="/home/xgrids/文档/data_need_pgo/data_1/0/database.db"
OUTPUT_ROOT="/home/xgrids/文档/data_need_pgo/data_1/0/test_pgo/three_stage_grid_$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --rig-config /home/xgrids/文档/data_need_pgo/data_1/0/rig_common.json
  --edge-input /home/xgrids/文档/data_need_pgo/data_1/0/sparse_pgo/pose_graph_edges.txt
  # --image-path /home/xgrids/文档/data_need_pgo/data_1/0/images/
)

# Stage-1
STAGE1_MAX_ITER_VALUES=(2)
STAGE1_TRI_MAX_PROJECT_ERROR_VALUES=(16.0)
STAGE1_FILTER_MAX_REPROJ_ERROR_VALUES=(16.0)

# Stage-2
STAGE2_MAX_ITER_VALUES=(2)
STAGE2_TRI_MAX_PROJECT_ERROR_VALUES=(4.0)
STAGE2_FILTER_MAX_REPROJ_ERROR_VALUES=(4.0)

# Stage-3
STAGE3_MAX_ITER_VALUES=(3)
STAGE3_TRI_MAX_PROJECT_ERROR_VALUES=(4.0)
STAGE3_FILTER_MAX_REPROJ_ERROR_VALUES=(4.0)

# 相对位姿约束权重
SNAPSHOT_RELATIVE_POSE_WEIGHT_VALUES=(10000.0)

############################################
# End user config.
############################################

sanitize_value() {
  local v="$1"
  v="${v//./p}"
  v="${v//-/m}"
  v="${v//\//_}"
  printf "%s" "$v"
}

require_nonempty() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "[ERROR] ${name} is empty. Edit script config first." >&2
    exit 1
  fi
}

require_nonempty "EXECUTABLE" "$EXECUTABLE"
require_nonempty "SPARSE_PATH" "$SPARSE_PATH"
require_nonempty "DATABASE_PATH" "$DATABASE_PATH"
require_nonempty "OUTPUT_ROOT" "$OUTPUT_ROOT"

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "[ERROR] Executable not found or not executable: $EXECUTABLE" >&2
  exit 1
fi

if [[ ! -d "$SPARSE_PATH" ]]; then
  echo "[ERROR] SPARSE_PATH does not exist: $SPARSE_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATABASE_PATH" ]]; then
  echo "[ERROR] DATABASE_PATH does not exist: $DATABASE_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/runs"
mkdir -p "$OUTPUT_ROOT/logs"

summary_file="$OUTPUT_ROOT/run_summary.tsv"
printf "run_idx\tstatus\telapsed_sec\toutput_dir\tlog_file\tstage1_max_iterations\tstage1_tri_max_project_error\tstage1_filter_max_reproj_error\tstage2_max_iterations\tstage2_tri_max_project_error\tstage2_filter_max_reproj_error\tstage3_max_iterations\tstage3_tri_max_project_error\tstage3_filter_max_reproj_error\tsnapshot_relative_pose_weight\n" > "$summary_file"

total_runs=$((
  ${#STAGE1_MAX_ITER_VALUES[@]} *
  ${#STAGE1_TRI_MAX_PROJECT_ERROR_VALUES[@]} *
  ${#STAGE1_FILTER_MAX_REPROJ_ERROR_VALUES[@]} *
  ${#STAGE2_MAX_ITER_VALUES[@]} *
  ${#STAGE2_TRI_MAX_PROJECT_ERROR_VALUES[@]} *
  ${#STAGE2_FILTER_MAX_REPROJ_ERROR_VALUES[@]} *
  ${#STAGE3_MAX_ITER_VALUES[@]} *
  ${#STAGE3_TRI_MAX_PROJECT_ERROR_VALUES[@]} *
  ${#STAGE3_FILTER_MAX_REPROJ_ERROR_VALUES[@]} *
  ${#SNAPSHOT_RELATIVE_POSE_WEIGHT_VALUES[@]}
))

if [[ "$total_runs" -le 0 ]]; then
  echo "[ERROR] total_runs is 0. Check your parameter arrays." >&2
  exit 1
fi

run_idx=0
failed=0

for s1_iter in "${STAGE1_MAX_ITER_VALUES[@]}"; do
  for s1_tri in "${STAGE1_TRI_MAX_PROJECT_ERROR_VALUES[@]}"; do
    for s1_filter in "${STAGE1_FILTER_MAX_REPROJ_ERROR_VALUES[@]}"; do
      for s2_iter in "${STAGE2_MAX_ITER_VALUES[@]}"; do
        for s2_tri in "${STAGE2_TRI_MAX_PROJECT_ERROR_VALUES[@]}"; do
          for s2_filter in "${STAGE2_FILTER_MAX_REPROJ_ERROR_VALUES[@]}"; do
            for s3_iter in "${STAGE3_MAX_ITER_VALUES[@]}"; do
              for s3_tri in "${STAGE3_TRI_MAX_PROJECT_ERROR_VALUES[@]}"; do
                for s3_filter in "${STAGE3_FILTER_MAX_REPROJ_ERROR_VALUES[@]}"; do
                  for sr_weight in "${SNAPSHOT_RELATIVE_POSE_WEIGHT_VALUES[@]}"; do
                      run_idx=$((run_idx + 1))

                      tag="s1i$(sanitize_value "$s1_iter")_s1tri$(sanitize_value "$s1_tri")_s1f$(sanitize_value "$s1_filter")_s2i$(sanitize_value "$s2_iter")_s2tri$(sanitize_value "$s2_tri")_s2f$(sanitize_value "$s2_filter")_s3i$(sanitize_value "$s3_iter")_s3tri$(sanitize_value "$s3_tri")_s3f$(sanitize_value "$s3_filter")_sw$(sanitize_value "$sr_weight")"
                      output_dir="$OUTPUT_ROOT/runs/$(printf "%03d" "$run_idx")_${tag}"
                      log_file="$OUTPUT_ROOT/logs/$(printf "%03d" "$run_idx")_${tag}.log"

                      mkdir -p "$output_dir"

                      cmd=(
                        "$EXECUTABLE"
                        "${COMMON_ARGS[@]}"
                        --enable-snapshot-relative-pose-constraints
                        --snapshot-relative-pose-rot-weight "$sr_weight"
                        --snapshot-relative-pose-trans-weight "$sr_weight"
                        --stage1-max-iterations "$s1_iter"
                        --stage1-tri-max-project-error "$s1_tri"
                        --stage1-filter-max-reproj-error "$s1_filter"
                        --stage2-max-iterations "$s2_iter"
                        --stage2-tri-max-project-error "$s2_tri"
                        --stage2-filter-max-reproj-error "$s2_filter"
                        --stage3-max-iterations "$s3_iter"
                        --stage3-tri-max-project-error "$s3_tri"
                        --stage3-filter-max-reproj-error "$s3_filter"
                        "$SPARSE_PATH"
                        "$DATABASE_PATH"
                        "$output_dir"
                      )

                      {
                        echo "[$(date '+%F %T')] Run ${run_idx}/${total_runs}"
                        echo "Output: $output_dir"
                        echo "Command: ${cmd[*]}"
                        echo "------------------------------------------------------------"
                      } | tee "$log_file"

                      start_ts=$(date +%s)
                      "${cmd[@]}" 2>&1 | tee -a "$log_file"
                      exit_code=${PIPESTATUS[0]}
                      end_ts=$(date +%s)
                      elapsed=$((end_ts - start_ts))

                      status="OK"
                      if [[ "$exit_code" -ne 0 ]]; then
                        status="FAIL(${exit_code})"
                        failed=$((failed + 1))
                      fi

                      printf "%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$run_idx" "$status" "$elapsed" "$output_dir" "$log_file" \
                        "$s1_iter" "$s1_tri" "$s1_filter" \
                        "$s2_iter" "$s2_tri" "$s2_filter" \
                        "$s3_iter" "$s3_tri" "$s3_filter" \
                        "$sr_weight" \
                        >> "$summary_file"

                      echo "[$(date '+%F %T')] ${status} elapsed=${elapsed}s" | tee -a "$log_file"
                      echo ""
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Finished. total_runs=${total_runs}, failed=${failed}"
echo "Summary: $summary_file"
exit 0
