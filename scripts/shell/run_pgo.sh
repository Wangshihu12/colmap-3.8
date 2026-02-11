#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_pgo.sh --bin <test_vpgo_pipeline> \
    --sparse <sparse_path> --database <db_path> --output-base <dir> \
    [--gt <gt_reconstruction_path>] \
    [--min-point-pairs "12,16"] \
    [--max-reproj-error "8.0,6.0"] \
    [--min-track-length "3,4"] \
    [--max-depth-ratio "10.0,6.0"] \
    [--ransac-max-error "auto,1e-3"] \
    [--ransac-min-inlier-ratio "0.25,0.35"] \
    [--extra "<extra args for test_vpgo_pipeline>"]

Notes:
  - ransac-max-error=auto will pass -1 to use the internal auto threshold.
  - If --gt is provided, summary.csv will include mean/max errors from
    loop_pose_errors.txt.
EOF
}

BIN=""
SPARSE=""
DB=""
OUT_BASE=""
GT=""

MIN_POINT_PAIRS="12"
MAX_REPROJ_ERROR="8.0"
MIN_TRACK_LENGTH="3"
MAX_DEPTH_RATIO="10.0"
RANSAC_MAX_ERROR="auto"
RANSAC_MIN_INLIER_RATIO="0.25"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)
      BIN="$2"; shift 2 ;;
    --sparse)
      SPARSE="$2"; shift 2 ;;
    --database)
      DB="$2"; shift 2 ;;
    --output-base)
      OUT_BASE="$2"; shift 2 ;;
    --gt)
      GT="$2"; shift 2 ;;
    --min-point-pairs)
      MIN_POINT_PAIRS="$2"; shift 2 ;;
    --max-reproj-error)
      MAX_REPROJ_ERROR="$2"; shift 2 ;;
    --min-track-length)
      MIN_TRACK_LENGTH="$2"; shift 2 ;;
    --max-depth-ratio)
      MAX_DEPTH_RATIO="$2"; shift 2 ;;
    --ransac-max-error)
      RANSAC_MAX_ERROR="$2"; shift 2 ;;
    --ransac-min-inlier-ratio)
      RANSAC_MIN_INLIER_RATIO="$2"; shift 2 ;;
    --extra)
      if [[ -n "${2:-}" ]]; then
        read -r -a extra_tokens <<< "$2"
        EXTRA_ARGS+=("${extra_tokens[@]}")
      fi
      shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      echo "Unknown arg: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ -z "$BIN" || -z "$SPARSE" || -z "$DB" || -z "$OUT_BASE" ]]; then
  usage
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not executable: $BIN" >&2
  exit 1
fi

mkdir -p "$OUT_BASE"

IFS=',' read -r -a MIN_POINT_PAIRS_ARR <<< "$MIN_POINT_PAIRS"
IFS=',' read -r -a MAX_REPROJ_ERROR_ARR <<< "$MAX_REPROJ_ERROR"
IFS=',' read -r -a MIN_TRACK_LENGTH_ARR <<< "$MIN_TRACK_LENGTH"
IFS=',' read -r -a MAX_DEPTH_RATIO_ARR <<< "$MAX_DEPTH_RATIO"
IFS=',' read -r -a RANSAC_MAX_ERROR_ARR <<< "$RANSAC_MAX_ERROR"
IFS=',' read -r -a RANSAC_MIN_INLIER_RATIO_ARR <<< "$RANSAC_MIN_INLIER_RATIO"

SUMMARY_CSV="${OUT_BASE}/summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_id,min_point_pairs,max_reproj_error_px,min_track_length,max_depth_ratio,ransac_max_error,ransac_min_inlier_ratio,mean_rot_error_deg,mean_trans_error,mean_trans_dir_error_deg,max_rot_error_deg,max_trans_error,max_trans_dir_error_deg" > "$SUMMARY_CSV"
fi

for mpp in "${MIN_POINT_PAIRS_ARR[@]}"; do
  for mre in "${MAX_REPROJ_ERROR_ARR[@]}"; do
    for mtl in "${MIN_TRACK_LENGTH_ARR[@]}"; do
      for mdr in "${MAX_DEPTH_RATIO_ARR[@]}"; do
        for rme_raw in "${RANSAC_MAX_ERROR_ARR[@]}"; do
          for rir in "${RANSAC_MIN_INLIER_RATIO_ARR[@]}"; do
            rme="$rme_raw"
            if [[ "$rme_raw" == "auto" ]]; then
              rme="-1"
            fi

            run_id="mpp${mpp}_mre${mre}_mtl${mtl}_mdr${mdr}_rme${rme_raw}_rir${rir}"
            run_id="${run_id//./p}"
            out_dir="${OUT_BASE}/sim3_${run_id}"
            mkdir -p "$out_dir"

            cmd=(
              "$BIN"
              "$SPARSE"
              "$DB"
              "$out_dir"
              --sim3-min-point-pairs "$mpp"
              --sim3-max-reproj-error "$mre"
              --sim3-min-track-length "$mtl"
              --sim3-max-depth-ratio "$mdr"
              --sim3-ransac-max-error "$rme"
              --sim3-ransac-min-inlier-ratio "$rir"
            )

            if [[ -n "$GT" ]]; then
              cmd+=(--gt-reconstruction "$GT")
            fi

            if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
              cmd+=("${EXTRA_ARGS[@]}")
            fi

            echo "Running: ${cmd[*]}"
            "${cmd[@]}" > "${out_dir}/run.log" 2>&1

            mean_rot="NA"; mean_trans="NA"; mean_trans_dir="NA"
            max_rot="NA"; max_trans="NA"; max_trans_dir="NA"
            if [[ -f "${out_dir}/loop_pose_errors.txt" ]]; then
              mean_rot=$(awk -F': ' '/^# Mean rot error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
              mean_trans=$(awk -F': ' '/^# Mean trans error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
              mean_trans_dir=$(awk -F': ' '/^# Mean trans dir error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
              max_rot=$(awk -F': ' '/^# Max rot error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
              max_trans=$(awk -F': ' '/^# Max trans error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
              max_trans_dir=$(awk -F': ' '/^# Max trans dir error/ {print $2}' "${out_dir}/loop_pose_errors.txt" | tail -n1)
            fi

            echo "${run_id},${mpp},${mre},${mtl},${mdr},${rme_raw},${rir},${mean_rot},${mean_trans},${mean_trans_dir},${max_rot},${max_trans},${max_trans_dir}" >> "$SUMMARY_CSV"
          done
        done
      done
    done
  done
done

echo "Done. Summary: ${SUMMARY_CSV}"

# ../scripts/shell/run_pgo.sh \
#   --bin ./bin/test_vpgo_pipeline \
#   --sparse /home/xgrids/文档/data_need_pgo/data_1/0/sfm_result/final/triangle/ \
#   --database /home/xgrids/文档/data_need_pgo/data_1/0/database.db \
#   --output-base /home/xgrids/文档/data_need_pgo/data_1/0/test_pgo \
#   --gt /home/xgrids/文档/data_need_pgo/data_1/0/sfm_result/final/ \
#   --min-point-pairs "12" \
#   --max-reproj-error "8.0" \
#   --min-track-length "3" \
#   --max-depth-ratio "10.0" \
#   --ransac-max-error "auto" \
#   --ransac-min-inlier-ratio "0.25" \
#   --extra "--rig-config /home/xgrids/文档/data_need_pgo/data_1/0/rig_common.json"