#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_EVAL=0
GPU_TYPE="H200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_p3_variants.sh [options]

Options:
  --dry-run   Print the sbatch commands without submitting.
  --skip-eval Do not launch evaluation arrays.
  --gpu-type  GPU type to request (e.g., H200).
  -h, --help  Show this message.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CMD=("$ROOT_DIR/scripts/run_p3_pipeline.sh" "--gpu-type" "$GPU_TYPE")
if [[ $DRY_RUN -eq 1 ]]; then
  CMD+=("--dry-run")
fi
if [[ $SKIP_EVAL -eq 1 ]]; then
  CMD+=("--skip-eval")
fi

"${CMD[@]}" --variant config
"${CMD[@]}" --variant dataset
"${CMD[@]}" --variant category
