#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
export DATA_DIR

log() {
  printf "[warmup-eval-data] %s\n" "$*" >&2
}

mkdir -p "$DATA_DIR"

log "Downloading IFEval..."
python - <<'PY'
import os
from federatedscope.core.data.utils import download_url

data_dir = os.environ.get("DATA_DIR")
url = ("https://huggingface.co/datasets/google/IFEval/"
       "resolve/main/ifeval_input_data.jsonl")
download_url(url, data_dir)
PY

log "Downloading GSM8K..."
python - <<'PY'
import os
from federatedscope.core.data.utils import download_url

data_dir = os.environ.get("DATA_DIR")
target = os.path.join(data_dir, "gsm8k_test.jsonl")
if not os.path.exists(target):
    download_url(
        "https://raw.githubusercontent.com/openai/"
        "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
        "grade_school_math/data/test.jsonl",
        data_dir,
    )
    os.rename(os.path.join(data_dir, "test.jsonl"), target)
PY

log "Downloading HumanEval..."
python - <<'PY'
import os
from federatedscope.core.data.utils import download_url

data_dir = os.environ.get("DATA_DIR")
download_url(
    "https://github.com/openai/human-eval/raw/"
    "463c980b59e818ace59f6f9803cd92c749ceae61/"
    "data/HumanEval.jsonl.gz",
    data_dir,
)
PY

log "Downloading MMLU..."
python - <<'PY'
import os
import tarfile
from federatedscope.core.data.utils import download_url

data_dir = os.environ.get("DATA_DIR")
mmlu_dir = os.path.join(data_dir, "mmlu")
if not os.path.exists(mmlu_dir):
    tar_path = download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar", data_dir)
    with tarfile.open(tar_path, "r:") as tar:
        os.makedirs(mmlu_dir, exist_ok=True)
        tar.extractall(path=mmlu_dir)
PY

log "Warmup complete. Data lives in $DATA_DIR."
