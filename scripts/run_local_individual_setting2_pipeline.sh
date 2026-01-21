#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$ROOT_DIR/scripts/run_local_individual_sharded_pipeline.sh" \
  --cfg-prefix individual_federated_sharded_setting2 \
  "$@"
