#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the existing GenImage directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR outside the repository}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"
# CKPT_ROOT or CKPT_PATH/FREQ_STATS_PATH are read by the Python entrypoint.
# Explicit templates are quoted as literals to preserve {generator} and [15000].
DEFAULT_CHECKPOINT_TEMPLATE='exclude_{generator}/ckpt/ddfsd_step[15000].pth'
DEFAULT_STATS_TEMPLATE='exclude_{generator}/freq_stats.pt'
exec "${PYTHON:-python}" -B -u test_ddfsd_r24.py \
  --experiment "${EXPERIMENT:-all}" \
  --exclude-class "${EXCLUDE_CLASS:-all}" \
  --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}" \
  --seeds "${SEEDS:-42,101,102,103,104}" \
  --noise-rates "${NOISE_RATES:-0.1,0.2,0.3}" \
  --batch-size "${BATCH_SIZE:-128}" --num-workers "${NUM_WORKERS:-8}" \
  --use-fp16 "${USE_FP16:-true}" \
  --checkpoint-template "${CHECKPOINT_TEMPLATE:-$DEFAULT_CHECKPOINT_TEMPLATE}" \
  --stats-template "${STATS_TEMPLATE:-$DEFAULT_STATS_TEMPLATE}" \
  --resume "$@"
