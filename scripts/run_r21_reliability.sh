#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing real/ ADM/ BigGAN/ glide/ Midjourney/ SD/ VQDM/}"
: "${CKPT_ROOT:?Set CKPT_ROOT to the directory containing all six exclude_* task directories}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a directory outside the repository, dataset and checkpoint roots}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULT_CHECKPOINT_TEMPLATE='exclude_{generator}/ddfsd_step[15000].pth'
DEFAULT_STATS_TEMPLATE='exclude_{generator}/freq_stats.pt'
cd "${REPO_ROOT}"
exec "${PYTHON:-python}" -u tools/run_r21_reliability.py \
  --data-root "${DATA_ROOT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --seeds "${SEEDS:-42,101,102,103,104}" \
  --support-shot "${SUPPORT_SHOT:-10}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --use-fp16 "${USE_FP16:-true}" \
  --checkpoint-template "${CHECKPOINT_TEMPLATE:-$DEFAULT_CHECKPOINT_TEMPLATE}" \
  --stats-template "${STATS_TEMPLATE:-$DEFAULT_STATS_TEMPLATE}" \
  --all-queries --resume --package "$@"
