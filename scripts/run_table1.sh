#!/usr/bin/env bash
# Reproduce the vision STABLE rows of Table 1.
#
#   scripts/run_table1.sh CELL [SEED ...]
#
# CELL is one of lora_imagenet_r, lora_cifar100, lora_imagenet_a, prompts_imagenet_r,
# prompts_cifar100, prompts_imagenet_a, or "all". The seeds default to 40 41 42 (the paper
# reports their mean). Results go to $OUTPUT_DIR/CELL/seedN/{log.txt,results.json}.
#
# Environment: PYTHON (default python), DATA_DIR (default data),
# CHECKPOINT (default checkpoints/vit_base_patch16_224.augreg_in21k.safetensors),
# OUTPUT_DIR (default outputs/table1).
set -euo pipefail
cd "$(dirname "$0")/.."

CELLS=(lora_imagenet_r lora_cifar100 lora_imagenet_a prompts_imagenet_r prompts_cifar100 prompts_imagenet_a)
if [[ $# -lt 1 ]]; then
  echo "usage: $0 {$(IFS='|'; echo "${CELLS[*]}")|all} [SEED ...]" >&2
  exit 1
fi
if [[ $1 == all ]]; then cells=("${CELLS[@]}"); else cells=("$1"); fi
shift
seeds=("$@")
[[ ${#seeds[@]} -eq 0 ]] && seeds=(40 41 42)

for cell in "${cells[@]}"; do
  [[ -f configs/table1/$cell.yaml ]] || { echo "unknown cell: $cell" >&2; exit 1; }
  for seed in "${seeds[@]}"; do
    "${PYTHON:-python}" -m stable.train --config "configs/table1/$cell.yaml" --seed "$seed" \
      --data-dir "${DATA_DIR:-data}" \
      --checkpoint "${CHECKPOINT:-checkpoints/vit_base_patch16_224.augreg_in21k.safetensors}" \
      --output-dir "${OUTPUT_DIR:-outputs/table1}"
  done
done
