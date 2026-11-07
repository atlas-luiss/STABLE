#!/bin/bash
# Example Slurm job for one Table 1 cell. Fill in the placeholders, then:
#   CELL=lora_cifar100 SEEDS="40 41 42" sbatch scripts/slurm_example.sbatch
#SBATCH --job-name=stable-table1
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# DataLoader workers create sockets under TMPDIR; a long TMPDIR path can exceed the
# AF_UNIX path limit, so use a short node-local directory.
export TMPDIR="/dev/shm/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Activate your environment here, e.g.: source <path-to-venv>/bin/activate
bash scripts/run_table1.sh "${CELL:-lora_cifar100}" ${SEEDS:-42}
