#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=60
#SBATCH --requeue

set -euo pipefail

export VENV=../venv

COMMAND=(
  lunar_lander.py
  "$@"
)

if [ -n "${SLURM_JOB_ID-}" ]; then
  COMMAND+=(--output "slurm-$SLURM_JOB_ID-model.npy")
  COMMAND+=(--run_name "slurm-$SLURM_JOB_ID-$SLURM_JOB_NAME")
fi

../run.sh "${COMMAND[@]}"
