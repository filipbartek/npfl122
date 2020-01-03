#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=60
#SBATCH --requeue

# ssh -L 6007:localhost:6007 cluster

# srun --mem=1G --time=60 --partition=gpu --gres=gpu:1 --pty bash -i

# VENV=../venv sbatch --time=240 --cpus-per-task=16 --job-name=walker-base ../run.sh walker.py
# VENV=../venv sbatch --time=240 --cpus-per-task=16 --partition=gpu --gres=gpu:1 --job-name=walker-base ../run.sh walker.py

set -euo pipefail

# https://stackoverflow.com/a/949391/4054250
echo Git commit: "$(git rev-parse --verify HEAD)"

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
VENV=${VENV:-$PWD/venv}
. "$VENV/bin/activate"

echo "$@"

python -O "$@"
