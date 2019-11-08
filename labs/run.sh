#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=60
#SBATCH --requeue

set -euo pipefail

# https://stackoverflow.com/a/949391/4054250
echo Git commit: "$(git rev-parse --verify HEAD)"

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
VENV=${VENV:-$PWD/venv}
. "$VENV/bin/activate"

python -O "$@"
