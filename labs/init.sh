#!/usr/bin/env bash

set -euo pipefail

if [ -n "${MODULESHOME-}" ]; then module load Python; fi
if [ ! -e venv ]; then python3 -m virtualenv venv; fi
source venv/bin/activate
pip install -r requirements.txt
