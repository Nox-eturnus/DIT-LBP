#!/usr/bin/env bash
# Run a Python script in the project's conda env (env "lbp") with its DLL dirs on PATH.
# Usage: bash run.sh scripts/run_experiment.py [args]
E="${LBP_ENV:-/c/miniconda3/envs/lbp}"
export PATH="$E:$E/Library/bin:$E/Scripts:$PATH"
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$E/python.exe" "$@"
