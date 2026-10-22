#!/usr/bin/env bash
# Run the checkout directly; only the Python dependencies need to be available.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source_dir="$PWD"
if [[ ! -d "$source_dir/ggpa" ]]; then
    source_dir="$(dirname -- "$source_dir")"
fi
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
exec "${PYTHON:-python}" -u quickstart/run.py "$@"
