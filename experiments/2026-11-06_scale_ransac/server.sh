#!/usr/bin/env bash
set -euo pipefail
# Run inside an allocated server environment or tmux. All arguments passed unchanged.
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${EXPERIMENT_PYTHON:-python}" -u "$task_dir/run_experiments.py" "$@"
