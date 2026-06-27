#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

WORK16_MAX_WORKERS="${WORK16_MAX_WORKERS:-2}"
export WORK16_MAX_WORKERS

printf '%s\n' \
  "Running work/16_kalman_joint_ice_firn experiment suite" \
  "Worker limit: ${WORK16_MAX_WORKERS}" \
  "Launcher: work/16_kalman_joint_ice_firn/run_all.py"

cd "${REPO_ROOT}"
uv run python "work/16_kalman_joint_ice_firn/run_all.py"
