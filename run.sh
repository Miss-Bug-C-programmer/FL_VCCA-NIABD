#!/usr/bin/env bash
set -Eeuo pipefail

# Stable repository-root entry point for the formal matrix.  The target
# script keeps rounds, devices, matrix size, and runtime controls configurable
# through environment variables while defaulting to 80 rounds and 80 runs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/run_cifar10_3090_matrix.sh" "$@"
