#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELLCHECK_BIN="${SHELLCHECK_BIN:-$ROOT_DIR/.venv/bin/shellcheck}"

if [ ! -x "$SHELLCHECK_BIN" ]; then
  echo "Missing shellcheck binary at $SHELLCHECK_BIN" >&2
  echo "Install with: $ROOT_DIR/.venv/bin/python -m pip install shellcheck-py" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  SHELL_FILES="$*"
else
  SHELL_FILES="$ROOT_DIR/scripts/vastai_bootstrap.sh $ROOT_DIR/scripts/vastai_launch_example.sh $ROOT_DIR/scripts/lint_shell.sh"
fi

# shellcheck disable=SC2086
"$SHELLCHECK_BIN" -x $SHELL_FILES
