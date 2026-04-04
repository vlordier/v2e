#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/research/v2e_imu/.env.vastai.local}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
LOCAL_PYTHON="${LOCAL_PYTHON:-$VENV_DIR/bin/python}"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-80}"
SEARCH_QUERY="${SEARCH_QUERY:-reliability > 0.98 num_gpus=1 gpu_ram>=20 dph<0.6 inet_up>100 inet_down>100}"
ACTION="${1:-launch}"
OFFER_ID="${2:-${OFFER_ID:-}}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/vastai_launch_example.sh search
  bash scripts/vastai_launch_example.sh launch [offer_id]
  bash scripts/vastai_launch_example.sh dry-run
  bash scripts/vastai_launch_example.sh print-onstart

Behavior:
  - auto-loads research/v2e_imu/.env.vastai.local if present
  - auto-installs the Vast.ai CLI into ./.venv if missing
  - can print the exact on-start command for the Vast.ai web UI
EOF
}

load_env_file() {
  if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required env var: $name" >&2
    exit 1
  fi
}

ensure_local_python() {
  if [ ! -x "$LOCAL_PYTHON" ]; then
    python3 -m venv "$VENV_DIR"
    "$LOCAL_PYTHON" -m pip install --upgrade pip >/dev/null
  fi
}

ensure_vastai() {
  if command -v vastai >/dev/null 2>&1; then
    VASTAI_BIN="$(command -v vastai)"
    return
  fi

  ensure_local_python
  "$LOCAL_PYTHON" -m pip install --upgrade vastai >/dev/null
  VASTAI_BIN="$VENV_DIR/bin/vastai"
}

configure_api_key() {
  if [ -n "${VAST_API_KEY:-}" ]; then
    "$VASTAI_BIN" set api-key "$VAST_API_KEY" >/dev/null
  fi
}

shell_quote() {
  printf '%q' "$1"
}

build_remote_env() {
  cat <<EOF
GITHUB_TOKEN=$(shell_quote "${GITHUB_TOKEN:-}")
GITHUB_REPO_URL=$(shell_quote "${GITHUB_REPO_URL:-https://github.com/vlordier/v2e.git}")
GITHUB_REMOTE=$(shell_quote "${GITHUB_REMOTE:-origin}")
RESEARCH_BRANCH=$(shell_quote "${RESEARCH_BRANCH:-research/vastai-$(date +%b%d | tr '[:upper:]' '[:lower:]')}")
GIT_AUTHOR_NAME=$(shell_quote "${GIT_AUTHOR_NAME:-vastai-autoresearch}")
GIT_AUTHOR_EMAIL=$(shell_quote "${GIT_AUTHOR_EMAIL:-vastai-autoresearch@example.com}")
PYTHON_BIN=$(shell_quote "${PYTHON_BIN:-python3}")
TRAIN_TIMEOUT_SECONDS=$(shell_quote "${TRAIN_TIMEOUT_SECONDS:-1200}")
AUTORESEARCH_PLAN=$(shell_quote "${AUTORESEARCH_PLAN:-research/v2e_imu/autoresearch_plan.example.json}")
MLFLOW_TRACKING_URI=$(shell_quote "${MLFLOW_TRACKING_URI:-}")
MLFLOW_EXPERIMENT_NAME=$(shell_quote "${MLFLOW_EXPERIMENT_NAME:-v2e-imu-vastai}")
MLFLOW_S3_ENDPOINT_URL=$(shell_quote "${MLFLOW_S3_ENDPOINT_URL:-https://s3.amazonaws.com}")
AWS_ACCESS_KEY_ID=$(shell_quote "${AWS_ACCESS_KEY_ID:-}")
AWS_SECRET_ACCESS_KEY=$(shell_quote "${AWS_SECRET_ACCESS_KEY:-}")
AWS_DEFAULT_REGION=$(shell_quote "${AWS_DEFAULT_REGION:-us-east-1}")
S3_RESULTS_PREFIX=$(shell_quote "${S3_RESULTS_PREFIX:-}")
EOF
}

build_onstart() {
  cat <<EOF
apt-get update && apt-get install -y git curl python3 python3-venv python3-pip && \
mkdir -p /workspace && \
cd /workspace && \
if [ ! -d v2e/.git ]; then git clone $(shell_quote "${GITHUB_REPO_URL:-https://github.com/vlordier/v2e.git}") v2e; fi && \
cd /workspace/v2e && \
cat > research/v2e_imu/.env.vastai.local <<'ENVVARS'\n$(build_remote_env)\nENVVARS
chmod +x scripts/vastai_bootstrap.sh && \
ENV_FILE=research/v2e_imu/.env.vastai.local bash scripts/vastai_bootstrap.sh
EOF
}

load_env_file

case "$ACTION" in
  -h|--help|help)
    usage
    exit 0
    ;;
  print-onstart)
    require_env GITHUB_TOKEN
    build_onstart
    exit 0
    ;;
  dry-run)
    require_env GITHUB_TOKEN
    echo "Search query: $SEARCH_QUERY"
    echo "Image: $IMAGE"
    echo "Disk: $DISK_GB GB"
    echo
    echo "--- onstart command ---"
    build_onstart
    exit 0
    ;;
  search)
    ensure_vastai
    configure_api_key
    "$VASTAI_BIN" search offers "$SEARCH_QUERY"
    exit 0
    ;;
  launch)
    require_env GITHUB_TOKEN
    ensure_vastai
    configure_api_key
    "$VASTAI_BIN" search offers "$SEARCH_QUERY"
    echo
    if [ -z "$OFFER_ID" ]; then
      read -r -p "Enter chosen offer id: " OFFER_ID
    fi
    "$VASTAI_BIN" create instance "$OFFER_ID" \
      --image "$IMAGE" \
      --disk "$DISK_GB" \
      --ssh \
      --onstart-cmd "$(build_onstart)"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
