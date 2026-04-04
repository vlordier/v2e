#!/usr/bin/env bash
set -euo pipefail

GITHUB_REPO_URL="${GITHUB_REPO_URL:-https://github.com/vlordier/v2e.git}"
REPO_DIR="${REPO_DIR:-$HOME/v2e}"
BASE_BRANCH="${BASE_BRANCH:-research/apr03}"
RESEARCH_BRANCH="${RESEARCH_BRANCH:-research/vastai-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTORESEARCH_PLAN="${AUTORESEARCH_PLAN:-research/v2e_imu/autoresearch_plan.example.json}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/research/v2e_imu/.env.vastai.local}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a fine-grained PAT or GitHub App token}"

AUTH_HEADER="$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')"
GIT_AUTH=(-c "http.extraheader=AUTHORIZATION: basic ${AUTH_HEADER}")

if [ ! -d "$REPO_DIR/.git" ]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  git "${GIT_AUTH[@]}" clone "$GITHUB_REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git "${GIT_AUTH[@]}" fetch origin --prune
if git show-ref --verify --quiet "refs/remotes/origin/$BASE_BRANCH"; then
  git checkout -B "$RESEARCH_BRANCH" "origin/$BASE_BRANCH"
else
  git checkout -B "$RESEARCH_BRANCH"
fi

$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mlops]' awscli timm

git config user.name "${GIT_AUTHOR_NAME:-vastai-autoresearch}"
git config user.email "${GIT_AUTHOR_EMAIL:-vastai-autoresearch@example.com}"

mkdir -p "$LOG_DIR"

if command -v aws >/dev/null 2>&1 && [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  aws sts get-caller-identity >/dev/null
fi

export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-v2e-imu-vastai}"
export RESEARCH_BRANCH
export GITHUB_REMOTE="${GITHUB_REMOTE:-origin}"

nohup python research/v2e_imu/autoresearch_runner.py \
  --env-file "$ENV_FILE" \
  --plan "$AUTORESEARCH_PLAN" \
  --python-bin "$REPO_DIR/.venv/bin/python" \
  --branch "$RESEARCH_BRANCH" \
  --remote "$GITHUB_REMOTE" \
  --push > "$LOG_DIR/vastai-autoresearch.out" 2>&1 &

printf '\nAutoresearch launched.\n'
printf 'Log file: %s\n' "$LOG_DIR/vastai-autoresearch.out"
printf 'Tail with: tail -f %s\n' "$LOG_DIR/vastai-autoresearch.out"
