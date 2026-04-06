#!/usr/bin/env bash
set -euo pipefail

START_DIR="$(pwd)"
DEFAULT_REPO_DIR="$HOME/v2e"
if [ -d "$START_DIR/.git" ]; then
  DEFAULT_REPO_DIR="$START_DIR"
fi

GITHUB_REPO_URL="${GITHUB_REPO_URL:-https://github.com/vlordier/v2e.git}"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
BASE_BRANCH="${BASE_BRANCH:-upgrades}"
RESEARCH_BRANCH="${RESEARCH_BRANCH:-research/vastai-$(date +%b%d | tr '[:upper:]' '[:lower:]')}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTORESEARCH_PLAN="${AUTORESEARCH_PLAN:-research/v2e_imu/autoresearch_plan.example.json}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/research/v2e_imu/.env.vastai.local}"

case "$ENV_FILE" in
  /*) ;;
  *) ENV_FILE="$START_DIR/$ENV_FILE" ;;
esac

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
git "${GIT_AUTH[@]}" fetch origin "$BASE_BRANCH:refs/remotes/origin/$BASE_BRANCH" || true
git "${GIT_AUTH[@]}" fetch origin "$RESEARCH_BRANCH:refs/remotes/origin/$RESEARCH_BRANCH" || true
if git show-ref --verify --quiet "refs/remotes/origin/$RESEARCH_BRANCH"; then
  git checkout -B "$RESEARCH_BRANCH" "origin/$RESEARCH_BRANCH"
elif git show-ref --verify --quiet "refs/remotes/origin/$BASE_BRANCH"; then
  git checkout -B "$RESEARCH_BRANCH" "origin/$BASE_BRANCH"
else
  git checkout -B "$RESEARCH_BRANCH"
fi

$PYTHON_BIN -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mlops]' timm

git config user.name "${GIT_AUTHOR_NAME:-vastai-autoresearch}"
git config user.email "${GIT_AUTHOR_EMAIL:-vastai-autoresearch@example.com}"

mkdir -p "$LOG_DIR"

echo "[bootstrap] repo_dir=$REPO_DIR"
echo "[bootstrap] base_branch=$BASE_BRANCH research_branch=$RESEARCH_BRANCH"
echo "[bootstrap] log_dir=$LOG_DIR"

if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "[bootstrap] validating AWS credentials"
  if ! "$REPO_DIR/.venv/bin/python" - <<'PY'
from research.v2e_imu.autoresearch_runner import aws_identity_preflight_ok
raise SystemExit(0 if aws_identity_preflight_ok() else 1)
PY
  then
    echo "[bootstrap] warning: AWS STS preflight failed; continuing without blocking startup" >&2
  fi
fi

export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-v2e-imu-vastai}"
export AUTORESEARCH_OPTUNA_ENABLED="${AUTORESEARCH_OPTUNA_ENABLED:-1}"
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
export V2E_TORCH_COMPILE="${V2E_TORCH_COMPILE:-1}"
export RESEARCH_BRANCH
export GITHUB_REMOTE="${GITHUB_REMOTE:-origin}"
export PYTHONUNBUFFERED=1

EXISTING_PIDS="$(pgrep -f 'research/v2e_imu/autoresearch_runner.py' || true)"
if [ -n "$EXISTING_PIDS" ]; then
  echo "[bootstrap] stopping existing autoresearch runner(s): $EXISTING_PIDS"
  # shellcheck disable=SC2086
  kill $EXISTING_PIDS || true
  sleep 2
fi

echo "[bootstrap] starting autoresearch runner"
nohup python -u research/v2e_imu/autoresearch_runner.py \
  --env-file "$ENV_FILE" \
  --plan "$AUTORESEARCH_PLAN" \
  --python-bin "$REPO_DIR/.venv/bin/python" \
  --branch "$RESEARCH_BRANCH" \
  --remote "$GITHUB_REMOTE" \
  --push > "$LOG_DIR/vastai-autoresearch.out" 2>&1 &
RUN_PID=$!
sleep 5

printf '\nAutoresearch launched.\n'
printf 'Runner PID: %s\n' "$RUN_PID"
printf 'Log file: %s\n' "$LOG_DIR/vastai-autoresearch.out"
printf 'Tail with: tail -f %s\n' "$LOG_DIR/vastai-autoresearch.out"
ps -p "$RUN_PID" -o pid=,stat=,etime=,command= || true
if [ -f "$LOG_DIR/vastai-autoresearch.out" ]; then
  echo "[bootstrap] initial runner log"
  tail -n 40 "$LOG_DIR/vastai-autoresearch.out" || true
fi
