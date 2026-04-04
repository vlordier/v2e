#!/usr/bin/env bash
set -euo pipefail

# Example Vast.ai launch helper.
# Prereqs:
#   pip install vastai
#   vastai set api-key <YOUR_VAST_API_KEY>
#   export GITHUB_TOKEN=...
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#   export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000
#   export S3_RESULTS_PREFIX=s3://<bucket>/v2e-autoresearch

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${AWS_DEFAULT_REGION:?Set AWS_DEFAULT_REGION}"
: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI}"
: "${S3_RESULTS_PREFIX:?Set S3_RESULTS_PREFIX}"

IMAGE="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
DISK_GB="80"
SEARCH_QUERY='reliability > 0.98 num_gpus=1 gpu_ram>=20 dph<0.6 inet_up>100 inet_down>100'
ONSTART=$(cat <<'EOF'
apt-get update && apt-get install -y git curl python3 python3-venv python3-pip && \
cd /workspace && \
git clone https://github.com/vlordier/v2e.git || true && \
cd /workspace/v2e && \
chmod +x scripts/vastai_bootstrap.sh && \
set -a && \
cat > research/v2e_imu/.env.vastai.local <<ENVVARS
GITHUB_TOKEN=${GITHUB_TOKEN}
GITHUB_REPO_URL=https://github.com/vlordier/v2e.git
GITHUB_REMOTE=origin
RESEARCH_BRANCH=research/vastai-$(date +%b%d | tr '[:upper:]' '[:lower:]')
GIT_AUTHOR_NAME=vastai-autoresearch
GIT_AUTHOR_EMAIL=vastai-autoresearch@example.com
PYTHON_BIN=python3
TRAIN_TIMEOUT_SECONDS=1200
AUTORESEARCH_PLAN=research/v2e_imu/autoresearch_plan.example.json
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
MLFLOW_EXPERIMENT_NAME=v2e-imu-vastai
MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL:-https://s3.amazonaws.com}
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
S3_RESULTS_PREFIX=${S3_RESULTS_PREFIX}
ENVVARS
source research/v2e_imu/.env.vastai.local && \
set +a && \
bash scripts/vastai_bootstrap.sh
EOF
)

vastai search offers "$SEARCH_QUERY"

echo
read -r -p "Enter chosen offer id: " OFFER_ID

vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh \
  --onstart-cmd "$ONSTART"
