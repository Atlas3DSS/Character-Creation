#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-192.168.1.90}"
REMOTE_USER="${REMOTE_USER:-orwel}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/orwel/dev_genius/experiments/Character Creation}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/orwel/dev_genius}"
REMOTE_VENV="${REMOTE_VENV:-/home/orwel/dev_genius/venv/bin/python}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATASET_REL="${DATASET_REL:-sweep_v4/qwen35_teacher_lora_pair_pilot}"
RUN_TAG="${RUN_TAG:-qwen35_pair_lora_pilot}"
MODEL_PATH="${MODEL_PATH:-/home/orwel/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
SAMPLE_STEPS="${SAMPLE_STEPS:-25}"
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-25}"

if [[ ! -f /tmp/codex_askpass.sh ]]; then
  cat >/tmp/codex_askpass.sh <<'SH'
#!/usr/bin/env bash
printf '%s' '214184'
SH
  chmod 700 /tmp/codex_askpass.sh
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no)
SCP_BASE=(scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no)
export DISPLAY="${DISPLAY:-:0}"
export SSH_ASKPASS=/tmp/codex_askpass.sh
export SSH_ASKPASS_REQUIRE=force

LOCAL_DATASET="$LOCAL_ROOT/$DATASET_REL"
REMOTE_DATASET="$REMOTE_ROOT/$DATASET_REL"
LOCAL_TRAINER="$LOCAL_ROOT/scripts/experiments/personality/train_qwen35_pair_lora_sft.py"
REMOTE_TRAINER="$REMOTE_ROOT/scripts/experiments/personality/train_qwen35_pair_lora_sft.py"

if [[ ! -d "$LOCAL_DATASET" ]]; then
  echo "Dataset dir not found: $LOCAL_DATASET" >&2
  exit 1
fi

setsid -w "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$REMOTE_ROOT/scripts/experiments/personality' '$REMOTE_DATASET'"
setsid -w "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "cat > '$REMOTE_TRAINER'" < "$LOCAL_TRAINER"
tar -C "$LOCAL_DATASET" -cf - . | setsid -w "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "tar -C '$REMOTE_DATASET' -xf -"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd '$REMOTE_ROOT'
mkdir -p logs sweep_v4/${RUN_TAG}
tmux kill-session -t ${RUN_TAG}_trace 2>/dev/null || true
tmux kill-session -t ${RUN_TAG}_persona 2>/dev/null || true
pkill -f 'sglang.launch_server --model-path Qwen/Qwen3.5-9B' 2>/dev/null || true
sleep 3
CUDA_VISIBLE_DEVICES=0 tmux new-session -d -s ${RUN_TAG}_trace "cd '$REMOTE_ROOT' && '$REMOTE_VENV' scripts/experiments/personality/train_qwen35_pair_lora_sft.py --model-path '$MODEL_PATH' --train-file '$REMOTE_DATASET/student_trace_train.jsonl' --val-file '$REMOTE_DATASET/student_trace_val.jsonl' --output-dir 'sweep_v4/${RUN_TAG}/trace_student' --max-length '$MAX_LENGTH' --epochs '$EPOCHS' --lr '$LR' --batch-size '$BATCH_SIZE' --grad-accum '$GRAD_ACCUM' --sample-steps '$SAMPLE_STEPS' --save-steps '$SAVE_STEPS' --eval-steps '$EVAL_STEPS' > 'logs/${RUN_TAG}_trace.log' 2>&1"
CUDA_VISIBLE_DEVICES=1 tmux new-session -d -s ${RUN_TAG}_persona "cd '$REMOTE_ROOT' && '$REMOTE_VENV' scripts/experiments/personality/train_qwen35_pair_lora_sft.py --model-path '$MODEL_PATH' --train-file '$REMOTE_DATASET/student_personality_train.jsonl' --val-file '$REMOTE_DATASET/student_personality_val.jsonl' --output-dir 'sweep_v4/${RUN_TAG}/personality_student' --max-length '$MAX_LENGTH' --epochs '$EPOCHS' --lr '$LR' --batch-size '$BATCH_SIZE' --grad-accum '$GRAD_ACCUM' --sample-steps '$SAMPLE_STEPS' --save-steps '$SAVE_STEPS' --eval-steps '$EVAL_STEPS' > 'logs/${RUN_TAG}_persona.log' 2>&1"
tmux ls | grep -E '${RUN_TAG}_(trace|persona)'
EOF
)

setsid -w "${SSH_BASE[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_CMD"
echo "Remote launch started for $RUN_TAG"
