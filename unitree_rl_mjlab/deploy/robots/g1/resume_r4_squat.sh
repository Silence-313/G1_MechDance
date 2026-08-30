#!/usr/bin/env bash
#
# resume_r4_squat.sh — round 4 续训: 4 支舞各再训 10001 轮 (即 +10000 轮)。
#
# 续训起点 (注意 957d1c6f 上轮 r3 没参加, 起点不同):
#   957d1c6f  : r2 model_19998.pt (iter 19998) -> 终点 30000 (10003 轮)
#   93ccdf54  : r3 model_25000.pt (iter 25000) -> 终点 35000
#   c2197703  : r3 model_25000.pt (iter 25000) -> 终点 35000
#   f85854e8  : r3 model_25000.pt (iter 25000) -> 终点 35000
#   排除 17fe0adf (RB+B, 冻结)。
#
# 4 卡 L20, 每卡一支舞, 错峰启动: GPU0=0s / GPU1=30s / GPU2=60s / GPU3=90s。
#
# 用法 (在训练服务器):
#   bash deploy/robots/g1/resume_r4_squat.sh
#   GPUS="0 1" bash deploy/robots/g1/resume_r4_squat.sh
#
set -uo pipefail

# ============================================================
# 定位仓库 + 激活 conda
# ============================================================
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$RL_DIR"

CONDA_ENV="${CONDA_ENV:-unitree_rl_mjlab}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  for _cb in "${CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [[ -n "$_cb" && -f "$_cb/etc/profile.d/conda.sh" ]] && { source "$_cb/etc/profile.d/conda.sh"; break; }
  done
fi
conda activate "$CONDA_ENV" 2>/dev/null

# ============================================================
# 配置
# ============================================================
TASK="${TASK:-Unitree-G1-Tracking-No-State-Estimation}"
GPUS="${GPUS:-0 1 2 3}"              # 四卡全上
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-10001}"      # 各再训 10001 轮 (即 +10000 轮)
SESSION="${SESSION:-resume_r4}"
STAGGER="${STAGGER:-30}"             # 错峰: GPU i 等 i*30 秒

# ============================================================
# 续训队列 (排除 17fe0adf/RB+B)
# ============================================================
declare -a JOBS=(
  "squat_957d1c6f"
  "squat_93ccdf54"
  "squat_c2197703"
  "squat_f85854e8b56c0af7ae94f60921563ff7"
)

# 每支舞从哪一轮(r2/r3)的哪个 checkpoint 续训
declare -A RESUME_ROUND=(
  ["squat_957d1c6f"]="r2"
  ["squat_93ccdf54"]="r3"
  ["squat_c2197703"]="r3"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="r3"
)
declare -A RESUME_CKPT=(
  ["squat_957d1c6f"]="model_19998.pt"
  ["squat_93ccdf54"]="model_25000.pt"
  ["squat_c2197703"]="model_25000.pt"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="model_25000.pt"
)
# 每支舞的续训轮数 (957d1c6f 起点 19998, 要落到 30000 需 10003 轮)
declare -A DANCE_ITERS=(
  ["squat_957d1c6f"]="10003"
  ["squat_93ccdf54"]="10001"
  ["squat_c2197703"]="10001"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="10001"
)

log() { echo "[$(date '+%F %T')] $*"; }

# ============================================================
# 续训单个舞 ($1=舞名, $2=gpu)
# ============================================================
resume_one() {
  local dance="$1" gpu="$2"
  local round="${RESUME_ROUND[$dance]}" ckpt="${RESUME_CKPT[$dance]}"
  local iters="${DANCE_ITERS[$dance]:-$MAX_ITERS}"
  local motion="src/assets/motions/g1/${dance}.npz"
  local name="${dance}_r4"
  log "续训 motion=$motion (run=${name}) GPU=$gpu, 从 ${round}/${ckpt} 再训 ${iters} 轮"
  python scripts/train.py "$TASK" \
    --motion_file="$motion" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.max-iterations="$iters" \
    --agent.run-name="$name" \
    --gpu-ids "[$gpu]" \
    --agent.resume=True \
    --agent.load_run=".*${dance}_${round}\$" \
    --agent.load_checkpoint="$ckpt"
  log "续训结束 motion=$motion (exit=$?)"
}

# ============================================================
# 队列 (flock 原子取任务)
# ============================================================
QDIR="$RL_DIR/logs/resume_r4_queue"
QUEUE="$QDIR/queue"; LOCK="$QDIR/lock"

pop_job() {
  (
    flock -x 9
    if [[ -s "$QUEUE" ]]; then
      head -n1 "$QUEUE"
      sed -i '1d' "$QUEUE"
    fi
  ) 9>"$LOCK"
}

# ============================================================
# worker: 每卡一个, 循环取任务直到队列空
# ============================================================
worker() {
  local gpu="$1"
  local logfile="$RL_DIR/logs/resume_r4_gpu${gpu}.log"
  local job delay
  while true; do
    job="$(pop_job)"
    [[ -z "$job" ]] && { log "GPU $gpu: 队列已空, worker 退出" | tee -a "$logfile"; break; }
    log "GPU $gpu: 领取任务 [$job]" | tee -a "$logfile"
    delay=$((gpu * STAGGER))
    if [[ "$delay" -gt 0 ]]; then
      log "GPU $gpu: 错峰启动, 等 ${delay}s 再开始 (避免同时建环境 OOM)" | tee -a "$logfile"
      sleep "$delay"
    fi
    resume_one "$job" "$gpu" 2>&1 | tee -a "$logfile"
  done
}

# ============================================================
# tmux 编排
# ============================================================
launch() {
  command -v tmux >/dev/null 2>&1 || { echo "[ERROR] 需要 tmux"; exit 1; }
  log "r4 续训任务数: ${#JOBS[@]}  GPU: $GPUS  num_envs: $NUM_ENVS  iters: $MAX_ITERS"

  mkdir -p "$QDIR"
  : > "$QUEUE"
  touch "$LOCK"
  for j in "${JOBS[@]}"; do echo "$j" >> "$QUEUE"; done

  tmux kill-session -t "$SESSION" 2>/dev/null
  tmux new-session -d -s "$SESSION"

  local i=0 gpu
  for gpu in $GPUS; do
    if [[ $i -eq 0 ]]; then
      tmux send-keys -t "$SESSION" "bash $SELF --worker $gpu" C-m
    else
      tmux split-window -t "$SESSION"
      tmux select-layout -t "$SESSION" tiled
      tmux send-keys -t "$SESSION" "bash $SELF --worker $gpu" C-m
    fi
    i=$((i+1))
  done

  log "已创建 tmux 会话 '$SESSION', attach 中... (脱离: Ctrl-b d)"
  tmux attach -t "$SESSION"
}

# ============================================================
# 入口
# ============================================================
case "${1:-}" in
  --worker) worker "${2:?需要 GPU id}";;
  "")       launch;;
  -h|--help) sed -n '2,40p' "$0";;
  *) echo "未知参数: $1"; exit 1;;
esac
