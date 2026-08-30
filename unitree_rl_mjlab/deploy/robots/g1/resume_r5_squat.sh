#!/usr/bin/env bash
#
# resume_r5_squat.sh — round 5: 用 v3 动作续训 4 支 + 原始动作续训 f85854e8。
#
# 动作修改 (仅开头):
#   957d1c6f (A): 删除 0.25s   -> squat_957d1c6f_v3.npz
#   93ccdf54 (X): 删除 1.25s   -> squat_93ccdf54_v3.npz
#   c2197703 (Y): 增加 1s      -> squat_c2197703_v3.npz
#   17fe0adf (B): 增加 1.25s   -> squat_17fe0adf_v3.npz
#   f85854e8 (left): 不改      -> squat_f85854e8b56c0af7ae94f60921563ff7.npz (原始)
#
# 续训起点 (r4 进行中的 checkpoint):
#   957d1c6f -> r4 model_22500 ; 93ccdf54 -> r4 model_27500 ; c2197703 -> r4 model_27500
#   17fe0adf -> r3 model_25000 ; f85854e8 -> r4 model_27500
#
# 轮数/终点:
#   957d1c6f 10001 -> 32500 ; 93ccdf54 10001 -> 37500 ; c2197703 10001 -> 37500
#   17fe0adf 2001  -> 27000 (快速, 只学新开头) ; f85854e8 10001 -> 37500
#
# 5 支舞 4 卡: 17fe0adf 最短, 跑完的 GPU 自动接着训 f85854e8。
# 错峰启动: GPU0=0s / GPU1=30s / GPU2=60s / GPU3=90s。
#
# 用法 (训练服务器, 需先把 4 个 v3 npz 放到 src/assets/motions/g1/):
#   bash deploy/robots/g1/resume_r5_squat.sh
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
GPUS="${GPUS:-0 1 2 3}"
NUM_ENVS="${NUM_ENVS:-8192}"
SESSION="${SESSION:-resume_r5}"
STAGGER="${STAGGER:-30}"

# ============================================================
# 续训队列 (5 支; 17fe0adf 最短, 其 GPU 跑完接着训 f85854e8)
# ============================================================
declare -a JOBS=(
  "squat_957d1c6f"
  "squat_93ccdf54"
  "squat_c2197703"
  "squat_17fe0adf"
  "squat_f85854e8b56c0af7ae94f60921563ff7"
)

# 每支舞的动作文件 (4 支用 v3, f85854e8 用原始)
declare -A MOTION=(
  ["squat_957d1c6f"]="squat_957d1c6f_v3.npz"
  ["squat_93ccdf54"]="squat_93ccdf54_v3.npz"
  ["squat_c2197703"]="squat_c2197703_v3.npz"
  ["squat_17fe0adf"]="squat_17fe0adf_v3.npz"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="squat_f85854e8b56c0af7ae94f60921563ff7.npz"
)

# 每支舞从哪一轮(r3/r4)续训
declare -A RESUME_ROUND=(
  ["squat_957d1c6f"]="r4"
  ["squat_93ccdf54"]="r4"
  ["squat_c2197703"]="r4"
  ["squat_17fe0adf"]="r3"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="r4"
)
# 每支舞的续训 checkpoint
declare -A RESUME_CKPT=(
  ["squat_957d1c6f"]="model_22500.pt"
  ["squat_93ccdf54"]="model_27500.pt"
  ["squat_c2197703"]="model_27500.pt"
  ["squat_17fe0adf"]="model_25000.pt"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="model_27500.pt"
)
# 每支舞的续训轮数
declare -A DANCE_ITERS=(
  ["squat_957d1c6f"]="10001"
  ["squat_93ccdf54"]="10001"
  ["squat_c2197703"]="10001"
  ["squat_17fe0adf"]="2001"
  ["squat_f85854e8b56c0af7ae94f60921563ff7"]="10001"
)

log() { echo "[$(date '+%F %T')] $*"; }

# ============================================================
# 续训单个舞 ($1=舞名, $2=gpu)
# ============================================================
resume_one() {
  local dance="$1" gpu="$2"
  local motion="${MOTION[$dance]}"
  local round="${RESUME_ROUND[$dance]}" ckpt="${RESUME_CKPT[$dance]}"
  local iters="${DANCE_ITERS[$dance]}"
  local name="${dance}_r5"
  log "续训 motion=$motion (run=${name}) GPU=$gpu, 从 ${round}/${ckpt} 再训 ${iters} 轮"
  python scripts/train.py "$TASK" \
    --motion_file="src/assets/motions/g1/$motion" \
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
QDIR="$RL_DIR/logs/resume_r5_queue"
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
  local logfile="$RL_DIR/logs/resume_r5_gpu${gpu}.log"
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
  log "r5 续训任务数: ${#JOBS[@]}  GPU: $GPUS  num_envs: $NUM_ENVS"

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
