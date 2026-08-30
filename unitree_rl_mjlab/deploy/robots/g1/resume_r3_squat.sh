#!/usr/bin/env bash
#
# resume_r3_squat.sh — 从 model_19998.pt 断点续训, 每支舞再训 5003 轮到 iter=25000 (round 3)。
#
# 断点续训原理:
#   train.py 加 --agent.resume + --agent.load_run=<匹配第2轮run目录的正则> \
#   + --agent.load_checkpoint=model_19998.pt。runner.load() 会恢复 iter=19998,
#   再跑 max-iterations=5003 就接着训到 iter=25000, 最终落盘 model_25000.pt。
#
# 4 卡 L20 (GPU 0 1 2 3), 每卡一支舞, 错峰启动: GPU0=0s / GPU1=30s / GPU2=60s / GPU3=90s。
# 排除 957d1c6f (RB+A, 冻结, 不续训)。
#
# 用法 (在训练服务器):
#   bash deploy/robots/g1/resume_r3_squat.sh            # 默认四卡 0 1 2 3 全上
#   GPUS="0 1" bash deploy/robots/g1/resume_r3_squat.sh # 或指定用哪几张卡
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
GPUS="${GPUS:-0 1 2 3}"              # 用哪些物理卡 (四卡全上)
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-5003}"       # 再训 5003 轮: 19998 -> 25000
RESUME_CKPT="${RESUME_CKPT:-model_19998.pt}"
SESSION="${SESSION:-resume_r3}"
STAGGER="${STAGGER:-30}"             # 错峰启动: 每卡领任务后多等 gpu*STAGGER 秒 (GPU0=0s/GPU1=30s/GPU2=60s/GPU3=90s)

# ============================================================
# 续训队列 (每行一个舞名, 即 npz 文件名去掉 .npz; 排除 957d1c6f)
# ============================================================
declare -a JOBS=(
  "squat_17fe0adf"
  "squat_93ccdf54"
  "squat_c2197703"
  "squat_f85854e8b56c0af7ae94f60921563ff7"
)

log() { echo "[$(date '+%F %T')] $*"; }

# ============================================================
# 续训单个舞 ($1=舞名, $2=gpu)
# ============================================================
resume_one() {
  local dance="$1" gpu="$2"
  local motion="src/assets/motions/g1/${dance}.npz"
  local name="${dance}_r3"              # round3 用新 run-name, 避免和 round2 目录混淆
  log "续训 motion=$motion (run=${name}) GPU=$gpu, 从 $RESUME_CKPT 再训 ${MAX_ITERS} 轮 (到 iter=25000)"
  python scripts/train.py "$TASK" \
    --motion_file="$motion" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.max-iterations="$MAX_ITERS" \
    --agent.run-name="$name" \
    --gpu-ids "[$gpu]" \
    --agent.resume=True \
    --agent.load_run=".*${dance}_r2\$" \
    --agent.load_checkpoint="$RESUME_CKPT"
  log "续训结束 motion=$motion (exit=$?)"
}

# ============================================================
# 队列 (flock 原子取任务, 独立目录避免和 dance_train 冲突)
# ============================================================
QDIR="$RL_DIR/logs/resume_r3_queue"
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
  local logfile="$RL_DIR/logs/resume_r3_gpu${gpu}.log"
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
  log "续训任务数: ${#JOBS[@]}  GPU: $GPUS  num_envs: $NUM_ENVS  iters: $MAX_ITERS  ckpt: $RESUME_CKPT"

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
