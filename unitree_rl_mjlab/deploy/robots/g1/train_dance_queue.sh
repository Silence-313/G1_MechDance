#!/usr/bin/env bash
#
# train_dance_queue.sh — 在训练服务器上, 用 3 张 L20 排队训练多支舞, 训练完一支自动接下一支。
#
# 两种调度模式 (MODE):
#   parallel (默认): 3 张卡各跑一支舞, 谁先跑完就自动从队列里取下一支 (吞吐最大)。
#   single          : 3 张卡同时服务同一支舞 (torchrunx 多卡), 一支接一支顺序跑。
#
# 队列: 5 支舞 (4 支 squat_*.npz + 新舞 squat_f85854e8...), 均已做半蹲防超伸处理。
#
# 用法 (在训练服务器 unitree_rl_mjlab 目录下, conda activate unitree_rl_mjlab):
#   # 3 卡并行 + 自动接续 (默认)
#   ./deploy/robots/g1/train_dance_queue.sh
#
#   # 3 卡服务同一支舞 (顺序)
#   MODE=single ./deploy/robots/g1/train_dance_queue.sh
#
#   # 调整 num_envs (默认 8192, 已按 benchmark 饱和点定好)
#   NUM_ENVS=8192 ./deploy/robots/g1/train_dance_queue.sh
#
# 说明:
#   - 脚本会新建 tmux 会话 "dance_train" (3 个 pane), 并 attach 进去; 断线后 tmux 仍在跑。
#   - 每张卡的 worker 日志写 logs/queue_worker_gpu<N>.log。
#   - 训练日志照常在 logs/rsl_rl/g1_tracking/<时间戳>_<run-name>/。
#
set -uo pipefail

# ============================================================
# 自动定位仓库根目录
# ============================================================
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$RL_DIR"

# ============================================================
# 激活 conda 环境。tmux 派生的 worker 是干净 shell, 不会继承交互式 shell 里
# 已 activate 的环境, 必须显式激活; 否则 python 会落到 base/系统环境导致 import 失败。
# 可用 CONDA_ENV / CONDA_BASE 覆盖。
# ============================================================
CONDA_ENV="${CONDA_ENV:-unitree_rl_mjlab}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  for _cb in "${CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [[ -n "$_cb" && -f "$_cb/etc/profile.d/conda.sh" ]] && { source "$_cb/etc/profile.d/conda.sh"; break; }
  done
fi
if conda activate "$CONDA_ENV" 2>/dev/null; then
  echo "[conda] 已激活 $CONDA_ENV (python=$(command -v python))"
else
  echo "[WARN] 未能激活 conda 环境 '$CONDA_ENV', 当前 python=$(command -v python 2>/dev/null || echo 无)"
fi

# ============================================================
# 配置 (可用环境变量覆盖)
# ============================================================
TASK="${TASK:-Unitree-G1-Tracking-No-State-Estimation}"
MODE="${MODE:-parallel}"                          # parallel | single
GPUS="${GPUS:-0 1 2}"                             # 用哪几张物理卡 (空格分隔)
NUM_ENVS="${NUM_ENVS:-8192}"                      # 每张卡的并行环境数 (两种模式都是"每卡"; single 模式 3 卡等效 batch = 单卡×3)
MAX_ITERS="${MAX_ITERS:-10000}"                   # 训练迭代数
SESSION="${SESSION:-dance_train}"                 # tmux 会话名

# ============================================================
# 任务队列 (一行一个 npz 动作路径, 直接训练; run-name 自动取文件名)
#   新增/删减训练只改这里。
# ============================================================
declare -a JOBS=(
  "src/assets/motions/g1/squat_957d1c6f.npz"
  "src/assets/motions/g1/squat_17fe0adf.npz"
  "src/assets/motions/g1/squat_93ccdf54.npz"
  "src/assets/motions/g1/squat_c2197703.npz"
  "src/assets/motions/g1/squat_f85854e8b56c0af7ae94f60921563ff7.npz"
)

log() { echo "[$(date '+%F %T')] $*"; }

# ============================================================
# 单任务执行
# ============================================================
# 训练一个 npz 动作。$2 为空表示 single 模式 (用全部 GPUS)。
train_npz() {
  local motion="$1" gpu="${2:-}"
  local name; name="$(basename "$motion")"; name="${name%.npz}"
  local gpu_args=()
  if [[ -z "$gpu" ]]; then
    # single 模式: 多卡。tyro 开了 UsePythonSyntaxForLiteralCollections, 必须是 "[0, 1, 2]"
    gpu_args+=(--gpu-ids "[$(echo ${GPUS} | tr ' ' ',')]")
    log "开始训练 motion=$motion (run=$name) 多卡 GPU=($GPUS)"
  else
    gpu_args+=(--gpu-ids "[$gpu]")                 # parallel 模式: 单卡 (--gpu-ids "[0]")
    log "开始训练 motion=$motion (run=$name) GPU=$gpu"
  fi
  python scripts/train.py "$TASK" \
    --motion_file="$motion" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.max-iterations="$MAX_ITERS" \
    --agent.run-name="$name" \
    "${gpu_args[@]}"
  log "训练结束 motion=$motion (exit=$?)"
}

# ============================================================
# 队列 (flock 原子取任务)。
# 队列文件放在固定路径, 由 launch() 一次性写入, 各 worker 进程共享同一份;
# 不能每个 worker 各建一份, 否则会重复领取同一个任务。
# ============================================================
QDIR="$RL_DIR/logs/dance_queue"
QUEUE="$QDIR/queue"; LOCK="$QDIR/lock"

pop_job() {
  # 在 flock 子 shell 内打印队首并删除之, 父 shell 通过 $(pop_job) 捕获输出。
  (
    flock -x 9
    if [[ -s "$QUEUE" ]]; then
      head -n1 "$QUEUE"
      sed -i '1d' "$QUEUE"
    fi
  ) 9>"$LOCK"
}

# ============================================================
# worker: parallel 模式, 每卡一个, 循环取任务直到队列空
# ============================================================
worker() {
  local gpu="$1"
  local logfile="$RL_DIR/logs/queue_worker_gpu${gpu}.log"
  local job
  while true; do
    job="$(pop_job)"
    [[ -z "$job" ]] && { log "GPU $gpu: 队列已空, worker 退出" | tee -a "$logfile"; break; }
    log "GPU $gpu: 领取任务 [$job]" | tee -a "$logfile"
    train_npz "$job" "$gpu" 2>&1 | tee -a "$logfile"
  done
}

# ============================================================
# worker: single 模式, 顺序多卡跑
# ============================================================
worker_single() {
  local logfile="$RL_DIR/logs/queue_worker_multi.log"
  local job
  while true; do
    job="$(pop_job)"
    [[ -z "$job" ]] && { log "队列已空, worker 退出" | tee -a "$logfile"; break; }
    log "领取任务 [$job] (多卡 GPU=($GPUS))" | tee -a "$logfile"
    train_npz "$job" "" 2>&1 | tee -a "$logfile"
  done
}

# ============================================================
# tmux 编排
# ============================================================
launch() {
  command -v tmux >/dev/null 2>&1 || { echo "[ERROR] 需要 tmux"; exit 1; }
  log "模式: $MODE  任务数: ${#JOBS[@]}  GPU: $GPUS  num_envs: $NUM_ENVS  iters: $MAX_ITERS"

  # 一次性写入共享队列 (先于任何 worker 启动, 避免竞争)
  mkdir -p "$QDIR"
  : > "$QUEUE"
  touch "$LOCK"
  for j in "${JOBS[@]}"; do echo "$j" >> "$QUEUE"; done

  tmux kill-session -t "$SESSION" 2>/dev/null
  tmux new-session -d -s "$SESSION"

  if [[ "$MODE" == "parallel" ]]; then
    local i=0 gpu
    for gpu in $GPUS; do
      if [[ $i -eq 0 ]]; then
        tmux send-keys -t "$SESSION" "$SELF --worker $gpu" C-m
      else
        tmux split-window -t "$SESSION"
        tmux select-layout -t "$SESSION" tiled
        tmux send-keys -t "$SESSION" "$SELF --worker $gpu" C-m
      fi
      i=$((i+1))
    done
  else
    tmux send-keys -t "$SESSION" "$SELF --worker-single" C-m
  fi

  log "已创建 tmux 会话 '$SESSION', attach 中... (脱离: Ctrl-b d)"
  tmux attach -t "$SESSION"
}

# ============================================================
# 入口
# ============================================================
case "${1:-}" in
  --worker)        worker "${2:?需要 GPU id}";;
  --worker-single) worker_single;;
  "")              launch;;
  -h|--help)       sed -n '2,40p' "$0";;
  *) echo "未知参数: $1"; exit 1;;
esac
