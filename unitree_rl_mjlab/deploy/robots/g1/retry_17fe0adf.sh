#!/usr/bin/env bash
#
# retry_17fe0adf.sh — 等现有 3 支舞(957d1c6f/93ccdf54/c2197703)全部结束后,
# 扫描空闲的两张卡(即没在跑 f85854e8 的卡), 用这两张卡一起(多卡)训练 17fe0adf。
# 若双卡训练失败(含瞬时 OOM), 自动降级为单卡重试。
#
# 用法 (建议放 tmux 里跑, 会先等 ~7.7h 再训练 ~7.7h):
#   tmux new-session -d -s retry17
#   tmux send-keys -t retry17 "./deploy/robots/g1/retry_17fe0adf.sh" C-m
#   tmux attach -t retry17
#
set -uo pipefail

# ============================================================
# 定位仓库 + 激活 conda
# ============================================================
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
RETRY_MOTION="src/assets/motions/g1/squat_17fe0adf.npz"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-10000}"

# 现在还在跑、需要等它们结束的 3 支舞 (用 pgrep 匹配命令行里的 motion_file)
WAIT_DANCES='squat_957d1c6f.npz|squat_93ccdf54.npz|squat_c2197703.npz'

log() { echo "[$(date '+%F %T')] $*"; }

# 空闲 GPU 列表 (nvidia-smi 上没有任何 compute 进程的物理卡)
free_gpus() {
  local free_list=() g n
  for g in 0 1 2; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null \
        | grep -cE '^[0-9]+$')
    [[ "$n" -eq 0 ]] && free_list+=("$g")
  done
  echo "${free_list[*]}"
}

# ============================================================
# 1) until: 等 3 支在跑的舞全部结束
# ============================================================
log "等待现有 3 支舞结束 (${WAIT_DANCES}) ..."
until [[ -z "$(pgrep -f "$WAIT_DANCES" 2>/dev/null)" ]]; do
  sleep 60
done
log "3 支舞已全部结束。"

# ============================================================
# 2) until: 等 f85854e8 被自动领取并启动, 恰好剩两张空闲卡
# ============================================================
log "等待 f85854e8 启动, 扫描空闲 GPU ..."
until [[ "$(free_gpus | wc -w)" -eq 2 ]]; do
  sleep 30
done
free_list="$(free_gpus)"
log "空闲 GPU: [${free_list}] (f85854e8 应在另一张卡上跑)"

# 给 f85854e8 的启动内存峰值留缓冲, 避免再次瞬时 OOM
log "再等 180s 让 f85854e8 进入稳态, 避免同时建环境触发 OOM ..."
sleep 180

# ============================================================
# 3) 训练函数 (参数: 逗号分隔的 GPU 列表, 单卡或双卡都走这里)
# ============================================================
run_train() {
  local gpus="$1"
  python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
    --motion_file="$RETRY_MOTION" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.max-iterations="$MAX_ITERS" \
    --agent.run-name=squat_17fe0adf \
    --gpu-ids "[$gpus]"
}

# ============================================================
# 4) 双卡训练, 失败(含 OOM)自动降级单卡重试
# ============================================================
gpu_list="$(echo "${free_list}" | tr ' ' ',')"
log "用双卡 GPU=[${gpu_list}] 训练 17fe0adf (num_envs=${NUM_ENVS}/卡, ${MAX_ITERS} iter) ..."
run_train "$gpu_list"
rc=$?
if [[ $rc -eq 0 ]]; then
  log "17fe0adf 双卡训练成功完成"
else
  log "双卡训练失败 (exit=$rc), 等 60s 让 GPU 清理后降级为单卡重试 ..."
  sleep 60
  single_gpu="$(free_gpus)"
  single_gpu="${single_gpu%% *}"          # 取第一张空闲卡
  if [[ -z "$single_gpu" ]]; then
    log "[ERROR] 无空闲卡可降级, 请手动处理 17fe0adf"
    exit 1
  fi
  log "用单卡 GPU=[$single_gpu] 重试 17fe0adf ..."
  run_train "$single_gpu"
  log "17fe0adf 单卡训练结束 (exit=$?)"
fi
