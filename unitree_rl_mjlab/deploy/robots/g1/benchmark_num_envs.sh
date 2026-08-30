#!/usr/bin/env bash
#
# benchmark_num_envs.sh — 测出一张 NVIDIA L20 (46GB) 能跑满多少环境 (num_envs)。
#
# 目的:
#   RL 训练吞吐受 num_envs 影响: env 太少 GPU 吃不饱, 太多则 OOM 或吞吐不再上升。
#   本脚本在一张卡上依次用不同的 num_envs 跑一小段训练, 采样 nvidia-smi,
#   汇报每个 num_envs 的峰值显存 / GPU 利用率 / 训练速度, 帮用户找到饱和点。
#
# 用法 (在训练服务器 unitree_rl_mjlab 目录下, conda activate unitree_rl_mjlab):
#   ./deploy/robots/g1/benchmark_num_envs.sh                       # 默认参数在 GPU 0 上扫
#   ./deploy/robots/g1/benchmark_num_envs.sh --gpu 1               # 在 GPU 1 上扫
#   ./deploy/robots/g1/benchmark_num_envs.sh --envs "2048 4096 8192"
#   ./deploy/robots/g1/benchmark_num_envs.sh --iters 50
#
# 并行加速: 三张卡各扫不同 num_envs (放 tmux):
#   tmux new-session -d -s bench
#   tmux send-keys -t bench "./deploy/robots/g1/benchmark_num_envs.sh --gpu 0 --envs '2048 4096 8192'" C-m
#   tmux split-window -t bench
#   tmux send-keys -t bench "./deploy/robots/g1/benchmark_num_envs.sh --gpu 1 --envs '4096 8192 12288'" C-m
#   tmux split-window -t bench
#   tmux send-keys -t bench "./deploy/robots/g1/benchmark_num_envs.sh --gpu 2 --envs '8192 12288 16384'" C-m
#
# 结果判读:
#   - GPU 利用率 ~100% 且再加大 num_envs 吞吐不再上升 -> 这就是饱和点
#   - 显存 OOM (报错 / 进程被杀)                       -> 该 num_envs 超限, 取上一个
#   - env_steps/s 随 num_envs 线性上升                 -> 还没饱和, 可继续加大
#
set -uo pipefail

# ============================================================
# 配置
# ============================================================
# 自动定位仓库根目录 (本脚本在 deploy/robots/g1/ 下)
RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$RL_DIR"

# ============================================================
# 激活 conda 环境 (让脚本不依赖交互式 shell 里预先 activate)。
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

TASK="Unitree-G1-Tracking-No-State-Estimation"
MOTION="${MOTION:-src/assets/motions/g1/squat_957d1c6f.npz}"   # 用真实训练动作更准
ENVS_LIST="${ENVS_LIST:-2048 4096 8192 12288 16384}"            # 要测的 num_envs
ITERS="${ITERS:-30}"                                            # 每个 num_envs 跑多少 iter (够到稳态即可)
GPU="${GPU:-0}"                                                 # 用哪张物理卡
STEPS_PER_ENV=24                                                # num_steps_per_env (PPO horizon)

log() { echo "[$(date '+%F %T')] $*"; }

# ============================================================
# 参数解析
# ============================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --envs)   ENVS_LIST="$2"; shift 2 ;;
    --iters)  ITERS="$2"; shift 2 ;;
    --gpu)    GPU="$2"; shift 2 ;;
    --motion) MOTION="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] 找不到 nvidia-smi (不是训练服务器?)"; exit 1
fi
if [[ ! -f "$MOTION" ]]; then
  echo "[ERROR] motion 不存在: $MOTION"; exit 1
fi

VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | head -1)
log "仓库: $RL_DIR"
log "GPU $GPU 总显存: ${VRAM_GB} MiB"
log "待测 num_envs: $ENVS_LIST  每个跑 $ITERS iter"

# ============================================================
# 单次测试
# ============================================================
bench_one() {
  local N="$1"
  local tag="bench_gpu${GPU}_env${N}"
  local sample sample_pid
  sample="$(mktemp)"
  log "── 测试 num_envs=$N (iter=$ITERS) ──"

  # 后台每 2s 采样一次显存/利用率
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 2 -i "$GPU" > "$sample" &
  sample_pid=$!

  local t0 t1 rc
  t0=$(date +%s)
  python scripts/train.py "$TASK" \
    --motion_file="$MOTION" \
    --env.scene.num-envs="$N" \
    --agent.max-iterations="$ITERS" \
    --agent.run-name="$tag" \
    --gpu-ids "[$GPU]" > "/tmp/${tag}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  kill "$sample_pid" 2>/dev/null
  wait "$sample_pid" 2>/dev/null

  local dt=$(( t1 - t0 ))
  if [[ $rc -ne 0 ]]; then
    log "  num_envs=$N 失败 (rc=$rc, 耗时 ${dt}s) — 大概率 OOM, 见 /tmp/${tag}.log 末尾:"
    tail -n 5 "/tmp/${tag}.log" | sed 's/^/    /'
    echo "N=$N STATUS=FAILED wall=${dt}s"
    return
  fi

  # 汇总采样: 峰值显存 / 峰值利用率
  local peak_mem peak_util
  peak_mem=$(awk -F', *' '{if($1>m)m=$1} END{print m+0}' "$sample")
  peak_util=$(awk -F', *' '{if($2>u)u=$2} END{print u+0}' "$sample")

  # 吞吐: env_steps/s = N * STEPS_PER_ENV * iters / wall
  local iter_s eps
  iter_s=$(awk -v a="$ITERS" -v t="$dt" 'BEGIN{printf "%.3f", a/t}')
  eps=$(awk -v n="$N" -v h="$STEPS_PER_ENV" -v t="$dt" -v a="$ITERS" \
    'BEGIN{printf "%.0f", n*h*a/t}')

  log "  num_envs=$N  峰值显存=${peak_mem}MiB (${VRAM_GB}MiB)  峰值利用率=${peak_util}%"
  log "               耗时=${dt}s  iter/s=$iter_s  env_steps/s=$eps"
  echo "N=$N MEM_MiB=$peak_mem UTIL=$peak_util wall=${dt}s iter_s=$iter_s env_steps_s=$eps"
  rm -f "$sample"
}

# ============================================================
# 主流程
# ============================================================
echo "==== 结果汇总 (N=num_envs, MEM=峰值显存MiB, UTIL=峰值利用率%) ===="
for n in $ENVS_LIST; do
  bench_one "$n"
done
log "全部完成。找到: 利用率接近 100% 且 env_steps/s 不再随 N 明显上升的那个 num_envs 即饱和点。"
