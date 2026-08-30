#!/usr/bin/env bash
#
# run_video_to_rl.sh — 一次完成 视频 → GVHMR(人体动作) → GMR(G1动作) → CSV → NPZ → RL训练
#
# 使用示例:
#   ./run_video_to_rl.sh --video /path/to/dance.mp4
#   ./run_video_to_rl.sh --video /path/to/dance.mp4 --name my_dance --num-envs 1024
#   ./run_video_to_rl.sh --video x.mp4 --no-gvhmr --no-gmr   # 只做 CSV→NPZ→训练（复用已有 hmr4d/g1 产物）
#
# 依赖三个 conda 环境（可用环境变量覆盖）:
#   gvhmr            -> 视频 → 人体动作  (tools/demo/demo.py)
#   gmr              -> 人体动作 → G1 动作 (scripts/gvhmr_to_robot.py)
#   unitree_rl_mjlab -> CSV → NPZ + RL 训练 (scripts/csv_to_npz.py / train.py)

set -euo pipefail

# ============================================================
# 配置（可用环境变量覆盖，适配不同机器）
# ============================================================
GVHMR_DIR="${GVHMR_DIR:-/home/silence/GVHMR}"
GMR_DIR="${GMR_DIR:-/home/silence/GMR}"
RL_DIR="${RL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

GVHMR_ENV="${GVHMR_ENV:-gvhmr}"
GMR_ENV="${GMR_ENV:-gmr}"
RL_ENV="${RL_ENV:-unitree_rl_mjlab}"

# ============================================================
# 参数解析
# ============================================================
VIDEO=""
NAME=""
ROBOT="g1"            # g1 或 g1_23dof
INPUT_FPS=30
OUTPUT_FPS=50
NUM_ENVS=1024
DO_GVHMR=1
DO_GMR=1
DO_TRAIN=1
KEEP_TMP=0

usage() {
  cat <<EOF
用法: $0 [选项]
  --video <path>        输入视频（必填）
  --name <name>         动作名，默认取视频文件名
  --robot <g1|g1_23dof> 机器人，默认 g1
  --input-fps <N>       GMR 输出 fps（默认 30）
  --output-fps <N>      RL motion fps（默认 50）
  --num-envs <N>        训练并行环境数（默认 1024）
  --no-gvhmr            跳过 GVHMR（复用已有 hmr4d_results.pt）
  --no-gmr              跳过 GMR（复用已有 G1 pkl/csv）
  --no-train            只生成 npz，不启动训练
  --keep-tmp            保留中间临时目录
  -h, --help            帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)      VIDEO="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --robot)      ROBOT="$2"; shift 2 ;;
    --input-fps)  INPUT_FPS="$2"; shift 2 ;;
    --output-fps) OUTPUT_FPS="$2"; shift 2 ;;
    --num-envs)   NUM_ENVS="$2"; shift 2 ;;
    --no-gvhmr)   DO_GVHMR=0; shift ;;
    --no-gmr)     DO_GMR=0; shift ;;
    --no-train)   DO_TRAIN=0; shift ;;
    --keep-tmp)   KEEP_TMP=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "未知参数: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "[ERROR] 必须提供 --video"
  usage; exit 1
fi
# 转绝对路径（避免 GVHMR 等子目录 cd 后找不到相对路径）
VIDEO="$(realpath "$VIDEO")"

if [[ ! -f "$VIDEO" ]]; then
  echo "[ERROR] 视频不存在: $VIDEO"
  exit 1
fi

STEM="$(basename "$VIDEO")"
STEM="${STEM%.*}"
NAME="${NAME:-$STEM}"

# 校验 robot
if [[ "$ROBOT" != "g1" && "$ROBOT" != "g1_23dof" ]]; then
  echo "[ERROR] --robot 仅支持 g1 / g1_23dof，得到: $ROBOT"
  exit 1
fi

echo "============================================================"
echo " 视频 → RL 全流程"
echo " 视频 : $VIDEO"
echo " 名称 : $NAME  机器人: $ROBOT"
echo " GVHMR: $GVHMR_DIR ($GVHMR_ENV)"
echo " GMR  : $GMR_DIR ($GMR_ENV)"
echo " RL   : $RL_DIR ($RL_ENV)"
echo "============================================================"

# 环境执行辅助（每个命令在独立子 shell 中 activate，互不干扰）
run_in_env() {
  local env="$1"; shift
  echo "  [conda] 激活环境: $env"
  (
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env"
    echo "  [conda] 当前环境: $CONDA_DEFAULT_ENV (python: $(which python))"
    "$@"
  )
}

# ============================================================
# Step 1: GVHMR — 视频 → 人体 SMPL-X 动作
# ============================================================
HMR4D_RESULTS="$GVHMR_DIR/outputs/demo/$STEM/hmr4d_results.pt"

if [[ "$DO_GVHMR" -eq 1 ]]; then
  echo ""
  echo "[Step 1/4] GVHMR: 视频 → 人体动作"
  echo "  运行: python tools/demo/demo.py --video=$VIDEO -s"
  (
    cd "$GVHMR_DIR"
    run_in_env "$GVHMR_ENV" python tools/demo/demo.py --video="$VIDEO" -s
  )
  if [[ ! -f "$HMR4D_RESULTS" ]]; then
    echo "[ERROR] GVHMR 未生成 $HMR4D_RESULTS"
    exit 1
  fi
  echo "[Step 1] 完成: $HMR4D_RESULTS"
else
  echo "[Step 1/4] 跳过 GVHMR（期望 $HMR4D_RESULTS）"
  [[ -f "$HMR4D_RESULTS" ]] || { echo "[ERROR] 缺少 $HMR4D_RESULTS，无法继续"; exit 1; }
fi

# ============================================================
# Step 2: GMR — 人体动作 → G1 动作 (pkl → csv)
# ============================================================
G1_PKL="$GMR_DIR/outputs/$NAME.pkl"
TMP_DIR="$GMR_DIR/outputs/tmp_$NAME"
G1_CSV="$TMP_DIR/csv/$NAME.csv"

if [[ "$DO_GMR" -eq 1 ]]; then
  echo ""
  echo "[Step 2/4] GMR: 人体动作 → G1 动作"
  echo "  运行: python scripts/gvhmr_to_robot.py --gvhmr_pred_file=$HMR4D_RESULTS --robot=unitree_g1 --save_path=$G1_PKL"
  (
    cd "$GMR_DIR"
    run_in_env "$GMR_ENV" python scripts/gvhmr_to_robot.py \
      --gvhmr_pred_file "$HMR4D_RESULTS" \
      --robot "unitree_g1" \
      --save_path "$G1_PKL"
  )
  if [[ ! -f "$G1_PKL" ]]; then
    echo "[ERROR] GMR 未生成 $G1_PKL"
    exit 1
  fi

  # pkl → csv（复用 batch_gmr_pkl_to_csv.py，在临时目录中只放本动作一个 pkl）
  echo "  转换 pkl → csv（临时目录 $TMP_DIR）"
  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"
  cp "$G1_PKL" "$TMP_DIR/$NAME.pkl"
  (
    cd "$GMR_DIR"
    run_in_env "$GMR_ENV" python scripts/batch_gmr_pkl_to_csv.py --folder "$TMP_DIR"
  )
  if [[ ! -f "$G1_CSV" ]]; then
    echo "[ERROR] 未生成 csv: $G1_CSV"
    exit 1
  fi
  echo "[Step 2] 完成: $G1_CSV"
else
  echo "[Step 2/4] 跳过 GMR（期望 $G1_CSV）"
  [[ -f "$G1_CSV" ]] || { echo "[ERROR] 缺少 $G1_CSV，无法继续"; exit 1; }
fi

# ============================================================
# Step 3: CSV → NPZ（RL 训练输入）
# ============================================================
MOTION_NPZ="$RL_DIR/src/assets/motions/g1/$NAME.npz"

echo ""
echo "[Step 3/4] CSV → NPZ"
echo "  运行: python scripts/csv_to_npz.py --input-file=$G1_CSV --output-name=$NAME --input-fps=$INPUT_FPS --output-fps=$OUTPUT_FPS --robot=$ROBOT"
(
  cd "$RL_DIR"
  run_in_env "$RL_ENV" python scripts/csv_to_npz.py \
    --input-file "$G1_CSV" \
    --output-name "$NAME" \
    --input-fps "$INPUT_FPS" \
    --output-fps "$OUTPUT_FPS" \
    --robot "$ROBOT"
)
if [[ ! -f "$MOTION_NPZ" ]]; then
  echo "[ERROR] 未生成 npz: $MOTION_NPZ"
  exit 1
fi
echo "[Step 3] 完成: $MOTION_NPZ"

# 清理临时目录
if [[ "$KEEP_TMP" -eq 0 ]]; then
  rm -rf "$TMP_DIR"
  echo "  (已清理临时目录 $TMP_DIR)"
fi

# ============================================================
# Step 4: RL 训练
# ============================================================
if [[ "$DO_TRAIN" -eq 1 ]]; then
  echo ""
  echo "[Step 4/4] RL 训练"
  echo "  运行: python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=$MOTION_NPZ --env.scene.num-envs=$NUM_ENVS"
  (
    cd "$RL_DIR"
    run_in_env "$RL_ENV" python scripts/train.py \
      Unitree-G1-Tracking-No-State-Estimation \
      --motion_file="$MOTION_NPZ" \
      --env.scene.num-envs="$NUM_ENVS"
  )
else
  echo ""
  echo "[Step 4/4] 跳过训练（--no-train）"
fi

echo ""
echo "============================================================"
echo " 全部完成！"
echo " 动作文件 : $MOTION_NPZ"
[[ "$DO_TRAIN" -eq 1 ]] && echo " 训练日志 : $RL_DIR/logs/rsl_rl/g1_tracking/<时间戳>/"
echo "============================================================"
