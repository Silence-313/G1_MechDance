#!/usr/bin/env bash
# 实时监测训练 checkpoint 是否生成，生成后立即渲染视频。
#
# 用法: render_when_done.sh <run_dir> <motion_npz> <output_mp4> [gpu_id] [target_iter]
#   run_dir      logs/rsl_rl/g1_tracking/ 下的 run 目录名（如 2026-08-16_06-06-03）
#   motion_npz   该 run 对应的参考动作 npz（相对 unitree_rl_mjlab 目录）
#   output_mp4   输出视频名
#   gpu_id       渲染用哪张卡（默认 0）
#   target_iter  等待哪个 checkpoint（默认 model_20000.pt）
#
# 在 unitree_rl_mjlab 目录下运行（和 train.py 同一目录）。
set -uo pipefail

RUN_DIR="$1"
MOTION="$2"
OUTPUT="$3"
GPU="${4:-0}"
TARGET="${5:-model_20000.pt}"

CKPT_PATH="logs/rsl_rl/g1_tracking/${RUN_DIR}/${TARGET}"

echo "[$(date '+%F %T')] 监测 ${CKPT_PATH} ..."
while [ ! -f "$CKPT_PATH" ]; do
  echo "[$(date '+%T')] 尚未生成，60s 后再查"
  sleep 60
done
echo "[$(date '+%T')] 检测到 ${TARGET}，开始渲染 ${OUTPUT} (cuda:${GPU})"

xvfb-run -a python scripts/render_policy_video.py Unitree-G1-Tracking-No-State-Estimation \
  --checkpoint-file "$CKPT_PATH" \
  --motion-file "$MOTION" \
  --video-length 3200 \
  --video-path "$OUTPUT" \
  --video-width 640 --video-height 360 \
  --device "cuda:${GPU}"

rc=$?
if [ $rc -eq 0 ]; then
  echo "[$(date '+%T')] 渲染完成: ${OUTPUT}"
else
  echo "[$(date '+%T')] 渲染失败 (exit ${rc})"
fi
