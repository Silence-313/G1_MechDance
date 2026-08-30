#!/usr/bin/env bash
# 完整编排：渲染当前两个训练(95/17) → 都完成后启动新训练(93cc/c2197) → 渲染新训练
# 在训练服务器 unitree_rl_mjlab 目录下运行。
set -uo pipefail

cd /public/home/zhaozhiyao/MechDance/unitree_rl_mjlab
LOG_ROOT="logs/rsl_rl/g1_tracking"

log() { echo "[$(date '+%F %T')] $*"; }

# 从某个 run 的 env.yaml 里提取 motion 文件名（不确定 95/17 哪个是哪个，读文件最准）
motion_of() {
  grep -oE 'knee_bend_[a-z0-9]+\.npz' "$LOG_ROOT/$1/params/env.yaml" 2>/dev/null | head -1
}

# 等某 run 出现 model_20000.pt 就渲染
wait_render() {
  local run="$1" motion="$2" out="$3" gpu="$4"
  log "等待 $run (motion=$motion) 生成 model_20000.pt ..."
  until [ -f "$LOG_ROOT/$run/model_20000.pt" ]; do sleep 60; done
  log "$run 完成，开始渲染 -> $out (cuda:$gpu)"
  xvfb-run -a python scripts/render_policy_video.py Unitree-G1-Tracking-No-State-Estimation \
    --checkpoint-file "$LOG_ROOT/$run/model_20000.pt" \
    --motion-file "src/assets/motions/g1/$motion" \
    --video-length 3200 --video-path "$out" \
    --video-width 640 --video-height 360 --device "cuda:$gpu"
  log "渲染完成 $out"
}

# ================= 阶段 1：渲染当前两个训练 =================
RUN_A="2026-08-16_11-26-04"
RUN_B="2026-08-16_11-26-49"
MA=$(motion_of "$RUN_A")
MB=$(motion_of "$RUN_B")
log "检测到: $RUN_A -> ${MA:-未知} ; $RUN_B -> ${MB:-未知}"

wait_render "$RUN_A" "$MA" "render_${MA%.npz}.mp4" 0 &
P1=$!
wait_render "$RUN_B" "$MB" "render_${MB%.npz}.mp4" 1 &
P2=$!
wait "$P1" "$P2"
log "阶段1完成：95/17 两个都渲染完"

# ================= 阶段 2：启动新训练 93ccdf54 / c2197703 =================
log "启动新训练 93ccdf54(GPU0) / c2197703(GPU1) ..."
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/knee_bend_93ccdf54.npz \
  --env.scene.num-envs=4096 --agent.max-iterations=20001 --gpu-ids '[0]' \
  --agent.run-name=93ccdf54 > train_93cc.log 2>&1 &

python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/knee_bend_c2197703.npz \
  --env.scene.num-envs=4096 --agent.max-iterations=20001 --gpu-ids '[1]' \
  --agent.run-name=c2197703 > train_c2197.log 2>&1 &

# 等新 run 目录出现（train.py 启动后立刻建目录并写 env.yaml）
sleep 20
RUN_93=$(ls -dt "$LOG_ROOT"/*_93ccdf54/ 2>/dev/null | head -1); RUN_93=${RUN_93%/}; RUN_93=${RUN_93##*/}
RUN_C2=$(ls -dt "$LOG_ROOT"/*_c2197703/ 2>/dev/null | head -1); RUN_C2=${RUN_C2%/}; RUN_C2=${RUN_C2##*/}
log "新训练 run 目录: 93=$RUN_93 ; c2197=$RUN_C2"

# ================= 阶段 3：渲染新训练 =================
if [ -n "$RUN_93" ] && [ -n "$RUN_C2" ]; then
  wait_render "$RUN_93" "knee_bend_93ccdf54.npz" "render_93ccdf54.mp4" 0 &
  P3=$!
  wait_render "$RUN_C2" "knee_bend_c2197703.npz" "render_c2197703.mp4" 1 &
  P4=$!
  wait "$P3" "$P4"
  log "阶段3完成：新训练都渲染完"
else
  log "警告：没定位到新训练 run 目录，请手动检查 train_93cc.log / train_c2197.log"
fi

log "全部完成"
