# RL_CODE_CHANGE_REPORT

日期：2026-08-05
基准：commit `1425b15`
范围：仅本地代码修改。未运行训练、未修改服务器配置、未做 git commit/push、未修改 termination threshold / reward / PPO / motion。

---

## # Modified Files

| 文件 | 类型 | 修改 |
|------|------|------|
| `src/tasks/tracking/config/g1/rl_cfg.py` | 修改 | `init_std` 1.0 → 0.3 |
| `src/tasks/tracking/tracking_env_cfg.py` | 修改 | `num_envs` 1 → 512；`MotionCommandCfg` import 切换到本地 curriculum 版本 |
| `src/tasks/tracking/config/g1/env_cfgs.py` | 修改 | 新增 RSI curriculum 三阶段参数 |
| `src/tasks/tracking/mdp/curriculum_commands.py` | **新增** | 三阶段 RSI curriculum + reset stale-state 修复 |
| `src/tasks/tracking/mdp/terminations.py` | **无改动** | 已恢复到 HEAD（不修改 termination threshold） |

---

## # Change Summary

### 任务 1：降低 action exploration noise ✅

**文件**：`src/tasks/tracking/config/g1/rl_cfg.py`
**修改位置**：`unitree_g1_tracking_ppo_runner_cfg()` → actor `distribution_cfg`
**修改前**：`"init_std": 1.0`
**修改后**：`"init_std": 0.3`
**保持**：`std_type="scalar"`；未动 `entropy_coef`、`learning_rate`、其余 PPO config
**影响**：动作分布初始 std 从 1.0 → 0.3。29 维动作空间全部应用。有效关节噪声 = action_scale × std：
- 髋/膝 (scale 0.35)：0.35×0.3 ≈ **0.105 rad**
- 髋pitch/yaw、腰、肘、肩、踝 (scale 0.44~0.55)：**0.13~0.16 rad**
- 手腕 (scale 0.0745)：**0.022 rad**

### 任务 2：num_envs 默认配置 ✅

**num_envs 配置链**：
```
默认值唯一控制点：src/tasks/tracking/tracking_env_cfg.py:288
    SceneCfg(terrain=..., num_envs=1)   ← 改为 512
        ↓
register_mjlab_task(env_cfg=..., play_env_cfg=...)
        ↓
scripts/train.py: tyro 解析 --env.scene.num-envs=<N>（CLI 覆盖默认值）
scripts/play.py: --num_envs=<N>（CLI 覆盖）
```
**修改前**：`num_envs=1`（审计确认原训练误用此默认值，导致采样失效）
**修改后**：`num_envs=512`
**CLI 覆盖不受影响**：`python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --env.scene.num-envs=1024`
**注意**：play 模式也继承 512，但 `play.py --num_envs` 可覆盖（建议 play 用 `--num_envs=1` 或 `--num_envs=8`）

### 任务 3：RSI curriculum（三阶段）✅

**调用链变化**：
```
修改前：
  tracking_env_cfg.py: from mjlab.tasks.tracking.mdp import MotionCommandCfg
  → MotionCommandCfg.build() → mjlab MotionCommand（sampling_mode 固定 adaptive）
修改后：
  tracking_env_cfg.py: from src.tasks.tracking.mdp.curriculum_commands import MotionCommandCfg
  → MotionCommandCfg.build() → 本地 MotionCommand（继承 mjlab，override _resample_command）
  → 根据 common_step_counter 分三阶段强制采样模式
```

**三阶段逻辑**（`env_cfgs.py` 配置，`curriculum_commands.py` 实现）：
```
step <  24000  (iteration <  1000)  → sampling_mode "start"（帧 0，最简单）
24000 ≤ step < 120000 (1000≤iter<5000) → sampling_mode "uniform"（随机帧）
step ≥ 120000  (iteration ≥ 5000)   → sampling_mode "adaptive"（按失败 bin 加权）
```
- env step ↔ iteration 换算：`iteration = common_step_counter / num_steps_per_env(24)`
- 通过继承 mjlab `MotionCommand` 实现，**未复制整个 command 系统**

### 任务 4：reset stale state 修复 ✅

**问题确认**：`env.reset()`（初始化 reset）调用 `_reset_idx()` 后**不执行** `command_manager.compute()`，因此 `body_pos_relative_w` / `body_quat_relative_w` 停留在初始 zeros。第一次 `step()` 的 termination 检查（先于 `command_manager.compute()`）用 zeros 对比真实 `robot_body_pos_w`，手腕 z≈1.3m 远超 0.25m 阈值 → 第一帧即误触发 `ee_body_pos`。

**修复**：本地 `MotionCommand.reset()` 在 `super().reset()`（写 RSI 状态）后立即调用 `_refresh_relative_state()`，重算 relative body buffers。
- **未改变正常 simulation step 顺序**（`_update_command` / termination 时序不变）
- `_refresh_relative_state()` 复刻 mjlab `_update_command` 的相对位姿计算块（不含 `time_steps += 1`）

---

## # Reasoning

1. **action_std**：审计实测 `Policy/mean_std` 851 iter 从 1.0006→1.0023 不收缩；原 std=1.0 下有效噪声 0.44~0.55 rad 使 29 DOF 人形从 RSI 帧起步即失衡。降到 0.3 后噪声为 0.022~0.16 rad，与动作帧间跳变均值 0.027 rad 同量级。
2. **num_envs**：审计确认原训练 `env.yaml` 实测 `num_envs=1`。单 env 下 PPO 采样、RSI adaptive 统计均失效，是训练无进展的关键因素。
3. **RSI curriculum**：数据特性分析确认 test_g1_motion 帧 0 是难度最低帧（2% 分位，近站立）。三阶段让策略先学"存活于帧 0"，再逐步过渡到全动作随机帧，避免直接面对复杂舞蹈姿态。
4. **reset stale state**：初始化第一帧误触发 termination 会污染所有 episode 起点，放大"8 步倒"现象；修复后 RSI 后 relative 状态立即一致。

---

## # Potential Risk

| 风险 | 说明 | 缓解 |
|------|------|------|
| curriculum_commands 依赖 mjlab 内部 API | 继承 `mjlab.tasks.tracking.mdp.commands.MotionCommand`、import `mjlab.utils.lab_api.math` | mjlab==1.2.0 已固定；升级 mjlab 时需回归 |
| `_refresh_relative_state` 与 mjlab `_update_command` 未来不同步 | 复刻了相对位姿计算块 | 若 mjlab 更新该计算逻辑，需同步 |
| num_envs=512 默认增大 GPU 显存 | 512 env 追踪任务约需更多显存 | CLI 可调低；play 建议 `--num_envs` 覆盖 |
| 三阶段切换可能造成训练分布突变 | iter 1000 / 5000 处采样策略跳变 | 切换点为软设计，PPO 可自适应；观察 episode length 曲线 |
| 初始化 reset 刷新全部 env relative | 非 resampled env 多做一次计算 | 计算量极小，无副作用 |

---

## # Need Server Validation

以下需在服务器（有 GPU/EGL）上执行：

### 1. 环境启动测试
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --agent=zero \
  --motion_file=src/assets/motions/g1/test_g1_motion.npz --num_envs=8
```
预期：环境正常构建，无 import/config 报错，robot 从帧 0 附近起步。

### 2. 单 episode 观察
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --agent=random \
  --motion_file=src/assets/motions/g1/test_g1_motion.npz --num_envs=8 --no-terminations
```
预期：机器人能跟随动作前段（帧 0 附近姿态），不再首帧误判终止。

### 3. 短训练（观察前 500 iter）
```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/test_g1_motion.npz \
  --env.scene.num-envs=1024
```
检查：
- `Train/mean_episode_length` 应显著 >8 步（curriculum start 阶段）
- `Episode_Termination/ee_body_pos` 触发率应下降（stale-state 修复生效）
- `Policy/mean_std` 应从 0.3 起随训练调整
- iter 1000 附近（切 uniform）观察 episode length 是否回落（属正常，采样变难）

### 4. 回滚
所有改动未提交，`git checkout` 可完整回滚；新增 `curriculum_commands.py` 手动删除即可。

---

## # Git Diff Summary

```bash
$ git diff --stat
 src/tasks/tracking/config/g1/env_cfgs.py            |  8 ++++++++
 src/tasks/tracking/config/g1/rl_cfg.py              |  2 +-
 src/tasks/tracking/tracking_env_cfg.py              |  2 +-
 3 files changed, 11 insertions(+), 3 deletions(-)

# 新增（未跟踪）：
src/tasks/tracking/mdp/curriculum_commands.py        # ~160 行

# 无改动（已恢复 HEAD）：
src/tasks/tracking/mdp/terminations.py
```
