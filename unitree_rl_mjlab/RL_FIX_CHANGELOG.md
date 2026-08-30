# RL_FIX_CHANGELOG

本文件记录针对 Unitree G1 模仿训练无法有效学习问题的全部代码修改。

修改基准：commit `1425b15`
备份位置：`backup_before_rl_fix/`（含全部被修改文件的原始版本）
日期：2026-08-05

> **重要更新（2026-08-05 第二轮）**：本轮按新指令，**不修改 termination threshold**。
> 原 M2（curriculum termination）已**全部撤销**：`terminations.py` 恢复 HEAD，`env_cfgs.py` 中
> termination 的 `ramp_iters`/`easy_threshold` 已移除。当前有效修改见下方 M1/M3/M4/M5，
> 以及 `RL_CODE_CHANGE_REPORT.md`。等待下一轮训练结果后再决定是否加入 termination curriculum。

---

## 修改总览（当前有效）

| # | 目标 | 文件 | 类型 | 状态 |
|---|------|------|------|------|
| M1 | 降低 action_std | `src/tasks/tracking/config/g1/rl_cfg.py` | config | 完成 |
| ~~M2~~ | ~~Curriculum termination~~ | ~~`terminations.py` + `env_cfgs.py`~~ | env code | **已撤销** |
| M3 | RSI curriculum（三阶段） | `src/tasks/tracking/mdp/curriculum_commands.py`(新增) + `tracking_env_cfg.py` + `env_cfgs.py` | env code | 完成 |
| M4 | num_envs 默认值 | `src/tasks/tracking/tracking_env_cfg.py` | config | 完成（1→512） |
| M5 | reset stale state 修复 | `src/tasks/tracking/mdp/curriculum_commands.py` | env code | 完成 |

未修改：reward 权重、termination threshold、mjlab 核心包、PPO、motion 数据。

---

## M3. RSI curriculum（三阶段，替换原两阶段）

**文件**：`src/tasks/tracking/mdp/curriculum_commands.py`（重写）、`tracking_env_cfg.py`（import）、`env_cfgs.py`（参数）

**三阶段逻辑**（env step ↔ iteration：iteration = step / num_steps_per_env(24)）：
```
step <  24000   (iteration <  1000) → sampling_mode "start"
24000 ≤ step < 120000 (1000≤iter<5000) → sampling_mode "uniform"
step ≥ 120000   (iteration ≥ 5000)  → sampling_mode "adaptive"
```

**调用链**：`tracking_env_cfg.py` import 本地 `MotionCommandCfg`（继承 mjlab）→ `build()` 返回本地 `MotionCommand` → `_resample_command()` 按 `_env.common_step_counter` 强制阶段模式。

## M5. reset stale state 修复（新增）

**文件**：`src/tasks/tracking/mdp/curriculum_commands.py`

**问题**：初始化 `env.reset()` 不执行 `command_manager.compute()`，`body_pos_relative_w`/`body_quat_relative_w` 保持 zeros，第一次 termination 检查用过期值误触发 `ee_body_pos`。

**修复**：`MotionCommand.reset()` 在 `super().reset()` 后调用 `_refresh_relative_state()` 重算 relative buffers。未改变正常 step 顺序。

## M4. num_envs 默认值（新增）

**文件**：`src/tasks/tracking/tracking_env_cfg.py`
**修改**：`SceneCfg(num_envs=1)` → `SceneCfg(num_envs=512)`，CLI `--env.scene.num-envs` 仍可覆盖。

---

## M1. 降低 action_std

**文件**：`src/tasks/tracking/config/g1/rl_cfg.py`

**修改位置**：`unitree_g1_tracking_ppo_runner_cfg()` → actor `distribution_cfg`

**修改内容**：
```
"init_std": 1.0  →  "init_std": 0.3
```

**修改原因**：
- 原值 `init_std=1.0`，经 `GaussianDistribution` 的 `std_param = 1.0 * ones(29)` 后，有效关节噪声 = action_scale × std ≈ 0.44~0.55 rad/关节（±25° 随机扰动）
- 29 DOF 人形从 RSI 初始帧出发时，±25° 随机目标足以使其立即失衡
- 训练日志证实 `Policy/mean_std` 从 1.0006 到 1.0023 在 851 iter 内几乎不收缩（无成功样本引导）

**修改后影响**：有效噪声 ≈ 0.13~0.17 rad/关节（约 ±8°），配合 G1 动作数据帧间跳变均值 0.027 rad，探索幅度合理。

**未改动**：`schedule="adaptive"`（只调 learning_rate）、`std_type="scalar"`、`entropy_coef=0.005`（std 无独立 decay，靠 PPO 训练自适应）。

---

## M2. Curriculum termination

### 2.1 `src/tasks/tracking/mdp/terminations.py`

**修改位置**：
- 新增辅助函数 `_curriculum_threshold(env, threshold, ramp_iters, easy_threshold)`
- `bad_anchor_pos_z_only`：新增参数 `ramp_iters: int = 0`、`easy_threshold: float | None = None`
- `bad_motion_body_pos_z_only`：新增参数 `ramp_iters: int = 0`、`easy_threshold: float | None = None`

**逻辑**：当 `ramp_iters > 0` 且 `env.common_step_counter < ramp_iters` 时，用 `easy_threshold` 替代 `threshold`；否则用原值。

**兼容性**：默认参数 `ramp_iters=0` / `easy_threshold=None` 时行为与原来完全一致（g1_23dof 未配置时不受影响）。

**修改原因**：训练日志显示 `ee_body_pos` termination 在 851 iter 中 ~100% 触发，机器人 8 步（0.16s）即倒，策略拿不到任何学习窗口。

### 2.2 `src/tasks/tracking/config/g1/env_cfgs.py`

**修改位置**：`unitree_g1_flat_tracking_env_cfg()` 内 `cfg.terminations[...].params` 配置

**修改内容**：
```
ee_body_pos: ramp_iters=24000, easy_threshold=0.6   (原 0.25 → 2.4×)
anchor_pos:  ramp_iters=24000, easy_threshold=0.5   (原 0.25 → 2.0×)
```

**ramp 时长换算**：`24000 env steps = 1000 × num_steps_per_env(24)`，即前 1000 个 PPO iteration 放宽。

**修改原因**：早期放宽终止阈值，让机器人能撑过前几步收集有效模仿经验。

**不会永久放宽**：超过 24000 步后恢复原始阈值 0.25，最终训练标准不降低。

---

## M3. RSI curriculum

### 3.1 新增 `src/tasks/tracking/mdp/curriculum_commands.py`

**内容**：`MotionCommandCfg` / `MotionCommand`，继承 mjlab 的 `MotionCommand`，新增：
- `curriculum_start_mode: str = ""`：curriculum 阶段强制使用的采样模式（如 `"start"`）
- `curriculum_iters: int = 0`：curriculum 持续的环境步数
- 重写 `_resample_command`：curriculum 阶段临时强制 `sampling_mode = curriculum_start_mode`，结束后恢复
- 重写 `build()`：返回本地的 `MotionCommand`

**继承自 mjlab 基类**，其他方法（`_update_command`、`_update_metrics`、debug_vis 等）完全复用，确保与 mjlab 行为一致。

### 3.2 `src/tasks/tracking/tracking_env_cfg.py`

**修改位置**：import

**修改内容**：
```
from mjlab.tasks.tracking.mdp import MotionCommandCfg
      ↓
from src.tasks.tracking.mdp.curriculum_commands import MotionCommandCfg
```

**修改原因**：让 tracking 任务的 motion 命令使用带 curriculum 能力的本地子类。本地 `MotionCommandCfg` 继承 mjlab 的，`isinstance(x, mjlab MotionCommandCfg)` 仍成立，g1/g1_23dof env_cfgs 中的类型断言不受影响。

### 3.3 `src/tasks/tracking/config/g1/env_cfgs.py`

**修改位置**：`unitree_g1_flat_tracking_env_cfg()` 内 `motion_cmd` 配置

**修改内容**：
```
motion_cmd.curriculum_start_mode = "start"
motion_cmd.curriculum_iters = 24000
```

**修改原因**：数据特性分析证实 test_g1_motion 帧 0 是难度最低帧（难度 2% 分位，近站立姿态）。前 1000 iteration 让机器人总是从帧 0 起步学习，24000 步后切回 `adaptive`（随机帧）覆盖全动作。

**未修改 motion 数据**：GMR/GVHMR 生成的 `test_g1_motion.npz` 保持原样。

---

## 不涉及的修改

- **reward 权重**：6 项 tracking reward + 3 项惩罚权重保持不变（避免引入新变量，先验证 M1-M3 效果）
- **mjlab / rsl_rl 核心包**：零修改
- **motion 数据**：零修改
- **g1_23dof**：继承 tracking_env_cfg 的本地命令类，但未配置 curriculum 参数 → 行为不变
- **num_envs=1**：训练时用 `--env.scene.num-envs` 覆盖（见下）

---

## 训练建议命令（M4）

```bash
# 单 GPU：至少 1024 env（审计发现原训练误用默认 num_envs=1）
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/test_g1_motion.npz \
  --env.scene.num-envs=1024
```

`num_envs=1` 时 RSI adaptive 统计、PPO 采样均失效，是训练无进展的关键因素之一。

---

## 回滚方式

```bash
# 方式一：用 git 恢复（推荐，全部改动未提交）
git checkout -- src/tasks/tracking/config/g1/rl_cfg.py \
               src/tasks/tracking/config/g1/env_cfgs.py \
               src/tasks/tracking/mdp/terminations.py \
               src/tasks/tracking/tracking_env_cfg.py
rm src/tasks/tracking/mdp/curriculum_commands.py

# 方式二：从备份恢复
cp backup_before_rl_fix/<file> src/tasks/tracking/...
```

---

## 附录：29-DOF 实证验证（2026-08-05）

用户强调机器人是 **29 关节**，对动作空间维度做了实证确认：

### 模型层面（编译 mjlab G1 spec）

| 属性 | 值 | 说明 |
|------|-----|------|
| `nq` | 36 | 位置维度 = 基座 free joint (7) + 29 关节 |
| `njnt` | 30 | 1 个 `floating_base_joint` + 29 个控制关节 |
| 控制关节 | 29 | 腿 12 + 腰 3 + 手臂 14 |

### 动作维度 = 29（确认）

- `BuiltinPositionActuatorCfg.edit_spec()` 注释明确 "**Add `<position>` actuator to spec, one per target**"（`mjlab/actuator/builtin_actuator.py:74-87`），即**每个匹配的实际关节生成一个独立 position actuator**
- `get_spec().compile()` 显示 `nu=0` 是因为 actuator 由 Entity 在 `edit_spec` 阶段注入，不在该编译路径上
- 因此 `GaussianDistribution` 输出维度 = **29**，`init_std=0.3` 对全部 29 个动作维度生效

### G1_ACTION_SCALE 覆盖全部 29 关节（确认）

16 个正则条目逐一匹配 29 个控制关节，**无遗漏、无冲突**（`floating_base_joint` 是基座，不属于动作）：

```
.*_elbow_joint → 2 个   .*_hip_pitch_joint → 2 个   waist_yaw_joint → 1 个
.*_shoulder_pitch/roll/yaw → 各 2 个   .*_hip_yaw/roll → 各 2 个
.*_wrist_roll/pitch/yaw → 各 2 个   .*_knee_joint → 2 个
waist_pitch/roll → 各 1 个   .*_ankle_pitch/roll → 各 2 个
```

### 修改后每关节有效噪声（std=0.3）

| 关节组 | action_scale | 有效噪声 = scale × 0.3 |
|--------|-------------|------------------------|
| 髋/膝 (7520_22) | 0.35 | ~0.105 rad |
| 髋 pitch/yaw、腰、肘、肩、踝 (5020) | 0.44~0.55 | 0.13~0.16 rad |
| 手腕 (4010) | 0.0745 | ~0.022 rad |

动作范围大的关节噪声相对大、手腕等小动作关节噪声小，符合物理直觉。

**结论：`init_std=0.3` 是对 29 维动作空间的统一初始标准差，全部关节生效，不存在遗漏关节沿用默认 std 的问题。**
