# RL Parameter Audit — Unitree G1 Motion Imitation

审计日期：2026-08-05
审计对象：`Unitree-G1-Tracking` / `Unitree-G1-Tracking-No-State-Estimation`
审计依据代码：commit `1425b15` + mjlab 1.2.0 + rsl_rl 5.x（已安装包）
训练实况数据源：`logs/rsl_rl/g1_tracking/2026-08-05_07-20-19/`（tfevents，851 iterations）

---

## # Current Training Issue

### 实测症状（来自训练日志）

| 指标 | 实测值 | 说明 |
|------|--------|------|
| `Train/mean_episode_length` | **≈ 8 步**（iter 0→851 一直如此） | 0.16s / episode（step_dt=0.02s） |
| `Policy/mean_std` | 1.0006 → 1.0023（851 iter 几乎不变） | action std **不收缩** |
| `Episode_Termination/ee_body_pos` | **851 iter 中 ~100% 触发** | 绝对主导的终止原因 |
| `Episode_Termination/time_out` | 0%（从未存活到 10s） | 无任何完整 episode |
| `Train/mean_reward` | ≈ -1.0（长期为负） | 惩罚项主导 |
| 所有 tracking reward 分量 | ≈ 0.005~0.02 | 机器人处于跌倒态，误差巨大 |
| `env.yaml` 实测 `num_envs` | **1** | 单环境训练 |

### 根本问题链

```
num_envs=1（采样效率极低）
   +
init_std=1.0 的 action 噪声（有效 0.44~0.55 rad/关节）
   +
RSI 从 587 帧复杂舞蹈的【任意帧】起步（无 curriculum）
   +
ee_body_pos 0.25m z 阈值 100% 触发（8 步即倒）
   +
entropy bonus(0.005) 推动 std 增大，无成功样本引导收缩
═════════════════════════════════════════════
→ 恶性循环：策略拿不到任何"存活+跟踪"样本 → 无法学习 → std 不降
```

结论：**不是单个参数问题，而是 4 个因素叠加**。单修 action_std 或单改 termination 都不足以让训练跑起来。

---

## # Code Structure

### 相关文件

| 角色 | 文件 |
|------|------|
| 基础 env 配置（观测/动作/命令/奖励/终止） | `src/tasks/tracking/tracking_env_cfg.py` |
| G1 专用 env 配置 | `src/tasks/tracking/config/g1/env_cfgs.py` |
| G1 RL 配置（PPO/网络/std） | `src/tasks/tracking/config/g1/rl_cfg.py` |
| 任务注册 | `src/tasks/tracking/config/g1/__init__.py` |
| Motion 命令/RSI（本项目复制件） | `src/tasks/tracking/mdp/commands.py` |
| 终止函数 | `src/tasks/tracking/mdp/terminations.py` |
| 奖励函数 | `src/tasks/tracking/mdp/rewards.py` |
| 观测函数 | `src/tasks/tracking/mdp/observations.py` |
| Runner（ONNX 导出） | `src/tasks/tracking/rl/runner.py` |
| 框架 motion 命令（真实实现，commands.py 是复制） | `mjlab/tasks/tracking/mdp/commands.py` |
| env 主循环/reset | `mjlab/envs/manager_based_rl_env.py` |
| command 管理器/reset 链 | `mjlab/managers/command_manager.py` |
| 分布实现（std） | `rsl_rl/modules/distribution.py` |
| PPO 算法 | `rsl_rl/algorithms/ppo.py` |

### 关键机制链

1. **reset 链**：`_reset_idx()` → `command_manager.reset(env_ids)` → `MotionCommand.reset()` → `_resample()` → `_resample_command()`（RSI 写入）
2. **step 顺序**：`simulate → termination_manager.compute() → reward_manager.compute() → _reset_idx(reset) → sim.forward() → command_manager.compute()`（注意 termination 先于 command 更新）
3. **观测**：actor 无 base_lin_vel/motion_anchor_pos_b（No-State-Estimation 变体移除），critic 有特权信息

---

## # Action Noise Analysis

### 参数调用链

```
配置文件：src/tasks/tracking/config/g1/rl_cfg.py:17-21
  distribution_cfg = {"class_name": "GaussianDistribution",
                      "init_std": 1.0, "std_type": "scalar"}

传递：unitree_g1_tracking_ppo_runner_cfg() → register_mjlab_task(rl_cfg=...)
   → load_rl_cfg(task_id)（scripts/train.py）→ 存为 agent_cfg["actor"]["distribution_cfg"]

使用：MjlabOnPolicyRunner(rsl_rl) → OnPolicyRunner.__init__ → MLPModel
   → dist_class(output_dim, **distribution_cfg)  [rsl_rl/models/mlp_model.py:67-69]
   → GaussianDistribution(29, init_std=1.0, std_type="scalar")
   → self.std_param = nn.Parameter(1.0 * ones(29))  [rsl_rl/modules/distribution.py:157]

运行时真实值：Policy/mean_std = 1.0006 (iter 0) → 1.0023 (iter 851)
```

### 有效动作幅度

- 动作是关节位置目标：`processed = raw_actions * action_scale + default_offset`（`mjlab/envs/mdp/actions/actions.py:148`）
- `G1_ACTION_SCALE`（16 种正则 actuator 类型覆盖全部 29 DOF）：min 0.075 / 中位 **0.439** / max 0.548
- **有效噪声 std = action_scale × 1.0 ≈ 0.44~0.55 rad/关节**（相当于 ±25° 随机扰动）
- 注意：动作分布 std=1.0 是**未缩放**的，采样后乘 scale。所以 std 降不降，实际扰动幅度都很大

### std 为何不衰减

- `std_type="scalar"`：std 是 learnable parameter，靠 PPO 梯度收缩
- PPO loss = surrogate + value − **entropy_coef × entropy**（`ppo.py:313`），entropy_coef=0.005 → **熵奖励推动 std 增大**
- `schedule="adaptive"` 只调 learning_rate，**与 std 无关**（`ppo.py:269-294`）
- 所有 episode 均失败、reward 全负 → 无"成功"优势信号引导 std 收缩 → 实测 std 缓慢上漂
- **结论：没有独立的 noise decay scheduler，std 收缩完全依赖策略学到正优势样本**

---

## # Termination Analysis

### 配置（tracking_env_cfg.py:259-281 + G1 env_cfgs.py:65-70）

| 终止项 | 函数 | 阈值 | body 范围 | 实测触发率 |
|--------|------|------|-----------|-----------|
| `ee_body_pos` | `bad_motion_body_pos_z_only` | **0.25 m**（仅 z） | left/right_ankle_roll_link, left/right_wrist_yaw_link | **~100%** |
| `anchor_pos` | `bad_anchor_pos_z_only` | 0.25 m（仅 z，torso_link） | — | 偶发 |
| `anchor_ori` | `bad_anchor_ori` | 0.8（重力投影差） | — | 0% |
| `time_out` | `time_out` | 10.0 s | — | 0% |

### 分析

- `bad_motion_body_pos_z_only`：`|body_pos_relative_w.z − robot_body_pos_w.z| > 0.25` 即终止（`terminations.py:73-86`）。4 个末端（脚踝+手腕）任一 z 偏差超 25cm 即倒计时归零
- **0.25m 对舞蹈动作过严**：舞蹈中手腕/脚踝瞬时位移大，跟随延迟很容易超 25cm
- **~100% 触发率意味着终止条件根本没有给策略任何"学习窗口"**
- 次要 bug（非主因）：reset 后第一个 step 的 termination 检查使用**过期的 `body_pos_relative_w`**（`command_manager.compute()` 在 termination 之后运行），放大了 reset 帧的误杀概率

### 时间换算

- step_dt = decimation(4) × physics_dt(0.005) = **0.02s**
- 8 步 = 0.16s 存活 → 几乎刚落地就触发

---

## # RSI Analysis

### 结论：RSI **存在且已开启**（用户怀疑"可能没开"不成立）

- 实现：`MotionCommand._resample_command()`（`commands.py:297-363`）
- reset 链已接线：`_reset_idx()` → `command_manager.reset()` → `_resample()` → `_resample_command()`（`command_manager.py:87-95, 203-209`）
- 顺序正确：`command_manager.reset()` 在 `sim.reset()` **之后**，RSI 写入不会被默认姿态覆盖（`manager_based_rl_env.py:489-527`）

### RSI 行为（`_resample_command`）

1. 采样起始帧：
   - `sampling_mode="adaptive"`（**默认，训练实测值**）：从 587 帧按 bin 失败计数加权采样（初期≈均匀）
   - `uniform`：均匀随机
   - `start`：固定帧 0
2. 写入参考状态：root_pos/ori/lin_vel/ang_vel + joint_pos/vel（来自 motion 数据）
3. 施加初始化噪声：
   - `pose_range`：x/y ±0.05, z ±0.01, roll/pitch ±0.1, yaw ±0.2（tracking_env_cfg.py:146-153）
   - `velocity_range`：x/y ±0.5, z ±0.2, roll/pitch ±0.52, yaw ±0.78
   - `joint_position_range`：±0.1

### 问题

- **无 curriculum**：每次 reset 从 587 帧复杂舞蹈的**任意帧**起步（adaptive 初期≈均匀），很多帧是大幅运动/单脚/失衡姿态，从零学策略必然立即失败
- 这解释了为何 `ee_body_pos` 100% 触发：RSI 落在剧烈动作帧 → 跟随不及 → 末端超差
- 次要 bug：reset 后 `body_pos_relative_w`/`body_quat_relative_w` 未在 `_resample_command` 内重算，直到下一次 `_update_command` 才刷新（见 Termination 分析）

---

## # Reward Analysis

### 配置（tracking_env_cfg.py:211-253）

**正向 tracking（exp(−error/std²) 形式）：**

| Reward | 权重 | std | 函数 |
|--------|------|-----|------|
| motion_global_root_pos | 0.5 | 0.3 | anchor 位置误差 |
| motion_global_root_ori | 0.5 | 0.4 | anchor 姿态误差 |
| motion_body_pos | 1.0 | 0.3 | 各 body 相对位置误差均值 |
| motion_body_ori | 1.0 | 0.4 | 各 body 姿态误差均值 |
| motion_body_lin_vel | 1.0 | 1.0 | 线速度误差 |
| motion_body_ang_vel | 1.0 | 3.14 | 角速度误差 |

**惩罚：**

| Reward | 权重 |
|--------|------|
| action_rate_l2 | −0.1 |
| joint_limit | −10.0 |
| self_collisions | −10.0 |

### 实测

- 6 项 tracking reward 全部 ≈ 0（跌倒态误差巨大）
- mean_reward ≈ −1.0（惩罚主导），没有信号让策略区分"好/坏"轨迹
- `action_rate_l2` 惩罚在探索期会抑制动作平滑性，但非当前主要矛盾

### 结构缺陷（非当前主因，记录待后续优化）

- 观测中只有 `last_action`，**无 action history / joint history**（模仿任务惯例需要观测历史窗口）
- 无 foot contact / gait phase 奖励项（velocity 任务有 gait 相位辅助，tracking 无）

---

## # Recommended Modification Plan

> 全部待用户确认后执行。修改顺序遵循：config → env code → 核心算法代码（尽量不改）。

### M1. 降低 action std（config，rl_cfg.py）

- `init_std: 1.0 → 0.3`
- 效果：有效噪声 std ≈ 0.44~0.55 → 0.13~0.17 rad/关节
- **不破坏 schedule**：`schedule="adaptive"` 只影响 learning_rate；`std` 无独立 decay，靠 PPO 训练自适应，本次不动机制
- 注意：`entropy_coef=0.005` 仍会推动 std 增大，故需 M2/M4 配合制造"成功样本"

### M2. Curriculum Termination（env code，env_cfgs.py + termination 包装）

- 训练早期（前 1000 iter）放宽终止阈值：
  - `ee_body_pos`: 0.25 → **0.6**（2.4×）
  - `anchor_pos`: 0.25 → **0.5**（2.0×）
- 之后恢复正常（0.25）
- 实现方式（可选其一，待确认）：
  - **A（推荐，纯 config）**：借助 mjlab CurriculumManager，按 `common_step_counter` 或全局 iter 线性/分段插值阈值。需先验证 mjlab curriculum 是否支持 termination 字段
  - **B（包装函数）**：在 `terminations.py` 新增带 schedule 参数的包装（不改原函数），env_cfgs 传入"当前阈值"
- **不永久放宽**：超过 1000 iter 恢复原阈值，不降低最终标准

### M3. RSI Curriculum（env code，env_cfgs.py 或 commands.py）

- 目标：让机器人从**简单→难**起步
- 方案：`sampling_mode` 分阶段：训练早期 `"start"`（帧 0 站立起步）→ 中期切 `"uniform"` → 后期 `"adaptive"`
- 纯 config 方案：通过 `MotionCommandCfg.sampling_mode` 传入，配合 M2 的 schedule 一起控制
- 若 mjlab CurriculumManager 无法驱动 command 字段，则在 `commands.py` 内加 schedule 逻辑（最小侵入）

### M4. 训练规模修正（CLI 参数，非代码）

- 实测 `num_envs=1` 是单环境训练。**推荐 `--env.scene.num-envs≥1024`**（参考 CLAUDE.md 示例 4096）
- 单 env 下 RSI adaptive 统计、PPO 采样均失效，这是最廉价也最必要的一改

### M5. 训练稳定性（可选，观察 M1-M4 效果后再定）

- 若 motion 难度仍过高：**不改 motion 数据**，先靠 M3 的 start→uniform 过渡，再不行降 reward std 或加观测历史
- 保留 `action_rate_l2`/`joint_limit`/`self_collisions` 权重不变（避免引入新变量）

### 修改记录要求

- 改前先建 `backup_before_rl_fix/` 备份
- 每次改动写 `RL_FIX_CHANGELOG.md`（文件/位置/原因/影响）
- 修改后先跑：环境启动测试 → 单 episode 测试 → 检查 mean episode length / termination 分布 / reward 分量 → 输出 `RL_FIX_VALIDATION_REPORT.md`
