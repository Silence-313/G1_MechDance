# Unitree G1 Motion Tracking RL — 现状文档

日期：2026-08-05
范围：仅描述 RL 训练部分（不含 GMR / GVHMR 动作生成项目）
对象：`Unitree-G1-Tracking` / `Unitree-G1-Tracking-No-State-Estimation`

---

## 1. 项目定位

在 unitree_rl_mjlab 框架上，用 PPO 训练 Unitree G1（**29 DOF** 人形）模仿参考动作（当前为舞蹈 `test_g1_motion`）。目标是让机器人通过 RL 学会跟踪参考运动，为后续更复杂动作（如 Breaking）训练打基础。

技术栈：mjlab 1.2.0（环境）+ rsl_rl 5.x（PPO）+ MuJoCo 3.5 物理。

两个任务变体：
- `Unitree-G1-Tracking`：actor 有基座位置观测（有状态估计）
- `Unitree-G1-Tracking-No-State-Estimation`：actor 无 `base_lin_vel` / `motion_anchor_pos_b`，更接近真机条件

---

## 2. 数据来源（视频 → G1 动作）

RL 训练所需的 motion 文件由外部项目生成，链路如下（**已跑通一次**，产物完整）：

```
test.mp4
  → [GVHMR]   outputs/demo/test/hmr4d_results.pt     人体 SMPL-X 动作
  → [GMR]     outputs/test_g1_motion.pkl             G1 动作 (29 dof)
  → [GMR]     outputs/csv/test_g1_motion.csv         BeyondMimic 格式 CSV
  → [本项目]  src/assets/motions/g1/test_g1_motion.npz  RL 训练输入
```

当前训练动作文件：`src/assets/motions/g1/test_g1_motion.npz`
- **587 帧，50 fps**（约 11.7 s），与 RL 环境 step_dt=0.02s 对齐
- 字段：`joint_pos(587,29)`、`joint_vel(587,29)`、`body_pos_w(587,30,3)`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w` —— 与 `MotionLoader` 所需完全匹配
- 关节顺序已验证：GMR 模型 / `csv_to_npz.py` joint_names / mjlab G1 **三者一致**
- 数据特性：帧间跳变均值 0.027 rad（平滑无抖动）；帧 0 是难度最低帧（2% 分位）

> 注意：若需生成**新**动作，GVHMR 预训练权重被省略（需补），GMR 需在 `gmr` conda 环境运行（依赖 `mink`）。

---

## 3. RL 训练架构

### 3.1 配置链路

```
src/tasks/tracking/tracking_env_cfg.py    ← 基础配置工厂 make_tracking_env_cfg()
  → src/tasks/tracking/config/g1/env_cfgs.py   ← G1 专用（关节、传感器、body 列表）
  → src/tasks/tracking/config/g1/rl_cfg.py     ← PPO / 网络 / init_std
  → src/tasks/tracking/config/g1/__init__.py   ← 任务注册 register_mjlab_task()
```

命令侧使用**本地** `curriculum_commands.py`（继承 mjlab `MotionCommand`），由 `tracking_env_cfg.py` import 接入。

### 3.2 观测（Actor/Critic 不对称）

| 观测 | Actor | Critic |
|------|-------|--------|
| motion 命令（joint_pos+joint_vel） | ✅ | ✅ |
| anchor pos/ori 偏差 | 仅 Tracking 变体 | ✅ |
| base_lin_vel / base_ang_vel | ✅（带噪声） | ✅（无噪声） |
| joint_pos / joint_vel | ✅ | ✅ |
| last_action | ✅ | ✅ |
| body_pos / body_ori（特权） | ❌ | ✅ |

Actor 有观测噪声（corruption），Critic 使用无噪声特权信息——标准 sim-to-real 模式。

### 3.3 动作

`JointPositionAction`，`actuator_names=(".*",)`，scale = `G1_ACTION_SCALE`（16 个正则覆盖全部 29 关节，scale 0.0745~0.5475）。动作分布 `GaussianDistribution`（29 维）。

### 3.4 命令 / RSI（Reference State Initialization）

`MotionCommand`：reset 时从参考运动采样起始帧，写入 robot 的 joint/root 状态（**RSI 已确认存在且开启**）。采样模式：
- `start`：固定帧 0
- `uniform`：随机帧
- `adaptive`：按失败 bin 加权（默认）

### 3.5 奖励

6 项正向 tracking（exp 误差）+ 3 项惩罚：

| 项 | 权重 |
|----|------|
| motion_global_root_pos / ori | 0.5 / 0.5 |
| motion_body_pos / ori | 1.0 / 1.0 |
| motion_body_lin_vel / ang_vel | 1.0 / 1.0 |
| action_rate_l2 | −0.1 |
| joint_limit | −10.0 |
| self_collisions | −10.0 |

### 3.6 终止

| 项 | 阈值 |
|----|------|
| ee_body_pos（脚踝+手腕 z 偏差） | 0.25 m |
| anchor_pos（torso z 偏差） | 0.25 m |
| anchor_ori | 0.8 |
| time_out | 10 s |

### 3.7 PPO

- 网络：MLP (512, 256, 128)，ELU，obs normalization
- `init_std=0.3`（原 1.0，已改）、`std_type="scalar"`、`entropy_coef=0.005`
- `learning_rate=1e-3`、`schedule="adaptive"`（只调 lr，不调 std）
- `num_steps_per_env=24`、`save_interval=500`、`max_iterations=30001`

---

## 4. 训练现状与核心问题（实测）

数据源：`logs/rsl_rl/g1_tracking/2026-08-05_07-20-19/`（851 iterations，tfevents）

| 指标 | 实测 | 说明 |
|------|------|------|
| mean_episode_length | **≈ 8 步**（851 iter 无改善） | 0.16s / episode |
| Policy/mean_std | 1.0006 → 1.0023 | std 不收缩 |
| ee_body_pos termination | **~100%** | 绝对主导终止 |
| time_out | 0% | 从未存活到 10s |
| mean_reward | ≈ −1.0 | 惩罚主导 |
| 训练 num_envs | **1** | 单环境 |

**根因链（已确认）**：单环境训练 + 大动作噪声（std=1.0）+ RSI 从复杂舞蹈任意帧起步 + 第一帧误触发终止（stale state）→ 策略拿不到任何"存活+跟踪"样本 → 恶性循环不学习。

**附带确认**：机器人确为 29 DOF；`init_std` 作用于全部 29 维动作；reset 后 relative buffers 未刷新导致第一帧误终止。

---

## 5. 已执行的代码修改

所有修改**未提交**，可通过 `git checkout` 回滚。详见 `RL_CODE_CHANGE_REPORT.md` 与 `RL_FIX_CHANGELOG.md`。

| 修改 | 文件 | 内容 |
|------|------|------|
| M1 action_std | `config/g1/rl_cfg.py` | `init_std` 1.0 → **0.3** |
| M3 RSI curriculum | `mdp/curriculum_commands.py`（新增）+ `tracking_env_cfg.py` + `env_cfgs.py` | 三阶段采样：start(<1000 iter)→uniform(<5000 iter)→adaptive(≥5000 iter) |
| M4 num_envs | `tracking_env_cfg.py` | 默认 1 → **512**（CLI 可覆盖） |
| M5 stale state | `mdp/curriculum_commands.py` | reset 后刷新 relative buffers，修复首帧误终止 |

**已撤销**：M2（curriculum termination）——本轮指令明确不修改 termination threshold，`terminations.py` 已恢复到 HEAD。

**修改前有效噪声对比**：std=1.0 时 0.44~0.55 rad/关节 → std=0.3 时 0.022~0.16 rad/关节。

---

## 6. 当前代码状态

```
git status（RL 相关）：
  M  src/tasks/tracking/config/g1/env_cfgs.py          （RSI curriculum 参数）
  M  src/tasks/tracking/config/g1/rl_cfg.py            （init_std=0.3）
  M  src/tasks/tracking/tracking_env_cfg.py            （num_envs=512 + import 切换）
  ?? src/tasks/tracking/mdp/curriculum_commands.py     （新增，三阶段 + stale fix）
  ?? RL_PARAMETER_AUDIT.md / RL_FIX_CHANGELOG.md / RL_FIX_VALIDATION_REPORT.md / RL_CODE_CHANGE_REPORT.md
  ?? backup_before_rl_fix/                             （原始文件备份）
  （terminations.py 与 HEAD 一致）
```

静态验证已通过：语法检查、三阶段逻辑 mock 验证、reset 刷新验证。**尚未在服务器实际运行训练**。

---

## 7. 待验证 / 下一步

1. **服务器验证**（需 GPU/EGL）：
   ```bash
   # 环境启动 + 单 episode 观察
   python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --agent=zero \
     --motion_file=src/assets/motions/g1/test_g1_motion.npz --num_envs=8

   # 短训练（前 500 iter）
   python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
     --motion_file=src/assets/motions/g1/test_g1_motion.npz --env.scene.num-envs=1024
   ```
2. **观察指标**：mean_episode_length（应显著 >8）、ee_body_pos 触发率（应下降）、mean_std（应从 0.3 调整）、tracking reward（应从 0 上升）。
3. **训练稳定后**再评估是否加入 termination curriculum（当前未加）。
4. 若 motion 难度仍高：评估观测历史（action/joint history）、foot contact 奖励等（当前未做）。

---

## 附：相关文档索引

| 文档 | 内容 |
|------|------|
| `RL_PARAMETER_AUDIT.md` | 全量参数审计（action_std 调用链 / termination / RSI / reward） |
| `RL_CODE_CHANGE_REPORT.md` | 本轮代码修改详情（本轮为准） |
| `RL_FIX_CHANGELOG.md` | 修改历史与回滚方式 |
| `RL_FIX_VALIDATION_REPORT.md` | 验证清单（静态已过，运行待做） |
| `backup_before_rl_fix/` | 被修改文件的原始备份 |
