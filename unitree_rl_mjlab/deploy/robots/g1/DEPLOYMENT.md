# G1 动作部署指南（服务器 + 机器人）

本文说明如何把训练好的 RL 动作（motion tracking）部署到服务器并驱动 Unitree G1 机器人。

**范围**：只覆盖「动作」层面的部署 —— 准备动作文件、注册 FSM、上传、运行、手柄触发。服务器连接与网络配置（SSH、netplan、机器人 IP）见仓库根目录 `11.md`（`/home/silence/MechDance/11.md`），这里不重复。

---

## 0. 总览

一个可部署的动作 = **三件套文件 + 一条 FSM 配置**：

```
config/policy/mimic/<动作名>/
├── exported/
│   └── policy.onnx        # actor-only 策略（~991KB）
└── params/
    ├── deploy.yaml         # 观测/动作规格（C++ 构造观测向量用）
    └── <动作名>.npz        # 参考动作（50 fps）
```

C++ 程序 `g1_ctrl` 启动时读 `config/config.yaml`，按其中注册的 FSM 状态，从对应目录加载三件套（`src/State_Mimic.cpp` 第 87–121 行）。

**动作名约定**：目录名、`<动作名>.npz` 文件名、FSM 状态名三处保持一致，一般用参考动作 npz 的 hash 短名（如 `957d1c6f`、`17fe0adf`）。

---

## 1. 三件套详解

### 1.1 policy.onnx —— actor-only 策略

训练时 `MotionTrackingOnPolicyRunner.save()` 导出**两个** ONNX：

| 文件 | 内容 | 部署用哪个 |
|------|------|-----------|
| `policy.onnx` | 只含 actor 网络 | ✅ 用这个 |
| `<run_name>.onnx` | actor + 参考动作打包（~3.9MB） | ❌ 不用（C++ 侧用独立 npz） |

取自训练 run 目录 `logs/rsl_rl/g1_tracking/<run>/policy.onnx`（对应最后一次 save 的 checkpoint）。

### 1.2 deploy.yaml —— 观测/动作规格

定义 C++ 侧如何构造观测向量（观测项 + JointPositionAction 的 scale/offset）。`State_Mimic.cpp` 从 `policy_dir/params/deploy.yaml` 加载。

**No-State-Estimation 任务的规格固定**，直接复制现有 mimic 目录的 `params/deploy.yaml` 即可（如 `17fe0adf/params/deploy.yaml`），无需改动。

### 1.3 <动作名>.npz —— 参考动作

`MotionLoader_`（`include/State_Mimic.h` 第 35–134 行）从 npz 读取 4 个 key，**硬编码 50 fps**（`dt = 1/50`）：

| key | shape | 说明 |
|-----|-------|------|
| `body_pos_w` | [frame, body_id, 3] | 世界系根位置 |
| `body_quat_w` | [frame, body_id, 4] | 世界系根姿态（wxyz） |
| `joint_pos` | [frame, 29] | 关节位置 |
| `joint_vel` | [frame, 29] | 关节速度 |

取自 `src/assets/motions/g1/`（或训练 server）。

---

## 2. 准备本地部署目录

```bash
cd /home/silence/MechDance/unitree_rl_mjlab

# 假设：训练 run 目录 RUNDIR=logs/rsl_rl/g1_tracking/<run>，动作 hash NAME=<hash>

# 1. 建目录
mkdir -p deploy/robots/g1/config/policy/mimic/$NAME/exported
mkdir -p deploy/robots/g1/config/policy/mimic/$NAME/params

# 2. policy.onnx（actor-only）
cp $RUNDIR/policy.onnx deploy/robots/g1/config/policy/mimic/$NAME/exported/policy.onnx

# 3. 参考动作 npz（重命名为 <NAME>.npz）
cp src/assets/motions/g1/<hash>.npz deploy/robots/g1/config/policy/mimic/$NAME/params/$NAME.npz

# 4. deploy.yaml（No-State-Estimation 规格固定，从现有目录复制）
cp deploy/robots/g1/config/policy/mimic/17fe0adf/params/deploy.yaml \
   deploy/robots/g1/config/policy/mimic/$NAME/params/deploy.yaml
```

---

## 3. 注册 FSM 状态（config.yaml）

编辑 `config/config.yaml`（相对 `deploy/robots/g1/`），**三处**都要加。

以一个名为 `<NAME>`、用按钮 `RB + Y` 的新动作为例：

**① 启用列表 `FSM._`**（分配唯一 id）：

```yaml
    Mimic_<NAME>:
      id: 8
      type: Mimic
```

**② `Velocity` 的 transitions**（从 Velocity 进入该动作的入口按钮）：

```yaml
  Velocity:
    transitions:
      ...
      Mimic_<NAME>: RB + Y.on_pressed
```

**③ 新增 state block**（该动作的内部 transitions + 三件套路径）：

```yaml
  Mimic_<NAME>:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: RT + A.on_pressed
      Mimic_Dance1_subject2: RB + A.on_pressed
      Mimic_cut_standing: RB + B.on_pressed

    motion_file: config/policy/mimic/<NAME>/params/<NAME>.npz
    policy_dir: config/policy/mimic/<NAME>/
    time_start: 0.0
    time_end: 1000.0
```

> `motion_file` / `policy_dir` 是相对 `deploy/robots/g1/` 的路径（`g1_ctrl` 二进制在 `build/` 下，`proj_dir` 为其父目录，见 `deploy/include/param.h`）。

按钮 DSL 语法见 `deploy/include/unitree_joystick_dsl.hpp`：`RB + Y.on_pressed` = RB 按住 + Y 刚按下（边沿触发），`x/y/a/b` 均有效（大小写不敏感）。

---

## 4. 上传到服务器

```bash
cd /home/silence/MechDance/unitree_rl_mjlab/deploy/robots/g1

# 服务器地址/路径（IP 每次连接前用 ip a 确认，见 11.md）
SERVER=ubuntu@10.101.194.227
DEPLOY=/home/ubuntu/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1

# 1. config.yaml（改过就必须重新上传）
scp config/config.yaml $SERVER:$DEPLOY/config/config.yaml

# 2. 动作目录（三件套）
scp -r config/policy/mimic/<NAME> $SERVER:$DEPLOY/config/policy/mimic/
```

---

## 5. 运行 + 触发

**服务器侧**（在服务器终端）：

```bash
cd ~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=enp6s0
```

- `enp6s0` 是连机器人的有线网口（用 `ifconfig` 确认，见 `11.md`）。
- 启动后打印 `Waiting for connection rt/lowstate` 是**正常**的，等机器人连接。

**机器人侧**：

1. 机器人上电。
2. 手柄 **L2+R2** 进入调试模式（debug mode）—— 这是机器人硬件层的开关，`lowcmd` 只有在这个模式下才允许下发关节指令。**必须**。
3. 连接建立后，`g1_ctrl` 日志开始初始化各 FSM 状态，确认出现 `Initializing State_Mimic_<NAME>` 和 `Loaded motion file '<NAME>' ... duration XX.XXs`。

**手柄触发**（`g1_ctrl` 的软件状态机，与调试模式是两回事）：

| 阶段 | 按钮 | 结果 |
|------|------|------|
| Passive → FixStand | `L2 + 上` | 站立 |
| FixStand → Velocity | `R2 + A` | 进入速度控制 |
| Velocity → 目标动作 | `RB + <按钮>` | 播放动作 |
| 任意 Mimic → Passive | `LT + B` | 停止 |
| 任意 Mimic → Velocity | `R2 + A` | 回速度控制 |

---

## 6. 当前按钮映射

（截至 2026-08-15，`config/config.yaml`）

| 动作 | FSM 状态 | 入口按钮 | id |
|------|---------|---------|----|
| 舞蹈 1 | `Mimic_Dance1_subject2` | RB + A | 5 |
| 站定收尾 | `Mimic_cut_standing` | RB + B | 6 |
| 新舞 957d1c6f | `Mimic_957d1c6f` | RB + X | 7 |
| 17fe0adf | `Mimic_17fe0adf` | RB + Y | 8 |

---

## 7. 故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| 一直 `Waiting for connection rt/lowstate` | DDS 没通 / 网口错 / 机器人未上电 | 检查 `--network` 是否为 `enp6s0`；`ping 192.168.123.99`；确认机器人已上电进调试模式 |
| 手柄按了没反应 | `config.yaml` 没更新到服务器 / `g1_ctrl` 没重启 | config 是**启动时**加载，改后必须重新 scp + 重启 `g1_ctrl` |
| 没出现 `Initializing State_Mimic_<NAME>` | config.yaml 三处没加全 / 状态名拼写不一致 | `grep -n "<NAME>" config/config.yaml` 应出现 3 次 |
| 报 `Loaded motion file` 但帧数/时长不对 | npz 不是 50fps / 缺 4 个 key 之一 | 用 Python 检查 npz key 与 shape，见 1.3 |
| 动作播放但动作不对 | policy.onnx 用错（用了 `<run>.onnx` 打包版）或 npz 与训练不对应 | 确认用 actor-only `policy.onnx`；比对 npz md5 与训练源一致 |
