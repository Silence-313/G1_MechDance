# MechDance 舞蹈训练与部署实操手册

> 本手册只讲**怎么操作**，少讲原理。覆盖 5 件事：
>
> 1. 用视频提取动作 → 转成机器人关节数据 → 强化学习（RL）训练
> 2. 通过网线直连服务器，把训练好的舞蹈在真机上测试
> 3. 把舞蹈部署到机器人**内部的 PC2**，脱离服务器独立运行
> 4. 用手柄控制机器人
> 5. **修改机器人腿部关节数据**（膝盖屈曲/半蹲等）
>
> 文中的所有命令、IP、路径、按钮映射均来自仓库内已实测通过的脚本和文档。
>
> **阅读约定**：标注 **【必须】** 的步骤**不可跳过**——跳过会导致后续步骤跑不通或机器人不安全；未标注的可按需选用。

---

## 0. 整体数据流（一图看懂）

```
视频 .mp4
  → [GVHMR]   人体 3D 动作      outputs/demo/<视频名>/hmr4d_results.pt
  → [GMR]     机器人关节动作     outputs/<名称>.pkl  →  csv
  → [csv→npz] 训练用动作         src/assets/motions/g1/<名称>.npz  (50 fps)
  → [PPO 训练] 训练出策略         logs/rsl_rl/g1_tracking/<run>/policy.onnx
  → [部署]    真机播放            config/policy/mimic/<名称>/ (onnx + deploy.yaml + npz)
```

三个子项目互相独立，靠**文件**衔接（不是代码互相 import），各自有独立 conda 环境：

| 项目 | conda 环境 | Python | 作用 |
|------|-----------|--------|------|
| GVHMR | `gvhmr` | 3.10 | 视频 → 人体 SMPL-X |
| GMR | `gmr` | 3.10 | 人体 → 机器人关节 |
| unitree_rl_mjlab | `unitree_rl_mjlab` | 3.11 | CSV→NPZ + RL 训练 + 部署 |

**本机**（当前 `/home/silence/MechDance`）只负责写代码/准备数据；**训练（需 GPU）和真机部署在离线服务器/机器人服务器上做**。

---

## 1. 环境准备（一次性）

三套 conda 环境按各项目的安装说明建好即可：

```bash
# GVHMR
conda create -n gvhmr python=3.10 -y && conda activate gvhmr
pip install -r requirements.txt && pip install -e .
# 预训练权重放到 inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt（需自行下载）

# GMR
conda create -n gmr python=3.10 -y && conda activate gmr
pip install -e . && conda install -c conda-forge libstdcxx-ng -y

# RL
conda create -n unitree_rl_mjlab python=3.11 -y && conda activate unitree_rl_mjlab
pip install -e .
```

> GVHMR 还需要 SMPL-X body model 文件（在 `assets/body_models/smplx/`）。缺少时 GMR 的 `gvhmr_to_robot.py` 会加载失败。

---

## 2. 视频 → RL 训练（四步，可全自动）

### 2.0 最快方式：一条脚本跑完（推荐）

> **【必须注意】** 本节的这一条脚本会把 **2.1 ~ 2.4 的全部分步操作自动跑完**（视频 → GVHMR → GMR → CSV → NPZ → RL 训练）。
> **执行了 2.0 之后，就不需要再手动执行后面的 2.1 / 2.2 / 2.3 / 2.4**——否则等于重复跑一遍。
> 只有需要**分步调试、复用中间产物、或单独改某步参数**时，才看 2.1~2.4。

```bash
cd /home/silence/MechDance

# 注意：脚本默认 GVHMR_DIR/GMR_DIR 指向 /home/silence/GVHMR、/home/silence/GMR（不存在），必须覆盖
GVHMR_DIR=/home/silence/MechDance/GVHMR \
GMR_DIR=/home/silence/MechDance/GMR \
  ./unitree_rl_mjlab/run_video_to_rl.sh --video /path/to/dance.mp4 --name my_dance --num-envs 1024
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--name <名>` | 动作名，默认取视频文件名 |
| `--robot g1\|g1_23dof` | 机器人，默认 g1（29 关节） |
| `--input-fps 30` / `--output-fps 50` | GMR 输出 / RL 动作帧率，默认 30/50 |
| `--num-envs 1024` | 训练并行环境数 |
| `--no-gvhmr` / `--no-gmr` | 跳过某步（复用已有中间产物） |
| `--no-train` | 只生成 npz，不训练 |
| `--keep-tmp` | 保留中间临时目录 |

脚本会依次激活 `gvhmr`、`gmr`、`unitree_rl_mjlab` 三个环境各跑一个子进程，所以**三个环境都必须存在**（**【必须】前置条件**）。

### 2.1 第 1 步：GVHMR 视频 → 人体动作

```bash
cd /home/silence/MechDance/GVHMR
conda activate gvhmr
python tools/demo/demo.py --video=/path/to/dance.mp4 -s
```

- `-s` = 静态相机（跳过视觉里程计）。拍摄时相机固定就加 `-s`。
- 产物：`outputs/demo/<视频名>/hmr4d_results.pt`（SMPL-X 人体动作，世界坐标）。

### 2.2 第 2 步：GMR 人体动作 → 机器人关节（pkl → csv）

```bash
cd /home/silence/MechDance/GMR
conda activate gmr

# 2a. GVHMR 结果 → G1 关节动作 pkl
python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file /home/silence/MechDance/GVHMR/outputs/demo/<视频名>/hmr4d_results.pt \
  --robot unitree_g1 \
  --save_path /home/silence/MechDance/GMR/outputs/<名称>.pkl

# 2b. pkl → csv（把该 pkl 单独放一个临时目录，脚本会批处理该目录下所有 pkl）
mkdir -p /home/silence/MechDance/GMR/outputs/tmp_<名称>
cp /home/silence/MechDance/GMR/outputs/<名称>.pkl /home/silence/MechDance/GMR/outputs/tmp_<名称>/
python scripts/batch_gmr_pkl_to_csv.py --folder /home/silence/MechDance/GMR/outputs/tmp_<名称>
```

- pkl 内部键：`fps`、`root_pos`(N×3)、`root_rot`(N×4, xyzw)、`dof_pos`(N×29)。
- csv 每帧 36 列 = `root_pos(3) + root_rot(4) + dof_pos(29)`，30 fps（>30 会下采样到 30）。
- 产物：`outputs/tmp_<名称>/csv/<名称>.csv`。

### 2.3 第 3 步：CSV → NPZ（RL 训练输入）

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python scripts/csv_to_npz.py \
  --input-file /home/silence/MechDance/GMR/outputs/tmp_<名称>/csv/<名称>.csv \
  --output-name <名称> \
  --input-fps 30 --output-fps 50 --robot g1
```

- 产物：`src/assets/motions/g1/<名称>.npz`（50 fps）。
- npz 关键字段（部署和训练都读这 4 个）：`joint_pos`(N×29)、`joint_vel`(N×29)、`body_pos_w`(N×body×3)、`body_quat_w`(N×body×4)。另有 `body_lin_vel_w`、`body_ang_vel_w`、`fps`。

### 2.4 第 4 步：RL 训练（PPO 模仿跟踪）

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/<名称>.npz \
  --env.scene.num-envs=1024
```

- 任务名 `Unitree-G1-Tracking-No-State-Estimation`：actor 无基座速度估计，贴近真机条件。
- 训练日志/模型在 `logs/rsl_rl/g1_tracking/<时间戳>/`。
- 训练会导出**两个** ONNX（见第 4 章）：`policy.onnx`（只用这个）和 `<run名>.onnx`（打包了参考动作，**部署不用**）。

### 2.5 训练前先验证动作（强烈建议）

```bash
# 用 zero-agent 只看参考动作（不回放策略），确认动作合理、脚不陷地
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --agent=zero --motion_file=src/assets/motions/g1/<名称>.npz --num_envs=8
```

---

## 3. 修改机器人腿部关节数据（重点）

### 3.1 先记住：G1 的 29 关节顺序（0-based 列号）

**腿部是前 12 列（0~11）**，这是所有腿部修改脚本使用的列号：

| 列 | 关节 | 列 | 关节 |
|----|------|----|------|
| 0 | left_hip_pitch（左髋俯仰） | 6 | right_hip_pitch |
| 1 | left_hip_roll（左髋横滚） | 7 | right_hip_roll |
| 2 | left_hip_yaw（左髋偏航） | 8 | right_hip_yaw |
| **3** | **left_knee（左膝）** | **9** | **right_knee（右膝）** |
| 4 | left_ankle_pitch（左踝俯仰） | 10 | right_ankle_pitch |
| 5 | left_ankle_roll（左踝横滚） | 11 | right_ankle_roll |

其余：12~14 腰部（yaw/roll/pitch），15~21 左臂，22~28 右臂。

**膝盖**：列 **3（左）、9（右）**。限位 `[-0.087, 2.88]` rad，**0 = 伸直，正值 = 屈膝**（正值越大越弯）。
矢状面三个「蹲」关节限位（来自 `g1.xml`）：

| 关节 | 限位 (rad) |
|------|-----------|
| hip_pitch | -2.5307 ~ 2.8798 |
| knee | -0.0873 ~ 2.8798 |
| ankle_pitch | -0.8727 ~ 0.5236 |

### 3.2 关键红线：【必须】改关节后必须重算 body FK

跟踪 npz 里 `joint_pos`（关节角）和 `body_pos_w / body_quat_w`（各连杆世界位姿）是**一套自洽数据**，训练和部署时奖励同时用这两组。**只改 `joint_pos` 不重算 `body_*`**，会让参考动作自相矛盾 → 策略收敛成高频抖动 → 真机电机过流、**BMS 弹电池**。

所以：**【必须】** 永远不要手改 `joint_pos` 后直接拿去训练/部署，要用下面的脚本，它们会顺带重算 FK。

### 3.3 三个腿部修改脚本（都在 `unitree_rl_mjlab/deploy/robots/g1/`）

| 脚本 | 作用 | 是否重算 FK | 适用 |
|------|------|:---:|------|
| `fix_knee_npz.py` | 只 clamp 膝盖到下限 | ❌ | **仅快速查看，别用于真机** |
| `regenerate_consistent_npz.py` | clamp 膝盖 + 重算全部 body FK | ✅ | 只改膝盖时用 |
| `squat_bias_npz.py` | 整体偏移成半蹲 + 落脚锚定 + 重算 FK | ✅ | **推荐**（既改腿又不改脚步） |

> 环境要求：后两个脚本需要 `mujoco` 和 `torch`，在 `unitree_rl_mjlab` conda 环境下跑即可。

#### 3.3.1 只把膝盖 clamp 到下限（并保持一致性）——推荐用 regenerate

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python deploy/robots/g1/regenerate_consistent_npz.py \
  src/assets/motions/g1/<原名>.npz \
  src/assets/motions/g1/<原名>_knee.npz \
  --knee-min 0.52
```

- `--knee-min 0.52` ≈ 30° 微屈下限，避免机器人膝盖伸直锁死。按需调。
- 脚本会打印每个膝盖列 clamp 前后的 min/median，确认改了哪些帧。

#### 3.3.2 整体改成半蹲（最常用，改腿部不破坏脚步）

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python deploy/robots/g1/squat_bias_npz.py \
  src/assets/motions/g1/<原名>.npz \
  src/assets/motions/g1/<原名>_squat.npz \
  --hip-offset -0.20 --knee-offset 0.35 --ankle-offset -0.15 --knee-floor 0.30
```

参数含义（正负方向以 G1 约定为准）：

| 参数 | 默认 | 作用 |
|------|------|------|
| `--hip-offset` | -0.20 | 髋俯仰偏移，负 = 髋更屈（蹲下） |
| `--knee-offset` | +0.35 | 膝偏移，正 = 更屈 |
| `--ankle-offset` | -0.15 | 踝偏移，负 = 背屈 |
| `--knee-floor` | 0.30 | 偏移后膝盖硬下限（不伸直） |
| `--no-root-comp` | — | 加此参数跳过「落脚锚定」（脚会陷地，一般别加） |

- 脚本会自动**降低骨盆高度**，让脚保持在原始世界高度 → 舞步/跳跃/抬脚完全保留，只是腿在下面蹲着。
- 完成后打印 `foot-anchor max residual`（脚锚定残差，越小越好）。

#### 3.3.3 改完后的三步收尾

1. **验证**：用 `analyze_npz_legs.py` 看腿部各列 min/median（改脚本里的 `FILES` 路径列表，或自己写 `np.load` 打印）。
2. **重新训练**：改的是**参考动作**，必须用新 npz 重新跑第 2.4 步训练（旧策略不认新动作）。
3. **重新导出部署**：拿新训练的 `policy.onnx` + 新 npz 走第 4 章部署。

### 3.4 快速查看腿部数据（不写脚本直接看）

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
conda activate unitree_rl_mjlab
python -c "
import numpy as np
d = np.load('src/assets/motions/g1/<名称>.npz')
jp = d['joint_pos']
for name, c in {'L_knee':3, 'R_knee':9, 'L_hip_pitch':0, 'L_ankle_pitch':4}.items():
    print(name, 'min %.3f med %.3f max %.3f' % (jp[:,c].min(), np.median(jp[:,c]), jp[:,c].max()))
"
```

---

## 4. 网线直连服务器做舞蹈测试

### 4.1 网络拓扑

```
本机 ──SSH(WiFi)──▶ 服务器 ──网线直连(enp6s0)──▶ 机器人
                    10.101.x.x              192.168.123.x
```

- 服务器 WiFi IP：`10.101.x.x`（每次登录后 `ip a` 确认，动态分配）。
- 服务器有线口 `enp6s0` 固定 `192.168.123.99`（netplan 配置）。
- 机器人运控 PC1 实测在 `192.168.123.161`（**不是** `.99`，`.99` 是服务器自己的网口 IP）。

### 4.2 连接服务器

```bash
# 服务器先开机登显示器（按 Enter 输密码），然后 ip a 看 10.101 开头的无线 IP
ssh ubuntu@<10.101.x.x>
```

### 4.3 打通服务器 → 机器人（netplan）

```bash
# 服务器上 ping 机器人
ping 192.168.123.99
```

- ✅ 有响应 → 已通，跳到 4.4。
- ❌ 100% 丢包 → 在**服务器**上执行（推荐用脚本，一步到位）：

```bash
sudo ./fix_robot_network.sh   # 该脚本在仓库根，会覆盖 netplan 并 apply
```

或手动（与脚本等效）：

```bash
sudo nano /etc/netplan/50-cloud-init.yaml   # 写入下面内容
sudo chmod 600 /etc/netplan/*.yaml
sudo netplan generate && sudo netplan try   # try 确认后按 Enter 保留，超时自动回滚
sudo netplan apply
ping 192.168.123.99                          # 再次验证
```

netplan 内容：

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp6s0:
      dhcp4: no
      addresses:
        - 192.168.123.99/24
      optional: true
```

### 4.4 准备部署三件套 + 注册 FSM

一个可部署舞蹈 = **三件套文件 + 一条 FSM 配置**：

```
deploy/robots/g1/config/policy/mimic/<动作名>/
├── exported/policy.onnx        # actor-only 策略（训练 run 里的 policy.onnx）
└── params/
    ├── deploy.yaml             # 观测/动作规格（No-State-Estimation 固定，直接复制）
    └── <动作名>.npz            # 参考动作（50 fps，第 2/3 章产物）
```

```bash
cd /home/silence/MechDance/unitree_rl_mjlab
NAME=<动作名>          # 一般用 npz 的 hash 短名
RUNDIR=logs/rsl_rl/g1_tracking/<run>

mkdir -p deploy/robots/g1/config/policy/mimic/$NAME/exported
mkdir -p deploy/robots/g1/config/policy/mimic/$NAME/params
cp $RUNDIR/policy.onnx deploy/robots/g1/config/policy/mimic/$NAME/exported/policy.onnx
cp src/assets/motions/g1/<原名>.npz deploy/robots/g1/config/policy/mimic/$NAME/params/$NAME.npz
# deploy.yaml 规格固定，从现有动作目录复制
cp deploy/robots/g1/config/policy/mimic/17fe0adf/params/deploy.yaml \
   deploy/robots/g1/config/policy/mimic/$NAME/params/deploy.yaml
```

然后编辑 `deploy/robots/g1/config/config.yaml`，**【必须】三处都要加**（以按钮 `RB + Y` 为例）：

1. 启用列表 `FSM._`：给唯一 id
2. `Velocity.transitions`：加 `Mimic_<NAME>: RB + Y.on_pressed`
3. 新增 state block（含内部 transitions + 三件套路径）

```yaml
# ① 启用列表
    Mimic_<NAME>:
      id: 8
      type: Mimic

# ② Velocity 的 transitions
  Velocity:
    transitions:
      Mimic_<NAME>: RB + Y.on_pressed

# ③ 新增 state block
  Mimic_<NAME>:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: RT + A.on_pressed
    motion_file: config/policy/mimic/<NAME>/params/<NAME>.npz
    policy_dir: config/policy/mimic/<NAME>/
    time_start: 0.0
    time_end: 1000.0
```

> **【必须】验证**：`grep -n "<NAME>" config/config.yaml` 应出现 **3 次**（三处都加全，否则动作不会被注册）。

### 4.5 上传到服务器并运行

**本机** scp 上传：

```bash
cd /home/silence/MechDance/unitree_rl_mjlab/deploy/robots/g1
SERVER=ubuntu@<10.101.x.x>
DEPLOY=/home/ubuntu/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1

scp config/config.yaml $SERVER:$DEPLOY/config/config.yaml
scp -r config/policy/mimic/<NAME> $SERVER:$DEPLOY/config/policy/mimic/
```

**服务器**运行：

```bash
cd ~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=enp6s0
```

- 启动打印 `Waiting for connection rt/lowstate` 是正常的，等机器人连。
- 连上后应看到 `Initializing State_Mimic_<NAME>` 和 `Loaded motion file '<NAME>' ... duration XX.XXs`。

> **【必须】** config.yaml 是**启动时加载**的：改了 config.yaml / 动作包后，必须重新 scp 上传并**重启 `g1_ctrl`**（Ctrl+C 后重新运行），否则改动不生效。

### 4.6 真机触发（务必先按 L2+R2）

1. 机器人上电。
2. 手柄 **L2+R2** 进**硬件调试模式**（lowcmd 放行闸门，**【必须】**，否则关节指令不下发）。
3. 连上后按手柄 FSM 触发（见第 6 章按钮表）。

---

## 5. 部署到机器人内部 PC2（脱离服务器独立运行）

让机器人**不依赖任何外部服务器/网线/WiFi**：把 `g1_ctrl` 编译并跑在机器人板载开发计算机 **PC2** 上，上电自动启动，全程手柄操作。

### 5.1 硬件与关键事实

| 单元 | 角色 | 是否开放 | IP |
|------|------|---------|-----|
| PC1 | 运控计算单元（官方运动控制） | ❌ | `192.168.123.161` |
| PC2 | 开发计算单元（可二次开发） | ✅ SSH | `192.168.123.164` |

- PC2：**aarch64 + Ubuntu 20.04**，SSH `unitree@192.168.123.164`，密码 `123`。
- PC2 连机器人内部网络的网口是 **`eth0`**（运行 `g1_ctrl --network=eth0`）。
- ⚠️ **PC2 只接受来自服务器网段（192.168.123.x）的 SSH**，本机直连会被拒。**【必须】所有文件经本机 → 服务器 → PC2 中转**，且从服务器发起 scp 到 PC2。

### 5.2 传输源码 + 依赖到 PC2

保持相对结构（CMakeLists 用 `../../thirdparty` 解析 onnxruntime）：

```
deploy/
├── robots/g1/       # main.cpp, CMakeLists.txt, include/, src/, config/
├── include/         # 共享 FSM/工具头文件
└── thirdparty/      # onnxruntime-linux-aarch64-1.22.0 + cnpy
```

**服务器终端**：

```bash
cd ~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy
scp -r robots/g1 include thirdparty unitree@192.168.123.164:~/
ssh unitree@192.168.123.164
ls ~/deploy/robots/g1/   # 应有 CMakeLists.txt main.cpp config include src
```

若服务器上是旧包，先从本机更新到服务器（4 支舞动作包 + config）：

```bash
SERVER=ubuntu@<10.101.x.x>
DEPLOY=~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1
cd /home/silence/MechDance/unitree_rl_mjlab/deploy/robots/g1
scp config/config.yaml $SERVER:$DEPLOY/config/config.yaml
scp -r config/policy/mimic/957d1c6f config/policy/mimic/17fe0adf \
      config/policy/mimic/93ccdf54 config/policy/mimic/c2197703 \
      $SERVER:$DEPLOY/config/policy/mimic/
```

### 5.3 补齐 PC2 缺失依赖（4 个坑，【必须】逐个照做）

> PC2 无外网，所需 deb / 头文件 / 库都要从本机（有外网）下载 → 服务器中转 → PC2 安装。**注意 arm64 架构**。

**坑 ① 缺 libfmt-dev**

```bash
# 本机（有外网）：下 arm64 版 deb
wget -P /tmp/ "https://ports.ubuntu.com/pool/universe/f/fmtlib/libfmt-dev_6.1.2+ds-2_arm64.deb"
scp /tmp/libfmt-dev_6.1.2+ds-2_arm64.deb ubuntu@<10.101.x.x>:~/

# 服务器：中转
scp ~/libfmt-dev_6.1.2+ds-2_arm64.deb unitree@192.168.123.164:~/

# PC2：安装
sudo dpkg -i ~/libfmt-dev_6.1.2+ds-2_arm64.deb
```

**坑 ② 缺新版 SDK2 头文件（dds_wrapper）**

```bash
# 服务器：整树推新版头文件
scp -r /usr/local/include/unitree unitree@192.168.123.164:/tmp/

# PC2：备份旧版 + 覆盖
echo '123' | sudo -S mv /usr/local/include/unitree /usr/local/include/unitree.bak.old
echo '123' | sudo -S cp -r /tmp/unitree /usr/local/include/
ls /usr/local/include/unitree/dds_wrapper/robots/go2/go2.h   # 应存在
```

**坑 ③ 缺新版 SDK2 aarch64 静态库（链接符号缺失）**

```bash
# 服务器：推 aarch64 静态库
scp /home/ubuntu/unitree_sdk2/lib/aarch64/libunitree_sdk2.a unitree@192.168.123.164:/tmp/

# PC2：替换
echo '123' | sudo -S cp /usr/local/lib/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a.bak.old
echo '123' | sudo -S cp /tmp/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a
```

**坑 ④ onnxruntime 符号链接被 scp 弄坏**

```bash
# PC2
cd ~/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib
rm -f libonnxruntime.so.1 libonnxruntime.so
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so.1
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so
ls -la libonnxruntime.so*   # 应显示 -> libonnxruntime.so.1.22.0
```

### 5.4 在 PC2 上编译 g1_ctrl

```bash
# PC2
cd ~/deploy/robots/g1
rm -rf build && mkdir -p build && cd build
cmake ..
make -j4
ls -la g1_ctrl   # 约 9.7MB 可执行文件
```

### 5.5 配置动作包 + systemd 开机自启

校验 config 与动作包（md5 与本机一致）：

```bash
cd ~/deploy/robots/g1
grep -nE "Mimic_|RB \+" config/config.yaml
md5sum config/policy/mimic/*/exported/policy.onnx
```

写 systemd 服务文件（`/etc/systemd/system/g1_ctrl.service`）：

```bash
sudo -S tee /etc/systemd/system/g1_ctrl.service > /dev/null <<'EOF'
[Unit]
Description=Unitree G1 Motion Controller (g1_ctrl)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=unitree
WorkingDirectory=/home/unitree/deploy/robots/g1/build
Environment=LD_LIBRARY_PATH=/usr/local/lib:/home/unitree/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib
ExecStart=/home/unitree/deploy/robots/g1/build/g1_ctrl --network=eth0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

加载 + 启用 + 启动（**【必须】三条缺一不可**）：

```bash
systemd-analyze verify /etc/systemd/system/g1_ctrl.service   # 先验证语法
sudo systemctl daemon-reload
sudo systemctl enable g1_ctrl    # 开机自启（不立即启动）
sudo systemctl start g1_ctrl     # 立即启动（不自启）
systemctl is-enabled g1_ctrl     # 应输出 enabled
systemctl is-active g1_ctrl      # 应输出 active
```

**判断真正就绪以日志为准，不是 `active (running)`**：

```bash
journalctl -u g1_ctrl -n 30 --no-pager
```

成功标志（应看到）：

```
Connected to robot.
Initializing State_Mimic_957d1c6f / 17fe0adf / 93ccdf54 / c2197703 ...
Loaded motion file '957d1c6f' duration 61.44s ...
FSM: Start Passive
```

> ⚠️ 改动作包后必须 `sudo systemctl restart g1_ctrl`（配置在启动时加载），且**不要手动 `./g1_ctrl`**——会和 systemd 抢 lowcmd 通道。

---

## 6. 手柄控制机器人

### 6.1 先分清「两个模式」

| | 是什么 | 触发 | 谁管 |
|---|--------|------|------|
| 硬件调试模式 | 机器人接受外部关节指令的**闸门** | `L2 + R2` 同时按 | 机器人硬件层 |
| FSM 状态机 | g1_ctrl 软件层状态切换 | 各种按键组合 | `config.yaml` |

**【必须】跳舞前先按 `L2+R2` 进硬件调试模式**，否则按什么都没反应。

### 6.2 手柄操作流程（通用）

```
1. 机器人上电（独立部署等 PC2 自动启动 g1_ctrl，约 30s；服务器模式先手动 ./g1_ctrl）
2. L2 + R2     → 进硬件调试模式（必须）
3. L2 + 上      → 站立（FixStand）
4. R2 + A      → 进 Velocity（速度控制）
5. RB + <按钮>  → 播放对应舞蹈
6. LT + B      → 停止（回 Passive 待机）
```

### 6.3 当前按钮映射（以 `config/config.yaml` 为准）

| 动作 | 入口按钮 | 说明 |
|------|---------|------|
| 957d1c6f | RB + A | 冻结版（已部署） |
| 17fe0adf | RB + B | 冻结版（已部署） |
| 93ccdf54 | RB + X | 原始 raw 版 |
| c2197703 | RB + Y | 原始 raw 版 |
| 8yue12_01 | RB + 下 | |
| clip_130_135s | RB + 上 | |
| f85854e8 | RB + 左 | |

状态切换按钮：

| 切换 | 按钮 |
|------|------|
| Passive → FixStand（站立） | L2 + 上 |
| FixStand → Velocity | R2 + A |
| 任意状态 → Passive（停止） | LT + B |
| 任意 Mimic → Velocity | R2 + A |
| Mimic 之间切换 | 各动作块里写的 `RB + X` 等（见 config.yaml） |

### 6.4 按钮 DSL 语法要点

- `RB + Y.on_pressed` = RB 按住 + Y **刚按下**（边沿触发）。
- `LT + B.on_pressed` = LT 按住 + B 刚按下。
- 大小写不敏感，`x/y/a/b/up/down/left/right` 均有效。

---

## 7. 故障排查速查

| 症状 | 原因 | 处理 |
|------|------|------|
| 一直 `Waiting for connection rt/lowstate` | DDS 没通 / 网口错 / 机器人没上电 | 检查 `--network`（服务器=enp6s0，PC2=eth0）；`ping 192.168.123.99`（服务器）/ `ping 192.168.123.161`（PC2）；确认机器人已上电 |
| 手柄按了没反应 | 没进硬件调试模式 / config 没更新 / 服务没重启 | 先按 `L2+R2`；config 启动时加载，改后重新 scp + 重启 g1_ctrl |
| 没出现 `Initializing State_Mimic_<NAME>` | config.yaml 三处没加全 / 状态名不一致 | `grep -n "<NAME>" config/config.yaml` 应出现 3 次 |
| `Loaded motion file` 帧数/时长不对 | npz 不是 50fps / 缺 key | Python 检查 npz 的 key 和 shape |
| 动作播了但不对 | 用了 `<run>.onnx` 打包版 / npz 与训练不对应 | 用 actor-only `policy.onnx`；比对 npz md5 |
| 真机抖动 / BMS 弹电池 | 手改 `joint_pos` 没重算 body FK | 用第 3 章脚本重生成一致性 npz，重新训练 |
| PC2 编译 `No SOURCES ... cnpy_lib` | 目录结构不对 | 保持 `deploy/robots/g1` + `deploy/thirdparty` 相对结构 |
| PC2 编译 `dds_wrapper/...go2.h: No such file` | 缺新版 SDK2 头文件 | 第 5.3 坑② |
| PC2 链接 `undefined reference to get_type_props` | 旧版 SDK2 静态库 | 第 5.3 坑③ |
| PC2 运行 `libonnxruntime.so.1: file too short` | scp 弄坏符号链接 | 第 5.3 坑④ |
| PC2 `is-active` 是 active 但机器人不动 | 服务活着 ≠ 连上运控 | 看 `journalctl -u g1_ctrl` 是否 `Connected to robot.` |

---

## 8. 命令速查

### 8.1 视频 → RL

```bash
# 一步到位（覆盖目录）
cd /home/silence/MechDance
GVHMR_DIR=/home/silence/MechDance/GVHMR GMR_DIR=/home/silence/MechDance/GMR \
  ./unitree_rl_mjlab/run_video_to_rl.sh --video x.mp4 --name my_dance --num-envs 1024
```

### 8.2 服务器（网线直连）

```bash
ssh ubuntu@<10.101.x.x>                        # 连服务器
sudo ./fix_robot_network.sh                     # 服务器上修复机器人网络
ping 192.168.123.99                             # 验通
scp <本地文件> ubuntu@<10.101.x.x>:~/server/beigongshang/mechdance/...   # 上传
cd ~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=enp6s0                      # 运行
```

### 8.3 独立部署（PC2）

```bash
ssh unitree@192.168.123.164                     # 密码 123
# 服务器 → PC2
scp -r robots/g1 include thirdparty unitree@192.168.123.164:~/
# PC2 编译
cd ~/deploy/robots/g1/build && cmake .. && make -j4
# 服务管理
sudo systemctl start|restart|enable|disable g1_ctrl
systemctl status g1_ctrl --no-pager
journalctl -u g1_ctrl -f                        # 实时日志
```

### 8.4 腿部修改

```bash
cd /home/silence/MechDance/unitree_rl_mjlab && conda activate unitree_rl_mjlab
# 半蹲（推荐）
python deploy/robots/g1/squat_bias_npz.py in.npz out.npz --hip-offset -0.2 --knee-offset 0.35 --ankle-offset -0.15 --knee-floor 0.3
# 只 clamp 膝盖 + 重算 FK
python deploy/robots/g1/regenerate_consistent_npz.py in.npz out.npz --knee-min 0.52
```

---

## 附：关键文件索引

| 文件 | 内容 |
|------|------|
| `unitree_rl_mjlab/run_video_to_rl.sh` | 视频 → RL 一键脚本 |
| `unitree_rl_mjlab/deploy/robots/g1/DEPLOYMENT.md` | 服务器模式部署 |
| `unitree_rl_mjlab/deploy/robots/g1/DEPLOYMENT_STANDALONE.md` | PC2 独立部署 |
| `11.md` | 服务器↔机器人网络（netplan/SSH） |
| `fix_robot_network.sh` | netplan 一键修复 |
| `unitree_rl_mjlab/deploy/robots/g1/squat_bias_npz.py` | 腿部半蹲偏移 + FK 重算 |
| `unitree_rl_mjlab/deploy/robots/g1/regenerate_consistent_npz.py` | 膝盖 clamp + FK 重算 |
| `unitree_rl_mjlab/deploy/robots/g1/fix_knee_npz.py` | 仅 clamp 膝盖（勿用于真机） |
| `unitree_rl_mjlab/deploy/robots/g1/analyze_npz_legs.py` | 腿部数据统计 |
| `unitree_rl_mjlab/scripts/csv_to_npz.py` | CSV → NPZ（含 29 关节顺序定义） |
