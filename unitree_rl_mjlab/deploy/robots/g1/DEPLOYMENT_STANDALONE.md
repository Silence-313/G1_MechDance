# G1 脱离服务器独立部署指南（板载 PC2 运行 g1_ctrl）

本文说明如何让 Unitree G1 **完全不依赖服务器/上位机**独立跳预设舞蹈：把控制程序 `g1_ctrl` 直接编译、运行在**机器人板载开发计算机（PC2）**上。机器人上电后 `g1_ctrl` 自动启动并连上机器人运控，之后全程用手柄操作。

**范围**：只覆盖「板载独立部署」这一条路径。传统「服务器 + 网线」模式见同目录 `DEPLOYMENT.md`，服务器/机器人网络拓扑见仓库根目录 `11.md`。

---

## 1. 原理与硬件架构

### 1.1 为什么能脱离服务器

Unitree G1-EDU 机载是**双计算单元**：

| 单元 | 角色 | 是否开放 | IP |
|---|---|---|---|
| **PC1** | 运控计算单元（跑官方运动控制） | ❌ 不开放 | `192.168.123.161` |
| **PC2** | 开发计算单元（开发者可二次开发） | ✅ 可 SSH | `192.168.123.164` |

传统模式里 `g1_ctrl` 跑在**外部服务器**上，通过网线 + DDS 组播远程控制机器人。而 PC2 就在机器人内部，**天然连着运控（PC1）的总线**——只要把 `g1_ctrl` 编译并跑在 PC2 上，就不再需要任何外部机器、网线、WiFi。

### 1.2 关键事实（本次实测确认）

- PC2：**aarch64（ARM64）+ Ubuntu 20.04**，默认 SSH `unitree@192.168.123.164`，密码 `123`
- PC2 连机器人内部网络用的是 **`eth0`**（`192.168.123.164/24`）——运行 g1_ctrl 时 `--network=eth0`
- DDS 靠**组播**发现运控发布的 lowstate，只要网口选对就能连上
- PC2 的 systemd 服务 `g1_ctrl.service` 已配好**开机自启 + 崩溃自动重启**

### 1.3 两种模式对比

| | 服务器模式（DEPLOYMENT.md） | 板载独立模式（本文） |
|---|---|---|
| g1_ctrl 跑在哪 | 外部服务器 | 机器人 PC2 |
| 需要网线/服务器 | ✅ 需要 | ❌ 不需要 |
| 上电后 | 手动 scp + 启动 | 自动启动 |
| 适用场景 | 开发调试、频繁换动作 | 演示、演出、稳定运行 |

---

## 2. 部署总览

```
[本机 x86_64] ──scp──▶ [服务器(中转)] ──scp──▶ [PC2 aarch64] ──运行──▶ 机器人
  编辑源码/动作包          保留一份源码            编译+运行 g1_ctrl         跳舞
```

- **本机**：源码编辑、动作包制作（本仓库所在机器，x86_64）
- **服务器**：中转站 + 源码备份（`~/server/beigongshang/mechdance/...`）
- **PC2**：最终编译、运行 g1_ctrl 的地方

> ⚠️ **PC2 只接受来自服务器网段（192.168.123.x）的 SSH**——从本机直连 PC2 会被拒（`kex_exchange_identification` / `Error reading SSH protocol banner`）。所以**所有文件必须经服务器中转**，且从服务器发起 `scp` 到 PC2。

首次部署共 5 步（3~7 章）；日常使用只有 1 步（第 8 章）。

---

## 3. 第一步：确认 PC2 环境

登录 PC2 并确认工具链。**在 PC2 终端**：

```bash
ssh unitree@192.168.123.164    # 密码 123

# ① 架构（必须 aarch64，Ubuntu 20.04）
uname -m
lsb_release -a

# ② 编译工具链（都应在）
which cmake g++ make

# ③ 依赖库头文件（都应在）
ls /usr/local/include/ddscxx
ls /usr/local/include/iceoryx/v2.0.2
ls /usr/include/eigen3

# ④ 已装库（确认 boost / yaml-cpp / spdlog / fmt）
dpkg -l | grep -E "libyaml-cpp|libspdlog|libfmt" | awk '{print $2}'
```

**注意**：PC2 的 `unitree_sdk2` 是**旧版**（`/usr/local/include/unitree/robot/go2/` 老结构），而本项目代码需要**新版**（`unitree/dds_wrapper/robots/go2/go2.h`）。后文第 5 步会覆盖处理。

---

## 4. 第二步：传输源码 + 依赖到 PC2

PC2 无外网，所有东西从服务器（或本机经服务器）手动 scp。

### 4.1 需要传输的内容

编译需要三个目录，**必须保持相对结构**（CMakeLists 用 `../../thirdparty` 相对路径解析 onnxruntime）：

```
deploy/
├── robots/g1/       # main.cpp, CMakeLists.txt, include/, src/, config/
├── include/         # 共享 FSM/工具头文件
└── thirdparty/      # onnxruntime-linux-aarch64-1.22.0 (编译必需) + cnpy
```

### 4.2 从服务器传给 PC2

**在【服务器终端】**执行（把 `deploy/` 下这三个目录 scp 到 PC2 主目录，目标结构是 `~/deploy/robots/g1`）：

```bash
cd ~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy

# 目的：把源码+依赖从服务器推到 PC2（PC2 只接受服务器网段，故从服务器发起）
scp -r robots/g1 include thirdparty unitree@192.168.123.164:~/

# 传完后登录 PC2 确认结构
ssh unitree@192.168.123.164
ls ~/deploy/robots/g1/    # 应有 CMakeLists.txt main.cpp config include src
```

> 若服务器上源码/动作包是旧版，先从**本机**把最新内容 scp 到服务器再往下走（见第 4.3 节）。

### 4.3 若服务器上是旧包，先从本机更新到服务器

**在【本机终端】**：

```bash
# 目的：把本机最新的 config.yaml + 4 支舞动作包推到服务器中转
SERVER=ubuntu@10.101.195.200
DEPLOY=~/server/beigongshang/mechdance/unitree_rl_mjlab/deploy/robots/g1
cd /home/silence/MechDance/unitree_rl_mjlab/deploy/robots/g1

scp config/config.yaml $SERVER:$DEPLOY/config/config.yaml
scp -r config/policy/mimic/957d1c6f config/policy/mimic/17fe0adf \
      config/policy/mimic/93ccdf54 config/policy/mimic/c2197703 \
      $SERVER:$DEPLOY/config/policy/mimic/
```

---

## 5. 第三步：补齐 PC2 缺失的依赖（4 个坑）

PC2 原装环境与本项目代码不匹配，需要补 4 样东西。**这些坑本次已逐个踩过并解决**，直接照做即可。

### 坑 ①：缺 libfmt-dev（编译报 `fmt` 找不到 / spdlog 依赖）

PC2 无外网，需从本机下载 **arm64 版** deb 上传安装。**注意架构**：本机是 x86_64，PC2 是 **arm64**，必须下 `_arm64.deb`。

**在【本机终端】**下载（本机有外网）：

```bash
# 目的：下载 Ubuntu 20.04 focal 的 arm64 libfmt-dev（静态库+头文件，无运行时依赖）
wget -P /tmp/ "https://ports.ubuntu.com/pool/universe/f/fmtlib/libfmt-dev_6.1.2+ds-2_arm64.deb"
scp /tmp/libfmt-dev_6.1.2+ds-2_arm64.deb ubuntu@10.101.195.200:~/
```

**在【服务器终端】**中转 + **在【PC2 终端】**安装：

```bash
# 服务器：推到 PC2
scp ~/libfmt-dev_6.1.2+ds-2_arm64.deb unitree@192.168.123.164:~/
# PC2：安装
sudo dpkg -i ~/libfmt-dev_6.1.2+ds-2_arm64.deb
```

> `libfmt-dev` 依赖只有 `libc6/libgcc-s1/libstdc++6`（PC2 全有），装完即可。

### 坑 ②：缺新版 SDK2 头文件（编译报 `unitree/dds_wrapper/robots/go2/go2.h: No such file`）

PC2 原装 SDK2 是旧版，没有 `dds_wrapper`。**从服务器拷新版头文件**。**在【服务器终端】**：

```bash
# 目的：把服务器新版 SDK2 头文件整树 scp 到 PC2 临时目录
scp -r /usr/local/include/unitree unitree@192.168.123.164:/tmp/
```

**在【PC2 终端】**备份旧版 + 覆盖：

```bash
# 目的：用新版头文件替换旧版（旧版备份为 .bak.old）
echo '123' | sudo -S mv /usr/local/include/unitree /usr/local/include/unitree.bak.old
echo '123' | sudo -S cp -r /tmp/unitree /usr/local/include/
ls /usr/local/include/unitree/dds_wrapper/robots/go2/go2.h   # 应存在
```

### 坑 ③：缺新版 SDK2 aarch64 静态库（链接报 `undefined reference to get_type_props<...>()`）

头文件换新后，链接 PC2 旧版 `libunitree_sdk2.a` 会因**符号缺失**失败（旧库没有新版 IDL 类型的序列化符号）。需换**新版 aarch64 静态库**。**在【服务器终端】**：

```bash
# 目的：服务器新版 SDK2 的 aarch64 静态库（官方 SDK lib/ 下同时带 aarch64+x86_64）
scp /home/ubuntu/unitree_sdk2/lib/aarch64/libunitree_sdk2.a unitree@192.168.123.164:/tmp/
```

**在【PC2 终端】**替换：

```bash
echo '123' | sudo -S cp /usr/local/lib/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a.bak.old
echo '123' | sudo -S cp /tmp/libunitree_sdk2.a /usr/local/lib/libunitree_sdk2.a
```

### 坑 ④：onnxruntime 符号链接被 scp 弄坏（运行报 `libonnxruntime.so.1: file too short`）

`scp` 会把符号链接传成普通文本文件（里面存链接目标字符串），运行时加载报错。**在【PC2 终端】**重建符号链接：

```bash
cd ~/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib
# 目的：scp 把符号链接传成了普通文件，删除并重建为真实软链
rm -f libonnxruntime.so.1 libonnxruntime.so
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so.1
ln -s libonnxruntime.so.1.22.0 libonnxruntime.so
ls -la libonnxruntime.so*   # 应显示 -> libonnxruntime.so.1.22.0
```

---

## 6. 第四步：在 PC2 上编译 g1_ctrl

**在【PC2 终端】**：

```bash
cd ~/deploy/robots/g1
# 目的：干净编译（若上次编译残留 build 需先删，否则 CMakeCache 路径报错）
rm -rf build && mkdir -p build && cd build

# 目的：cmake 配置，自动检测 aarch64 走 onnxruntime-linux-aarch64 分支
cmake ..
# 目的：编译 g1_ctrl（-j4 用 4 核并行）
make -j4

# 确认产物
ls -la g1_ctrl    # 约 9.7MB 可执行文件
```

**成功标志**：`make` 无错误，生成 `g1_ctrl`。若中途报错，对照第 5 章四个坑逐一检查。

---

## 7. 第五步：配置动作包 + systemd 开机自启

### 7.1 确认动作包已就位

**在【PC2 终端】**校验 config.yaml 与动作包（关键：md5 必须与本机一致，否则动作不对）：

```bash
cd ~/deploy/robots/g1

# 目的：确认 4 支舞按键映射（应只有这 4 个，无残留旧舞）
grep -nE "Mimic_|RB \+" config/config.yaml

# 目的：校验 policy.onnx 是否为最新版（对比下表的期望 md5）
md5sum config/policy/mimic/957d1c6f/exported/policy.onnx \
        config/policy/mimic/17fe0adf/exported/policy.onnx \
        config/policy/mimic/93ccdf54/exported/policy.onnx \
        config/policy/mimic/c2197703/exported/policy.onnx
```

| 舞 | 按键 | 版本 | policy.onnx 期望 md5 |
|---|---|---|---|
| 957d1c6f | RB + A | 冻结版（回溯） | `2ad5cdc5014827d8efa8e86184c80626` |
| 17fe0adf | RB + B | 冻结版（回溯） | `b08052c76d184b14275515bf2ead65c7` |
| 93ccdf54 | RB + X | 原始 raw 版 | `3dab373dca22ccf67bd8a30fd296652d` |
| c2197703 | RB + Y | 原始 raw 版 | `8beeab6cbafa2abe29b1536f68a118ce` |

### 7.2 配置 systemd 开机自启

systemd 负责在开机时把 `g1_ctrl` 拉起来并保持存活。**创建服务文件的完整流程**分三步：写文件 → 让 systemd 认识它 → 启用并启动。全部在 **PC2 终端** 执行。

#### 7.2.1 前置确认：服务文件里要写的路径必须真实存在

```bash
# 目的：确认二进制与 onnxruntime 库路径正确，服务文件才写不烂
ls -la /home/unitree/deploy/robots/g1/build/g1_ctrl
ls /home/unitree/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib/libonnxruntime.so.1.22.0
```

这两个路径如果不存在，先回头做第 4/6 章，再继续。

#### 7.2.2 写服务文件

用 `tee` + heredoc 创建，`sudo` 是因为要写到系统目录 `/etc/systemd/system/`：

```bash
# 目的：把服务文件内容写入 /etc/systemd/system/g1_ctrl.service
#   tee 会把 stdin（heredoc）写入文件并带 sudo 权限
#   > /dev/null 丢弃 tee 的回显，只看错误
#   <<'EOF' 是 heredoc，引号保证 $ 变量和反斜杠不被 shell 解释
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

**服务文件逐字段说明**：

| 字段 | 值 | 用途 |
|---|---|---|
| `Description` | G1 Motion Controller | 服务说明，`systemctl status` 显示用 |
| `After=network-online.target` | | 在系统网络就绪**之后**才启动（DDS 需要 eth0 有 IP） |
| `Wants=network-online.target` | | 请求激活网络就绪目标（软依赖，不强制） |
| `Type=simple` | | 认为 ExecStart 启动的进程就是主进程（g1_ctrl 是前台常驻程序，适合） |
| `User=unitree` | | 以 unitree 身份运行（不用 root 跑控制程序） |
| `WorkingDirectory` | build 目录 | 进程工作目录，配置加载按相对路径解析时依赖它 |
| `Environment=LD_LIBRARY_PATH` | /usr/local/lib + onnxruntime lib | 运行时找 DDS 库和 onnxruntime 库的路径（关键，缺了运行报库找不到） |
| `ExecStart` | g1_ctrl --network=eth0 | 实际执行的命令 + 参数（eth0 是 PC2 连运控的网口） |
| `Restart=always` | | 无论因何退出都自动重启（崩溃/被杀都拉起） |
| `RestartSec=5` | | 重启前等待 5 秒，防止疯狂重启打日志 |
| `WantedBy=multi-user.target` | | 挂到多用户启动目标，**开机时自动启动**（这就是自启的关键） |

> 三个 `[Install]` 相关字段只在 `systemctl enable` 时生效（生成开机启动链接）；`[Unit]/[Service]` 字段在 `start`/开机时生效。

#### 7.2.3 验证服务文件写对了

```bash
# 目的：检查文件语法 + 内容是否正确（systemd-analyze 会报语法错误）
systemd-analyze verify /etc/systemd/system/g1_ctrl.service
cat /etc/systemd/system/g1_ctrl.service
```

无报错、内容完整，继续下一步。

#### 7.2.4 加载、启用、启动

```bash
# ① daemon-reload：让 systemd 重新扫描服务文件（改了文件/新建文件后必须做）
sudo systemctl daemon-reload

# ② enable：开机自启（生成 /etc/systemd/system/multi-user.target.wants/ 软链）
sudo systemctl enable g1_ctrl

# ③ start：立即启动（等效于手动 ./g1_ctrl，但归 systemd 托管）
sudo systemctl start g1_ctrl

# ④ 确认：is-enabled 应输出 enabled，is-active 应输出 active
systemctl is-enabled g1_ctrl
systemctl is-active g1_ctrl
systemctl status g1_ctrl --no-pager
```

**三条命令各自目的**（缺一不可）：
- `daemon-reload` —— 让 systemd 读到新写的服务文件，**新建/修改服务文件后必须执行**，否则 systemd 用的还是旧配置
- `enable` —— 只做**开机自启**（写入启动软链），**不会立即启动**
- `start` —— 只做**立即启动**，**不做**开机自启

> 只 `enable` 不开机生效；只 `start` 不随开机启动。两者要一起用才「现在能跑 + 以后开机自动跑」。

#### 7.2.5 确认真正连上运控

```bash
# 目的：看启动日志，确认连上运控 + 4 支舞加载成功
journalctl -u g1_ctrl -n 30 --no-pager
```

**成功标志**（journalctl 应看到）：

```
Connected to robot.
Initializing State_Mimic_957d1c6f / 17fe0adf / 93ccdf54 / c2197703 ...
Loaded motion file '957d1c6f' duration 61.44s  ...（各舞一行）
FSM: Start Passive
```

> ⚠️ `systemctl status` 显示 `active (running)` **只代表进程活着，不代表连上运控**。判断真正就绪永远以 journalctl 里的 `Connected to robot.` 为准。

### 7.3 systemd 自动拉起的完整链路（上电 → 连上运控）

理解这条链路的每一环，才能在出问题时定位卡在哪一步。g1_ctrl 由 systemd 托管，**从上电到连上运控**的过程如下：

```
机器人上电
  → PC2 系统启动（Ubuntu 20.04 开机）
  → systemd 进入 multi-user.target
  → 激活 network-online.target（等 eth0 网络就绪，DDS 需要它）
  → 启动 g1_ctrl.service（WantedBy=multi-user.target，依赖 Wants=network-online.target）
  → 执行 ExecStart=.../g1_ctrl --network=eth0
  → g1_ctrl 初始化 DDS（绑定 eth0，走组播）
  → 收到机器人运控（PC1）发布的 lowstate → 打印 "Connected to robot."
  → 初始化 4 个 Mimic 状态，加载动作文件
  → 进入 FSM: Passive，等待手柄
```

**每一步的验证方法**（在 PC2 上）：

```bash
# ① 服务是否被 systemd 接管且开机自启已启用
systemctl is-enabled g1_ctrl        # 应输出 enabled
systemctl is-active g1_ctrl         # 应输出 active

# ② 服务进程是否在跑（Main PID 是 g1_ctrl）
systemctl status g1_ctrl --no-pager

# ③ 是否已连上运控 + 动作加载成功（核心验证）
journalctl -u g1_ctrl --since "1 min ago" --no-pager
#    应看到 "Connected to robot." 和 4 行 "Loaded motion file ... duration ...s"

# ④ eth0 网络是否就绪（network-online.target 依赖它）
ip a show eth0 | grep "state UP"

# ⑤ 若服务反复重启（Restart=always），看失败原因
systemctl status g1_ctrl --no-pager | grep -A5 "Active:"
journalctl -u g1_ctrl -b --no-pager | grep -iE "error|fail|waiting"
```

**各环节故障 → 对应处理**：

| 链路环节 | 故障表现 | 处理 |
|---|---|---|
| PC2 开机 | 上电后 `is-active` 是 inactive/dead | 检查服务文件、`daemon-reload` 后重新 `enable` |
| 等网络就绪 | 卡在 `Waiting for connection` 且 eth0 DOWN | `ip a` 看 eth0；确认网线/内部总线 |
| DDS 组播发现 | `Waiting for connection` 但 eth0 UP | `ping 192.168.123.161`（运控）；确认网口参数 `eth0` |
| 加载动作 | 日志缺某支舞的 `Loaded motion file` | config.yaml 三处注册、mimic 目录/文件是否齐全 |
| 崩溃后重启 | 日志反复出现启动 → 退出 | `journalctl -u g1_ctrl -b` 看退出原因，对照第 5 章四个坑 |

> **关键认知**：systemd 只负责「把 g1_ctrl 拉起来并保持存活」，**「连上运控」是 g1_ctrl 自己通过 DDS 完成的**。所以「服务 running」≠「连上运控」——判断是否真正就绪，永远以 journalctl 里的 `Connected to robot.` 为准，而不是 `systemctl status` 的绿色 active。

---

## 8. 日常使用（上电即跳，无需任何手动操作）

**上电后**：机器人上电 → PC2 启动 → systemd 自动拉起 g1_ctrl 并连上运控（约 30 秒）。然后：

### 手柄操作表

| 按键 | 作用 |
|---|---|
| **L2 + R2** 同时按 | 进**硬件调试模式**（机器人接受外部关节指令的闸门，**必须**先按） |
| **L2 + 上** | 站立（FixStand） |
| **R2 + A** | 进 Velocity（速度控制） |
| **RB + A** | 播放 957d1c6f |
| **RB + B** | 播放 17fe0adf |
| **RB + X** | 播放 93ccdf54 |
| **RB + Y** | 播放 c2197703 |
| **LT + B** | 停止动作（回 Passive 待机） |

### 完整操作流程

```
1. 机器人上电，等待 PC2 自动启动 g1_ctrl（约 30s）
2. 手柄 L2+R2 → 进硬件调试模式
3. L2 + 上 → 站立
4. R2 + A → Velocity
5. RB + A/B/X/Y → 播放对应舞蹈
6. LT + B → 停止
```

> **两个"模式"是两回事**：`L2+R2` 是机器人硬件层的调试模式（lowcmd 放行开关）；`L2+上 / R2+A / RB+* / LT+B` 是 g1_ctrl 软件层的 FSM 状态切换。跳舞前必须先按 `L2+R2`。

---

## 9. 换动作 / 更新模型

```bash
# 目的：替换某支舞的模型/动作后重启服务（config/npz/onnx 都是启动时加载）
# 在 PC2 上：
# 1) 用第 4/7 章方式把新 config.yaml + mimic 目录传到 PC2
# 2) 重启服务使其生效
sudo systemctl restart g1_ctrl
journalctl -u g1_ctrl -f    # 实时看日志确认加载成功
```

> 改动作包后**必须 `systemctl restart`**（加载发生在启动时），且**不要手动 `./g1_ctrl`**——会和 systemd 抢 lowcmd 通道。

---

## 10. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Waiting for connection rt/lowstate` | DDS 没通 / 网口错 | 确认 `--network=eth0`；PC2 上 `ping 192.168.123.161` |
| 编译报 `No SOURCES ... cnpy_lib` | 目录结构不对（`../../thirdparty` 解析失败） | 确认 `deploy/robots/g1` + `deploy/thirdparty` 相对结构 |
| 编译报 `dds_wrapper/...go2.h: No such file` | 缺新版 SDK2 头文件 | 第 5 章坑 ② |
| 链接报 `undefined reference to get_type_props` | 用了旧版 SDK2 静态库 | 第 5 章坑 ③ |
| 运行报 `libonnxruntime.so.1: file too short` | 符号链接被 scp 弄坏 | 第 5 章坑 ④ |
| 手柄按了没反应 | 没进硬件调试模式 / 服务没起 | 先按 L2+R2；`systemctl status g1_ctrl` |
| 没出现 `Initialized State_Mimic_*` | config.yaml 三处注册不全 | `grep -n "<NAME>" config/config.yaml` 应出现 3 次 |

---

## 11. 命令速查

| 目的 | 命令 |
|---|---|
| 登录 PC2 | `ssh unitree@192.168.123.164`（密码 123） |
| 服务器推源码给 PC2 | `scp -r robots/g1 include thirdparty unitree@192.168.123.164:~/` |
| PC2 编译 | `cd ~/deploy/robots/g1/build && cmake .. && make -j4` |
| 启动 g1_ctrl | `sudo systemctl start g1_ctrl` |
| 重启 g1_ctrl | `sudo systemctl restart g1_ctrl` |
| 开机自启开关 | `sudo systemctl enable/disable g1_ctrl` |
| 看状态 | `systemctl status g1_ctrl --no-pager` |
| 实时日志 | `journalctl -u g1_ctrl -f` |

---

*首次部署时间：2026-08-20。本路径已实测打通：PC2 上 g1_ctrl 编译成功、systemd 开机自启、4 支舞全部加载、DDS 连上运控。*
