#!/usr/bin/env bash
# 自动执行《11.md》第二章「连接机器人」的 2/3/4/6 步
# 用法：sudo ./fix_robot_network.sh

set -uo pipefail

NETPLAN_FILE="/etc/netplan/50-cloud-init.yaml"
INTERFACE="enp6s0"
ROBOT_IP="192.168.123.99"

if [[ $EUID -ne 0 ]]; then
  echo "请以 root 运行：sudo $0" >&2
  exit 1
fi

# ── 步骤 2：直接覆盖 netplan 配置 ──────────────────────────────
if [[ -f "$NETPLAN_FILE" ]]; then
  cp "$NETPLAN_FILE" "${NETPLAN_FILE}.bak.$(date +%Y%m%d%H%M%S)"
  echo "[2] 已备份原配置 → ${NETPLAN_FILE}.bak.*"
fi
cat > "$NETPLAN_FILE" <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    ${INTERFACE}:
      dhcp4: no
      addresses:
        - ${ROBOT_IP}/24
      optional: true
EOF
echo "[2] 已覆盖 ${NETPLAN_FILE}"

# ── 步骤 3：修正权限 ───────────────────────────────────────────
chmod 600 /etc/netplan/*.yaml
echo "[3] 已执行 chmod 600 /etc/netplan/*.yaml"

# ── 步骤 4：生成并测试 ─────────────────────────────────────────
netplan generate && echo "[4] netplan generate 完成"

# 非交互环境自动确认（送入一个换行即视为按 Enter）；若 netplan 版本支持可加 --timeout 30
if printf '\n' | netplan try; then
  echo "[4] netplan try 已确认，配置保留"
else
  echo "[!] netplan try 未确认/已回滚，将在步骤 6 重新应用" >&2
fi

# ── 步骤 6：永久应用 ───────────────────────────────────────────
netplan apply && echo "[6] netplan apply 完成"

echo "全部完成。请验证连通性：ping ${ROBOT_IP}"
