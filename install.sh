#!/usr/bin/env bash
# One-shot installer/preflight for DeepSeek-V4-Flash on 2× DGX Spark.
#
# Run it on the WORKER node (the one you'll launch from). It is idempotent:
# run it as many times as you want, it only fixes what's missing and prints
# the exact commands for the two steps that need sudo (netplan IPs).
#
# Usage:
#   HEAD_SSH=user@spark-head ./install.sh
#
# What it does:
#   1. checks the basics (nvidia-smi/GB10, docker, ssh, python3, curl)
#   2. installs uv + sparkrun if missing (user-level, no sudo)
#   3. checks the SSH mesh to the head node (key-based, both directions)
#   4. checks the 200G ConnectX-7 link (carrier + IPs) — prints netplan/ip
#      commands if the IPs are missing (sudo required, on each node)
#   5. applies the NCCL GID patch to sparkrun (patches/apply_sparkrun_patch.py)
#   6. checks docker on the head node
#
# When everything is green:  ./scripts/launch-deepseek.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HEAD_SSH="${HEAD_SSH:-user@spark-head}"
IP_LOCAL="${IP_LOCAL:-10.99.1.2}"
IP_HEAD="${IP_HEAD:-10.99.1.1}"
CX7_IF="${CX7_IF:-enp1s0f0np0}"

ok=0; warn=0
say(){  echo -e "  \033[32m✓\033[0m $*"; ok=$((ok+1)); }
bad(){  echo -e "  \033[31m✗\033[0m $*"; warn=$((warn+1)); }
info(){ echo -e "    \033[2m$*\033[0m"; }
hdr(){  echo -e "\n\033[1m$*\033[0m"; }

hdr "1/6 Basics"
command -v nvidia-smi >/dev/null && say "nvidia-smi" || bad "nvidia-smi missing (is this a DGX Spark?)"
if nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | grep -q "N/A"; then
  say "GB10 unified memory detected"
else
  info "note: no [N/A] VRAM — not a GB10? The method still applies to 2-node vLLM/ray."
fi
command -v docker >/dev/null && say "docker" || bad "docker missing (DGX OS ships it; else: https://docs.docker.com/engine/install/)"
docker ps >/dev/null 2>&1 && say "docker usable without password" || \
  { sudo -n docker ps >/dev/null 2>&1 && say "docker via passwordless sudo" || \
    bad "docker needs an interactive password — add your user to the docker group: sudo usermod -aG docker \$USER"; }
command -v python3 >/dev/null && say "python3" || bad "python3 missing"
command -v curl >/dev/null && say "curl" || bad "curl missing"

hdr "2/6 sparkrun"
if command -v sparkrun >/dev/null || [ -x "$HOME/.local/bin/sparkrun" ]; then
  say "sparkrun present ($($HOME/.local/bin/sparkrun --version 2>/dev/null || sparkrun --version 2>/dev/null || echo '?'))"
else
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "  installing uv (user-level)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 && say "uv installed" || bad "uv install failed"
  fi
  UV="$(command -v uv || echo $HOME/.local/bin/uv)"
  echo "  installing sparkrun (uv tool)…"
  "$UV" tool install sparkrun >/dev/null 2>&1 && say "sparkrun installed" || bad "sparkrun install failed (try: $UV tool install sparkrun)"
fi

hdr "3/6 SSH mesh to the head node ($HEAD_SSH)"
if [ "$HEAD_SSH" = "user@spark-head" ]; then
  bad "HEAD_SSH not set — run:  HEAD_SSH=youruser@head-hostname ./install.sh"
else
  if ssh -o BatchMode=yes -o ConnectTimeout=6 "$HEAD_SSH" true 2>/dev/null; then
    say "SSH key-based to head OK"
    ssh -o BatchMode=yes "$HEAD_SSH" "ssh -o BatchMode=yes -o ConnectTimeout=6 $(whoami)@$IP_LOCAL true" 2>/dev/null \
      && say "reverse SSH (head → worker over the 200G link) OK" \
      || info "reverse SSH not verified — sparkrun mostly needs worker→head; fine for a first try"
  else
    bad "no key-based SSH to $HEAD_SSH — run: ssh-copy-id $HEAD_SSH"
  fi
fi

hdr "4/6 200G ConnectX-7 link"
if ip -br link show "$CX7_IF" 2>/dev/null | grep -q UP; then
  say "$CX7_IF is UP"
else
  info "carrier check: the cabled pair is port f0↔f0 on both Sparks (NOT the gigabit LAN port)"
  bad "$CX7_IF not UP — is the QSFP-DD cable seated? (dmesg | grep -i mlx)"
fi
if ip -br addr show "$CX7_IF" 2>/dev/null | grep -q "$IP_LOCAL"; then
  say "$IP_LOCAL present on $CX7_IF"
else
  bad "$IP_LOCAL missing on $CX7_IF — persistent way (recommended):"
  info "sudo cp $HERE/netplan/90-cx7-200g.yaml /etc/netplan/  # edit: worker=$IP_LOCAL"
  info "sudo chmod 600 /etc/netplan/90-cx7-200g.yaml && sudo netplan apply"
  info "quick way: sudo ip addr add $IP_LOCAL/24 dev $CX7_IF && sudo ip link set $CX7_IF up"
fi
if [ "$HEAD_SSH" != "user@spark-head" ] && ssh -o BatchMode=yes "$HEAD_SSH" "ip -br addr show | grep -q $IP_HEAD" 2>/dev/null; then
  say "$IP_HEAD present on head"
  ping -c1 -W2 "$IP_HEAD" >/dev/null 2>&1 && say "ping over the 200G link OK" || bad "$IP_HEAD does not ping"
else
  bad "$IP_HEAD missing on the head node — same netplan there (head=$IP_HEAD)"
fi

hdr "5/6 sparkrun NCCL GID patch (trap #3)"
python3 "$HERE/patches/apply_sparkrun_patch.py" && say "patch OK" || bad "patch failed (see patches/README in the diff)"

hdr "6/6 head node runtime"
if [ "$HEAD_SSH" != "user@spark-head" ]; then
  ssh -o BatchMode=yes "$HEAD_SSH" "docker ps >/dev/null 2>&1 || sudo -n docker ps >/dev/null 2>&1" 2>/dev/null \
    && say "docker usable on head" || bad "docker not usable passwordless on head"
fi

echo
if [ "$warn" -eq 0 ]; then
  echo -e "\033[1;32mAll green ($ok checks). Launch with:\033[0m"
  echo "  HEAD_SSH=$HEAD_SSH ./scripts/launch-deepseek.sh"
  echo "First launch downloads ~149 GB per node, then several minutes of JIT/graph capture."
else
  echo -e "\033[1;33m$warn item(s) to fix above, then re-run ./install.sh\033[0m"
fi
