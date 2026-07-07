#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash TP=2 on 2× NVIDIA DGX Spark over the 200G ConnectX-7 link.
# Idempotent: checks the 200G link, re-applies the sparkrun NCCL patch, cleans up
# stale Ray/containers, then launches the fixed recipe with the DeepGEMM nightly image.
#
# Run this from the WORKER node (the head node is reached over SSH).
#
# Usage:  ./launch-deepseek.sh              # no-op if the endpoint already serves
#         ./launch-deepseek.sh --restart    # stop then relaunch
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ─── Adjust these to your setup ──────────────────────────────────────────────
SPARKRUN="${SPARKRUN:-$HOME/.local/bin/sparkrun}"
RECIPE="${RECIPE:-$HERE/../recipes/deepseek-v4-flash.yaml}"
IMAGE="${IMAGE:-ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026070501}"  # vLLM 0.23.1 + DeepGEMM
HEAD_SSH="${HEAD_SSH:-user@spark-head}"   # SSH to the OTHER Spark (head, serves :8000)
IP_LOCAL="${IP_LOCAL:-10.99.1.2}"         # this node on the 200G link
IP_HEAD="${IP_HEAD:-10.99.1.1}"           # head node on the 200G link
ENDPOINT="${ENDPOINT:-http://spark-head:8000}"   # head's LAN address for the API
CX7_IF="${CX7_IF:-enp1s0f0np0}"           # the cabled ConnectX-7 port (f0 on both Sparks)
# ─────────────────────────────────────────────────────────────────────────────

say(){ echo "[deepseek] $*"; }

# 0. Already serving?
if curl -sf -m4 "$ENDPOINT/v1/models" >/dev/null 2>&1; then
  if [ "${1:-}" != "--restart" ]; then
    say "already online ($ENDPOINT) — nothing to do (use --restart to relaunch)"
    exit 0
  fi
  say "--restart: stopping current cluster…"
fi

# 1. 200G link: both IPs must be up (netplan or manual)
ip -br addr show | grep -q "$IP_LOCAL" || {
  say "ERROR: $IP_LOCAL missing on this node. Apply the netplan (see netplan/) or:"
  say "  sudo ip addr add $IP_LOCAL/24 dev $CX7_IF && sudo ip link set $CX7_IF up"
  exit 1
}
ssh -o BatchMode=yes -o ConnectTimeout=6 "$HEAD_SSH" "ip -br addr show | grep -q $IP_HEAD" || {
  say "ERROR: $IP_HEAD missing on the head node. There:"
  say "  sudo ip addr add $IP_HEAD/24 dev $CX7_IF && sudo ip link set $CX7_IF up"
  exit 1
}
ping -c1 -W2 "$IP_HEAD" >/dev/null || { say "ERROR: $IP_HEAD does not ping (cable?)"; exit 1; }
say "200G link OK ($IP_LOCAL <-> $IP_HEAD)"

# 2. sparkrun NCCL patch (wiped by every sparkrun upgrade — see patches/)
python3 "$HERE/../patches/apply_sparkrun_patch.py"

# 3. Clean up leftovers (host-level Ray + sparkrun containers) on BOTH nodes.
#    Stale Ray pollutes the NCCL rendezvous: "ActorHandleNotFoundError … not
#    valid across Ray sessions".
say "cleaning stale Ray/containers…"
"$SPARKRUN" stop --all >/dev/null 2>&1 || true
pkill -9 -f "gcs_server|raylet|ray start" 2>/dev/null || true
ssh -o BatchMode=yes "$HEAD_SSH" 'pkill -9 -f "gcs_server|raylet|ray start" 2>/dev/null; \
  sudo docker ps -aq --filter name=sparkrun | xargs -r sudo docker rm -f' 2>/dev/null || true
sudo docker ps -aq --filter name=sparkrun | xargs -r sudo docker rm -f 2>/dev/null || true

# 4. Launch (head = IP_HEAD, worker = IP_LOCAL; order of --hosts matters)
say "launching sparkrun (recipe $RECIPE, DeepGEMM nightly image)…"
exec "$SPARKRUN" run "$RECIPE" \
  --hosts "$IP_HEAD,$IP_LOCAL" \
  --image "$IMAGE" \
  --rootful
