#!/usr/bin/env bash
# Quick decode-throughput + RDMA-traffic benchmark for the 2-node cluster.
#
# IMPORTANT: during inference the *netdev* counters (/sys/class/net/*/statistics)
# stay at ZERO — RoCE is kernel-bypass. The real traffic is in the InfiniBand
# HW counters, in 4-byte units:
#   /sys/class/infiniband/<hca>/ports/1/counters/port_{xmit,rcv}_data
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://spark-head:8000}"
HCA="${HCA:-rocep1s0f0}"        # the CX-7 HCA that carries the link (the one WITH an IPv4)
NTOK="${NTOK:-120}"
MODEL=$(curl -sf "$ENDPOINT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")

cnt(){ echo $(( $(cat /sys/class/infiniband/$HCA/ports/1/counters/port_${1}_data) * 4 )); }

TX0=$(cnt xmit); RX0=$(cnt rcv); T0=$(date +%s.%N)
OUT=$(curl -sf "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Écris un poème de $NTOK mots sur les réseaux 200G.\"}],
  \"max_tokens\": $NTOK, \"temperature\": 0.7}")
T1=$(date +%s.%N); TX1=$(cnt xmit); RX1=$(cnt rcv)

TOK=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])")
python3 - "$T0" "$T1" "$TOK" "$((TX1-TX0))" "$((RX1-RX0))" <<'EOF'
import sys
t0, t1, tok, tx, rx = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dt = t1 - t0
print(f"tokens: {tok}  |  wall: {dt:.1f}s  |  ~{tok/dt:.1f} tok/s (incl. TTFT)")
print(f"RDMA on {dt:.1f}s : TX {tx/1e6:.0f} MB  RX {rx/1e6:.0f} MB  ({(tx+rx)/tok/1e6:.1f} MB/token)")
EOF
