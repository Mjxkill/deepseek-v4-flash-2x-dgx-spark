#!/usr/bin/env bash
# Stop the DeepSeek cluster on both Sparks and clean Ray leftovers.
set -uo pipefail
SPARKRUN="${SPARKRUN:-$HOME/.local/bin/sparkrun}"
HEAD_SSH="${HEAD_SSH:-user@spark-head}"

"$SPARKRUN" stop --all || true
pkill -9 -f "gcs_server|raylet|ray start" 2>/dev/null || true
sudo docker ps -aq --filter name=sparkrun | xargs -r sudo docker rm -f 2>/dev/null || true
ssh -o BatchMode=yes "$HEAD_SSH" '~/.local/bin/sparkrun stop --all >/dev/null 2>&1; \
  pkill -9 -f "gcs_server|raylet|ray start" 2>/dev/null; \
  sudo docker ps -aq --filter name=sparkrun | xargs -r sudo docker rm -f' 2>/dev/null || true
echo "[deepseek] cluster stopped on both nodes"
