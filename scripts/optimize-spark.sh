#!/usr/bin/env bash
# Turn a DGX Spark into a PURE headless MODEL FARM. Idempotent.
#
# Usage:  sudo ./optimize-spark.sh          # apply
#         ./optimize-spark.sh --dry-run     # preview, no changes
#
# ── WHY THIS SCRIPT EXISTS ──────────────────────────────────────────────────
# DGX OS ships `earlyoom` configured with
#   --prefer '(vllm|VLLM|sglang|llama-server|...|ray|python3|python)'
# i.e. below 2% available memory it SIGTERMs — PREFERRING inference engines by
# name. Caught red-handed killing a Ray TP worker mid CUDA-graph capture, with
# zero error in the vLLM logs. Any serving above ~85% memory utilization is
# doomed while it runs. The kernel's own OOM-killer remains as a last resort,
# so disabling it is safe.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"
[ -z "$DRY" ] && [ "$(id -u)" != "0" ] && { echo "sudo required (or --dry-run)"; exit 1; }

avail(){ awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo; }
echo "MemAvailable before: $(avail)"

off(){ # stop + disable + mask (mask = nothing can pull it back in via deps)
  for u in "$@"; do
    systemctl list-unit-files "$u" >/dev/null 2>&1 || continue
    $DRY systemctl disable --now "$u" 2>/dev/null
    $DRY systemctl mask "$u" 2>/dev/null
    echo "  off: $u"
  done
}

echo "-- 1. earlyoom (the model killer -- see header)"
off earlyoom.service

echo "-- 2. Graphical stack (headless: everything goes through SSH)"
$DRY systemctl set-default multi-user.target
off gdm.service gdm3.service gnome-remote-desktop.service accounts-daemon.service \
    switcheroo-control.service colord.service rtkit-daemon.service

echo "-- 3. Desktop peripherals with no purpose on a model farm"
off bluetooth.service cups.service cups-browsed.service cups.socket cups.path \
    ModemManager.service avahi-daemon.service avahi-daemon.socket \
    wpa_supplicant.service upower.service udisks2.service \
    fwupd.service fwupd-refresh.timer whoopsie.service apport.service kerneloops.service

echo "-- 4. What we KEEP (and why):"
cat <<'KEEP'
  ssh docker containerd     : access + model containers
  NetworkManager systemd-*  : network/boot/journal
  nvidia-persistenced       : GPU ready without wake latency
  rdma-ndd                  : RDMA device naming -> REQUIRED for the RoCE 200G link
  rasdaemon smartmontools   : hardware health (ECC RAM, NVMe)
  multipathd cron rsyslog polkit dbus : base plumbing
  (tailscaled if you use it)
KEEP

echo "-- 5. Optional (uncomment if you accept the trade-off):"
cat <<'OPT'
  # NVIDIA web dashboard + telemetry (useless if you admin over SSH):
  # systemctl disable --now dgx-dashboard.service dgx-dashboard-admin.service nvidia-dgx-telemetry.service lldpd.service
  # snapd + DESKTOP snaps (yes, DGX OS ships firefox/thunderbird/snap-store on a compute node):
  # snap remove firefox thunderbird snap-store firmware-updater snapd-desktop-integration
  # systemctl disable --now snapd.service snapd.socket snapd.seeded.service
OPT

echo "-- 6. USER units (audio/desktop -- no sudo needed, applied for the invoking user)"
USER_UNITS="pipewire.service pipewire-pulse.service wireplumber.service filter-chain.service \
  gvfs-daemon.service xdg-document-portal.service xdg-permission-store.service \
  snap.snapd-desktop-integration.snapd-desktop-integration.service"
for u in $USER_UNITS; do
  $DRY sudo -u "${SUDO_USER:-$USER}" XDG_RUNTIME_DIR="/run/user/$(id -u "${SUDO_USER:-$USER}")" \
    systemctl --user disable --now "$u" 2>/dev/null && echo "  off(user): $u"
  $DRY sudo -u "${SUDO_USER:-$USER}" XDG_RUNTIME_DIR="/run/user/$(id -u "${SUDO_USER:-$USER}")" \
    systemctl --user mask "$u" 2>/dev/null
done

echo ""
echo "MemAvailable after: $(avail)   (most of the GUI gain shows at next boot)"
echo "Done. A reboot is recommended to start clean in multi-user.target."
