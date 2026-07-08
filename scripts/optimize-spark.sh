#!/usr/bin/env bash
# Turn a DGX Spark into a PURE headless MODEL FARM.
# Basé sur l'inventaire réel gx10-1/gx10-2 du 2026-07-08. Idempotent.
#
# Usage :  sudo ./optimize-spark.sh          # applique
#          ./optimize-spark.sh --dry-run     # montre sans rien faire
#
# ── LA DÉCOUVERTE QUI JUSTIFIE CE SCRIPT ────────────────────────────────────
# DGX OS livre `earlyoom` avec --prefer '(vllm|VLLM|sglang|...|ray|python)' :
# il tue EN PRIORITÉ les moteurs d'inférence dès que la mémoire dispo < 2 %.
# Flagrant délit (gx10-2, 2026-07-08 19:16:42) : SIGTERM à "ray::RayWorkerP"
# en pleine capture CUDA graphs d'Ornith 397B. Tout serving à >85 % d'util
# est condamné tant qu'il tourne. Le kernel garde son propre OOM-killer en
# dernier recours -> on peut le couper sans risque de figer la machine.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo [dry-run]"
[ -z "$DRY" ] && [ "$(id -u)" != "0" ] && { echo "sudo requis (ou --dry-run)"; exit 1; }

avail(){ awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo; }
echo "MemAvailable avant : $(avail)"

off(){ # stop + disable + mask (mask = rien ne peut le relancer par dépendance)
  for u in "$@"; do
    systemctl list-unit-files "$u" >/dev/null 2>&1 || continue
    $DRY systemctl disable --now "$u" 2>/dev/null
    $DRY systemctl mask "$u" 2>/dev/null
    echo "  off: $u"
  done
}

echo "── 1. earlyoom (le tueur de modèles — voir en-tête)"
off earlyoom.service

echo "── 2. Pile graphique (headless : tout passe par SSH)"
$DRY systemctl set-default multi-user.target
off gdm.service gdm3.service gnome-remote-desktop.service accounts-daemon.service \
    switcheroo-control.service colord.service rtkit-daemon.service

echo "── 3. Périphériques / bureau sans objet sur une ferme à modèles"
off bluetooth.service cups.service cups-browsed.service cups.socket cups.path \
    ModemManager.service avahi-daemon.service avahi-daemon.socket \
    wpa_supplicant.service upower.service udisks2.service \
    fwupd.service fwupd-refresh.timer whoopsie.service apport.service kerneloops.service

echo "── 4. Ce qu'on GARDE (et pourquoi) :"
cat <<'KEEP'
  ssh docker containerd     : accès + conteneurs des modèles
  NetworkManager systemd-*  : réseau/boot/journal
  tailscaled                : mesh d'admin
  nvidia-persistenced       : GPU prêt sans latence de réveil
  rdma-ndd                  : nommage RDMA -> indispensable au lien RoCE 200G
  rasdaemon smartmontools   : santé matérielle (RAM ECC, NVMe)
  multipathd cron rsyslog polkit dbus : plomberie de base
KEEP

echo "── 5. Optionnel (décommenter si assumé) :"
cat <<'OPT'
  # NVIDIA dashboard web + télémétrie (inutiles si admin par SSH) :
  # systemctl disable --now dgx-dashboard.service dgx-dashboard-admin.service nvidia-dgx-telemetry.service lldpd.service
  # snapd + snaps de BUREAU (firefox, thunderbird, snap-store sur un noeud de calcul !) :
  # snap remove firefox thunderbird snap-store firmware-updater snapd-desktop-integration
  # systemctl disable --now snapd.service snapd.socket snapd.seeded.service
OPT

echo "── 6. Services UTILISATEUR (audio/bureau — aucun sudo requis, relancé pour l'user courant)"
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
echo "MemAvailable après : $(avail)   (le gros du gain GUI apparaît au prochain boot)"
echo "Terminé. Un reboot est recommandé pour partir propre (multi-user.target)."
