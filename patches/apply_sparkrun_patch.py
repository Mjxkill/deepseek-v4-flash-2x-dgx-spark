#!/usr/bin/env python3
"""Ré-applique le patch NCCL GID de ft-studio sur sparkrun (idempotent).

sparkrun est un tool uv : toute mise à jour / réinstallation ÉCRASE le patch dans
site-packages. Ce script le ré-applique. Lancé automatiquement par launch-deepseek.sh.

Le patch : filtrer de NCCL_IB_HCA les HCA RoCE sans GID RoCEv2 IPv4 à l'index 3
(2e port ConnectX-7 sans IP -> GID fe80 seulement -> ibv_modify_qp "local GID ::"
-> crash NCCL cross-node).
"""
import glob
import re
import sys

MARKER = "PATCH (ft-studio)"

PATCH_BLOCK = '''    if ib_info.get("DETECTED_HCA_LIST"):
        # PATCH (ft-studio): drop RoCE HCAs that have no RoCEv2 IPv4 GID at
        # NCCL_IB_GID_INDEX (3). Such a HCA (netdev without an IPv4, e.g. a 2nd
        # ConnectX-7 port left unaddressed) exposes only a link-local IPv6 GID,
        # so ibv_modify_qp fails ("local GID ::") and NCCL crashes cross-node.
        def _hca_has_rocev2_ipv4_gid(hca: str, gid_index: int = 3) -> bool:
            try:
                with open(
                    "/sys/class/infiniband/%s/ports/1/gids/%d" % (hca, gid_index)
                ) as fh:
                    gid = fh.read().strip().lower()
            except OSError:
                return True  # can't introspect -> don't filter it out
            # RoCEv2 over IPv4 => IPv4-mapped IPv6 GID: 0000:...:ffff:AABB:CCDD
            # with a non-zero embedded IPv4. Anything else (fe80 link-local,
            # all-zero) has no usable v2 GID at this index.
            if "ffff:" not in gid:
                return False
            tail = gid.split("ffff:", 1)[1]
            return tail not in ("0000:0000", "", "0000:0000\\n")

        _hcas = [h.strip() for h in ib_info["DETECTED_HCA_LIST"].split(",") if h.strip()]
        _good = [h for h in _hcas if _hca_has_rocev2_ipv4_gid(h)]
        # Only apply the filter if it leaves at least one HCA; otherwise fall
        # back to the detected list so we never disable IB entirely by mistake.
        env["NCCL_IB_HCA"] = ",".join(_good) if _good else ib_info["DETECTED_HCA_LIST"]
'''

# La ligne d'origine (non patchée) que l'on remplace.
ORIGINAL_RE = re.compile(
    r'''[ \t]*if ib_info\.get\("DETECTED_HCA_LIST"\):\n'''
    r'''[ \t]*env\["NCCL_IB_HCA"\] = ib_info\["DETECTED_HCA_LIST"\]\n'''
)


def find_target() -> str | None:
    hits = glob.glob(
        "/home/*/.local/share/uv/tools/sparkrun/lib/python3*/site-packages/"
        "sparkrun/orchestration/infiniband.py"
    )
    return hits[0] if hits else None


def main() -> int:
    path = find_target()
    if not path:
        print("sparkrun introuvable (uv tool non installé ?)")
        return 1
    src = open(path).read()
    if MARKER in src:
        print(f"déjà patché : {path}")
        return 0
    new, n = ORIGINAL_RE.subn(PATCH_BLOCK, src, count=1)
    if n != 1:
        print(f"ÉCHEC : bloc d'origine introuvable dans {path} — "
              "sparkrun a changé, patch à réviser à la main.")
        return 2
    open(path, "w").write(new)
    print(f"patch appliqué : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
