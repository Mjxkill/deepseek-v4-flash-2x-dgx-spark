# DeepSeek-V4-Flash on 2× NVIDIA DGX Spark (TP=2 over 200G RoCE)

Everything needed to serve **deepseek-ai/DeepSeek-V4-Flash** (fp8, ~149 GB of weights,
500k context, MTP speculative decoding, DeepSeek Sparse Attention) across **two DGX
Spark (GB10)** connected by a single QSFP-DD cable — including the **four traps** that
each crash the launch, and the fixes for all of them.

Status: **working**. OpenAI-compatible endpoint on the head node (`:8000`),
~**16 tok/s** single-stream decode, RDMA traffic confirmed on the 200G link.

```
GET /v1/models → deepseek-ai/DeepSeek-V4-Flash  (max_model_len 500000)
```

## Hardware

| Piece | Detail |
|---|---|
| 2× DGX Spark | GB10, 128 GB unified memory each (model needs ~95 GB/GPU → TP=2 mandatory) |
| 1× QSFP-DD DAC cable | e.g. NVIDIA/Mellanox `MCP1660-W001E30` (1 m), directly between the two Sparks |
| Ports | **`enp1s0f0np0` ↔ `enp1s0f0np0`** (the *f0* ConnectX-7 port on both). ⚠ Each Spark has **two** CX-7 ports (`enp1s0f0np0` + `enP2p1s0f0np0`) and also a gigabit LAN port — don't confuse them. `ethtool` shows 200 Gb/s carrier on the cabled pair. |

## Software

- [`sparkrun`](https://github.com/eugr/sparkrun) (tested with v0.2.40) with an SSH mesh
  between both nodes (key-based, both directions).
- Recipe: **forked** `@eugr/deepseek-v4-flash` → [`recipes/deepseek-v4-flash.yaml`](recipes/deepseek-v4-flash.yaml)
  (the upstream recipe is broken, see trap #1).
- Image: **`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026070501`** (vLLM 0.23.1 +
  DeepGEMM). The default `vllm-node:latest` build does **not** work (trap #2).

## Quick start

```bash
git clone https://github.com/Mjxkill/deepseek-v4-flash-2x-dgx-spark
cd deepseek-v4-flash-2x-dgx-spark

# 1. Installer / preflight (idempotent — installs uv+sparkrun if missing,
#    checks docker, SSH mesh, the 200G link, and applies the NCCL patch;
#    prints the exact sudo commands for the netplan IPs, the only manual step)
HEAD_SSH=user@spark-head ./install.sh

# 2. When all checks are green:
HEAD_SSH=user@spark-head ENDPOINT=http://spark-head:8000 ./scripts/launch-deepseek.sh
```

The netplan IPs (the one sudo step, on **both** Sparks): head = `10.99.1.1/24`,
worker = `10.99.1.2/24` on the cabled `enp1s0f0np0` — see
[`netplan/90-cx7-200g.yaml`](netplan/90-cx7-200g.yaml).

The launcher is idempotent: it checks the link, re-applies the sparkrun patch
(trap #3), cleans stale Ray state (trap #4), then runs
`sparkrun run recipes/deepseek-v4-flash.yaml --hosts 10.99.1.1,10.99.1.2 --rootful --image …nightly:2026070501`.

First launch downloads ~149 GB on each node. Loading takes **several minutes**
(TileLang JIT of the `mhc_*` kernels + CUDA-graph capture); the
`shm_broadcast: No available shared memory broadcast block in 60s` warnings during
that phase are **benign**.

## The five traps 🪤

### 1. The upstream recipe generates invalid JSON (`--speculative-config`)

sparkrun's template engine (`arg_substitute`, regex `\{(.*?)\}`) does **not** collapse
`{{` → `{` the way `str.format` does. The upstream recipe writes:

```yaml
--speculative-config '{{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}}'
```

…which reaches vLLM with the double braces intact → `json.loads` fails on node
rank 0 at step [6/6]. Same bug on `--reasoning-config`.

**Fix** (in [`recipes/deepseek-v4-flash.yaml`](recipes/deepseek-v4-flash.yaml)):
single braces and hardcoded values:

```yaml
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
--reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
```

(a single-brace block is not a known template variable, so `arg_substitute` leaves it
untouched, while `{port}`/`{tensor_parallel}` still substitute normally).

### 2. `RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM`

DeepSeek-V4-Flash uses Sparse Attention (Lightning Indexer) whose CUDA op needs
**DeepGEMM** compiled into the image. The default `vllm-node:latest` built by the eugr
builder (vLLM 0.22.x) doesn't have it: everything mounts (TP=2, RoCE, weights start
loading) then it crashes when building the attention layers.

**Fix**: override the image with a nightly that bundles it:

```
--image ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026070501    # vLLM 0.23.1
```

Verified in the logs: `Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM`.

### 3. NCCL `unhandled system error` at the first all-reduce

With `NCCL_DEBUG=INFO` (already set in the recipe) the real error is:

```
ibv_modify_qp failed with 61, dev roceP2p1s0f0, local GID index 3, local GID ::
```

Each Spark has **two** CX-7 HCAs. The un-cabled/un-addressed one (`roceP2p1s0f0`,
netdev `enP2p1s0f0np0`) has **no RoCEv2 IPv4 GID** at index 3 (only a link-local
`fe80::`), but sparkrun force-sets `NCCL_IB_HCA` to *the full detected list*. NCCL then
dies bringing up a queue pair on the bad HCA.

**Fix** ([`patches/`](patches/)): filter `NCCL_IB_HCA` down to HCAs that actually have
an IPv4-mapped GID at index 3 (`::ffff:a.b.c.d`). Apply with:

```bash
python3 patches/apply_sparkrun_patch.py     # idempotent; the launcher runs it for you
```

⚠ sparkrun is a pip/uv package: **every upgrade wipes the patch** — hence the
idempotent re-applicator. Durable alternatives: give the second port an IP too
(second cable → 2× 200G), or `sudo ip link set enP2p1s0f0np0 down`.

### 4. Stale Ray state between attempts

A crashed run leaves host-level Ray processes and containers behind; the next launch
then fails with `ActorHandleNotFoundError … handle is not valid across Ray sessions`
or hangs at the NCCL rendezvous. Clean both nodes between attempts:

```bash
sparkrun stop --all
pkill -9 -f "gcs_server|raylet|ray start"
sudo docker ps -aq --filter name=sparkrun | xargs -r sudo docker rm -f
```

(`scripts/launch-deepseek.sh` does this automatically; also avoid `--no-rm`.)

### 5. `Free memory on device cuda:0 (…) is less than desired GPU memory utilization`

Unified-memory trap. After downloading (or reading) ~200 GB of weights, the Linux
page cache holds them; vLLM's fail-fast startup check measures **free** memory and
refuses to start even though the cache is reclaimable. Bonus confusion: after the
engine retries internally, the log ends with
`ActorHandleNotFoundError … not valid across Ray sessions` — which looks like trap
#4 but is only a **symptom**; always scroll up to the first `ValueError`.

**Fix**: [`scripts/drop_hf_cache.py`](scripts/drop_hf_cache.py) returns the blob
pages to the kernel (`posix_fadvise DONTNEED`, no sudo) on both nodes — the
launcher runs it automatically. Also budget ~9 GiB/node for the OS + container
stack when picking `gpu_memory_utilization`: on a 121.6 GiB GB10, 0.90 is the
realistic ceiling (0.93 fails by ~1 GiB even with a clean cache).

## Measuring — don't trust the netdev counters

During inference `/sys/class/net/*/statistics` stays at **zero**: RoCE is
kernel-bypass. The real traffic is in the InfiniBand HW counters (4-byte units):

```
/sys/class/infiniband/rocep1s0f0/ports/1/counters/port_xmit_data   # ×4 = bytes
/sys/class/infiniband/rocep1s0f0/ports/1/counters/port_rcv_data
```

[`scripts/bench-rdma.sh`](scripts/bench-rdma.sh) wraps a completion request with these
counters. Measured here: **~315 MB TX / 316 MB RX for 120 generated tokens**
(~5 MB/token of all-reduce traffic — this is why the LAN won't do).

## Performance notes

- **~16 tok/s** single-stream decode. Cross-node TP=2 is **latency-bound**
  (~1 ms per all-reduce on this link), not bandwidth-bound — don't expect the
  single-node MoE numbers.
- MTP speculative decoding (`num_speculative_tokens: 2`) is active and included in
  that figure.
- Memory: ~104/121 GB used per node → leave the Sparks alone while it serves.
- `max_model_len 500000`, `kv-cache fp8`, `max_num_seqs 4` (see recipe defaults).

## Repo layout

| Path | What |
|---|---|
| `install.sh` | Idempotent installer/preflight (uv+sparkrun, SSH mesh, 200G link, patch) |
| `recipes/deepseek-v4-flash.yaml` | Fixed recipe (trap #1) |
| `patches/sparkrun-nccl-gid-filter.patch` | The NCCL GID filter as a unified diff (trap #3) |
| `patches/apply_sparkrun_patch.py` | Idempotent re-applicator (survives sparkrun upgrades) |
| `scripts/launch-deepseek.sh` | One-shot idempotent launcher (checks link, patch, cleanup, run) |
| `scripts/stop-deepseek.sh` | Clean stop on both nodes |
| `scripts/bench-rdma.sh` | tok/s + real RDMA traffic measurement |
| `scripts/drop_hf_cache.py` | Return HF blob page cache to the kernel (trap #5, no sudo) |
| `netplan/90-cx7-200g.yaml` | Persistent IPs for the 200G link (both nodes) |
| `docs/TROUBLESHOOTING.md` | Error signature → fix, in one table |

## Credits

- [eugr/sparkrun](https://github.com/eugr/sparkrun) and the
  [spark-arena dgx-vllm](https://github.com/spark-arena/dgx-vllm) nightlies — this repo
  is a field report + fixes on top of their work.
- Debugged and written with Claude (Anthropic) driving two Sparks over SSH.
