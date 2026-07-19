# Troubleshooting — error signature → fix

| You see | It means | Fix |
|---|---|---|
| `json.loads` / `JSONDecodeError` on `--speculative-config` or `--reasoning-config` at launch step [6/6] | Upstream recipe's `{{ }}` braces reach vLLM verbatim (sparkrun's `arg_substitute` doesn't collapse them) | Use `recipes/deepseek-v4-flash.yaml` (single braces, hardcoded JSON) |
| `RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed` | Image lacks DeepGEMM (default `vllm-node:latest`, vLLM 0.22.x) | `--image ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026070501` or newer |
| NCCL `unhandled system error` at first all-reduce; `NCCL_DEBUG=INFO` shows `ibv_modify_qp failed with 61 … local GID ::` on the HCA **without** an IP | The unaddressed 2nd CX-7 port has no RoCEv2 IPv4 GID at index 3, but sparkrun put it in `NCCL_IB_HCA` | `python3 patches/apply_sparkrun_patch.py` (or give that port an IP / `ip link set … down`) |
| `ActorHandleNotFoundError … not valid across Ray sessions`, or rendezvous hangs | Stale Ray processes / containers from a previous crashed attempt | `sparkrun stop --all` + `pkill -9 -f "gcs_server\|raylet\|ray start"` + `docker rm -f` sparkrun containers, on **both** nodes |
| `ValueError: Free memory on device cuda:0 (X/121.63 GiB) … less than desired GPU memory utilization` | Page cache full of model blobs (unified memory) and/or util too high — OS+container need ~9 GiB | `scripts/drop_hf_cache.py` on both nodes (launcher does it), and keep `gpu_memory_utilization` ≤ 0.90 |
| `ActorHandleNotFoundError … not valid across Ray sessions` at the END of a failed launch | Often a SYMPTOM of the engine retrying after an earlier crash (scroll up for the first `ValueError`) — only a real trap-#4 if no earlier error and old containers exist | Find the root error first; then clean both nodes as per trap #4 |
| `RayWorkerProc rank=[N] died unexpectedly` with NO error in its own log, near/after graph capture | **earlyoom** (DGX OS default) killed it — check `journalctl -u earlyoom` | `sudo ./scripts/optimize-spark.sh` (disables earlyoom + desktop stack) |
| `AssertionError: In Mamba cache align mode, block_size (4176) must be <= max_num_batched_tokens` | Hybrid-attention model (Mamba layers): 4176 is a hard floor, not a tuning knob | Set `max_num_batched_tokens: 4176` (or higher) |
| `shm_broadcast: No available shared memory broadcast block in 60s` (repeated) during load | TileLang JIT + CUDA-graph capture takes minutes; workers idle-wait | **Benign** — wait |
| Netdev RX/TX counters stay at 0 during inference | RoCE is kernel-bypass; traffic doesn't hit the netdev stats | Read `/sys/class/infiniband/<hca>/ports/1/counters/port_{xmit,rcv}_data` (×4 = bytes) |
| Decode feels "slow" (~16 tok/s) vs single-node MoE benchmarks | Cross-node TP=2 is latency-bound (~1 ms/all-reduce) | Expected. MTP (`num_speculative_tokens: 2`) already helps; batch more requests for throughput |
| Second launch re-downloads nothing but still slow to start | Normal: JIT + graph capture re-run | Wait; keep the container image cached |
| `sparkrun` can't find the other node | No auto-discovery exists | Always pass `--hosts <head_ip>,<worker_ip>` (head first) |
| Patch "block d'origine introuvable" from `apply_sparkrun_patch.py` | sparkrun upgrade changed `infiniband.py` | Re-derive the patch from `patches/sparkrun-nccl-gid-filter.patch` against the new source |
| Throughput ~halved for no visible reason; both nodes show high GPU utilization | One node is stuck at a low SM clock (e.g. 721 MHz vs 2489) — TP=2 runs at the slowest node's speed; no throttle reason is flagged | Compare `nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader` on **both** nodes; reboot the slow one (trap #7) |
