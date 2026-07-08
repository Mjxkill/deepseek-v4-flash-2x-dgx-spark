#!/usr/bin/env python3
"""Return HF blob pages to the kernel (posix_fadvise DONTNEED, no sudo needed).

GB10 unified-memory trap: after downloading/reading ~200 GB of weights, the
kernel page cache holds them, and vLLM's fail-fast startup check sees
"Free memory (108/121 GiB) < gpu_memory_utilization (0.93 = 113 GiB)" and
refuses to start — even though that cache is reclaimable. Dropping the blob
pages beforehand fixes it. Run on BOTH nodes before launching."""
import os, pathlib

hub = pathlib.Path.home() / ".cache/huggingface/hub"
freed = 0
for blob in hub.glob("models--*/blobs/*"):
    try:
        size = blob.stat().st_size
        fd = os.open(blob, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        freed += size
    except OSError:
        pass
print(f"page cache returned to kernel: ~{freed/1e9:.0f} GB of blobs processed")
