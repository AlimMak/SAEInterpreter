"""Read-time access to the frozen activation cache.

Batch composition is decided here, at read time, by an RNG seeded from
`data_seed` alone. Two consequences the experiment depends on:

  * same data_seed  -> identical batch sequence, regardless of `seed`  (Arm A)
  * different seeds -> different batch sequence                        (Arm B)

test_sampler.py asserts both rather than trusting them.

Why a shuffle buffer over sequential reads, rather than a full random
permutation over the memmap:

A full permutation issues 4096 scattered reads per batch. Each row is 1536
bytes -- smaller than a page -- so a batch touches ~4096 distinct pages, and
measured on the 8GB laptop that cost ~400ms per batch against ~6ms once the
pages were warm. For the 18GB full preset the pages can never all be warm, so
that cost would be permanent: roughly 3.3 hours of pure I/O per run, 33 hours
across the ten runs, with the GPU idle for most of it.

The fix is split across the two stages. capture.py already applied one fixed
permutation at write time, so *disk order is uncorrelated with corpus order*.
That means a sequential block of rows on disk is already a random sample of the
corpus, and the read-time buffer only has to break up the residual structure
within a block and make batch composition depend on data_seed. Sequential
reads, seed-dependent batches, no page-fault cliff.

The thing this must not become is a write-time shuffle *alone*: that would fix
batch order identically for every run and silently collapse Arm B into Arm A.
Both shuffles are required, and they do different jobs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActivationStore:
    """Memory-maps the activation shards and reads rows by global index."""

    def __init__(self, act_dir: Path):
        self.act_dir = Path(act_dir)
        manifest_path = self.act_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no activation cache at {self.act_dir}. Run capture.py for this preset first."
            )
        self.manifest = json.loads(manifest_path.read_text())

        n_shards = self.manifest["n_shards"]
        self.acts = [
            np.load(self.act_dir / f"acts_{i:05d}.npy", mmap_mode="r") for i in range(n_shards)
        ]
        self.toks = [
            np.load(self.act_dir / f"toks_{i:05d}.npy", mmap_mode="r") for i in range(n_shards)
        ]

        rows = self.manifest["shard_rows"]
        self.offsets = np.concatenate([[0], np.cumsum(rows)])
        self.total_rows = int(self.offsets[-1])
        self.d_model = self.manifest["d_model"]

        # Read once from the manifest, never recomputed. A per-run scalar would
        # be a fresh source of cross-run variation in an experiment whose whole
        # subject is cross-run variation.
        if "norm_scale" not in self.manifest:
            raise RuntimeError(
                f"{manifest_path} has no norm_scale -- it predates the normalisation "
                f"change. Re-run capture.py for this preset."
            )
        self.norm_scale = float(self.manifest["norm_scale"])

        for i, (a, t) in enumerate(zip(self.acts, self.toks)):
            if a.shape[0] != t.shape[0]:
                raise RuntimeError(
                    f"shard {i}: {a.shape[0]} activation rows vs {t.shape[0]} token rows"
                )

    def read_block(self, start: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Read `n` consecutive rows starting at global index `start`.

        Returns (rows_fp16, global_indices). Wraps at the end of the cache.
        Contiguous within each shard, so this is a sequential read -- the whole
        point of the write-time permutation.

        The indices are returned because test_sampler.py needs to measure batch
        overlap between arms by *identity*. Comparing row contents instead is
        not equivalent: 0.81% of rows in this cache are exact byte-duplicates
        of another row (repeated boilerplate in the corpus produces identical
        activations), which adds ~33 false matches to a 4096-row batch and
        would make an at-chance overlap look like 2x chance.

        Rows come back in the cache's native fp16. Widening to fp32 happens per
        batch instead, because the buffer is the largest allocation in the
        process: 262144 x 768 is 384MB as fp16 and 768MB as fp32, and on the
        8GB development machine the fp32 version drove the system 10.9GB into
        swap. The values are identical either way -- the cache is fp16 on disk.
        """
        out = np.empty((n, self.d_model), dtype=np.float16)
        idx = np.empty(n, dtype=np.int64)
        filled = 0
        pos = start % self.total_rows
        while filled < n:
            s = int(np.searchsorted(self.offsets, pos, side="right") - 1)
            local = pos - self.offsets[s]
            take = min(n - filled, self.acts[s].shape[0] - local)
            out[filled : filled + take] = self.acts[s][local : local + take]
            idx[filled : filled + take] = np.arange(pos, pos + take)
            filled += take
            pos = (pos + take) % self.total_rows
        return out, idx

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """Fetch arbitrary rows by global index. Used for eval, not training."""
        out = np.empty((idx.shape[0], self.d_model), dtype=np.float32)
        bounds = np.searchsorted(idx, self.offsets)
        for s in range(len(self.acts)):
            lo, hi = bounds[s], bounds[s + 1]
            if lo == hi:
                continue
            out[lo:hi] = self.acts[s][idx[lo:hi] - self.offsets[s]]
        return out

    def tokens_at(self, idx: np.ndarray) -> np.ndarray:
        """Token IDs for the same global rows -- used from Phase 4 onward."""
        out = np.empty(idx.shape[0], dtype=np.int32)
        bounds = np.searchsorted(idx, self.offsets)
        for s in range(len(self.toks)):
            lo, hi = bounds[s], bounds[s + 1]
            if lo == hi:
                continue
            out[lo:hi] = self.toks[s][idx[lo:hi] - self.offsets[s]]
        return out


def batch_stream(
    store: ActivationStore,
    batch_size: int,
    n_steps: int,
    data_seed: int,
    buffer_rows: int,
    normalize: bool = True,
    with_indices: bool = False,
):
    """Yield one batch of activations per training step.

    Half-buffer refill: the buffer is drained to 50%, then the consumed rows
    are overwritten from the next sequential block and the draw order is
    reshuffled. This keeps every batch a mixture of rows read at different
    times rather than a single contiguous block, and keeps reads sequential.

    `data_seed` drives both the shuffle inside the buffer and the starting
    offset into the cache, so two runs with different data_seeds see different
    batches from step 0 rather than converging on the same first epoch.

    The buffer is allocated once and mutated in place, and the shuffle is a
    permutation of an *index* array rather than of the data. Permuting the data
    (`buf = buf[p]`) allocates a second full buffer for the duration of the
    copy; at fp32 that peaked around 1.5GB and pushed the 8GB dev machine into
    swap, which cost more than the page faults this sampler exists to avoid.
    """
    rng = np.random.default_rng(data_seed)
    buffer_rows = min(buffer_rows, store.total_rows)
    half = buffer_rows // 2

    read_pos = int(rng.integers(0, store.total_rows))
    buf, bidx = store.read_block(read_pos, buffer_rows)  # fp16, allocated once
    read_pos += buffer_rows
    order = rng.permutation(buffer_rows)
    take = 0

    scale = np.float32(store.norm_scale if normalize else 1.0)

    for _ in range(n_steps):
        if take + batch_size > buffer_rows:
            # Overwrite exactly the rows already handed out. Their positions
            # are scattered (order is a permutation), which is fine -- they are
            # being replaced, and writing into an existing allocation avoids
            # the temporary copy.
            n_new = max(half, take)
            fresh, fidx = store.read_block(read_pos, n_new)
            read_pos += n_new
            slots = order[:n_new]
            buf[slots] = fresh
            bidx[slots] = fidx
            order = rng.permutation(buffer_rows)
            take = 0
            del fresh, fidx

        sel = order[take : take + batch_size]
        take += batch_size
        batch = buf[sel].astype(np.float32) * scale
        yield (batch, bidx[sel]) if with_indices else batch
