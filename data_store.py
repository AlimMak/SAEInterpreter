"""Read-time access to the frozen activation cache.

This module is where `data_seed` earns its keep. capture.py deliberately wrote
the shards in corpus order; the permutation that decides batch composition is
generated here, at read time, from `data_seed` alone. Two consequences the
experiment depends on:

  * same data_seed  -> identical batch sequence, regardless of `seed`  (Arm A)
  * different seeds -> different batch sequence                        (Arm B)

test_sampler.py asserts both rather than trusting them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActivationStore:
    """Memory-maps the activation shards and gathers arbitrary rows.

    Memory-mapped rather than loaded: the full preset is 18GB against 8GB of
    VRAM and a laptop with finite RAM. The OS page cache handles residency,
    and because we only ever touch one batch's worth of rows at a time, the
    working set stays small.
    """

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
        # offsets[i] is the global index of shard i's first row; used to turn a
        # global row index into (shard, local index).
        self.offsets = np.concatenate([[0], np.cumsum(rows)])
        self.total_rows = int(self.offsets[-1])
        self.d_model = self.manifest["d_model"]

        for i, (a, t) in enumerate(zip(self.acts, self.toks)):
            if a.shape[0] != t.shape[0]:
                raise RuntimeError(
                    f"shard {i}: {a.shape[0]} activation rows vs {t.shape[0]} token rows"
                )

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """Fetch rows by global index. `idx` is expected pre-sorted."""
        if len(self.acts) == 1:
            return np.asarray(self.acts[0][idx], dtype=np.float32)

        out = np.empty((idx.shape[0], self.d_model), dtype=np.float32)
        # Split the sorted index array at shard boundaries; each slice is then
        # a contiguous, ascending read within one memmap.
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


def batch_index_stream(
    total_rows: int, batch_size: int, n_steps: int, data_seed: int
) -> "list[np.ndarray]":
    """Yield one array of row indices per training step.

    A **full permutation** over every row, reshuffled each time the pool is
    exhausted -- not a block shuffle over shards or a shuffle within a buffer.
    Block shuffling would leave rows from the same document adjacent in a
    batch, so a batch would be a sample of a few documents rather than of the
    corpus, and the gradient would be correlated in a way that varies with the
    block layout rather than with the seed.

    Indices within a batch are **sorted** before being returned. Sorting does
    not change which rows are in the batch -- the batch composition is fixed by
    the permutation, which is fixed by data_seed -- it only changes the order
    they are read from the memmap, turning a scattered random read into an
    ascending one. The SAE has no notion of within-batch order, so this is free.
    """
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(total_rows)
    pos = 0
    for _ in range(n_steps):
        if pos + batch_size > total_rows:
            perm = rng.permutation(total_rows)  # next epoch, same RNG stream
            pos = 0
        idx = perm[pos : pos + batch_size]
        pos += batch_size
        yield np.sort(idx)
