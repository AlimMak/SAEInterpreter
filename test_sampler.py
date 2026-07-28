"""Assertions the two-arm design rests on.

Arm A claims to hold batch order fixed while varying initialisation. That claim
is only true if the batch stream depends on `data_seed` and on nothing else. If
torch's RNG or the weight-init seed leaked into it, Arm A would secretly be
Arm B, the two arms would agree, and the headline result -- "the gap between the
arms" -- would be an artifact of a broken sampler.

The sampler changed in Phase 3: a fixed write-time permutation plus a
data_seed-driven read-time shuffle buffer, replacing a full random permutation
over the memmap. The risk introduced by that change is that the buffer is too
small or too structured for batches to still sample across the corpus. The
overlap test below is the check: if two runs with different data_seeds draw
batches that overlap at chance rate, batch composition is still effectively
independent between arms and the experiment is intact.

Run:  python test_sampler.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from config import SHUFFLE_BUFFER_ROWS, get_config, set_seed
from data_store import ActivationStore, batch_stream

BATCH, STEPS = 4096, 12
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


cfg = get_config("smoke")
store = ActivationStore(cfg.act_dir)
print(f"cache: {store.total_rows:,} rows, {store.manifest['n_shards']} shard(s), "
      f"shuffled_on_disk={store.manifest['shuffled_on_disk']}, "
      f"write_shuffle_seed={store.manifest['write_shuffle_seed']}")
print(f"buffer: {SHUFFLE_BUFFER_ROWS:,} rows "
      f"({100 * SHUFFLE_BUFFER_ROWS / store.total_rows:.1f}% of cache)\n")


def stream(data_seed: int, init_seed: int | None = None):
    """Optionally disturb the global RNGs first, mimicking a real run.

    Returns [(rows, global_indices)]. Overlap is measured on the indices, not on
    row contents: 0.81% of rows in this cache are exact byte-duplicates of some
    other row (repeated boilerplate in pile-10k giving identical activations),
    which adds ~33 false matches per 4096-row batch. Hashing contents made an
    at-chance overlap read as 2x chance -- the sampler was fine, the ruler
    was wrong.
    """
    if init_seed is not None:
        set_seed(init_seed)
        torch.randn(12288, 768)  # what SparseAutoencoder.__init__ consumes
        np.random.random(1000)
    return [(b.copy(), i.copy()) for b, i in
            batch_stream(store, BATCH, STEPS, data_seed, SHUFFLE_BUFFER_ROWS,
                         normalize=False, with_indices=True)]


print("Arm A -- same data_seed, different init seed:")
a0, a1 = stream(0, init_seed=0), stream(0, init_seed=4)
n_same = sum(np.array_equal(x[1], y[1]) for x, y in zip(a0, a1))
check("identical batch sequence", n_same == STEPS, f"{n_same}/{STEPS} batches identical")

print("\nArm B -- data_seed follows init seed:")
b0, b1 = stream(0), stream(4)
check("batch sequences differ", not any(np.array_equal(x[1], y[1]) for x, y in zip(b0, b1)))

# THE number. Under the old full random permutation this was 26/4096, matching
# BATCH^2 / total_rows. If the buffered sampler still hits chance, batch
# composition between arms is as independent as it was before and the
# throughput fix cost the experiment nothing.
expected = BATCH * BATCH / store.total_rows
overlaps = [len(np.intersect1d(x[1], y[1])) for x, y in zip(b0, b1)]
check(
    "Arm B batch-0 overlap at chance",
    overlaps[0] < 2 * expected,
    f"{overlaps[0]}/{BATCH} rows vs {expected:.1f} expected by chance",
)
check(
    "overlap stays at chance across steps",
    max(overlaps) < 2 * expected,
    f"mean {np.mean(overlaps):.1f}, max {max(overlaps)} vs {expected:.1f} expected",
)

print("\nBuffer mixing:")
check(
    "consecutive batches are disjoint",
    len(np.intersect1d(b0[0][1], b0[1][1])) == 0,
    f"{len(np.intersect1d(b0[0][1], b0[1][1]))} shared rows between batch 0 and 1",
)
# A batch pulled straight from an unshuffled sequential block would be rows
# adjacent on disk. The buffer must be breaking that up.
gaps = np.diff(np.sort(b0[0][1]))
check(
    "batch is not a contiguous disk block",
    gaps.mean() > 10,
    f"mean index gap {gaps.mean():.0f} (a raw sequential read would be 1)",
)
check("batch shape", all(x[0].shape == (BATCH, cfg.d_model) for x in b0))
check("no non-finite values", all(np.isfinite(x[0]).all() for x in b0[:3]))

print("\nNormalisation:")
raw = next(iter(batch_stream(store, BATCH, 1, 0, SHUFFLE_BUFFER_ROWS, normalize=False)))
nrm = next(iter(batch_stream(store, BATCH, 1, 0, SHUFFLE_BUFFER_ROWS, normalize=True)))
mean_raw = np.linalg.norm(raw, axis=1).mean()
mean_nrm = np.linalg.norm(nrm, axis=1).mean()
target = cfg.d_model ** 0.5
check(
    "scaled to sqrt(d_model)",
    abs(mean_nrm - target) / target < 0.05,
    f"mean||x|| {mean_raw:.1f} -> {mean_nrm:.1f} (target {target:.1f})",
)
# A uniform scalar must not change direction -- this is the property that makes
# normalisation safe for a cosine-similarity-based experiment.
cos = (raw * nrm).sum(1) / (np.linalg.norm(raw, axis=1) * np.linalg.norm(nrm, axis=1))
check("scaling preserves direction", cos.min() > 0.9999, f"min cosine {cos.min():.8f}")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all sampler invariants hold")
