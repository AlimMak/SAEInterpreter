"""Assertions the two-arm design rests on.

Arm A claims to hold batch order fixed while varying initialisation. That claim
is only true if the batch index stream depends on `data_seed` and on nothing
else. If torch's RNG or the weight-init seed leaked into it, Arm A would
secretly be Arm B, the two arms would agree, and the headline result -- "the
gap between the arms" -- would be an artifact of a broken sampler.

Run:  python test_sampler.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from config import get_config, set_seed
from data_store import batch_index_stream

N_ROWS, BATCH, STEPS = 635_000, 4096, 12
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def stream(data_seed: int, init_seed: int | None = None) -> list[np.ndarray]:
    """Optionally disturb the global RNGs first, mimicking a real run."""
    if init_seed is not None:
        set_seed(init_seed)
        torch.randn(12288, 768)  # what SparseAutoencoder.__init__ consumes
        np.random.random(1000)
    return list(batch_index_stream(N_ROWS, BATCH, STEPS, data_seed))


print("Arm A -- same data_seed, different init seed:")
a0, a1 = stream(0, init_seed=0), stream(0, init_seed=4)
check(
    "identical batch sequence",
    all(np.array_equal(x, y) for x, y in zip(a0, a1)),
    f"{sum(np.array_equal(x, y) for x, y in zip(a0, a1))}/{STEPS} batches identical",
)

print("\nArm B -- data_seed follows init seed:")
b0, b1 = stream(0), stream(4)
check(
    "batch sequences differ",
    not any(np.array_equal(x, y) for x, y in zip(b0, b1)),
    f"overlap of batch 0: {len(np.intersect1d(b0[0], b1[0]))}/{BATCH} rows",
)

print("\nPermutation integrity:")
flat = np.concatenate(list(batch_index_stream(N_ROWS, BATCH, N_ROWS // BATCH, 0)))
check("no row repeats within an epoch", len(np.unique(flat)) == len(flat))
check("indices in range", flat.min() >= 0 and flat.max() < N_ROWS)
check("batches are sorted (memmap locality)", all(np.all(np.diff(x) > 0) for x in a0))
check(
    "coverage is a true permutation, not blocked",
    # A block shuffle would leave long runs of consecutive indices. In a full
    # permutation the mean gap between sorted neighbours in a batch is
    # N_ROWS/BATCH ~= 155; in a block shuffle it would be ~1.
    np.diff(a0[0]).mean() > 50,
    f"mean gap {np.diff(a0[0]).mean():.0f} (block-shuffled would be ~1)",
)

print("\nEpoch boundary:")
long_stream = list(batch_index_stream(N_ROWS, BATCH, (N_ROWS // BATCH) + 2, 0))
check(
    "reshuffles rather than repeating the same epoch",
    not np.array_equal(long_stream[0], long_stream[N_ROWS // BATCH]),
)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all sampler invariants hold")
