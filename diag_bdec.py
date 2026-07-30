"""Follow-up diagnostics: reconcile check 4 at l1=0.2, then test the b_dec/
dense-offset-feature hypothesis for the flat L0 floor.

No training. Loads existing checkpoints, runs forward passes on cached
batches.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from config import get_config
from data_store import ActivationStore, batch_stream
from sae import SparseAutoencoder

cfg = get_config("full")
store = ActivationStore(cfg.act_dir)

print("=" * 70)
print("CHECK 1 (redo): mse, l1, l1_coeff*l1, loss printed separately, l1=0.2")
print("=" * 70)
for run_name, l1_coeff in [("sweep_l1_0.2", 0.2), ("sweep_l1_0.002", 0.002)]:
    rows = [
        json.loads(line)
        for line in (cfg.run_dir(run_name) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    last = rows[-1]
    mse, l1, loss = last["mse"], last["l1"], last["loss"]
    print(f"\n[{run_name}]  step={last['step']}")
    print(f"  mse            = {mse!r}")
    print(f"  l1             = {l1!r}")
    print(f"  l1_coeff       = {l1_coeff!r}")
    print(f"  l1_coeff * l1  = {l1_coeff * l1!r}")
    print(f"  mse + l1_coeff*l1 = {mse + l1_coeff * l1!r}")
    print(f"  logged loss       = {loss!r}")
    print(f"  abs diff          = {abs(loss - (mse + l1_coeff * l1)):.8f}")


def load(run_name: str) -> SparseAutoencoder:
    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae)
    sae.load_state_dict(torch.load(cfg.run_dir(run_name) / "sae_final.pt", map_location="cpu"))
    return sae


# A few thousand tokens across several batches (different draws, same store).
stream = batch_stream(store, cfg.batch_size, 4, data_seed=0, buffer_rows=262_144, normalize=True)
batches = [torch.from_numpy(b) for b in stream]
x_all = torch.cat(batches, dim=0)  # 16384 tokens
print(f"\n(using {x_all.shape[0]} tokens across {len(batches)} batches for checks 2-4)")

# Data mean at the same scale b_dec was initialised from (normalized acts).
sample_idx = np.sort(
    np.random.default_rng(0).choice(store.total_rows, size=65_536, replace=False)
)
data_sample = torch.from_numpy(store.gather(sample_idx) * store.norm_scale)
data_mean = data_sample.mean(0)
print(f"data mean norm (||mean of normalized activations||) = {data_mean.norm().item():.4f}")

for run_name in ("sweep_l1_0.2", "sweep_l1_0.002"):
    sae = load(run_name)
    print("\n" + "=" * 70)
    print(f"[{run_name}]")
    print("=" * 70)

    with torch.no_grad():
        f = sae.encode(x_all)  # (n_tokens, d_sae)

    print("\nCHECK 2: per-feature firing frequency")
    freq = (f > 0).float().mean(0)  # fraction of tokens each feature fires on
    hist, edges = np.histogram(freq.numpy(), bins=[0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"  freq in [{lo:.3f}, {hi:.3f}): {c} features")
    dense = (freq > 0.9).sum().item()
    print(f"  features firing on >90% of tokens: {dense}")
    print(f"  features firing on >99% of tokens: {(freq > 0.99).sum().item()}")
    print(f"  features that never fire (freq==0): {(freq == 0).sum().item()}")
    l0_mean = (f > 0).float().sum(-1).mean().item()
    print(f"  mean L0 over this sample: {l0_mean:.2f}")
    print(f"  dense (>90%) features as fraction of mean L0: {dense / l0_mean:.3f}")

    print("\nCHECK 3: b_dec norm vs data mean norm")
    b_dec_norm = sae.b_dec.data.norm().item()
    print(f"  b_dec nn.Parameter default: zeros (sae.py __init__)")
    print(f"  BUT train.py calls init_b_dec_from_data() before step 0 -- "
          f"b_dec is data-mean-initialised, not left at zero")
    print(f"  b_dec learned norm (this checkpoint, after full training) = {b_dec_norm:.4f}")
    print(f"  data mean norm                        = {data_mean.norm().item():.4f}")
    print(f"  cos(b_dec, data_mean) = "
          f"{torch.nn.functional.cosine_similarity(sae.b_dec.data, data_mean, dim=0).item():.4f}")

    print("\nCHECK 4: b_enc distribution at convergence")
    b_enc = sae.b_enc.data
    pct = np.percentile(b_enc.numpy(), [1, 10, 25, 50, 75, 90, 99])
    print(f"  b_enc percentiles [1/10/25/50/75/90/99] = {pct}")
    print(f"  b_enc mean = {b_enc.mean().item():.4f}  std = {b_enc.std().item():.4f}")
    print(f"  fraction of b_enc < 0: {(b_enc < 0).float().mean().item():.4f}")
    print(f"  fraction of b_enc < -1: {(b_enc < -1).float().mean().item():.4f}")
