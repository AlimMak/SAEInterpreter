"""Coherence diagnostics: pairwise decoder-row cosine similarity and the
reconstruction coherence ratio, at both l1 extremes. No training.

Tests the mutual-incoherence hypothesis: L1 penalizes sum(|f_i|) and cannot
distinguish a few large, near-orthogonal activations from many small,
correlated ones -- so once the decoder dictionary is coherent, dense solutions
can match a sparse solution's L1 budget while reconstructing just as well.
"""

from __future__ import annotations

import torch

from config import get_config
from data_store import ActivationStore, batch_stream
from sae import SparseAutoencoder

cfg = get_config("full")
store = ActivationStore(cfg.act_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(run_name: str) -> SparseAutoencoder:
    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae)
    sae.load_state_dict(torch.load(cfg.run_dir(run_name) / "sae_final.pt", map_location="cpu"))
    return sae.to(device)


stream = batch_stream(store, cfg.batch_size, 4, data_seed=0, buffer_rows=262_144, normalize=True)
x_all = torch.cat([torch.from_numpy(b) for b in stream], dim=0).to(device)
print(f"using {x_all.shape[0]} tokens, device={device}")

RANDOM_BASELINE_SD = 1.0 / (cfg.d_model ** 0.5)
print(f"random-unit-vector baseline in R^{cfg.d_model}: mean~0, sd~{RANDOM_BASELINE_SD:.4f}")

for run_name in ("sweep_l1_0.2", "sweep_l1_0.002"):
    sae = load(run_name)
    print("\n" + "=" * 70)
    print(f"[{run_name}]")
    print("=" * 70)

    with torch.no_grad():
        xhat, f = sae(x_all)
    freq = (f > 0).float().mean(0)
    alive = (freq > 0).nonzero(as_tuple=True)[0]
    print(f"alive features (fired >=1x in sample): {alive.numel()} / {sae.d_sae}")

    print("\nCHECK A: pairwise cosine similarity of W_dec rows (alive only)")
    W = sae.W_dec.data[alive]  # rows already unit-norm (normalize_decoder invariant)
    sim = W @ W.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=device)
    offdiag = sim[mask]
    print(f"  n alive features = {n:,}   n ordered pairs = {offdiag.numel():,}")
    print(f"  max off-diagonal cosine sim        = {offdiag.max().item():.4f}")
    print(f"  mean |cosine sim| (off-diagonal)   = {offdiag.abs().mean().item():.5f}  "
          f"vs random baseline sd {RANDOM_BASELINE_SD:.4f}")
    print(f"  fraction of pairs |sim| > 0.5       = {(offdiag.abs() > 0.5).float().mean().item():.6f}")
    print(f"  fraction of pairs |sim| > 0.9       = {(offdiag.abs() > 0.9).float().mean().item():.6f}")

    print("\nCHECK B: coherence ratio, per token, averaged over the batch")
    xhat_norm = xhat.norm(dim=-1)
    l2f = f.pow(2).sum(-1).sqrt()
    l0 = (f > 0).float().sum(-1)
    print(f"  ||x_hat|| (incl. b_dec) mean        = {xhat_norm.mean().item():.3f}")
    print(f"  sqrt(sum f^2) mean                  = {l2f.mean().item():.3f}")
    print(f"  ratio ||x_hat|| / sqrt(sum f^2)      = {(xhat_norm / l2f).mean().item():.3f}  "
          f"(user's literal formula, x_hat includes b_dec)")

    delta = f @ sae.W_dec  # feature contribution only, excludes b_dec
    delta_norm = delta.norm(dim=-1)
    l1f = f.sum(-1)  # == training's l1 statistic, per-token here instead of batch-mean
    print(f"  [excl. b_dec] ||delta|| mean        = {delta_norm.mean().item():.3f}")
    print(f"  [excl. b_dec] ||delta||/sqrt(sum f^2)= {(delta_norm / l2f).mean().item():.4f}  "
          f"(1.0 if orthogonal, up to sqrt(L0) if perfectly parallel)")
    print(f"  [excl. b_dec] ||delta||/sum(f)       = {(delta_norm / l1f).mean().item():.4f}  "
          f"(1.0 if perfectly parallel/coherent, ~1/sqrt(L0) if orthogonal)")
    print(f"  mean L0 this sample = {l0.mean().item():.1f}  ->  1/sqrt(L0) = {l0.mean().item() ** -0.5:.4f}")
