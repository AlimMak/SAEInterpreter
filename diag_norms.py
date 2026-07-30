"""One-off diagnostics for the flat-L0-floor hypothesis: decoder norm inflation.

Not part of the sweep/plot pipeline. Answers four specific questions raised
about whether normalize_decoder() is actually keeping ||W_dec_i|| == 1, and
whether the L1 penalty and the logged metrics are computed the way train.py
claims. No training -- loads existing checkpoints and runs a handful of
forward passes on one cached batch.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import torch

from config import get_config
from data_store import ActivationStore, batch_stream
from sae import SparseAutoencoder, compute_metrics
import train as train_module

torch.manual_seed(0)


def norm_report(tag: str, run_name: str, cfg) -> None:
    ckpt_path = cfg.run_dir(run_name) / "sae_final.pt"
    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae)
    sae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    norms = sae.W_dec.data.norm(dim=-1)
    off = (norms - 1.0).abs() > 1e-4
    print(f"\n[{tag}] {run_name}  ({ckpt_path})")
    print(f"  min={norms.min().item():.6f}  max={norms.max().item():.6f}  "
          f"mean={norms.mean().item():.6f}  median={norms.median().item():.6f}")
    print(f"  fraction outside 1 +/- 1e-4: {off.float().mean().item():.6f} "
          f"({int(off.sum())}/{norms.numel()})")
    return sae


print("=" * 70)
print("CHECK 1: W_dec row norms, final checkpoint, low and high l1_coeff")
print("=" * 70)
cfg = get_config("full")
sae_hi = norm_report("l1=0.2 (high)", "sweep_l1_0.2", cfg)
sae_lo = norm_report("l1=0.002 (low)", "sweep_l1_0.002", cfg)


print("\n" + "=" * 70)
print("CHECK 2: training-loop step sequence (source, not paraphrase)")
print("=" * 70)
src = inspect.getsource(train_module.train)
start = src.index("xhat, f = sae(x)")
end = src.index("with torch.no_grad():\n            steps_since_fired")
print(src[start:end])


print("=" * 70)
print("CHECK 3: activation magnitude distribution + L0 at multiple thresholds")
print("=" * 70)
store = ActivationStore(cfg.act_dir)
stream = batch_stream(store, cfg.batch_size, 1, data_seed=0, buffer_rows=262_144, normalize=True)
batch_np = next(stream)
x = torch.from_numpy(batch_np)

for tag, sae in (("l1=0.2 (high)", sae_hi), ("l1=0.002 (low)", sae_lo)):
    with torch.no_grad():
        f = sae.encode(x)
    nz = f[f > 0]
    pct = np.percentile(nz.numpy(), [1, 10, 50, 90, 99])
    print(f"\n[{tag}] nonzero activation magnitude percentiles "
          f"(1/10/50/90/99): {pct}")
    print(f"  n_nonzero total = {nz.numel()}  (of {f.numel()} entries, "
          f"{x.shape[0]} tokens x {sae.d_sae} features)")

    per_feature_max = f.max(dim=0).values.clamp_min(1e-12)
    for thresh_name, thresh in [
        ("0", 0.0),
        ("1e-8", 1e-8),
        ("1e-6", 1e-6),
        ("1e-4", 1e-4),
        ("1e-2", 1e-2),
        ("1% of own max", None),
    ]:
        if thresh_name == "1% of own max":
            l0 = (f > 0.01 * per_feature_max).float().sum(-1).mean().item()
        else:
            l0 = (f > thresh).float().sum(-1).mean().item()
        print(f"  L0 @ threshold {thresh_name:<15} = {l0:.2f}")


print("\n" + "=" * 70)
print("CHECK 4: logged l1 == acts.abs().sum(-1).mean(); loss == mse + l1*l1_coeff")
print("=" * 70)
for run_name, l1_coeff in [("sweep_l1_0.2", 0.2), ("sweep_l1_0.002", 0.002)]:
    rows = [
        json.loads(line)
        for line in (cfg.run_dir(run_name) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    last = rows[-1]
    recon_loss = last["mse"] + l1_coeff * last["l1"]
    print(f"\n[{run_name}] last logged row (step {last['step']}):")
    print(f"  logged loss = {last['loss']:.6f}")
    print(f"  mse + l1_coeff*l1 = {last['mse']:.6f} + {l1_coeff:g}*{last['l1']:.6f} "
          f"= {recon_loss:.6f}")
    print(f"  match: {abs(last['loss'] - recon_loss) < 1e-4}")

    # Recompute l1 fresh on this checkpoint + a live batch, confirm it's the
    # same quantity as f.abs().sum(-1).mean() -- i.e. logged l1 is not some
    # other statistic silently substituted in.
    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae)
    sae.load_state_dict(torch.load(cfg.run_dir(run_name) / "sae_final.pt", map_location="cpu"))
    with torch.no_grad():
        f = sae.encode(x)
    fresh_l1 = f.abs().sum(-1).mean().item()
    print(f"  fresh f.abs().sum(-1).mean() on a live batch = {fresh_l1:.6f} "
          f"(compare to logged l1={last['l1']:.6f} -- won't match exactly, "
          f"different batch/step, sanity-checking magnitude only)")
