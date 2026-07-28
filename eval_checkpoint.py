"""Score a trained checkpoint with loss recovered.

Separate from train.py on purpose. Loss recovered needs GPT-2 resident, and on
an 8GB unified-memory machine holding GPT-2 alongside the SAE, the optimiser
state and the shuffle buffer pushed the system into swap hard enough that
training effectively stopped (0.1% CPU, 14MB resident). Running the eval after
training, in its own process, means the two never compete for RAM -- and it also
means a checkpoint can be re-scored later without retraining.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformer_lens import HookedTransformer

from config import get_config, get_device
from data_store import ActivationStore
from evaluate import build_eval_batch, loss_recovered
from sae import SparseAutoencoder, compute_metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="smoke", choices=["smoke", "full"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--ckpt", default="sae_final.pt")
    p.add_argument("--n_eval_seqs", type=int, default=32)
    args = p.parse_args()

    cfg = get_config(args.preset)
    device = get_device()
    run_dir = cfg.run_dir(args.run_name)

    store = ActivationStore(cfg.act_dir)
    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae).to(device)
    sae.load_state_dict(torch.load(run_dir / args.ckpt, map_location=device))
    sae.eval()

    model = HookedTransformer.from_pretrained(cfg.model_name, device=str(device))
    model.eval()
    tokens = build_eval_batch(cfg, model, args.n_eval_seqs)

    out = loss_recovered(sae, model, tokens, cfg.hook_name, store.norm_scale)

    # Geometric metrics on the same held-out text, so the two views of quality
    # are measured on identical data and can be compared directly.
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens.to(device), names_filter=cfg.hook_name, stop_at_layer=9
        )
        acts = cache[cfg.hook_name]
        if cfg.drop_bos:
            acts = acts[:, 1:, :]
        x = acts.reshape(-1, cfg.d_model).float() * store.norm_scale
        xhat, f = sae(x)
        out.update(compute_metrics(x, xhat, f))

    print(f"run={args.run_name}  ckpt={args.ckpt}  ({args.n_eval_seqs} held-out seqs)")
    print(f"  L0                 {out['l0']:.1f}")
    print(f"  explained variance {out['explained_variance']:.4f}")
    print(f"  MSE                {out['mse']:.4f}")
    print(f"  CE clean           {out['ce_clean']:.4f}")
    print(f"  CE with SAE        {out['ce_sae']:.4f}")
    print(f"  CE zero-ablated    {out['ce_zero']:.4f}")
    print(f"  LOSS RECOVERED     {out['loss_recovered']:.4f}")

    (run_dir / "eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
