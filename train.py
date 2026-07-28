"""Phase 3 -- train one SAE.

One invocation produces one run. The ten runs of the experiment are ten
invocations differing only in --seed and --data_seed:

    Arm A:  --seed {0..4} --data_seed 0      (init varies, batch order fixed)
    Arm B:  --seed {0..4}                    (both vary)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import Config, get_config, get_device, set_seed
from data_store import ActivationStore, batch_index_stream
from sae import SparseAutoencoder, compute_metrics


def train(cfg: Config, run_name: str, project_grad: bool) -> Path:
    device = get_device()
    run_dir = cfg.run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    store = ActivationStore(cfg.act_dir)
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed

    print(f"run={run_name}  device={device}")
    print(f"seed={cfg.seed}  data_seed={data_seed}"
          f"{'  (follows seed)' if cfg.data_seed is None else '  (pinned)'}")
    print(f"rows={store.total_rows:,}  d_sae={cfg.d_sae}  "
          f"steps={cfg.n_steps:,}  epochs={cfg.batch_size * cfg.n_steps / store.total_rows:.1f}")
    print(f"decoder grad projection: {'ON' if project_grad else 'OFF (ablation)'}")

    # set_seed is called with `seed`, not data_seed: it governs weight init and
    # any other torch-side randomness. Batch order is driven separately by the
    # numpy Generator inside batch_index_stream, which is the entire point --
    # the two sources of randomness must not share a stream or Arm A cannot
    # hold one fixed while varying the other.
    set_seed(cfg.seed)

    sae = SparseAutoencoder(cfg.d_model, cfg.d_sae).to(device)

    # Seed b_dec from a sample of real activations rather than zeros.
    sample_idx = np.sort(
        np.random.default_rng(data_seed).choice(
            store.total_rows, size=min(65_536, store.total_rows), replace=False
        )
    )
    sample = torch.from_numpy(store.gather(sample_idx)).to(device)
    sae.init_b_dec_from_data(sample)
    del sample

    sae.normalize_decoder()  # unit norm from step 0, not just after step 1
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    # Dead-feature tracking: steps since each feature last fired. Counted, not
    # resampled -- resampling is a seed-dependent intervention and adding it in
    # v1 would confound the thing this project measures.
    steps_since_fired = torch.zeros(cfg.d_sae, dtype=torch.long, device=device)

    log_path = run_dir / "metrics.jsonl"
    log_f = log_path.open("w")
    t_start = time.time()
    throughput: list[float] = []

    for step, idx in enumerate(
        batch_index_stream(store.total_rows, cfg.batch_size, cfg.n_steps, data_seed)
    ):
        t0 = time.time()
        x = torch.from_numpy(store.gather(idx)).to(device)

        xhat, f = sae(x)
        mse = ((xhat - x) ** 2).sum(-1).mean()
        # Plain L1 on the feature activations. This is only a valid sparsity
        # penalty because the decoder rows are held at unit norm -- see
        # SparseAutoencoder.normalize_decoder.
        l1 = f.abs().sum(-1).mean()
        loss = mse + cfg.l1_coeff * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if project_grad:
            sae.project_decoder_grad()  # BEFORE the step: keeps Adam's v clean

        opt.step()
        sae.normalize_decoder()  # still required -- Adam's step is not tangential

        with torch.no_grad():
            fired = (f > 0).any(dim=0)
            steps_since_fired += 1
            steps_since_fired[fired] = 0

        if step < 100:
            throughput.append(cfg.batch_size / (time.time() - t0))
        if step == 99:
            print(f"throughput (first 100 steps): {np.mean(throughput):,.0f} samples/s "
                  f"-> ~{cfg.n_steps * cfg.batch_size / np.mean(throughput) / 60:.1f} min projected")

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            with torch.no_grad():
                m = compute_metrics(x, xhat, f)
            m.update(
                step=step,
                loss=loss.item(),
                l1=l1.item(),
                dead=int((steps_since_fired > cfg.dead_feature_window).sum()),
                elapsed=round(time.time() - t_start, 1),
            )
            log_f.write(json.dumps(m) + "\n")
            log_f.flush()
            print(f"step {step:>6}  mse {m['mse']:>9.2f}  L0 {m['l0']:>8.1f}  "
                  f"EV {m['explained_variance']:>6.3f}  dead {m['dead']:>5}")

        if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
            torch.save(sae.state_dict(), run_dir / f"sae_step{step}.pt")

    log_f.close()
    torch.save(sae.state_dict(), run_dir / "sae_final.pt")

    # The config travels with the checkpoint. A checkpoint whose
    # hyperparameters live only in a shell history is not reproducible, and
    # this project's entire claim is about reproducibility.
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    meta.update(
        run_name=run_name,
        resolved_data_seed=data_seed,
        decoder_grad_projection=project_grad,
        total_rows=store.total_rows,
        cache_manifest=store.manifest,
        wall_clock_s=round(time.time() - t_start, 1),
    )
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

    print(f"\nsaved -> {run_dir}  ({time.time() - t_start:.0f}s)")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="smoke", choices=["smoke", "full"])
    p.add_argument("--seed", type=int, default=0, help="weight initialisation")
    p.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="batch order; omit to follow --seed (Arm B), pin to a constant for Arm A",
    )
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--no_decoder_projection",
        action="store_true",
        help="ablation: skip the decoder gradient projection",
    )
    args = p.parse_args()

    cfg = get_config(args.preset, seed=args.seed, data_seed=args.data_seed)
    run_name = args.run_name or (
        f"seed{args.seed}" + ("" if args.data_seed is None else f"_data{args.data_seed}")
    )
    train(cfg, run_name, project_grad=not args.no_decoder_projection)


if __name__ == "__main__":
    main()
