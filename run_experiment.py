"""Launch the 10-run TopK reproducibility experiment (Arm A + Arm B).

Green-lit after the k=30 confirmation run (checkpoints/full/topk_k30) and its
convergence check -- EV slope ~0.0006/1000 steps over the last 3000 steps
(essentially flat) and loss_recovered non-monotonic-but-bounded across steps
5000/10000/14999. See NOTES.md Phase 4 for the full readout. STEP_COUNT below
reflects that check's conclusion: 15,000 stands, not 30,000.

Runs sequentially -- one GPU. Each run's full config lands beside its
checkpoint as run_config.json (train.py does this unconditionally), so no
separate config-saving step is needed here.

topk=30 (not l1_coeff) is passed to every run identically via
experiment_runs()'s **overrides, so the ten runs differ from each other in
exactly seed/data_seed, per the project's whole design premise.
"""

from __future__ import annotations

import time

from config import experiment_runs
from train import train

STEP_COUNT = 15_000  # see NOTES.md Phase 4 convergence check
TOPK = 30
EVAL_EVERY = 5_000  # matches the topk_k30 confirmation run exactly


def main() -> None:
    runs = experiment_runs("full", topk=TOPK, n_steps=STEP_COUNT)
    print(f"launching {len(runs)} runs: {[name for name, _ in runs]}")
    print(f"topk={TOPK}  n_steps={STEP_COUNT}  eval_every={EVAL_EVERY}\n")

    t_sweep0 = time.time()
    for i, (run_name, cfg) in enumerate(runs):
        print(f"\n=== run {i + 1}/{len(runs)}  {run_name}  "
              f"seed={cfg.seed}  data_seed={cfg.data_seed} ===")
        t0 = time.time()
        run_dir = train(cfg, run_name, eval_every=EVAL_EVERY)
        print(f"--> {run_name} done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")

    print(f"\nall {len(runs)} runs complete in {(time.time() - t_sweep0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
