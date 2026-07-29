"""Sweep l1_coeff to find the L0 / explained-variance operating point.

Why this exists: arm_a_seed0 ran to completion at l1_coeff=5e-4 (the config
default) and plateaued at L0~1280 -- far above d_model=768. With d_sae=12288
(16x expansion), any L0 above d_model can span the activation space and
reconstruct near-perfectly without having found a sparse decomposition, so
that run's EV=0.999 measured spanning coverage, not sparsity. 5e-4 was tuned
for a pre-normalization activation scale (see NOTES.md Phase 3) and was never
retuned after normalization landed.

Plateau floor: the rolling 2000-step slope of L0 on arm_a_seed0 falls from a
peak of about -180 L0/1000 steps (around step 6000) to a single-digit noise
floor by step ~13,000 and stays there through step 29,000, the last logged
step. SWEEP_STEPS is set at 15,000 -- above that floor, not below it. Phase 3
already made the mistake of reading a 400-step sweep as the l1 response when
it was actually initialisation dynamics (L0 starts near d_sae/2 and takes
thousands of steps to come down); this sweep does not repeat it.

Everything except l1_coeff is held identical to arm_a_seed0 -- same preset
(full), same seed and data_seed (0, 0) -- so the six points differ in exactly
the one variable being swept.

This script sweeps l1_coeff only. It does NOT launch the 10-run
reproducibility experiment (config.experiment_runs) -- that launch is gated
on a human reading this sweep's L0/EV frontier first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import get_config
from train import train

SWEEP_STEPS = 15_000  # plateau floor from arm_a_seed0, see module docstring
L1_LO, L1_HI, N_POINTS = 2e-3, 2e-1, 6
SEED, DATA_SEED = 0, 0  # matches arm_a_seed0 exactly except l1_coeff
SMOOTH_POINTS = 10  # average the last N logged rows (1000 steps) per operating point


def summarize(run_dir: Path) -> dict:
    """Operating-point stats for one sweep run.

    l0/ev/mse are averaged over the last SMOOTH_POINTS logged rows rather than
    read from the single final row -- step-to-step noise is large relative to
    the plateau-to-plateau differences this sweep is trying to resolve (see
    arm_a_seed0, where single-step L0 swings ~30-100 around its trend).
    dead is read from the last row only: it is close to monotone (a feature
    only leaves the dead set by firing again) and averaging would blur the
    count at the point that matters, the end of the run.
    """
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tail = rows[-SMOOTH_POINTS:]
    return {
        "l0": float(np.mean([r["l0"] for r in tail])),
        "explained_variance": float(np.mean([r["explained_variance"] for r in tail])),
        "mse": float(np.mean([r["mse"] for r in tail])),
        "dead": rows[-1]["dead"],
        "final_step": rows[-1]["step"],
        "n_logged_rows": len(rows),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps", type=int, default=SWEEP_STEPS)
    p.add_argument("--out", default=None, help="defaults to results/full/l1_sweep.json")
    p.add_argument("--l1_lo", type=float, default=L1_LO)
    p.add_argument("--l1_hi", type=float, default=L1_HI)
    p.add_argument("--n_points", type=int, default=N_POINTS)
    args = p.parse_args()

    base_cfg = get_config("full")
    out_path = Path(args.out) if args.out else base_cfg.results_dir / "full" / "l1_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    l1_values = np.logspace(np.log10(args.l1_lo), np.log10(args.l1_hi), args.n_points)
    print(f"l1_coeff sweep: {[f'{v:.4g}' for v in l1_values]}")
    print(f"n_steps={args.n_steps}  seed={SEED}  data_seed={DATA_SEED}  preset=full")
    print(f"writing incrementally -> {out_path}")

    results: list[dict] = []
    t_sweep0 = time.time()
    for i, l1 in enumerate(l1_values):
        l1 = float(l1)
        run_name = f"sweep_l1_{l1:.4g}"
        cfg = get_config(
            "full", seed=SEED, data_seed=DATA_SEED, l1_coeff=l1, n_steps=args.n_steps
        )
        print(f"\n=== sweep {i + 1}/{len(l1_values)}  l1_coeff={l1:.4g}  run={run_name} ===")
        t0 = time.time()
        run_dir = train(cfg, run_name)
        stats = summarize(run_dir)
        stats.update(l1_coeff=l1, run_name=run_name, wall_s=round(time.time() - t0, 1))
        results.append(stats)
        print(
            f"--> L0={stats['l0']:.1f}  EV={stats['explained_variance']:.4f}  "
            f"dead={stats['dead']}  ({stats['wall_s'] / 60:.1f} min)"
        )

        out_path.write_text(json.dumps(results, indent=2))

    print(
        f"\nswept {len(l1_values)} values of l1_coeff in "
        f"{(time.time() - t_sweep0) / 60:.1f} min -> {out_path}"
    )


if __name__ == "__main__":
    main()
