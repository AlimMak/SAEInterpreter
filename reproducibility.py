"""Cross-seed feature reproducibility: does the same sparse feature reappear
across independently trained SAEs?

Requires all 10 runs of the TopK reproducibility experiment
(config.experiment_runs, launched by run_experiment.py) to exist under
checkpoints/full/. This script does not run as part of that experiment --
invoke it separately, after training finishes: `python reproducibility.py`.
It refuses to run against a partial set of checkpoints rather than silently
reporting a number computed from whichever runs happen to exist.

Method
------
Both SAE variants in this repo hold decoder rows at unit norm throughout
training (SparseAutoencoder.normalize_decoder), so cosine similarity between
two runs' decoder rows is just a dot product -- no renormalisation needed at
comparison time.

"Does feature i in run A correspond to feature j in run B" is posed as a
matching problem. Greedy max-cosine allows collisions: several run-A features
can all claim the same run-B partner, silently inflating the apparent
reproducibility rate. Every number in this script therefore uses Hungarian
assignment (scipy.optimize.linear_sum_assignment) for a global one-to-one
pairing; greedy is not used at all here (it was compared once, historically,
to justify this choice -- see NOTES.md Phase 5, ~67% collision rate).

Version 2 -- reworked after a review of the first pass found two problems
with reporting a single threshold as the headline:

  1. The null-calibrated threshold (~0.17) was far more permissive than real
     matched similarities (~0.69 mean) -- "clears the null" and "is the same
     feature" are not the same claim, and a single pass/fail cutoff hid that
     gap.
  2. Both arms saturated near the ceiling (median score 1.0 on a 0/4..4/4
     scale) at that permissive threshold, so the Arm A vs Arm B comparison
     carried no information -- a ceiling effect, not a null result.

The fix is to never collapse a feature's match quality to a single pass/fail
bit before comparing arms. Three views on the same underlying data (every
feature's Hungarian-matched cosine similarity to each of its 4 same-arm
peers), all threshold-light or threshold-swept rather than threshold-fixed:

  1. The raw similarity distribution itself, Arm A vs Arm B vs null, with
     percentiles and a KS test -- no threshold at all.
  2. Fraction of features clearing a threshold against ALL 4 peers, swept
     across thresholds -- the curve, not one point on it.
  3. The strict set: clears 0.9 against all 4 peers. This is the one number
     meant to be used downstream (e.g. a steering phase) as "these are real."

The null
--------
d_sae~12288 directions packed into d_model=768 dimensions are, by pigeonhole,
never going to be mutually orthogonal -- some pairwise correlation is
unavoidable even in a dictionary with zero learned structure. This script
matches a real SAE's decoder against independently drawn random unit-norm
dictionaries of the same size and carries that null distribution through
every plot above, not just past a single threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

from config import experiment_runs, get_config
from data_store import ActivationStore, batch_stream
from sae import TopKSparseAutoencoder

ALIVE_SAMPLE_TOKENS = 16_384   # tokens used to decide which features ever fire
N_NULL_GROUPS = 5              # independent "4 fake peers" draws for the null threshold curve
THRESHOLDS = np.round(np.arange(0.17, 0.951, 0.05), 2)
STRICT_THRESHOLD = 0.9         # item 3 -- the number a downstream steering phase should use
CALLOUT_THRESHOLDS = (0.5, 0.7, 0.9)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e0e0dc",
                  arm_a="#2a78d6", arm_b="#eb6834", null="#9a9890"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7", grid="#3a3a38",
                 arm_a="#3987e5", arm_b="#d95926", null="#75736a"),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def require_all_runs_complete(cfg, run_names: list[str]) -> None:
    missing = [
        name for name in run_names
        if not (cfg.run_dir(name) / "sae_final.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(run_names)} runs not finished yet: {missing}. "
            f"reproducibility.py must not run against a partial experiment -- "
            f"wait for run_experiment.py to complete."
        )


def load_decoder(cfg, run_name: str) -> torch.Tensor:
    sae = TopKSparseAutoencoder(cfg.d_model, cfg.d_sae, cfg.topk)
    sae.load_state_dict(torch.load(cfg.run_dir(run_name) / "sae_final.pt", map_location="cpu"))
    return sae.W_dec.data.clone()  # (d_sae, d_model), unit-norm rows


def alive_mask(cfg, run_name: str, x_sample: torch.Tensor) -> torch.Tensor:
    sae = TopKSparseAutoencoder(cfg.d_model, cfg.d_sae, cfg.topk)
    sae.load_state_dict(torch.load(cfg.run_dir(run_name) / "sae_final.pt", map_location="cpu"))
    with torch.no_grad():
        f = sae.encode(x_sample)
    return (f > 0).any(dim=0)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def hungarian_match(W1: torch.Tensor, W2: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global one-to-one assignment maximising total cosine similarity.

    Returns (row_idx_in_W1, col_idx_in_W2, matched_similarity). If W1 and W2
    differ in size, len(row_idx) == min(len(W1), len(W2)) -- the larger set's
    unmatched rows are simply absent, not padded with a fabricated score.
    """
    S = (W1 @ W2.T).numpy()
    row, col = linear_sum_assignment(-S)
    return row, col, S[row, col]


def random_unit_dictionary(n: int, d_model: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, d_model, generator=g)
    return R / R.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# The shared primitive: per-feature, per-peer matched similarity. No
# thresholding anywhere in this section -- every downstream view (items 1-3)
# is a different way of looking at the same matrix.
# --------------------------------------------------------------------------
def pairwise_sims_matrix(
    run_names: list[str], decoders: dict[str, torch.Tensor], alive: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """For each run, the Hungarian-matched cosine similarity of every alive
    feature to its partner in each of the other runs in `run_names`.

    Returns {run_name: matrix of shape (n_alive_features, len(run_names)-1)}.
    A feature that wasn't assigned a partner in some comparison (only
    possible if alive-counts differ between the two runs) gets 0.0 there,
    which fails every threshold this script uses -- it never inflates a
    reproducibility number by omission.
    """
    result: dict[str, np.ndarray] = {}
    for name_i in run_names:
        Wi = decoders[name_i][alive[name_i]]
        n_i = Wi.shape[0]
        peers = [n for n in run_names if n != name_i]
        M = np.zeros((n_i, len(peers)), dtype=np.float32)
        for k, name_j in enumerate(peers):
            Wj = decoders[name_j][alive[name_j]]
            row, _, sims = hungarian_match(Wi, Wj)
            M[row, k] = sims
        result[name_i] = M
    return result


def null_sims(reference_W: torch.Tensor, d_model: int, n_peers: int, n_groups: int,
              seed0: int = 90_000) -> np.ndarray:
    """Hungarian-match `reference_W` against n_groups independent sets of
    n_peers random unit-norm dictionaries. Shape (n_features, n_groups,
    n_peers) -- same per-feature, per-peer structure as a real run's matrix,
    against pure noise instead of another trained SAE.
    """
    n = reference_W.shape[0]
    out = np.zeros((n, n_groups, n_peers), dtype=np.float32)
    trial = 0
    for g in range(n_groups):
        for p in range(n_peers):
            R = random_unit_dictionary(n, d_model, seed=seed0 + trial)
            row, _, sims = hungarian_match(reference_W, R)
            out[row, g, p] = sims
            trial += 1
    return out


# --------------------------------------------------------------------------
# Item 1: threshold-free distribution comparison
# --------------------------------------------------------------------------
def distribution_stats(flat_sims: np.ndarray) -> dict:
    p10, p25, p50, p75, p90 = np.percentile(flat_sims, [10, 25, 50, 75, 90])
    return {
        "mean": float(flat_sims.mean()), "median": float(p50),
        "p10": float(p10), "p25": float(p25), "p75": float(p75), "p90": float(p90),
        "n": int(flat_sims.size),
    }


def plot_distribution(arm_a_flat, arm_b_flat, null_flat, out: Path, mode: str) -> None:
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=t["surface"])
    bins = np.linspace(0, 1, 81)
    for data, color, label in (
        (null_flat, t["null"], "null (random dictionary)"),
        (arm_a_flat, t["arm_a"], "Arm A"),
        (arm_b_flat, t["arm_b"], "Arm B"),
    ):
        ax.hist(data, bins=bins, density=True, histtype="step", linewidth=1.8,
                 color=color, label=label, zorder=3)
    ax.set_xlabel("Hungarian-matched cosine similarity to one peer run", fontsize=9, color=t["muted"])
    ax.set_ylabel("density", fontsize=9, color=t["ink"])
    ax.set_title(
        "Matched-similarity distribution: Arm A vs Arm B vs null\n"
        "every feature x peer pair, no threshold applied",
        fontsize=10.5, color=t["ink"], loc="left",
    )
    ax.legend(fontsize=8.5, frameon=False, labelcolor=t["ink"])
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Item 2: reproducibility-vs-threshold curve
# --------------------------------------------------------------------------
def threshold_curve(sims_matrices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of features (pooled across all runs in this arm) that clear
    EVERY one of their peers -- min over the peer axis -- at each of
    THRESHOLDS. Returns (curve, per_feature_min_similarities).
    """
    mins = np.concatenate([M.min(axis=1) for M in sims_matrices.values()])
    curve = np.array([(mins >= th).mean() for th in THRESHOLDS])
    return curve, mins


def plot_threshold_curve(arm_a_curve, arm_b_curve, null_curve, out: Path, mode: str) -> None:
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=t["surface"])
    ax.plot(THRESHOLDS, arm_a_curve, color=t["arm_a"], marker="o", ms=4, lw=1.6, label="Arm A", zorder=3)
    ax.plot(THRESHOLDS, arm_b_curve, color=t["arm_b"], marker="o", ms=4, lw=1.6, label="Arm B", zorder=3)
    ax.plot(THRESHOLDS, null_curve, color=t["null"], marker="o", ms=4, lw=1.6, ls="--",
             label="null (chance)", zorder=2)
    ax.axvline(STRICT_THRESHOLD, color=t["muted"], lw=1, ls=":", zorder=1)
    ax.text(STRICT_THRESHOLD, 1.0, f" strict ({STRICT_THRESHOLD})", fontsize=8, color=t["muted"],
             va="top", ha="left")
    ax.set_xlabel("threshold (must clear ALL 4 peers)", fontsize=9, color=t["muted"])
    ax.set_ylabel("fraction of features", fontsize=9, color=t["ink"])
    ax.set_title("Reproducibility vs threshold -- Arm A vs Arm B vs null",
                  fontsize=10.5, color=t["ink"], loc="left")
    ax.legend(fontsize=8.5, frameon=False, labelcolor=t["ink"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
def main() -> None:
    cfg = get_config("full", topk=30, n_steps=15_000)  # must match run_experiment.py exactly
    runs = experiment_runs("full", topk=30, n_steps=15_000)
    run_names = [name for name, _ in runs]
    arm_a_names = [n for n in run_names if n.startswith("arm_a")]
    arm_b_names = [n for n in run_names if n.startswith("arm_b")]

    require_all_runs_complete(cfg, run_names)

    store = ActivationStore(cfg.act_dir)
    stream = batch_stream(store, cfg.batch_size, 4, data_seed=0,
                           buffer_rows=262_144, normalize=True)
    x_sample = torch.cat([torch.from_numpy(b) for b in stream], dim=0)[:ALIVE_SAMPLE_TOKENS]

    print(f"loading {len(run_names)} decoders + alive masks ({x_sample.shape[0]} sample tokens)...")
    decoders = {name: load_decoder(cfg, name) for name in run_names}
    alive = {name: alive_mask(cfg, name, x_sample) for name in run_names}
    for name in run_names:
        print(f"  {name}: {int(alive[name].sum())}/{cfg.d_sae} alive")

    out_dir = cfg.results_dir / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "repro_raw_sims"
    raw_dir.mkdir(exist_ok=True)

    print("\ncomputing within-arm pairwise Hungarian matches...")
    arm_a_sims = pairwise_sims_matrix(arm_a_names, decoders, alive)
    arm_b_sims = pairwise_sims_matrix(arm_b_names, decoders, alive)
    for name, M in {**arm_a_sims, **arm_b_sims}.items():
        np.save(raw_dir / f"{name}.npy", M)
    print(f"saved raw per-feature, per-peer matched similarities -> {raw_dir}")

    n_peers = len(arm_a_names) - 1  # == len(arm_b_names) - 1 == 4
    ref_name = arm_a_names[0]
    ref_W = decoders[ref_name][alive[ref_name]]
    print(f"\nmeasuring null ({N_NULL_GROUPS} groups x {n_peers} random dictionaries, "
          f"reference={ref_name})...")
    null = null_sims(ref_W, cfg.d_model, n_peers, N_NULL_GROUPS)
    null_flat = null.reshape(-1)
    null_mins = null.min(axis=2).reshape(-1)

    # ---- Item 1: threshold-free distribution ----
    arm_a_flat = np.concatenate([M.reshape(-1) for M in arm_a_sims.values()])
    arm_b_flat = np.concatenate([M.reshape(-1) for M in arm_b_sims.values()])
    stats_a = distribution_stats(arm_a_flat)
    stats_b = distribution_stats(arm_b_flat)
    stats_null = distribution_stats(null_flat)
    ks_stat, ks_p = ks_2samp(arm_a_flat, arm_b_flat)

    print("\n=== ITEM 1: threshold-free distribution (every feature x peer pair) ===")
    print(f"Arm A: mean={stats_a['mean']:.4f}  median={stats_a['median']:.4f}  "
          f"p10={stats_a['p10']:.4f}  p25={stats_a['p25']:.4f}  "
          f"p75={stats_a['p75']:.4f}  p90={stats_a['p90']:.4f}  n={stats_a['n']}")
    print(f"Arm B: mean={stats_b['mean']:.4f}  median={stats_b['median']:.4f}  "
          f"p10={stats_b['p10']:.4f}  p25={stats_b['p25']:.4f}  "
          f"p75={stats_b['p75']:.4f}  p90={stats_b['p90']:.4f}  n={stats_b['n']}")
    print(f"null:  mean={stats_null['mean']:.4f}  median={stats_null['median']:.4f}")
    print(f"KS test (Arm A vs Arm B): statistic={ks_stat:.4f}  p-value={ks_p:.4g}")
    print("  (caveat: samples are not iid -- each feature contributes 4 correlated "
          "values, one per peer, and peer decoders are shared across many features' "
          "comparisons -- read the p-value as indicative, not a formal test)")

    for mode in ("light", "dark"):
        plot_distribution(arm_a_flat, arm_b_flat, null_flat,
                            out_dir / f"repro_distribution_{mode}.png", mode)

    # ---- Item 2: threshold curve ----
    arm_a_curve, arm_a_mins = threshold_curve(arm_a_sims)
    arm_b_curve, arm_b_mins = threshold_curve(arm_b_sims)
    null_curve = np.array([(null_mins >= th).mean() for th in THRESHOLDS])

    print(f"\n=== ITEM 2: fraction clearing threshold against ALL {n_peers} peers ===")
    print(f"{'threshold':>10} {'Arm A':>10} {'Arm B':>10} {'null':>10}")
    for th, a, b, nu in zip(THRESHOLDS, arm_a_curve, arm_b_curve, null_curve):
        print(f"{th:>10.2f} {a:>10.4f} {b:>10.4f} {nu:>10.4f}")

    # CALLOUT_THRESHOLDS (0.5/0.7/0.9) don't land on the THRESHOLDS grid (step
    # 0.05 from 0.17), so they're computed directly here rather than hoped for
    # from the sweep -- a prior version silently never printed them because of
    # exactly this mismatch.
    print(f"\ncallouts (computed directly, not read off the grid above):")
    callouts = {}
    for th in CALLOUT_THRESHOLDS:
        fa = (arm_a_mins >= th).mean()
        fb = (arm_b_mins >= th).mean()
        fn = (null_mins >= th).mean()
        callouts[th] = {"arm_a": float(fa), "arm_b": float(fb), "null": float(fn)}
        print(f"  threshold={th:.2f}: Arm A={fa:.4f}  Arm B={fb:.4f}  "
              f"diff={fa - fb:+.4f}  null={fn:.4f}")

    for mode in ("light", "dark"):
        plot_threshold_curve(arm_a_curve, arm_b_curve, null_curve,
                               out_dir / f"repro_threshold_curve_{mode}.png", mode)

    # ---- Item 3: strict reproducibility at 0.9 ----
    strict_a = arm_a_mins >= STRICT_THRESHOLD
    strict_b = arm_b_mins >= STRICT_THRESHOLD
    print(f"\n=== ITEM 3: STRICT reproducibility (clears {STRICT_THRESHOLD} against ALL "
          f"{n_peers} peers) ===")
    print(f"Arm A: {int(strict_a.sum())}/{len(strict_a)} = {strict_a.mean():.4f}")
    print(f"Arm B: {int(strict_b.sum())}/{len(strict_b)} = {strict_b.mean():.4f}")

    strict_features: dict[str, list[int]] = {}
    for arm_names, sims_dict in ((arm_a_names, arm_a_sims), (arm_b_names, arm_b_sims)):
        for name in arm_names:
            alive_idx = alive[name].nonzero(as_tuple=True)[0].numpy()
            mins = sims_dict[name].min(axis=1)
            strict_features[name] = alive_idx[mins >= STRICT_THRESHOLD].tolist()
    (out_dir / "repro_strict_0.9_features.json").write_text(json.dumps(strict_features))
    print(f"wrote per-run strict feature indices (original d_sae indices) -> "
          f"{out_dir / 'repro_strict_0.9_features.json'}")

    summary = {
        "n_peers": n_peers,
        "arm_a": stats_a, "arm_b": stats_b, "null": stats_null,
        "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p),
        "thresholds": THRESHOLDS.tolist(),
        "arm_a_curve": arm_a_curve.tolist(), "arm_b_curve": arm_b_curve.tolist(),
        "null_curve": null_curve.tolist(),
        "callouts": callouts,
        "strict_threshold": STRICT_THRESHOLD,
        "strict_arm_a_count": int(strict_a.sum()), "strict_arm_a_total": int(len(strict_a)),
        "strict_arm_b_count": int(strict_b.sum()), "strict_arm_b_total": int(len(strict_b)),
    }
    (out_dir / "repro_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'repro_summary.json'}")


if __name__ == "__main__":
    main()
