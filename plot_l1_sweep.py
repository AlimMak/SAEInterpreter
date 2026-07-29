"""Plot the l1_coeff sweep's L0 / explained-variance frontier.

One scatter: L0 on a log x-axis against explained variance on y, one point
per swept l1_coeff. Marker size encodes the dead-feature count so the reader
can see, at a glance, whether the sparsity gained toward the L0=30 target is
being bought with dead features. Reference lines mark the target region from
the sweep brief: L0 near 30, EV above 0.85.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config

# Same palette as plot_ablation.py -- one categorical colour per project, kept
# consistent so colour means the same thing across every figure in this repo.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  grid="#e0e0dc", mark="#2a78d6", target="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 grid="#3a3a38", mark="#3987e5", target="#d95926"),
}


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not data:
        raise ValueError(f"{path} is empty -- sweep has not written any points yet")
    return sorted(data, key=lambda d: d["l1_coeff"])


def marker_sizes(dead: list[int]) -> list[float]:
    """Area-scaled marker size: 40pt^2 floor, up to 400pt^2 at the sweep's max dead count."""
    peak = max(dead) or 1
    return [40 + 360 * (d / peak) for d in dead]


# Fan-out offsets (points, in reading order once sorted by L0) for the label
# leader lines. Six sweep points is small enough to hand-tune rather than
# pull in a label-placement dependency; several points land within a few
# percent of each other in log(L0) and would otherwise overlap.
LABEL_OFFSETS = [(-75, -22), (60, 4), (55, -34), (60, 34), (60, 60), (-15, -46)]


def plot(data: list[dict], mode: str, out: Path) -> None:
    t = THEMES[mode]
    ordered = sorted(data, key=lambda d: d["l0"])
    xs = [d["l0"] for d in ordered]
    ys = [d["explained_variance"] for d in ordered]
    dead = [d["dead"] for d in ordered]
    l1 = [d["l1_coeff"] for d in ordered]

    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    ax.scatter(
        xs, ys, s=marker_sizes(dead), color=t["mark"], edgecolor=t["ink"],
        linewidth=0.6, alpha=0.9, zorder=3,
    )
    for i, (x, y, coeff, d) in enumerate(zip(xs, ys, l1, dead)):
        dx, dy = LABEL_OFFSETS[i % len(LABEL_OFFSETS)]
        ax.annotate(
            f"l1={coeff:.3g}  dead={d}", (x, y), textcoords="offset points",
            xytext=(dx, dy), fontsize=8, color=t["ink"], zorder=4,
            arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.7, shrinkA=4, shrinkB=4),
        )

    # x-axis spans down to below the L0=30 target so the reference line sits
    # inside the plot -- the whole point of this figure is showing how far
    # the swept points are from that target, and a line pinned at the axis
    # edge hides the gap instead of showing it.
    ax.set_xlim(20, max(xs) * 1.6)
    ax.set_ylim(min(ys) - 0.018, 1.028)
    ax.axvline(30, color=t["target"], lw=1.2, ls=":", zorder=1)
    ax.text(32, 1.021, "L0 = 30 (target)", ha="left", va="top",
             fontsize=8.5, color=t["target"], clip_on=True)
    ax.axhline(0.85, color=t["target"], lw=1.2, ls=":", zorder=1)
    # clip_on: EV=0.85 can fall outside the y-range actually observed (every
    # swept point cleared it easily) -- clipping keeps the label from
    # rendering, and inflating the saved bbox, off the bottom of the figure.
    ax.text(0.995, 0.85, "EV = 0.85 (target) ", transform=ax.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8.5, color=t["target"], clip_on=True)

    ax.set_xscale("log")
    ax.set_xlabel("L0  (mean features firing per token, log scale)", fontsize=9, color=t["muted"])
    ax.set_ylabel("Explained variance", fontsize=9, color=t["ink"])
    ax.set_title(
        "l1_coeff sweep: L0 / explained-variance frontier\n"
        "marker size = dead-feature count out of d_sae=12288",
        fontsize=11, color=t["ink"], pad=12, loc="left",
    )
    ax.grid(True, color=t["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)

    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full", choices=["smoke", "full"])
    p.add_argument("--sweep_json", default=None, help="defaults to results/<preset>/l1_sweep.json")
    args = p.parse_args()
    cfg = get_config(args.preset)

    sweep_path = Path(args.sweep_json) if args.sweep_json else cfg.results_dir / args.preset / "l1_sweep.json"
    data = load(sweep_path)

    out_dir = cfg.results_dir / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        plot(data, mode, out_dir / f"l1_sweep_frontier_{mode}.png")

    print("\nl1_coeff   L0        EV        dead   final_step")
    for d in data:
        print(f"{d['l1_coeff']:<10.4g} {d['l0']:<9.1f} {d['explained_variance']:<9.4f} "
              f"{d['dead']:<6} {d['final_step']}")


if __name__ == "__main__":
    main()
