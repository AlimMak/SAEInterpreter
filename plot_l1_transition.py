"""Plot L0 and dead-feature count directly against l1_coeff.

Complements plot_l1_sweep.py's L0/EV frontier, which compresses this
project's actual finding: EV barely moves across the swept range (0.97-1.0),
so the frontier plot cannot show the thing that matters here -- that L0 vs
l1_coeff is non-monotonic, with a narrow, well-behaved transition zone
sitting between two failure modes (a runaway dead-feature cascade below it,
an unconverged climb back up above it). This plot puts l1_coeff on the x-axis
directly so that transition is visible.

Reads both the coarse sweep (results/full/l1_sweep.json) and the narrow
follow-up (results/full/l1_sweep_narrow.json) and merges them by l1_coeff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  grid="#e0e0dc", l0="#2a78d6", dead="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 grid="#3a3a38", l0="#3987e5", dead="#d95926"),
}


def load_merged(paths: list[Path]) -> list[dict]:
    by_l1: dict[float, dict] = {}
    for p in paths:
        if not p.exists():
            continue
        for d in json.loads(p.read_text()):
            by_l1[d["l1_coeff"]] = d  # later files win on exact-coeff collision
    if not by_l1:
        raise ValueError(f"none of {paths} exist / contain data")
    return [by_l1[k] for k in sorted(by_l1)]


def plot(data: list[dict], mode: str, out: Path) -> None:
    t = THEMES[mode]
    l1 = [d["l1_coeff"] for d in data]
    l0 = [d["l0"] for d in data]
    dead = [d["dead"] for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, facecolor=t["surface"])
    for ax in (ax1, ax2):
        ax.set_facecolor(t["surface"])
        ax.set_xscale("log")
        ax.grid(True, color=t["grid"], lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=t["muted"], labelsize=8, length=0)

    ax1.plot(l1, l0, color=t["l0"], lw=1.5, marker="o", ms=5, zorder=3)
    ax1.axhline(30, color=t["muted"], lw=1, ls=":", zorder=1)
    ax1.text(l1[0], 30, " L0 = 30 target", va="bottom", ha="left", fontsize=8, color=t["muted"])
    ax1.set_ylabel("L0", fontsize=9, color=t["ink"])
    ax1.set_title(
        "L0 and dead-feature count vs l1_coeff\n"
        "(coarse sweep + narrow follow-up merged, both at 15,000 steps)",
        fontsize=11, color=t["ink"], pad=12, loc="left",
    )

    ax2.plot(l1, dead, color=t["dead"], lw=1.5, marker="o", ms=5, zorder=3)
    ax2.set_ylabel("dead features\n(of 12,288)", fontsize=9, color=t["ink"])
    ax2.set_xlabel("l1_coeff  (log scale)", fontsize=9, color=t["muted"])

    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="full", choices=["smoke", "full"])
    args = p.parse_args()
    cfg = get_config(args.preset)
    res_dir = cfg.results_dir / args.preset

    data = load_merged([res_dir / "l1_sweep.json", res_dir / "l1_sweep_narrow.json"])

    for mode in ("light", "dark"):
        plot(data, mode, res_dir / f"l1_transition_{mode}.png")

    print("\nl1_coeff    L0        dead")
    for d in data:
        print(f"{d['l1_coeff']:<11.4g} {d['l0']:<9.1f} {d['dead']}")


if __name__ == "__main__":
    main()
