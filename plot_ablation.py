"""Plot the decoder-gradient-projection ablation.

Three panels, one per logged metric, two series each (projection on/off).
Small multiples rather than one chart with two y-axes: L0, explained variance
and MSE have unrelated scales, and a dual-axis chart would let the reader infer
a crossing point that is an artifact of the axis choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config

# Categorical slots 1 and 2 from the validated palette. Fixed assignment:
# blue is always "projection on", orange always "off", in every plot in this
# repo, so colour means the same thing across figures.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
                  grid="#e0e0dc", on="#2a78d6", off="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                 grid="#3a3a38", on="#3987e5", off="#d95926"),
}

PANELS = [
    ("l0", "L0  (features firing per token)", "log"),
    ("explained_variance", "Explained variance", "linear"),
    ("mse", "MSE  (per token, summed over d_model)", "log"),
]


def load(run_dir: Path) -> dict[str, list[float]]:
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return {k: [r[k] for r in rows] for k in rows[0]}


def plot(cfg, run_on: str, run_off: str, mode: str, out: Path) -> None:
    t = THEMES[mode]
    a, b = load(cfg.run_dir(run_on)), load(cfg.run_dir(run_off))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=t["surface"])
    for ax, (key, label, scale) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        ax.plot(a["step"], a[key], color=t["on"], lw=2, label="projection ON", zorder=3)
        ax.plot(b["step"], b[key], color=t["off"], lw=2, label="projection OFF", zorder=3)

        if key == "l0":
            # The target the l1 coefficient was supposed to hit. Drawn because
            # the distance from it is the headline of this run.
            ax.axhline(30, color=t["muted"], lw=1, ls=":", zorder=2)
            ax.text(0.98, 30, " L0 target = 30", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color=t["muted"])
        if key == "explained_variance":
            ax.axhline(0, color=t["grid"], lw=1, zorder=1)
            ax.set_ylim(-1.05, 1.05)  # early steps go to -56; clip to the useful band

        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(label, fontsize=10, color=t["ink"], pad=10, loc="left")
        ax.set_xlabel("step", fontsize=9, color=t["muted"])
        ax.grid(True, color=t["grid"], lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=t["muted"], labelsize=8, length=0)

    # One legend for the figure: identity is stated once, not repeated per panel.
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", frameon=False, fontsize=9, ncol=2,
                     bbox_to_anchor=(0.995, 1.02))
    for txt in leg.get_texts():
        txt.set_color(t["ink"])

    fig.suptitle("Decoder gradient projection ablation", fontsize=12, color=t["ink"],
                 x=0.007, ha="left", y=1.0, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=t["surface"], bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="smoke", choices=["smoke", "full"])
    p.add_argument("--run_on", default="smoke_proj")
    p.add_argument("--run_off", default="smoke_noproj")
    args = p.parse_args()
    cfg = get_config(args.preset)
    out_dir = cfg.results_dir / args.preset
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        plot(cfg, args.run_on, args.run_off, mode, out_dir / f"ablation_projection_{mode}.png")


if __name__ == "__main__":
    main()
