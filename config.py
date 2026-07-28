"""Single source of truth for every hyperparameter and path in the project.

Why a config module instead of argparse defaults scattered across scripts:
the core experiment trains 10 SAEs that must differ in *exactly one* controlled
way. If capture.py and train.py could disagree about d_model, or if two runs
silently used different l1_coeff, the reproducibility result would be
meaningless. Centralising means a run is fully described by (preset, seed,
data_seed) and nothing else.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.resolve()


# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
def get_device() -> torch.device:
    """cuda -> mps -> cpu.

    This project runs on two machines: a MacBook (Apple Silicon / MPS) for
    pipeline smoke tests and an RTX 2070 Super (CUDA, 8GB) for the real runs.
    Preferring cuda means the same script picks the fast path on the desktop
    without an env var, and degrades rather than crashes anywhere else.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training.

    torch.manual_seed covers CPU + CUDA + MPS generators in current torch.
    We also seed python/numpy because the dataset streaming and any shuffling
    helper may reach for them. This function is deliberately *not* called
    automatically on import -- train.py calls it at well-defined points so the
    init RNG and the batch-order RNG can be separated (see RunConfig.data_seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Frozen so a run's hyperparameters cannot drift mid-script.

    Mutation is done with `replace(cfg, field=value)`, which produces a new
    object -- this makes any deviation from a preset explicit and greppable
    rather than an invisible in-place assignment.
    """

    # ---- Preset identity -------------------------------------------------
    preset: str = "smoke"

    # ---- Model + hook point ---------------------------------------------
    model_name: str = "gpt2-small"
    # blocks.8 of 12: mid-late. Early layers are still dominated by token
    # identity and positional information; the last few layers have begun
    # collapsing the representation toward the next-token logit direction,
    # which makes features less about "what is being represented" and more
    # about "what comes next". Layer 8 is also where much of the published SAE
    # work on GPT-2 small sits, so the features are comparable to prior results.
    hook_name: str = "blocks.8.hook_resid_pre"
    # resid_pre rather than resid_post: we read the residual stream as it
    # *enters* the block, i.e. the accumulated output of all previous blocks.
    # This is the canonical "what does the model know at layer 8" object.
    d_model: int = 768

    # ---- Data ------------------------------------------------------------
    dataset_name: str = "NeelNanda/pile-10k"
    # Streaming: the Pile shard is large and we only need a fixed prefix of it.
    # Streaming also means the *order* of sequences is deterministic given the
    # dataset, which matters -- the seed experiment requires that all runs see
    # byte-identical activations.
    streaming: bool = True
    n_seqs: int = 5_000
    seq_len: int = 128
    drop_bos: bool = True
    # GPT-2's BOS/<|endoftext|> position carries a hugely outlying activation.
    # Measured at this hook point: ||BOS|| = 3119 vs a mean of 116 for ordinary
    # tokens -- ~27x. It is an attention-sink artifact, not content. Left in, a
    # squared-error loss would be dominated by it and the SAE would burn
    # features reconstructing a single constant vector.

    # ---- SAE architecture ------------------------------------------------
    expansion: int = 16
    # d_sae = 768 * 16 = 12288. This is an *expansion*, not a bottleneck: the
    # premise of superposition is that the model packs more features than it
    # has dimensions, so recovering them requires more slots than d_model.
    # 16x is the standard operating point in the literature -- large enough to
    # unpack, small enough to train on 8GB.

    # ---- Training --------------------------------------------------------
    l1_coeff: float = 5e-4
    # Tuned so L0 (mean number of features firing per token) lands near 30.
    # L0 is the real target; l1_coeff is just the knob that gets us there, and
    # it has to be re-tuned if d_sae or the activation scale changes.
    lr: float = 3e-4
    batch_size: int = 4_096
    # Batch of *activation vectors*, not sequences. 4096 x 12288 fp32 hidden
    # activations is ~200MB, which fits the 2070 Super with room for backward.
    n_steps: int = 3_000
    dead_feature_window: int = 2_000
    # A feature silent for this many steps is flagged dead. Flag only in v1;
    # resampling is deliberately deferred so that the reproducibility numbers
    # are not confounded by a heuristic intervention that itself depends on
    # the seed.
    log_every: int = 100
    ckpt_every: int = 5_000

    # ---- Seeding ---------------------------------------------------------
    seed: int = 0
    # Controls weight initialisation.
    data_seed: int | None = None
    # Controls batch sampling order. None means "follow `seed`".
    #
    # This split is the whole two-arm design:
    #   Arm A -- data_seed fixed, seed varies: are features stable to
    #            initialisation alone?
    #   Arm B -- data_seed None so both vary: are features stable to simply
    #            rerunning the script?
    # The gap between A and B is the fraction of features that are artifacts
    # of what the model saw in what order, rather than of structure in the data.

    # ---- Paths -----------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    ckpt_dir: Path = REPO_ROOT / "checkpoints"
    results_dir: Path = REPO_ROOT / "results"

    # ---- Derived ---------------------------------------------------------
    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion

    @property
    def tokens_per_seq(self) -> int:
        """Activations contributed per sequence, after dropping BOS."""
        return self.seq_len - 1 if self.drop_bos else self.seq_len

    @property
    def n_activations(self) -> int:
        return self.n_seqs * self.tokens_per_seq

    @property
    def n_epochs(self) -> float:
        """How many times training revisits the activation pool.

        Surfaced as a first-class number because it is a known limitation of
        this project: heavy re-epoching over a small pool gives each seed more
        room to memorise noise, which *inflates* the apparent fragility of
        features. The core result has to be read with this number next to it.
        """
        return self.batch_size * self.n_steps / self.n_activations

    @property
    def activation_bytes(self) -> int:
        """On-disk size of the activation cache (fp16)."""
        return self.n_activations * self.d_model * 2

    @property
    def act_dir(self) -> Path:
        """Cache is namespaced by preset -- smoke and full must never mix."""
        return self.data_dir / f"activations_{self.preset}"

    def run_dir(self, run_name: str) -> Path:
        return self.ckpt_dir / self.preset / run_name


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
# SMOKE: MacBook / MPS. This exists only to prove the pipeline runs end to end.
# Nothing is tuned here and no result from it goes in the writeup -- ~1GB of
# activations and ~10 epochs is not enough to say anything about feature
# reproducibility.
SMOKE = Config(
    preset="smoke",
    n_seqs=5_000,
    n_steps=3_000,
)

# FULL: RTX 2070 Super / CUDA. The real 10-run experiment.
# 100k sequences ~= 12.7M activations ~= 19.5GB on disk, which is why this
# never runs on the laptop.
FULL = Config(
    preset="full",
    n_seqs=100_000,
    n_steps=30_000,
)

PRESETS: dict[str, Config] = {"smoke": SMOKE, "full": FULL}


def get_config(preset: str = "smoke", **overrides) -> Config:
    """Fetch a preset, optionally overriding fields.

    Overrides go through dataclasses.replace so the returned object is still
    frozen and still self-consistent.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    cfg = PRESETS[preset]
    return replace(cfg, **overrides) if overrides else cfg


# --------------------------------------------------------------------------
# Write-time shuffle
# --------------------------------------------------------------------------
# Hardcoded, never a CLI flag, never derived from `seed`. capture.py applies
# this one fixed permutation when writing shards, so on-disk order is
# decorrelated from corpus order -- which is what lets a moderate read-time
# buffer still sample across the whole corpus with sequential reads.
#
# It is a *constant* because every run must read the identical on-disk layout.
# If this varied per run it would become a second uncontrolled variable, which
# is precisely what the frozen cache exists to prevent. Batch order still comes
# from `data_seed` at read time; this permutation only decides where rows live.
WRITE_SHUFFLE_SEED = 1234

# Rows held in the read-time shuffle buffer. 1M rows x 768 fp32 = 3.1GB, too
# much for the 8GB laptop; 262144 rows = 768MB and still mixes ~2% of the full
# corpus per buffer on top of an already-decorrelated disk order.
SHUFFLE_BUFFER_ROWS = 262_144


# --------------------------------------------------------------------------
# The 10-run experiment matrix
# --------------------------------------------------------------------------
N_SEEDS = 5
ARM_A_DATA_SEED = 0  # held constant across Arm A so batch order is identical


def experiment_runs(preset: str = "full") -> list[tuple[str, Config]]:
    """(run_name, config) for all 10 runs of the reproducibility experiment.

    Arm A isolates initialisation. Arm B is the honest "rerun the script" test.
    Both arms read the *same* activation cache on disk -- the data itself is
    never regenerated, only the order it is consumed in changes (and in Arm A,
    not even that).
    """
    runs: list[tuple[str, Config]] = []
    for seed in range(N_SEEDS):
        runs.append(
            (f"arm_a_seed{seed}", get_config(preset, seed=seed, data_seed=ARM_A_DATA_SEED))
        )
    for seed in range(N_SEEDS):
        runs.append((f"arm_b_seed{seed}", get_config(preset, seed=seed, data_seed=None)))
    return runs


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


if __name__ == "__main__":
    # `python config.py` prints the budget for both presets. Useful sanity
    # check before committing to a multi-hour capture run.
    print(f"device: {get_device()}")
    for name, cfg in PRESETS.items():
        print(f"\n[{name}]")
        print(f"  d_sae            {cfg.d_sae}")
        print(f"  sequences        {cfg.n_seqs:,} x {cfg.seq_len} tok")
        print(f"  activations      {cfg.n_activations:,}")
        print(f"  cache on disk    {_fmt_bytes(cfg.activation_bytes)}")
        print(f"  train samples    {cfg.batch_size * cfg.n_steps:,}")
        print(f"  epochs over pool {cfg.n_epochs:.1f}")
