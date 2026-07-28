"""Loss recovered -- does the reconstruction preserve what GPT-2 actually uses?

Explained variance answers a question about geometry: how much of the
activation vector's L2 magnitude did we reproduce? That is the quantity the
training loss optimises, so it is close to guaranteed to look good, and it is
not the question anyone cares about. A reconstruction can capture 95% of the
variance while destroying a low-magnitude direction the network reads heavily,
and L2 will never notice -- variance is dominated by the largest directions,
and importance to the model is not.

Loss recovered asks the model instead. Splice the SAE's reconstruction into the
residual stream in place of the true activation and measure the damage to
next-token cross-entropy, normalised against a floor:

    recovered = (CE_zero - CE_sae) / (CE_zero - CE_clean)

  CE_clean  model untouched
  CE_sae    activation replaced by the SAE reconstruction
  CE_zero   activation zero-ablated -- the "reconstruction that carries no
            information" baseline

1.0 means the reconstruction is as good as the real activation; 0.0 means it is
worth no more than deleting the layer's contribution entirely. Normalising
against CE_zero rather than reporting raw CE matters because raw CE degradation
is uninterpretable on its own -- you cannot tell whether +0.3 nats is
catastrophic or trivial without knowing what total destruction costs.

The eval texts are drawn from the same corpus but are *held out* by construction
(taken from the tail of pile-10k, past the documents any preset consumes), so
this is not scored on the activations the SAE trained on.
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

from config import Config

# Drawn from the tail of pile-10k. The full preset consumes ~5,400 of the
# 10,000 documents from the front, so starting here keeps eval text unseen.
EVAL_DOC_START = 9_000


def build_eval_batch(cfg: Config, model: HookedTransformer, n_seqs: int = 32) -> torch.Tensor:
    """Fixed set of held-out token sequences. Deterministic -- no seed involved,
    so every run is scored on identical text.

    Cached to disk: reaching document 9,000 of a *streaming* dataset means
    pulling the preceding 9,000, which cost ~3 minutes of dead time at the
    start of every run. Ten runs would pay it ten times for a byte-identical
    result.
    """
    cache = cfg.data_dir / f"eval_tokens_{n_seqs}x{cfg.seq_len}.npy"
    if cache.exists():
        return torch.from_numpy(np.load(cache))

    ds = load_dataset(cfg.dataset_name, split="train", streaming=True)
    content_len = cfg.seq_len - 1
    bos = model.tokenizer.bos_token_id
    seqs: list[list[int]] = []
    for i, doc in enumerate(ds):
        if i < EVAL_DOC_START:
            continue
        ids = model.tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < content_len:
            continue
        seqs.append([bos] + ids[:content_len])
        if len(seqs) == n_seqs:
            break

    arr = np.array(seqs, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return torch.from_numpy(arr)


@torch.no_grad()
def loss_recovered(
    sae,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    norm_scale: float,
) -> dict[str, float]:
    """CE under three conditions, and the normalised recovery score."""
    device = next(sae.parameters()).device
    tokens = tokens.to(device)

    def ce(hook_fn=None) -> float:
        hooks = [(hook_name, hook_fn)] if hook_fn else []
        loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=hooks)
        return loss.item()

    def splice(act, hook):
        # The SAE was trained on normalised activations, so scale in, and scale
        # the reconstruction back out before handing it to the model. Skipping
        # either half would make the metric meaningless while still producing a
        # plausible-looking number.
        shape = act.shape
        flat = act.reshape(-1, shape[-1]).to(sae.W_enc.dtype) * norm_scale
        recon, _ = sae(flat)
        return (recon / norm_scale).reshape(shape).to(act.dtype)

    def zero(act, hook):
        # Zero-ablate the *whole* residual stream at this point, which is the
        # honest floor: it is what the model scores with no information from
        # this hook at all.
        return torch.zeros_like(act)

    ce_clean = ce()
    ce_sae = ce(splice)
    ce_zero = ce(zero)

    denom = ce_zero - ce_clean
    recovered = (ce_zero - ce_sae) / denom if abs(denom) > 1e-6 else float("nan")
    return {
        "ce_clean": ce_clean,
        "ce_sae": ce_sae,
        "ce_zero": ce_zero,
        "loss_recovered": recovered,
    }
