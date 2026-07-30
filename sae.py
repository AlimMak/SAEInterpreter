"""Phase 3 -- the sparse autoencoder.

Written from scratch rather than pulled from SAELens: the decoder
normalisation and the gradient projection below are the two places where an
SAE differs from a textbook autoencoder, and they are the two things worth
being able to explain.

Architecture (L1 variant, SparseAutoencoder):
    f    = ReLU((x - b_dec) @ W_enc + b_enc)      encode
    xhat = f @ W_dec + b_dec                      decode
    loss = ||x - xhat||^2 + l1_coeff * ||f||_1

TopKSparseAutoencoder below is the second variant: the l1_coeff sweep in
NOTES.md found no coefficient, across two orders of magnitude, that reaches
L0~30 without either a runaway dead-feature cascade or non-convergence. L1
penalises sum(|f_i|), which cannot distinguish 30 large activations from 1000+
small ones at the same total budget -- it induces a *magnitude* target, not a
*count* target, and L0 was only ever a hoped-for side effect. TopK (Gao et al.,
2024, "Scaling and evaluating sparse autoencoders") makes L0 == k a structural
property of the forward pass instead: keep the k largest pre-activations per
token, zero the rest. No coefficient to tune, no activation shrinkage, and
every run in the reproducibility matrix gets identical L0 by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # W_dec rows are the feature directions -- the objects the whole
        # experiment compares across seeds. Initialised as random unit vectors.
        W_dec = torch.randn(d_sae, d_model, dtype=dtype)
        W_dec /= W_dec.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        # Tied-transpose init: the encoder starts as the decoder's transpose.
        # Not a tie during training (they diverge immediately) -- just a
        # starting point where every encoder row already points along its own
        # decoder direction, so features begin able to detect what they
        # reconstruct instead of having to discover the pairing from noise.
        self.W_enc = nn.Parameter(W_dec.t().clone())

        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype))
        # b_dec is set from the data by init_b_dec_from_data before training.
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=dtype))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_b_dec_from_data(self, sample: torch.Tensor) -> None:
        """Initialise b_dec to the mean activation.

        GPT-2's residual stream has a large constant offset -- the mean
        activation is nowhere near the origin. Starting b_dec there means the
        features only ever have to explain *deviation* from the corpus mean,
        which is the part that carries information. Left at zero, the first
        few thousand steps are spent with every feature collectively
        reconstructing one constant vector.

        (Anthropic use the geometric median; the mean is a cheaper
        approximation and b_dec is a free parameter that trains anyway, so the
        init only needs to be in the right neighbourhood.)
        """
        self.b_dec.data = sample.mean(0).to(self.b_dec.dtype)

    def encode_preacts(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation, before the nonlinearity. Shared by encode() and
        TopKSparseAutoencoder.encode(), which apply different nonlinearities
        to the same linear map."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Subtracting b_dec before the encoder re-centres the input. Because
        # the decoder adds b_dec back, this makes the autoencoder operate in
        # residual-from-offset space, and keeps the encoder pre-activations
        # from being dominated by a constant the features cannot represent.
        return torch.relu(self.encode_preacts(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    # ------------------------------------------------------------------
    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Force every decoder row back to unit norm.

        This is what makes the L1 penalty mean something. Without it the model
        has a free win available: halve every feature activation, double every
        decoder vector, and the reconstruction is unchanged while ||f||_1 has
        halved. The sparsity penalty would fall to near zero without a single
        feature actually turning off. Fixing ||W_dec_i|| = 1 removes that
        escape route, so the only way to reduce L1 is to genuinely stop firing.
        """
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Strip the component of W_dec.grad parallel to each decoder row.

        Called after backward() and *before* optimizer.step().

        Why the parallel component is garbage: normalize_decoder() runs after
        every step and rescales each row back to unit norm, which deletes any
        movement along the row's own direction. So the parallel component of
        the gradient produces no net parameter change -- it is guaranteed
        wasted before it is ever computed.

        Why it must be removed *before* the optimizer rather than after:
        Adam does not just use the gradient for direction, it accumulates it
        into the second-moment estimate v. A parallel component that changes
        nothing still enters v, inflates it, and therefore shrinks the
        effective learning rate (grad / sqrt(v)) applied to the perpendicular
        component -- the part that actually moves the feature. Projecting the
        *update* after Adam would fix the step direction but leave v polluted
        by a quantity known in advance to be meaningless, and that pollution
        persists for many steps through the moving average.

        What this does NOT do: make the update exactly tangential. Adam
        rescales element-wise by sqrt(v), which is not a rotation, so a
        gradient that is perpendicular going in generally is not perpendicular
        coming out. Momentum from earlier steps has the same effect. The
        projection reduces waste and keeps the moments clean; it does not
        replace normalize_decoder(), which is still required after every step.
        """
        if self.W_dec.grad is None:
            return
        g = self.W_dec.grad
        w = self.W_dec.data
        # Row-wise <g, w> w, with ||w|| = 1 so no denominator is needed.
        parallel = (g * w).sum(dim=-1, keepdim=True) * w
        g -= parallel


class TopKSparseAutoencoder(SparseAutoencoder):
    """L0 == k by construction: keep the k largest pre-activations per token,
    zero the rest. No L1 term -- see the module docstring for why L1 was
    dropped rather than retuned again.

    Everything else (decoder unit-norm, gradient projection, b_dec/b_enc,
    dead-feature tracking) is unchanged; only the nonlinearity differs.
    """

    def __init__(self, d_model: int, d_sae: int, k: int, dtype: torch.dtype = torch.float32):
        super().__init__(d_model, d_sae, dtype=dtype)
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        preacts = self.encode_preacts(x)
        topk_vals, topk_idx = preacts.topk(self.k, dim=-1)
        # A pre-activation can rank in the top k while still negative (e.g.
        # early in training, or a token where fewer than k features have a
        # positive case for firing). Clamping to zero keeps "at most k
        # nonzero, all positive" -- the ReLU contract -- without changing k
        # for tokens where every top-k pre-activation is already positive.
        topk_vals = torch.relu(topk_vals)
        f = torch.zeros_like(preacts)
        f.scatter_(-1, topk_idx, topk_vals)
        return f


# ----------------------------------------------------------------------
# Metrics. Only three are logged, per the project spec: L0, explained
# variance, MSE. Anything else is a distraction from the question being asked.
# ----------------------------------------------------------------------
def compute_metrics(x: torch.Tensor, xhat: torch.Tensor, f: torch.Tensor) -> dict[str, float]:
    # MSE summed over d_model, averaged over the batch: the per-token
    # reconstruction error, in the same units as ||x||^2 so it can be compared
    # against activation norms directly.
    mse = ((xhat - x) ** 2).sum(-1).mean()

    # L0 = mean number of features firing per token. This is the real sparsity
    # target (~30); l1_coeff is only the knob used to reach it.
    l0 = (f > 0).float().sum(-1).mean()

    # Fraction of variance explained, measured against the *batch mean* as the
    # baseline rather than against zero. Against zero, the large constant
    # offset in GPT-2's residual stream would make even a useless model that
    # predicts the mean look like it explains ~99% of the variance.
    resid_var = ((x - xhat) ** 2).sum()
    total_var = ((x - x.mean(0)) ** 2).sum()
    ev = 1.0 - resid_var / total_var

    return {"mse": mse.item(), "l0": l0.item(), "explained_variance": ev.item()}
