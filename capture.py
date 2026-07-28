"""Phase 2 -- cache GPT-2 residual-stream activations to disk.

The seed experiment requires that every one of the ten training runs sees
*byte-identical* data. That is only guaranteed if the activations are computed
once and frozen on disk; recomputing them per run would introduce a second
uncontrolled variable (nondeterministic kernel reductions, dataset streaming
order, tokeniser version) on top of the one we are trying to measure.

Three invariants this file exists to protect:

1. **On-disk order is a fixed permutation, shared by every run.** Rows are
   shuffled at write time under `WRITE_SHUFFLE_SEED`, a hardcoded constant.
   This decorrelates disk order from corpus order, so a moderate read-time
   buffer over *sequential* reads still draws a batch from across the whole
   corpus -- which is what makes training I/O-bound-free without block
   structure leaking into batch composition.

   Critically this is not the *only* shuffle: batch order is still imposed at
   read time from `data_seed` (see data_store.py). A write-time shuffle alone
   would freeze batch order identically for every run and collapse Arm B into
   Arm A. A write-time shuffle *plus* a read-time buffer gives both sequential
   I/O and seed-dependent batches.

2. **Row i of the activation cache is row i of the token cache.** Every
   downstream feature label depends on mapping an activation back to the token
   that produced it. A misalignment here is invisible until Phase 4, where it
   shows up as plausible-looking but wrong labels. The permutation is applied
   to both arrays with the same index array, and alignment is asserted.

3. **The normalisation scalar is computed once and stored.** Every run reads
   the stored value rather than deriving its own; a per-run scalar would be a
   new source of cross-run variation in an experiment whose entire subject is
   cross-run variation.

Two passes:
  pass 1 -- stream the corpus, run the model, route each row to a random
            bucket, append to raw .bin files (all sequential writes)
  pass 2 -- load each bucket, shuffle within it, save as .npy

Random bucket assignment followed by a within-bucket shuffle *is* a uniform
global permutation, and it never needs more than one bucket in RAM -- which is
what makes it work for the 18GB full preset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import WRITE_SHUFFLE_SEED, Config, get_config, get_device

# ~1GB buckets: large enough that the full preset is ~18 files rather than
# thousands, small enough that pass 2 can hold one in RAM to shuffle it.
TARGET_BUCKET_BYTES = 1024**3


def layer_from_hook(hook_name: str) -> int:
    """'blocks.8.hook_resid_pre' -> 8."""
    parts = hook_name.split(".")
    if len(parts) < 2 or parts[0] != "blocks":
        raise ValueError(f"cannot parse layer index from hook name {hook_name!r}")
    return int(parts[1])


def stop_layer_for(hook_name: str) -> int:
    """How many blocks to run before bailing out of the forward pass.

    Blocks above the hook point cannot influence it, and this loop runs for
    hours, so we stop early. TransformerLens `stop_at_layer=n` runs blocks
    [0, n) -- and hook_resid_pre fires *inside* the block it names, so reading
    blocks.8.hook_resid_pre requires n = 9, not 8. (n = 8 raises a KeyError on
    the cache, which is at least a loud failure rather than a wrong tensor.)

    This does run block 8 needlessly. The alternative is to read the identical
    tensor as blocks.7.hook_resid_post and stop at 8, saving ~8% of the forward
    pass -- not worth quietly using a hook name that differs from the one the
    experiment is documented against.
    """
    return layer_from_hook(hook_name) + 1


def model_revision(model_name: str) -> str:
    """Pin the exact HF commit the weights came from."""
    try:
        from huggingface_hub import model_info
        from transformer_lens.loading_from_pretrained import get_official_model_name

        return model_info(get_official_model_name(model_name)).sha
    except Exception as exc:  # network down, API change, offline run
        return f"unknown ({type(exc).__name__})"


def iter_sequences(cfg: Config, tokenizer, bos_id: int):
    """Yield token sequences of exactly `seq_len`, each starting with BOS.

    * **No padding, ever.** A pad token has no meaning in the residual stream
      but still produces an activation, and those activations would become
      training data -- the SAE would learn features for an artifact of our
      batching. Short documents and trailing remainders are dropped instead.

    * **No cross-document packing.** Concatenating the corpus into one stream
      would put unrelated documents in the same context window, so activations
      late in a sequence would be conditioned on text from a different
      document. Fine for LM training, bad for interpreting a feature.
    """
    content_len = cfg.seq_len - 1  # one slot reserved for BOS
    ds = load_dataset(cfg.dataset_name, split="train", streaming=cfg.streaming)

    for doc in ds:
        # add_special_tokens=False is load-bearing, not defensive tidiness.
        # TransformerLens ships its tokenizer with add_bos_token=True, so the
        # default call returns [BOS] + content. Prepending our own BOS then
        # produced [BOS, BOS, content...]; dropping position 0 removed one and
        # left the second sitting in the cache as a normal row -- with a norm
        # ~27x everything around it, i.e. exactly the outlier drop_bos exists
        # to exclude. It survives one full round of eyeballing because the
        # shapes are all correct.
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        n_windows = len(ids) // content_len  # trailing remainder dropped
        for w in range(n_windows):
            yield [bos_id] + ids[w * content_len : (w + 1) * content_len]


class BucketWriter:
    """Pass 1: route rows to random buckets, appending to raw .bin files.

    Appending raw bytes rather than .npy because bucket sizes are not known in
    advance under random routing. Pass 2 converts to .npy.
    """

    def __init__(self, out_dir: Path, cfg: Config, n_buckets: int, rng: np.random.Generator):
        self.out_dir = out_dir
        self.cfg = cfg
        self.n_buckets = n_buckets
        self.rng = rng
        self.act_f = [(out_dir / f"_raw_acts_{i:05d}.bin").open("wb") for i in range(n_buckets)]
        self.tok_f = [(out_dir / f"_raw_toks_{i:05d}.bin").open("wb") for i in range(n_buckets)]
        self.counts = np.zeros(n_buckets, dtype=np.int64)

    def add(self, acts: np.ndarray, toks: np.ndarray) -> None:
        assert acts.shape[0] == toks.shape[0]
        # Uniform random bucket per row. Combined with the within-bucket
        # shuffle in pass 2, this is a uniform global permutation.
        assign = self.rng.integers(0, self.n_buckets, size=acts.shape[0])
        for b in range(self.n_buckets):
            m = assign == b
            if not m.any():
                continue
            acts[m].tofile(self.act_f[b])
            toks[m].tofile(self.tok_f[b])
            self.counts[b] += int(m.sum())

    def close(self) -> None:
        for f in self.act_f + self.tok_f:
            f.close()


def finalize_buckets(out_dir: Path, cfg: Config, counts: np.ndarray,
                     rng: np.random.Generator) -> list[int]:
    """Pass 2: shuffle within each bucket and write .npy shards."""
    shard_rows: list[int] = []
    for b in tqdm(range(len(counts)), desc="pass 2 (shuffle)", unit="shard"):
        n = int(counts[b])
        acts = np.fromfile(out_dir / f"_raw_acts_{b:05d}.bin", dtype=np.float16).reshape(n, cfg.d_model)
        toks = np.fromfile(out_dir / f"_raw_toks_{b:05d}.bin", dtype=np.int32)

        if acts.shape[0] != toks.shape[0]:
            raise RuntimeError(
                f"bucket {b}: {acts.shape[0]} activation rows vs {toks.shape[0]} token rows; "
                f"refusing to write a misaligned cache"
            )

        # One index array applied to both -- this is what preserves alignment
        # through the permutation.
        perm = rng.permutation(n)
        np.save(out_dir / f"acts_{b:05d}.npy", acts[perm])
        np.save(out_dir / f"toks_{b:05d}.npy", toks[perm])
        shard_rows.append(n)

        (out_dir / f"_raw_acts_{b:05d}.bin").unlink()
        (out_dir / f"_raw_toks_{b:05d}.bin").unlink()
        del acts, toks, perm
    return shard_rows


def capture(cfg: Config, fwd_batch: int, overwrite: bool) -> None:
    out_dir = cfg.act_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            sys.exit(
                f"{out_dir} already contains a cache. Re-running would change the data "
                f"underneath any checkpoints trained on it. Pass --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device={device}  preset={cfg.preset}")
    print(f"target: {cfg.n_seqs:,} seqs x {cfg.seq_len} tok "
          f"-> {cfg.n_activations:,} rows, ~{cfg.activation_bytes / 1024**3:.1f}GB")

    model = HookedTransformer.from_pretrained(cfg.model_name, device=str(device))
    model.eval()
    tokenizer = model.tokenizer
    bos_id = tokenizer.bos_token_id
    stop_layer = stop_layer_for(cfg.hook_name)

    n_buckets = max(1, int(np.ceil(cfg.activation_bytes / TARGET_BUCKET_BYTES)))
    rng = np.random.default_rng(WRITE_SHUFFLE_SEED)
    writer = BucketWriter(out_dir, cfg, n_buckets, rng)
    print(f"buckets: {n_buckets} (~{cfg.activation_bytes / n_buckets / 1024**3:.2f}GB each)")
    print(f"write-time shuffle seed: {WRITE_SHUFFLE_SEED} (fixed for all runs)")

    # Running accumulators for the normalisation scalar. Computed over the
    # whole cache in one pass so it is a property of the *data*, not of any run.
    norm_sum, norm_count = 0.0, 0

    seq_iter = iter_sequences(cfg, tokenizer, bos_id)
    n_done = 0
    batch: list[list[int]] = []
    exhausted = False

    pbar = tqdm(total=cfg.n_seqs, desc="pass 1 (capture)", unit="seq")
    with torch.no_grad():
        while n_done < cfg.n_seqs:
            batch.clear()
            while len(batch) < min(fwd_batch, cfg.n_seqs - n_done):
                try:
                    batch.append(next(seq_iter))
                except StopIteration:
                    exhausted = True
                    break
            if not batch:
                break

            tokens = torch.tensor(batch, dtype=torch.long, device=device)
            _, cache = model.run_with_cache(
                tokens, names_filter=cfg.hook_name, stop_at_layer=stop_layer
            )
            acts = cache[cfg.hook_name]  # [batch, seq_len, d_model]

            if cfg.drop_bos:
                acts = acts[:, 1:, :]
                tok_out = tokens[:, 1:]
            else:
                tok_out = tokens

            flat = acts.reshape(-1, cfg.d_model)
            norm_sum += flat.norm(dim=-1).sum().item()
            norm_count += flat.shape[0]

            acts_np = flat.to(torch.float16).cpu().numpy()
            toks_np = tok_out.reshape(-1).to(torch.int32).cpu().numpy()

            # A BOS surviving into the cache means BOS handling is wrong
            # somewhere upstream, and the consequence is a handful of rows with
            # ~27x the norm of everything else quietly dominating the MSE.
            if cfg.drop_bos:
                n_bos = int((toks_np == bos_id).sum())
                if n_bos:
                    raise RuntimeError(
                        f"{n_bos} BOS token(s) survived into the cache after dropping "
                        f"position 0 -- BOS handling is wrong; refusing to write."
                    )

            writer.add(acts_np, toks_np)
            n_done += len(batch)
            pbar.update(len(batch))
            if exhausted:
                break
    pbar.close()
    writer.close()

    if n_done < cfg.n_seqs:
        print(
            f"\nWARNING: corpus exhausted after {n_done:,} sequences, "
            f"short of the requested {cfg.n_seqs:,}.\n"
            f"         Actual epoch count will be higher than config predicts."
        )

    shard_rows = finalize_buckets(out_dir, cfg, writer.counts, rng)

    # ---- normalisation scalar -------------------------------------------
    mean_norm = norm_sum / norm_count
    target_norm = float(np.sqrt(cfg.d_model))
    scale = target_norm / mean_norm
    # Why a single global scalar is safe here specifically: it is a uniform
    # dilation. It cannot rotate anything, so every decoder direction and every
    # cross-seed cosine similarity -- the quantities the whole experiment is
    # built on -- is unchanged by it. It only moves the activations onto the
    # scale that l1_coeff is defined against, so the sparsity penalty is
    # comparable with published values instead of being a rounding error.

    manifest = {
        "preset": cfg.preset,
        "model_name": cfg.model_name,
        "model_revision": model_revision(cfg.model_name),
        "hook_name": cfg.hook_name,
        "dataset_name": cfg.dataset_name,
        "d_model": cfg.d_model,
        "seq_len": cfg.seq_len,
        "drop_bos": cfg.drop_bos,
        "rows_per_seq": cfg.tokens_per_seq,
        "n_seqs_requested": cfg.n_seqs,
        "n_seqs_captured": n_done,
        "total_rows": int(sum(shard_rows)),
        "n_shards": len(shard_rows),
        "shard_rows": shard_rows,
        "act_dtype": "float16",
        "tok_dtype": "int32",
        "shuffled_on_disk": True,
        "write_shuffle_seed": WRITE_SHUFFLE_SEED,
        # Read by every run; never recomputed per run.
        "mean_activation_norm": mean_norm,
        "norm_scale": scale,
        "norm_target": target_norm,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {sum(shard_rows):,} rows across {len(shard_rows)} shard(s)")
    print(f"mean ||x|| = {mean_norm:.2f} -> norm_scale = {scale:.6f} (target {target_norm:.2f})")
    print(f"manifest: {out_dir / 'manifest.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="smoke", choices=["smoke", "full"])
    p.add_argument(
        "--fwd_batch",
        type=int,
        default=32,
        help="sequences per model forward pass (memory knob only; does not "
             "affect the contents of the cache)",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    capture(get_config(args.preset), args.fwd_batch, args.overwrite)


if __name__ == "__main__":
    main()
