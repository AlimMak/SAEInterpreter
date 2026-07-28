"""Phase 2 -- cache GPT-2 residual-stream activations to disk.

The seed experiment requires that every one of the ten training runs sees
*byte-identical* data. That is only guaranteed if the activations are computed
once and frozen on disk; recomputing them per run would introduce a second
uncontrolled variable (nondeterministic kernel reductions, dataset streaming
order, tokeniser version) on top of the one we are trying to measure.

Two invariants this file exists to protect:

1. **Corpus order is preserved on disk.** Shards are written in the order the
   documents arrive. Batch order is imposed later, at *read* time, by sampling
   indices with an RNG seeded from `data_seed`. If the shuffle were baked into
   the files, on-disk order would fix batch order for every run and Arm B
   would silently collapse into Arm A -- the experiment would report a
   difference of zero and look like a clean result.

2. **Row i of the activation cache is row i of the token cache.** Every
   downstream feature label depends on mapping an activation back to the token
   that produced it. A misalignment here is invisible until Phase 4, where it
   shows up as plausible-looking but wrong labels. So the alignment is asserted
   per shard and the run dies rather than writing a corrupt cache.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import Config, get_config, get_device

# ~1GB shards: large enough that the full preset is ~18 files rather than
# thousands, small enough to memory-map cheaply and to lose little work if a
# capture is interrupted partway.
TARGET_SHARD_BYTES = 1024**3


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
    """Pin the exact HF commit the weights came from.

    'gpt2-small' is a moving target in principle -- recording the commit sha
    means the cache can be regenerated identically on the other machine, which
    is the whole point of freezing the data.
    """
    try:
        from huggingface_hub import model_info

        # TransformerLens maps its own alias onto the HF repo id.
        from transformer_lens.loading_from_pretrained import get_official_model_name

        return model_info(get_official_model_name(model_name)).sha
    except Exception as exc:  # network down, API change, offline run
        return f"unknown ({type(exc).__name__})"


def iter_sequences(cfg: Config, tokenizer, bos_id: int):
    """Yield token sequences of exactly `seq_len`, each starting with BOS.

    Documents are tokenised and cut into non-overlapping windows of
    `seq_len - 1` content tokens, with BOS prepended to each window. Two
    deliberate choices:

    * **No padding, ever.** A pad token has no meaning in the residual stream,
      but it still produces an activation, and those activations would become
      training data -- the SAE would learn features for an artifact of our
      batching. Documents (and trailing remainders) shorter than one full
      window are dropped instead.

    * **No cross-document packing.** The usual LM trick of concatenating the
      corpus into one stream and slicing it would put unrelated documents in
      the same context window, so activations late in a sequence would be
      conditioned on text from a different document. That is fine for training
      a language model and bad for interpreting what a feature responds to.
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
            window = ids[w * content_len : (w + 1) * content_len]
            yield [bos_id] + window


class ShardWriter:
    """Buffers rows and flushes ~1GB activation/token shard pairs to disk."""

    def __init__(self, out_dir: Path, cfg: Config):
        self.out_dir = out_dir
        self.cfg = cfg
        # Keep shards a whole number of sequences so a sequence is never split
        # across two files -- Phase 4 needs to show a token in its context.
        rows_per_seq = cfg.tokens_per_seq
        bytes_per_row = cfg.d_model * 2  # fp16
        seqs_per_shard = max(1, (TARGET_SHARD_BYTES // bytes_per_row) // rows_per_seq)
        self.rows_per_shard = seqs_per_shard * rows_per_seq

        self.acts = np.empty((self.rows_per_shard, cfg.d_model), dtype=np.float16)
        self.toks = np.empty(self.rows_per_shard, dtype=np.int32)
        self.fill = 0
        self.shard_idx = 0
        self.shard_rows: list[int] = []

    def add(self, acts: np.ndarray, toks: np.ndarray) -> None:
        assert acts.shape[0] == toks.shape[0], (
            f"row-count mismatch before buffering: {acts.shape[0]} acts vs {toks.shape[0]} toks"
        )
        pos = 0
        n = acts.shape[0]
        while pos < n:
            take = min(self.rows_per_shard - self.fill, n - pos)
            self.acts[self.fill : self.fill + take] = acts[pos : pos + take]
            self.toks[self.fill : self.fill + take] = toks[pos : pos + take]
            self.fill += take
            pos += take
            if self.fill == self.rows_per_shard:
                self.flush()

    def flush(self) -> None:
        if self.fill == 0:
            return
        acts = self.acts[: self.fill]
        toks = self.toks[: self.fill]

        # The alignment guarantee, checked rather than assumed. A mismatch here
        # would produce a cache that trains fine and mislabels every feature.
        if acts.shape[0] != toks.shape[0]:
            raise RuntimeError(
                f"shard {self.shard_idx}: activation rows ({acts.shape[0]}) != "
                f"token rows ({toks.shape[0]}); refusing to write a misaligned cache"
            )

        np.save(self.out_dir / f"acts_{self.shard_idx:05d}.npy", acts)
        np.save(self.out_dir / f"toks_{self.shard_idx:05d}.npy", toks)
        self.shard_rows.append(int(acts.shape[0]))
        self.shard_idx += 1
        self.fill = 0

    @property
    def total_rows(self) -> int:
        return sum(self.shard_rows)


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

    writer = ShardWriter(out_dir, cfg)
    print(f"shard size: {writer.rows_per_shard:,} rows "
          f"({writer.rows_per_shard * cfg.d_model * 2 / 1024**3:.2f}GB)")

    seq_iter = iter_sequences(cfg, tokenizer, bos_id)
    n_done = 0
    batch: list[list[int]] = []
    exhausted = False

    pbar = tqdm(total=cfg.n_seqs, unit="seq")
    with torch.no_grad():
        while n_done < cfg.n_seqs:
            # ---- gather one forward batch --------------------------------
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

            # names_filter + stop_at_layer: cache exactly one tensor and skip
            # every block above the hook point.
            _, cache = model.run_with_cache(
                tokens,
                names_filter=cfg.hook_name,
                stop_at_layer=stop_layer,
            )
            acts = cache[cfg.hook_name]  # [batch, seq_len, d_model]

            if cfg.drop_bos:
                acts = acts[:, 1:, :]
                tok_out = tokens[:, 1:]
            else:
                tok_out = tokens

            # Flatten [batch, pos, d_model] -> [batch*pos, d_model]. Both
            # tensors are flattened with the same row-major ordering, which is
            # what makes row i correspond in both files.
            acts_np = acts.reshape(-1, cfg.d_model).to(torch.float16).cpu().numpy()
            toks_np = tok_out.reshape(-1).to(torch.int32).cpu().numpy()

            # A BOS surviving into the cache means BOS handling is wrong
            # somewhere upstream, and the consequence is a handful of rows with
            # ~27x the norm of everything else quietly dominating the MSE.
            # Cheap to check per batch; expensive to discover in Phase 5.
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
    writer.flush()

    if n_done < cfg.n_seqs:
        # Not fatal, but it changes the compute budget, so it must be loud and
        # it must end up in the manifest rather than only in a scrollback buffer.
        print(
            f"\nWARNING: corpus exhausted after {n_done:,} sequences, "
            f"short of the requested {cfg.n_seqs:,}.\n"
            f"         {cfg.dataset_name} did not contain enough text at "
            f"seq_len={cfg.seq_len}. Actual epoch count will be higher than "
            f"config predicts."
        )

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
        "total_rows": writer.total_rows,
        "n_shards": len(writer.shard_rows),
        "shard_rows": writer.shard_rows,
        "rows_per_shard_nominal": writer.rows_per_shard,
        "act_dtype": "float16",
        "tok_dtype": "int32",
        "shuffled_on_disk": False,  # asserted for the reader's benefit
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {writer.total_rows:,} rows across {len(writer.shard_rows)} shard(s)")
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
