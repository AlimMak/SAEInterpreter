# NOTES

Working log. One entry per phase: what I built, why I chose it, what I'd say if
asked in an interview. Kept honest — failures and limitations go in here too.

---

## Phase 1 — Scaffold, config, requirements

**What I built.** Flat repo layout (one file per pipeline stage, matching the
build order in the README), a frozen-dataclass `config.py` holding every
hyperparameter and path, `smoke`/`full` presets for the two machines,
`requirements.txt` with torch excluded, `.gitignore` that excludes `data/` and
`checkpoints/` but keeps `results/`, and a `python config.py` entry point that
prints the compute budget for both presets.

**Why this design choice.** Three decisions carried real weight:

1. *Frozen dataclass, not module-level globals or argparse defaults.* The core
   experiment trains 10 SAEs that must differ in exactly one controlled way. If
   `capture.py` and `train.py` could disagree about `d_model`, or if a field
   could be mutated mid-run, the reproducibility number would be measuring the
   wrong thing. A run is now fully described by `(preset, seed, data_seed)`.

2. *Splitting `seed` from `data_seed`.* Originally the plan was one seed
   controlling both weight init and batch order. That conflates two sources of
   variation. Splitting them gives two arms — A holds batch order fixed and
   varies init only; B varies both — and the *gap between them* is a result in
   itself, not just a robustness check.

3. *Two presets instead of one config I keep editing.* The MacBook run is a
   pipeline test at 5k sequences; the real run is 100k sequences on the 2070.
   Making that a named preset rather than a hand-edit means I can't accidentally
   publish a number from the laptop run, and the activation caches are
   namespaced by preset so they can't mix.

**What I'd say if asked about it in an interview.** "The config is frozen and
the cache is namespaced by preset because the experiment is a controlled
comparison — the whole result rests on ten runs being identical except for one
variable, so I made it structurally hard for them not to be. The thing I'd point
at is the seed split: most people ask 'do features survive a different seed?',
which bundles initialisation and data order together. I separated them, so I can
say how much instability comes from where the model started versus what it saw
in what order."

**Honest notes / limitations logged now, before they become excuses later.**

- `n_epochs` is a first-class property on the config because it's the weakest
  part of the setup. Even `full` re-epochs ~10× over 12.7M activations, against
  the hundreds of millions of unique activations in published work. This isn't
  just a quality problem — it *biases the core result*. Fewer unique activations
  means more room for a seed to memorise noise, which inflates measured
  fragility. So my headline number is a lower bound on stability, and I need to
  say that before a reviewer says it to me.
- Python 3.13 was the system default; the TransformerLens stack doesn't have
  reliable wheels there. Moved to a 3.12 venv rather than debug wheel builds
  four days before applications. Cheap insurance, not a technical insight.
- **Dependency break, found and fixed during setup.** First pin was
  transformer-lens 2.11.0, which import-crashes against transformers 5.x —
  TL 2.x reads `transformers.TRANSFORMERS_CACHE`, which has since been removed.
  Two ways out: pin transformers back below 4.56, or move to TL 3.5.1. Took
  TL 3.5.1 — pinning a dependency backwards to keep an old version of the thing
  that depends on it is how you end up stuck. Both `transformer-lens` and
  `transformers` are now pinned to the exact pair verified to load the model.
- Verified the hook point empirically rather than trusting the config string:
  loaded gpt2-small, confirmed `blocks.8.hook_resid_pre` exists and
  `d_model == 768`. Also **measured the BOS outlier** instead of citing it —
  at this hook point ‖BOS‖ = 3119 against a mean of 116 for ordinary tokens,
  ~27×. That's the concrete number behind the `drop_bos` decision.
- `ckpt_every=5000` means the `smoke` preset (3000 steps) writes no intermediate
  checkpoint. Fine — it only needs the final one — but worth remembering if a
  smoke run ever needs to be resumed.

---

## Phase 2 — Activation capture

**What I built.** `capture.py`: streams pile-10k, cuts documents into
non-overlapping 128-token windows, runs GPT-2 small to layer 8, and writes
fp16 activation shards (~1GB) with row-aligned int32 token IDs and a
`manifest.json`. Smoke preset captured 635,000 rows in one shard.

**Why this design choice.** The three that matter:

1. *Corpus order preserved on disk; shuffling happens at read time.* If the
   shuffle were baked into the shards, on-disk order would fix batch order for
   every run, `data_seed` would have nothing left to vary, and Arm B would
   collapse into Arm A. The experiment would then report a difference of zero
   and it would look like a clean result rather than a dead control.

2. *No padding, no cross-document packing.* Pad activations would become
   training data and the SAE would learn features for our batching. Packing
   unrelated documents into one context would mean late-sequence activations
   are conditioned on unrelated text — fine for LM training, poison for
   interpreting what a feature responds to. Documents under 127 tokens (12% of
   the corpus) are dropped instead.

3. *Alignment asserted, not assumed.* Row *i* of `acts` and row *i* of `toks`
   are written from the same flatten of the same tensor, and a mismatch raises
   rather than writes.

**What I'd say if asked about it in an interview.** "The subtle decision is
that the cache is in corpus order, not shuffled. It looks like the lazy choice
and it's actually the one the experiment depends on — my second arm varies
batch order, so if I'd frozen the order into the files that arm would have
silently measured nothing. Baking a transform into your data is how you delete
a control without noticing."

**What went wrong — the BOS double-prepend.** This is the honest one.

TransformerLens ships its tokenizer with `add_bos_token=True`, which I didn't
check. So `tokenizer(text)` returned `[BOS] + content`, I prepended my own BOS,
and every document's first window was `[BOS, BOS, content...]`. Dropping
position 0 removed one and left the other sitting in the cache as an ordinary
row — carrying the ~27× outlier norm that `drop_bos` exists specifically to
remove.

Everything looked correct: shapes right, row counts exactly `n_seqs × 127`,
manifest consistent, no error anywhere. It surfaced only because I decoded the
token cache back to text as a check and saw a sequence beginning with
`<|endoftext|>`. Roughly 1 row in 800 was affected, but the squared norm ratio
is ~723×, so those rows would have contributed on the order of the entire rest
of the dataset to the reconstruction loss.

Fixed with `add_special_tokens=False`, plus a per-batch assertion that no BOS
survives into the cache. Post-fix row norms: mean 101.6, max 143.2, zero
outliers.

The lesson worth keeping is not "remember this tokenizer flag." It's that a
data bug with correct shapes produces no error signal at all, and the only
thing that caught it was decoding the artifact back into the domain a human can
read. That check now lives in the script.

**Other things verified rather than assumed.**

- *MPS correctness.* TransformerLens warns that torch 2.13 MPS "may produce
  silently incorrect results". Compared MPS against CPU on identical inputs:
  max abs diff 0.00073, min per-row cosine 0.99999982. For scale, fp16 storage
  quantisation at these magnitudes is ~1.47 — the cache's own dtype introduces
  ~2000× more error than the backend does. MPS is not the weak link.
- *Off-by-one in the early-exit.* `stop_at_layer=8` runs blocks [0,8), so the
  hook *inside* block 8 never fires — KeyError. Correct value is 9. Loud
  failure, cheap fix, but worth noting the early-exit optimisation is exactly
  the kind of thing that could have silently returned the wrong layer.
- *Corpus capacity.* pile-10k is 10k documents, but `full` wants 100k
  sequences. Measured: mean 18.5 windows/doc after filtering, so ~185k
  sequences available. 100k is reachable with 1.85× headroom — checked before
  committing to an 18GB capture rather than discovering it 80% through.
- *Model revision pinned* in the manifest (`607a30d7…`) so the cache can be
  regenerated identically on the Windows machine.

**Predicted vs actual.** Config predicted 635,000 rows / 975,360,000 bytes.
Actual: 635,000 rows / 975,360,128 bytes — the 128-byte delta is the `.npy`
header.

**Known limitation.** Because long documents yield many windows each, 5,000
smoke sequences come from only ~270 documents. Fine for a pipeline test,
meaningless for feature diversity — another reason nothing from smoke goes in
the writeup. The `full` preset draws ~5,400 documents.
