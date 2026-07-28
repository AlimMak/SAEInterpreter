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
