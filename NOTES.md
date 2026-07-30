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

---

## Phase 3 — SAE, sampler, training loop

**What I built.** `sae.py` (the autoencoder, decoder renormalisation, gradient
projection), `data_store.py` (read-time shuffle buffer), `train.py`,
`evaluate.py` + `eval_checkpoint.py` (loss recovered), `test_sampler.py`,
`plot_ablation.py`.

**Why this design choice — the gradient projection.** Projected after
`backward()` and before `optimizer.step()`, not applied to the update
afterwards. The parallel component of the decoder gradient is deleted by
renormalisation every step, so it produces no movement — it is known to be
useless before it is computed. But Adam does not only use the gradient for
direction: it accumulates it into the second-moment estimate `v`. A parallel
component that moves nothing still inflates `v` and therefore shrinks the
effective learning rate `grad/sqrt(v)` on the perpendicular component that does
the work. Projecting the update post-Adam fixes the step direction but leaves
`v` polluted, and that pollution persists through the moving average for many
steps. It does *not* make the update tangential — Adam rescales element-wise,
which is not a rotation — so renormalisation after every step is still
required. The projection reduces waste; it does not replace normalisation.

**Why this design choice — two shuffles doing different jobs.** A fixed
write-time permutation (`WRITE_SHUFFLE_SEED = 1234`, hardcoded, identical for
all ten runs) decorrelates disk order from corpus order. A `data_seed`-driven
read-time shuffle buffer then decides batch composition. Neither alone works:
write-time only would freeze batch order identically for every run and collapse
Arm B into Arm A; read-time only over a full random permutation is what caused
the I/O problem below.

**Evidence the experiment survived the sampler change** (the number to quote):

| | Arm B batch overlap | chance | ratio |
|---|---|---|---|
| smoke (buffer = 41% of cache) | 43.2 / 4096 | 26.4 | 1.63× |
| full (buffer = 2.1% of cache) | 0.0 / 4096 | 1.3 | 0.00× |

Smoke sits above chance only because the buffer covers 41% of a small cache, so
two streams' buffers overlap heavily. At full scale the buffer is 2% of the
cache and overlap falls to zero. Arm A stays byte-identical across different
init seeds even when torch's and numpy's global RNGs are deliberately disturbed
first.

**Why this design choice — loss recovered.** Explained variance measures the
quantity the loss optimises, so it is nearly guaranteed to look good, and it is
not the question anyone cares about. Variance is dominated by the largest
directions; importance to the network is not. Loss recovered splices the
reconstruction into the residual stream and measures CE damage normalised
against zero-ablation: `(CE_zero - CE_sae) / (CE_zero - CE_clean)`. Normalising
matters because raw CE degradation is uninterpretable — +0.3 nats means nothing
without knowing what total destruction costs. First reading, at step 0 of an
untrained SAE: CE_clean 3.59, CE_sae 5.30, CE_zero 12.05 → 0.798 recovered.
A randomly-initialised autoencoder already "recovers 80% of the loss," which is
the clearest possible argument for why this metric needs its floor stated.

**What went wrong, in order.**

1. *`l1_coeff = 5e-4` was ~3 orders of magnitude too weak.* At step 600 of the
   first run the sparsity term was **0.36% of the loss** and L0 was 3,544
   against a target of 30. Cause: 5e-4 is the standard value for activations
   normalised so `E‖x‖ = √d_model ≈ 27.7`; raw layer-8 activations have
   `‖x‖ ≈ 101.8`. MSE scales with `‖x‖²`, L1 with `‖x‖`. Fixed by storing one
   global scalar (`norm_scale = 0.272329`) in the manifest at capture time, read
   by every run and never recomputed per run. A uniform scalar is a dilation: it
   cannot rotate anything, so decoder directions and every cross-seed cosine
   similarity — the quantities this project measures — are untouched. Asserted
   in the test: min cosine between raw and scaled rows is 0.99999976.

2. *My first l1 sweep was worthless, and the lesson is the useful part.* L0
   moved 2082 → 2101 across an 80× change in `l1_coeff` and I briefly read that
   as a flat sparsity response. It was not. 400 steps at batch 2048 is far too
   early: L0 starts near `d_sae/2` (half the features fire at init, measured
   6,141 of 12,288) and takes thousands of steps to come down. **Any sparsity
   measurement taken before L0 plateaus is measuring initialisation dynamics,
   not the l1 response.** Sweep only after locating the plateau, and locate it
   with one long run rather than several short ones.

3. *Killed the first real run at step 600 of 3000 to diagnose, which threw away
   the converged number I needed.* L0 was still falling monotonically when I
   stopped it. Diagnosing early was right; stopping the only run producing the
   answer was not.

4. *A full random permutation over the memmap was I/O-suicide.* 4096 scattered
   reads per batch, each row 1536 bytes — smaller than a page — so a batch
   touched ~4096 pages: 400ms cold versus 6ms warm. Projected 3.3h of pure I/O
   per run on the 18GB preset, ~33h across ten runs with the GPU idle. This is
   what motivated the two-shuffle scheme.

5. *The overlap test was measuring the corpus, not the sampler.* It compared
   batches by hashing row bytes, and **0.81% of rows in the cache are exact
   byte-duplicates** of another row — repeated boilerplate in pile-10k produces
   identical activations. That adds ~33 false matches per 4096-row batch and
   made an at-chance overlap read as 2× chance. Fixed by threading true global
   indices through the buffer. The sampler was fine; the ruler was wrong. Same
   shape of error as the BOS bug: a plausible number with no error raised.

6. *Then a memory failure that stopped training without failing.* The buffer
   held float32 and the shuffle was `buf = buf[p]`, which allocates a second
   full buffer for the copy — ~1.5GB peak on a machine with 8.6GB total. With
   GPT-2 also resident for the in-loop eval, the process went to **0.1% CPU and
   14MB resident** with 9.9GB in swap. It was not crashed, not erroring, just
   swapped out and making no progress. Fixed three ways: buffer kept in the
   cache's native fp16 (384MB not 768MB), the shuffle permutes an index array
   instead of the data so the buffer is allocated once and mutated in place, and
   loss recovered moved to a separate process (`eval_checkpoint.py`) so GPT-2 is
   never resident during training.

**Machine constraint worth stating plainly.** This laptop has 8.6GB of unified
memory, shared between CPU and GPU, and runs with ~59MB of free pages. Measured
step time is linear in batch size (4096→1.18s, 2048→0.59s, 1024→0.33s,
512→0.20s) and at batch 512 works out to ~185 GFLOPS, an order of magnitude
under what the GPU should manage — so smoke is memory-bound, not compute-bound,
and the fix is the RTX box rather than more tuning here. Reinforces the original
decision that smoke is a pipeline test and nothing more.

**What I'd say if asked about it in an interview.** "The two failures worth
talking about are both the same failure. The BOS bug and the batch-overlap bug
both produced correct-looking numbers with no error raised, and in both cases
what caught them was checking the measurement against something independent —
decoding tokens back to text, and switching from content-hashing to true
indices. The l1 mistake is the other kind: I measured a real quantity at the
wrong time. L0 starts at half of d_sae because half the features fire at
initialisation, and it takes thousands of steps to come down, so a 400-step
sweep tells you about initialisation and nothing about your sparsity penalty."

---

## Phase 3.5 — Landing the Mac session's work on the RTX box, and the real-scale check

The commit above (normalization, two-shuffle sampler, loss recovered) was done
on the Mac and pushed, but this machine's `main` had never been pulled, so at
the start of this session it still looked like Phase 3 raw. Nearly redid it
from scratch before `git push` bounced with "fetch first" and surfaced the
real history. Lesson: `git fetch` before reconstructing anything a past
session might already have shipped.

Ran it here to confirm it isn't Mac-specific: `capture.py --preset smoke`
reproduced `mean ||x|| = 101.76 -> norm_scale = 0.272329` exactly, and
`test_sampler.py` reproduced the same Arm B overlap (`mean 43.2, max 52 vs
26.4 expected`) byte-for-byte. Cross-machine determinism holds.

**The number that actually matters — full preset, RTX 2070 Super, 12.7M rows:**

```
buffer: 262,144 rows (2.1% of cache)
Arm B batch-0 overlap at chance  -- 0/4096 rows vs 1.3 expected by chance
overlap stays at chance across steps  -- mean 0.0, max 0 vs 1.3 expected
```

Matches the Mac's own full-scale number (`0.0/4096 vs 1.3 chance`) exactly.
At this scale the buffer is only 2.1% of the cache, so the chance-level
expectation itself is under 2 rows per 4096-row batch — a run of zeros here
isn't a sign of anything, it's what a Poisson mean of 1.3 looks like most of
the time. The write-time shuffle plus read-time buffer costs the experiment
nothing at the scale the ten real runs actually read.

Added `--preset` to `test_sampler.py` (defaults to `smoke`) rather than
writing a second script, since the checks that matter are identical — only
the scale changes, and the scale is exactly the thing chance depends on.

---

## Phase 4 — L1 could not reach the sparsity target; switched to TopK

**What happened.** `arm_a_seed0` (the first of the ten planned reproducibility
runs, `l1_coeff=5e-4`) plateaued at `L0=1288`, `EV=0.999`. `L0 > d_model=768`
with `d_sae=12288` means any 1288-feature set can span the 768-dim activation
space and reconstruct near-perfectly with no sparse decomposition at all —
`EV=0.999` was measuring spanning coverage, not quality. Stopped the other
nine runs before launching them.

**Sweep, not a single retune.** Located the true plateau on `arm_a_seed0`
first (rolling-slope of L0 flattens to noise by step ~13,000) rather than
repeating an earlier mistake (see Phase 3, item 2) of reading pre-plateau
dynamics as the l1 response. Swept `l1_coeff` log-spaced 2e-3 to 2e-1 (6
points, 15,000 steps each — above the plateau floor), then narrowed to
0.013–0.03 to map the transition precisely (5 of 6 points; the 6th was
stopped by an external process kill mid-run, not a bug, and wasn't rerun —
the trend from the other ten points was already unambiguous).

**Result: non-monotonic, two failure regimes, no viable point anywhere in a
100x range.**

| l1_coeff | L0 | dead |
|---|---|---|
| 0.002 | 1221 | 246 |
| 0.005 | 1114 | 6134 (50%, still accelerating) |
| 0.0126 | 1095 | 1106 |
| 0.0154 | 1099 | 204 (stable) |
| 0.0182 | 1122 | 23 (stable) |
| 0.0215 | 1226 | 1 |
| 0.0254 | 1601 | 0 |
| 0.032–0.2 | 1487–2482 | 0 |

Below ~0.013: a runaway dead-feature cascade (l1=0.005 killed 50% of all
12,288 features and was still accelerating at step 15,000). Above ~0.02: zero
dead features, but L0 never converges — it climbs back up, ending denser than
the low-l1 side. Across the whole range, the lowest L0 ever reached was ~1095
— 36x the L0≈30 target. No coefficient fixes this; the floor is flat.

**Diagnosed rather than guessed at why, in order:**

1. *Decoder-norm inflation (the standard "SAE escapes L1 by growing decoder
   norms" failure) — ruled out by direct measurement.* `W_dec` row norms were
   exactly 1.0 (0/12288 outside 1±1e-4) at both l1 extremes.
   `normalize_decoder()` is unconditional, every step, immediately after
   `opt.step()` — confirmed from the source, not paraphrased. L0 was also
   stable across six count-thresholds spanning 0 to 1e-2 (1465→1311 at the
   loosest, not an order-of-magnitude swing), so the ~1200-1465 firing
   features are real, not sub-threshold noise being counted as "firing."

2. *Loss/l1 bookkeeping — exact.* `logged loss == mse + l1_coeff * l1` to
   float32 precision at both extremes; `l1` is exactly `f.abs().sum(-1).mean()`
   used in the backward pass, not a different statistic.

3. *"Optimizer failure" (b_enc stuck near zero, never learns to gate
   features off) — looked plausible, then a loss-arithmetic correction ruled
   it out.* The first-pass argument ("a hypothetical L0=30 solution is ~3x
   cheaper under the exact loss being optimized") assumed *perfect*
   reconstruction at L0=30. Redone honestly using the EV≈0.85 that published
   SAEs actually achieve at L0≈30 (so mse≈115, not ≈0): the dense solution
   (loss≈2.61 at l1=0.002) beats the sparse one (loss≈115.3) by ~40x. The
   optimizer is correctly minimizing the loss it was given; that loss does not
   prefer the sparse solution once real reconstruction cost is accounted for.
   b_enc staying near zero is a symptom of that, not an independent bug.

4. *Mutual-incoherence (L1 can't tell concentrated mass from spread mass once
   the decoder dictionary lets features add constructively) — mixed evidence,
   not a clean confirmation.* Population-level pairwise cosine similarity
   between decoder rows sat at the random-unit-vector baseline (mean|cos|
   0.030 vs baseline sd 0.036, signed mean ≈0) at both l1 extremes — no
   detectable learned correlation in the dictionary as a whole. The realized
   reconstruction did show a real ~2x amplification over the pure-orthogonal
   prediction at the high-l1 end (and a *below*-1.0, mildly destructive
   ratio at the low-l1 end) — real, but smaller than an initial back-of-
   envelope estimate suggested, and not the clean smoking gun the hypothesis
   predicted.

**The one point that doesn't depend on resolving (4) precisely:** L1
penalizes `sum(|f_i|)` — by construction, it cannot distinguish 30 features at
magnitude ~5 from 1000+ features at magnitude ~0.15 when both sum to the same
budget. L0 was always a *hoped-for side effect* of that penalty, never a
constraint. No coefficient across a 100x sweep produced it.

**Fix: TopK (Gao et al., 2024 — "Scaling and evaluating sparse autoencoders").**
Added `TopKSparseAutoencoder` in `sae.py` alongside (not replacing) the L1
`SparseAutoencoder` — keep the k largest pre-activations per token, zero the
rest, no L1 term. L0 becomes a structural property of the forward pass, not
something a coefficient has to induce. `Config.topk: int | None` gates it in
`train.py`; `None` (default) leaves every existing config/run untouched.

**One confirmation run, `full` preset, seed=0/data_seed=0 (same as every
other run in this investigation), k=30, 15,000 steps:**

```
L0 = 29.998   EV = 0.8674   mse = 59.25   dead = 2 / 12288
loss_recovered = 0.9273   (ce_clean 3.591, ce_sae 4.205, ce_zero 12.050)
wall clock: 40.2 min
```

L0 lands at exactly k from step 0 (not tuned to get there). EV clears the
0.85 target and is now a meaningful number, since L0=30 is well under
`d_model=768`. Dead features: 2, against 204–6134 for any L1 point that
wasn't still climbing. Loss recovered: 0.927, against a randomly-initialized
floor of ~0.80 (Phase 3) — first time in this project a sparsity metric and a
functional (loss-recovered) metric have agreed.

**Kept, not deleted:** the L1 implementation (`SparseAutoencoder` in
`sae.py`), both sweep scripts (`sweep_l1.py`, the narrow-sweep invocation),
both result files (`results/full/l1_sweep.json`,
`results/full/l1_sweep_narrow.json`), and the frontier/transition plots. The
sweep is a result — a documented negative one — not dead code.

**Still open.** The ten-run reproducibility matrix (`config.experiment_runs`)
has not been launched. It should use `topk=30`, not `l1_coeff`, once cleared
to run.

---

## Phase 5 — the 10-run matrix, and a reproducibility analysis that needed a rework

**Convergence check before launching.** On `topk_k30`: EV slope over the last
3000 steps ≈0.0006/1000 steps (flat, 31-point regression). `loss_recovered`
only has 4 logged points (`eval_every=5000`) so a true 3000-step rolling slope
isn't available at that resolution, but the available points (0.924 → 0.937 →
0.927) move up then down, not a sustained climb. 15,000 steps stands.

**Launch.** `run_experiment.py`: Arm A (seeds 0-4, `data_seed=0` pinned) and
Arm B (seeds 0-4, `data_seed` following seed), `topk=30`, `n_steps=15000`,
`eval_every=5000` — identical to the confirmation run. Re-ran the Arm A/B
batch-overlap sampler check first and logged it
(`results/full/sampler_check_topk_experiment.log`) so the arm-separation
invariant is on record for this exact config, not just the original sampler
validation from Phase 3. All 10 completed in 3.55h (faster than the ~6-7h
estimate — likely warm disk cache after the first run). Every run landed
`L0=30.0`, `EV≈0.86-0.87`, `dead≤5`, consistent with the confirmation run.

**First-pass analysis (superseded below) — two problems found on review.**
`reproducibility.py` matched decoder rows across same-arm runs with Hungarian
assignment (`scipy.optimize.linear_sum_assignment`, not greedy — greedy's
collision rate against a random dictionary measured 67.4%, i.e. two-thirds of
"best matches" were claimed by more than one feature), calibrated a threshold
against a random-dictionary null (99th percentile = 0.1715), and reported the
fraction of each feature's 4 same-arm peers that cleared it. Result: Arm A
mean=0.984, Arm B mean=0.984, both median=1.000. Two things wrong with
reporting that as the headline:

1. The threshold (0.1715) was far more permissive than real matched
   similarities (mean 0.69) — "clears the null" and "is the same feature"
   are different claims, and collapsing to one pass/fail cutoff hid the gap
   between them.
2. Both arms saturated near the ceiling (median exactly 1.000) at that
   threshold, so the arm comparison carried no information — a ceiling
   effect, not evidence that the arms don't differ. The original writeup of
   this phase said "data order adds no extra instability"; that claim did
   not survive review and has been withdrawn.

**Reworked analysis — three threshold-light views on the same underlying
per-feature, per-peer matched-similarity data** (saved raw, uncollapsed, in
`results/full/repro_raw_sims/*.npy` this time, so a future rework doesn't
require re-matching):

1. *Distribution, no threshold.* Arm A: mean=0.6910, median=0.7416,
   p10/p25/p75/p90 = 0.345/0.540/0.876/0.945. Arm B: mean=0.6900,
   median=0.7395, p10/p25/p75/p90 = 0.345/0.542/0.874/0.942. Null:
   mean=0.135, median=0.134. Real matches sit far from null in both arms
   (visible in `repro_distribution_{light,dark}.png` as two heavily-
   overlapping broad curves well clear of a sharp null peak near 0.13-0.15)
   — but Arm A and Arm B are themselves nearly indistinguishable. Two-sample
   KS test: statistic=0.0066, p=5.08e-05. The p-value is small only because
   n≈240,000 gives enormous power to detect even a trivial effect; the
   statistic itself (max CDF gap under 1%) says the practical difference is
   negligible. **Read this as: a real but tiny distributional difference,
   not "no difference" and not "a meaningful difference" — the test is
   underpowered for the second claim and overpowered for the first.**
2. *Fraction clearing threshold against all 4 peers, swept 0.17→0.92 step
   0.05* (`repro_threshold_curve_{light,dark}.png`). The two arms track
   within ~1 percentage point of each other at every threshold in the sweep;
   null is ~0.0000 throughout (a random-dictionary decoder essentially never
   clears even the loosest threshold against all 4 fake peers
   simultaneously). Explicit callouts (computed directly, not read off the
   swept grid — 0.5/0.7/0.9 don't land on a 0.05-step grid starting at 0.17,
   and a first version of this script silently never printed them because of
   exactly that mismatch): at θ=0.5, Arm A=0.5605 vs Arm B=0.5613
   (diff −0.0008); at θ=0.7, Arm A=0.3005 vs Arm B=0.2935 (diff +0.0070); at
   θ=0.9, Arm A=0.0688 vs Arm B=0.0676 (diff +0.0012).
3. *Strict set (θ=0.9 against all 4 peers) — the defensible "these are real"
   set.* Arm A: 4141/60188 (6.88%). Arm B: 4069/60226 (6.76%). Saved per-run,
   as original `d_sae` indices, in `results/full/repro_strict_0.9_features.json`
   — this is the number and the feature set a downstream steering phase
   should use, not the 98.4%-at-a-permissive-threshold figure from the first
   pass.

**Honest bottom line on the arm comparison: inconclusive, not null.** Arm A
and Arm B's matched-similarity distributions are close enough that this
experiment cannot support a claim that varying data order adds meaningfully
more feature instability than varying initialisation alone — but a KS test
run on ~240k paired samples did detect a statistically real (if tiny)
difference, so "the arms are identical" is also not a claim this data
supports. The honest statement is: whatever gap exists between the arms is
small enough that this experiment, at this scale, cannot resolve it either
way.

**Config bug found and fixed while writing this up: `arm_a_seed0` and
`arm_b_seed0` were the exact same run.** Arm B's `data_seed` followed `seed`
(`None`), and `ARM_A_DATA_SEED=0` — so at `seed=0` both arms resolved to
`seed=0, data_seed=0`, an identical config. Fixed in `config.py` with
`ARM_B_DATA_SEED_OFFSET=100` (`data_seed = seed + 100` for Arm B, so B's
data_seeds are 100-104, disjoint from A's pinned 0 and from each other).
**The 10-run results above predate this fix and are unaffected by it** — the
reproducibility analysis only ever compares within an arm (each run against
its 4 same-arm peers), never across arms, so the seed=0 collision was never
exercised. It would only have mattered for a cross-arm comparison, which
this analysis doesn't do. Any future cross-arm work should use the fixed
config (or re-run `arm_b_seed0` under the new `data_seed=100`).
