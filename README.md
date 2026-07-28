# Are SAE features real, or artifacts of the seed?

Sparse autoencoders trained on GPT-2 small's residual stream, plus a
reproducibility audit of the features they find.

Most SAE projects train one autoencoder, find an attractive feature, and stop.
This one asks the question that comes next: **if you retrain with a different
random seed, do you get the same features back?** And then the question after
that: **does a feature's reproducibility predict whether you can actually steer
the model with it?**

A feature that vanishes when you change the seed is not a fact about GPT-2. It
is a fact about that training run.

## The experiment

Ten SAEs are trained on one byte-identical cache of activations, in two arms:

| Arm | `seed` (init) | `data_seed` (batch order) | Question it answers |
|-----|---------------|---------------------------|---------------------|
| A   | 0–4           | fixed at 0                | Are features stable to **initialisation**? |
| B   | 0–4           | follows `seed`            | Are features stable to **rerunning the script**? |

Features are matched across runs by maximum cosine similarity of their decoder
directions. A feature's *reproducibility score* is how consistently it finds a
high-similarity partner in the other runs of its arm.

The gap between Arm A and Arm B is the interesting number: it isolates how much
of the instability comes from **what the model saw in what order**, as opposed
to where it started.

## Install

Python **3.12**. Not 3.13 — the TransformerLens dependency stack does not have
reliable wheels there yet.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

torch installs separately per platform, because the CUDA and Apple Silicon
builds come from different indexes:

```bash
# macOS / Apple Silicon (MPS)
pip install torch

# Windows / RTX 2070 Super (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then the shared dependencies:

```bash
pip install -r requirements.txt
```

Verify device selection and print the compute budget for both presets:

```bash
python config.py
```

## Presets

The project runs on two machines with very different budgets, so there are two
configs and the difference between them is deliberate.

| | `smoke` (MacBook / MPS) | `full` (RTX 2070 Super / CUDA) |
|---|---|---|
| sequences | 5,000 | 100,000 |
| activations | 635k | 12.7M |
| cache on disk | ~0.9 GB | ~18.2 GB |
| train steps | 3,000 | 30,000 |
| epochs over pool | ~19 | ~9.7 |

**`smoke` is a pipeline test, not an experiment.** It exists to prove the code
runs end to end before committing hours of GPU time. Nothing is tuned on it and
no number from it appears in the writeup.

## Running it

```bash
python capture.py  --preset full                          # Phase 2
python train.py    --preset full --run_name arm_a_seed0 \
                   --seed 0 --data_seed 0                 # Phase 3
python analyze.py  --preset full --run_name arm_a_seed0   # Phase 4
python reproducibility.py --preset full                   # Phase 5
python steer.py    --preset full                          # Phase 6
python correlate.py --preset full                         # Phase 7
```

## Design notes

**Hook point — `blocks.8.hook_resid_pre`.** Layer 8 of 12. Early layers are
still dominated by token identity and position; the last layers have begun
collapsing toward the next-token logit direction, which makes their features
more about *what comes next* than *what is represented*. Layer 8 is also where
much published GPT-2-small SAE work sits, so results are comparable.

**Expansion, not compression — `d_sae = 768 × 16 = 12288`.** The superposition
hypothesis says the model packs more features than it has dimensions. Recovering
them requires more slots than `d_model`, not fewer.

**BOS is dropped.** Measured at this hook point, `‖BOS‖ = 3119` against a mean
of `116` for ordinary tokens — about 27×. It's an attention-sink artifact, not
content. Left in, a squared-error loss is dominated by it and the SAE spends
features reconstructing one constant vector.

**Activations are cached to disk, not regenerated.** The experiment's validity
depends on every run seeing identical data. Recomputing per run would introduce
a second uncontrolled variable.

**No SAELens or other prebuilt SAE library.** The autoencoder, the decoder
normalisation, and the gradient projection are written from scratch in `sae.py`.

## Known limitations

Named here rather than left for a reviewer to find:

- **Epoch count.** Even the `full` preset revisits its activation pool ~10
  times, against the hundreds of millions of *unique* activations used in
  published work. This biases the core result in a specific direction: fewer
  unique activations means more room for a seed to memorise noise, which
  **inflates** the measured fragility. The reproducibility numbers should be
  read as a lower bound on stability.
- **Dead features are flagged, not resampled.** Resampling is a seed-dependent
  heuristic intervention; adding it in v1 would confound the thing being
  measured.
- **Matching is greedy max-cosine**, which is not a bijection — two features in
  one run can claim the same partner in another. Reported alongside the score.

## Status

- [x] Phase 1 — scaffold, config, requirements
- [x] Phase 2 — activation capture
- [ ] Phase 3 — SAE + training
- [ ] Phase 4 — max-activating examples + auto-labelling
- [ ] Phase 5 — cross-seed reproducibility (core experiment)
- [ ] Phase 6 — steering
- [ ] Phase 7 — reproducibility vs steering success
- [ ] Phase 8 — feature browser + writeup

Working notes and per-phase design rationale: [NOTES.md](NOTES.md)
