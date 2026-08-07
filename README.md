# SafeNest

Reference implementation of **Nested Policy Learning (NPL)** — a middleware
architecture that adapts large language model interaction to the developmental
stage of a child user, aged 3 to 17.

The framework sits between a foundation model and the user. It does not modify
model weights, fine-tuning or inference. It comprises three subsystems:

- a **developmental constraint lattice** over five tiers, derived from Piaget,
  Vygotsky and Erikson, with provable monotonicity;
- a **multi-signal Bayesian age assurance module** with an exponential
  convergence guarantee and an explicitly specified differential-privacy model;
- a **Socratic Protection Engine**, a finite-horizon MDP that replaces direct
  answers with guided questioning calibrated to the Zone of Proximal
  Development.

Four safety invariants are stated in linear temporal logic and verified
mechanically, by exhaustive enumeration where the state space permits and by
randomised search over traces otherwise.

## Install

Python 3.11 or later. NumPy is the only runtime dependency; `pytest` and
`matplotlib` are needed for the tests and figures.

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 -m pytest tests -q          # 81 tests, ~2 s
python3 experiments/run_all.py      # 7 experiments, ~13 s
python3 experiments/make_figures.py # regenerate figures/
python3 experiments/make_latex.py   # regenerate tables/tables.tex
```

Results are written to `results/*.json`. Each file records the master seed,
Python and NumPy versions, platform and commit, so any number can be traced back
to the run that produced it.

## Layout

| Path | Contents |
|---|---|
| `safenest/tiers.py` | Developmental tier definitions |
| `safenest/lattice.py` | Constraint lattice and feature-gating matrix |
| `safenest/privacy.py` | Privacy accounting: unit, adjacency, sensitivity, composition |
| `safenest/signals.py` | Tier-conditional signal models, KL divergences, population profiles |
| `safenest/estimator.py` | Bayesian age assurance, fail-safe, bypass and discordance detection |
| `safenest/policy.py` | Nested Policy Engine layers L0–L4, severity gating, fail-closed degradation |
| `safenest/socratic.py` | Socratic MDP, backward induction, reward sensitivity |
| `safenest/corpus.py` | Seeded 7,000-prompt evaluation corpus |
| `safenest/labeling.py` | Two ground-truth labellers: specification-derived and independent |
| `safenest/baselines.py` | Age-agnostic and age-aware comparison frameworks |
| `safenest/metrics.py` | Developmental Safety Rate, Wilson intervals, McNemar test |
| `experiments/` | Seven experiment scripts plus figure and table generators |
| `tests/` | Test suite, including mechanical verification of the four invariants |
| `results/` | Experiment output, with provenance |
| `figures/` | Figures generated from `results/` |
| `tables/` | LaTeX tables generated from `results/` |

## Experiments

| Script | Reports |
|---|---|
| `exp01_convergence.py` | Estimator accuracy by interaction count; confusion matrix |
| `exp02_separability.py` | KL divergence matrix, D_min, Sanov bound reconciliation |
| `exp03_socratic.py` | Optimal action distribution, trajectories, reward sensitivity sweep |
| `exp04_comparative.py` | Comparative evaluation under both ground truths |
| `exp05_privacy.py` | Privacy/utility trade-off under two privacy models |
| `exp06_populations.py` | Atypical and non-Western population robustness |
| `exp07_overhead.py` | Latency profiling; conditional-independence cost |

## Two findings worth reading the code for

**The privacy model determines the result.** For a clipped per-modality
log-likelihood-ratio release, the per-release signal-to-noise ratio is
`ε_m / (2(K−1))` — independent of the clipping bound, so tightening the clip
attenuates signal and noise equally. At ε = 1.0 with five tiers this is 0.05, and
local differential privacy over the live user's own signals yields near-chance
five-way accuracy. High accuracy at ε = 1.0 is attainable only when ε protects
the calibration corpus rather than the live user. `safenest/privacy.py`
implements both readings and `PrivacyConfig.describe()` emits the exact
statement for each.

**Ground truth determines the margin.** Labels derived from the feature-gating
matrix are circular: the matrix is the specification under test. `labeling.py`
therefore provides a second labeller built from chronological age, regulatory
thresholds and Piagetian criteria, which never reads the matrix. The two agree
on 77% of prompts (Cohen's κ = 0.645), and the measured advantage over baselines
differs substantially between them. Both are reported.

## Scope

This is a specification-level implementation with simulation. The evaluation
corpus consists of structured policy records rather than natural language, so
what is measured is the correctness of policy decisions, not the linguistic
quality of responses. The baselines are reimplementations of each system's
documented decision rule, not the shipped systems. No human subjects are
involved and no child-produced text is used; the tier-conditional distributions
are synthetic, parameterised from published developmental linguistics norms.

Results should be read as evidence about the specification, not about children.

## Licence

MIT. See `LICENSE`.
