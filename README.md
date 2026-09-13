# SafeNest

Reference implementation of **Nested Policy Learning (NPL)**, middleware that
conditions what an unmodified language model may emit on the inferred
developmental tier of a child user aged 3 to 17.

Every table and figure in the accompanying article is generated from this
repository. Nothing is transcribed by hand.

> Ejaz, R.A.R.; Bangash, Y.A.; Iradat, F.; Kumail, M. *SafeNest: Nested Policy
> Learning for Developmentally Adaptive Child–AI Interaction.* Extended version
> of the UKCI 2026 paper *A Multi-Timescale Safety Architecture for
> Developmentally Adaptive Child–AI Interaction via Nested Policy Learning.*

## What is in it

- a **monotone tier policy**: nine capabilities, five tiers, four access levels
  (Blocked < Socratic < Limited < Available), validated on every policy edit;
- a **nested policy engine** of five layers at decreasing frequencies, composed
  by conjunction so no layer can relax another;
- a **multi-signal Bayesian tier estimator** with a confidence floor, a proved
  finite-sample misassignment bound (Chernoff exponent, not KL) and
  discordance and bypass checks;
- a **privacy module** that fixes the privacy unit, adjacency, sensitivity and
  protected output for three readings: architectural minimisation, a
  corpus-level (ε, δ) release calibrated by the analytic Gaussian mechanism,
  and local DP on the live child;
- a **Socratic Protection Engine**, a finite-horizon MDP solved by backward
  induction;
- a **verification suite** that checks three safety invariants over a
  region-complete abstraction (8,640 responses × 5 tiers), which covers every
  input the engine can distinguish.

## Install

Python 3.11 or later. NumPy is the only runtime dependency.

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python3 -m pytest tests -q            # 96 tests, about 7 s
python3 experiments/run_all.py        # 12 experiments, about 65 s
python3 experiments/make_tables.py    # tex/tab_*.tex
python3 experiments/make_tikz.py      # tex/fig*.tex
```

Results are written to `results/*.json`. Each file records the master seed
(20260806), Python and NumPy versions, platform and commit. All experiments
except the overhead timings are deterministic given the seed.

## Layout

| Path | Contents |
|---|---|
| `safenest/tiers.py` | Tier definitions |
| `safenest/lattice.py` | Feature-gating policy and monotonicity validation |
| `safenest/policy.py` | Layers L0–L4, severity gate, fail-closed degradation |
| `safenest/verification.py` | Region-complete abstraction of the response space |
| `safenest/signals.py` | Signal model, population profiles, corpus release, Chernoff bound |
| `safenest/estimator.py` | Bayesian tier estimator, discordance and bypass checks |
| `safenest/privacy.py` | Privacy readings, sensitivity, analytic Gaussian calibration |
| `safenest/socratic.py` | Socratic MDP and backward induction |
| `safenest/corpus.py` | Seeded 7,000-record evaluation corpus |
| `safenest/labeling.py` | Policy-derived and independent rubric labellers |
| `safenest/baselines.py` | Age-agnostic and age-aware comparison frameworks |
| `safenest/metrics.py` | Developmental safety rate, Wilson intervals, McNemar |
| `experiments/` | Twelve experiment scripts and the table and figure generators |
| `tests/` | Unit tests and the invariant verification |
| `results/` | Experiment outputs with provenance |
| `tex/` | Generated LaTeX tables and TikZ figures |

## Experiments

| Script | Reports |
|---|---|
| `exp01_convergence.py` | Estimator accuracy by interaction count; assignment matrices at n = 1, 3, 10 |
| `exp02_separability.py` | KL divergences; Chernoff exponents; Proposition 1 bound |
| `exp03_socratic.py` | Optimal Socratic actions, trajectories, 80-configuration sweep |
| `exp04_comparative.py` | Nine frameworks under both ground truths; per-category results |
| `exp05_privacy.py` | Corpus-level and local DP against accuracy |
| `exp06_populations.py` | Simulated atypical and second-language cohorts |
| `exp07_overhead.py` | Latency per component; cost of conditional independence |
| `exp08_ceiling.py` | Cross-validated ceiling on the engine's own features |
| `exp09_operating.py` | Sweeps of the confidence and severity thresholds |
| `exp10_bypass.py` | Graded impersonation adversary; detector ROC |
| `exp11_reliability.py` | Seed variance, calibration, Holm–Bonferroni, estimated tiers, deployment mix |
| `exp12_ablation.py` | Component ablation |

## Three results worth reading the code for

**The privacy reading decides the result.** With the live child as the privacy
unit, each modality releases K−1 clipped log-likelihood ratios with Laplace
noise. The per-coordinate signal-to-noise ratio is ε_m / (2(K−1)), independent
of the clip bound, and ten interactions at ε = 1 give 40% five-way accuracy.
With corpus children as the privacy unit, a clipped-mean release with sensitivity
2√2·c·√d / n and analytic Gaussian noise keeps 98.9% accuracy at ε = 1. That
guarantee does not cover the live child. See `safenest/privacy.py`.

**The ground truth decides the margin.** Labels derived from the gating policy
are circular. `labeling.py` provides a rubric built from age, regulatory
thresholds and Piagetian criteria that never reads the policy. The two agree on
77.1% of records (κ = 0.645), and NPL's margin over the strongest age-agnostic
baseline is 36.0 points under the circular labels but 23.0 under the rubric.

**Exhaustive means region-complete.** Every gating layer reads a response only
through threshold predicates, so one representative per interval between
thresholds realises every distinguishable input. `verification.py` builds that
abstraction from the thresholds themselves, and `tests/test_invariants.py`
checks all three invariants on it. An earlier, sampled abstraction missed two
defects that this one found.

## Scope

This is a specification-level implementation evaluated in simulation. The corpus
is structured policy records rather than natural language, so the evaluation
measures policy decisions, not response quality. Signal parameters, tier
boundaries, the rubric and the reward weights are author-specified; no
child-produced text is used, and no calibration to real data is claimed. The
baselines re-implement each system's documented decision rule; they are not the
shipped systems. Results are evidence about the specification, not about
children.

## Licence

MIT. See `LICENSE`.
