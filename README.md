# SafeNest

[![tests](https://github.com/Abraheem13/safenest/actions/workflows/tests.yml/badge.svg)](https://github.com/Abraheem13/safenest/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Reference implementation of **Nested Policy Learning (NPL)**: middleware that
decides what an unmodified language model may say to a child aged 3 to 17,
based on the child's inferred developmental tier.

Every number, table and figure in the accompanying article is produced by the
code in this repository. None is transcribed by hand.

> Ejaz, R.A.R.; Iradat, F.; Iqbal, W.; Bangash, Y.A.; Kumail, M. *SafeNest: Nested
> Policy Learning for Developmentally Adaptive Child–AI Interaction.* Extended
> version of the UKCI 2026 paper *A Multi-Timescale Safety Architecture for
> Developmentally Adaptive Child–AI Interaction via Nested Policy Learning.*

## What is in it

- **A monotone tier policy**: nine capabilities, five tiers and four access
  levels (Blocked < Socratic < Limited < Available). The policy store rejects
  any edit that breaks monotonicity.
- **A nested policy engine** of five layers running at decreasing
  frequencies. Layers compose by conjunction, so no layer can relax another.
  Crisis disclosures are routed to a referral protocol, never refused on
  severity alone.
- **Tier estimators.** The specification's Bayesian estimator has a proved
  finite-sample misassignment bound. It can be learned from real children's
  language, with or without differential privacy, alongside discriminative
  estimators (logistic regression, gradient boosting, TF-IDF and
  sentence-embedding classifiers).
- **A privacy module** that states, for each of three readings, the privacy
  unit, adjacency, sensitivity and protected output. The three readings are
  architectural minimisation, a corpus-level (ε, δ) release, and local DP on
  the live child.
- **A Socratic engine**: a finite-horizon Markov decision process solved by
  backward induction.
- **A verification suite.** It checks three safety invariants exhaustively,
  over a region-complete abstraction of every response the engine can
  distinguish and over every transition of the cross-session tier-authority
  automaton.

## Install

Python 3.11 or later.

```bash
pip install -e .              # specification, simulations, verification (NumPy only)
pip install -e ".[real]"      # + real-data experiments
python -m spacy download en_core_web_sm
```

## Reproduce

```bash
python -m pytest -q                         # unit tests and the invariant verification
python experiments/run_all.py --synthetic   # Experiments 1-12 (NumPy only, about 2 min)
```

The real-data experiments need the external corpora. None is redistributed
here; see [`data/README.md`](data/README.md) for sources, licences and the
TalkBank login that CHILDES requires.

```bash
python scripts/download_data.py             # PERSUADE 2.0, ASAP, ELLIPSE, checksum-verified
python scripts/prepare_features.py          # parse and featurise every corpus present
python scripts/embed_windows.py written 50  # sentence embeddings for Experiment 13
python experiments/run_all.py --real        # Experiments 13-16
```

Then regenerate the manuscript's tables and figures:

```bash
python experiments/make_tables.py           # tex/tab_*.tex
python experiments/make_figures.py          # figures/*.pdf (+ 600-dpi PNG); needs pdflatex, poppler
```

Each file in `results/` records the master seed (20260806), library versions,
platform and commit, and whether the working tree was clean when it was run.
All experiments except the latency measurements are deterministic given the
seed.

## Experiments

| # | Script | Question |
|---|---|---|
| 1 | `exp01_convergence` | How fast does the specified estimator converge, and in which direction does it err? |
| 2 | `exp02_separability` | Chernoff exponents and the Proposition 1 bound |
| 3 | `exp03_socratic` | Optimal Socratic actions across 80 reward configurations |
| 4 | `exp04_comparative` | Nine frameworks, both ground truths, tuned baselines, bootstrap intervals, margin decomposition |
| 5 | `exp05_privacy` | Corpus-level versus local differential privacy in simulation |
| 6 | `exp06_populations` | Simulated atypical signal profiles and the discordance rule |
| 7 | `exp07_overhead` | Latency of every component, including feature extraction |
| 8 | `exp08_ceiling` | The best DSR any rule on the engine's features could reach |
| 9 | `exp09_operating` | Sweeps of the confidence and severity thresholds |
| 10 | `exp10_bypass` | A graded impersonation adversary against bypass detection |
| 11 | `exp11_reliability` | Seed variance, calibration, estimated tiers, deployment mix |
| 12 | `exp12_ablation` | Component ablation, including both amendments |
| 13 | `exp13_real_estimation` | Tier estimation learned from three essay corpora, leave-one-corpus-out; the prompt confound; Proposition 1 on real text |
| 14 | `exp14_real_privacy` | Private release of parameters learned from real text |
| 15 | `exp15_real_subgroups` | English learners, disability, economic status and gender, within grade |
| 16 | `exp16_real_spoken` | The youngest tiers from children's speech (CHILDES Gillam and ENNI) |

## Layout

| Path | Contents |
|---|---|
| `safenest/tiers.py`, `lattice.py`, `policy.py` | Tiers, the gating policy, the five-layer engine |
| `safenest/verification.py` | Region-complete abstraction of the response space |
| `safenest/signals.py`, `estimator.py` | Simulation signal model, Bayesian estimator, Proposition 1 |
| `safenest/privacy.py` | Privacy readings, sensitivity, analytic Gaussian calibration |
| `safenest/socratic.py` | Socratic MDP |
| `safenest/corpus.py`, `labeling.py`, `baselines.py`, `metrics.py` | Structured policy corpus, ground truths, comparison frameworks, metrics |
| `safenest/data/` | Registry, download and loaders for the real corpora |
| `safenest/features.py` | The five linguistic features, computed from text |
| `safenest/learning.py` | Learned estimators, the two-stage private release, session decisions |
| `experiments/` | Experiments 1–16, the runner, and the table and figure generators |
| `scripts/` | Data download, feature preparation, sentence embeddings |
| `tests/` | Unit tests and the invariant verification |
| `results/`, `tex/`, `figures/` | Recorded results, generated tables, generated figures |

## Scope

The policy evaluation (Experiments 4, 8, 9, 11 and 12) uses structured
records, not natural language. It measures policy decisions against an
author-designed rubric, and the rule-based baselines model documented
behaviour rather than run the shipped systems. The real-data experiments
measure the tier estimator on real children's writing and speech. They do not
measure effects on children. The tier definitions are an engineering proposal
awaiting review by developmental specialists.

## Licence

Code: MIT, see [`LICENSE`](LICENSE). The external corpora keep their own
licences, listed in [`data/README.md`](data/README.md).
