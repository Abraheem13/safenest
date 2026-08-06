# SafeNest — Nested Policy Learning reference implementation

Runnable implementation of the architecture specified in *SafeNest: Nested Policy
Learning for Multi-Timescale Safety in Child-AI Interaction*, written so that
every quantitative claim in the manuscript is regenerated from code rather than
transcribed.

Start with **[REVISION_NOTES.md](REVISION_NOTES.md)** — it maps each reviewer
comment to what the code shows and what has to change in the paper.

## Quick start

```bash
cd ~/Desktop/safenest && python3 -m pytest tests -q
```

```bash
cd ~/Desktop/safenest && python3 experiments/run_all.py
```

The full suite takes about 15 seconds on a laptop. Results land in `results/*.json`,
each with provenance (timestamp, Python and NumPy versions, platform, git commit,
master seed).

## Layout

| Path | Contents |
|---|---|
| `safenest/tiers.py` | Tier definitions (Table 2) |
| `safenest/lattice.py` | Constraint lattice and feature-gating matrix (Eq. 1–3, Table 7) |
| `safenest/privacy.py` | DP accounting: privacy unit, adjacency, sensitivity, composition |
| `safenest/signals.py` | Tier-conditional signal models (Table 3), KL divergences, population profiles |
| `safenest/estimator.py` | Bayesian age assurance (Eq. 6–11), fail-safe, bypass and discordance detection |
| `safenest/policy.py` | Nested Policy Engine L0–L4 (Eq. 12), fail-closed degradation |
| `safenest/socratic.py` | Socratic MDP (Eq. 14–16), backward induction, reward sensitivity |
| `safenest/corpus.py` | The 7,000-prompt evaluation corpus, seeded |
| `safenest/labeling.py` | Two ground-truth labellers: circular and independent |
| `safenest/baselines.py` | Age-agnostic and age-aware comparison frameworks |
| `safenest/metrics.py` | DSR definition, Wilson intervals, McNemar |

## Experiments

| Script | Regenerates |
|---|---|
| `exp01_convergence.py` | Table 8, Figure 6 |
| `exp02_separability.py` | Tables 4 and 9, plus the D_min reconciliation |
| `exp03_socratic.py` | Tables 10 and 11, plus reward-weight sensitivity |
| `exp04_comparative.py` | Tables 14, 15, 16 under both ground truths |
| `exp05_privacy.py` | Table 12 and the formal privacy statement |
| `exp06_populations.py` | Neurodivergent and non-WEIRD analyses (new) |
| `exp07_overhead.py` | Table 13, plus the conditional-independence cost |

## Tests

81 tests. The ones that matter for the paper's claims are in
`tests/test_invariants.py`: Invariant I is checked over every response/tier pair,
Invariant II over 2,000 random traces including tier upgrades, and Invariant IV
by exhaustive analysis of all 15 single- and pairwise-layer failure combinations.

## Scope

This is a specification-level implementation. The evaluation corpus is structured
policy records, not natural language, and the baselines are reimplementations of
each system's documented decision rule rather than the shipped systems. No human
subjects are involved. Results should be described as simulation throughout.

## Requirements

Python 3.11+, `numpy`, `pytest`. No other dependencies.
