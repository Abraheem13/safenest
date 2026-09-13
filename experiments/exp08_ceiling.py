"""Experiment 08 -- the achievable ceiling on the Developmental Safety Rate.

The comparative evaluation reports how far NPL is from a ground truth. It does
not report how far *any* policy could get. Those are different questions, and
without the second the first cannot be interpreted: an error rate is only a
defect if some reachable policy would have avoided it.

This experiment computes the best DSR attainable by any decision rule that sees
exactly what the Nested Policy Engine sees -- capability, tier, directness,
academic status and assessed harm severity -- and nothing else. The gap between
that ceiling and NPL's score is the part of the error that is a policy defect;
the remainder is information the architecture cannot observe, principally the
child's chronological age within the tier band.

The ceiling is estimated by cross-validation rather than in-sample, because a
majority-label rule fitted and scored on the same prompts is optimistically
biased. Both figures are reported so the size of that bias is visible.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.baselines import make_npl  # noqa: E402
from safenest.corpus import build_corpus  # noqa: E402
from safenest.labeling import rubric_label  # noqa: E402
from safenest.metrics import evaluate_framework  # noqa: E402
from safenest.tiers import ALL_TIERS  # noqa: E402

N_REPEATS = 5
N_FOLDS = 5
#: Harm severity is continuous; the engine thresholds it. Binning at this
#: resolution keeps the ceiling honest -- a rule allowed unlimited resolution on
#: a continuous feature could memorise individual prompts.
SEVERITY_BINS = 10


def observable(prompt) -> tuple:
    """Exactly the fields `baselines.make_npl` passes to the policy engine."""
    return (
        prompt.capability,
        prompt.tier,
        prompt.is_direct_request,
        prompt.is_academic,
        int(prompt.harm_severity * SEVERITY_BINS),
    )


def _in_sample(prompts, labels, keyfn) -> float:
    groups: dict = defaultdict(Counter)
    for p in prompts:
        groups[keyfn(p)][labels[p.idx]] += 1
    return sum(c.most_common(1)[0][1] for c in groups.values()) / len(prompts)


def _cross_validated(prompts, labels, keyfn, rng) -> tuple[float, float]:
    scores = []
    for _ in range(N_REPEATS):
        order = rng.permutation(len(prompts))
        folds = np.array_split(order, N_FOLDS)
        hits = 0
        for k in range(N_FOLDS):
            held = set(folds[k].tolist())
            groups: dict = defaultdict(Counter)
            for i in order:
                if i not in held:
                    p = prompts[i]
                    groups[keyfn(p)][labels[p.idx]] += 1
            rule = {key: c.most_common(1)[0][0] for key, c in groups.items()}
            for i in folds[k]:
                p = prompts[i]
                # A cell never seen in training counts as a miss, so the
                # ceiling is not inflated by cells the rule cannot decide.
                hits += rule.get(keyfn(p)) == labels[p.idx]
        scores.append(hits / len(prompts))
    return float(np.mean(scores)), float(np.std(scores))


def run() -> dict:
    rng = rng_for("ceiling")
    prompts = build_corpus()
    labels = {p.idx: rubric_label(p) for p in prompts}
    npl = make_npl()

    achieved = evaluate_framework(npl, prompts, rubric_label)
    npl_dsr = achieved["overall"]["dsr"]

    ceiling, sd = _cross_validated(prompts, labels, observable, rng)
    ceiling_in = _in_sample(prompts, labels, observable)

    # Note: extending the key with age or abstraction demand fragments the
    # cell space faster than 7,000 prompts can populate it, so the resulting
    # cross-validated figure falls rather than rises. That is an artefact of
    # the estimator, not a property of the architecture, and is therefore not
    # reported. Quantifying the value of unobserved features needs a smoothed
    # model rather than a per-cell majority rule.

    print("Achievable DSR on the engine's own observable features")
    print(f"  NPL as specified                    {pct(npl_dsr)}%")
    print(f"  Ceiling, {N_REPEATS}x{N_FOLDS}-fold cross-validated  "
          f"{pct(ceiling)}% (sd {100 * sd:.2f})")
    print(f"  Ceiling, in-sample (biased)         {pct(ceiling_in)}%")
    print(f"  -> fixable policy defect            {100 * (ceiling - npl_dsr):.1f} pp")
    print(f"  -> irreducible, unobservable        {100 * (1 - ceiling):.1f} pp")

    # Where the recoverable error sits.
    rows = []
    by_cell = defaultdict(lambda: {"n": 0, "npl": 0, "best": Counter()})
    decisions = {p.idx: npl(p, p.tier) for p in prompts}
    from safenest.labeling import decision_to_label

    for p in prompts:
        cell = by_cell[(p.tier.label, p.category.value)]
        cell["n"] += 1
        cell["npl"] += decision_to_label(decisions[p.idx]) == labels[p.idx]
        cell["best"][labels[p.idx]] += 1
    losses = []
    for (tier, cat), c in by_cell.items():
        best = c["best"].most_common(1)[0][1]
        losses.append({
            "tier": tier, "category": cat,
            "npl": 100 * c["npl"] / c["n"],
            "ceiling": 100 * best / c["n"],
            "gap_pp": 100 * (best - c["npl"]) / c["n"],
            "prompts_lost": best - c["npl"],
        })
    losses.sort(key=lambda r: -r["prompts_lost"])
    table(
        [{"Tier": r["tier"], "Risk category": r["category"],
          "NPL (%)": f"{r['npl']:.1f}", "Ceiling (%)": f"{r['ceiling']:.1f}",
          "Gap (pp)": f"{r['gap_pp']:+.1f}"} for r in losses[:8]],
        ["Tier", "Risk category", "NPL (%)", "Ceiling (%)", "Gap (pp)"],
        "Where the recoverable error sits -- eight worst cells",
    )
    recoverable = sum(r["prompts_lost"] for r in losses)
    print(f"\n  Recoverable prompts concentrated in the top 3 cells: "
          f"{100 * sum(r['prompts_lost'] for r in losses[:3]) / max(recoverable, 1):.0f}%")

    return {
        "npl_dsr": npl_dsr,
        "ceiling_cv": ceiling,
        "ceiling_cv_sd": sd,
        "ceiling_in_sample": ceiling_in,
        "fixable_headroom_pp": 100 * (ceiling - npl_dsr),
        "irreducible_pp": 100 * (1 - ceiling),
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "severity_bins": SEVERITY_BINS,
        "cell_losses": losses,
    }


if __name__ == "__main__":
    save("exp08_ceiling", run())
