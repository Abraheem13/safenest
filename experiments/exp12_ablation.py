"""Experiment 12 -- component ablation.

The comparative evaluation shows that the framework as a whole outperforms its
baselines. It does not show which part is responsible. This experiment removes
one mechanism at a time and reports what each contributes, so a reader can see
whether the result rests on the developmental lattice, on Socratic substitution,
on the severity gate, or on the scaffolding rule added below t3.

Each variant is a complete framework evaluated on the same 7,000 prompts against
the same independent rubric, so the differences are attributable to the removed
component rather than to a change of protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, save, table  # noqa: E402
from safenest.baselines import make_npl, npl_socratic_only  # noqa: E402
from safenest.corpus import HARM_CATEGORIES, build_corpus  # noqa: E402
from safenest.labeling import rubric_label  # noqa: E402
from safenest.metrics import correctness_vector, evaluate_framework, mcnemar  # noqa: E402
from safenest.policy import (  # noqa: E402
    L0TokenFilter, L1SocraticGuard, L2TierRefiner, L3MemoryLayer, L4PolicyStore,
    NestedPolicyEngine,
)
from safenest.tiers import ALL_TIERS  # noqa: E402


def _engine(scaffold_below_t3: bool = True, severity_gate: bool = True):
    return NestedPolicyEngine(layers=[
        L0TokenFilter(),
        L1SocraticGuard(scaffold_below_t3=scaffold_below_t3,
                        severity_gate=severity_gate),
        L2TierRefiner(), L3MemoryLayer(), L4PolicyStore(),
    ])


def run() -> dict:
    prompts = build_corpus()
    harm = [p for p in prompts if p.category in HARM_CATEGORIES]

    variants = {
        "Full framework": make_npl(_engine()),
        "-- scaffolding below $t_3$": make_npl(_engine(scaffold_below_t3=False)),
        "-- severity gate": make_npl(_engine(severity_gate=False)),
        "-- both refinements": make_npl(
            _engine(scaffold_below_t3=False, severity_gate=False)),
        "-- Socratic substitution": npl_socratic_only,
    }

    full_vec = correctness_vector(variants["Full framework"], prompts, rubric_label)
    rows, detail = [], {}
    for name, fw in variants.items():
        overall = evaluate_framework(fw, prompts, rubric_label)["overall"]
        harm_r = evaluate_framework(fw, harm, rubric_label)["overall"]
        test = (
            {"chi2": 0.0, "p_value": 1.0, "odds_ratio": 1.0}
            if name == "Full framework"
            else mcnemar(full_vec, correctness_vector(fw, prompts, rubric_label))
        )
        detail[name] = {
            "dsr": overall["dsr"],
            "under_protection": overall["under_protection"],
            "over_restriction": overall["over_restriction"],
            "harm_dsr": harm_r["dsr"],
            "delta_pp": 100 * (overall["dsr"] - evaluate_framework(
                variants["Full framework"], prompts, rubric_label)["overall"]["dsr"]),
            "mcnemar": test,
        }
        rows.append({
            "Variant": name,
            "DSR (%)": pct(overall["dsr"]),
            "Delta (pp)": f"{detail[name]['delta_pp']:+.1f}",
            "Under (%)": pct(overall["under_protection"]),
            "Over (%)": pct(overall["over_restriction"]),
            "Harm DSR (%)": pct(harm_r["dsr"]),
        })

    table(rows, ["Variant", "DSR (%)", "Delta (pp)", "Under (%)", "Over (%)",
                 "Harm DSR (%)"],
          "Component ablation against the independent rubric (N = 7,000)")

    # Which mechanism protects which tier?
    by_tier = {}
    for name, fw in variants.items():
        r = evaluate_framework(fw, prompts, rubric_label)["by_tier"]
        by_tier[name] = {t.label: r[t.label]["dsr"] for t in ALL_TIERS}
    table(
        [{"Variant": n, **{t.label: pct(v[t.label]) for t in ALL_TIERS}}
         for n, v in by_tier.items()],
        ["Variant", *[t.label for t in ALL_TIERS]],
        "Ablation by tier (DSR %)",
    )

    full = detail["Full framework"]
    nogate = detail["-- severity gate"]
    print("\n  Reading. Socratic substitution carries the headline result: removing")
    print(f"  it costs {abs(detail['-- Socratic substitution']['delta_pp']):.1f} points and "
          "trebles over-restriction.")
    print("  The severity gate does NOT buy accuracy on the harm categories: removing")
    print(f"  it raises harm-category DSR from {pct(full['harm_dsr'])}% to "
          f"{pct(nogate['harm_dsr'])}% and overall DSR from {pct(full['dsr'])}% to "
          f"{pct(nogate['dsr'])}%.")
    print(f"  What it buys is under-protection: {pct(full['under_protection'])}% against "
          f"{pct(nogate['under_protection'])}% without it.")
    print("  The gate therefore trades "
          f"{nogate['delta_pp']:.1f} points of accuracy for "
          f"{100 * (nogate['under_protection'] - full['under_protection']):.1f} points of")
    print("  under-protection. Whether that is a good trade depends on how the two")
    print("  error types are weighted; at the adopted threshold it is plausibly")
    print("  mis-tuned, which the threshold sweep in Experiment 09 supports.")

    return {
        "variants": detail,
        "by_tier": by_tier,
        "n_prompts": len(prompts),
        "n_harm_prompts": len(harm),
    }


if __name__ == "__main__":
    save("exp12_ablation", run())
