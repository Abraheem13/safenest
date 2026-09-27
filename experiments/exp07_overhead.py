"""Experiment 07 -- computational overhead and the cost of the
conditional-independence assumption.

Latencies are measured on the host machine rather than asserted, and the
platform is recorded alongside them in the result file.
"""
from __future__ import annotations

import platform
import sys
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import pct, rng_for, save, table  # noqa: E402
from safenest.estimator import BayesianAgeEstimator, EstimatorState  # noqa: E402
from safenest.lattice import Capability  # noqa: E402
from safenest.policy import (  # noqa: E402
    L0TokenFilter,
    L1SocraticGuard,
    L4PolicyStore,
    NestedPolicyEngine,
    Response,
)
from safenest.privacy import PrivacyConfig, PrivacyMode  # noqa: E402
from safenest.signals import SignalModel  # noqa: E402
from safenest.socratic import SocraticMDP  # noqa: E402
from safenest.tiers import ALL_TIERS, Tier  # noqa: E402

REPEATS = 20_000
CORRELATIONS = (0.0, 0.2, 0.4, 0.6, 0.8)


def _time(fn, repeats: int = REPEATS) -> float:
    """Median-of-5 seconds per call, to blunt scheduler noise."""
    times = [timeit.timeit(fn, number=repeats) / repeats for _ in range(5)]
    return float(np.median(times))


def run() -> dict:
    rng = rng_for("overhead")
    engine = NestedPolicyEngine()
    l0, l1, l4 = L0TokenFilter(), L1SocraticGuard(), L4PolicyStore()
    resp = Response(
        capability=Capability.HOMEWORK_ANSWER,
        tokens=tuple(f"w{i}" for i in range(40)),
        fk_grade=4.0, is_academic_query=True,
    )
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    signals = est.model.sample(Tier.T3, rng)
    state = EstimatorState()
    mdp = SocraticMDP()
    mdp.solve(Tier.T3)  # warm the offline table

    measurements = {
        "L0: Token Filter (per response, 40 tokens)": _time(lambda: l0.evaluate(resp, Tier.T3)),
        "L1: Socratic Guard (per response)": _time(lambda: l1.evaluate(resp, Tier.T3)),
        "L4: Policy Store (per query)": _time(lambda: l4.evaluate(resp, Tier.T3)),
        "Full engine (per response)": _time(lambda: engine.evaluate(resp, Tier.T3)),
        "L2: Bayesian update (per interaction)": _time(
            lambda: est.observe(state, signals, rng), repeats=2_000
        ),
        "Socratic MDP lookup (per step)": _time(
            lambda: mdp.solve(Tier.T3)[0][0, 25], repeats=20_000
        ),
    }

    # The engine consumes features; computing them from text is the expensive
    # part in a deployment, so it is measured too when the tools are installed.
    measurements.update(_feature_extraction())

    rows = [
        {"Component": k, "Latency": _fmt(v), "seconds": f"{v:.3e}"}
        for k, v in measurements.items()
    ]
    table(rows, ["Component", "Latency", "seconds"],
          f"Measured overhead on {platform.processor() or platform.machine()}")

    sync = measurements["Full engine (per response)"]
    print(f"\n  Synchronous per-response overhead: {_fmt(sync)}")
    for llm_ms in (500, 2000):
        print(f"    as a share of a {llm_ms} ms LLM response: "
              f"{100 * sync / (llm_ms / 1000):.4f}%")

    # ---- cost of the conditional-independence assumption -------------------
    print("\n  Cost of the conditional-independence assumption:")
    ci_rows = []
    ci = {}
    for rho in CORRELATIONS:
        gen = SignalModel(correlation=rho)
        hits = 0
        n = 0
        for tier in ALL_TIERS:
            for _ in range(120):
                got, _ = est.run_session(tier, 10, rng, generating_model=gen)
                hits += got is tier
                n += 1
        ci[rho] = hits / n
        ci_rows.append({"feature correlation rho": f"{rho:.1f}", "accuracy (%)": pct(ci[rho])})
    table(ci_rows, ["feature correlation rho", "accuracy (%)"],
          "  Estimator assumes independence; data generated with correlated features")
    drop = ci[0.0] - ci[max(CORRELATIONS)]
    print(f"  Accuracy cost from rho=0 to rho={max(CORRELATIONS)}: {100 * drop:.1f} pp")

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurements_seconds": measurements,
        "synchronous_per_response_seconds": sync,
        "share_of_500ms_llm": sync / 0.5,
        "share_of_2000ms_llm": sync / 2.0,
        "conditional_independence": {
            "correlations": list(CORRELATIONS),
            "accuracy": ci,
            "accuracy_drop_pp": 100 * drop,
        },
    }


#: A 50-word message, the interaction window used for the real-data experiments.
MESSAGE = ("I think that schools should let students choose their own summer projects "
           "because it helps them learn about things they really care about. When my "
           "class did a project on volcanoes last year, everyone worked harder and we "
           "remembered much more than we usually do.")


def _feature_extraction() -> dict[str, float]:
    """Latency of computing the five linguistic features and a sentence embedding."""
    out: dict[str, float] = {}
    try:
        from safenest.features import Featurizer

        featurizer = Featurizer()
        featurizer.window_features(featurizer.parse([MESSAGE], spoken=False)[0])  # warm up
        out["Feature extraction (per 50-word message)"] = _time(
            lambda: featurizer.window_features(featurizer.parse([MESSAGE], spoken=False)[0]),
            repeats=200)
    except (ImportError, OSError):
        pass
    try:
        from sentence_transformers import SentenceTransformer

        from experiments.real_common import EMBEDDER

        model = SentenceTransformer(EMBEDDER, device="cpu")
        model.encode([MESSAGE])
        out["Sentence embedding (per 50-word message, CPU)"] = _time(
            lambda: model.encode([MESSAGE], show_progress_bar=False), repeats=100)
    except (ImportError, OSError):
        pass
    return out


def _fmt(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


if __name__ == "__main__":
    save("exp07_overhead", run())
