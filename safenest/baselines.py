"""Frameworks under comparison.

Comparing a developmental architecture only against age-agnostic guardrails is
not an informative test, so three age-aware baselines are included alongside
four age-agnostic ones:

  age-agnostic   four content rules with no model of the user. Each models the
                 documented decision behaviour of a deployed system -- a
                 topical rail (NeMo Guardrails), a harm-taxonomy classifier
                 (Llama Guard), an agent firewall (LlamaFirewall) and a
                 constitutional critique (Constitutional AI) -- as a threshold
                 rule on harm severity. They are not the shipped systems, which
                 are configurable and could be made to implement any policy.
                 Their thresholds are parameters: `experiments/exp04_comparative`
                 reports them at the default values below and tuned to their
                 best on a held-out tuning corpus.
  age-aware      coppa_binary       under-13 / 13-plus rule split
                 age_conditioned    hand-written policy per age band, with the
                                    true age band given (oracle); the band
                                    partition is a parameter
                 child_safety_clf   binary child-mode classifier with 90%
                                    detection accuracy, feeding an adult filter

Each framework is a function (Prompt, assigned_tier) -> Decision. Baselines that
do not model developmental stage ignore the tier argument.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .corpus import Category, Prompt
from .lattice import Access, access_level
from .policy import Decision, L1SocraticGuard, NestedPolicyEngine, Response
from .tiers import Tier

Framework = Callable[[Prompt, Tier], Decision]


# ------------------------------------------------------------ age-agnostic
def make_topical_rail(crisis_modify_above: float = 0.6) -> Framework:
    """Topical rails: block configured harmful topics, pass everything else."""
    def framework(prompt: Prompt, tier: Tier) -> Decision:
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if prompt.category is Category.CRISIS and prompt.harm_severity > crisis_modify_above:
            return Decision.MODIFY  # safe-completion rail
        return Decision.ACCEPT
    return framework


def make_harm_classifier(crisis_reject_above: float = 0.5,
                         substance_reject_above: float = 0.8) -> Framework:
    """Harm-taxonomy classifier: binary safe/unsafe, tuned for adults."""
    def framework(prompt: Prompt, tier: Tier) -> Decision:
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if prompt.category is Category.CRISIS and prompt.harm_severity > crisis_reject_above:
            return Decision.REJECT
        if (prompt.category is Category.SUBSTANCE
                and prompt.harm_severity > substance_reject_above):
            return Decision.REJECT
        return Decision.ACCEPT
    return framework


def make_agent_firewall(crisis_reject_above: float = 0.7) -> Framework:
    """Agent-oriented guardrail: injection and misalignment focus."""
    def framework(prompt: Prompt, tier: Tier) -> Decision:
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if prompt.category is Category.CRISIS and prompt.harm_severity > crisis_reject_above:
            return Decision.REJECT
        return Decision.ACCEPT
    return framework


def make_constitutional(crisis_reject_above: float = 0.4,
                        substance_reject_above: float = 0.6) -> Framework:
    """Self-critique against a uniform constitution; the strongest age-agnostic
    rule at its default thresholds, because it scaffolds marginal harm cases."""
    def framework(prompt: Prompt, tier: Tier) -> Decision:
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if prompt.category is Category.CRISIS:
            return (Decision.REJECT if prompt.harm_severity > crisis_reject_above
                    else Decision.MODIFY)
        if prompt.category is Category.SUBSTANCE:
            return (Decision.REJECT if prompt.harm_severity > substance_reject_above
                    else Decision.MODIFY)
        return Decision.ACCEPT
    return framework


#: Threshold grid searched when a rule is tuned to its best on a tuning corpus.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(float(x), 2) for x in np.linspace(0.0, 1.0, 21))
AGE_AGNOSTIC_FACTORIES: dict[str, tuple[Callable[..., Framework], tuple[str, ...]]] = {
    "Topical rail": (make_topical_rail, ("crisis_modify_above",)),
    "Harm classifier": (make_harm_classifier,
                        ("crisis_reject_above", "substance_reject_above")),
    "Agent firewall": (make_agent_firewall, ("crisis_reject_above",)),
    "Constitutional rule": (make_constitutional,
                            ("crisis_reject_above", "substance_reject_above")),
}

nemo_guardrails = make_topical_rail()
llama_guard = make_harm_classifier()
llama_firewall = make_agent_firewall()
constitutional_ai = make_constitutional()


# --------------------------------------------------------------- age-aware
def coppa_binary(prompt: Prompt, tier: Tier) -> Decision:
    """A single under-13 / 13-plus split, the rule most deployments actually use."""
    child = int(tier) <= 3  # t1-t3 are ages 3-12
    if prompt.category is Category.AGE_INAPPROPRIATE:
        return Decision.REJECT
    if prompt.category in (Category.CRISIS, Category.SUBSTANCE):
        return Decision.REJECT if child else Decision.MODIFY
    if child and prompt.is_academic and prompt.is_direct_request:
        return Decision.REJECT
    return Decision.ACCEPT


#: The default partition: young child {t1, t2}, older child {t3}, teen {t4, t5}.
DEFAULT_BANDS: tuple[tuple[int, ...], ...] = ((1, 2), (3,), (4, 5))


def contiguous_three_band_partitions() -> list[tuple[tuple[int, ...], ...]]:
    """All six ways to cut the five ordered tiers into three contiguous bands."""
    out = []
    for a in range(1, 4):
        for b in range(a + 1, 5):
            out.append((tuple(range(1, a + 1)), tuple(range(a + 1, b + 1)),
                        tuple(range(b + 1, 6))))
    return out


def make_age_conditioned(bands: tuple[tuple[int, ...], ...] = DEFAULT_BANDS) -> Framework:
    """Three age bands (young / middle / oldest) with the true band given.

    A hand-written policy per band: it knows the user's band exactly but has
    no stage model, no Socratic substitution and no monotone lattice.
    """
    band_of = {t: i for i, band in enumerate(bands) for t in band}

    def framework(prompt: Prompt, tier: Tier) -> Decision:
        band = band_of[int(tier)]
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if prompt.category is Category.CRISIS:
            return Decision.REJECT if band == 0 else Decision.MODIFY
        if prompt.category is Category.SUBSTANCE:
            return Decision.REJECT if band < 2 else Decision.MODIFY
        if prompt.is_academic:
            if band == 0:
                return Decision.REJECT
            if band == 1:
                return Decision.MODIFY
            return Decision.ACCEPT if not prompt.is_direct_request else Decision.MODIFY
        if band == 0 and prompt.is_direct_request:
            return Decision.MODIFY
        return Decision.ACCEPT

    return framework


age_conditioned = make_age_conditioned()


def make_child_safety_classifier(seed: int = 7, accuracy: float = 0.90) -> Framework:
    """Binary child-mode detector feeding an adult-safety filter.

    Models the deployed pattern of "detect a child, then switch on kid mode",
    including detection error.
    """
    def framework(prompt: Prompt, tier: Tier) -> Decision:
        # Detection error is a deterministic function of the prompt, not of a
        # shared RNG: otherwise the same framework scores differently depending
        # on how many times it has already been called.
        draw = np.random.default_rng([seed, prompt.idx]).random()
        true_child = int(tier) <= 3
        detected = true_child if draw < accuracy else (not true_child)
        if prompt.category is Category.AGE_INAPPROPRIATE:
            return Decision.REJECT
        if detected:
            if prompt.category in (Category.CRISIS, Category.SUBSTANCE):
                return Decision.REJECT
            if prompt.is_academic and prompt.is_direct_request:
                return Decision.MODIFY
            return Decision.ACCEPT
        return constitutional_ai(prompt, tier)

    return framework


# ---------------------------------------------------------------------- NPL
def make_npl(engine: NestedPolicyEngine | None = None) -> Framework:
    """Nested Policy Learning: the full five-layer engine at the assigned tier."""
    engine = engine or NestedPolicyEngine()

    def framework(prompt: Prompt, tier: Tier) -> Decision:
        response = Response(
            capability=prompt.capability,
            tokens=(),
            fk_grade=0.0,
            is_direct_answer=prompt.is_direct_request,
            is_academic_query=prompt.is_academic,
            session_minutes=0.0,  # records carry no session time; see Section 5
            harm_severity=prompt.harm_severity,
        )
        return engine.evaluate(response, tier)

    return framework


def npl_engine(**l1_options) -> NestedPolicyEngine:
    """The default engine with L1 options changed (for ablations and sweeps)."""
    layers = NestedPolicyEngine().layers
    return NestedPolicyEngine(layers=[layers[0], L1SocraticGuard(**l1_options), *layers[2:]])


def npl_socratic_only(prompt: Prompt, tier: Tier) -> Decision:
    """Ablation: the lattice without the Socratic substitution mechanism, so
    everything the lattice marks SOCRATIC is refused outright instead."""
    level = access_level(prompt.capability, tier)
    if level is Access.AVAILABLE:
        return Decision.ACCEPT
    return Decision.REJECT


AGE_AGNOSTIC: dict[str, Framework] = {
    "Topical rail": nemo_guardrails,
    "Harm classifier": llama_guard,
    "Agent firewall": llama_firewall,
    "Constitutional rule": constitutional_ai,
}
AGE_AWARE: dict[str, Framework] = {
    "COPPA binary rule": coppa_binary,
    "Child-safety classifier": make_child_safety_classifier(),
    "Age-band oracle": age_conditioned,
}
#: The system whose documented behaviour each age-agnostic rule models.
MODELLED_ON: dict[str, str] = {
    "Topical rail": "NeMo Guardrails",
    "Harm classifier": "Llama Guard",
    "Agent firewall": "LlamaFirewall",
    "Constitutional rule": "Constitutional AI",
}
NPL_FULL = "NPL (full)"
NPL_NO_SOCRATIC = "NPL without Socratic substitution"


def all_frameworks(engine: NestedPolicyEngine | None = None) -> dict[str, Framework]:
    return {
        **AGE_AGNOSTIC,
        **AGE_AWARE,
        NPL_NO_SOCRATIC: npl_socratic_only,
        NPL_FULL: make_npl(engine),
    }
