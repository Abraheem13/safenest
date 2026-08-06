"""Frameworks under comparison.

Reviewer 1's major comment 5 is that comparing NPL only against age-agnostic
guardrails is an unfair fight. Three stronger, age-*aware* baselines are
therefore included, and the paper should report the full table:

  age-agnostic   NeMo-style, LlamaGuard-style, LlamaFirewall-style,
                 Constitutional-AI-style. Content filters with no user model.
  age-aware      coppa_binary       under-13 / over-13 rule split
                 age_conditioned    three coarse age bands, oracle age
                 child_safety_clf   binary child-mode classifier with realistic
                                    error, applied to an adult-safety filter

Each framework is a function (Prompt, assigned_tier) -> Decision. Baselines that
do not model developmental stage simply ignore the tier argument.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .corpus import Category, Prompt
from .lattice import Access, access_level
from .policy import Decision, NestedPolicyEngine, Response
from .tiers import Tier

Framework = Callable[[Prompt, Tier], Decision]


# ------------------------------------------------------------ age-agnostic
def nemo_guardrails(prompt: Prompt, tier: Tier) -> Decision:
    """Topical rails: blocks configured harmful topics, passes everything else."""
    if prompt.category is Category.AGE_INAPPROPRIATE:
        return Decision.REJECT
    if prompt.category is Category.CRISIS and prompt.harm_severity > 0.6:
        return Decision.MODIFY  # safe-completion rail
    return Decision.ACCEPT


def llama_guard(prompt: Prompt, tier: Tier) -> Decision:
    """Harm-taxonomy classifier: binary safe/unsafe, tuned for adults."""
    if prompt.category is Category.AGE_INAPPROPRIATE:
        return Decision.REJECT
    if prompt.category is Category.CRISIS and prompt.harm_severity > 0.5:
        return Decision.REJECT
    if prompt.category is Category.SUBSTANCE and prompt.harm_severity > 0.8:
        return Decision.REJECT
    return Decision.ACCEPT


def llama_firewall(prompt: Prompt, tier: Tier) -> Decision:
    """Agent-oriented guardrail: injection and misalignment focus."""
    if prompt.category is Category.AGE_INAPPROPRIATE:
        return Decision.REJECT
    if prompt.category is Category.CRISIS and prompt.harm_severity > 0.7:
        return Decision.REJECT
    return Decision.ACCEPT


def constitutional_ai(prompt: Prompt, tier: Tier) -> Decision:
    """Self-critique against a uniform constitution: the strongest age-agnostic
    baseline, because its harmlessness principles catch more marginal cases."""
    if prompt.category is Category.AGE_INAPPROPRIATE:
        return Decision.REJECT
    if prompt.category is Category.CRISIS:
        return Decision.REJECT if prompt.harm_severity > 0.4 else Decision.MODIFY
    if prompt.category is Category.SUBSTANCE:
        return Decision.REJECT if prompt.harm_severity > 0.6 else Decision.MODIFY
    return Decision.ACCEPT


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


def age_conditioned(prompt: Prompt, tier: Tier) -> Decision:
    """Three coarse bands (young child / older child / teen) with an oracle age.

    This is the strongest non-developmental baseline: it knows the user's band
    exactly and applies a hand-written policy per band, but has no stage model,
    no Socratic substitution and no lattice.
    """
    band = 0 if int(tier) <= 2 else (1 if int(tier) == 3 else 2)
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


def make_child_safety_classifier(seed: int = 7, accuracy: float = 0.90) -> Framework:
    """Binary child-mode detector feeding an adult-safety filter.

    Models the deployed pattern of "detect a child, then switch on kid mode",
    including realistic detection error.
    """
    rng = np.random.default_rng(seed)

    def framework(prompt: Prompt, tier: Tier) -> Decision:
        true_child = int(tier) <= 3
        detected = true_child if rng.random() < accuracy else (not true_child)
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
            session_minutes=0.0,  # session limits are evaluated separately
            harm_severity=prompt.harm_severity,
        )
        return engine.evaluate(response, tier)

    return framework


def npl_socratic_only(prompt: Prompt, tier: Tier) -> Decision:
    """Ablation: the lattice without the Socratic substitution mechanism, so
    everything the lattice marks SOCRATIC is refused outright instead."""
    level = access_level(prompt.capability, tier)
    if level is Access.AVAILABLE:
        return Decision.ACCEPT
    return Decision.REJECT


AGE_AGNOSTIC: dict[str, Framework] = {
    "NeMo Guardrails": nemo_guardrails,
    "Llama Guard": llama_guard,
    "LlamaFirewall": llama_firewall,
    "Constitutional AI": constitutional_ai,
}
AGE_AWARE: dict[str, Framework] = {
    "COPPA binary rule": coppa_binary,
    "Age-conditioned (oracle)": age_conditioned,
    "Child-safety classifier": make_child_safety_classifier(),
}


def all_frameworks(engine: NestedPolicyEngine | None = None) -> dict[str, Framework]:
    return {
        **AGE_AGNOSTIC,
        **AGE_AWARE,
        "NPL (no Socratic)": npl_socratic_only,
        "NPL (ours)": make_npl(engine),
    }
