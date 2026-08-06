"""Executable checks of the four safety invariants of Section 4.

These are the tests that matter most for the paper's claims: each invariant is
checked by exhaustive or randomised search over admissible transitions rather
than by restating the proof.
"""
import itertools

import numpy as np
import pytest

from safenest.corpus import build_corpus
from safenest.labeling import Label, decision_to_label
from safenest.lattice import Access, Capability, access_level, allowed_set
from safenest.policy import (
    SEVERITY_GATED, Decision, L0TokenFilter, L1SocraticGuard, NestedPolicyEngine,
    Response, conjoin,
)
from safenest.tiers import ALL_TIERS, Tier

ENGINE = NestedPolicyEngine()
LAYER_NAMES = ENGINE.layer_names()


def _responses():
    """A spanning set of candidate responses."""
    for cap in Capability:
        for fk in (0.0, 5.0, 14.0):
            for direct in (True, False):
                for academic in (True, False):
                    yield Response(
                        capability=cap, fk_grade=fk, is_direct_answer=direct,
                        is_academic_query=academic,
                    )


# --------------------------------------------------- Invariant I: monotonicity
def test_invariant_I_tier_monotonicity_holds_for_every_response():
    for r in _responses():
        severity = [
            {Decision.ACCEPT: 0, Decision.MODIFY: 1, Decision.REJECT: 2}[ENGINE.evaluate(r, t)]
            for t in ALL_TIERS
        ]
        assert all(b <= a for a, b in zip(severity, severity[1:])), (r, severity)


def test_invariant_I_allowed_sets_grow_with_tier():
    for a, b in itertools.combinations(ALL_TIERS, 2):
        assert allowed_set(a) <= allowed_set(b)


# ------------------------------------------- Invariant II: non-bypassability
def test_invariant_II_conjunction_cannot_be_relaxed_by_any_layer():
    """No single layer can turn a REJECT from another layer into an emission."""
    for r in _responses():
        for t in ALL_TIERS:
            per_layer = [layer.evaluate(r, t) for layer in ENGINE.layers]
            composite = ENGINE.evaluate(r, t)
            assert composite == conjoin(*per_layer)
            if Decision.REJECT in per_layer:
                assert composite is Decision.REJECT


def test_invariant_II_blocked_capability_never_emitted():
    """A capability whose access level is BLOCKED is never emitted. (LIMITED is
    emitted by design -- it is permitted but monitored -- so `allowed_set`,
    which excludes everything not fully AVAILABLE, is not the right predicate.)"""
    for r in _responses():
        for t in ALL_TIERS:
            if access_level(r.capability, t) is Access.BLOCKED:
                assert ENGINE.evaluate(r, t) is Decision.REJECT


def test_invariant_II_socratic_capability_never_yields_a_direct_answer():
    for r in _responses():
        for t in ALL_TIERS:
            if access_level(r.capability, t) is Access.SOCRATIC:
                assert not ENGINE.evaluate(r, t).emits_direct_answer


def test_invariant_II_severity_gate_applies_at_every_tier():
    """High-severity harm content is refused even at t5."""
    for cap in SEVERITY_GATED:
        r = Response(capability=cap, harm_severity=0.9)
        for t in ALL_TIERS:
            assert ENGINE.evaluate(r, t) is Decision.REJECT


def test_invariant_II_holds_across_random_traces_with_tier_upgrades():
    """A trace that walks a user up the tiers must never emit content that was
    blocked at the tier in force at the time of emission."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        tier = Tier.T1
        for _step in range(10):
            if rng.random() < 0.3 and int(tier) < 5:
                tier = Tier(int(tier) + 1)  # confirmed upgrade
            r = Response(
                capability=list(Capability)[int(rng.integers(len(Capability)))],
                fk_grade=float(rng.uniform(0, 14)),
                is_direct_answer=bool(rng.random() < 0.5),
                is_academic_query=bool(rng.random() < 0.5),
            )
            if ENGINE.evaluate(r, tier).emits_direct_answer:
                assert access_level(r.capability, tier) in (Access.AVAILABLE, Access.LIMITED)


def test_invariant_II_session_limit_is_enforced_per_tier():
    from safenest.tiers import TIER_SPECS

    for t in ALL_TIERS:
        limit = TIER_SPECS[t].session_limit_min
        if limit is None:
            continue
        r = Response(capability=Capability.OPEN_ENDED_CHAT, session_minutes=limit + 1)
        assert ENGINE.evaluate(r, t) is Decision.REJECT


# ------------------------------------------- Invariant IV: graceful degradation
#: Layers that actually gate emission. L2 and L3 are asynchronous and
#: contribute no synchronous constraint, so their failure is a no-op.
ENFORCING = {"L0_token_filter", "L1_socratic_guard", "L4_policy_store"}


def test_invariant_IV_all_single_and_pairwise_failures_are_safe():
    """Exhaustive case analysis over the 15 single- and pairwise-layer failures.

    Safe means: whenever any *enforcing* layer fails, no direct answer is
    emitted at all; and in every case the degraded engine is at least as
    restrictive as the nominal one.
    """
    combos = [(n,) for n in LAYER_NAMES] + list(itertools.combinations(LAYER_NAMES, 2))
    assert len(combos) == 15
    for combo in combos:
        degraded = ENGINE.with_failures(*combo)
        touches_enforcement = bool(set(combo) & ENFORCING)
        for r in _responses():
            for t in ALL_TIERS:
                d = degraded.evaluate(r, t)
                if touches_enforcement:
                    assert not d.emits_direct_answer, (combo, r, t)
                else:
                    assert d == ENGINE.evaluate(r, t), (combo, r, t)


def test_invariant_IV_degradation_is_never_less_restrictive_than_nominal():
    combos = [(n,) for n in LAYER_NAMES] + list(itertools.combinations(LAYER_NAMES, 2))
    sev = {Decision.ACCEPT: 0, Decision.MODIFY: 1, Decision.REJECT: 2}
    for combo in combos:
        degraded = ENGINE.with_failures(*combo)
        for r in _responses():
            for t in ALL_TIERS:
                assert sev[degraded.evaluate(r, t)] >= sev[ENGINE.evaluate(r, t)]


def test_total_layer_failure_suppresses_everything():
    dead = ENGINE.with_failures(*LAYER_NAMES)
    for r in _responses():
        for t in ALL_TIERS:
            assert dead.evaluate(r, t) is Decision.REJECT


# ------------------------------------------------------------ engine structure
def test_layers_are_strictly_frequency_ordered():
    freqs = [layer.update_frequency_hz for layer in ENGINE.layers]
    assert all(a > b for a, b in zip(freqs, freqs[1:]))


def test_engine_rejects_misordered_layers():
    with pytest.raises(ValueError):
        NestedPolicyEngine(layers=[L1SocraticGuard(), L0TokenFilter()])


def test_conjoin_is_commutative_associative_and_reject_absorbing():
    ds = list(Decision)
    for a, b in itertools.product(ds, repeat=2):
        assert conjoin(a, b) == conjoin(b, a)
        assert conjoin(a, Decision.REJECT) is Decision.REJECT
    for a, b, c in itertools.product(ds, repeat=3):
        assert conjoin(conjoin(a, b), c) == conjoin(a, conjoin(b, c))


# ---------------------------------------------------------- no under-protection
def test_npl_never_under_protects_on_the_harm_categories():
    """On the corpus, NPL must not allow anything the independent rubric blocks
    in the three content-harm categories."""
    from safenest.baselines import make_npl
    from safenest.corpus import HARM_CATEGORIES
    from safenest.labeling import rubric_label

    npl = make_npl(ENGINE)
    for p in build_corpus(n_per_cell=40, seed=1):
        if p.category not in HARM_CATEGORIES:
            continue
        if rubric_label(p) is Label.BLOCK:
            assert decision_to_label(npl(p, p.tier)) is not Label.ALLOW
