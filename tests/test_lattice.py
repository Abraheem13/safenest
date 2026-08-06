import itertools

import pytest

from safenest.lattice import (
    Access, Capability, allowed_set, constraint_set, join, leq, meet, validate_lattice,
)
from safenest.tiers import ALL_TIERS, Tier


def test_lattice_validates():
    validate_lattice()


def test_constraint_sets_form_a_descending_chain():
    sets = [constraint_set(t) for t in ALL_TIERS]
    for lower, higher in zip(sets, sets[1:]):
        assert higher <= lower


def test_tier_monotonicity_invariant_I():
    """A(t_i) subset-or-equal A(t_j) for all i < j."""
    for ti, tj in itertools.combinations(ALL_TIERS, 2):
        assert allowed_set(ti) <= allowed_set(tj)


def test_partial_order_is_reflexive_antisymmetric_transitive():
    for t in ALL_TIERS:
        assert leq(t, t)
    for a, b in itertools.product(ALL_TIERS, repeat=2):
        if leq(a, b) and leq(b, a):
            assert constraint_set(a) == constraint_set(b)
    for a, b, c in itertools.product(ALL_TIERS, repeat=3):
        if leq(a, b) and leq(b, c):
            assert leq(a, c)


def test_meet_is_most_restrictive_and_join_least():
    for a, b in itertools.product(ALL_TIERS, repeat=2):
        m, j = meet(a, b), join(a, b)
        assert constraint_set(m) >= constraint_set(a) | constraint_set(b)
        assert constraint_set(j) <= constraint_set(a) & constraint_set(b)


def test_meet_join_absorption_and_idempotence():
    for a, b in itertools.product(ALL_TIERS, repeat=2):
        assert meet(a, a) is a and join(a, a) is a
        assert meet(a, join(a, b)) is a
        assert join(a, meet(a, b)) is a


def test_membership_queries_are_set_operations():
    for t in ALL_TIERS:
        assert allowed_set(t) | constraint_set(t) == frozenset(Capability)
        assert not (allowed_set(t) & constraint_set(t))


def test_age_inappropriate_never_available():
    for t in ALL_TIERS:
        assert Capability.AGE_INAPPROPRIATE in constraint_set(t)


def test_t1_is_bottom_and_t5_is_top():
    assert all(leq(Tier.T1, t) for t in ALL_TIERS)
    assert all(leq(t, Tier.T5) for t in ALL_TIERS)


def test_validate_lattice_rejects_a_non_monotone_edit(monkeypatch):
    from safenest import lattice

    broken = {k: dict(v) for k, v in lattice.FEATURE_GATING.items()}
    broken[Capability.ESSAY_WRITING][Tier.T5] = Access.BLOCKED  # more restrictive than t4
    monkeypatch.setattr(lattice, "FEATURE_GATING", broken)
    with pytest.raises(ValueError):
        lattice.validate_lattice()
