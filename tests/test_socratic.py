import numpy as np
import pytest

from safenest.socratic import (
    ACTIONS, Action, MAX_DELTA, RewardParams, SocraticMDP, delta, sensitivity_analysis,
)
from safenest.tiers import ALL_TIERS, Tier


@pytest.fixture(scope="module")
def mdp():
    return SocraticMDP()


def test_reward_params_reject_a_too_small_reveal_penalty():
    with pytest.raises(ValueError):
        SocraticMDP(params=RewardParams(alpha_reveal=0.01))


def test_default_params_satisfy_the_no_premature_reveal_condition():
    p = RewardParams()
    assert p.alpha_reveal > p.alpha_learn * MAX_DELTA


def test_transition_rows_are_probability_distributions(mdp):
    for t in ALL_TIERS:
        for a in ACTIONS:
            p = mdp.transition_matrix(a, t)
            assert np.allclose(p.sum(axis=1), 1.0)
            assert (p >= 0).all()


def test_expected_knowledge_increases_toward_the_zpd_target(mdp):
    q = mdp.q_grid
    for t in ALL_TIERS:
        gain = mdp.transition_matrix(Action.GUIDE, t) @ q - q
        assert gain[: mdp.params.n_q // 2].mean() > 0


def test_partial_explain_is_never_the_optimal_opening_action(mdp):
    """The central claim of Section 3.5."""
    for t in ALL_TIERS:
        assert not mdp.opens_with_revelation(t)


def test_verify_request_usage_increases_with_tier(mdp):
    usage = [mdp.action_distribution(t)[Action.VERIFY_REQUEST] for t in ALL_TIERS]
    assert usage[-1] >= usage[0]


def test_action_distribution_sums_to_one_hundred(mdp):
    for t in ALL_TIERS:
        assert np.isclose(sum(mdp.action_distribution(t).values()), 100.0)


def test_delta_is_monotone_in_tier_for_every_action():
    for i, a in enumerate(ACTIONS):
        vals = [delta(a, t) for t in ALL_TIERS]
        assert all(b >= a_ for a_, b in zip(vals, vals[1:])), a


def test_trajectory_is_non_decreasing_in_knowledge(mdp):
    traj = mdp.trajectory(Tier.T2, q0=0.15)
    assert len(traj) == mdp.params.horizon
    for _step, _action, q_before, q_after in traj:
        assert q_after >= q_before - 1e-9


def test_bellman_update_count_matches_the_stated_bound(mdp):
    assert mdp.bellman_updates() <= 12_500


def test_policy_shape_and_validity(mdp):
    for t in ALL_TIERS:
        policy, value = mdp.solve(t)
        assert policy.shape == (mdp.params.horizon, mdp.params.n_q)
        assert value.shape == policy.shape
        assert policy.min() >= 0 and policy.max() < len(ACTIONS)


def test_the_guarantee_is_not_vacuous():
    """Without the reveal penalty, PartialExplain *is* the optimal opening
    action -- so the guarantee is doing work rather than restating the dynamics."""
    weak = SocraticMDP(
        params=RewardParams(alpha_reveal=0.0, alpha_frust=0.0), strict=False
    )
    assert any(weak.opens_with_revelation(t) for t in ALL_TIERS)


def test_binding_threshold_is_higher_in_the_sequential_setting():
    """The one-step condition alpha_reveal > alpha_learn * max_delta is
    sufficient but not tight: the continuation value suppresses revelation at
    penalties below it too. Worth reporting, since the paper states only the
    one-step bound."""
    marginal = SocraticMDP(
        params=RewardParams(alpha_reveal=0.30, alpha_frust=0.0), strict=False
    )
    assert not any(marginal.opens_with_revelation(t) for t in ALL_TIERS)


def test_sensitivity_analysis_reports_stability_over_a_plausible_range():
    rows = sensitivity_analysis(
        alpha_reveal_grid=np.array([0.5, 0.6, 0.8, 1.0]),
        alpha_frust_grid=np.array([0.0, 0.25, 0.5]),
        discount_grid=np.array([0.9, 0.95, 0.99]),
    )
    valid = [r for r in rows if r["valid"]]
    assert valid
    assert all(r["no_premature_reveal"] for r in valid)
