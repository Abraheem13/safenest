import numpy as np
import pytest

from safenest.estimator import BayesianAgeEstimator, EstimatorState
from safenest.privacy import PrivacyConfig, PrivacyMode
from safenest.signals import (
    NEURODIVERGENT_VERBAL, SignalModel, linguistic_kl_matrix, min_adjacent_kl,
    sanov_interactions,
)
from safenest.tiers import ALL_TIERS, K_TIERS, Tier


@pytest.fixture
def estimator():
    return BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))


def test_prior_is_uniform():
    s = EstimatorState()
    assert np.allclose(s.posterior, 1.0 / K_TIERS)


def test_posterior_is_normalised_after_updates(estimator):
    rng = np.random.default_rng(0)
    s = EstimatorState()
    for _ in range(10):
        estimator.observe(s, estimator.model.sample(Tier.T3, rng), rng)
        assert np.isclose(s.posterior.sum(), 1.0)


def test_fail_safe_defaults_to_t1_when_unconfident(estimator):
    s = EstimatorState()  # uniform posterior, max = 0.2 < gamma
    assert estimator.assign(s) is Tier.T1


def test_confident_posterior_returns_map_tier(estimator):
    s = EstimatorState()
    s.log_posterior = np.log(np.array([0.01, 0.01, 0.01, 0.01, 0.96]))
    assert estimator.assign(s) is Tier.T5


def test_estimator_recovers_the_true_tier_without_inference_noise(estimator):
    rng = np.random.default_rng(3)
    for tier in ALL_TIERS:
        hits = 0
        for _ in range(200):
            got, _ = estimator.run_session(tier, n_interactions=10, rng=rng)
            hits += got is tier
        assert hits / 200 > 0.85, (tier, hits / 200)


def test_accuracy_is_monotone_in_the_number_of_interactions(estimator):
    rng = np.random.default_rng(11)
    accs = []
    for n in (1, 3, 10):
        hits = sum(
            estimator.run_session(Tier.T2, n_interactions=n, rng=rng)[0] is Tier.T2
            for _ in range(300)
        )
        accs.append(hits / 300)
    assert accs[0] <= accs[1] <= accs[2] + 0.02


def test_local_dp_noise_degrades_accuracy_as_theory_predicts():
    """The SNR argument in privacy.py says eps=1.0 local DP cannot classify
    reliably. This test pins that finding so it cannot silently regress."""
    est = BayesianAgeEstimator(
        privacy=PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0)
    )
    rng = np.random.default_rng(5)
    hits = sum(
        est.run_session(Tier.T5, n_interactions=10, rng=rng)[0] is Tier.T5
        for _ in range(300)
    )
    assert hits / 300 < 0.60


def test_misclassification_under_local_dp_is_protective():
    """When the noisy estimator is unconfident it must fall back to t1, so the
    error is over-protection rather than under-protection."""
    est = BayesianAgeEstimator(
        privacy=PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0)
    )
    rng = np.random.default_rng(6)
    got = [est.run_session(Tier.T2, 5, rng)[0] for _ in range(400)]
    over = sum(int(t) <= int(Tier.T2) for t in got)
    assert over / len(got) > 0.8


def test_mahalanobis_flags_an_implausible_profile(estimator):
    mean, std = estimator.model.linguistic_params(Tier.T1)
    assert not estimator.bypass_detected(mean, Tier.T1)
    assert estimator.bypass_detected(mean + 6 * std, Tier.T1)


def test_discordance_flag_protects_a_verbally_advanced_young_child():
    """A highly verbal 8-year-old whose linguistic profile reads as t4 must be
    held at the attested tier, not promoted."""
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    gen = SignalModel(profile=NEURODIVERGENT_VERBAL)
    rng = np.random.default_rng(9)
    unprotected, _ = est.run_session(Tier.T2, 10, rng, generating_model=gen)
    protected, _ = est.run_session(
        Tier.T2, 10, rng, generating_model=gen, attested_tier=Tier.T2
    )
    assert int(protected) <= int(unprotected)
    assert protected is Tier.T2


def test_kl_matrix_is_positive_off_diagonal_and_zero_on_diagonal():
    m = linguistic_kl_matrix()
    assert np.allclose(np.diag(m), 0.0)
    assert (m[~np.eye(K_TIERS, dtype=bool)] > 0).all()


def test_sanov_bound_decreases_with_separability():
    assert sanov_interactions(1.2, 0.05) < sanov_interactions(0.5, 0.05)


def test_min_adjacent_kl_is_at_the_t4_t5_boundary():
    m = linguistic_kl_matrix()
    adjacent = [min(m[i, i + 1], m[i + 1, i]) for i in range(K_TIERS - 1)]
    assert np.argmin(adjacent) == 3
    assert np.isclose(min(adjacent), min_adjacent_kl())


def test_proposition_1_bound_dominates_simulated_error():
    """The proved misassignment bound must sit above the simulated error rate
    for every tier at every milestone the manuscript reports."""
    from safenest.signals import misassignment_bound
    from safenest.estimator import DEFAULT_GAMMA
    model = SignalModel()
    est = BayesianAgeEstimator(privacy=PrivacyConfig(mode=PrivacyMode.CORPUS))
    rng = np.random.default_rng(11)
    for tier in Tier:
        errors = {1: 0, 3: 0}
        trials = 300
        for _ in range(trials):
            state = EstimatorState()
            for n in range(1, 4):
                est.observe(state, model.sample(tier, rng), rng)
                if n in errors:
                    errors[n] += est.assign(state) is not tier
        for n, e in errors.items():
            # three standard errors of slack for the finite simulation
            slack = 3 * np.sqrt(max(e / trials, 1e-3) / trials)
            assert e / trials <= misassignment_bound(model, tier, n, DEFAULT_GAMMA) + slack


def test_chernoff_closed_form_matches_monte_carlo():
    from safenest.signals import log_mgf_llr
    model = SignalModel()
    rng = np.random.default_rng(5)
    t, r, s = Tier.T4, Tier.T5, 0.4
    vals = []
    for _ in range(60_000):
        x = model.sample(t, rng)
        l = sum(model.log_likelihood(m, x[m], r) - model.log_likelihood(m, x[m], t) for m in x)
        vals.append(np.exp(s * l))
    assert np.log(np.mean(vals)) == pytest.approx(log_mgf_llr(model, t, r, s), abs=0.03)
