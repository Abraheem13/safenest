import numpy as np
import pytest

import math

from safenest.estimator import BayesianAgeEstimator, EstimatorState
from safenest.privacy import (
    Accounting, PrivacyConfig, PrivacyMode, PrivacyUnit, analytic_gaussian_sigma,
    clip_llr, gaussian_mechanism_delta, laplace_noise,
)
from safenest.signals import SignalModel, release_corpus_parameters
from safenest.tiers import Tier


def test_budget_must_sum_to_one():
    with pytest.raises(ValueError):
        PrivacyConfig(budget={"linguistic": 0.5, "behavioural": 0.2,
                              "device": 0.1, "contextual": 0.1})


def test_epsilon_must_be_positive():
    with pytest.raises(ValueError):
        PrivacyConfig(epsilon_total=0.0)


def test_sensitivity_matches_the_closed_form():
    c = PrivacyConfig(clip=3.0, n_tiers=5)
    assert c.llr_sensitivity() == pytest.approx(2 * 3.0 * 4)


def test_noise_scale_is_sensitivity_over_budget_share():
    c = PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0)
    assert c.noise_scale("linguistic") == pytest.approx(c.llr_sensitivity() / 0.40)


def test_local_dp_snr_is_independent_of_the_clip_bound():
    """The result that makes eps=1.0 local DP unusable for this task."""
    a = PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, clip=0.5)
    b = PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, clip=50.0)
    assert a.per_release_snr("linguistic") == pytest.approx(b.per_release_snr("linguistic"))
    assert a.per_release_snr("linguistic") == pytest.approx(0.4 / 8)


def test_basic_composition_grows_linearly_under_per_interaction_accounting():
    c = PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0)
    assert c.basic_composition(10) == pytest.approx(10.0)
    assert c.advanced_composition(10) > 0


def test_corpus_and_none_modes_do_not_compose_over_interactions():
    for mode in (PrivacyMode.CORPUS, PrivacyMode.NONE):
        c = PrivacyConfig(mode=mode, epsilon_total=1.0)
        assert c.basic_composition(1000) == pytest.approx(1.0)


def test_privacy_units_are_mode_specific():
    assert PrivacyConfig(mode=PrivacyMode.CORPUS).unit is PrivacyUnit.CORPUS_CHILD
    assert PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE).unit is PrivacyUnit.INTERACTION
    assert PrivacyConfig(mode=PrivacyMode.CORPUS).accounting is Accounting.ONE_SHOT


def test_corpus_sensitivity_matches_the_closed_form():
    c = PrivacyConfig(mode=PrivacyMode.CORPUS, corpus_n_per_tier=400,
                      corpus_clip_sd=3.0, corpus_n_features=5)
    assert c.corpus_l2_sensitivity() == pytest.approx(math.sqrt(2) * 6 * math.sqrt(5) / 400)


def test_corpus_noise_shrinks_with_corpus_size():
    small = PrivacyConfig(mode=PrivacyMode.CORPUS, corpus_n_per_tier=100)
    large = PrivacyConfig(mode=PrivacyMode.CORPUS, corpus_n_per_tier=10_000)
    assert large.corpus_gaussian_sigma() < small.corpus_gaussian_sigma()


def test_analytic_gaussian_sigma_attains_the_target_delta():
    for eps in (0.1, 1.0, 10.0):
        sigma = analytic_gaussian_sigma(eps, 1e-5, 1.0)
        assert gaussian_mechanism_delta(eps, sigma, 1.0) <= 1e-5
        assert gaussian_mechanism_delta(eps, 0.99 * sigma, 1.0) > 1e-5


def test_analytic_gaussian_is_never_looser_than_the_classical_bound_below_eps_1():
    for eps in (0.1, 0.5, 0.9):
        c = PrivacyConfig(mode=PrivacyMode.CORPUS, epsilon_total=eps)
        assert c.corpus_gaussian_sigma() <= c.corpus_classical_sigma() + 1e-12


def test_corpus_release_perturbs_only_the_linguistic_means():
    rng = np.random.default_rng(0)
    cfg = PrivacyConfig(mode=PrivacyMode.CORPUS, epsilon_total=1.0)
    base = SignalModel()
    released = release_corpus_parameters(base, cfg, rng)
    for t in Tier:
        m0, s0 = base.linguistic_params(t)
        m1, s1 = released.linguistic_params(t)
        assert np.array_equal(s0, s1)
        assert not np.array_equal(m0, m1)
        assert base.typing_params(t) == released.typing_params(t)


def test_local_release_is_k_minus_1_clipped_ratios_against_t1(monkeypatch):
    """Noise is drawn for exactly K-1 released coordinates per modality, which is
    what the 2C(K-1) sensitivity assumes."""
    import safenest.estimator as estimator_module
    shapes = []

    def spy(scale, shape, rng):
        shapes.append(shape)
        return np.zeros(shape)

    monkeypatch.setattr(estimator_module, "laplace_noise", spy)
    rng = np.random.default_rng(1)
    est = BayesianAgeEstimator(
        privacy=PrivacyConfig(mode=PrivacyMode.LOCAL_INFERENCE, epsilon_total=1.0))
    est.observe(EstimatorState(), SignalModel().sample(Tier.T3, rng), rng)
    assert shapes == [(4,)] * 4


def test_describe_reports_the_four_required_elements():
    for mode in PrivacyMode:
        d = PrivacyConfig(mode=mode).describe()
        assert {"privacy_unit", "adjacency", "protected_output", "mechanism"} <= set(d)


def test_none_mode_makes_no_epsilon_claim():
    d = PrivacyConfig(mode=PrivacyMode.NONE).describe()
    assert d["epsilon_total"] is None
    assert "NOT an eps guarantee" in d["claim"]


def test_clip_bounds_the_llr():
    x = np.array([-10.0, 0.0, 10.0])
    assert np.array_equal(clip_llr(x, 3.0), np.array([-3.0, 0.0, 3.0]))


def test_laplace_noise_has_the_requested_scale():
    rng = np.random.default_rng(0)
    s = laplace_noise(2.0, (200_000,), rng)
    assert np.isclose(s.std(), 2.0 * np.sqrt(2), rtol=0.05)
