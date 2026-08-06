import numpy as np
import pytest

from safenest.privacy import (
    Accounting, PrivacyConfig, PrivacyMode, PrivacyUnit, clip_llr, laplace_noise,
)


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


def test_corpus_parameter_noise_shrinks_with_corpus_size():
    small = PrivacyConfig(mode=PrivacyMode.CORPUS, corpus_n_children=100)
    large = PrivacyConfig(mode=PrivacyMode.CORPUS, corpus_n_children=10_000)
    assert large.corpus_parameter_noise_std() < small.corpus_parameter_noise_std()


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
