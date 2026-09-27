"""The learned tier estimator, its private release and the session rule."""
import pytest

pytest.importorskip("pandas", reason="real-data modules need the [real] extra")

import numpy as np
import pandas as pd

from safenest.features import to_unit
from safenest.learning import (
    FEATURES,
    DiscriminativeTierModel,
    GaussianTierModel,
    expand_admissible,
    outcome,
    session_decisions,
)

MEANS = {3: np.array([50.0, 6.0, 12.0, 4.0, 0.03]),
         4: np.array([60.0, 8.0, 15.0, 4.6, 0.02]),
         5: np.array([70.0, 10.0, 18.0, 5.2, 0.01])}
SDS = {t: np.array([8.0, 1.0, 2.0, 0.4, 0.005]) for t in MEANS}


def _frame(n_children: int = 300, windows: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in MEANS:
        for c in range(n_children):
            for w in range(windows):
                x = rng.normal(MEANS[t], SDS[t])
                rows.append({"doc_id": f"t{t}-{c}", "window": w, "tier": t,
                             "admissible": str(t), **dict(zip(FEATURES, x))})
    return pd.DataFrame(rows)


def test_the_gaussian_fit_recovers_the_generating_parameters():
    m = GaussianTierModel().fit(_frame())
    for t in MEANS:
        assert np.allclose(m.mean[t], MEANS[t], rtol=0.02, atol=0.002)
        assert np.allclose(m.sd[t], SDS[t], rtol=0.08, atol=0.001)


def test_the_learned_estimator_separates_well_separated_tiers():
    train, test = _frame(seed=0), _frame(n_children=50, seed=1)
    m = GaussianTierModel().fit(train)
    dec = session_decisions(test, m.window_log_scores(test), m.tiers, (6,))
    truth = test.groupby("doc_id")["tier"].first()
    assert (dec.set_index("doc_id")["tier_n6"] == truth).mean() > 0.95


def test_the_clipped_release_without_noise_is_close_to_maximum_likelihood():
    f = _frame()
    ml = GaussianTierModel().fit(f)
    clipped = GaussianTierModel().fit_private(f, None, 1e-5, np.random.default_rng(0))
    for t in MEANS:
        assert np.allclose(clipped.mean[t], ml.mean[t], rtol=1e-6)
        # Deviations here are far inside the clip, so variances agree too.
        assert np.allclose(clipped.sd[t], ml.sd[t], rtol=0.05)


def test_the_private_release_is_noisy_but_centred():
    """Averaged over releases, the private means sit on the non-private ones,
    within three standard errors of the release noise (spelling rate, whose
    mean lies near the clipping boundary, is excluded)."""
    f = _frame(n_children=2000, windows=2)
    rng = np.random.default_rng(0)
    draws = np.stack([GaussianTierModel().fit_private(f, 1.0, 1e-5, rng).mean[4]
                      for _ in range(60)])
    ml = GaussianTierModel().fit(f).mean[4]
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert draws.std(axis=0)[:4].min() > 0        # noise was added
    assert np.all(np.abs(draws.mean(axis=0) - ml)[:4] <= 3 * se[:4] + 1e-9)


def test_replacing_one_child_moves_the_stage_one_sums_by_at_most_sqrt_d():
    """The sensitivity the mean release is calibrated to, checked directly."""
    f = _frame(n_children=20)
    mom = GaussianTierModel._child_moments(f)
    cols = [f"u_{c}" for c in FEATURES]
    before = mom[mom["tier"] == 4][cols].sum().to_numpy()
    worst = f.copy()
    victim = worst["doc_id"] == "t4-0"
    for c, extreme in zip(FEATURES, (0.0, -1e9, 1e9, 1e9, 0.0)):
        worst.loc[victim, c] = extreme
    after_mom = GaussianTierModel._child_moments(worst)
    after = after_mom[after_mom["tier"] == 4][cols].sum().to_numpy()
    assert np.linalg.norm(after - before) <= np.sqrt(len(FEATURES)) + 1e-12
    assert np.all((to_unit(worst[FEATURES].to_numpy()) >= 0)
                  & (to_unit(worst[FEATURES].to_numpy()) <= 1))


def test_boundary_documents_enter_training_with_half_weight_per_tier():
    f = pd.DataFrame({"doc_id": ["a", "b"], "window": [0, 0], "tier": [4, 0],
                      "admissible": ["4", "4,5"], **{c: [1.0, 1.0] for c in FEATURES}})
    e = expand_admissible(f)
    assert sorted(zip(e["doc_id"], e["tier"], e["weight"])) == [
        ("a", 4, 1.0), ("b", 4, 0.5), ("b", 5, 0.5)]


def test_an_unconfident_session_falls_to_the_floor():
    f = pd.DataFrame({"doc_id": ["x", "x"], "window": [0, 1]})
    flat = np.zeros((2, 3))
    sure = np.array([[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
    assert session_decisions(f, flat, [3, 4, 5], (2,))["tier_n2"].iloc[0] == 1
    assert session_decisions(f, sure, [3, 4, 5], (2,))["tier_n2"].iloc[0] == 4


def test_a_short_session_is_scored_on_the_windows_it_has():
    f = pd.DataFrame({"doc_id": ["x"], "window": [0]})
    d = session_decisions(f, np.array([[0.0, 9.0, 0.0]]), [3, 4, 5], (1, 10))
    assert d["tier_n10"].iloc[0] == d["tier_n1"].iloc[0] == 4
    assert d["n_windows"].iloc[0] == 1


@pytest.mark.parametrize("assigned,admissible,expected", [
    (5, "4,5", "correct"), (4, "4,5", "correct"), (5, "4", "under"),
    (3, "4", "over"), (1, "3,4", "over"),
])
def test_outcomes_follow_the_admissible_tiers(assigned, admissible, expected):
    assert outcome(assigned, admissible) == expected


def test_the_discriminative_model_learns_and_weights_children_equally():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = _frame(seed=0), _frame(n_children=50, seed=1)
    m = DiscriminativeTierModel(
        make=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        kind="features").fit(train)
    dec = session_decisions(test, m.window_log_scores(test), m.tiers, (6,))
    truth = test.groupby("doc_id")["tier"].first()
    assert (dec.set_index("doc_id")["tier_n6"] == truth).mean() > 0.95
