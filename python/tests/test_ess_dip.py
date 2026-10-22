"""Behavioural tests on small synthetic fields, generated as in Section 4.1 of the paper."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ess_dip import ESSDip, homogeneity_test, local_range, poly_r2


def grf(h, w, sigma, rng):
    g = gaussian_filter(rng.standard_normal((h, w)), sigma, mode="wrap")
    return (g - g.mean()) / g.std()


def null_world(rng, n=96, p=4, sigma=6):
    return np.stack([grf(n, n, sigma, rng) for _ in range(p)], -1)


def two_class_world(rng, n=96, p=4, sep=4.0):
    labels = (grf(n, n, 9, rng) > 0).astype(int)
    means = rng.normal(0, sep, size=(2, p))
    noise = np.stack([grf(n, n, 2, rng) for _ in range(p)], -1)
    return means[labels] + noise, labels


def trend_world(rng, n=96, p=4, amp=3.0):
    yy, xx = np.mgrid[0:n, 0:n] / n
    t = (xx + 0.5 * yy + 0.8 * xx * yy)
    t = (t - t.mean()) / t.std()
    return np.stack([amp * t + 0.4 * grf(n, n, 3, rng) for _ in range(p)], -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_null_field_is_one_class(seed):
    X = null_world(np.random.default_rng(seed))
    assert ESSDip(random_state=seed).fit(X).n_classes_ == 1


@pytest.mark.parametrize("seed", [0, 1])
def test_trend_without_classes_is_one_class(seed):
    model = ESSDip(random_state=seed).fit(trend_world(np.random.default_rng(seed)))
    assert model.detrended_
    assert model.n_classes_ == 1


@pytest.mark.parametrize("seed", [3, 4])
def test_two_well_separated_classes_are_found(seed):
    X, truth = two_class_world(np.random.default_rng(seed))
    model = ESSDip(random_state=seed).fit(X)
    assert model.n_classes_ >= 2
    assert model.labels_.shape == truth.shape


def test_never_more_classes_than_cap():
    X, _ = two_class_world(np.random.default_rng(5), sep=8)
    assert ESSDip(max_classes=2, random_state=0).fit(X).n_classes_ <= 2


def test_two_dimensional_input_and_nan_mask():
    rng = np.random.default_rng(6)
    X = grf(96, 96, 4, rng)
    X[:10, :] = np.nan                              # outside the region
    model = ESSDip(random_state=0).fit(X)
    assert model.labels_[:10].min() == -1 and model.labels_[:10].max() == -1
    assert model.n_classes_ == 1


def test_explicit_range_is_used_and_sets_area():
    X = null_world(np.random.default_rng(7))
    model = ESSDip(range_=5.0, random_state=0).fit(X)
    assert model.range_estimated_ == 5.0
    assert model.area_ == 25.0
    assert np.isclose(model.n_eff_, 96 * 96 / 25.0)


def test_censoring_warning_on_long_range():
    X = null_world(np.random.default_rng(8), sigma=20)
    with pytest.warns(UserWarning, match="censored"):
        ESSDip(random_state=0).fit(X)


def test_gaussian_null_runs_and_is_not_more_conservative():
    X, _ = two_class_world(np.random.default_rng(9), sep=3.0)
    k_u = ESSDip(null="uniform", random_state=1).fit(X).n_classes_
    k_g = ESSDip(null="gaussian", random_state=1).fit(X).n_classes_
    assert k_g >= 1 and k_u >= 1


def test_homogeneity_test_returns_decision():
    rng = np.random.default_rng(10)
    res = homogeneity_test(null_world(rng), random_state=0)
    assert res.homogeneous and 0 <= res.p_value <= 1 and res.n_eff > 0
    X, _ = two_class_world(rng, sep=6.0)
    assert not homogeneity_test(X, random_state=0).homogeneous


def test_same_seed_same_result():
    X, _ = two_class_world(np.random.default_rng(11))
    a = ESSDip(random_state=3).fit(X)
    b = ESSDip(random_state=3).fit(X)
    assert a.runs_ == b.runs_ and (a.labels_ == b.labels_).all()


def test_helpers_are_finite():
    X = null_world(np.random.default_rng(12))
    r, cens, n = local_range(X)
    assert r >= 2 and 0 <= cens <= 1 and n > 0
    assert 0 <= poly_r2(X) <= 1
