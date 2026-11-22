"""Tests for the regime-switching model, random-variate helpers, risk measures,
data loading and the backtest."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fxrisk import backtest, data, gbm, regime, risk, rvgen

P_TRUE = np.array([[0.99, 0.01], [0.05, 0.95]])
MU_TRUE = np.array([0.0001, 0.002])
SD_TRUE = np.array([0.004, 0.015])


def _simulate_ms_returns(n, seed=0):
    rng = np.random.default_rng(seed)
    s = regime.simulate_chain(P_TRUE, n, rng)
    r = MU_TRUE[s] + SD_TRUE[s] * rng.standard_normal(n)
    return r, s


# ---------------------------------------------------------------- Markov chain
def test_transition_matrix_rows_sum_to_one():
    regime.validate_transition_matrix(P_TRUE)
    with pytest.raises(AssertionError):
        regime.validate_transition_matrix(np.array([[0.9, 0.2], [0.1, 0.9]]))


def test_stationary_distribution_is_fixed_point():
    pi = regime.stationary_distribution(P_TRUE)
    np.testing.assert_allclose(pi @ P_TRUE, pi, atol=1e-12)
    np.testing.assert_allclose(pi, [5 / 6, 1 / 6], atol=1e-12)


def test_simulated_regime_frequencies_match_stationary():
    rng = np.random.default_rng(1)
    s = regime.simulate_chain(P_TRUE, 400_000, rng)
    pi = regime.stationary_distribution(P_TRUE)
    # autocorrelated chain: allow generous tolerance
    assert abs(s.mean() - pi[1]) < 0.01


def test_fitted_matrix_valid_and_parameters_recovered():
    r, _ = _simulate_ms_returns(6000, seed=2)
    p = regime.fit(r, 250.0)
    regime.validate_transition_matrix(p.P)
    np.testing.assert_allclose(p.sigma, SD_TRUE, rtol=0.15)
    np.testing.assert_allclose(np.diag(p.P), np.diag(P_TRUE), atol=0.03)


def test_hamilton_filter_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    r, _ = _simulate_ms_returns(3000, seed=3)
    mine = regime.fit(r, 250.0)
    res = sm.tsa.MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True).fit(
        disp=False, search_reps=5)
    assert abs(mine.loglik - res.llf) < 1e-3 * abs(res.llf)


def test_regime_simulation_reduces_to_gbm_when_regimes_identical():
    """With identical regimes, log S_T is N(h*mu, h*sd^2) exactly."""
    p = regime.MSParams(mu=np.array([2e-4, 2e-4]), sigma=np.array([0.006, 0.006]),
                        P=P_TRUE, periods_per_year=250)
    rng = np.random.default_rng(4)
    s_T = regime.simulate_terminal(100.0, p, 120, 50_000, rng)
    ks = stats.kstest(np.log(s_T / 100.0), lambda x: stats.norm.cdf(x, 120 * 2e-4, 0.006 * np.sqrt(120)))
    assert ks.pvalue > 0.001


# ---------------------------------------------------------------- random variates
def test_box_muller_is_standard_normal():
    z = rvgen.box_muller(100_000, np.random.default_rng(5))
    assert stats.kstest(z, "norm").pvalue > 0.001
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02


def test_inverse_transform_discrete():
    probs = np.array([0.2, 0.5, 0.3])
    u = np.random.default_rng(6).random(200_000)
    k = rvgen.sample_discrete(probs, u)
    np.testing.assert_allclose(np.bincount(k, minlength=3) / k.size, probs, atol=0.005)


def test_fit_candidates_prefers_t_for_heavy_tails():
    x = stats.t.rvs(3, size=5000, random_state=7) * 0.005
    tab = rvgen.fit_candidates(x)
    assert tab.loc["Student-t", "AIC"] < tab.loc["Normal", "AIC"]


# ---------------------------------------------------------------- risk measures
def test_var_es_on_known_distribution():
    x = np.random.default_rng(8).standard_normal(2_000_000)
    var, es = risk.var_es(x, 0.95)
    assert abs(var - stats.norm.ppf(0.95)) < 0.005
    assert abs(es - stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05) < 0.01


def test_analytic_var_matches_lognormal_quantile():
    var, _ = risk.analytic_var_es(400, 0.05, 0.2, 0.5, 1.0, 0.95)
    q = stats.lognorm.ppf(0.95, s=0.2 * np.sqrt(0.5), scale=400 * np.exp((0.05 - 0.02) * 0.5))
    assert abs(var - (q - 400)) < 1e-9


# ---------------------------------------------------------------- data
def test_load_frankfurter_csv_both_directions(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("date,base,quote,rate\n2024-01-02,GBP,LKR,400.0\n2024-01-01,GBP,LKR,399.0\n")
    s = data.load_csv(f)
    assert list(s.values) == [399.0, 400.0]
    g = tmp_path / "b.csv"
    g.write_text("date,base,quote,rate\n2024-01-01,LKR,GBP,0.0025\n")
    assert abs(data.load_csv(g).iloc[0] - 400.0) < 1e-9


def test_save_load_roundtrip(tmp_path):
    s = data.make_synthetic_series(end="2012-12-31")
    data.save_csv(s, tmp_path / "x.csv")
    t = data.load_csv(tmp_path / "x.csv")
    np.testing.assert_allclose(s.values, t.values)


def test_regular_segment_drops_data_before_last_long_gap():
    idx = pd.to_datetime(["2012-01-02", "2012-01-03", "2012-03-01", "2012-03-02",
                          "2012-04-20", "2012-04-23", "2012-04-24"])
    s = pd.Series(np.arange(1.0, 8.0), index=idx)
    out = data.regular_segment(s, max_gap_days=14)
    assert out.index[0] == pd.Timestamp("2012-04-20") and len(out) == 3
    assert len(data.find_gaps(s, 14)) == 2


# ---------------------------------------------------------------- backtest
def test_backtest_has_no_lookahead():
    s = data.make_synthetic_series(end="2015-12-31")
    seen = []

    def spy(window, h):
        seen.append(window[-1])
        return 0.0

    pos = backtest.month_start_positions(s.index, "2013-01-01", "2014-12-31")
    bt = backtest.run(s, pos, horizon_obs=60, window_obs=250, reserve_fns={"spy": spy})
    np.testing.assert_allclose(seen, bt["S0"].values)   # window ends exactly at the plan date


def test_kupiec_at_expected_rate():
    b = np.zeros(1000, bool)
    b[:50] = True
    k = backtest.kupiec_pof(b, 0.95)
    assert k["LR"] < 1e-9 and k["p_value"] > 0.99
    b[:150] = True
    assert backtest.kupiec_pof(b, 0.95)["p_value"] < 1e-6


def test_gbm_reserve_fn_matches_analytic():
    s = data.make_synthetic_series(end="2014-12-31")
    w = s.values[-500:]
    fn = backtest.gbm_reserve_fn(250.0)
    p = gbm.calibrate(w, 250.0)
    expected = gbm.analytic_quantile(w[-1], p.mu, p.sigma, 125 / 250, 0.95) - w[-1]
    assert abs(fn(w, 125) - expected) < 1e-9
