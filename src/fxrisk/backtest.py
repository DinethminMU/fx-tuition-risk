"""Rolling out-of-sample backtest of reserve rules against realised CBSL rates.

For each planning date t:
  1. calibrate a model on the trailing ``window_obs`` observations up to t only
     (no look-ahead),
  2. compute the rupee reserve per GBP needed for 95% cover over ``horizon_obs``,
  3. compare with the realised cost  S_{t+h} - S_t.

A *breach* is a planning date where the realised shortfall exceeded the
reserve. A correctly calibrated 95% rule should breach about 5% of the time;
the Kupiec (1995) proportion-of-failures test checks this.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from . import gbm, regime

ReserveFn = Callable[[np.ndarray, int], float]


def month_start_positions(index: pd.DatetimeIndex, start: str, end: str) -> list[int]:
    """Positions of the first observation in each calendar month in [start, end]."""
    idx = pd.DatetimeIndex(index)
    mask = (idx >= start) & (idx <= end)
    sub = pd.Series(np.arange(len(idx))[mask], index=idx[mask])
    return sub.groupby([sub.index.year, sub.index.month]).first().tolist()


def gbm_reserve_fn(periods_per_year: float, alpha: float = 0.95) -> ReserveFn:
    """Closed-form GBM VaR per GBP (exact, so no Monte Carlo noise in the backtest)."""
    def fn(window: np.ndarray, h: int) -> float:
        p = gbm.calibrate(window, periods_per_year)
        T = h / periods_per_year
        return float(gbm.analytic_quantile(window[-1], p.mu, p.sigma, T, alpha) - window[-1])
    return fn


def regime_reserve_fn(periods_per_year: float, alpha: float = 0.95, n_paths: int = 20_000,
                      seed: int = 0) -> ReserveFn:
    """Monte Carlo VaR per GBP under the 2-state switching model (warm-started fits)."""
    state: dict = {"prev": None, "calls": 0, "cache": {}}

    def fn(window: np.ndarray, h: int) -> float:
        key = (window.size, float(window[0]), float(window[-1]))
        params = state["cache"].get(key)
        if params is None:   # one fit per planning date, shared by all horizons
            r = np.diff(np.log(window))
            params = regime.fit(r, periods_per_year, warm_start=state["prev"])
            state["cache"][key] = params
            state["prev"] = params
        rng = np.random.default_rng([seed, state["calls"]])
        state["calls"] += 1
        s_T = regime.simulate_terminal(window[-1], params, h, n_paths, rng)
        return float(np.quantile(s_T, alpha) - window[-1])
    fn.state = state  # type: ignore[attr-defined]
    return fn


def zero_reserve_fn() -> ReserveFn:
    """Baseline: budget at today's rate and set nothing aside."""
    return lambda window, h: 0.0


def trailing(fn: ReserveFn, n_obs: int) -> ReserveFn:
    """Wrap a reserve rule so it only sees the last ``n_obs`` observations."""
    return lambda window, h: fn(window[-(n_obs + 1):], h)


def run(prices: pd.Series, plan_positions: list[int], horizon_obs: int,
        window_obs: int | None, reserve_fns: dict[str, ReserveFn], amount: float = 1.0,
        min_obs: int = 500) -> pd.DataFrame:
    """Run the backtest. Reserves and shortfalls are scaled by ``amount`` (GBP).

    ``window_obs=None`` uses an *expanding* window (all data up to the plan date,
    which must contain at least ``min_obs`` observations); an integer uses a
    fixed trailing window.
    """
    vals = prices.to_numpy(dtype=float)
    idx = prices.index
    rows = []
    for pos in plan_positions:
        if pos + horizon_obs >= len(vals):
            continue
        if window_obs is None:
            if pos < min_obs:
                continue
            window = vals[: pos + 1]
        else:
            if pos - window_obs < 0:
                continue
            window = vals[pos - window_obs: pos + 1]
        s0, sT = vals[pos], vals[pos + horizon_obs]
        row = {"plan_date": idx[pos], "pay_date": idx[pos + horizon_obs],
               "S0": s0, "ST": sT, "shortfall": amount * (sT - s0)}
        for name, fn in reserve_fns.items():
            res = amount * fn(window, horizon_obs)
            row[f"reserve_{name}"] = res
            row[f"breach_{name}"] = bool(row["shortfall"] > res)
            row[f"uncovered_{name}"] = max(row["shortfall"] - res, 0.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("plan_date")


def kupiec_pof(breaches: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    """Kupiec proportion-of-failures likelihood-ratio test.

    H0: breach probability = 1 - alpha. LR ~ chi^2(1) under H0.
    Caveat: overlapping horizons make breaches serially dependent, so the
    p-value is indicative only.
    """
    b = np.asarray(breaches, dtype=bool)
    n, x = b.size, int(b.sum())
    p = 1.0 - alpha
    phat = x / n if n else np.nan

    def ll(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    lr = -2.0 * (ll(p) - ll(phat)) if n else np.nan
    return {"n": n, "breaches": x, "rate": phat, "expected_rate": p,
            "LR": float(lr), "p_value": float(stats.chi2.sf(lr, 1)) if n else np.nan}


def summarise(bt: pd.DataFrame, names: list[str], alpha: float = 0.95) -> pd.DataFrame:
    rows = []
    for nm in names:
        k = kupiec_pof(bt[f"breach_{nm}"].to_numpy(), alpha)
        rows.append({
            "model": nm,
            "plans": k["n"],
            "breaches": k["breaches"],
            "breach_rate": k["rate"],
            "kupiec_p": k["p_value"],
            "mean_reserve": bt[f"reserve_{nm}"].mean(),
            "mean_uncovered": bt[f"uncovered_{nm}"].mean(),
            "worst_uncovered": bt[f"uncovered_{nm}"].max(),
        })
    return pd.DataFrame(rows).set_index("model")
