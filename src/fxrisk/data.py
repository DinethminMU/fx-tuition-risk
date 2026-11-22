"""Data access for CBSL GBP/LKR indicative exchange rates.

The Central Bank of Sri Lanka (CBSL) publishes daily indicative rates
(LKR per unit of foreign currency) at
https://www.cbsl.gov.lk/rates-and-indicators/exchange-rates .
The open-source Frankfurter API re-serves the same CBSL series in machine-readable
form (https://frankfurter.dev/providers/cbsl/), which is what we download here.

All loaders return a ``pandas.Series`` of rates (LKR per GBP) indexed by date,
sorted, de-duplicated and strictly positive.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

FRANKFURTER_URL = "https://api.frankfurter.dev/v2/rates"
FRANKFURTER_CSV_URL = "https://api.frankfurter.dev/v2/rates.csv"


def _clean(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    s.name = "GBP/LKR"
    s.index.name = "date"
    return s


def download_cbsl(
    base: str = "GBP",
    quote: str = "LKR",
    start: str = "2011-01-01",
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Download daily CBSL indicative rates from the Frankfurter v2 API.

    Requests one calendar year at a time so no single response is too large.
    Returns LKR per 1 unit of ``base``.
    """
    import requests  # imported lazily so the package works offline

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    frames = []
    for year in range(start_ts.year, end_ts.year + 1):
        lo = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        hi = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        params = {
            "base": base,
            "quotes": quote,
            "providers": "CBSL",
            "from": lo.strftime("%Y-%m-%d"),
            "to": hi.strftime("%Y-%m-%d"),
        }
        resp = requests.get(FRANKFURTER_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("Frankfurter returned no data for the requested range.")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["base"] == base) & (df["quote"] == quote)]
    return _clean(pd.Series(df["rate"].values, index=df["date"].values))


def load_csv(path: str | Path) -> pd.Series:
    """Load a CSV of rates.

    Accepts the Frankfurter CSV export (columns ``date,base,quote,rate[,providers]``)
    or any two-column ``date,rate`` file (as written by :func:`save_csv`).
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"{path}: no 'date' column found (columns: {list(df.columns)})")
    rate_col = cols.get("rate") or cols.get("gbp/lkr") or df.columns[-1]
    if "base" in cols and "quote" in cols:
        b = df[cols["base"]].astype(str).str.upper()
        q = df[cols["quote"]].astype(str).str.upper()
        # Frankfurter may return the pair either way round; normalise to LKR per GBP.
        direct = (b == "GBP") & (q == "LKR")
        inverse = (b == "LKR") & (q == "GBP")
        if direct.any():
            df = df[direct]
            vals = df[rate_col].astype(float)
        elif inverse.any():
            df = df[inverse]
            vals = 1.0 / df[rate_col].astype(float)
        else:
            raise ValueError(f"{path}: no GBP/LKR rows found")
    else:
        vals = df[rate_col].astype(float)
    return _clean(pd.Series(vals.values, index=df[cols["date"]].values))


def save_csv(series: pd.Series, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    series.rename("rate").to_frame().to_csv(path, index_label="date")


def make_synthetic_series(
    start: str = "2011-01-03",
    end: str = "2026-11-18",
    s0: float = 175.0,
    seed: int = 2026,
) -> pd.Series:
    """**Synthetic stand-in data for offline testing only.**

    Generates a business-day GBP/LKR-like path from a two-regime process with a
    scripted 2022-style devaluation. It is NOT real data and must never be used
    for reported results; it exists so the code and tests run without network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    mu = np.array([0.00004, 0.0003])  # daily drift: calm, crisis
    sd = np.array([0.0045, 0.0140])
    P = np.array([[0.995, 0.005], [0.03, 0.97]])
    s = 0
    r = np.empty(n)
    for t in range(n):
        s = s if rng.random() < P[s, s] else 1 - s
        r[t] = mu[s] + sd[s] * rng.standard_normal()
    # scripted 2022-style float: ~55% depreciation over March-April 2022
    crash = (idx >= "2022-03-08") & (idx <= "2022-04-29")
    if crash.any():
        r[crash] += np.log(1.55) / crash.sum()
    r[0] = 0.0
    prices = s0 * np.exp(np.cumsum(r))
    out = _clean(pd.Series(prices, index=idx))
    out.attrs["synthetic"] = True
    return out


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log-returns r_t = ln(S_t / S_{t-1})."""
    return np.log(prices).diff().dropna()


def find_gaps(prices: pd.Series, max_gap_days: int = 14) -> pd.Series:
    """Calendar-day gaps longer than ``max_gap_days`` (indexed by the date after the gap)."""
    idx = prices.index
    gaps = pd.Series((idx[1:] - idx[:-1]).days, index=idx[1:])
    return gaps[gaps > max_gap_days]


def regular_segment(prices: pd.Series, max_gap_days: int = 14) -> pd.Series:
    """Return the series from the end of its last long gap onwards.

    A return across a multi-week gap is not a daily return; mixing such returns
    into a daily model inflates the volatility estimate. The early CBSL series
    (2011-2013) has gaps of up to two months, so the model is calibrated on the
    regular segment only.
    """
    gaps = find_gaps(prices, max_gap_days)
    if gaps.empty:
        return prices
    return prices.loc[gaps.index[-1]:]


def observations_per_year(prices: pd.Series) -> float:
    """Median number of observations in each *complete* calendar year.

    CBSL publishes on Sri Lankan business days (~240-250 per year), so we
    measure the sampling frequency instead of assuming 252.
    """
    counts = prices.groupby(prices.index.year).size()
    years = counts.index
    full = [y for y in years if y not in (years.min(), years.max())]
    if not full:
        return float(len(prices) / max((prices.index[-1] - prices.index[0]).days / 365.25, 1e-9))
    return float(np.median(counts.loc[full]))


def get_rates(
    csv_path: str | Path = "data/cbsl_gbp_lkr.csv",
    start: str = "2011-01-01",
    allow_download: bool = True,
    allow_synthetic: bool = True,
    verbose: bool = True,
) -> tuple[pd.Series, str]:
    """Return (rates, source) using the first source that works.

    Order: local CSV -> Frankfurter download (cached to ``csv_path``) ->
    synthetic stand-in (with a loud warning).
    """
    p = Path(csv_path)
    if p.exists():
        s = load_csv(p)
        if verbose:
            print(f"Loaded {len(s)} CBSL observations from {p}")
        return s, f"CBSL via local file {p.name}"
    if allow_download:
        try:
            s = download_cbsl(start=start)
            save_csv(s, p)
            if verbose:
                print(f"Downloaded {len(s)} CBSL observations; cached to {p}")
            return s, "CBSL via Frankfurter API"
        except Exception as exc:  # network blocked, API down, ...
            if verbose:
                print(f"Download failed: {exc!r}")
    if allow_synthetic:
        if verbose:
            print("!" * 78)
            print("WARNING: using SYNTHETIC stand-in data. Results are NOT real CBSL figures.")
            print("!" * 78)
        return make_synthetic_series(), "SYNTHETIC (testing only)"
    raise RuntimeError("No data source available.")


def read_csv_text(text: str) -> pd.Series:
    """Parse CSV content held in a string (convenience for notebooks)."""
    return load_csv(io.StringIO(text))
