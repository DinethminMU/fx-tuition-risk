# fx-tuition-risk

**Simulating Currency Risk in Overseas Tuition Payments for Sri Lankan Families: A Monte Carlo Study**
**Live demo:** `docs/index.html` (open it in a browser, or via GitHub Pages once published)

> How many additional rupees must a family set aside today to be 95% confident of covering a
> **GBP 20,000** tuition payment due in **3, 6 or 12 months**?

The GBP/LKR rate is modelled with **geometric Brownian motion** calibrated to daily **Central Bank of Sri Lanka**
indicative rates. Thousands of future paths are simulated, and the **95% Value at Risk** (the reserve) and
**Expected Shortfall** of the rupee shortfall are estimated with confidence intervals over independent replications.
A **two-state Markov regime-switching** model (calm/crisis) is the stretch goal. Both models are backtested against
what actually happened, including the 2022 float of the rupee.

## Results (CBSL data to 20 November 2026)

GBP/LKR = 435.56, so the budget at today's rate is LKR 8.71 million. Extra rupees to set aside for 95% confidence:

| Horizon | GBM VaR | Regime VaR | GBM ES | Regime ES | Backtest breach rate (target 5%) |
|---|---|---|---|---|---|
| 3 months | 1,012,972 (11.6%) | 1,046,446 (12.0%) | 1,254,211 | 1,453,497 | 6.0% |
| 6 months | 1,552,964 (17.8%) | 1,624,835 (18.7%) | 1,918,326 | 2,159,670 | 8.4% |
| 12 months | 2,461,894 (28.3%) | 2,562,265 (29.4%) | 3,027,589 | 3,318,731 | 14.5% |

The 3-month reserve passes its backtest; the 6- and 12-month reserves are breached more often than 5%. No rule
calibrated on pre-2022 data would have covered the March 2022 float of the rupee. See Sections 6–7 of the notebook.

## Repository layout

```
src/fxrisk/          the library
  data.py            CBSL data via the Frankfurter API, CSV loading/caching
  gbm.py             GBM calibration, exact simulation, Euler–Maruyama, Milstein, closed forms
  risk.py            VaR / ES, replications and Student-t confidence intervals
  rvgen.py           input modelling (Normal vs Student-t), Box–Muller, inverse transform
  regime.py          Hamilton filter, Kim smoother, ML fit, DTMC simulation
  backtest.py        rolling out-of-sample backtest, Kupiec test
notebooks/
  fx_tuition_risk.ipynb   the full study (narrative, results, figures, verification asserts)
tests/               pytest suite (31 tests)
figures/             figures written by the notebook
data/cbsl_gbp_lkr.csv  CBSL GBP/LKR indicative rates, 20 Jan 2011 – 20 Nov 2026
tools/build_notebook.py   regenerates the notebook from source
```

## Quick start

```bash
git clone https://github.com/<your-username>/fx-tuition-risk.git
cd fx-tuition-risk
pip install -e ".[dev]"
pytest -q                                   # 31 verification tests
jupyter notebook notebooks/fx_tuition_risk.ipynb
```

**Google Colab:** `!git clone …`, then `%cd fx-tuition-risk/notebooks`, then *Run all*. Colab already has
every dependency installed.

On the first run, the notebook downloads the CBSL series from
`https://api.frankfurter.dev/v2/rates?base=GBP&quotes=LKR&providers=CBSL` and caches it to
`data/cbsl_gbp_lkr.csv`. You can also place a CSV there yourself, either the Frankfurter export
(`date,base,quote,rate`) or a plain `date,rate` file. If neither the file nor the network is available, the code falls
back to a clearly labelled **synthetic** series so that it still runs; synthetic output is never valid as a result.

## Methods

| Step | Method | Module topic |
|---|---|---|
| Input modelling | Log-returns; Normal vs Student-t fits (loglik, AIC, KS); Jarque–Bera; rolling volatility | Input modelling |
| Variate generation | Box–Muller normals (KS-tested); inverse transform for regime switches | Random variate generation |
| Calibration | Maximum likelihood GBM with standard errors | Input modelling |
| Simulation | Exact GBM solution; Euler–Maruyama and Milstein time-stepping | Monte Carlo; numerical solvers |
| Risk measures | 95% VaR and ES of `20,000 × (S_T − S_0)`; 30 replications × 100k paths; Student-t CIs | Monte Carlo integration, LLN |
| Stretch model | 2-state Markov switching; Hamilton filter ML; stationary distribution | Markov chains (DTMC) |
| Backtest (X4) | Monthly planning dates 2014–present, expanding/trailing windows, Kupiec test | — |
| Sensitivity (X5) | Calibration window, μ ± 2 SE, σ ± 2 SE, confidence level, (μ, σ) surface | — |
| Vectorisation (X6) | Pure-Python loops vs NumPy across paths vs full array vs exact sampler | — |

## Verification (X2)

All of these are `assert`s in the notebook and/or tests in `tests/`:

* Monte Carlo VaR and ES agree with the closed-form lognormal values within 4 standard errors and 1% relative error.
* The Monte Carlo mean agrees with `S0·exp(μT)` within 4 standard errors, and the running mean converges (LLN).
* The RMSE of the quantile estimator falls like `N^(-1/2)` (measured log–log slope).
* Euler–Maruyama strong order ≈ 0.5, weak order ≈ 1; Milstein strong order ≈ 1, measured on log–log plots using the same Brownian paths.
* MLE calibration recovers known GBM and regime parameters from simulated data.
* The Hamilton-filter log-likelihood matches `statsmodels.MarkovRegression`.
* Transition-matrix rows sum to 1; `πP = π`; simulated regime frequencies match the stationary distribution.
* The backtest has no look-ahead (each calibration window ends exactly on the planning date).

## Data cleaning

Before July 2013 the CBSL series has gaps of up to 65 days. A return across such a gap is not a daily return, so
the model is calibrated on the regular segment from 29 July 2013 (3,216 observations, about 241 a year); see
`data.regular_segment`.

## Limitations

CBSL indicative rates exclude the bank's spread on telegraphic transfers; interest earned on the rupee reserve
is ignored; GBM understates tail risk; and a policy-driven devaluation like 2022's is close to a structural break.
See Sections 6–7 of the notebook.

## Licence

MIT. See `LICENSE`. Data: Central Bank of Sri Lanka, via the Frankfurter API.

## AI use

Developed with assistance from Claude (Anthropic). See `AI_USAGE_LOG.md`.
