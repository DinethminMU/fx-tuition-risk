"""fxrisk: Monte Carlo simulation of currency risk in overseas tuition payments.

Models the GBP/LKR exchange rate (Central Bank of
Sri Lanka indicative rate) and estimates how many extra rupees a family must set
aside today to cover a GBP-denominated tuition payment due in the future.

Modules
-------
data      Download / load CBSL GBP/LKR rates (via the Frankfurter API).
gbm       Geometric Brownian motion: calibration, exact simulation,
          Euler-Maruyama / Milstein schemes, closed-form results.
risk      Value at Risk, Expected Shortfall and replication confidence intervals.
rvgen     Input modelling and random variate generation helpers.
regime    Two-state Markov regime-switching model (Hamilton filter, DTMC simulation).
backtest  Rolling out-of-sample backtest and Kupiec coverage test.
"""

__version__ = "1.0.0"

from . import backtest, data, gbm, regime, risk, rvgen  # noqa: F401
