# Data

## `cbsl_gbp_lkr.csv`

Daily indicative exchange rates, **LKR per 1 GBP**, published by the **Central Bank of Sri Lanka (CBSL)**.

| Item | Detail |
|---|---|
| Source | Central Bank of Sri Lanka, *Exchange rates*: <https://www.cbsl.gov.lk/rates-and-indicators/exchange-rates> |
| Obtained via | Frankfurter open-source API, CBSL provider: <https://frankfurter.dev/providers/cbsl/> |
| Download URL | `https://api.frankfurter.dev/v2/rates.csv?base=GBP&quotes=LKR&providers=CBSL&from=2011-01-01` |
| Downloaded | 22 November 2026 |
| Coverage | 20 Jan 2011 – 20 Nov 2026, 3,507 business days |
| Columns | `date` (YYYY-MM-DD), `base` (GBP), `quote` (LKR), `rate` (LKR per GBP) |

**Cleaning.** Before July 2013 the series has gaps longer than 14 calendar days (the longest is 65 days). The notebook calibrates on the regular segment from 29 July 2013 (3,216 observations, about 241 per year); see `data.regular_segment` in `src/fxrisk/data.py`. The raw file is kept unchanged.

**Licence and use.** The rates are public data published by the Central Bank of Sri Lanka; this copy is included only so the analysis is reproducible. Credit the Central Bank of Sri Lanka as the source, and check the CBSL website for its terms of use before reusing the data. If the file is missing, the notebook downloads it again from the Frankfurter API.
