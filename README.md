# congress-signal

**Can a portfolio follow the stock purchases of members of the US Congress?**
A study on twelve years of official disclosures from both chambers, and a test portfolio that applies the rule in public.

Members of Congress must disclose their stock trades within 45 days. Trackers and funds that copy them became popular around 2020–2023. This project tests the idea the way an investor would live it: acting only after the disclosure, paying costs, and comparing with the index and with random portfolios.

**Result (development period, January 2016 to November 2024).** The average disclosed purchase does not beat the S&P 500; no selection rule tested improves on it; the 15 most bought stocks match the index with more risk. The last 24 months of data are kept locked for one final test. Details, figures and limits: [`notebook/congress_signal.ipynb`](notebook/congress_signal.ipynb).

## What is here

| Path | Content |
|---|---|
| `notebook/congress_signal.ipynb` | the report: question, data, methods, results, limits |
| `csig/data` | readers for the House and Senate filings, prices, company information |
| `csig/models` | LASSO and comparators (walk-forward), Robust PCA (Principal Component Pursuit) |
| `csig/portfolio` | backtest with costs, random-portfolio benchmark, performance statistics |
| `csig/live`, `scripts/trade.py` | the daily decision of the test portfolio and its safety rules |
| `data/store` | tables built from the public filings (trades, text of the filings, list of eligible stocks) |
| `results` | outputs of the research scripts |
| `tests` | tests on real filings and on synthetic markets with a planted effect |

## What is not here, and why

- **Prices.** The licences of the price sources do not allow redistribution. `scripts/build_dataset.py --stage market` downloads them (free accounts).
- **The rules of the investment mandate.** The portfolio is restricted to stocks allowed by a private mandate (balance-sheet limits and excluded activities). Its rules are not published; the resulting list of eligible stocks on each date is provided in `data/store/eligibility.parquet` and is an input of the study.

## Reproduce

```bash
pip install -r requirements.txt
pytest                                              # tests
python scripts/build_dataset.py --stage ingest      # downloads and reads the filings (about 2 hours, resumable)
python scripts/build_dataset.py --stage market      # prices and company information (free API keys: SEC user agent, Alpaca)
python scripts/detectability.py                     # what size of effect could be detected at all
python scripts/run_research.py --null 200           # walk-forward experiment and portfolios, development period
```

## Test portfolio

From 1 December 2026 the rule fixed before the final test (hold, in equal parts, every eligible stock with a purchase disclosed in the last 63 trading days) runs on a broker's test account, with test money. Performance against the S&P 500 is published on the [project page](https://abenguedouar.github.io/congress-signal/); holdings are shown one month late.

## How this was built

By Abderrahmene Benguedouar, with extensive use of an AI coding assistant (Claude, Anthropic), which wrote most of the code under direction. The log of what the assistant did, including the errors it made and how they were found, is in [`docs/AI_USAGE_LOG.md`](docs/AI_USAGE_LOG.md).

## Notes

Code under the MIT licence (see `LICENSE`). Research and education only; not investment advice. The disclosure reports are public records; US law (5 U.S.C. § 13107) forbids their use for commercial purposes, and this project has none.
