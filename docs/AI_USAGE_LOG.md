# AI usage log

The course requires a section describing exactly how AI was used. This file is
the raw log that section will be written from. It is kept honest on purpose:
it records what the assistant did and what the student decided or checked.

## 2026-11-10 (session 1, Claude)

**Asked by the student**
- Read the course material (three slide decks, lecture notes, three labs, exercise sheet).
- Assess whether a portfolio built on congressional disclosures is feasible and fits the course.
- Start building.

**Done by the assistant**
- Literature and feasibility search (legal status of congressional trading, academic evidence, data sources, brokers). Sources are listed in the conversation.
- Proposed the two convex-relaxation angles: sparse regression (LASSO) for "which members matter", Robust PCA for "herd vs unusual purchases".
- Wrote the whole code base of this repository (data connectors, screen, event table, models, optimiser, backtest, tests).
- Ported the balance-sheet screen from the student's earlier screening engine (tag priorities and sector tables).
- Designed and ran the detectability study (`scripts/detectability.py`) and the synthetic end-to-end check.
- Found and fixed its own errors during testing: the first random-portfolio benchmark paid far more
  trading costs than the strategy it was compared with (it redrew names at every rebalance), which made
  any strategy look significant. It now mirrors the strategy's holding pattern exactly.

**Decided by the student**
- The idea and the scope (congressional purchases, long only, small real account).
- The universe constraint and its strictness (four standards, all required, missing = excluded).
- "Take whichever method tests best" for the modelling choice.

**To be done by the student before submission** (the professor may ask for an oral defence)
- [ ] Re-derive why l1 is the convex envelope of l0 on the unit cube and connect it to `models/sparse.py`.
- [ ] Explain the two proximal steps of `models/rpca.py` (singular value thresholding, soft thresholding).
- [ ] Explain why the training set must stop where the last outcome was observed ("purging").
- [ ] Explain what the random-portfolio benchmark controls for that the S&P 500 does not.
- [ ] Read the detectability figure and state its consequence for the project in two sentences.
- [ ] Decide the go / no-go rule for real money before looking at the real backtest.

## 2026-11-10 (session 1, continued)

**First contact with the live sources (20-filing probe on GitHub Actions)**
- Senate site: 18 of 20 filings read, every filer matched to a member.
- House site: only 9 of 20 filings read. The assistant's parser assumed the asset-type code always came
  before the dates; in real filings long names wrap and the code lands on the next line. The parser was
  rewritten from the real text and two live filings were added as test fixtures.
- Prices: one ticker of 26 missing (a company acquired in 2025): a first measured case of the
  delisted-stock problem.

**Confirmed by the student and added by the assistant**
- Locked final test (last 24 months untouched until all choices are frozen), factor regression,
  deflated Sharpe ratio.
- Disclosed sales used as a veto, ignoring sales from large "liquidation" filings.
- Option purchases read as signals on the underlying stock (options themselves are never traded).
- New candidate features: committee chair or ranking member, party holding the chamber, home-state company,
  first-term member, member with hardly any trading history.
- Resumable, time-boxed download of the history.

**Decided by the student**
- No instrument other than plain stocks; no borrowing.
- Not to automate a reaction to a possible trading ban: the student stops the system himself.

## 2026-11-11 (session 2)

**Full House history downloaded (2014-2026, 8,440 filings, 63 minutes on GitHub Actions)**
- 5,987 filings are electronic and were read; 2,453 (29%) are scans of paper forms and were not
  (more than half of the filings in 2014, about one in eight in 2025-2026). This is a selection
  effect to state in the report: members who file on paper are missing.
- The Senate site refused the connection on that run (it had answered on both probes).

**Three reading errors found by the assistant when profiling the result, and their fix**
- The text layer of many PDFs has broken letter case ("(aaPL) [sT]"). The parser expected capitals, so
  about 5,500 stock rows had no ticker and about 2,300 were not recognised as stocks.
- Before April 2018 the form has no asset-type code at all: every 2014-2017 row had been stored as
  "not a stock". The type is now read from the wording (a cell ending with a ticker that is not an option
  or a fund), and each row records how it was identified (`id_src`), so results can be recomputed
  without the inferred rows.
- Transaction ids and owner codes at the start of the cell were read as part of the name.
- Effect measured on the stored rows: usable stock rows 27,027 -> about 44,500.

**Design change proposed by the assistant**
- The extracted text of every electronic filing is now kept (one small file per filing month) and the
  trade table is rebuilt from it on each run, so a parser correction no longer needs a new download
  and the parser can be checked against the whole corpus.
- A name-to-ticker matching step was tried and dropped: it recovered only 93 rows and added a risk of
  wrong matches.

**Later the same day**
- Text of all 5,987 electronic House filings stored; reader checked against it: 57,671 of 57,673 transaction
  lines recognised. Descriptions and account names are now read: they give the direction of option trades,
  identify dividend reinvestments and similar non-decisions (dropped), and flag adviser-run accounts (new feature).
- Senate: 1,813 electronic filings read (605 on paper, not readable), 15,249 trades, every filer matched.
- Prices: Yahoo throttled the first full download (half the tickers missing). The assistant added waiting and
  retries, then a second source (Alpaca) for renamed and delisted companies. Coverage of stock trades with
  prices: 63% -> 88% -> 98.9%. Delisted companies have prices from 2016 only, so rules are judged from 2016.
- Impossible quotes found in the free data (negative prices, one-day moves of thousands of percent on thinly
  traded foreign lines): 9 tickers dropped, 47 isolated jumps neutralised.
- First development-period results (locked period closed): no rule beats the market or a random portfolio of
  eligible stocks. The assistant noticed that the first evaluation mixed two things (ranking purchases of a
  given month, and the level of the forecast moving over time) and added a month-by-month evaluation.
- Mandate screen audited on real company accounts: debt had been read under a single tag, so about a quarter
  of the stocks that passed did so with understated debt. Corrected (wider search, same balance date for every
  item, cross-check against interest paid) and an audit table written.

**Choices made by the assistant that the student should be able to defend**
- Judging rules from January 2016 (price history of delisted companies).
- Dropping reinvested dividends, spin-offs, mergers and inheritances from the signals.
- Reading the type of pre-2018 rows from the wording, and flagging those rows (`id_src`).
- The random benchmark draws from stocks still listed today; small survivors flatter it. A size-matched
  version is planned.

**Afternoon of 11 November**
- Senate completed (1,822 electronic filings; the 9 that had failed were read on a new attempt).
- Random benchmark made size-matched (stand-ins drawn in the same size third of the eligible stocks). The
  conclusion does not change: no rule in the top 5%, most in the lower half.
- Detectability study rerun on the full structure with the measured noise (14.5%).
- First draft of the notebook written by the assistant (`notebook/congress_signal.ipynb`, built from
  `notebook/_build.py`). Its section "Use of AI" is a draft: the part on what the student checked and decided is
  left for the student to write.

**Evening of 11 November (live system, outside the course deliverable)**
- Year-by-year stability table added to the research and the notebook after the student asked whether the
  early years were relevant.
- Daily decision built in dry-run mode: target portfolio from the tested rule, order planner with safety rules,
  client for the broker's test account. First dry run on GitHub: account of 100,000 (test money), 39 stocks in
  the target, no order sent. Launch decided by the student for 1 December 2026.
