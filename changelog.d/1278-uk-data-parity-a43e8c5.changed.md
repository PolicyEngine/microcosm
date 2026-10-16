The UK parity register now compares Microcosm with uk-data at `a43e8c5` (15 October 2026) rather than `8629dbb` (microcosm#1278). The inventory has 57 modules: three source files changed and three are new. No existing concern's classification changes. A second re-pin follows when uk-data#482 and #484 merge.

- **Four new concerns.**
  - Pension-age Housing Benefit and Pension Credit are ported (`dwp.hb.amount_pension_age` and the pension-age caseloads; `obr.pension_credit` and the `dwp.pension_credit` rows).
  - Working-age Housing Benefit is routed to microcosm#867 and fenced against binding the full line.
  - LFS employment is a blocked source, since the feed has no LFS series.
- **Updated citations.** The five `obr.py` line citations point at the moved functions.
- **Two corrected verdicts.**
  - `local_council_tax_band_d_rate`'s origin now says Band D entered uk-data's loss only as the multiplier in England's `housing/council_tax_net` proxy.
  - `credibility_local_council_tax_outliers` is a ported compile contract: Microcosm binds dwelling counts per authority from that authority's own row, and the local compile-parity preflight pins every cell.
