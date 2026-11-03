The UK parity register now compares Microcosm with uk-data at `a43e8c5` (15 October 2026) rather than `8629dbb` (microcosm#1278). The inventory has 57 modules: two source files and a test changed, and three sources are new. The re-extraction changes no existing concern's classification; one is corrected by hand (the council-tax outlier guard, below). A second re-pin follows when uk-data#482 and #484 merge.

- **Four new concerns.**
  - Pension-age Housing Benefit and Pension Credit are ported (`dwp.hb.amount_pension_age` and the pension-age caseloads; `obr.pension_credit` and the `dwp.pension_credit` rows). The Housing Benefit age split differs and the concern says how: uk-data applies the pension-age regulations, leaving out units on a working-age means-tested benefit, while Microcosm's rows read `eldest_adult_age` alone; re-binding goes with microcosm#867.
  - Working-age Housing Benefit is routed to microcosm#867 and fenced against binding the full line.
  - LFS employment is a blocked source, since the feed has no LFS series.
- **Updated citations.** The five `obr.py` line citations point at the moved functions.
- **The local employment-income count.** uk-data#529 made it validation-only in the incumbent's local calibration, outside the inventory's scope; its concern says so, and that whether Microcosm follows is a ruling owed on microcosm#1095.
- **Two corrected verdicts.**
  - `local_council_tax_band_d_rate`'s origin now says Band D entered uk-data's loss only as the multiplier in England's `housing/council_tax_net` proxy.
  - `credibility_local_council_tax_outliers` is a ported compile contract: Microcosm binds dwelling counts per authority from that authority's own row, and the local compile-parity preflight pins every cell.
