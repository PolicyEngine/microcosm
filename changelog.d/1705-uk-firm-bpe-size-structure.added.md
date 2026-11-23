Add opt-in BPE 2025 size-structure switches to the UK firm build (microcosm#1705), all needing `--bpe-dir`:
- `--bpe-mid-headcount` remaps headcount of ONS-frame rows inside the ONS 5-9 to 100-249 bands to BPE Table 1 shapes by weighted rank, keeping bands, ranks and weights.
- `--bpe-large-headcount` gives the top private 250+ firms BPE's 1000+ and 500-999 counts and shapes (a truncated Pareto on [1000, 500,000] for 1000+, an assumption). It raises paybill to about 1.28× ASHE annual jobs × mean, so it stays out of the reference build.
- `--out-of-frame-no-employees` gives the HMRC zero-turnover traders outside the ONS frame no headcount, an assumption with no published pin.
- `--bpe-turnover` draws the open ONS turnover band on a truncated Pareto tail pinned to BPE whole-economy turnover instead of the log-uniform draw capped at £50m, and re-pairs employment with turnover inside each SIC division by a Gaussian copula fitted to BPE turnover by employment band.
With `--bpe-dir` the receipt also adds a report-only ratio (`paybill_vs_bpe_employees_x_ashe_mean`) of paybill to BPE employees × ASHE all-employee mean beside the usual ASHE annual-jobs check.
