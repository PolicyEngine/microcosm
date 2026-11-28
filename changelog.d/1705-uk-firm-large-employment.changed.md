`UKFirmGenerationConfig` gains `large_employment_mean` (microcosm#1705), which lifts the 2,000-staff cap on UK firms with 250 or more employees. By default (`None`) the open 250+ band keeps the paper's draw: a lognormal (median 400, log-sd 0.8) clamped to [250, 2000], whose weighted mean is about 620. With a mean set, `assign_employment` ranks the band's rows by that same lognormal draw, so torch's generator stream and every other column are unchanged, and gives them consecutive base-weight-sized slices of a truncated Pareto on [250, 500,000] (`LARGE_EMPLOYMENT_CAP`) with that mean. The band's base-weighted mean is then the chosen mean, with no sampling noise from the heavy tail. `LARGE_EMPLOYMENT_MEAN`, 1,347.3, is DBT's private-sector mean: Business population estimates 2024, Table 1 (UK private sector, start 2024), 11,115 thousand employment over 8,250 businesses with 250 or more employees. At that mean the fitted shape (about 1.14) puts 54% / 25% / 21% of the band in 250–499 / 500–999 / 1,000+, against DBT's 51% / 26% / 23%. The three DBT rows are fixtured, with the workbook's sha256, under `tests/fixtures/uk/firm_2024_25_size/`.

The default is unchanged because the model's 250+ population is not DBT's. On the 20-iteration thinned 2024-25 test build (seed 42), the band holds 12,898 weighted firms: 11,387 in the ONS frame, close to BPE's whole-economy 11,265 (Table 2: private, government and non-profit), and 1,511 among the appended zero-turnover HMRC traders. DBT counts 8,250 private businesses. Measured on that build:

| 250+ setting | 250+ employment | 250+ employment outside ONS statuses 5–6 | Total weighted paybill ÷ ASHE £1.007tn | Paybill in firms above £3m |
| --- | --- | --- | --- | --- |
| Default (clamped at 2,000) | 7.98m (mean 618) | 7.33m | 1.06 | £486bn |
| DBT mean, 1,347.3 | 17.85m (mean 1,384) | 16.15m | 1.48 | £901bn |
| DBT total, mean 839.8 | 11.01m (mean 854) | 10.01m | 1.19 | £618bn |

The DBT mean overshoots DBT's 11.1 million private 250+ employment and falls outside the ASHE paybill guard in `test_uk_firm_extended_export.py` (0.85–1.25), which is why it is opt-in.

The default stays the legacy draw. Changing it is blocked until public and non-profit 250+ firms are separated and out-of-frame zero-turnover traders stop drawing 250+ headcounts (1,511 weighted firms on the thinned build). Paybill is already 1.064× ASHE jobs × mean at 7.98m 250+ employment, so reaching DBT's 11.1m may also need sub-250 employment or per-head pay rebalanced.

The "DBT total" row sets the mean to DBT's 11,115,000 divided by the thinned build's base-weighted 250+ count (13,235). It was measured on one local build and is not fixtured. A mean the Pareto cannot hit exactly, outside about 263 to 56,876 on [250, 500,000], is refused.
