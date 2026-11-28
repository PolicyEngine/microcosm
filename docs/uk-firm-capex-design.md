# UK firm capital expenditure by asset class: design note

Status: design only. Nothing is built and nothing is fed to policyengine-uk.

## Why it is not fed to pe-uk

`firm_taxable_trading_profit` is allotted from HMRC CT liability bands. Those are profits after capital allowances and after losses brought forward. pe-uk's capital allowance inputs (`firm_capex_main_pool`, `firm_capex_special_rate_pool`, full expensing and the AIA) deduct allowances again. So supplying capex while profit is drawn after allowances would deduct them twice.

The capex columns become pe-uk inputs only when profit is drawn **before** allowances. That needs a source for profit before allowances, such as HMRC CT statistics Table 11 (capital allowances claimed), or a CT600 aggregate. Losses brought forward would be fixture-only for the same reason.

## Sources (pinned 26 Nov 2026; fixture `tests/fixtures/uk/firm_capex_2024/`, tool `tools/extract_uk_firm_capex_tables.py`)

| Source | Edition | Use |
|---|---|---|
| ONS Business investment by asset, CP (Annual) | 30 Sep 2026 | 2024 business totals: transport £30.8bn (EDRB), ICT and other M&E £56.6bn (EDWH), other buildings and structures £81.1bn (EDYP), IPP £140.5bn (EDYS); total £309.1bn (NPEK) |
| ONS Annual Business Survey | 2024 results (26 May 2026) | Capex acquisitions and disposals by section A–S (£243.9bn and £39.9bn); no asset split |

## Proposed method (for a later PR)

1. **Section totals:** ABS acquisitions by section, minus disposals, are the business total. They cover the non-financial business economy only, so K and public bodies are excluded.
2. **Asset split:** each section takes the Blue Book 2026 GFCF by industry and asset mix of its divisions (not pinned yet; pin it when the method is built). The mix is whole-economy, so it is an "assumed" proxy for private firms. Sector mixes are then raked so that the asset totals match ONS business investment by asset.
3. **Within section:** capex is shared across firms by turnover, in proportion. Firms with no turnover get 0. This is assumed. ABS publishes no size split, so the share by size band is an open question.
4. **Mapping to pe-uk pools:**
   - Plant and machinery (ICT and other M&E) goes to the main pool or full expensing.
   - Long-life items and integral features go to the special-rate pool. These have no source, so the split is assumed.
   - Structures go to SBA (3%).
   - IPP is not a capital allowance; it falls under intangible fixed assets (CTA 2009 Part 8) and R&D.
   - Vehicles go by CO2 band, and nothing pins that.
5. **Held-out check:** HMRC CT statistics on capital allowances claimed, by type.

## Open questions

- The ABS (2024) and business investment (2024) concepts differ: the ABS counts acquisitions; business investment is national accounts GFCF.
- Lumpiness: a firm-year capex draw needs a zero mass. Most small firms have no capex in a year, and nothing pinned here gives the share.
