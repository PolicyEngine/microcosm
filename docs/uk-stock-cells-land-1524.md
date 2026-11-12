# Land under homes in England and Wales: the stock cell table (microcosm#1524)

This note describes the first output of the land value tax project
(PolicyEngine/microcosm#1521, scoped in PolicyEngine/uk-land-value-tax#20): a
table of every home in England and Wales, pooled into cells of a small area
(an MSOA) × a council tax band, with what the homes are worth, what their
buildings would cost to rebuild after depreciation, and the land as the
remainder. Authority and constituency figures are sums of cells.
`tools/build_uk_stock_cells.py` builds it; `uk_runtime.stock_cells` reads it.

The builder reads public administrative data only. It reads no survey
microdata, so its outputs need no disclosure floor. Addresses stay in the
build caches; the resources hold aggregates.

## The resources

| File | Holds |
|---|---|
| `uk/stock_cells_value.json` | One row per cell with homes: VOA's count, the private and social counts, homes seen sold, the owner-occupied share of the private stock, the mean value and 13 value knots at calendar-2024 prices, and where the cell's means and knots come from. Also each cell's split across constituencies, the social homes per authority and each MSOA's authority. |
| `uk/stock_cells_structure.json` | Aligned to the value block: floor area, the share of the building left after depreciation, the unit cost at κ = 1, cost and value along the value rank, and the central case's land per home. |
| `uk/la_land_shares.json` | Per authority: dwellings, social homes, flags, and land (and value where it differs) under every variant and under MHCLG's pattern. λ = land ÷ value is the loader's. |
| `uk/constituency_land_totals.json` | Land and value by July 2024 constituency, every variant. |
| `uk/stock_cells.provenance.json` | Every input's pin and licence, the parameters, the level and its status, the receipts, the decision rule and the measurements. |

The digests chain: the structure block records the value block's digest, the
rollups record both, and the provenance records all four. The structure
block, the rollups and the provenance never feed the solve
(`solve_scope.UK_POST_SOLVE_RESOURCES`); the value block does, since the
frame's home values will be read from it.

## Inputs

Every raw input is checked against its Chronicle registration (sha256 and
size) before it is read; the provenance records each registration and
licence.

| Input | Chronicle package | What the builder reads |
|---|---|---|
| HM Land Registry Price Paid, 1995 to August 2026 | `hmlr-price-paid-data-yearly-august-2026` (raw only) | category A sales of houses and flats, through the home-stock builder's dwelling table (#1240) |
| UK House Price Index | `hmlr-uk-hpi-full-file-july-2026` (raw only) | each sale moved to the calendar-2024 average of its authority and type |
| ONS NSPL, August 2026 | `ons-nspl-august-2026` (raw only) | each postcode's 2021 output area |
| England and Wales atomic support | committed (`uk_atomic_area_supports.provenance.json`) | output areas with their MSOA, authority, region, constituency and households |
| VOA CTSOP 1.1, 3.1, 4.1 (31 March 2026) | `voa-ctsop1-1-lsoa-2026`, `voa-ctsop3-1-lsoa-2026`, `voa-ctsop4-1-lsoa-2026` (raw only) | counts by MSOA × band, type shares, build periods |
| UCL's Price Paid–EPC linked file and certificates | `ucl-house-price-per-square-metre-856542` (raw only) | each linked sale's floor area and age band; social rented certificates |
| Tax Policy Associates' model | `tpa-lvt-model-2026` (raw only) | cost per m² by built form and regional factors (read as syntax, never run); TPA's own land shares, for a check |
| Chronicle consumer feed | see the provenance's `ledger_feed` | Census 2021 TS054 tenure, MHCLG's land value estimates and densities, the Market Value Survey 2024, Band D and tax base, EPC D3, ONS's land under dwellings, the balance sheet land rows, LPS's housing stock |

## Method

### 1. The table of homes

The home-stock builder's dwelling table: each home's latest category A sale
that is not a bulk transfer, moved to the calendar-2024 average of the UK HPI
for its authority and type, and placed through its postcode's output area in
the committed support. A home is its postcode, PAON and SAON. The table is
cached under the build's cache folder and shared with the home-stock builder;
its name records the NSPL it indexes.

Each home then takes its floor area and construction age band from UCL's
linked file, joined on the same dwelling key. Each attribute comes from the
home's latest linked sale that carries it, so a home sold after October 2023
takes an earlier linked sale's. A home with no linked floor area takes its
cell's mean (the provenance counts them).

### 2. Cells and counts

A cell is an MSOA × a council tax band. Its count is VOA's at 31 March 2026
as printed (CTSOP 1.1), so homes that never sold are counted. VOA prints a
count from 1 to 4 as `[c]`; it reads as 2.5, and the provenance counts the
cells imputed.

### 3. Bands

Price Paid has no band. Within each MSOA × property type, homes are ranked by
value (ties by dwelling key) and take bands against the private quota of each
band: VOA's count × the type's share in that MSOA and band (CTSOP 3.1, its
43 type columns grouped into detached, semi-detached, terraced and flats;
bungalows join the house types in the row's house mix, and annexes, caravans
and unknown types spread over all four).

### 4. Tenure and social rented homes

An authority's social rented homes are VOA's count there × the Census 2021
social rented share (TS054), as in the home-stock builder. They spread over
the authority's MSOAs by each MSOA's own census share when the feed carries
TS054 by MSOA (requested in PolicyEngine/chronicle#438, item 3), and in
proportion to each MSOA's count until then; the provenance records which.
Within an MSOA every band loses the same share. Social rented homes rarely
sell, so they form one cell per authority: their count by the rule above,
their value the Market Value Survey's regional social rented mean (moved
from July 2024 to the calendar-2024 basis by the region's UK HPI), and their
floor area and age from the social rented certificates in UCL's bundle 2.
Wales, which the survey does not cover, takes England's ratio of social to
private-sector means × its own sales-based private mean.

### 5. Weights and thin cells

Each sold home weighs its stratum's private quota over the stratum's sold
homes, so a cell's mean is post-stratified by type. A cell with fewer than 30
sold homes reads its means from its authority × band, then its region ×
band, then the nation × band; a cell with fewer than 100 reads its knots the
same way. A fallback never leaves the cell's own parent, and the provenance
counts the cells at each level.

### 6. The value block

Per cell: the counts, the owner-occupied share of the private stock, the
exact weighted mean value and 13 knots of the value distribution (0, 0.05,
0.1, 0.2 … 0.9, 0.95, 1), capped at £20 million, with the exact mean of the
capped values beside them, so totals never depend on the grid.

### 7. Building cost

Unit cost = floor area × TPA's cost per m² for the built form × TPA's
regional factor × the share of the building left after depreciation. κ = 1
is TPA's own level. TPA has no factor for Wales, which takes England's
stock-weighted mean factor.

### 8. Depreciation

Depreciation is `min(cap, θ × shape(age))`, with the age from the EPC's
construction age band and θ scaled so the cost-weighted mean depreciation
equals one less ONS's net ÷ gross capital stock of dwellings: the whole
economy's MJF8 and MJF7 for 2024, from ONS's capital stocks of 27 November
2025 (Tables 2.1.1 and 1.1.1). PolicyEngine/chronicle#439 registers them; until
the pinned feed carries them, the build takes them as stated parameters, read
from the workbook that PR pins (sha256 `71a995c3…0de7`). The central shape is
a straight line in age, capped at 0.8; the workbooks state no depreciation
profile, so whether ONS's is close to it is not verified. Without the ratio
the build runs with depreciation off and says so.

E46Y is a net stock, so the two must go together: with depreciation off, the
level absorbs the average depreciation instead (see the results).

### 9. The level

One national cost level κ multiplies every building.

- **Target (a), the central case:** the buildings add up to ONS's dwellings
  (E46Y, whole economy, end-2024, 2025 balance sheet) × England and Wales's
  share of UK dwellings (VOA's count over the UK's, with Scotland's council
  tax dwellings and Northern Ireland's LPS housing stock). Closed form.
- **Target (b), a check:** land ÷ value equals ONS's ratio of land under
  dwellings to land and dwellings at end-2021, solved exactly (the zero floor
  makes the condition piecewise linear in κ).

Each input resolves as the feed's fact, then a value stated on the command
line and recorded as a parameter, then absent; without either target the
build runs at κ = 1 and records that.

### 10. Land

Cell land per home = max(0, mean value − κ × mean unit cost), taken once per
cell over its pooled means. Land is never more than value, so λ never
exceeds 1 and the residual needs no cap. The variant "land at the rank"
reads the residual at each of the cell's ten value deciles instead.

### 11. Authorities, constituencies and regions

An authority's land and value are sums of its cells and its social cell. A
cell is split across constituencies by its homes' output areas (a cell with
no sold home by its MSOA's household shares), and an authority's social homes
by its households; the split is the same for every variant.

### 12. MHCLG's pattern, the comparison

Land per new home = MHCLG's £ per hectare ÷ its homes per hectare (the
floor-adjusted sheet, medium density, median new-build price, 1 October
2023). MHCLG's own caveat stands: the estimates "do not represent the market
value of land". As a pattern it spreads England's central land total across
authorities in proportion to land per new home × dwellings, capped at each
authority's value and rescaled. The Isles of Scilly take Cornwall's values,
flagged.

## Variants

| Recomputes | Variants |
|---|---|
| The structure block | κ = 1, 0.75 and 1.25; targets (a) and (b); regional factors flat, and their spread doubled; depreciation off, steep (front-loaded), and from VOA's build-period mix; floor area from EPC D3's authority means; land at the rank |
| The value block too (`--variants all`) | uniform tenure; post-stratified by build period (declared groups of VOA's periods and the EPC's bands); bands within MSOA without type |
| MHCLG's side | density ±25%; the gross and net density bounds; new-build price percentiles 25 and 75; the sheet before MHCLG's floor; Scilly without a value |

## The decision rule

MHCLG's pattern becomes the central case again if any of these fails. The
thresholds are proposals for Max and Vahid.

1. **Stability:** the authority land shares keep a Spearman correlation of at
   least 0.9 with the central case across the regional, depreciation and
   floor-area variants.
2. **Level:** the solved κ lies between 0.75 and 1.25.
3. **Costs:** in every English region, the cost per m² at the solved κ sits
   below the lower quartile of new-build sale prices per m².

## Measurements

- **The grain test:** for each pattern (cell land, an authority share × cell
  value, MHCLG's pattern, land at the rank), the share of council tax (VOA's
  private count × the band's ratio to Band D × the authority's Band D) that a
  land tax at the balancing rate would move between cells, in points,
  nationally and by seat. Authority shares are enough if they differ from
  cell land by under 3 points nationally and under 5 in 90% of seats.
- **κ diagnostics:** the share of cells with no land, by region (proposed:
  under 5% in every region), and seat ranks across the cost and depreciation
  variants (proposed: correlation 0.95).
- **Stock value:** the sales-based private-sector mean against the Market
  Value Survey's private-sector mean (owner occupiers and private renters),
  England within 5% and each region within 10%.

Also recorded: TPA's authority ranking against the central case (TPA shares
Price Paid, the HPI and the cost constants, so this is no independent
check), the authorities whose |log ratio| exceeds 0.4 against both MHCLG's
pattern and TPA, household land against E44N, and a like-for-like
comparison of the residual per new home with MHCLG's land per new home.

## Results

From the build of 12 November 2026 on Chronicle feed 80a3482 (facts sha256
`cb3342c3…`). ONS's four level inputs are stated parameters until the pinned
feed carries them: E46Y for 2024 (£2,525,540m) and 2021 (£2,097,345m), from
the 2025 balance sheet, and the net (£2,526bn) and gross (£4,632bn) capital
stock of dwellings for 2024. ONS's land under dwellings for 2021 (£5.9tn) comes
from the feed.

### The level

| | κ |
|---|---|
| Target (a), central: buildings = E46Y × 0.888 = £2,241.9bn | 0.795 |
| Target (b), check: land ÷ value = ONS's 2021 ratio, 0.738 | 0.882 |
| Depreciation off (target (a)) | 0.437 |
| Depreciation steep / from VOA's build periods | 0.797 / 0.802 |
| Regional factors flat / spread doubled | 0.801 / 0.789 |
| Floor area from EPC D3's authority means | 0.836 |
| Bands within MSOA without type (a value variant) | 0.764 |

England and Wales's share of UK dwellings is 0.888: VOA's 27,438,780 over
the sum of that, Scotland's 2,623,149 council tax dwellings and Northern
Ireland's 848,541 (LPS, 2026). The mean depreciation is 0.455 (one less 2,526 ÷ 4,632). Without
it, target (a) solves at 0.437, outside the band: E46Y is net of
depreciation and TPA's costs are not. The first build ran that way, because
the capital stocks were not yet an input.

### Totals, England and Wales, calendar-2024 prices

| | Land | Value | Land ÷ value |
|---|---|---|---|
| Central (κ = 0.795) | £7.24tn | £9.48tn | 0.764 |
| κ = 1 | £6.66tn | £9.48tn | 0.703 |
| Target (b) | £7.00tn | £9.48tn | 0.738 |

Under target (a), the buildings add up to the same total with depreciation on
or off, so national land is the same to within £0.1bn; only its spread
differs.

### The decision rule: all three pass

1. **Stability:** authority land shares correlate with the central case at
   0.968 (depreciation off), 0.983 (steep), 0.996 (EPC D3 floor areas), 0.992
   (regional spread doubled) and 0.995 (flat); the threshold is 0.9.
2. **Level:** κ = 0.795, inside 0.75 to 1.25.
3. **Costs:** in every English region the cost per m² at κ = 0.795 is below
   the lower quartile of new-build prices per m² (the North East: £1,504
   against £2,315).

So cell land stays the central case.

### Measurements

- **The grain test:** a land tax at the balancing rate would move 24.4 points
  of council tax between cells under cell land, 23.0 under an authority
  share × cell value, 24.4 under land at the rank and 26.5 under MHCLG's
  pattern. Cell land and authority shares differ by 1.44 points nationally,
  and by under 5 points in 99.8% of seats: **authority shares are enough**
  by the declared thresholds.
- **κ diagnostics: pass.** No region has more than 0.12% of its cells without
  land, and seat ranks correlate at 0.962 to 0.994 across the cost and
  depreciation variants.
- **Stock value: fails.** The sales-based private-sector mean is above the
  Market Value Survey's in England (£364,796 against £341,754, +6.7%; the
  threshold is 5%) and in every region: the North West +17.8%, the South East
  +13.7%, the East +11.3% (over the 10% threshold), the North East +9.4%, the
  South West +9.1%, the West Midlands +7.6%, Yorkshire and the Humber +6.9%,
  London +4.9% and the East Midlands +3.9%. Land, as the residual, carries the
  whole gap.

Also recorded: TPA's authority ranking correlates with the central case at
0.958; the residual per new home is a median 2.86 times MHCLG's land per new
home across 295 authorities, with a rank correlation of 0.927; 34
authorities differ from both MHCLG's pattern and TPA by a log ratio above
0.4; and the land under England and Wales's homes (£7.24tn) is above E44N,
UK households' land (£4.56tn), so that check fails. It is not like for like:
E44N is households' land only.

### Coverage

- 15,567,861 homes valued from 29,753,107 transfers; 10,835,909 of them
  linked to UCL's file. 4,733,362 homes take their cell's mean floor area.
- 55,617 cells. Means: 42,333 from the cell's own sales, 12,993 from the
  authority, 291 from the region. Knots: 34,342, 19,257 and 2,018.
- 5,672 MSOA × band counts printed as `[c]` read as 2.5.
- Tenure by MSOA is not in the feed yet, so social homes spread over each
  authority's MSOAs by their counts (PolicyEngine/chronicle#439 adds TS054 by
  MSOA).

## Not verified, and what is approximate

- ONS's depreciation profile for dwellings: the capital stocks workbooks
  state none.
- The stock value check above: why the sales-based means run above the
  survey is not yet explained.
- UCL's column names were checked against its readme by the `discover` stage;
  the provenance records the readme's sha256.
- The build-period groups join VOA's periods and the EPC's age bands where
  their boundaries nearly meet; some edges are approximate.
- VOA counts homes at 31 March 2026 and ONS's buildings are at end-2024:
  about a year of new homes apart. No adjustment is made.

## Disclosure and licences

The resources hold aggregates of public administrative data and published
tables only. Price Paid, the NSPL and UCL's linked file carry Royal Mail and
Ordnance Survey address data under their own terms; no address reaches any
resource. The attribution each publisher asks for is in the provenance's
`licences`, as its Chronicle manifest states it. The manifests of Price Paid,
the UK HPI and the NSPL state no licence; for those three the builder records
the licence and attribution the home-stock builder states (reviewed in
microcosm#1240), marked `license_source: declared`.

## Rebuilding

```bash
uv run --no-sync python tools/build_uk_stock_cells.py --stage discover --ucl-bundle-1 … --ucl-bundle-2 … --ctsop1-zip … --ctsop3-zip … --ctsop4-zip … --cache-dir …
```

```bash
uv run --no-sync python tools/build_uk_stock_cells.py --stage all --variants all --built-on YYYY-MM-DD --cache-dir … [the raw inputs and manifests] --ew-support … --ledger-facts … --ledger-manifest …
```

Until the pinned feed carries ONS's level inputs, add them as parameters:
`--dwellings-whole-economy-gbp 2525540000000
--dwellings-whole-economy-2021-gbp 2097345000000
--dwellings-net-capital-stock-gbp 2526000000000
--dwellings-gross-capital-stock-gbp 4632000000000`. Once the feed carries one,
drop its parameter: a parameter that differs from the feed is refused.

`--check` rebuilds and fails if a committed resource differs; the build date
alone is not a difference. A full build takes about 12 minutes and peaks near
12 GiB of memory.
