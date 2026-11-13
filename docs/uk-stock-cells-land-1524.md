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
cell's mean, pooled as in step 5. The provenance's
`floor_area_imputation` counts these homes by region, by sale year and by
the pool their mean comes from.

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
social rented share (TS054), as in the home-stock builder; TS054 prints
renting from the council and other social renting as two rows, and both
count. They spread over the authority's MSOAs by each MSOA's own census share
(TS054 by MSOA, PolicyEngine/chronicle#439); the variant "uniform tenure"
spreads them in proportion to each MSOA's count instead, and the provenance
records which source the build read.
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
2025 (Tables 2.1.1 and 1.1.1), which the pinned feed carries
(PolicyEngine/chronicle#439, workbook sha256 `71a995c3…0de7`). The central shape is
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

- **The grain test:** each pattern (cell land, an authority share × cell
  value, MHCLG's pattern, land at the rank) carries a land tax at the rate
  that balances council tax (VOA's private count × the band's ratio to Band
  D × the authority's Band D) over all homes. The test is how much of that
  tax falls on other cells under an authority share than under cell land:
  50 × Σ|cell land tax − authority-share land tax| ÷ council tax, in points,
  over cells nationally and over each seat's cells. Authority shares are
  enough if that is under 3 points nationally and under 5 in 90% of seats.
  The first build in review compared how much each pattern moves from
  council tax instead; two patterns can move the same amount to different
  cells, so the statistic changed (microcosm#1597 review). The thresholds
  are the ones fixed before the builder ran. The amount each pattern moves,
  and how far seat totals differ (100 × |a seat's difference| ÷ its council
  tax), are recorded beside it.
- **κ diagnostics:** the share of cells with no land, by region (proposed:
  under 5% in every region), and seat ranks across the cost and depreciation
  variants (proposed: correlation 0.95).
- **Stock value:** the sales-based private-sector mean against the Market
  Value Survey's private-sector mean (owner occupiers and private renters),
  England within 5% and each region within 10%.

Also recorded: TPA's authority ranking against the central case (TPA shares
Price Paid, the HPI and the cost constants, so this is no independent
check), the authorities whose |log ratio| exceeds 0.4 against both MHCLG's
pattern and TPA, a like-for-like comparison of the residual per new home
with MHCLG's land per new home, and the land against ONS's land. That last
one gives, under both κ targets, the land under owner-occupied, privately
rented and social homes, beside three ONS figures:

- E44N, the UK households sector's land of every kind;
- E44R, the UK's land of every kind and every sector;
- ONS's land underlying UK dwellings at end-2021, carried to the 2024 mean by
  the UK HPI for the United Kingdom.

Households hold the owner-occupied land and part of the privately rented
land, so their land under these homes lies between the owner-occupied land
and all private land. The owner-occupied land alone must be below E44N,
which also holds Scotland's and Northern Ireland's household land and
households' land under other buildings and farms. The land under every home
must be below E44R.

## Results

From the build of 13 November 2026 on the pinned Chronicle feed `4ce631f5`
(PolicyEngine/chronicle#439; facts sha256 `c294ba20…`, 2,160,233 rows). Every
level input comes from the feed: E46Y for 2024 (£2,525,540m) and 2021
(£2,097,345m) from the 2025 balance sheet, the net (£2,526bn, MJF8) and gross
(£4,632bn, MJF7) capital stock of dwellings for 2024, ONS's land under
dwellings for 2021 (£5.9tn), and the 2025 balance sheet's land at end-2024
for households (E44N, £4,559,831m) and the whole economy (E44R,
£7,117,771m). Social homes spread over MSOAs by TS054 by MSOA.

### The level

| | κ |
|---|---|
| Target (a), central: buildings = E46Y × 0.888 = £2,241.9bn | 0.807 |
| Target (b), check: land ÷ value = ONS's 2021 ratio, 0.738 | 0.880 |
| Depreciation off (target (a)) | 0.443 |
| Depreciation steep / from VOA's build periods | 0.807 / 0.812 |
| Regional factors flat / spread doubled | 0.813 / 0.800 |
| Floor area from EPC D3's authority means | 0.852 |
| Value variants: uniform tenure / build-period strata / bands without type | 0.813 / 0.795 / 0.778 |

England and Wales's share of UK dwellings is 0.888: VOA's 27,438,780 over
the sum of that, Scotland's 2,623,149 council tax dwellings and Northern
Ireland's 848,541 (LPS, 2026). The mean depreciation is 0.455 (one less
2,526 ÷ 4,632). Without it, target (a) solves at 0.443, outside the band:
E46Y is net of depreciation and TPA's costs are not. The first build ran
that way, before the capital stocks were an input.

### Totals, England and Wales, calendar-2024 prices

| | Land | Value | Land ÷ value |
|---|---|---|---|
| Central (κ = 0.807) | £7.08tn | £9.33tn | 0.760 |
| κ = 1 | £6.55tn | £9.33tn | 0.702 |
| Target (b) | £6.88tn | £9.33tn | 0.738 |
| Uniform tenure within authorities | £6.91tn | £9.15tn | 0.755 |

The private stock is 22,756,246 homes and the social stock 4,700,438. Under
target (a), the buildings add up to the same total with depreciation on or
off, so national land barely moves; only its spread differs. Placing social
homes by each MSOA's own census share leaves more of the private stock in
dearer MSOAs than spreading them by count does, hence the higher value.

### The decision rule: all three pass

1. **Stability:** authority land shares correlate with the central case at
   0.968 (depreciation off), 0.983 (steep), 0.996 (EPC D3 floor areas), 0.992
   (regional spread doubled) and 0.995 (flat); the threshold is 0.9.
2. **Level:** κ = 0.807, inside 0.75 to 1.25.
3. **Costs:** in every English region the cost per m² at κ = 0.807 is below
   the lower quartile of new-build prices per m² (the North East: £1,527
   against £2,315).

So cell land stays the central case.

### Measurements

- **The grain test: cells are needed.** An authority share × cell value
  places 3.26 points of the land tax on other cells than cell land does,
  against a threshold of 3. Within seats it is under 5 points in 92.3% of
  seats, which meets the 90% threshold, so the national test decides, and
  only just. Seat totals differ by under 5 points in 91.3% of seats.
  For context, a land tax at the balancing rate moves 24.0 points of council
  tax between cells under cell land, 22.6 under an authority share, 24.0
  under land at the rank and 26.3 under MHCLG's pattern. The first build in
  review compared those amounts instead: cell land and authority shares
  differed by 1.40 points, which read as "authority shares are enough", but
  that statistic could not see where the tax falls.

  What this means for the cells: an authority's land share spread by value
  is not a substitute for the cells' own land, so the cells are needed.
  They are already the central case, which the decision rule keeps. The
  gap is near the threshold, and the threshold is a proposal, so the reading
  is for Max and Vahid to confirm.
- **κ diagnostics: pass.** Almost no cell has zero land, and seat ranks keep
  their order across the cost and depreciation variants.
- **Stock value: passes.** The sales-based private-sector mean is £372,470 in
  England against the Market Value Survey's £367,739 (+1.3%; the threshold
  is 5%). By region: the North West +8.1%, London +4.7%, the West Midlands
  +1.8%, the South West +1.0%, the South East +0.9%, the East +0.8%,
  Yorkshire and the Humber 0.0%, the East Midlands −3.2% and the North East
  −4.0% (all within 10%). The survey's private-sector mean weights its
  owner-occupier and private-renter means by the private stock's
  owner-occupied share in the same cells (0.738 in England).

An earlier build failed this check (England +6.7%, the North West +17.8%).
It had two causes, both fixed: the check took the survey's two tenure means
half and half, and the census tenure shares kept only one of TS054's two
social-rent rows, which halved the social stock (2.41 million homes instead of
4.70 million) and lowered the owner share.

Also recorded: TPA's authority ranking correlates with the central case at
0.956; the residual per new home is a median 2.85 times MHCLG's land per new
home across 295 authorities, with a rank correlation of 0.927; and 34
authorities differ from both MHCLG's pattern and TPA by a log ratio above
0.4.

### The land against ONS's land

| England and Wales, £tn, calendar-2024 prices | Target (a), central | Target (b) |
|---|---|---|
| Under owner-occupied homes | 4.52 | 4.39 |
| Under privately rented homes | 1.86 | 1.81 |
| Under social homes | 0.71 | 0.68 |
| Under every home | 7.08 | 6.88 |

| ONS, the UK | £tn |
|---|---|
| E44N, the households sector's land of every kind, end-2024 | 4.56 |
| E44R, the land of every kind and every sector, end-2024 | 7.12 |
| The land underlying dwellings, end-2021 | 5.90 |
| The same, carried to the 2024 mean by the UK HPI (× 1.062) | 6.27 |

Both checks pass, but only just. Under target (a):

- **Owner-occupied land alone is 99.1% of E44N.** E44N also holds Scotland's
  and Northern Ireland's household land and households' land under other
  buildings and farms. Households also own part of the privately rented
  land. Once household landlords hold more than 2.2% of it, households'
  land under England and Wales's homes alone is above E44N. Owner-occupied
  and privately rented land together are 140% of it.
- **The land under every home is 99.5% of E44R**, the UK's land of every
  kind.
- **It is 13.0% above ONS's land under all of the UK's dwellings**, carried
  to 2024.

Under target (b), owner-occupied land is 96.2% of E44N, and the land under
every home is 9.8% above ONS's land under dwellings. Target (a) fixes the
buildings to E46Y, so wherever the sales-based value is above ONS's implied
dwelling value, the difference lands in land. Which route anchors the level
is open on PolicyEngine/uk-land-value-tax#20; these figures are for that
choice.

### Coverage

- 15,567,861 homes valued from 29,753,107 transfers; 10,835,909 of them
  linked to UCL's file.
- 4,733,362 of the 15,567,722 sold homes in cells (30.4%) take a pooled mean
  floor area: 97.9% of them from their own cell, 2.0% from their authority ×
  band and 0.04% from their region × band. By region the share runs from
  29.4% (London) to 32.7% (Wales). By the year of the home's latest sale it
  is 59.2% for 1995 to 2008, 7.5% for 2009 to 2021, 18.6% for 2022 and 2023,
  and 42.3% for 2024 to 2026. UCL's file ends in October 2023, and the
  earliest sales predate most certificates.
- 55,617 cells. Means: 42,333 from the cell's own sales, 12,993 from the
  authority, 291 from the region. Knots: 34,342, 19,257 and 2,018.
- 5,672 MSOA × band counts printed as `[c]` read as 2.5.

## Not verified, and what is approximate

- ONS's depreciation profile for dwellings: the capital stocks workbooks
  state none.
- The North West's sales-based mean stays 8.1% above the survey's, inside
  the threshold but the largest gap; why is not yet explained.
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

The pinned feed carries ONS's level inputs, so no `--dwellings-*` parameter
is needed. On a feed without them, each can be stated as a parameter, which
the provenance records as such; a parameter that differs from the feed is
refused, and so is a feed that carries an input in a shape the declaration
does not select.

`--check` rebuilds and fails if a committed resource differs; the build date
alone is not a difference. A full build takes about 12 minutes and peaks near
12 GiB of memory.
