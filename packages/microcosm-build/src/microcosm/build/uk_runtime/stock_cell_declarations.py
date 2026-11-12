"""The UK stock cell table's declarations: data, no logic (microcosm#1524).

Every choice the builder (``tools/build_uk_stock_cells.py``) makes before it
reads any result is written here, so a reviewer reads the method without
reading code and a change to it shows in one diff: the pooling and its
minimums, the knot grid, the age bands and depreciation profiles, the
columns of UCL's files, the level's inputs, the variants, the decision rule
and the measurements. Thresholds are proposals for Max and Vahid to set.

UCL's column names are unverified until the builder's ``discover`` stage has
printed the files' headers against them; ``UCL_COLUMNS_VERIFIED_AGAINST``
then records the sha256 of the readme they were checked against.

This module is loaded by file path from the builder tool, so it imports no
``uk_runtime`` sibling.
"""

from __future__ import annotations

from microcosm.build.cell_pooling import PoolingDeclaration

METHOD_VERSION = "uk-stock-cells-v1"
ISSUE = "PolicyEngine/microcosm#1524"

# ----------------------------------------------------------------------------
# Dates and the dwelling table

#: VOA counts the stock at this date.
STOCK_DATE = "2026-03-31"
#: Values are at calendar-2024 average prices, the frame's valuation basis.
BASIS_YEAR = 2024
#: The dwelling table and its cache are the home-stock builder's (#1240):
#: its method string and stock date name the shared cache.
DWELLING_CACHE_METHOD = "uk-high-value-home-stock-v1"
DWELLING_STOCK_DATE = "2025-12-31"
#: A building's age is counted at the date of ONS's buildings (end-2024).
BUILDING_AGE_AT_YEAR = 2025.0

# ----------------------------------------------------------------------------
# Cells

BANDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
#: Price Paid's types, in the home-stock library's ``ptype`` order.
PROPERTY_TYPES: tuple[str, ...] = ("D", "S", "T", "F")
#: VOA prints a count from 1 to 4 as [c]; it reads as their midpoint.
VOA_SUPPRESSED_COUNT = 2.5
#: A cell is MSOA × band. A thin cell reads its means and knots from its
#: authority × band, then region × band, then nation × band. The count is
#: always VOA's. Minimums are homes seen sold, proposed before any run.
POOLING = PoolingDeclaration(
    levels=("msoa_band", "local_authority_band", "region_band", "nation_band"),
    minimums={"means": 30, "knots": 100},
)
#: Thirteen knots; the exact weighted mean is stored beside them, so land
#: totals do not depend on the grid.
KNOT_GRID: tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1.0,
)
KNOT_CAP = 20_000_000.0
#: Value-rank bins of each cell, for unit cost along the value rank.
RANK_BINS = 10
#: A certificate's floor area outside these bounds is a data error.
FLOOR_AREA_BOUNDS_M2: tuple[float, float] = (15.0, 1500.0)

# ----------------------------------------------------------------------------
# UCL's linked file and certificates (PolicyEngine/chronicle#426)

#: The EPC's construction age bands for England and Wales: label, first and
#: last year. The open bands carry the years the age assumes.
EPC_AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("before 1900", 1850, 1899),
    ("1900-1929", 1900, 1929),
    ("1930-1949", 1930, 1949),
    ("1950-1966", 1950, 1966),
    ("1967-1975", 1967, 1975),
    ("1976-1982", 1976, 1982),
    ("1983-1990", 1983, 1990),
    ("1991-1995", 1991, 1995),
    ("1996-2002", 1996, 2002),
    ("2003-2006", 2003, 2006),
    ("2007-2011", 2007, 2011),
    ("2012 onwards", 2012, 2023),
)
#: Build-period groups for the variant that post-stratifies by build
#: period: each group holds VOA's CTSOP 4.1 periods and the EPC's age bands
#: whose years mostly fall in it. Their boundaries meet only in places, so
#: some edges are approximate (the EPC's 1950-1954 counts with 1955-1966,
#: and 1996-1999 with 2000 onwards). VOA's period of unknown age joins none.
BUILD_PERIOD_GROUPS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("before 1900", ("bp_pre_1900",), ("before 1900",)),
    ("1900-1929", ("bp_1900_1918", "bp_1919_1929"), ("1900-1929",)),
    ("1930-1954", ("bp_1930_1939", "bp_1945_1954"), ("1930-1949",)),
    ("1955-1966", ("bp_1955_1964",), ("1950-1966",)),
    ("1965-1982", ("bp_1965_1972", "bp_1973_1982"), ("1967-1975", "1976-1982")),
    ("1983-1999", ("bp_1983_1992", "bp_1993_1999"), ("1983-1990", "1991-1995")),
    (
        "2000 onwards",
        ("bp_2000_2008", *(f"bp_{year}" for year in range(2009, 2027))),
        ("1996-2002", "2003-2006", "2007-2011", "2012 onwards"),
    ),
)
#: The linked file's columns by role. UNVERIFIED: the names follow Price
#: Paid's and the EPC's own field names; the discover stage checks them.
UCL_LINKED_COLUMNS: dict[str, str | None] = {
    "transaction_id": "transactionid",
    "date": "dateoftransfer",
    "floor_area": "tfarea",
    "age_band": "CONSTRUCTION_AGE_BAND",
    "postcode": "postcode",
    "paon": "paon",
    "saon": "saon",
}
#: The certificate file's columns by role. UNVERIFIED, as above.
UCL_CERTIFICATE_COLUMNS: dict[str, object] = {
    "local_authority": "LOCAL_AUTHORITY",
    "tenure": "TENURE",
    "property_type": "PROPERTY_TYPE",
    "built_form": "BUILT_FORM",
    "floor_area": "TOTAL_FLOOR_AREA",
    "age_band": "CONSTRUCTION_AGE_BAND",
    "lodgement_date": "LODGEMENT_DATE",
    "building_id": "UPRN",
    "social_tenures": ("rental (social)", "Rented (social)"),
}
#: The sha256 of UCL's readme the column names were checked against; None
#: until the discover stage has run on the pinned bundle.
UCL_COLUMNS_VERIFIED_AGAINST: str | None = None

# ----------------------------------------------------------------------------
# Building cost

#: Tax Policy Associates' relative costs: cost per m² by type and regional
#: factors, read from the pinned config.py (``tpa-lvt-model-2026``). TPA has
#: no factor for Wales, which takes England's stock-weighted mean factor.
WALES_REGION_FACTOR_RULE = "england_stock_weighted_mean"
WALES_REGION_CODE = "W92000004"
#: Depreciation is ``min(cap, θ × shape(age))``, θ scaled so the
#: cost-weighted mean is one less ONS's net ÷ gross capital stock of
#: dwellings. Without that ratio the profiles cannot be scaled and the
#: build runs with depreciation off, recorded as such.
DEPRECIATION_PROFILES: dict[str, dict[str, object]] = {
    "central": {
        "shape": "straight_line",
        "cap": 0.8,
        "source": (
            "ONS's capital stocks profile for dwellings (straight line assumed): "
            "NOT VERIFIED against the capital stocks release"
        ),
    },
    "steep": {"shape": "front_loaded", "scale_years": 25.0, "cap": 0.8},
    "off": {"shape": "none"},
}

# ----------------------------------------------------------------------------
# The level (PolicyEngine/chronicle#438)

#: Each resolves as the feed's fact, then the command-line parameter, then
#: absent. Amounts are pounds, as the feed holds the balance sheet rows.
LEVEL_INPUTS: dict[str, dict[str, object]] = {
    "dwellings_whole_economy": {
        "concept": "ons.e46y",
        "spec_prefix": "ons.nbs2025.",
        "period": "2024",
        "parameter": "--dwellings-whole-economy-gbp",
        "request": "PolicyEngine/chronicle#438, item 1",
    },
    "dwellings_whole_economy_2021": {
        "concept": "ons.e46y",
        "spec_prefix": "ons.nbs2025.",
        "period": "2021",
        "parameter": "--dwellings-whole-economy-2021-gbp",
        "request": "PolicyEngine/chronicle#438, item 1",
    },
    "dwellings_net_capital_stock": {
        "concept": None,
        "parameter": "--dwellings-net-capital-stock-gbp",
        "request": "PolicyEngine/chronicle#438, item 2",
    },
    "dwellings_gross_capital_stock": {
        "concept": None,
        "parameter": "--dwellings-gross-capital-stock-gbp",
        "request": "PolicyEngine/chronicle#438, item 2",
    },
    "land_underlying_dwellings_2021": {
        "concept": "ons.land_article_2022.land_underlying_dwellings",
        "spec_prefix": "ons.land_article_2022.",
        "period": "2021",
        "parameter": "--land-underlying-dwellings-2021-gbp",
        "request": "in the pinned feed (ons-land-underlying-buildings-2022)",
    },
    "households_land": {
        "concept": "ons.e44n",
        "spec_prefix": "ons.nbs2025.",
        "period": "2024",
        "parameter": None,
        "request": "in the pinned feed (ons-national-balance-sheet-land-2025)",
    },
}
#: The central case solves κ on target (a): buildings add up to the UK's
#: dwellings (E46Y) × England and Wales's share of UK dwellings. Target (b)
#: solves land ÷ value to ONS's 2021 ratio and is reported beside it.
CENTRAL_KAPPA_TARGET = "a"
#: England and Wales's share of UK dwellings: its VOA count over the UK's,
#: Scotland from the vendored council tax dwellings and Northern Ireland
#: from LPS's housing stock in the feed.
UK_DWELLING_SPLIT: dict[str, str] = {
    "england_and_wales": "VOA CTSOP 1.1, ENGWAL all properties, 31 March 2026",
    "scotland": "uk/scotgov_chargeable_dwellings_by_band.json, all bands",
    "northern_ireland": "lps.housing_stock, Northern Ireland, 2026",
}

# ----------------------------------------------------------------------------
# Variants

#: Structure-block variants: each recomputes only the costs and the level.
STRUCTURE_VARIANTS: tuple[dict[str, object], ...] = (
    {"name": "central"},
    {"name": "kappa_1", "kappa": 1.0},
    {"name": "kappa_0_75", "kappa": 0.75},
    {"name": "kappa_1_25", "kappa": 1.25},
    {"name": "target_a", "kappa": "a"},
    {"name": "target_b", "kappa": "b"},
    {"name": "regional_flat", "regional": "flat"},
    {"name": "regional_doubled", "regional": "doubled"},
    {"name": "depreciation_off", "depreciation": "off"},
    {"name": "depreciation_steep", "depreciation": "steep"},
    {"name": "depreciation_voa_build_period", "depreciation": "voa_build_period"},
    {"name": "floor_area_epc_d3", "floor_area": "epc_d3"},
    {"name": "land_at_rank", "residual": "rank"},
)
#: Value-block variants (``--variants all``): each rebuilds the cells.
VALUE_VARIANTS: tuple[dict[str, object], ...] = (
    {"name": "tenure_uniform", "tenure": "uniform"},
    {"name": "post_stratify_build_period", "strata": "build_period"},
    {"name": "bands_without_type", "strata": "msoa"},
)
#: MHCLG's pattern across authorities: land per new home × dwellings, each
#: variant scaled to the central land total, capped at value and rescaled.
#: Each variant changes the central entry's settings it names. MHCLG
#: publishes no value for the Isles of Scilly, which takes Cornwall's in
#: the central case and is flagged (the issue's rule); the last variant
#: leaves it without a value instead.
MHCLG_VARIANTS: tuple[dict[str, object], ...] = (
    {
        "name": "central",
        "sheet": "floor_adjusted",
        "density": "medium",
        "percentile": 50,
        "density_factor": 1.0,
        "scilly": "cornwall",
    },
    {"name": "density_minus_25", "density_factor": 0.75},
    {"name": "density_plus_25", "density_factor": 1.25},
    {"name": "gross_net_upper", "density_factor": 0.8},
    {"name": "gross_net_lower", "density_factor": 4.0 / 3.0},
    {"name": "price_p25", "percentile": 25},
    {"name": "price_p75", "percentile": 75},
    {"name": "before_floor", "sheet": "before_floor"},
    {"name": "scilly_unimputed", "scilly": "none"},
)
MHCLG_SHEETS: dict[str, str] = {
    "floor_adjusted": "mhclg.residential_land_value_floor_adjusted",
    "before_floor": "mhclg.residential_land_value_before_floor",
}
ISLES_OF_SCILLY = "E06000053"
CORNWALL = "E06000052"
#: Councils merged since 2019, by their old code: an energy certificate or
#: an EPC D3 quarter recorded before the merger names the old council, and
#: counts towards its successor on Microcosm's April 2023 roster. These are
#: different councils, so their counts add up; a recoded council (Barnsley,
#: Sheffield) is an alias instead and counts once. The 2020 to 2023 codes
#: and names are as EPC D3 prints them.
AUTHORITY_PREDECESSORS: dict[str, str] = {
    # Bournemouth, Christchurch and Poole, and Dorset, April 2019.
    "E06000028": "E06000058",
    "E06000029": "E06000058",
    "E07000048": "E06000058",
    "E07000049": "E06000059",
    "E07000050": "E06000059",
    "E07000051": "E06000059",
    "E07000052": "E06000059",
    "E07000053": "E06000059",
    # East Suffolk and West Suffolk, April 2019.
    "E07000205": "E07000244",
    "E07000206": "E07000244",
    "E07000201": "E07000245",
    "E07000204": "E07000245",
    # Taunton Deane and West Somerset (Somerset West and Taunton, 2019), then
    # Somerset, April 2023.
    "E07000190": "E06000066",
    "E07000191": "E06000066",
    # Buckinghamshire, April 2020.
    "E07000004": "E06000060",
    "E07000005": "E06000060",
    "E07000006": "E06000060",
    "E07000007": "E06000060",
    # North and West Northamptonshire, April 2021.
    "E07000150": "E06000061",
    "E07000152": "E06000061",
    "E07000153": "E06000061",
    "E07000156": "E06000061",
    "E07000151": "E06000062",
    "E07000154": "E06000062",
    "E07000155": "E06000062",
    # Cumberland, and Westmorland and Furness, April 2023.
    "E07000026": "E06000063",
    "E07000028": "E06000063",
    "E07000029": "E06000063",
    "E07000027": "E06000064",
    "E07000030": "E06000064",
    "E07000031": "E06000064",
    # North Yorkshire, April 2023.
    "E07000163": "E06000065",
    "E07000164": "E06000065",
    "E07000165": "E06000065",
    "E07000166": "E06000065",
    "E07000167": "E06000065",
    "E07000168": "E06000065",
    "E07000169": "E06000065",
    # Somerset, April 2023.
    "E07000187": "E06000066",
    "E07000188": "E06000066",
    "E07000189": "E06000066",
    "E07000246": "E06000066",
}

# ----------------------------------------------------------------------------
# The decision rule and the measurements, fixed before the builder runs

#: MHCLG's pattern becomes the central case again if any of these fails.
DECISION_RULE: tuple[dict[str, object], ...] = (
    {
        "id": "rank_stability",
        "statistic": "Spearman correlation of authority land shares with the central case",
        "at_least": 0.9,
        "variants": (
            "regional_flat",
            "regional_doubled",
            "depreciation_off",
            "depreciation_steep",
            "floor_area_epc_d3",
        ),
    },
    {"id": "kappa_band", "statistic": "the solved cost level", "within": (0.75, 1.25)},
    {
        "id": "cost_below_new_build_lower_quartile",
        "statistic": (
            "cost per m² at the solved level against the lower quartile of "
            "new-build price per m², every English region"
        ),
    },
)
MEASUREMENTS: tuple[dict[str, object], ...] = (
    {
        "id": "grain_test",
        "statistic": (
            "council tax (band ratio × Band D) against rate × land, at the rate "
            "that balances them over all homes"
        ),
        "national_points": 3.0,
        "seat_points": 5.0,
        "seat_share": 0.9,
        "patterns": ("cell_land", "authority_share", "mhclg_pattern", "land_at_rank"),
    },
    {
        "id": "kappa_diagnostics",
        "zero_land_cell_share_by_region": 0.05,
        "seat_rank_correlation": 0.95,
    },
    {
        "id": "stock_value",
        "statistic": "private-sector mean value against the Market Value Survey",
        "england_relative_gap": 0.05,
        "region_relative_gap": 0.1,
    },
)
#: Authorities where the central case and two official series disagree.
LOG_RATIO_FLAG: dict[str, float] = {"threshold": 0.4, "minimum_references": 2}

# ----------------------------------------------------------------------------
# Feed selectors

TENURE_SOURCES = ("msoa", "uniform")
MVS_REGION_SPEC = "mhclg.mvs2024.at1_3.region.p2024."
MVS_ENGLAND_SPEC = "mhclg.mvs2024.at1_1a.mean.p2024."
MVS_PRIVATE_TENURES = ("owner_occupiers", "private_renters")
MVS_SOCIAL_TENURE = "all_social_renters"
MVS_MONTH = (2024, 7)
BAND_D_ENGLAND_SPEC = "uk.local_geography.council_tax.levels_england.fy2026."
BAND_D_WALES_SPEC = "uk.local_geography.council_tax.levels_wales.band_d."
TAX_BASE_ENGLAND_SPEC = "uk.local_geography.council_tax.levels_england.fy2026."
EPC_D3_SPEC = "mhclg.epc_d3_2026q3.local_authorities."
#: The ratio of Band D to each band's charge, set in law (England's and
#: Wales's nine bands; band I is Wales's alone).
BAND_D_RATIOS: dict[str, float] = {
    "A": 6 / 9,
    "B": 7 / 9,
    "C": 8 / 9,
    "D": 1.0,
    "E": 11 / 9,
    "F": 13 / 9,
    "G": 15 / 9,
    "H": 18 / 9,
    "I": 21 / 9,
}
