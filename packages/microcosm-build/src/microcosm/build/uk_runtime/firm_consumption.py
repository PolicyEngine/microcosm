"""UK firm consumption links: who buys each firm's output, from the ONS Supply and Use Tables.

The columns here let the firm table be matched later to household
consumption (LCFS, COICOP). Each firm's turnover is split into sales to
households, sales to other industries (intermediate consumption) and other
final demand (NPISH, central and local government, gross fixed capital
formation, valuables, changes in inventories and exports) by the shares of
its SIC 2007 division in the ONS *Supply and Use Tables, 1997 - 2024*
(consistent with Blue Book 2026), year 2024. The extracted CSVs sit in
``tests/fixtures/uk/firm_supply_use_2024/``
(``tools/extract_uk_firm_supply_use_tables.py`` re-creates them).

The columns are Microcosm-only (not policyengine-uk inputs) and their status
is ``"assumed"``: they are a split by published industry averages, not an
observation of any firm. The assumptions, stated once:

- **product = industry.** Each CPA product is made by its own SIC division,
  so a division's sales mix is the demand mix of its products. The tables
  are product-by-product (the 'Combined Use' matrix); secondary production
  is ignored.
- **imports used like domestic output.** The demand of a product (at
  purchasers' prices, domestic output and imports together) is split in the
  same proportions for both, so a division's shares are the product's
  demand shares.
- **aggregation by sums.** A division with several products (division 10
  has eight) takes the sum of each demand column over its products divided
  by the sum of their total demand, not an average of product shares. A
  product that spans several divisions (``CPA_B05 & B06``, ``CPA_B07 &
  B08``, ``CPA_C11.01-6 & C12``, ``CPA_F41, F42 & F43``, ``CPA_J59 & J60``,
  ``CPA_Q87 & Q88``) counts whole in each, since the tables give no split.
- **purchasers' prices.** Demand is valued at purchasers' prices, so trade
  margins and product taxes on a good count in its maker's household share,
  which leans high for goods sold mostly to households through retail
  (division 14, clothing, is about 0.84).
- **denominator = total demand** (intermediate consumption + total final
  demand = total supply at purchasers' prices). Household demand is Table
  2's national-concept households column. A negative demand component (a
  drawdown of inventories) is floored at zero in numerator and denominator
  alike, so shares stay in [0, 1]. Changes in inventories are negative on
  32 of the 103 products (6,275 million pounds in all); without the floor
  divisions 18 (printing) and 37 (sewerage) would get a small negative other
  final share (about -1.2 and -0.1 percent).
- **imputed rent is not a firm product.** ``CPA_L68A`` (imputed rents of
  owner-occupied dwellings: 284,071 million pounds of domestic output;
  288,906 of its 289,717 million pounds of demand from households, 811
  exports) is produced by owner-occupiers, not firms, so it is left out of
  division 68 and every aggregate here; division 68 is real estate on own
  account (``CPA_L68BXL683``) and on a fee basis (``CPA_L683``).

**Wholesale and retail (divisions 45, 46, 47).** Table 1's distributors'
trading margins column is negative on exactly the three trade products
(``CPA_G45`` -38,934, ``CPA_G46`` -202,054, ``CPA_G47`` -182,703 million
pounds) and positive on goods products (and the publishing products ``J58``
and ``J59 & J60``); it sums to zero. The trade output is therefore
redistributed onto the goods it is a margin on, and the trade products' own
rows hold almost no demand (``CPA_G47`` total demand 0, ``CPA_G46`` 276), so
their own-row shares are undefined. A trade division's shares are instead
margin-weighted over the products with a positive margin:
``s_trade = sum_p m_p s_p / sum_p m_p`` where ``s_p`` is product ``p``'s
demand share. Limitations, stated rather than fixed: the margins column is
one combined column (wholesale, retail and motor trade together), so 45, 46
and 47 get identical shares, although retail margins sit mostly on household
purchases and wholesale margins on intermediate ones; and ``CPA_G45``'s own
direct (non-margin) demand (21,469 of its 33,460 million pounds from
households) is not used.

**Fallback.** A division with no product (98, 99, or a code that is not a
SIC 2007 division) takes its section's aggregate (sums over the section's
products; the trade margin weighting for section G), and a section with no
product (U) or an unreadable code takes the economy-wide aggregate (sums
over every product but imputed rent). The fallbacks are counted.

COICOP shares of household sales (:func:`uk_firm_division_coicop_shares`)
use Table 3 (household final consumption, domestic concept, product by
COICOP class) aggregated to COICOP divisions 01-12 and to SIC divisions the
same way; for 45-47 each product's household purchases by COICOP division
are weighted by its margin per pound of total demand
(``sum_p m_p T3_pk / D_p``, normalised), so a trade division's household
sales follow the goods households buy, in proportion to the margin they
carry. Table 3 prints ``CPA_C27`` and ``CPA_C28`` as one row; it counts
whole in divisions 27 and 28.

Table 3's COICOP totals equal ONS Consumer Trends 2024 (the vendored 30
September 2026 release, the same Blue Book 2026 vintage) in every division:
:func:`check_uk_table3_consumer_trends` is a source-consistency check of the
extraction and the vintage, not a validation of the shares, which it never
reads. It will fail once Consumer Trends is re-vendored at a later vintage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.occupation_constants import (
    UK_SIC2007_SECTION_DIVISIONS,
    uk_sic2007_sections,
)

#: The extracted CSVs' file names in a supply and use source directory.
UK_FIRM_SUT_USES_FILE = "sut_product_uses.csv"
UK_FIRM_SUT_HHFCE_FILE = "sut_hhfce_product_coicop.csv"
UK_FIRM_SUT_DIVISION_FILE = "sut_product_sic_division.csv"
UK_FIRM_SUT_FILES = (
    UK_FIRM_SUT_USES_FILE,
    UK_FIRM_SUT_HHFCE_FILE,
    UK_FIRM_SUT_DIVISION_FILE,
)
UK_FIRM_SUT_USES_COLUMNS = (
    "product_code",
    "product_label",
    "intermediate_consumption",
    "households",
    "npish",
    "central_government",
    "local_government",
    "gfcf",
    "valuables",
    "changes_in_inventories",
    "exports",
    "total_demand",
    "domestic_output",
    "imports",
    "distributors_margins",
    "taxes_less_subsidies",
    "total_supply",
)
UK_FIRM_SUT_HHFCE_COLUMNS = ("product_code", "coicop", "value_gbp_million")
UK_FIRM_SUT_DIVISION_COLUMNS = (
    "product_code",
    "sic_division",
    "sic_section",
    "map_sic_codes",
    "map_a103_codes",
)
#: The supply and use year the shares describe.
UK_FIRM_SUT_YEAR = 2024
#: Final demand other than households.
UK_FIRM_OTHER_FINAL_DEMAND = (
    "npish",
    "central_government",
    "local_government",
    "gfcf",
    "valuables",
    "changes_in_inventories",
    "exports",
)
#: Products left out of every firm aggregate (owner-occupiers' imputed rent).
UK_FIRM_SUT_EXCLUDED_PRODUCTS = ("CPA_L68A",)
#: SIC divisions whose shares are margin-weighted over goods products.
UK_FIRM_TRADE_DIVISIONS = (45, 46, 47)
#: The firm columns, in order; they sum to turnover.
UK_FIRM_CONSUMPTION_COLUMNS = (
    "firm_household_sales",
    "firm_intermediate_sales",
    "firm_other_final_sales",
)
UK_FIRM_CONSUMPTION_STATUS = "assumed"
UK_FIRM_CONSUMPTION_METHOD = (
    "firm turnover split by its SIC 2007 division's shares of total demand "
    "(households; intermediate consumption; NPISH, government, capital "
    "formation and exports) in the ONS Supply and Use Tables 2024 (Blue Book "
    "2026), product = industry, imports used like domestic output, divisions "
    "45-47 margin-weighted over goods products"
)
#: COICOP divisions, as Consumer Trends and Table 3 code them.
UK_COICOP_DIVISIONS = tuple(f"{k:02d}" for k in range(1, 13))
#: Classes both Table 3 and Consumer Trends publish that household surveys
#: do not record: narcotics and imputed rentals. Table 3 has no 12.6.1
#: (FISIM) class, so that one is excluded on neither side.
UK_COICOP_GATE_EXCLUSIONS = ("02.3", "04.2")
#: The Table 3 vs Consumer Trends consistency check's tolerance per COICOP division, in
#: pounds million (both are published in whole pounds million).
UK_FIRM_TABLE3_GATE_TOLERANCE_GBP_MILLION = 0.5

#: The highest SIC 2007 division (99, extra-territorial organisations).
_LAST_SIC_DIVISION = max(last for _, last in UK_SIC2007_SECTION_DIVISIONS.values())
_SHARE_COLUMNS = ("household_share", "intermediate_share", "other_final_share")


@dataclass(frozen=True)
class UKFirmSupplyUseSource:
    """The extracted supply and use tables.

    ``uses`` is indexed by product code with the numeric columns of
    ``UK_FIRM_SUT_USES_COLUMNS``; ``hhfce`` is Table 3 in long form (Table 3
    row code, COICOP column, pounds million); ``divisions`` maps each product
    to its SIC divisions (int). ``sha256`` maps each file name to its hash.
    """

    uses: pd.DataFrame
    hhfce: pd.DataFrame
    divisions: pd.DataFrame
    sha256: Mapping[str, str]


def _read_csv(path: Path, columns: Sequence[str]) -> tuple[pd.DataFrame, str]:
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(
            f"{path.name} columns {list(frame.columns)} are not {list(columns)}."
        )
    return frame, hashlib.sha256(payload).hexdigest()


def read_uk_firm_supply_use_source(path: str | Path) -> UKFirmSupplyUseSource:
    """Read the three supply and use CSVs from their directory.

    Raises ``ValueError`` on wrong columns, a non-numeric value, a product
    whose demand components do not add up to its total demand, or a product
    without a SIC division.
    """

    directory = Path(path)
    uses, uses_hash = _read_csv(
        directory / UK_FIRM_SUT_USES_FILE, UK_FIRM_SUT_USES_COLUMNS
    )
    hhfce, hhfce_hash = _read_csv(
        directory / UK_FIRM_SUT_HHFCE_FILE, UK_FIRM_SUT_HHFCE_COLUMNS
    )
    divisions, divisions_hash = _read_csv(
        directory / UK_FIRM_SUT_DIVISION_FILE, UK_FIRM_SUT_DIVISION_COLUMNS
    )
    if uses["product_code"].duplicated().any():
        raise ValueError(f"{UK_FIRM_SUT_USES_FILE} repeats a product.")
    uses = uses.set_index("product_code")
    numeric = [c for c in UK_FIRM_SUT_USES_COLUMNS[2:]]
    try:
        uses[numeric] = uses[numeric].astype(np.float64)
        hhfce["value_gbp_million"] = hhfce["value_gbp_million"].astype(np.float64)
        divisions["sic_division"] = divisions["sic_division"].astype(np.int64)
    except ValueError:
        raise ValueError("UK supply and use values must be numbers.") from None
    demand = uses["intermediate_consumption"] + uses[
        ["households", *UK_FIRM_OTHER_FINAL_DEMAND]
    ].sum(axis=1)
    if not np.allclose(demand, uses["total_demand"], rtol=0, atol=0.5):
        raise ValueError("UK supply and use demand components != total demand.")
    if set(divisions["product_code"]) != set(uses.index):
        raise ValueError("Every UK supply and use product needs a SIC division.")
    unknown = set(hhfce["coicop"].str[:2]) - set(UK_COICOP_DIVISIONS)
    if unknown:
        raise ValueError(f"Table 3 COICOP columns {sorted(unknown)} are not 01-12.")
    for code in hhfce["product_code"].unique():
        _table3_products(code, uses.index)
    return UKFirmSupplyUseSource(
        uses=uses,
        hhfce=hhfce,
        divisions=divisions,
        sha256=MappingProxyType(
            {
                UK_FIRM_SUT_USES_FILE: uses_hash,
                UK_FIRM_SUT_HHFCE_FILE: hhfce_hash,
                UK_FIRM_SUT_DIVISION_FILE: divisions_hash,
            }
        ),
    )


def _table3_products(code: str, products: Sequence[str] | pd.Index) -> list[str]:
    """The Table 1 products a Table 3 row covers (``CPA_C27 & C28`` -> two)."""

    if code in products:
        return [code]
    parts = ["CPA_" + part.strip() for part in code[len("CPA_") :].split("&")]
    if len(parts) < 2 or any(part not in products for part in parts):
        raise ValueError(f"Table 3 row {code} is not a union of Table 1 products.")
    return parts


def _firm_products(source: UKFirmSupplyUseSource) -> pd.DataFrame:
    return source.uses.drop(
        index=[p for p in UK_FIRM_SUT_EXCLUDED_PRODUCTS if p in source.uses.index]
    )


def _components(uses: pd.DataFrame) -> np.ndarray:
    """Household, intermediate, other final and total demand per product.

    Negative components (changes in inventories, a drawdown of stocks, on
    some products) are floored at zero and the total is the sum of the
    floored components, so every share is in [0, 1].
    """

    other = uses[list(UK_FIRM_OTHER_FINAL_DEMAND)].clip(lower=0).sum(axis=1)
    parts = np.stack(
        [
            uses["households"].clip(lower=0).to_numpy(dtype=np.float64),
            uses["intermediate_consumption"].clip(lower=0).to_numpy(dtype=np.float64),
            other.to_numpy(dtype=np.float64),
        ]
    )
    return np.vstack([parts, parts.sum(axis=0)])


def _demand_sums(uses: pd.DataFrame) -> np.ndarray:
    """Household, intermediate, other final and total demand, summed."""

    return _components(uses).sum(axis=1)


def _trade_sums(uses: pd.DataFrame) -> np.ndarray:
    """Margin-weighted demand over products with a positive margin.

    Each product's demand is scaled by ``margin / total demand``, so the
    totals are ``sum_p m_p s_p`` and their sum ``sum_p m_p``.
    """

    goods = uses[uses["distributors_margins"] > 0]
    parts = _components(goods)
    scale = goods["distributors_margins"].to_numpy(dtype=np.float64) / parts[3]
    return (parts * scale).sum(axis=1)


def _row(sums: np.ndarray, basis: str) -> dict[str, Any]:
    household, intermediate, other, total = (float(v) for v in sums)
    return {
        "household_share": household / total,
        "intermediate_share": intermediate / total,
        "other_final_share": other / total,
        "household_gbp_million": household,
        "intermediate_gbp_million": intermediate,
        "other_final_gbp_million": other,
        "total_demand_gbp_million": total,
        "basis": basis,
    }


def _division_products(source: UKFirmSupplyUseSource) -> dict[int, list[str]]:
    kept = set(_firm_products(source).index)
    out: dict[int, list[str]] = {}
    for code, division in zip(
        source.divisions["product_code"],
        source.divisions["sic_division"],
        strict=True,
    ):
        if code in kept:
            out.setdefault(int(division), []).append(code)
    return out


def uk_firm_division_use_shares(source: UKFirmSupplyUseSource) -> pd.DataFrame:
    """Each SIC division's household, intermediate and other final shares.

    Indexed by division (int). The ``*_gbp_million`` columns are the summed
    numerators and denominator, the denominator being total demand with
    negative components floored at zero (for 45-47 the margin-weighted
    sums, whose total is the goods products' margins); ``basis`` is ``"products"`` or
    ``"trade_margin"``.
    """

    uses = _firm_products(source)
    trade = _trade_sums(uses)
    rows = {}
    for division, products in sorted(_division_products(source).items()):
        if division in UK_FIRM_TRADE_DIVISIONS:
            rows[division] = _row(trade, "trade_margin")
        else:
            rows[division] = _row(_demand_sums(uses.loc[products]), "products")
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "sic_division"
    return frame


def _section_use_shares(source: UKFirmSupplyUseSource) -> pd.DataFrame:
    """Each section's aggregate over its products (each product once)."""

    uses = _firm_products(source)
    by_division = _division_products(source)
    rows = {}
    for section, (first, last) in UK_SIC2007_SECTION_DIVISIONS.items():
        divisions = [d for d in by_division if first <= d <= last]
        if not divisions:
            continue
        if all(d in UK_FIRM_TRADE_DIVISIONS for d in divisions):
            rows[section] = _row(_trade_sums(uses), "section")
            continue
        products = sorted({p for d in divisions for p in by_division[d]})
        rows[section] = _row(_demand_sums(uses.loc[products]), "section")
    return pd.DataFrame.from_dict(rows, orient="index")


def _economy_use_shares(source: UKFirmSupplyUseSource) -> dict[str, Any]:
    return _row(_demand_sums(_firm_products(source)), "economy")


def _divisions(division: Any) -> np.ndarray:
    """SIC divisions as int, -1 where a value is missing or not 1-99."""

    raw = pd.to_numeric(pd.Series(np.asarray(division, dtype=object)), errors="coerce")
    values = raw.to_numpy(dtype=np.float64)
    ok = np.isfinite(values) & (values == np.round(values))
    ok &= (values >= 1) & (values <= _LAST_SIC_DIVISION)
    return np.where(ok, values, -1).astype(np.int64)


def _lookup(
    division: Any,
    by_division: pd.DataFrame,
    by_section: pd.DataFrame,
    economy: Mapping[str, Any],
    columns: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Rows of ``columns`` for each division, with section and economy fallback."""

    divisions = _divisions(division)
    out = np.empty((len(divisions), len(columns)), dtype=np.float64)
    counts = {"division": 0, "section": 0, "economy": 0}
    sections = uk_sic2007_sections(np.where(divisions < 0, 0, divisions))
    economy_row = np.array([economy[c] for c in columns], dtype=np.float64)
    division_table = by_division[list(columns)].to_numpy(dtype=np.float64)
    division_position = {int(d): i for i, d in enumerate(by_division.index)}
    section_table = by_section[list(columns)].to_numpy(dtype=np.float64)
    section_position = {str(s): i for i, s in enumerate(by_section.index)}
    for value in np.unique(divisions):
        mask = divisions == value
        if int(value) in division_position:
            out[mask] = division_table[division_position[int(value)]]
            counts["division"] += int(mask.sum())
            continue
        section = str(sections[mask][0]) if value > 0 else ""
        if section in section_position:
            out[mask] = section_table[section_position[section]]
            counts["section"] += int(mask.sum())
        else:
            out[mask] = economy_row
            counts["economy"] += int(mask.sum())
    return out, counts


@dataclass(frozen=True)
class UKFirmConsumptionSales:
    """The firm columns and how many rows each share level supplied."""

    columns: pd.DataFrame
    fallback_counts: Mapping[str, int]
    status: str = UK_FIRM_CONSUMPTION_STATUS
    method: str = UK_FIRM_CONSUMPTION_METHOD


def uk_firm_consumption_sales(
    division: Any, turnover: Any, source: UKFirmSupplyUseSource
) -> UKFirmConsumptionSales:
    """Split each firm's turnover into household, intermediate and other final sales.

    ``division`` is each firm's SIC 2007 division (the firm frame's
    ``sic_code``); ``turnover`` its turnover in any unit (the export's
    ``firm_turnover``, pounds), returned in the same unit. Household and
    intermediate sales are turnover times the division's shares; other final
    sales is the remainder, so the three columns sum to turnover up to one
    floating-point rounding (zero and negative turnover split the same way).
    ``fallback_counts`` counts rows by the level whose shares they took
    (``division``, ``section``, ``economy``).
    """

    values = np.asarray(turnover, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(np.asarray(division, dtype=object)):
        raise ValueError("UK firm division and turnover must be equal-length 1-d.")
    if not np.all(np.isfinite(values)):
        raise ValueError("UK firm turnover must be finite.")
    shares, counts = _lookup(
        division,
        uk_firm_division_use_shares(source),
        _section_use_shares(source),
        _economy_use_shares(source),
        _SHARE_COLUMNS,
    )
    household = values * shares[:, 0]
    intermediate = values * shares[:, 1]
    other = values - household - intermediate
    columns = pd.DataFrame(
        {
            "firm_household_sales": household,
            "firm_intermediate_sales": intermediate,
            "firm_other_final_sales": other,
        }
    )
    return UKFirmConsumptionSales(
        columns=columns, fallback_counts=MappingProxyType(counts)
    )


def _table3_by_row(source: UKFirmSupplyUseSource) -> pd.DataFrame:
    """Table 3 rows (row code) x COICOP division, pounds million."""

    frame = source.hhfce.assign(division=source.hhfce["coicop"].str[:2])
    wide = frame.pivot_table(
        index="product_code",
        columns="division",
        values="value_gbp_million",
        aggfunc="sum",
        fill_value=0.0,
    )
    return wide.reindex(columns=list(UK_COICOP_DIVISIONS), fill_value=0.0)


def _coicop_row(sums: np.ndarray, basis: str) -> dict[str, Any] | None:
    total = float(sums.sum())
    if total <= 0:
        return None
    return {
        **{k: float(v) / total for k, v in zip(UK_COICOP_DIVISIONS, sums, strict=True)},
        "household_gbp_million": total,
        "basis": basis,
    }


def _coicop_parts(
    source: UKFirmSupplyUseSource,
) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    """Table 3 rows of firm products, the products each covers, trade sums."""

    kept = set(_firm_products(source).index)
    table3 = _table3_by_row(source)
    covers = {code: _table3_products(code, source.uses.index) for code in table3.index}
    table3 = table3.loc[[c for c in table3.index if set(covers[c]) & kept]]
    uses = source.uses
    trade = np.zeros(len(UK_COICOP_DIVISIONS))
    for code, row in table3.iterrows():
        products = covers[code]
        margin = float(uses.loc[products, "distributors_margins"].sum())
        if margin > 0:
            demand = float(_components(uses.loc[products])[3].sum())
            trade += margin / demand * row.to_numpy(dtype=np.float64)
    return table3, covers, trade


def _coicop_tables(
    source: UKFirmSupplyUseSource,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    table3, covers, trade = _coicop_parts(source)
    by_division = _division_products(source)

    def rows_for(divisions: Sequence[int]) -> np.ndarray:
        wanted = {p for d in divisions for p in by_division[d]}
        codes = [c for c in table3.index if set(covers[c]) & wanted]
        return table3.loc[codes].to_numpy(dtype=np.float64).sum(axis=0)

    divisions = {}
    for division in sorted(by_division):
        if division in UK_FIRM_TRADE_DIVISIONS:
            row = _coicop_row(trade, "trade_margin")
        else:
            row = _coicop_row(rows_for([division]), "products")
        if row is not None:
            divisions[division] = row
    sections = {}
    for section, (first, last) in UK_SIC2007_SECTION_DIVISIONS.items():
        members = [d for d in by_division if first <= d <= last]
        if not members:
            continue
        if all(d in UK_FIRM_TRADE_DIVISIONS for d in members):
            row = _coicop_row(trade, "section")
        else:
            row = _coicop_row(rows_for(members), "section")
        if row is not None:
            sections[section] = row
    economy = _coicop_row(rows_for(sorted(by_division)), "economy")
    assert economy is not None
    division_frame = pd.DataFrame.from_dict(divisions, orient="index")
    division_frame.index.name = "sic_division"
    return division_frame, pd.DataFrame.from_dict(sections, orient="index"), economy


def uk_firm_division_coicop_shares(source: UKFirmSupplyUseSource) -> pd.DataFrame:
    """SIC division x COICOP division (``"01"``..``"12"``) shares of household sales.

    From Table 3 (domestic concept), summed over each division's products;
    divisions 45-47 margin-weighted over goods products. A division with
    products but no Table 3 household purchases takes its section's shares
    (``basis`` ``"section"``) or the economy-wide ones (``"economy"``);
    ``household_gbp_million`` is the summed Table 3 purchases behind the
    row's own shares (the fallback level's for a fallback row).
    """

    divisions, sections, economy = _coicop_tables(source)
    all_divisions = sorted(_division_products(source))
    columns = [*UK_COICOP_DIVISIONS, "household_gbp_million"]
    values, _ = _lookup(all_divisions, divisions, sections, economy, columns)
    frame = pd.DataFrame(values, index=pd.Index(all_divisions), columns=columns)
    frame.index.name = "sic_division"
    basis = []
    for division in all_divisions:
        if division in divisions.index:
            basis.append(divisions.loc[division, "basis"])
        else:
            section = str(uk_sic2007_sections([division])[0])
            basis.append("section" if section in sections.index else "economy")
    frame["basis"] = basis
    return frame


def _consumer_trends(year: int) -> dict[str, float]:
    """Consumer Trends domestic HFCE by COICOP division and gate class, £ million."""

    from microcosm.build.uk_runtime.energy_pricing import (
        UK_ONS_HFCE_ROW_DIMENSIONS,
        UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE,
    )
    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    rows = vendored_rows(
        UK_ONS_HOUSEHOLD_EXPENDITURE_RESOURCE,
        concept="ons.household_final_consumption_expenditure",
        period_type="calendar_year",
        period_value=year,
        dimensions={"consumption_concept": "domestic", **UK_ONS_HFCE_ROW_DIMENSIONS},
    )
    wanted = {*UK_COICOP_DIVISIONS, *UK_COICOP_GATE_EXCLUSIONS}
    out: dict[str, float] = {}
    for row in rows:
        code = str(row["dimensions"]["coicop"])
        if code in wanted:
            if code in out:
                raise ValueError(f"Consumer Trends {year} COICOP {code} appears twice.")
            out[code] = float(row["value"]) / 1e6
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"Consumer Trends {year} lacks COICOP {missing}.")
    return out


def uk_table3_consumer_trends_comparison(
    source: UKFirmSupplyUseSource, year: int = UK_FIRM_SUT_YEAR
) -> pd.DataFrame:
    """Table 3 COICOP-division totals vs ONS Consumer Trends (domestic concept).

    Indexed by COICOP division. ``table3`` and ``consumer_trends`` are pounds
    million less the classes in ``UK_COICOP_GATE_EXCLUSIONS`` (both sides
    publish them; 12.6.1 FISIM is in neither exclusion because Table 3 has no
    such class). ``difference`` is Table 3 minus Consumer Trends.
    """

    trends = _consumer_trends(year)
    hhfce = source.hhfce
    table3 = hhfce.groupby(hhfce["coicop"].str[:2])["value_gbp_million"].sum()
    rows = {}
    for division in UK_COICOP_DIVISIONS:
        t3 = float(table3.get(division, 0.0))
        ct = trends[division]
        excluded = [c for c in UK_COICOP_GATE_EXCLUSIONS if c[:2] == division]
        for code in excluded:
            t3 -= float(hhfce.loc[hhfce["coicop"] == code, "value_gbp_million"].sum())
            ct -= trends[code]
        rows[division] = {
            "table3": t3,
            "consumer_trends": ct,
            "difference": t3 - ct,
            "excluded": ";".join(excluded),
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "coicop_division"
    return frame


def check_uk_table3_consumer_trends(
    source: UKFirmSupplyUseSource,
    tolerance: float = UK_FIRM_TABLE3_GATE_TOLERANCE_GBP_MILLION,
) -> pd.DataFrame:
    """Source consistency: every COICOP division within ``tolerance`` pounds million.

    Returns the comparison; raises ``ValueError`` naming the divisions
    outside it.
    """

    frame = uk_table3_consumer_trends_comparison(source)
    outside = frame.index[frame["difference"].abs() > tolerance].tolist()
    if outside:
        raise ValueError(
            f"Table 3 differs from Consumer Trends by more than {tolerance} "
            f"pounds million in COICOP divisions {outside}."
        )
    return frame


def uk_firm_household_sales_by_coicop(
    division: Any,
    household_sales: Any,
    weight: Any,
    source: UKFirmSupplyUseSource,
    year: int = UK_FIRM_SUT_YEAR,
    unit: float = 1e6,
) -> pd.DataFrame:
    """Diagnostic only: weighted firm household sales by COICOP division vs Consumer Trends.

    ``household_sales`` (in pounds over ``unit``; default pounds -> pounds
    million) is spread over COICOP divisions by the firm's division shares,
    weighted and summed. ``consumer_trends_excluded`` is Consumer Trends
    less narcotics (02.3) and imputed rentals (04.2). No firm sells
    narcotics or owner-occupiers' imputed rent, but Table 3 puts 7,544
    million pounds of 04.2 on ``CPA_L68BXL683`` (real estate services), a
    firm product, so division 68's household sales carry it while this
    comparison leaves it out;
    ``ratio`` is against it. This cannot close to Consumer Trends and is not a
    gate. Directions of the separate biases, not a net sign:

    - gross turnover counts goods twice along the chain (the maker's
      household share and the retailer's whole turnover times the margin-
      weighted share): upward;
    - turnover is net of VAT and product taxes, Consumer Trends is at
      purchasers' prices: downward (largest for 02 alcohol and tobacco and
      07 fuel);
    - imported goods and services households buy are not UK firm output
      except through retail turnover: downward;
    - household shares are Table 2's national concept (residents' spending
      abroad in, non-residents' spending in the UK out), Consumer Trends is
      domestic: small, either way by product;
    - the business population's turnover need not equal output (registered
      firms only, government and non-market output largely absent): downward
      for 06, 10 and 12 especially.
    """

    sales = np.asarray(household_sales, dtype=np.float64) / unit
    weights = np.asarray(weight, dtype=np.float64)
    columns = list(UK_COICOP_DIVISIONS)
    divisions, sections, economy = _coicop_tables(source)
    shares, _ = _lookup(division, divisions, sections, economy, columns)
    firm = (shares * (sales * weights)[:, None]).sum(axis=0)
    trends = _consumer_trends(year)
    frame = pd.DataFrame(
        {
            "firm_household_sales": firm,
            "consumer_trends": [trends[c] for c in columns],
            "consumer_trends_excluded": [
                trends[c]
                - sum(trends[e] for e in UK_COICOP_GATE_EXCLUSIONS if e[:2] == c)
                for c in columns
            ],
        },
        index=pd.Index(columns, name="coicop_division"),
    )
    frame["ratio"] = frame["firm_household_sales"] / frame["consumer_trends_excluded"]
    return frame


__all__ = [
    "UK_COICOP_DIVISIONS",
    "UK_COICOP_GATE_EXCLUSIONS",
    "UK_FIRM_CONSUMPTION_COLUMNS",
    "UK_FIRM_CONSUMPTION_METHOD",
    "UK_FIRM_CONSUMPTION_STATUS",
    "UK_FIRM_OTHER_FINAL_DEMAND",
    "UK_FIRM_SUT_DIVISION_COLUMNS",
    "UK_FIRM_SUT_DIVISION_FILE",
    "UK_FIRM_SUT_EXCLUDED_PRODUCTS",
    "UK_FIRM_SUT_FILES",
    "UK_FIRM_SUT_HHFCE_COLUMNS",
    "UK_FIRM_SUT_HHFCE_FILE",
    "UK_FIRM_SUT_USES_COLUMNS",
    "UK_FIRM_SUT_USES_FILE",
    "UK_FIRM_SUT_YEAR",
    "UK_FIRM_TABLE3_GATE_TOLERANCE_GBP_MILLION",
    "UK_FIRM_TRADE_DIVISIONS",
    "UKFirmConsumptionSales",
    "UKFirmSupplyUseSource",
    "check_uk_table3_consumer_trends",
    "read_uk_firm_supply_use_source",
    "uk_firm_consumption_sales",
    "uk_firm_division_coicop_shares",
    "uk_firm_division_use_shares",
    "uk_firm_household_sales_by_coicop",
    "uk_table3_consumer_trends_comparison",
]
