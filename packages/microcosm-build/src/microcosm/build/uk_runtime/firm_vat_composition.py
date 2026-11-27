"""UK firm VAT composition: zero-rated, reduced-rated and exempt supplies by SIC section.

Opt-in, and **columns only**. The firm export scores VAT on wholly
standard-rated turnover with input VAT at 20% of input costs. This module
works out, for each SIC section, the split of sales into standard-,
reduced-, zero-rated and exempt supplies, the export share, the VAT rate on
purchases and the ratio of intermediate consumption to sales, from the ONS
Supply and Use Tables 2024 (Blue Book 2026) read through the VAT Act 1994
schedules. :func:`add_uk_firm_vat_composition_columns` attaches them to the
firm table as Microcosm-only columns (``firm_sut_*``, status ``assumed``):
no policyengine-uk input changes, and the build's weights are untouched.

**Not supported: the calibration seam.** :class:`UKFirmVATCompositionRules`
(VAT evaluated on the composition during calibration) and
:func:`rescale_uk_firm_input_costs` (input costs rescaled to the SUT ratio)
are kept for experiments and failed review (gate B2, 27 November 2026). On a
full 2024-25 build the VAT liability band accuracy fell to 0.998973 (seeds
2 and 3: 0.997182, 0.998228; bar 0.999), and the held-out error of VAT by
section doubled (sum of absolute section errors £168.0bn against £79.8bn;
section I at 3.6 times HMRC, G at 0.19 times). Applied to the reference
build's weights without a re-solve, the composition cuts VAT from £182.6bn
to £110.6bn: the frame carries about half of BPE's private turnover, so
realistic net rates cannot reach the HMRC liability totals and the solver
moves weight into high-net sections. (Those runs used the treatment table
before the ``exports`` rows and the ABS division weights for section G were
added.)

**Sources** (``tests/fixtures/uk/firm_vat_composition_2024/``, with
``firm_supply_use_2024/`` for the product demand table):

- ``vat_product_treatment.csv`` (hand-written): for each of the 103 SUT
  products, the split of its sales into standard, reduced, zero and exempt
  shares, by buyer where the law differs (``households``, ``other``
  domestic buyers, ``exports``; ``all`` covers every buyer not listed), with
  the VATA 1994 group it rests on. ``statute`` rows have one treatment by
  law; ``assumed`` rows carry a share set by judgement, never tuned to a
  receipt or relief figure. Two rows are computed: construction
  (``DWELLINGS``) and wholesale and retail (``GOODS``), below.
- ``sut_intermediate_consumption.csv``: the 'Combined Use' intermediate
  consumption matrix, product by industry (industries written under the
  product they make).
- ``sut_dwellings_gfcf.csv``: dwellings GFCF of the construction product.
- ``abs_section_turnover.csv``: ABS 2024 turnover by section and for
  divisions 45, 46 and 47.
- ``hmrc_vat_relief_costs.csv``: HMRC tax relief statistics (January 2026),
  for the checks only.

**Output shares.** A product's sales are its Table 2 demand at purchasers'
prices (so they include imports, distributors' margins and product taxes,
as in ``firm_consumption.py``): households (the national-concept column),
other domestic uses (intermediate consumption, NPISH, government, GFCF,
valuables, inventories floored at 0) and exports. Each takes its buyer's
treatment. Exports are zero-rated (VATA s.30(6); services supplied abroad
are outside the scope with the right to recover, the same effect here),
except for the consumer services with an ``exports`` row (I55, I56, P85,
Q86, R90-R93, S96): their SUT exports are mostly non-residents' spending in
the UK, which is taxed as a UK supply. Construction's zero share is
dwellings GFCF times ``UK_FIRM_NEW_DWELLING_FRACTION`` (0.6, assumed) over
its total demand (Sch 8 Group 5). Retail (G47) sales take the treatment mix
of household purchases of goods (products with a positive distributors'
margin; it includes vehicles and fuel, which households also buy through
G45); wholesale (G46) takes the mix of non-household domestic and export
demand for goods; motor trade (G45) is standard. A section's shares are the
domestic-output weighted mean of its products' shares, each product counted
once per section, except section G, whose three trade products are weighted
by ABS 2024 division turnover (gross sales; the SUT's trade output is the
margin only). Imputed rent (``CPA_L68A``) is excluded.

**Input VAT rate.** For industry ``j``, the intermediate-consumption
weighted mean of each product's VAT rate on business purchases
(``0.20 * standard + 0.05 * reduced`` of its ``other`` treatment). A
section's rate weights its industries by intermediate consumption. Capital
purchases are not included (stated, not patched). pe-uk applies the partial
exemption restriction itself (taxable / (taxable + exempt), rounded up).

**Input ratio.** The SUT ratio of intermediate consumption to output by
section. Section G is on a gross-sales base: the ratio is
``1 - (margin - IC_G) / gross_sales`` with gross sales = ABS 2024 section G
turnover (assumed to share the SUT concept; wholesale sales to retailers
count in both). Section G's input VAT rate prices the goods resold
(gross sales less margin) at the margin-weighted business purchase rate of
goods and its own intermediate consumption at its SUT rate. The generator
clamp [0.10, 0.95] applies, so section T (no intermediate consumption) shows
0.10.

**Assumptions stated once.** Shares are uniform across a section's firms (no
all-or-nothing draw; a wholly exempt bank gets the section mix). Public
administration (O84) is treated as exempt: its s.33 and s.41 refunds are not
modelled. The 2024-25 part year of VAT on private school fees (from 1
January 2025) is folded into P85's assumed standard share. Zero-rating of
prescription drugs at dispensing is not modelled.

**Checks, never targets** (:func:`uk_firm_vat_relief_checks`): the modelled
household base of eight zero- or reduced-rated reliefs against HMRC's 2024-25
cost over the rate replaced. All eight ratios are pinned in the tests,
including the poor ones (printed matter, domestic passenger transport) and
prescription drugs at 0 (not modelled).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_consumption import (
    UKFirmSupplyUseSource,
    read_uk_firm_supply_use_source,
)

UK_FIRM_VAT_TREATMENT_FILE = "vat_product_treatment.csv"
UK_FIRM_VAT_INT_CON_FILE = "sut_intermediate_consumption.csv"
UK_FIRM_VAT_DWELLINGS_FILE = "sut_dwellings_gfcf.csv"
UK_FIRM_VAT_ABS_FILE = "abs_section_turnover.csv"
UK_FIRM_VAT_RELIEF_FILE = "hmrc_vat_relief_costs.csv"
UK_FIRM_VAT_COMPOSITION_FILES = (
    UK_FIRM_VAT_TREATMENT_FILE,
    UK_FIRM_VAT_INT_CON_FILE,
    UK_FIRM_VAT_DWELLINGS_FILE,
    UK_FIRM_VAT_ABS_FILE,
    UK_FIRM_VAT_RELIEF_FILE,
)
UK_FIRM_VAT_COMPOSITION_STATUS = "assumed"
#: The supplies, in the order of the treatment table's share columns.
UK_FIRM_VAT_TREATMENTS = ("standard", "reduced", "zero", "exempt")
UK_FIRM_VAT_USERS = ("all", "households", "other", "exports")
UK_FIRM_VAT_SUPPLY_COLUMNS = {
    "standard": "firm_standard_rated_supplies",
    "reduced": "firm_reduced_rated_supplies",
    "zero": "firm_zero_rated_supplies",
    "exempt": "firm_exempt_supplies",
}
UK_FIRM_VAT_STANDARD_RATE = 0.20
UK_FIRM_VAT_REDUCED_RATE = 0.05
#: Share of construction's dwellings GFCF that is new build (Sch 8 Group 5), assumed.
UK_FIRM_NEW_DWELLING_FRACTION = 0.6
#: The clamp the generator puts on the input ratio
#: (``firm_generation.INPUT_RATIO_LOWER`` / ``INPUT_RATIO_UPPER``).
UK_FIRM_INPUT_RATIO_BOUNDS = (0.10, 0.95)
#: Microcosm-only columns the columns-only mode adds (not policyengine-uk inputs).
UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS = {
    "standard": "firm_sut_standard_rated_share",
    "reduced": "firm_sut_reduced_rated_share",
    "zero": "firm_sut_zero_rated_share",
    "exempt": "firm_sut_exempt_share",
    "export_share": "firm_sut_export_share",
    "input_vat_rate": "firm_sut_input_vat_rate",
    "input_ratio_target": "firm_sut_input_ratio",
}
_CONSTRUCTION = "CPA_F41, F42 & F43"
_TRADE = {"CPA_G45": "45", "CPA_G46": "46", "CPA_G47": "47"}
_EXCLUDED = ("CPA_L68A",)
_COMPUTED = ("DWELLINGS", "GOODS")
_ZERO = np.array([0.0, 0.0, 1.0, 0.0])
_OTHER_DOMESTIC = (
    "intermediate_consumption",
    "npish",
    "central_government",
    "local_government",
    "gfcf",
    "valuables",
    "changes_in_inventories",
)
_WHOLESALE_DOMESTIC = (
    "intermediate_consumption",
    "npish",
    "central_government",
    "local_government",
    "gfcf",
)
#: Each relief check: (HMRC code, rate the relief replaces, treatment, products;
#: ``None`` = food products). NS5 is domestic passenger transport: rail and
#: bus/coach only (water and air passenger transport are mostly international).
_RELIEF_CHECKS: Mapping[str, tuple[str, float, str, tuple[str, ...] | None]] = {
    "NS7 food (zero)": ("VAT - NS7", 0.20, "zero", None),
    "NS1 printed matter (zero)": ("VAT - NS1", 0.20, "zero", ("CPA_J58",)),
    "NS2 children's clothing (zero)": (
        "VAT - NS2",
        0.20,
        "zero",
        ("CPA_C14", "CPA_C15"),
    ),
    "NS3 new dwellings (zero)": ("VAT - NS3", 0.20, "zero", ()),
    "NS4 domestic fuel and power (reduced)": (
        "VAT - NS4",
        0.15,
        "reduced",
        ("CPA_D351", "CPA_D352_3"),
    ),
    "NS5 domestic passenger transport (zero)": (
        "VAT - NS5",
        0.20,
        "zero",
        ("CPA_H491_2", "CPA_H493_5"),
    ),
    "NS6 prescription drugs (zero)": ("VAT - NS6", 0.20, "zero", ("CPA_C21",)),
    "NS16 water and sewerage (zero)": (
        "VAT - NS16",
        0.20,
        "zero",
        ("CPA_E36", "CPA_E37"),
    ),
}


@dataclass(frozen=True)
class UKFirmVATCompositionSource:
    """The VAT composition inputs: treatment table, IC matrix, dwellings, ABS, reliefs, SUT."""

    treatment: pd.DataFrame
    intermediate: pd.DataFrame
    dwellings_gfcf_gbp_million: float
    abs_turnover: pd.DataFrame
    relief_costs: pd.DataFrame
    supply_use: UKFirmSupplyUseSource
    sha256: Mapping[str, str]


def _validate_treatment(treatment: pd.DataFrame, products: set[str]) -> None:
    if not set(treatment["user"]) <= set(UK_FIRM_VAT_USERS):
        raise ValueError(f"VAT treatment user must be one of {UK_FIRM_VAT_USERS}.")
    if treatment.duplicated(["product_code", "user"]).any():
        raise ValueError("VAT treatment repeats a (product, user) row.")
    if not set(treatment["status"]) <= {"statute", "assumed"}:
        raise ValueError("VAT treatment status must be statute or assumed.")
    if set(treatment["product_code"]) != products:
        raise ValueError("Every SUT product needs a VAT treatment, and only those.")
    for product, rows in treatment.groupby("product_code"):
        users = set(rows["user"])
        if "all" not in users and not {"households", "other"} <= users:
            raise ValueError(
                f"{product} needs an all row, or households and other rows."
            )
    computed = treatment["standard"].isin(_COMPUTED)
    allowed = {_CONSTRUCTION, "CPA_G46", "CPA_G47"}
    if set(treatment.loc[computed, "product_code"]) - allowed:
        raise ValueError("Only construction and G46/G47 have computed treatments.")
    numeric = treatment.loc[~computed, list(UK_FIRM_VAT_TREATMENTS)].astype(np.float64)
    if ((numeric < 0) | (numeric > 1)).any().any():
        raise ValueError("VAT treatment shares must lie in [0, 1].")
    if not np.allclose(numeric.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("VAT treatment shares must sum to 1 on every row.")


def read_uk_firm_vat_composition_source(
    path: str | Path, supply_use_path: str | Path
) -> UKFirmVATCompositionSource:
    """Read the fixture directory and the supply and use directory it builds on.

    Raises ``ValueError`` when a file's sha256 differs from ``provenance.json``,
    a treatment row is malformed (unknown user or status, repeated, shares
    outside [0, 1] or not summing to 1), a product lacks a treatment, the
    matrix names an unknown product, or the ABS table lacks section G or
    divisions 45-47.
    """

    directory = Path(path)
    provenance = json.loads((directory / "provenance.json").read_text())
    hashes: dict[str, str] = {}
    for name in UK_FIRM_VAT_COMPOSITION_FILES:
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if provenance["files"][name]["sha256"] != digest:
            raise ValueError(f"{name} sha256 differs from provenance.json.")
        hashes[name] = digest
    supply_use = read_uk_firm_supply_use_source(supply_use_path)
    products = set(supply_use.uses.index)
    treatment = pd.read_csv(directory / UK_FIRM_VAT_TREATMENT_FILE, dtype=str)
    _validate_treatment(treatment, products)
    intermediate = pd.read_csv(directory / UK_FIRM_VAT_INT_CON_FILE)
    named = set(intermediate["product_code"]) | set(
        intermediate["industry_product_code"]
    )
    if not named <= products:
        raise ValueError("The intermediate consumption matrix names unknown products.")
    dwellings = pd.read_csv(directory / UK_FIRM_VAT_DWELLINGS_FILE)
    abs_turnover = pd.read_csv(
        directory / UK_FIRM_VAT_ABS_FILE, dtype={"sic_section": str}
    )
    if not {"G", *_TRADE.values()} <= set(abs_turnover["sic_section"]):
        raise ValueError("The ABS table needs section G and divisions 45, 46 and 47.")
    relief_costs = pd.read_csv(directory / UK_FIRM_VAT_RELIEF_FILE)
    return UKFirmVATCompositionSource(
        treatment=treatment,
        intermediate=intermediate,
        dwellings_gfcf_gbp_million=float(
            dwellings["dwellings_gfcf_gbp_million"].iloc[0]
        ),
        abs_turnover=abs_turnover,
        relief_costs=relief_costs,
        supply_use=supply_use,
        sha256=MappingProxyType(hashes | dict(supply_use.sha256)),
    )


_LOOKUP = {
    "households": ("households", "all"),
    "other": ("other", "all"),
    "exports": ("exports",),
    "all": ("all", "other"),
}


def _treatment(
    source: UKFirmVATCompositionSource, product: str, user: str
) -> pd.Series | None:
    rows = source.treatment[source.treatment["product_code"] == product]
    for name in _LOOKUP[user]:
        row = rows[rows["user"] == name]
        if not row.empty:
            return row.iloc[0]
    if user == "exports":
        return None
    raise ValueError(f"No VAT treatment for {product}.")


def _goods(uses: pd.DataFrame) -> list[str]:
    return [p for p in uses.index if uses.loc[p, "distributors_margins"] > 0]


def _vector(
    source: UKFirmVATCompositionSource, product: str, user: str, *, nested: bool = False
) -> np.ndarray:
    """Standard, reduced, zero, exempt shares of ``product``'s sales to ``user``."""

    row = _treatment(source, product, user)
    if row is None:  # exports with no exports row: zero-rated (s.30(6))
        return _ZERO.copy()
    uses = source.supply_use.uses
    marker = row["standard"]
    if marker == "DWELLINGS":
        zero = (
            source.dwellings_gfcf_gbp_million
            * UK_FIRM_NEW_DWELLING_FRACTION
            / uses.loc[product, "total_demand"]
        )
        return np.array([1.0 - zero, 0.0, zero, 0.0])
    if marker == "GOODS":
        if nested:
            raise ValueError(f"{product}: a trade product cannot resell trade.")
        goods = _goods(uses)
        if product == "CPA_G47":
            weight = uses.loc[goods, "households"].clip(lower=0)
            mix = sum(
                weight[g] * _vector(source, g, "households", nested=True) for g in goods
            )
            return mix / weight.sum()
        total = np.zeros(4)
        mass = 0.0
        for g in goods:
            domestic = float(uses.loc[g, list(_WHOLESALE_DOMESTIC)].clip(lower=0).sum())
            exported = max(float(uses.loc[g, "exports"]), 0.0)
            total += domestic * _vector(source, g, "other", nested=True)
            total += exported * _vector(source, g, "exports", nested=True)
            mass += domestic + exported
        return total / mass
    return row[list(UK_FIRM_VAT_TREATMENTS)].astype(np.float64).to_numpy()


def _purchase_rate(source: UKFirmVATCompositionSource, product: str) -> float:
    """VAT rate a business pays when it buys ``product`` (its ``other`` treatment)."""

    if product in _EXCLUDED:
        return 0.0
    v = _vector(source, product, "other")
    return UK_FIRM_VAT_STANDARD_RATE * v[0] + UK_FIRM_VAT_REDUCED_RATE * v[1]


def uk_firm_vat_product_shares(source: UKFirmVATCompositionSource) -> pd.DataFrame:
    """Per product: output shares, export share, domestic output, IC and input VAT rate."""

    uses = source.supply_use.uses
    goods = _goods(uses)
    rows = {}
    for p in uses.index:
        if p in _EXCLUDED:
            continue
        if p in ("CPA_G46", "CPA_G47"):
            shares = _vector(source, p, "all")
            if p == "CPA_G46":
                ex = sum(max(float(uses.loc[g, "exports"]), 0.0) for g in goods)
                dom = sum(
                    float(uses.loc[g, list(_WHOLESALE_DOMESTIC)].clip(lower=0).sum())
                    for g in goods
                )
                export_share = ex / (ex + dom)
            else:
                export_share = 0.0
        else:
            hh = max(float(uses.loc[p, "households"]), 0.0)
            dom = float(uses.loc[p, list(_OTHER_DOMESTIC)].clip(lower=0).sum())
            ex = max(float(uses.loc[p, "exports"]), 0.0)
            total = hh + dom + ex
            shares = (
                hh * _vector(source, p, "households")
                + dom * _vector(source, p, "other")
                + ex * _vector(source, p, "exports")
            ) / total
            export_share = ex / total
        rows[p] = dict(zip(UK_FIRM_VAT_TREATMENTS, shares, strict=True)) | {
            "export_share": export_share,
            "domestic_output": float(uses.loc[p, "domestic_output"]),
        }
    frame = pd.DataFrame(rows).T
    rate = pd.Series({p: _purchase_rate(source, p) for p in uses.index})
    ic = source.intermediate
    ic = ic.assign(vat=ic["value_gbp_million"] * ic["product_code"].map(rate))
    by_industry = ic.groupby("industry_product_code")[
        ["vat", "value_gbp_million"]
    ].sum()
    frame["intermediate_consumption"] = (
        by_industry["value_gbp_million"].reindex(frame.index).fillna(0.0)
    )
    frame["input_vat_rate"] = (
        by_industry["vat"] / by_industry["value_gbp_million"]
    ).reindex(frame.index)
    return frame.astype(np.float64)


def uk_firm_vat_section_shares(source: UKFirmVATCompositionSource) -> pd.DataFrame:
    """By SIC section: output shares, export share, input VAT rate and input ratio."""

    products = uk_firm_vat_product_shares(source)
    abs_turnover = source.abs_turnover.set_index("sic_section")["turnover_gbp_million"]
    sections = source.supply_use.divisions[
        ["product_code", "sic_section"]
    ].drop_duplicates()
    sections = sections[sections["product_code"].isin(products.index)]
    keys = (*UK_FIRM_VAT_TREATMENTS, "export_share")
    rows = {}
    for section, group in sections.groupby("sic_section"):
        p = products.loc[group["product_code"]]
        if section == "G":
            # Gross-sales weights: ABS 2024 turnover of divisions 45, 46, 47.
            w = pd.Series({c: float(abs_turnover[d]) for c, d in _TRADE.items()})[
                p.index
            ]
        else:
            w = p["domestic_output"]
        ic = p["intermediate_consumption"]
        row: dict[str, Any] = {k: float(np.average(p[k], weights=w)) for k in keys}
        row["input_vat_rate"] = (
            float(np.average(p["input_vat_rate"].fillna(0.0), weights=ic))
            if ic.sum() > 0
            else 0.0
        )
        row["input_ratio_target"] = float(ic.sum() / p["domestic_output"].sum())
        row["output_gbp_million"] = float(p["domestic_output"].sum())
        rows[section] = row
    frame = pd.DataFrame(rows).T
    uses = source.supply_use.uses
    trade = products.loc[[c for c in _TRADE if c in products.index]]
    margin = float(trade["domestic_output"].sum())
    ic_g = float(trade["intermediate_consumption"].sum())
    gross = float(abs_turnover["G"])
    goods = _goods(uses)
    m = uses.loc[goods, "distributors_margins"]
    goods_rate = float(sum(m[g] * _purchase_rate(source, g) for g in goods) / m.sum())
    goods_resold = gross - margin
    frame.loc["G", "input_vat_rate"] = (
        goods_resold * goods_rate + ic_g * frame.loc["G", "input_vat_rate"]
    ) / (goods_resold + ic_g)
    frame.loc["G", "input_ratio_target"] = 1.0 - (margin - ic_g) / gross
    frame.loc["G", "output_gbp_million"] = gross
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    frame["input_ratio_target"] = frame["input_ratio_target"].clip(lo, hi)
    shares = list(UK_FIRM_VAT_TREATMENTS)
    frame[shares] = frame[shares].div(frame[shares].sum(axis=1), axis=0)
    frame.index.name = "sic_section"
    return frame


def add_uk_firm_vat_composition_columns(
    firms: pd.DataFrame, shares: pd.DataFrame
) -> pd.DataFrame:
    """``firms`` with the section composition as Microcosm-only share columns.

    The supported (columns-only) use: every policyengine-uk input, including
    the supply columns and ``firm_input_vat``, keeps the value the build
    solved its weights for. Refuses a table that already has a share column
    and a section with no shares.
    """

    clash = sorted(
        set(UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS.values()) & set(firms.columns)
    )
    if clash:
        raise ValueError(f"firm table already has {clash}.")
    sections = firms["firm_sic_section"].astype(str)
    missing = sorted(set(sections) - set(shares.index))
    if missing:
        raise ValueError(f"No VAT composition shares for sections {missing}.")
    rows = shares.reindex(sections.to_numpy())
    out = firms.copy()
    for key, column in UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS.items():
        out[column] = rows[key].to_numpy(np.float64)
    return out


def compose_uk_firm_vat_supplies(
    firms: pd.DataFrame, shares: pd.DataFrame, in_scope: np.ndarray | None = None
) -> pd.DataFrame:
    """Experimental (failed B2): the supply columns and ``firm_input_vat`` from shares.

    Rows in VAT scope (all, when ``in_scope`` is None) take their section's
    shares of ``firm_turnover``; other rows stay wholly exempt, as the default
    export has them. ``firm_input_vat`` is the section rate times
    ``firm_input_costs``. Raises ``ValueError`` for a section with no shares.
    """

    sections = firms["firm_sic_section"].astype(str)
    missing = sorted(set(sections) - set(shares.index))
    if missing:
        raise ValueError(f"No VAT composition shares for sections {missing}.")
    rows = shares.reindex(sections.to_numpy())
    turnover = firms["firm_turnover"].to_numpy(np.float64)
    scope = (
        np.ones(len(firms), bool) if in_scope is None else np.asarray(in_scope, bool)
    )
    out = firms.copy()
    for key, column in UK_FIRM_VAT_SUPPLY_COLUMNS.items():
        part = turnover * rows[key].to_numpy(np.float64)
        out[column] = np.where(scope, part, turnover if key == "exempt" else 0.0)
    out["firm_input_vat"] = rows["input_vat_rate"].to_numpy(np.float64) * out[
        "firm_input_costs"
    ].to_numpy(np.float64)
    return out


def uk_firm_vat_voluntary_registration(
    firms: pd.DataFrame, vat_registered: np.ndarray, threshold_gbp: float
) -> np.ndarray:
    """Experimental (failed B2): registered rows at or below the threshold on taxable turnover.

    pe-uk registers a firm on taxable (standard + reduced + zero) turnover, so
    a registered row the composition takes to the threshold or below keeps
    its registration through ``firm_vat_voluntarily_registered``.
    """

    taxable = sum(
        firms[UK_FIRM_VAT_SUPPLY_COLUMNS[k]].to_numpy(np.float64)
        for k in ("standard", "reduced", "zero")
    )
    return np.asarray(vat_registered, bool) & (taxable <= threshold_gbp)


def uk_firm_input_ratio_factors(
    sections: np.ndarray,
    ratio: np.ndarray,
    turnover: np.ndarray,
    targets: pd.Series,
    *,
    iterations: int = 200,
) -> dict[str, dict[str, float]]:
    """Experimental (failed B2): ``k_s`` per section so the clipped mean ratio hits the target.

    Deterministic bisection on ``log k`` in [1e-3, 1e3]; the mean is
    turnover-weighted over ``clip(ratio * k_s, 0.10, 0.95)``. A section with
    no positive turnover, or no target, gets ``k = 1`` and ``rescaled = 0``.
    """

    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    sections = np.asarray(sections).astype(str)
    out: dict[str, dict[str, float]] = {}
    for section in sorted(set(sections)):
        mask = sections == section
        r, w = np.asarray(ratio)[mask], np.asarray(turnover, np.float64)[mask]

        def mean(k: float, r: np.ndarray = r, w: np.ndarray = w) -> float:
            return float(np.average(np.clip(r * k, lo, hi), weights=w))

        if w.sum() <= 0 or section not in targets.index:
            out[section] = {
                "k": 1.0,
                "target": float("nan"),
                "achieved": float("nan"),
                "before": float("nan"),
                "rescaled": 0.0,
            }
            continue
        target = float(targets[section])
        a, b = 1e-3, 1e3
        for _ in range(iterations):
            mid = float(np.sqrt(a * b))
            if mean(mid) < target:
                a = mid
            else:
                b = mid
        k = float(np.sqrt(a * b))
        out[section] = {
            "k": k,
            "target": target,
            "achieved": mean(k),
            "before": mean(1.0),
            "rescaled": 1.0,
        }
    return out


def rescale_uk_firm_input_costs(
    turnover: np.ndarray,
    input_costs: np.ndarray,
    sections: np.ndarray,
    targets: pd.Series,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Experimental (failed B2): input costs rescaled by section to the target ratio.

    Rows with positive turnover in a section that has a target become
    ``turnover * clip(input / turnover * k_s, 0.10, 0.95)``; every other row
    keeps its input costs unchanged.
    """

    turnover = np.asarray(turnover, np.float64)
    input_costs = np.asarray(input_costs, np.float64)
    sections = np.asarray(sections).astype(str)
    positive = turnover > 0
    ratio = np.divide(
        input_costs, turnover, out=np.zeros_like(turnover), where=positive
    )
    factors = uk_firm_input_ratio_factors(
        sections[positive], ratio[positive], turnover[positive], targets
    )
    k = np.array([factors.get(s, {"k": 1.0})["k"] for s in sections])
    moved = positive & np.array(
        [factors.get(s, {"rescaled": 0.0})["rescaled"] > 0 for s in sections],
        dtype=bool,
    )
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    return np.where(moved, turnover * np.clip(ratio * k, lo, hi), input_costs), factors


#: Candidate columns the composition rules hand to policyengine-uk.
UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS = (
    "firm_id",
    "firm_turnover",
    "firm_input_costs",
    "firm_employees",
    "firm_sic_section",
    *UK_FIRM_VAT_SUPPLY_COLUMNS.values(),
    "firm_input_vat",
)


@dataclass(frozen=True)
class UKFirmVATCompositionRules:
    """Experimental (failed B2): ``UKFirmRules`` that evaluate VAT on the section composition.

    ``PolicyEngineUKFirmRules`` hands only its fixed input columns to the
    simulation, which have no zero- or reduced-rated supplies, so recovery
    would be standard / (standard + exempt). This class composes the
    candidate table and calls the simulation factory itself with
    ``UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS``. Every candidate row is
    composed: the rules return the liability if registered, before VAT scope
    and registration are assigned. ``simulation_factory`` defaults to
    ``policyengine_uk_firm_simulation``.
    """

    shares: pd.DataFrame
    simulation_factory: Callable[[pd.DataFrame, int], object] | None = None
    name: str = "policyengine_uk_vat_composition"

    def evaluate(self, firms: pd.DataFrame, *, period: int):
        from microcosm.build.uk_runtime.firm_policyengine_uk_rules import (
            policyengine_uk_firm_simulation,
            policyengine_uk_version,
        )
        from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs

        period = int(period)
        table = compose_uk_firm_vat_supplies(firms, self.shares)
        table = table.loc[:, list(UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS)].reset_index(
            drop=True
        )
        factory = self.simulation_factory or policyengine_uk_firm_simulation
        simulation = factory(table, period)
        liability = np.asarray(
            simulation.calculate("firm_vat_liability_if_registered", period),
            dtype=np.float64,
        )
        firm_id = table["firm_id"].to_numpy(np.int64)
        simulated = np.asarray(simulation.calculate("firm_id", period)).astype(np.int64)
        if liability.shape != firm_id.shape or not np.array_equal(simulated, firm_id):
            raise ValueError(
                "policyengine-uk's firm_id order differs from the candidate table's."
            )
        if not np.all(np.isfinite(liability)):
            raise ValueError("policyengine-uk returned a non-finite VAT liability.")
        return UKFirmRuleOutputs(
            firm_id=firm_id,
            vat_liability_if_registered=liability,
            provenance={
                "rules": self.name,
                "policyengine_uk_version": policyengine_uk_version(),
                "variable": "firm_vat_liability_if_registered",
                "period": str(period),
                "output": "vat_liability_if_registered",
                "vat_composition": "section shares, firm_vat_composition.py (experimental)",
            },
        )


def uk_firm_vat_relief_checks(source: UKFirmVATCompositionSource) -> pd.DataFrame:
    """Modelled relief bases against HMRC relief cost / rate (pounds billion); a check only."""

    uses = source.supply_use.uses
    costs = source.relief_costs.set_index("code")["cost_2024_25_gbp_million"]
    food = [
        p for p in uses.index if p.startswith("CPA_C10") or p in ("CPA_A01", "CPA_A03")
    ]

    def household(product: str, key: str) -> float:
        share = _vector(source, product, "households")[
            UK_FIRM_VAT_TREATMENTS.index(key)
        ]
        return float(uses.loc[product, "households"]) * share

    rows = {}
    for name, (code, rate, key, products) in _RELIEF_CHECKS.items():
        if name.startswith("NS3"):
            model = source.dwellings_gfcf_gbp_million * UK_FIRM_NEW_DWELLING_FRACTION
        else:
            model = sum(
                household(p, key) for p in (food if products is None else products)
            )
        base = float(costs[code]) / rate
        rows[name] = {
            "model_base_gbp_bn": model / 1e3,
            "relief_base_gbp_bn": base / 1e3,
            "ratio": model / base,
        }
    return pd.DataFrame(rows).T


__all__ = [
    "UK_FIRM_INPUT_RATIO_BOUNDS",
    "UK_FIRM_NEW_DWELLING_FRACTION",
    "UK_FIRM_VAT_COMPOSITION_FILES",
    "UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS",
    "UK_FIRM_VAT_COMPOSITION_SHARE_COLUMNS",
    "UK_FIRM_VAT_COMPOSITION_STATUS",
    "UK_FIRM_VAT_SUPPLY_COLUMNS",
    "UK_FIRM_VAT_TREATMENTS",
    "UK_FIRM_VAT_USERS",
    "UKFirmVATCompositionRules",
    "UKFirmVATCompositionSource",
    "add_uk_firm_vat_composition_columns",
    "compose_uk_firm_vat_supplies",
    "read_uk_firm_vat_composition_source",
    "rescale_uk_firm_input_costs",
    "uk_firm_input_ratio_factors",
    "uk_firm_vat_product_shares",
    "uk_firm_vat_relief_checks",
    "uk_firm_vat_section_shares",
    "uk_firm_vat_voluntary_registration",
]
