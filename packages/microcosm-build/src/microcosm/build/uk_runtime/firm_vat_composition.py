"""UK firm VAT composition: zero-rated, reduced-rated and exempt supplies by SIC section.

Opt-in. Today every candidate firm is wholly standard-rated and its input VAT
is 20% of its input costs. This module replaces that, for a build that asks
for it, with the composition of each SIC section's sales and purchases in the
ONS Supply and Use Tables 2024 (Blue Book 2026), read through the VAT Act
1994 schedules. Every column it sets has status ``"assumed"``: the shares are
section averages, not observations of a firm.

**Sources** (``tests/fixtures/uk/firm_vat_composition_2024/``, with
``firm_supply_use_2024/`` for the product demand table):

- ``vat_product_treatment.csv`` (hand-written, 111 rows over the 104 SUT
  products): each product's domestic sales split into standard, reduced,
  zero and exempt shares, by buyer (``households`` / ``other``) where the law
  differs, with the VATA 1994 group it rests on. ``statute`` rows have one
  treatment by law; ``assumed`` rows carry a share set by judgement. Two
  rows are computed: construction (``DWELLINGS``) and wholesale and retail
  (``GOODS``), below.
- ``sut_intermediate_consumption.csv``: the 'Combined Use' intermediate
  consumption matrix, product by industry (industries written under the
  product they make).
- ``sut_dwellings_gfcf.csv``: dwellings GFCF of the construction product.
- ``abs_section_turnover.csv``: ABS 2024 turnover by section (only section
  G's is used).

**Output shares.** A product's sales are its Table 2 demand at purchasers'
prices: households, other domestic uses (intermediate consumption, NPISH,
government, GFCF, valuables, inventories floored at 0) and exports.
Households take the product's household treatment, other domestic uses its
other treatment, and exports are zero-rated (VATA s.30(6); exported services
are outside the scope with the right to recover, the same effect here).
Construction's zero share is dwellings GFCF times
``UK_FIRM_NEW_DWELLING_FRACTION`` (0.6, assumed) over its total demand
(Sch 8 Group 5). Retail (G47) turnover takes the treatment mix of household
purchases of goods (products with a positive distributors' margin); wholesale
(G46) takes the mix of non-household domestic and export demand for goods;
motor trade (G45) is standard. A section's shares are the domestic-output
weighted mean of its products' shares, each product counted once per section;
imputed rent (``CPA_L68A``) is excluded.

**Input VAT rate.** For industry ``j``, the intermediate-consumption-weighted
mean of each product's VAT rate on business purchases
(``0.20 * standard + 0.05 * reduced`` of its ``other`` treatment). A
section's rate weights its industries by intermediate consumption. Capital
purchases are not included (stated, not patched). pe-uk applies the partial
exemption restriction itself (taxable / (taxable + exempt), rounded up).

**Input ratio** (``input_ratio_target``). The SUT ratio of intermediate
consumption to output by section. Section G is on a gross-sales base: trade
output in the SUT is the margin only, so the target is
``1 - (margin - IC_G) / gross_sales`` with gross sales = ABS 2024 section G
turnover (assumed to share the SUT concept; wholesale sales to retailers are
counted in both). Section G's input VAT rate is
``(goods_resold * output_rate_G + IC_G * rate_IC_G) / (goods_resold + IC_G)``
with goods resold = gross sales - margin. A build applies the target by
rescaling each section's drawn input costs by one factor ``k_s``
(:func:`uk_firm_input_ratio_factors`) after the unchanged draw, so the
generator's random stream is untouched.

**Assumptions stated once.** Shares are uniform across a section's firms (no
all-or-nothing draw; a wholly exempt bank gets the section mix). Public
administration (O84) is treated as exempt: its s.33 and s.41 refunds are not
modelled. The 2024-25 part year of VAT on private school fees (from 1 January
2025) is folded into P85's assumed standard share. The generator's
``NEGATIVE_LIABILITY_SECTORS`` / ``HIGH_LIABILITY_SECTORS`` input-ratio shifts
still apply before the rescale and now overlap with the composition.

**Checks, never targets** (:func:`uk_firm_vat_relief_checks`): the modelled
household base of each zero- or reduced-rated relief against HMRC tax relief
statistics (January 2026) cost / rate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

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
UK_FIRM_VAT_COMPOSITION_FILES = (
    UK_FIRM_VAT_TREATMENT_FILE,
    UK_FIRM_VAT_INT_CON_FILE,
    UK_FIRM_VAT_DWELLINGS_FILE,
    UK_FIRM_VAT_ABS_FILE,
)
UK_FIRM_VAT_COMPOSITION_STATUS = "assumed"
#: The supplies, in the order of the treatment table's share columns.
UK_FIRM_VAT_TREATMENTS = ("standard", "reduced", "zero", "exempt")
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
#: The clamp the generator puts on the input ratio.
UK_FIRM_INPUT_RATIO_BOUNDS = (0.10, 0.95)
_CONSTRUCTION = "CPA_F41, F42 & F43"
_TRADE = ("CPA_G45", "CPA_G46", "CPA_G47")
_EXCLUDED = ("CPA_L68A",)
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
#: HMRC Tax relief statistics (January 2026), Table_2, 2024-25 cost (pounds
#: million) and the rate the relief replaces, for the checks only.
UK_FIRM_VAT_RELIEF_COSTS = {
    "NS7 food (zero)": (26300.0, 0.20),
    "NS1 printed matter (zero)": (1700.0, 0.20),
    "NS2 children's clothing (zero)": (2200.0, 0.20),
    "NS3 new dwellings (zero)": (14400.0, 0.20),
    "NS4 domestic fuel and power (reduced)": (6800.0, 0.15),
    "NS5 passenger transport (zero)": (5600.0, 0.20),
    "NS6 prescription drugs (zero)": (3800.0, 0.20),
    "NS16 water and sewerage (zero)": (2900.0, 0.20),
}


@dataclass(frozen=True)
class UKFirmVATCompositionSource:
    """The VAT composition inputs: treatment table, IC matrix, dwellings, ABS, SUT uses."""

    treatment: pd.DataFrame
    intermediate: pd.DataFrame
    dwellings_gfcf_gbp_million: float
    abs_turnover: pd.DataFrame
    supply_use: UKFirmSupplyUseSource
    sha256: Mapping[str, str]


def read_uk_firm_vat_composition_source(
    path: str | Path, supply_use_path: str | Path
) -> UKFirmVATCompositionSource:
    """Read the fixture directory and the supply and use directory it builds on.

    Raises ``ValueError`` when a file's sha256 differs from ``provenance.json``,
    a treatment row's shares do not sum to 1, a product lacks a treatment, or
    the matrix names an unknown product.
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
    treatment = pd.read_csv(directory / UK_FIRM_VAT_TREATMENT_FILE, dtype=str)
    computed = treatment["standard"].isin(["DWELLINGS", "GOODS"])
    numeric = treatment.loc[~computed, list(UK_FIRM_VAT_TREATMENTS)].astype(np.float64)
    if not np.allclose(numeric.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("VAT treatment shares must sum to 1 on every row.")
    if not set(treatment["user"]) <= {"all", "households", "other"}:
        raise ValueError("VAT treatment user must be all, households or other.")
    products = set(supply_use.uses.index)
    if set(treatment["product_code"]) != products:
        raise ValueError("Every SUT product needs a VAT treatment, and only those.")
    intermediate = pd.read_csv(directory / UK_FIRM_VAT_INT_CON_FILE)
    if (
        not set(intermediate["product_code"])
        | set(intermediate["industry_product_code"])
        <= products
    ):
        raise ValueError("The intermediate consumption matrix names unknown products.")
    dwellings = pd.read_csv(directory / UK_FIRM_VAT_DWELLINGS_FILE)
    abs_turnover = pd.read_csv(directory / UK_FIRM_VAT_ABS_FILE)
    return UKFirmVATCompositionSource(
        treatment=treatment,
        intermediate=intermediate,
        dwellings_gfcf_gbp_million=float(
            dwellings["dwellings_gfcf_gbp_million"].iloc[0]
        ),
        abs_turnover=abs_turnover,
        supply_use=supply_use,
        sha256=MappingProxyType(hashes | dict(supply_use.sha256)),
    )


def _treatment(
    source: UKFirmVATCompositionSource, product: str, user: str
) -> pd.Series:
    rows = source.treatment[source.treatment["product_code"] == product]
    for name in (user, "all", "other"):
        row = rows[rows["user"] == name]
        if not row.empty:
            return row.iloc[0]
    raise ValueError(f"No VAT treatment for {product}.")


def _goods(uses: pd.DataFrame) -> list[str]:
    return [p for p in uses.index if uses.loc[p, "distributors_margins"] > 0]


def _vector(
    source: UKFirmVATCompositionSource, product: str, user: str, *, nested: bool = False
) -> np.ndarray:
    """Standard, reduced, zero, exempt shares of ``product``'s sales to ``user``."""

    uses = source.supply_use.uses
    row = _treatment(source, product, user)
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
            exported = float(uses.loc[g, "exports"])
            total += domestic * _vector(source, g, "other", nested=True)
            total += exported * np.array([0.0, 0.0, 1.0, 0.0])
            mass += domestic + exported
        return total / mass
    return row[list(UK_FIRM_VAT_TREATMENTS)].astype(np.float64).to_numpy()


def uk_firm_vat_product_shares(source: UKFirmVATCompositionSource) -> pd.DataFrame:
    """Per product: output shares, export share, domestic output and input VAT rate."""

    uses = source.supply_use.uses
    exports = np.array([0.0, 0.0, 1.0, 0.0])
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
                + ex * exports
            ) / total
            export_share = ex / total
        rows[p] = dict(zip(UK_FIRM_VAT_TREATMENTS, shares, strict=True)) | {
            "export_share": export_share,
            "domestic_output": float(uses.loc[p, "domestic_output"]),
        }
    frame = pd.DataFrame(rows).T
    purchase_rate = pd.Series(
        {
            p: UK_FIRM_VAT_STANDARD_RATE * _vector(source, p, "other")[0]
            + UK_FIRM_VAT_REDUCED_RATE * _vector(source, p, "other")[1]
            for p in uses.index
            if p not in _EXCLUDED
        }
        | {p: 0.0 for p in _EXCLUDED}
    )
    ic = source.intermediate
    ic = ic.assign(vat=ic["value_gbp_million"] * ic["product_code"].map(purchase_rate))
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
    """By SIC section: output shares, input VAT rate, export share and input ratio target."""

    products = uk_firm_vat_product_shares(source)
    sections = source.supply_use.divisions[
        ["product_code", "sic_section"]
    ].drop_duplicates()
    sections = sections[sections["product_code"].isin(products.index)]
    rows = {}
    for section, group in sections.groupby("sic_section"):
        p = products.loc[group["product_code"]]
        w = p["domestic_output"]
        ic = p["intermediate_consumption"]
        row = {
            k: float(np.average(p[k], weights=w))
            for k in (*UK_FIRM_VAT_TREATMENTS, "export_share")
        }
        row["input_vat_rate"] = (
            float(np.average(p["input_vat_rate"].fillna(0.0), weights=ic))
            if ic.sum() > 0
            else 0.0
        )
        row["input_ratio_target"] = float(ic.sum() / w.sum())
        row["output_gbp_million"] = float(w.sum())
        rows[section] = row
    frame = pd.DataFrame(rows).T
    trade = products.loc[list(_TRADE)]
    margin = float(trade["domestic_output"].sum())
    ic_g = float(trade["intermediate_consumption"].sum())
    abs_g = source.abs_turnover.set_index("sic_section").loc[
        "G", "turnover_gbp_million"
    ]
    goods_resold = float(abs_g) - margin
    output_rate_g = (
        UK_FIRM_VAT_STANDARD_RATE * frame.loc["G", "standard"]
        + UK_FIRM_VAT_REDUCED_RATE * frame.loc["G", "reduced"]
    )
    frame.loc["G", "input_vat_rate"] = (
        goods_resold * output_rate_g + ic_g * frame.loc["G", "input_vat_rate"]
    ) / (goods_resold + ic_g)
    frame.loc["G", "input_ratio_target"] = 1.0 - (margin - ic_g) / float(abs_g)
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    frame["input_ratio_target"] = frame["input_ratio_target"].clip(lo, hi)
    shares = list(UK_FIRM_VAT_TREATMENTS)
    frame[shares] = frame[shares].div(frame[shares].sum(axis=1), axis=0)
    frame.index.name = "sic_section"
    return frame


def compose_uk_firm_vat_supplies(
    firms: pd.DataFrame, shares: pd.DataFrame, in_scope: np.ndarray | None = None
) -> pd.DataFrame:
    """``firms`` with the four supply columns and ``firm_input_vat`` from section shares.

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
    """Registered rows whose taxable turnover is at or below the threshold.

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
    """``k_s`` per section: turnover-weighted mean of clip(ratio * k_s) = target.

    Deterministic bisection on ``log k`` in [1e-3, 1e3]. A section with no
    positive turnover, or no target, keeps ``k = 1``. Returns, per section,
    ``k``, the target, the achieved mean and the mean before the rescale.
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
        }
    return out


def rescale_uk_firm_input_costs(
    turnover: np.ndarray,
    input_costs: np.ndarray,
    sections: np.ndarray,
    targets: pd.Series,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Input costs rescaled by section to the target ratio, and the factors used.

    Only rows with positive turnover move; the result is
    ``turnover * clip(input / turnover * k_s, 0.10, 0.95)``.
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
    lo, hi = UK_FIRM_INPUT_RATIO_BOUNDS
    return np.where(positive, turnover * np.clip(ratio * k, lo, hi), 0.0), factors


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
    """``UKFirmRules`` that evaluate VAT on the section composition through policyengine-uk.

    ``PolicyEngineUKFirmRules`` hands only its fixed input columns to the
    simulation, which have no zero- or reduced-rated supplies, so recovery
    would be standard / (standard + exempt). This class composes the
    candidate table (:func:`compose_uk_firm_vat_supplies`) and calls the
    simulation factory itself with ``UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS``.
    ``simulation_factory`` defaults to ``policyengine_uk_firm_simulation``.
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
                "vat_composition": "section shares, firm_vat_composition.py",
            },
        )


def uk_firm_vat_relief_checks(source: UKFirmVATCompositionSource) -> pd.DataFrame:
    """Modelled relief bases against HMRC relief cost / rate (pounds billion); a check only."""

    uses = source.supply_use.uses

    def household(product: str, key: str) -> float:
        share = _vector(source, product, "households")[
            UK_FIRM_VAT_TREATMENTS.index(key)
        ]
        return float(uses.loc[product, "households"]) * share

    food = [
        p for p in uses.index if p.startswith("CPA_C10") or p in ("CPA_A01", "CPA_A03")
    ]
    model = {
        "NS7 food (zero)": sum(household(p, "zero") for p in food),
        "NS1 printed matter (zero)": household("CPA_J58", "zero"),
        "NS2 children's clothing (zero)": household("CPA_C14", "zero")
        + household("CPA_C15", "zero"),
        "NS3 new dwellings (zero)": source.dwellings_gfcf_gbp_million
        * UK_FIRM_NEW_DWELLING_FRACTION,
        "NS4 domestic fuel and power (reduced)": household("CPA_D351", "reduced")
        + household("CPA_D352_3", "reduced"),
        "NS5 passenger transport (zero)": sum(
            household(p, "zero")
            for p in ("CPA_H491_2", "CPA_H493_5", "CPA_H50", "CPA_H51")
        ),
        "NS6 prescription drugs (zero)": 0.0,
        "NS16 water and sewerage (zero)": household("CPA_E36", "zero")
        + household("CPA_E37", "zero"),
    }
    rows = {
        name: {
            "model_base_gbp_bn": model[name] / 1e3,
            "relief_base_gbp_bn": cost / rate / 1e3,
            "ratio": model[name] / (cost / rate),
        }
        for name, (cost, rate) in UK_FIRM_VAT_RELIEF_COSTS.items()
    }
    return pd.DataFrame(rows).T


__all__ = [
    "UK_FIRM_INPUT_RATIO_BOUNDS",
    "UK_FIRM_NEW_DWELLING_FRACTION",
    "UK_FIRM_VAT_COMPOSITION_FILES",
    "UK_FIRM_VAT_COMPOSITION_RULE_COLUMNS",
    "UK_FIRM_VAT_COMPOSITION_STATUS",
    "UK_FIRM_VAT_RELIEF_COSTS",
    "UK_FIRM_VAT_SUPPLY_COLUMNS",
    "UK_FIRM_VAT_TREATMENTS",
    "UKFirmVATCompositionRules",
    "UKFirmVATCompositionSource",
    "compose_uk_firm_vat_supplies",
    "read_uk_firm_vat_composition_source",
    "rescale_uk_firm_input_costs",
    "uk_firm_input_ratio_factors",
    "uk_firm_vat_product_shares",
    "uk_firm_vat_relief_checks",
    "uk_firm_vat_section_shares",
    "uk_firm_vat_voluntary_registration",
]
