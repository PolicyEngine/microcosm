"""How each count-like constant of the UK spine meets the geographic clone count.

The clone-first spine gives every FRS household ``K`` geographic clones
(microcosm#931), each at ``1 / K`` of its weight. A threshold on rows then
sees ``K`` times the rows it saw at one clone, while one on distinct true
source households, on survey records or on weight does not. Each constant
below is classified by what it counts and by how the code meets ``K``, and
pinned to the code's value by import, so a change to one fails here until
it is classified again. The gates' count-like maxima and minima are pinned
the same way: a new one in the packaged gates must join the table.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files

import pytest

from microcosm.build import stochastic_assignment
from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import (
    aps_occupation,
    area_home_values,
    cgt_support,
    child_benefit_take_up,
    company_cars,
    frs_person_draws,
    frs_take_up,
    high_value_home_support,
    private_rent_localisation,
    property_localisation,
    salary_sacrifice,
    spi_band_donors,
    spi_benefit_coherence,
    spi_employment_shell,
    spi_rank_transport,
    stage_health,
    support_clip,
    uc_capital_coherence,
)

#: What a constant counts: "ratio" bounds a ratio of weighted means, which
#: K clones at 1 / K of their weight leave as it was; "share" a share of a
#: group's weight, which they leave as it was too; "value" a value in the
#: data's own units, which they do not change; "published cells" the cells
#: of a published table a stage loads or compares with, which the frame does
#: not set.
COUNTS = frozenset(
    {
        "source count",
        "survey records",
        "rows",
        "weight",
        "ratio",
        "share",
        "value",
        "hyperparameter",
        "published cells",
    }
)
#: How the code meets the clone count, each with what it means.
TREATMENTS = {
    "per clone": (
        "declared for one clone: the stage applies it within each clone, and a "
        "check that reads it over every clone holds K times the stage's figure "
        "to it"
    ),
    "invariant": "what it counts does not move with K",
    "kept": "it moves with K, and is kept as it is for the reason given",
    "scaled by K": (
        "a row allowance the stage-health check multiplies by the receipt's "
        "clone_count, which a stage gives only from two clones up"
    ),
    "clone 0": "counted in geographic clone 0's rows, which do not move with K",
    "later": "taken up later in microcosm#931, for the reason given",
}


@dataclass(frozen=True)
class KSensitiveConstant:
    """One constant: where it lives, its value, what it counts and how the
    code meets ``K``. ``read`` returns the code's value now."""

    name: str
    where: str
    read: Callable[[], object]
    value: object
    counts: str
    treatment: str
    reason: str
    #: The packaged gates' (gate id, parameter) pairs that declare it.
    gate_parameters: frozenset[tuple[str, str]] = frozenset()


def _gates() -> list[dict]:
    payload = files("microcosm.build.uk").joinpath("gates.json").read_text()
    return json.loads(payload)["gates"]


def _declared_parameters(check: str, *keys: str) -> frozenset[tuple[str, str]]:
    """The packaged gates of ``check`` that declare each of ``keys``."""

    return frozenset(
        (gate["id"], key)
        for gate in _gates()
        if (gate.get("parameters") or {}).get("check") == check
        for key in keys
        if key in gate["parameters"]
    )


def _declared_values(
    pairs: tuple[tuple[str, str], ...],
) -> frozenset[tuple[str, str, object]]:
    """Each (gate id, parameter) pair with the value its packaged gate
    declares."""

    parameters = {gate["id"]: gate.get("parameters") or {} for gate in _gates()}
    return frozenset((gate, key, parameters[gate][key]) for gate, key in pairs)


def _company_car_donor_households() -> int:
    stage = load_country_spec("uk").sources.stage_map()["company_cars"]
    return company_cars.CompanyCarsSpec.from_stage(stage).minimum_donor_households


def _source_names(function: Callable[..., object], *names: str) -> bool:
    source = inspect.getsource(function)
    return all(name in source for name in names)


_SUPPORT_CLIP_STAGES = (
    "was_wealth",
    "was_lisa",
    "nts_bus_travel",
    "lcfs_consumption",
    "etb_vat",
    "etb_services",
)
_CLIP_ALLOWANCES = (
    "max_clipped_low_rows_by_column",
    "max_clipped_high_rows_by_column",
)
_PROPERTY_MAXIMA = (
    "maximum_rows_receipts_below_profit",
    "maximum_rows_with_profit_above_band",
)
#: The band donor gate's minimum seats per band.
_BAND_DONOR_MINIMUM = (
    "uk_stage_spi_income_band_donors_support",
    "minimum_donors_per_band",
)
#: The stage gates' floors of one, each on rows counted over every clone.
_FLOORS_OF_ONE = (
    ("uk_stage_spi_support_channel_mass", "minimum_spi_households"),
    ("uk_stage_hmrc_spi_income_spine_identity", "minimum_identity_rows"),
    ("uk_stage_pension_credit_take_up", "minimum_entitled_units"),
    ("uk_stage_child_benefit_take_up", "minimum_eligible_family_units"),
    ("uk_stage_cgt_incidence_anchor_composition", "minimum_pair_count"),
)
#: The stage gates' floors on the cells of a published table: the SPI income
#: stage's HMRC income targets, and the CGT gains summary's rows, one per
#: published gain band.
_PUBLISHED_CELL_FLOORS = (
    ("uk_stage_hmrc_spi_income_spine_identity", "minimum_target_count"),
    ("uk_stage_hmrc_cgt_gains_spine_summary", "minimum_band_rows"),
)

K_SENSITIVE_CONSTANTS: tuple[KSensitiveConstant, ...] = (
    KSensitiveConstant(
        name="private rents' sampled region renters",
        where="private_rent_localisation.MIN_SAMPLED_REGION_RENTERS",
        read=lambda: private_rent_localisation.MIN_SAMPLED_REGION_RENTERS,
        value=10,
        counts="source count",
        treatment="invariant",
        reason=(
            "a sampled spine ranks and scales a region against Great Britain "
            "where it holds fewer FRS renters than this, counted in distinct "
            "true source households, so a renter's K clones and copies count "
            "once (private_rent, microcosm#931)"
        ),
    ),
    KSensitiveConstant(
        name="private rents' scale band",
        where="private_rent_localisation.SCALE_BAND",
        read=lambda: private_rent_localisation.SCALE_BAND,
        value=(0.25, 4.0),
        counts="ratio",
        treatment="invariant",
        reason=(
            "bounds each region's scale, the FRS mean rent over the census mix "
            "of its BRMAs' list means, both weighted means that K clones at "
            "1 / K of their weight leave unchanged (private_rent, "
            "microcosm#931)"
        ),
    ),
    KSensitiveConstant(
        name="band donor maximum weight",
        where="spi_band_donors.SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT",
        read=lambda: spi_band_donors.SPI_INCOME_BAND_MAXIMUM_DONOR_WEIGHT,
        value=250.0,
        counts="weight",
        treatment="per clone",
        reason=(
            "each clone's donors weigh a 1 / K share, and the band donor "
            "check holds K times a donor's weight to the maximum"
        ),
    ),
    KSensitiveConstant(
        name="CGT support copy maximum weight",
        where="cgt_support.CGT_SUPPORT_MAXIMUM_COPY_WEIGHT",
        read=lambda: cgt_support.CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
        value=60.0,
        counts="weight",
        treatment="per clone",
        reason="a copy weighs at most the maximum divided by K",
    ),
    KSensitiveConstant(
        name="band donor minimum seats",
        where=(
            "spi_band_donors.SPI_INCOME_BAND_MINIMUM_DONORS, and the band donor "
            "gate's minimum_donors_per_band (stage_health), which the check "
            "holds the receipt's minimum to"
        ),
        read=lambda: (
            spi_band_donors.SPI_INCOME_BAND_MINIMUM_DONORS,
            _declared_values((_BAND_DONOR_MINIMUM,)),
        ),
        value=(120, frozenset({(*_BAND_DONOR_MINIMUM, 120)})),
        counts="source count",
        treatment="per clone",
        reason=(
            "every band seats at least this many donors in each clone, and "
            "the band donor check requires K times it"
        ),
        gate_parameters=frozenset({_BAND_DONOR_MINIMUM}),
    ),
    KSensitiveConstant(
        name="band donor household share",
        where="spi_band_donors.SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE",
        read=lambda: spi_band_donors.SPI_INCOME_BAND_MAXIMUM_DONOR_HOUSEHOLD_SHARE,
        value=0.15,
        counts="source count",
        treatment="invariant",
        reason=(
            "a share of the eligible base households counted in distinct "
            "source households"
        ),
    ),
    KSensitiveConstant(
        name="company car donor households",
        where=(
            "the company_cars stage's minimum_donor_households "
            "(company_cars.CompanyCarsSpec)"
        ),
        read=_company_car_donor_households,
        value=3,
        counts="source count",
        treatment="invariant",
        reason="distinct WAS donor households, and donors are never cloned",
    ),
    KSensitiveConstant(
        name="bonus sacrifice donor sources",
        where="salary_sacrifice.BONUS_SACRIFICE_SPI_MIN_DONOR_SOURCES",
        read=lambda: salary_sacrifice.BONUS_SACRIFICE_SPI_MIN_DONOR_SOURCES,
        value=20,
        counts="source count",
        treatment="invariant",
        reason="distinct true source households among a cell's donors",
    ),
    KSensitiveConstant(
        name="UC capital cell donors",
        where="uc_capital_coherence.UC_CAPITAL_MINIMUM_CELL_DONORS",
        read=lambda: uc_capital_coherence.UC_CAPITAL_MINIMUM_CELL_DONORS,
        value=20,
        counts="survey records",
        treatment="invariant",
        reason=(
            "a benefit unit's geographic clones count once, whether they fold "
            "into one row or donate apart"
        ),
    ),
    KSensitiveConstant(
        name="CGT support correction candidates",
        where="cgt_support.CGT_SUPPORT_CORRECTION_CANDIDATES",
        read=lambda: cgt_support.CGT_SUPPORT_CORRECTION_CANDIDATES,
        value=64,
        counts="rows",
        treatment="kept",
        reason=(
            "the heaviest whole households tried for the exact-total "
            "correction; they cover fewer sources at K clones, and the "
            "correction they place is of rounding size"
        ),
    ),
    KSensitiveConstant(
        name="SPI transport group minimum rows",
        where="spi_rank_transport.SPI_TRANSPORT_GROUP_MINIMUM_ROWS",
        read=lambda: spi_rank_transport.SPI_TRANSPORT_GROUP_MINIMUM_ROWS,
        value=10,
        counts="rows",
        treatment="clone 0",
        reason=(
            "a statistical floor of a uniformity diagnostic only the receipt "
            "reports, which measures geographic clone 0's recipients, since "
            "each clone draws its own copies; a group also needs the "
            "disclosure floor's source households among them"
        ),
    ),
    KSensitiveConstant(
        name="support-clip row allowances",
        where=(
            "the six support-clip gates' max_clipped_low_rows_by_column and "
            "max_clipped_high_rows_by_column (stage_health, support_clip)"
        ),
        read=lambda: _declared_parameters("support_clip", *_CLIP_ALLOWANCES),
        value=frozenset(
            (f"uk_stage_{stage}_support", key)
            for stage in _SUPPORT_CLIP_STAGES
            for key in _CLIP_ALLOWANCES
        ),
        counts="rows",
        treatment="scaled by K",
        reason="every clone's draws are clipped",
        gate_parameters=frozenset(
            (f"uk_stage_{stage}_support", key)
            for stage in _SUPPORT_CLIP_STAGES
            for key in _CLIP_ALLOWANCES
        ),
    ),
    KSensitiveConstant(
        name="property components row maxima",
        where=(
            "the property components gate's maximum_rows_receipts_below_profit "
            "and maximum_rows_with_profit_above_band (stage_health)"
        ),
        read=lambda: _declared_parameters("property_components", *_PROPERTY_MAXIMA),
        value=frozenset(
            ("uk_stage_property_components", key) for key in _PROPERTY_MAXIMA
        ),
        counts="rows",
        treatment="scaled by K",
        reason="every clone's landlords are walked",
        gate_parameters=frozenset(
            ("uk_stage_property_components", key) for key in _PROPERTY_MAXIMA
        ),
    ),
    KSensitiveConstant(
        name="area home values sources at the cap",
        where=(
            "the area home values ranking gate's maximum_sources_at_cap "
            "(uk_stage_area_home_values_ranking, stage_health), against the "
            "WAS stage's area home values sources_at_cap (area_home_values)"
        ),
        read=lambda: _declared_parameters("area_home_values", "maximum_sources_at_cap"),
        value=frozenset(
            {("uk_stage_area_home_values_ranking", "maximum_sources_at_cap")}
        ),
        counts="source count",
        treatment="clone 0",
        reason=(
            "each clone ranks on its own draw and reads its own constituency's "
            "knots, so the sources at the cap over every clone grow with K"
        ),
        gate_parameters=frozenset(
            {("uk_stage_area_home_values_ranking", "maximum_sources_at_cap")}
        ),
    ),
    KSensitiveConstant(
        name="area home values knots cap",
        where=(
            "the cap of the constituency knots the WAS stage's area home "
            "values read (high_value_home_value_knots.json, area_home_values)"
        ),
        read=lambda: (
            area_home_values.load_committed_area_home_values_resources().knots.cap
        ),
        value=20_000_000.0,
        counts="value",
        treatment="invariant",
        reason=(
            "a home value the published knots are winsorised at, which every "
            "clone reads the same; what moves with K is the count of sources "
            "at it, counted in clone 0 (above)"
        ),
    ),
    KSensitiveConstant(
        name="high-value support stratum",
        where=(
            "property_localisation.SUPPORT_STRATUM_LOWER, the high-value "
            "home support's stratum_lower "
            "(high_value_home_support.HIGH_VALUE_SUPPORT_STRATUM_LOWER)"
        ),
        read=lambda: (
            property_localisation.SUPPORT_STRATUM_LOWER,
            high_value_home_support.HIGH_VALUE_SUPPORT_STRATUM_LOWER,
        ),
        value=(0.95, 0.95),
        counts="share",
        treatment="invariant",
        reason=(
            "a weighted mid-rank within each FRS region's owners, a share of "
            "their weight, which K clones at 1 / K of it leave as it was"
        ),
    ),
    KSensitiveConstant(
        name="high-value support copies",
        where="high_value_home_support.HIGH_VALUE_SUPPORT_COPIES",
        read=lambda: high_value_home_support.HIGH_VALUE_SUPPORT_COPIES,
        value=6,
        counts="hyperparameter",
        treatment="invariant",
        reason=(
            "the copies each selected household row becomes, at an equal "
            "share of its weight; the count does not move with K, though the "
            "stratum ranks every clone's rows together (C-34)"
        ),
    ),
    KSensitiveConstant(
        name="relationship grid mismatches",
        where=(
            "the household composition gate's max_grid_reciprocity_mismatches "
            "(stage_health), against frs_relationships' receipt"
        ),
        read=lambda: _declared_parameters(
            "household_composition", "max_grid_reciprocity_mismatches"
        ),
        value=frozenset(
            {
                (
                    "uk_stage_frs_relationships_composition",
                    "max_grid_reciprocity_mismatches",
                )
            }
        ),
        counts="survey records",
        treatment="invariant",
        reason=(
            "frs_relationships runs before geographic_expand, so the grids it "
            "counts are the survey households' own"
        ),
        gate_parameters=frozenset(
            {
                (
                    "uk_stage_frs_relationships_composition",
                    "max_grid_reciprocity_mismatches",
                )
            }
        ),
    ),
    KSensitiveConstant(
        name="HMRC leaf signal rows",
        where=(
            "the HMRC leaves' source signal gate's minimum_signal_rows "
            "(stage_health), against frs_hmrc_spine_leaves' source_signal_rows"
        ),
        read=lambda: _declared_values(
            (("uk_stage_frs_hmrc_spine_leaves_signal", "minimum_signal_rows"),)
        ),
        value=frozenset(
            {("uk_stage_frs_hmrc_spine_leaves_signal", "minimum_signal_rows", 1)}
        ),
        counts="survey records",
        treatment="invariant",
        reason=(
            "frs_hmrc_spine_leaves runs before geographic_expand, so the rows it "
            "counts are the survey's own"
        ),
        gate_parameters=frozenset(
            {("uk_stage_frs_hmrc_spine_leaves_signal", "minimum_signal_rows")}
        ),
    ),
    KSensitiveConstant(
        name="Child Benefit age rows",
        where=(
            "the Child Benefit take-up gate's minimum_age_child_rows "
            "(stage_health), against child_benefit_take_up's eligible_child_rows "
            "at each age, which the gate's sampling band reads as its n too"
        ),
        read=lambda: _declared_values(
            (("uk_stage_child_benefit_take_up", "minimum_age_child_rows"),)
        ),
        value=frozenset(
            {("uk_stage_child_benefit_take_up", "minimum_age_child_rows", 200)}
        ),
        counts="rows",
        treatment="clone 0",
        reason=(
            "the stage counts each age's eligible child rows among geographic "
            "clone 0's rows, the rows one clone has, so the gate measures the "
            "same ages against the same bands at every K; counted over every "
            "clone, an age too thin to measure at one clone would be measured "
            "at K clones, against a band narrower by the square root of K"
        ),
        gate_parameters=frozenset(
            {("uk_stage_child_benefit_take_up", "minimum_age_child_rows")}
        ),
    ),
    KSensitiveConstant(
        name="floors of one row",
        where=(
            "the stage gates' floors of one: the SPI support channel's "
            "minimum_spi_households, the SPI income spine's minimum_identity_rows, "
            "Pension Credit's minimum_entitled_units, Child Benefit's "
            "minimum_eligible_family_units and the CGT incidence anchor's "
            "minimum_pair_count (stage_health)"
        ),
        read=lambda: _declared_values(_FLOORS_OF_ONE),
        value=frozenset((gate, key, 1) for gate, key in _FLOORS_OF_ONE),
        counts="rows",
        treatment="kept",
        reason=(
            "each asks only that its stage's population is not empty, counted in "
            "rows over every clone, which hold clone 0's rows among them, so more "
            "clones can only meet a floor of one more easily"
        ),
        gate_parameters=frozenset(_FLOORS_OF_ONE),
    ),
    KSensitiveConstant(
        name="published cell floors",
        where=(
            "the SPI income spine gate's minimum_target_count, against the count "
            "of HMRC income targets its stage loads, and the CGT gains summary "
            "gate's minimum_band_rows, against its rows, one per published gain "
            "band (stage_health)"
        ),
        read=lambda: _declared_values(_PUBLISHED_CELL_FLOORS),
        value=frozenset((gate, key, 1) for gate, key in _PUBLISHED_CELL_FLOORS),
        counts="published cells",
        treatment="invariant",
        reason=(
            "each counts the cells of a published table, which the stage loads or "
            "compares with whatever the frame holds"
        ),
        gate_parameters=frozenset(_PUBLISHED_CELL_FLOORS),
    ),
    KSensitiveConstant(
        name="take-up signal under-2 usage population support",
        where=(
            "frs_take_up.uk_take_up_signal_receipt's population_support of the "
            "under-2 usage row"
        ),
        read=lambda: _source_names(
            frs_take_up.uk_take_up_signal_receipt,
            "population_support",
            "UK_UNDER_TWO_USAGE_SIGNAL_KEY",
        ),
        value=True,
        counts="source count",
        treatment="clone 0",
        reason=(
            "the population is the children under 2 in units each clone draws "
            "as claiming the extended hours, so its sources over every clone "
            "grow with K, and with them the rows whose share is withheld"
        ),
    ),
    KSensitiveConstant(
        name="take-up signal distinct values",
        where="frs_take_up.uk_take_up_signal_receipt's distinct_values",
        read=lambda: _source_names(
            frs_take_up.uk_take_up_signal_receipt, "distinct_values"
        ),
        value=True,
        counts="rows",
        treatment="clone 0",
        reason=(
            "each clone draws its own flags, so a flag constant on one clone "
            "can vary over K"
        ),
    ),
    KSensitiveConstant(
        name="apprentice flagged support",
        where="frs_person_draws.derive_is_apprentice's flagged_support",
        read=lambda: _source_names(
            frs_person_draws.derive_is_apprentice, "flagged_support"
        ),
        value=True,
        counts="source count",
        treatment="clone 0",
        reason=(
            "each clone flags its own employees, so the flagged sources over "
            "every clone grow with K, and with them the figures it withholds"
        ),
    ),
    KSensitiveConstant(
        name="occupation leaf minimum",
        where=(
            "min_samples_leaf of spi_employment_shell's occupation model and "
            "aps_occupation's imputation"
        ),
        read=lambda: (
            spi_employment_shell.UK_SPI_EMPLOYMENT_SHELL_OPERATION["occupation"][
                "boosted"
            ]["min_samples_leaf"],
            aps_occupation.UK_APS_OCCUPATION_IMPUTATION["boosted"]["min_samples_leaf"],
        ),
        value=(40, 40),
        counts="hyperparameter",
        treatment="later",
        reason=(
            "rows of a fit, where geographic clones fold together unless they "
            "differ; it is taken up with the fits' cross-fitted folds by true "
            "source, later in #931"
        ),
    ),
    KSensitiveConstant(
        name="band donor reallocation share",
        where="spi_band_donors.SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE",
        read=lambda: spi_band_donors.SPI_INCOME_BAND_MAXIMUM_REALLOCATION_SHARE,
        value=0.5,
        counts="weight",
        treatment="later",
        reason=(
            "a share of each funding stratum's mass, the stratum now a region "
            "within a clone; its common mass scale is measured on a spine of "
            "fifteen clones before any change"
        ),
    ),
    KSensitiveConstant(
        name="take-up target count",
        where=(
            "frs_take_up.derive_frs_take_up's Child Benefit, Pension Credit and "
            "Universal Credit draws: the int(rate x n) of "
            "stochastic_assignment.assign_binary_with_anchored_residual, taken "
            "within its strata"
        ),
        read=lambda: (
            _source_names(
                stochastic_assignment.assign_binary_with_anchored_residual,
                "int(rate * int(population.sum()))",
                "np.unique(strata)",
            ),
            inspect.getsource(frs_take_up.derive_frs_take_up).count("strata=strata"),
            _source_names(
                frs_take_up.add_frs_take_up, "clone_index=geographic_clone_indices"
            ),
        ),
        value=(True, 3, True),
        counts="source count",
        treatment="per clone",
        reason=(
            "frs_take_up hands the draws each benefit unit's geographic clone "
            "as their strata, so each clone counts its own units, each FRS "
            "benefit unit once, and clone 0's count is one clone's"
        ),
    ),
    KSensitiveConstant(
        name="SPI take-up target count",
        where=(
            "spi_benefit_coherence._redraw_spi_uc_take_up's Universal Credit "
            "draw: the int(rate x n) of "
            "stochastic_assignment.assign_binary_with_anchored_residual"
        ),
        read=lambda: (
            _source_names(
                spi_benefit_coherence._redraw_spi_uc_take_up,
                "assign_binary_with_anchored_residual",
            ),
            _source_names(spi_benefit_coherence._redraw_spi_uc_take_up, "strata"),
        ),
        value=(True, False),
        counts="rows",
        treatment="kept",
        reason=(
            "int(rate x n) rounds once over the SPI units of every clone: the "
            "redraw lies outside the stages #931 nests clone 0 on, so it keeps "
            "one count over every clone, which its clone index as strata would "
            "take per clone"
        ),
    ),
    KSensitiveConstant(
        name="apprentice target",
        where=(
            "frs_person_draws.derive_is_apprentice's walk target, the rate times "
            "the weight aged 21 to 24"
        ),
        read=lambda: (
            _source_names(
                frs_person_draws.derive_is_apprentice,
                "_clone_rows(person_clone_index",
                "weights[in_ages & rows]",
            ),
            _source_names(
                frs_person_draws.add_frs_person_draws_with_receipt,
                "person_clone_index=geographic_clone_indices",
            ),
        ),
        value=(True, True),
        counts="weight",
        treatment="per clone",
        reason=(
            "each clone walks its own employees to the rate times its own "
            "weight aged 21 to 24, a 1 / K share, so clone 0's walk is one "
            "clone's at its weights over K"
        ),
    ),
    KSensitiveConstant(
        name="zero-weight SPI strata rows",
        where="the zero-weight strata gate's maximum_zero_weight_rows",
        # The gate and key of each declared maximum, with its value.
        read=lambda: frozenset(
            (
                "uk_zero_weight_strata",
                "maximum_zero_weight_rows",
                declaration["maximum_zero_weight_rows"],
            )
            for gate in _gates()
            if gate["id"] == "uk_zero_weight_strata"
            for declaration in gate["parameters"]["declarations"]
            if "maximum_zero_weight_rows" in declaration
        ),
        value=frozenset({("uk_zero_weight_strata", "maximum_zero_weight_rows", 0)}),
        counts="rows",
        treatment="kept",
        reason=(
            "a terminal gate whose every declared maximum is zero rows, which "
            "K clones leave at zero: the dense line, which builds on the "
            "packaged geography-first spine at the K its clone count pin holds "
            "it to, reads K rows where one clone has one, and any SPI-synthetic "
            "row at zero weight still fails; the national line builds at one "
            "clone, where K is one"
        ),
        gate_parameters=frozenset(
            {("uk_zero_weight_strata", "maximum_zero_weight_rows")}
        ),
    ),
    KSensitiveConstant(
        name="local area support rows",
        where="the local area support gate's minimum_rows",
        read=lambda: next(
            gate["parameters"]["minimum_rows"]
            for gate in _gates()
            if gate["id"] == "uk_local_area_support"
        ),
        value=50,
        counts="rows",
        treatment="kept",
        reason=(
            "the rows each area must hold on the dense pool, which is the "
            "bound spine with its K clones (#931), so an area holds K times "
            "its rows at one clone; releases run at the doctrine's K = 15 "
            "only, the count the floor was set at"
        ),
        gate_parameters=frozenset({("uk_local_area_support", "minimum_rows")}),
    ),
    KSensitiveConstant(
        name="local area support ESS",
        where="the local area support gate's minimum_effective_sample_size",
        read=lambda: next(
            gate["parameters"]["minimum_effective_sample_size"]
            for gate in _gates()
            if gate["id"] == "uk_local_area_support"
        ),
        value=50.0,
        counts="rows",
        treatment="kept",
        reason=(
            "each area's effective sample size over its household rows on the "
            "dense pool, where a source's K clones are K rows at 1 / K of its "
            "weight, so it grows with K as the row floor does; releases run "
            "at the doctrine's K = 15 only, the count the floor was set at"
        ),
    ),
    KSensitiveConstant(
        name="local area support sources",
        where="the local area support gate's minimum_distinct_sources",
        read=lambda: next(
            gate["parameters"]["minimum_distinct_sources"]
            for gate in _gates()
            if gate["id"] == "uk_local_area_support"
        ),
        value=50,
        counts="source count",
        treatment="kept",
        reason=(
            "the distinct true sources each area must hold: a source counts "
            "once in an area however many of its clones land there, but its "
            "K clones are placed apart, so an area meets more sources as K "
            "grows; releases run at the doctrine's K = 15 only, the count the "
            "floor was set at"
        ),
        gate_parameters=frozenset(
            {("uk_local_area_support", "minimum_distinct_sources")}
        ),
    ),
    KSensitiveConstant(
        name="high-value homes support sources",
        where=(
            "the high-value homes support gate's minimum_distinct_sources "
            "(high_value_home_gates.high_value_home_support_gate)"
        ),
        read=lambda: _declared_values(
            (("uk_high_value_homes_support", "minimum_distinct_sources"),)
        ),
        value=frozenset(
            {("uk_high_value_homes_support", "minimum_distinct_sources", 3)}
        ),
        counts="source count",
        treatment="kept",
        reason=(
            "the distinct true sources behind each assessed area's homes above "
            "a threshold on the calibrated pool: a source counts once in an "
            "area however many of its clones and copies land there, but its K "
            "clones are placed apart, so an area meets more sources as K "
            "grows; the floor is the disclosure floor, which counts sources "
            "this way at every K, on a diagnostic gate"
        ),
        gate_parameters=frozenset(
            {("uk_high_value_homes_support", "minimum_distinct_sources")}
        ),
    ),
)


@pytest.mark.parametrize(
    "constant", K_SENSITIVE_CONSTANTS, ids=lambda constant: constant.name
)
def test_each_constant_is_the_codes(constant: KSensitiveConstant) -> None:
    assert constant.read() == constant.value, constant.where


def test_each_constant_is_classified() -> None:
    names = [constant.name for constant in K_SENSITIVE_CONSTANTS]
    assert len(set(names)) == len(names)
    for constant in K_SENSITIVE_CONSTANTS:
        assert constant.counts in COUNTS, constant.name
        assert constant.treatment in TREATMENTS, constant.name
        assert constant.where and constant.reason, constant.name
    # A row threshold is never invariant: it is scaled, counted in clone 0,
    # kept for its reason or taken up later.
    assert not [
        constant.name
        for constant in K_SENSITIVE_CONSTANTS
        if constant.counts == "rows" and constant.treatment == "invariant"
    ]


#: A gate parameter that bounds a count: a maximum or a minimum (a name that
#: starts with max or min) of rows, sources, households, records, donors,
#: cells, mismatches, units or a count.
_COUNT_BOUND = re.compile(
    r"^(max|min)(imum)?_(.*_)?"
    r"(rows|sources|households|records|donors|cells|mismatches|units|count)(_.*)?$"
)


def _count_bounds(value: object) -> set[str]:
    if isinstance(value, dict):
        found = {key for key in value if _COUNT_BOUND.match(key)}
        for item in value.values():
            found |= _count_bounds(item)
        return found
    if isinstance(value, list):
        return set().union(*(_count_bounds(item) for item in value))
    return set()


def test_the_count_bound_pattern_reads_minima_as_well_as_maxima() -> None:
    for name in (
        "maximum_rows",
        "max_clipped_low_rows_by_column",
        "minimum_age_child_rows",
        "min_rows",
        "minimum_distinct_sources",
        "minimum_eligible_family_units",
        "minimum_pair_count",
    ):
        assert _COUNT_BOUND.match(name), name
    for name in (
        "minimum",
        "maximum_relative_deviation",
        "min_family_share",
        "minimum_effective_sample_size",
        "minimum_target_homes",
        "rows_minimum",
    ):
        assert not _COUNT_BOUND.match(name), name


def test_every_count_bound_in_the_gates_is_classified() -> None:
    """A count-like maximum or minimum a packaged gate declares is in the
    table, so a new one is classified before a spine of several clones reads
    it."""

    declared = {
        (gate["id"], key)
        for gate in _gates()
        for key in _count_bounds(gate.get("parameters") or {})
    }
    classified = set().union(
        *(constant.gate_parameters for constant in K_SENSITIVE_CONSTANTS)
    )
    assert declared == classified
    # The wider pattern reads minima: the Child Benefit gate's row floor
    # among them.
    assert ("uk_stage_child_benefit_take_up", "minimum_age_child_rows") in declared


def test_the_scaled_and_clone_zero_treatments_are_the_codes() -> None:
    """The row allowances the table scales are read through the receipt's
    clone count, which the shared clip receipt and the property components
    stage give; the counts the table counts in clone 0 are the stages'."""

    for function in (
        stage_health._support_clip_gate,
        stage_health._property_components_gate,
    ):
        assert _source_names(function, "_receipt_clone_count", "clone_count")
    assert _source_names(support_clip.UKSupportClipReceipt.evidence, '"clone_count"')
    # The area home values count the sources at the knots' cap among geographic
    # clone 0's rows (microcosm#931), and their check reads its declared
    # maximum as it stands.
    assert _source_names(
        area_home_values.localise_area_home_values, "clone_zero_rows", "at_cap"
    )
    assert _source_names(area_home_values.clone_zero_rows, "_CLONE_INDEX_COLUMN")
    assert not _source_names(stage_health._area_home_values_gate, "clone_count")
    assert _source_names(
        frs_take_up.uk_take_up_signal_receipt, "clone_zero", "population_support"
    )
    assert _source_names(frs_take_up._measure_take_up_signal, "clone_zero")
    assert _source_names(
        frs_person_draws.derive_is_apprentice, "person_clone_zero", "flagged_support"
    )
    # The Child Benefit receipt counts each age's eligible child rows among
    # geographic clone 0's rows (microcosm#931), and its check reads its
    # declared floor and its band's n from them as they stand.
    assert _source_names(
        child_benefit_take_up.redraw_child_benefit_take_up,
        "child_clone_zero=geographic_clone_zero_rows(frame",
    )
    assert _source_names(
        child_benefit_take_up.solve_child_benefit_claim_rates,
        '"eligible_child_rows": int((of_age & counted).sum())',
    )
    assert not _source_names(stage_health._child_benefit_take_up_gate, "clone_count")
    # The SPI transport's uniformity receipt reads geographic clone 0's
    # recipients, and its groups the statistical floor of those rows beside
    # the disclosure floor of their source households (microcosm#931).
    assert _source_names(
        spi_rank_transport._uniformity_by_group,
        "counted",
        "SPI_TRANSPORT_GROUP_MINIMUM_ROWS",
        "UK_DISCLOSURE_FLOOR",
    )
