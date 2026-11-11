"""How each count-like constant of the UK spine meets the geographic clone count.

The clone-first spine gives every FRS household ``K`` geographic clones
(microcosm#931), each at ``1 / K`` of its weight. A threshold on rows then
sees ``K`` times the rows it saw at one clone, while one on distinct true
source households, on survey records or on weight does not. Each constant
below is classified by what it counts and by how the code meets ``K``, and
pinned to the code's value by import, so a change to one fails here until
it is classified again. The gates' count-like maxima are pinned the same
way: a new one in the packaged gates must join the table.
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
    cgt_support,
    company_cars,
    frs_person_draws,
    frs_take_up,
    regional_home_values,
    salary_sacrifice,
    spi_band_donors,
    spi_benefit_coherence,
    spi_employment_shell,
    spi_rank_transport,
    stage_health,
    support_clip,
    uc_capital_coherence,
)

#: What a constant counts.
COUNTS = frozenset(
    {"source count", "survey records", "rows", "weight", "hyperparameter"}
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

K_SENSITIVE_CONSTANTS: tuple[KSensitiveConstant, ...] = (
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
        where="spi_band_donors.SPI_INCOME_BAND_MINIMUM_DONORS",
        read=lambda: spi_band_donors.SPI_INCOME_BAND_MINIMUM_DONORS,
        value=120,
        counts="source count",
        treatment="per clone",
        reason=(
            "every band seats at least this many donors in each clone, and "
            "the band donor check requires K times it"
        ),
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
        treatment="kept",
        reason=(
            "a statistical floor of a uniformity diagnostic only the receipt "
            "reports; at K clones it reports more groups"
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
        name="regional home values sources at the cap",
        where=(
            "the regional home values gate's maximum_sources_at_cap "
            "(stage_health), against the stage's sources_at_cap "
            "(regional_home_values)"
        ),
        read=lambda: _declared_parameters(
            "regional_home_values", "maximum_sources_at_cap"
        ),
        value=frozenset(
            {("uk_stage_regional_home_values_ranking", "maximum_sources_at_cap")}
        ),
        counts="source count",
        treatment="clone 0",
        reason=(
            "each clone ranks on its own draw, so the sources at the cap over "
            "every clone grow with K"
        ),
        gate_parameters=frozenset(
            {("uk_stage_regional_home_values_ranking", "maximum_sources_at_cap")}
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
        read=lambda: frozenset(
            ("uk_zero_weight_strata", "maximum_zero_weight_rows")
            for gate in _gates()
            if gate["id"] == "uk_zero_weight_strata"
            for declaration in gate["parameters"]["declarations"]
            if "maximum_zero_weight_rows" in declaration
        ),
        value=frozenset({("uk_zero_weight_strata", "maximum_zero_weight_rows")}),
        counts="rows",
        treatment="later",
        reason=(
            "a terminal gate: the national and dense builds that run it refuse "
            "a clone-first spine until #931 group (c)"
        ),
        gate_parameters=frozenset(
            {("uk_zero_weight_strata", "maximum_zero_weight_rows")}
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


#: A gate parameter that caps a count: maxima of rows, sources, households,
#: records, donors, cells or mismatches.
_COUNT_MAXIMUM = re.compile(
    r"^max(imum)?_(.*_)?"
    r"(rows|sources|households|records|donors|cells|mismatches|count)(_.*)?$"
)


def _count_maxima(value: object) -> set[str]:
    if isinstance(value, dict):
        found = {key for key in value if _COUNT_MAXIMUM.match(key)}
        for item in value.values():
            found |= _count_maxima(item)
        return found
    if isinstance(value, list):
        return set().union(*(_count_maxima(item) for item in value))
    return set()


def test_every_count_maximum_in_the_gates_is_classified() -> None:
    """A count-like maximum a packaged gate declares is in the table, so a
    new one is classified before a spine of several clones reads it."""

    declared = {
        (gate["id"], key)
        for gate in _gates()
        for key in _count_maxima(gate.get("parameters") or {})
    }
    classified = set().union(
        *(constant.gate_parameters for constant in K_SENSITIVE_CONSTANTS)
    )
    assert declared == classified


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
    assert _source_names(
        regional_home_values.assign_regional_home_values,
        "clone_zero",
        "GEOGRAPHIC_COPY_ID_FIELD",
    )
    # The regional home values check reads its declared maximum as it stands.
    assert not _source_names(stage_health._regional_home_values_gate, "clone_count")
    assert _source_names(
        frs_take_up.uk_take_up_signal_receipt, "clone_zero", "population_support"
    )
    assert _source_names(frs_take_up._measure_take_up_signal, "clone_zero")
    assert _source_names(
        frs_person_draws.derive_is_apprentice, "person_clone_zero", "flagged_support"
    )
