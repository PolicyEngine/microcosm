"""UK calibration diagnostics carry release evidence in the common format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.diagnostics import (
    UK_DIAGNOSTICS_SCHEMA_VERSION,
    UK_TARGET_GEOGRAPHY_LEVELS,
    uk_calibration_diagnostics_payload,
    uk_hierarchy_label_choices,
    uk_support_limited_misses,
    uk_weakest_areas_by_fit,
    uk_weakest_families,
    uk_weight_summary,
    uk_zero_weight_strata,
    write_uk_calibration_diagnostics,
)
from microcosm.calibrate import (
    CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION,
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
    diagnostics_payload,
    score_targets,
)
from microcosm.calibrate.geography_constants import UK_GEOGRAPHY_ID_TO_LABEL
from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    DiagnosticsWriteFailure,
    DiagnosticsWriteSuccess,
    HierarchyLabelChoices,
    catalog_calibration_diagnostics,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.microcosm_diagnostics.json_schema import (
    load_strict_json,
    published_json_schema_errors,
)

_SPI_COLUMN = "household_is_spi_synthetic"
_CG_COLUMN = "household_is_capital_gains_clone"


def _fixture_hierarchy(name: str, geography_level: str) -> CalibrationHierarchy:
    geography_ids = {
        "national": "K02000001",
        "region": "E12000001",
        "country": "E92000001",
        "local_authority": "E09000001",
        "constituency": "E14000001",
        "brma": "S33999999",
    }
    return CalibrationHierarchy(
        provider=HierarchyNode("fixture", "Fixture provider"),
        category=HierarchyCategory(
            "fixture.diagnostics",
            "Diagnostic fixtures",
            "fixture",
        ),
        geography=HierarchyGeography(
            geography_ids[geography_level],
            geography_level.replace("_", " ").title(),
            geography_level,
        ),
        dimensions=(),
        target=HierarchyNode(name, name.replace("_", " ").title()),
    )


def _local_target_row(
    name: str,
    *,
    family: str,
    area_type: str,
    area_code: str,
    relative_error: float,
    loss_contribution: float,
) -> dict[str, object]:
    return {
        "name": name,
        "relative_error": relative_error,
        "final_loss_contribution": loss_contribution,
        "registry": {"family": family},
        "metadata": {"area_type": area_type, "area_code": area_code},
    }


def test_local_weakest_rollups_pin_family_area_and_country_shapes() -> None:
    rows = [
        _local_target_row(
            "wales-a",
            family="income",
            area_type="constituency",
            area_code="W07000041",
            relative_error=0.30,
            loss_contribution=0.20,
        ),
        _local_target_row(
            "wales-b",
            family="households",
            area_type="constituency",
            area_code="W07000041",
            relative_error=0.05,
            loss_contribution=0.01,
        ),
        _local_target_row(
            "england-a",
            family="income",
            area_type="local_authority",
            area_code="E09000001",
            relative_error=0.15,
            loss_contribution=0.09,
        ),
    ]
    support = pd.DataFrame(
        {
            "geography_level": ["constituency", "local_authority"],
            "area_code": ["W07000041", "E09000001"],
            "nonzero_households": [80, 90],
            "nonzero_source_households": [70, 75],
            "effective_sample_size": [65.0, 72.0],
        }
    )

    families = uk_weakest_families(rows)
    assert [row["family"] for row in families] == ["income", "households"]
    assert set(families[0]) == {
        "family",
        "n_targets",
        "n_within_10pct",
        "pass_rate",
        "worst_target",
        "worst_abs_relative_error",
        "loss_contribution",
        "loss_share",
    }
    assert families[0]["n_targets"] == 2
    assert families[0]["loss_contribution"] == pytest.approx(0.29)

    areas = uk_weakest_areas_by_fit(rows, support)
    assert set(areas) == {"limit", "n_areas_scored", "bottom_by_fit", "countries"}
    # The reported list is keyed by role and carries its own limit, so the key
    # never asserts a count the list does not have.
    assert areas["limit"] == 15
    assert areas["n_areas_scored"] == 2
    assert [row["area_code"] for row in areas["bottom_by_fit"]] == [
        "W07000041",
        "E09000001",
    ]
    assert set(areas["bottom_by_fit"][0]) == {
        "geography_level",
        "area_code",
        "country",
        "n_targets",
        "n_within_10pct",
        "pass_rate",
        "worst_target",
        "worst_abs_relative_error",
        "loss_contribution",
        "nonzero_households",
        "nonzero_source_households",
        "effective_sample_size",
    }
    assert [row["country"] for row in areas["countries"]] == [
        "England",
        "Wales",
    ]
    wales = areas["countries"][1]
    assert wales["geography_level"] == "constituency"
    assert wales["n_targets"] == 2
    assert wales["pass_rate"] == pytest.approx(0.5)

    # A caller-supplied limit is reported alongside the list it produced,
    # rather than being contradicted by a fixed key name.
    trimmed = uk_weakest_areas_by_fit(rows, support, limit=1)
    assert trimmed["limit"] == 1
    assert trimmed["n_areas_scored"] == 2
    assert [row["area_code"] for row in trimmed["bottom_by_fit"]] == ["W07000041"]

    # Each row carries its area's rows, ESS and sources, and the floor flag
    # beside them when the table has it (María's ruling of 2026-10-26).
    wales_row = areas["bottom_by_fit"][0]
    assert (
        wales_row["nonzero_households"],
        wales_row["nonzero_source_households"],
        wales_row["effective_sample_size"],
    ) == (80, 70, 65.0)
    flagged = uk_weakest_areas_by_fit(
        rows, support.assign(support_below_floor=[True, False])
    )
    assert {
        row["area_code"]: row["support_below_floor"] for row in flagged["bottom_by_fit"]
    } == {"W07000041": True, "E09000001": False}
    assert all(
        type(row["support_below_floor"]) is bool for row in flagged["bottom_by_fit"]
    )
    # The support figures are required of the support table.
    with pytest.raises(ValueError, match="missing rollup column"):
        uk_weakest_areas_by_fit(rows, support.loc[:, ["geography_level", "area_code"]])


def test_local_fit_diagnostics_filter_national_rows_and_pin_support_shape() -> None:
    diagnostics = pd.DataFrame(
        {
            "area_type": ["constituency", "constituency", "la", None],
            "area_code": ["E14000001", "E14000002", "E06000001", None],
            "abs_relative_error": [0.30, 0.05, 0.40, 0.90],
        }
    )
    support = {
        "constituency": pd.DataFrame(
            {
                "area_code": ["E14000001", "E14000002"],
                "assigned_households": [2, 20],
                "effective_sample_size": [1.0, 20.0],
                "nonzero_source_households": [1, 20],
            }
        ),
        "la": pd.DataFrame(
            {
                "area_code": ["E06000001"],
                "assigned_households": [3],
                "effective_sample_size": [2.0],
                "nonzero_source_households": [2],
            }
        ),
    }

    result = uk_support_limited_misses(
        diagnostics,
        support,
        max_abs_relative_error=0.25,
    )
    assert set(result) == {"constituency", "la"}
    assert result["constituency"] == {
        "n_failing_cells": 1,
        "share_failing_cells_in_bottom_ess_decile": 1.0,
        "spearman_worst_abs_relative_error_vs_ess": -1.0,
        "worst_areas": [
            {
                "area_code": "E14000001",
                "worst_abs_relative_error": 0.3,
                "rows": 2,
                "ess": 1.0,
                "sources": 1,
            },
            {
                "area_code": "E14000002",
                "worst_abs_relative_error": 0.05,
                "rows": 20,
                "ess": 20.0,
                "sources": 20,
            },
        ],
    }
    assert result["la"]["n_failing_cells"] == 1
    assert result["la"]["spearman_worst_abs_relative_error_vs_ess"] is None

    # With the floor flag in the support table, each worst area carries it
    # beside its rows, ESS and sources (María's ruling of 2026-10-26).
    flagged = uk_support_limited_misses(
        diagnostics,
        {
            grain: frame.assign(support_below_floor=True)
            for grain, frame in support.items()
        },
        max_abs_relative_error=0.25,
    )
    assert [
        (area["area_code"], area["rows"], area["support_below_floor"])
        for area in flagged["constituency"]["worst_areas"]
    ] == [("E14000001", 2, True), ("E14000002", 20, True)]

    rows = [
        _local_target_row(
            "local",
            family="income",
            area_type="constituency",
            area_code="E14000001",
            relative_error=0.30,
            loss_contribution=0.20,
        ),
        {
            "name": "national",
            "relative_error": 0.90,
            "final_loss_contribution": 0.80,
            "registry": {"family": "national_income"},
            "metadata": {"geography_level": "national"},
        },
    ]
    long_support = support["constituency"].assign(
        geography_level="constituency",
        nonzero_households=lambda frame: frame["assigned_households"],
    )
    areas = uk_weakest_areas_by_fit(rows, long_support)
    assert areas["n_areas_scored"] == 1
    assert areas["bottom_by_fit"][0]["area_code"] == "E14000001"


def _consumption_columns(n: int) -> dict[str, np.ndarray]:
    """The lcfs_consumption columns the drift block reads, two houses with gas."""

    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
    )

    columns = {name: np.full(n, 10.0) for name in UK_LCFS_COICOP_DIVISION_COLUMNS}
    columns["electricity_consumption"] = np.full(n, 2.0)
    columns["gas_consumption"] = np.where(np.arange(n) < 2, 3.0, 0.0)
    columns["domestic_energy_consumption"] = (
        columns["electricity_consumption"] + columns["gas_consumption"]
    )
    columns["petrol_spending"] = np.where(np.arange(n) % 2 == 0, 4.0, 0.0)
    columns["diesel_spending"] = np.zeros(n)
    columns["has_fuel_consumption"] = np.arange(n) % 2 == 0
    return columns


def _diagnostics_case(*, with_skipped: bool = False, with_consumption: bool = False):
    """One row per geography level, each carried by one household."""

    levels_and_weights = (
        ("national", 11.0),
        ("region", 12.0),
        ("country", 9.0),
        ("local_authority", 15.0),
        ("constituency", 10.0),
    )
    n_households = 10
    household = pd.DataFrame(
        {
            "household_id": np.arange(n_households, dtype=np.int64),
            _SPI_COLUMN: [False] * 5 + [True] * 5,
            _CG_COLUMN: [False, False, True, True, False] * 2,
        }
    )
    if with_consumption:
        for name, values in _consumption_columns(n_households).items():
            household[name] = values
    specs = []
    geography = {}
    for row_index, (level, _) in enumerate(levels_and_weights):
        measure = f"{level}_measure"
        household[measure] = 0.0
        household.loc[row_index, measure] = 1.0
        name = f"{level}_target"
        specs.append(
            TargetSpec(
                name=name,
                entity="household",
                value=10.0,
                measure=measure,
                period=2023,
                source="Synthetic UK diagnostics fixture",
                family="fixture",
                metadata=(
                    {"observation_basis": "annual_flow"} if row_index == 0 else {}
                ),
                hierarchy=_fixture_hierarchy(name, level),
            )
        )
        geography[f"{name}@2023"] = "la" if level == "local_authority" else level
    if with_skipped:
        specs.append(
            TargetSpec(
                name="skipped_national_target",
                entity="household",
                value=10.0,
                measure="missing_measure",
                period=2023,
                source="Synthetic UK diagnostics fixture",
                family="fixture",
                hierarchy=_fixture_hierarchy(
                    "skipped_national_target",
                    "national",
                ),
            )
        )
        geography["skipped_national_target@2023"] = "national"

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n_households, dtype=np.int64),
                    "person_household_id": np.arange(
                        n_households,
                        dtype=np.int64,
                    ),
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.ones(n_households),
                WeightKind.DESIGN,
            )
        },
    )
    registry = TargetRegistry(specs, country="uk")
    final_weights = np.asarray(
        [weight for _, weight in levels_and_weights] + [0.0, 1.0, 1.0, 1.0, 1.0]
    )
    result = score_targets(
        frame,
        registry.to_target_set(),
        weights=final_weights,
    )
    diagnostic_frame = Frame(
        {
            "person": frame.table("person"),
            "household": household,
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                final_weights,
                WeightKind.DESIGN,
            )
        },
    )
    return result, diagnostic_frame, registry, geography


def test_uk_weight_summary_reports_kish_ess_and_concentration() -> None:
    summary = uk_weight_summary(np.asarray([0.0, 1.0, 1.0, 2.0, 6.0]))

    assert summary == {
        "n_records": 5,
        "positive_weight_records": 4,
        "zero_weight_records": 1,
        "total_weight": 10.0,
        "effective_sample_size": pytest.approx(100.0 / 42.0),
        "ess_fraction": pytest.approx(20.0 / 42.0),
        "median_positive_weight": 1.5,
        "max_weight": 6.0,
        "max_to_median_positive_weight": 4.0,
        "top_1pct_weight_share": 0.6,
    }


def test_uk_weight_summary_keeps_all_zero_vectors_reportable() -> None:
    assert uk_weight_summary([0.0, 0.0, 0.0]) == {
        "n_records": 3,
        "positive_weight_records": 0,
        "zero_weight_records": 3,
        "total_weight": 0.0,
        "effective_sample_size": 0.0,
        "ess_fraction": 0.0,
        "median_positive_weight": None,
        "max_weight": 0.0,
        "max_to_median_positive_weight": None,
        "top_1pct_weight_share": 0.0,
    }


@pytest.mark.parametrize(
    "weights",
    (
        [],
        [[1.0]],
        [1.0, -1.0],
        [1.0, np.nan],
        [1.0, np.inf],
    ),
)
def test_uk_weight_summary_rejects_invalid_vectors(weights) -> None:
    with pytest.raises(ValueError, match="UK diagnostic weights"):
        uk_weight_summary(weights)


def test_zero_weight_strata_exposes_june_like_spi_rows() -> None:
    frs_rows_per_cg_stratum = 167_540
    spi_rows_per_cg_stratum = 100_000
    spi = np.concatenate(
        (
            np.zeros(frs_rows_per_cg_stratum, dtype=bool),
            np.ones(spi_rows_per_cg_stratum, dtype=bool),
            np.zeros(frs_rows_per_cg_stratum, dtype=bool),
            np.ones(spi_rows_per_cg_stratum, dtype=bool),
        )
    )
    capital_gains = np.concatenate(
        (
            np.zeros(
                frs_rows_per_cg_stratum + spi_rows_per_cg_stratum,
                dtype=bool,
            ),
            np.ones(
                frs_rows_per_cg_stratum + spi_rows_per_cg_stratum,
                dtype=bool,
            ),
        )
    )
    household = pd.DataFrame({_SPI_COLUMN: spi, _CG_COLUMN: capital_gains})
    weights = np.where(spi, 0.0, 1.0)

    rows = uk_zero_weight_strata(household, weights)

    assert len(household) == 535_080
    assert sum(row["rows"] for row in rows) == 535_080
    assert sum(row["zero_weight_rows"] for row in rows) == 200_000
    assert rows == [
        {
            "stratum": {_SPI_COLUMN: False, _CG_COLUMN: False},
            "rows": 167_540,
            "positive_weight_rows": 167_540,
            "zero_weight_rows": 0,
            "weight_sum": 167_540.0,
        },
        {
            "stratum": {_SPI_COLUMN: True, _CG_COLUMN: False},
            "rows": 100_000,
            "positive_weight_rows": 0,
            "zero_weight_rows": 100_000,
            "weight_sum": 0.0,
        },
        {
            "stratum": {_SPI_COLUMN: False, _CG_COLUMN: True},
            "rows": 167_540,
            "positive_weight_rows": 167_540,
            "zero_weight_rows": 0,
            "weight_sum": 167_540.0,
        },
        {
            "stratum": {_SPI_COLUMN: True, _CG_COLUMN: True},
            "rows": 100_000,
            "positive_weight_rows": 0,
            "zero_weight_rows": 100_000,
            "weight_sum": 0.0,
        },
    ]


def test_zero_weight_strata_validates_alignment_and_columns() -> None:
    household = pd.DataFrame({_SPI_COLUMN: [False], _CG_COLUMN: [False]})

    with pytest.raises(ValueError, match="align with weights"):
        uk_zero_weight_strata(household, [1.0, 2.0])
    with pytest.raises(ValueError, match="missing stratum column"):
        uk_zero_weight_strata(
            household,
            [1.0],
            stratum_columns=("missing",),
        )
    with pytest.raises(ValueError, match="must be unique"):
        uk_zero_weight_strata(
            household,
            [1.0],
            stratum_columns=(_SPI_COLUMN, _SPI_COLUMN),
        )


def test_payload_preserves_common_schema_and_adds_versioned_uk_evidence() -> None:
    result, frame, registry, geography = _diagnostics_case()

    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        build={"release_id": "fixture"},
    )

    assert payload["schema_version"] == CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION
    assert "ess_fraction" not in payload
    assert all("geography_level" not in row for row in payload["targets"])
    assert payload["target_registry"] == {
        "country": "uk",
        "version": registry.version,
        "n_specs": len(registry),
    }
    assert payload["build"] == {"release_id": "fixture"}
    uk = payload["uk_diagnostics"]
    assert set(uk) == {
        "schema_version",
        "weights",
        "zero_weight_rows_by_stratum",
        "target_pass_rates_by_geography_level",
        "target_observation_basis",
    }
    assert uk["schema_version"] == UK_DIAGNOSTICS_SCHEMA_VERSION
    assert set(uk["weights"]) == {
        "n_records",
        "positive_weight_records",
        "zero_weight_records",
        "total_weight",
        "effective_sample_size",
        "ess_fraction",
        "median_positive_weight",
        "max_weight",
        "max_to_median_positive_weight",
        "top_1pct_weight_share",
    }
    assert uk["weights"]["effective_sample_size"] == payload["effective_sample_size"]
    assert uk["weights"]["top_1pct_weight_share"] == payload["top_1pct_weight_share"]
    assert uk["weights"]["zero_weight_records"] == 1
    assert uk["weights"]["ess_fraction"] == pytest.approx(
        payload["effective_sample_size"] / frame.n("household")
    )
    assert uk["target_observation_basis"] == {"national_target@2023": "annual_flow"}
    assert uk["target_pass_rates_by_geography_level"] == [
        {
            "geography_level": "national",
            "n_targets": 1,
            "n_scored": 1,
            "n_skipped": 0,
            "n_within_10pct": 1,
            "pass_rate": 1.0,
        },
        {
            "geography_level": "region",
            "n_targets": 1,
            "n_scored": 1,
            "n_skipped": 0,
            "n_within_10pct": 0,
            "pass_rate": 0.0,
        },
        {
            "geography_level": "country",
            "n_targets": 1,
            "n_scored": 1,
            "n_skipped": 0,
            "n_within_10pct": 1,
            "pass_rate": 1.0,
        },
        {
            "geography_level": "local_authority",
            "n_targets": 1,
            "n_scored": 1,
            "n_skipped": 0,
            "n_within_10pct": 0,
            "pass_rate": 0.0,
        },
        {
            "geography_level": "constituency",
            "n_targets": 1,
            "n_scored": 1,
            "n_skipped": 0,
            "n_within_10pct": 1,
            "pass_rate": 1.0,
        },
        # Every level is written; Scotland's BRMA rows bind no target here.
        {
            "geography_level": "brma",
            "n_targets": 0,
            "n_scored": 0,
            "n_skipped": 0,
            "n_within_10pct": 0,
            "pass_rate": None,
        },
    ]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_consumption_drift_compares_design_and_final_weights() -> None:
    """Stage-levelled spend and composition, design against final (#1113)."""

    from microcosm.build.uk_runtime.diagnostics import uk_consumption_drift

    household = pd.DataFrame(_consumption_columns(4))
    design = np.array([1.0, 1.0, 1.0, 1.0])
    final = np.array([2.0, 0.0, 1.0, 1.0])

    drift = uk_consumption_drift(household, design, final)

    assert drift["report_only"] is True
    assert drift["totals"]["gas_consumption"] == {
        "design": 6.0,
        "final": 6.0,
        "change": 0.0,
        "relative_change": 0.0,
    }
    assert drift["totals"]["petrol_spending"]["final"] == 12.0
    assert drift["totals"]["petrol_spending"]["relative_change"] == pytest.approx(0.5)
    assert drift["totals"]["consumption"]["design"] == 4 * 120.0
    assert drift["shares"]["gas_connected_households"]["design"] == 0.5
    assert drift["shares"]["gas_connected_households"]["final"] == 0.5
    assert drift["shares"]["fuel_buying_households"]["final"] == 0.75
    assert drift["shares"]["fuel_car_households_without_fuel"] == {
        "design": 0.0,
        "final": 0.0,
        "change": 0.0,
        "relative_change": None,
    }
    assert (
        uk_consumption_drift(household.drop(columns="gas_consumption"), design, final)
        is None
    )
    with pytest.raises(ValueError, match="align"):
        uk_consumption_drift(household, design[:3], final)
    assert "full_rate_vat_base" not in drift["totals"]


def test_consumption_drift_carries_the_heating_fuels_where_the_frame_has_them() -> None:
    """The liquid and solid fuels the stage draws (pe-uk#2572), read only where
    the frame carries them, so an older spine keeps its drift block."""

    from microcosm.build.uk_runtime.diagnostics import uk_consumption_drift

    household = pd.DataFrame(_consumption_columns(4))
    design = np.array([1.0, 1.0, 1.0, 1.0])
    final = np.array([2.0, 0.0, 1.0, 1.0])
    assert (
        "liquid_fuel_consumption"
        not in uk_consumption_drift(household, design, final)["totals"]
    )
    household["liquid_fuel_consumption"] = [100.0, 0.0, 50.0, 0.0]
    household["solid_fuel_consumption"] = [0.0, 30.0, 0.0, 0.0]
    totals = uk_consumption_drift(household, design, final)["totals"]
    assert totals["liquid_fuel_consumption"]["final"] == 250.0
    assert totals["solid_fuel_consumption"]["relative_change"] == -1.0


def test_drift_flags_mark_totals_past_two_percent_and_shares_past_half_a_point() -> (
    None
):
    """Warn-only flags at #1278's R1 thresholds; the payload keeps its shape."""

    from microcosm.build.uk_runtime.diagnostics import (
        UK_DRIFT_SHARE_FLAG,
        UK_DRIFT_TOTAL_FLAG,
        uk_consumption_drift_flags,
    )

    drift = {
        "totals": {
            "moves": {"relative_change": 0.021, "change": 21.0},
            "edge": {"relative_change": -0.02, "change": -20.0},
            "none": {"relative_change": None, "change": 0.0},
        },
        "shares": {
            "moves": {"change": -0.0051},
            "edge": {"change": 0.005},
        },
    }
    flags = uk_consumption_drift_flags(drift)
    assert flags["total_relative_change_above"] == UK_DRIFT_TOTAL_FLAG == 0.02
    assert flags["share_change_above"] == UK_DRIFT_SHARE_FLAG == 0.005
    assert flags["totals"] == ["moves"]
    assert flags["shares"] == ["moves"]


def test_drift_flags_reach_the_payload_as_warnings() -> None:
    """Vahid's review of #1383, item 11: the flags are wired, not a hand run.

    Each flagged total or share is one warn-only diagnostic warning, naming it
    and the threshold, never its figures.
    """

    from microcosm.build.uk_runtime.diagnostics import (
        UK_CONSUMPTION_DRIFT_WARNING_CODE,
        uk_consumption_drift_warnings,
    )

    drift = {
        "totals": {"solid_fuel_consumption": {"relative_change": -0.3, "change": -1.0}},
        "shares": {"gas_connected_households": {"change": 0.0001}},
    }
    (warning,) = uk_consumption_drift_warnings(drift)
    assert warning["code"] == UK_CONSUMPTION_DRIFT_WARNING_CODE
    assert warning["severity"] == "warning"
    assert "solid_fuel_consumption" in warning["message"]
    assert "2%" in warning["message"] and "-0.3" not in warning["message"]
    result, frame, registry, geography = _diagnostics_case(with_consumption=True)
    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )
    flagged = [
        warning
        for warning in payload["diagnostic_warnings"]
        if warning["code"] == UK_CONSUMPTION_DRIFT_WARNING_CODE
    ]
    assert flagged == uk_consumption_drift_warnings(
        payload["uk_diagnostics"]["consumption_drift"]
    )


def test_consumption_drift_carries_the_full_rate_vat_base() -> None:
    """Consumption times the ETB VAT stage's full-rate share (#1207)."""

    from microcosm.build.uk_runtime.diagnostics import uk_consumption_drift

    household = pd.DataFrame(_consumption_columns(4))
    household["full_rate_vat_expenditure_rate"] = [0.5, 0.5, 0.25, 0.0]
    design = np.array([1.0, 1.0, 1.0, 1.0])
    final = np.array([2.0, 0.0, 1.0, 1.0])

    drift = uk_consumption_drift(household, design, final)

    base = drift["totals"]["full_rate_vat_base"]
    assert base["design"] == pytest.approx(120.0 * (0.5 + 0.5 + 0.25))
    assert base["final"] == pytest.approx(120.0 * (2 * 0.5 + 0.25))


def test_consumption_drift_carries_the_stage_s_kwh() -> None:
    """The electricity and gas kWh at both weights, from the spend (#1207)."""

    from microcosm.build.uk_runtime.diagnostics import uk_consumption_drift
    from microcosm.build.uk_runtime.domestic_energy_targets import (
        household_energy_kwh,
        household_energy_prices,
    )

    household = pd.DataFrame(_consumption_columns(4))
    household["electricity_consumption"] = [900.0, 1100.0, 700.0, 1300.0]
    household["gas_consumption"] = [800.0, 0.0, 950.0, 0.0]
    household["region"] = ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"]
    design = np.array([1.0, 1.0, 1.0, 1.0])
    final = np.array([2.0, 0.0, 1.0, 1.0])

    drift = uk_consumption_drift(household, design, final)

    kwh = household_energy_kwh(
        household["electricity_consumption"],
        household["gas_consumption"],
        region=household["region"],
        prices=household_energy_prices(),
    )
    for name in ("electricity_kwh", "gas_kwh"):
        assert drift["totals"][name]["design"] == pytest.approx(design @ kwh[name])
        assert drift["totals"][name]["final"] == pytest.approx(final @ kwh[name])
    assert drift["totals"]["gas_kwh"]["design"] > 0
    without_region = uk_consumption_drift(
        household.drop(columns="region"), design, final
    )
    assert "electricity_kwh" not in without_region["totals"]


def test_consumption_drift_carries_the_excise_quantities() -> None:
    """Each excise quantity and base as stored, at both weights (#1359)."""

    from microcosm.build.uk_runtime.diagnostics import uk_consumption_drift

    household = pd.DataFrame(_consumption_columns(4))
    household["beer_litres"] = [100.0, 0.0, 50.0, 10.0]
    household["cigarettes_per_week"] = [20.0, 0.0, 0.0, 40.0]
    household["remote_gaming_net_losses"] = [0.0, 30.0, 0.0, 0.0]
    household["beer_draught_share"] = 0.4
    design = np.array([1.0, 1.0, 1.0, 1.0])
    final = np.array([2.0, 0.0, 1.0, 1.0])

    totals = uk_consumption_drift(household, design, final)["totals"]

    assert totals["beer_litres"]["design"] == pytest.approx(160.0)
    assert totals["beer_litres"]["final"] == pytest.approx(260.0)
    assert totals["cigarettes_per_week"]["final"] == pytest.approx(80.0)
    assert totals["remote_gaming_net_losses"]["relative_change"] == -1.0
    # A draught share is a ratio, not a quantity, and no other input is on
    # the frame.
    assert "beer_draught_share" not in totals
    assert "wine_litres" not in totals
    plain = uk_consumption_drift(pd.DataFrame(_consumption_columns(4)), design, final)[
        "totals"
    ]
    assert "beer_litres" not in plain


def test_payload_carries_the_consumption_drift_when_the_frame_has_it() -> None:
    result, frame, registry, geography = _diagnostics_case(with_consumption=True)

    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    drift = payload["uk_diagnostics"]["consumption_drift"]
    assert set(drift["totals"]) == {
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
        "petrol_spending",
        "diesel_spending",
        "consumption",
    }
    assert drift["totals"]["electricity_consumption"]["design"] == pytest.approx(
        2.0 * frame.n("household")
    )
    assert drift["totals"]["electricity_consumption"]["final"] == pytest.approx(
        2.0 * float(np.sum(frame.weights_for("household").values))
    )


def test_uk_payload_shared_layer_matches_shared_diagnostics_format() -> None:
    result, frame, registry, geography = _diagnostics_case()
    build = {"release_id": "fixture"}

    shared = diagnostics_payload(result, target_registry=registry, build=build)
    uk = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        build=build,
    )

    uk_shared_layer = {
        key: value for key, value in uk.items() if key != "uk_diagnostics"
    }
    assert uk_shared_layer == shared
    assert set(uk) == {*shared.keys(), "uk_diagnostics"}
    assert set(uk["uk_diagnostics"]) == {
        "schema_version",
        "weights",
        "zero_weight_rows_by_stratum",
        "target_pass_rates_by_geography_level",
        "target_observation_basis",
    }


def test_payload_requires_exact_explicit_geography_mapping() -> None:
    result, frame, registry, geography = _diagnostics_case()
    missing = dict(geography)
    missing.pop("national_target@2023")
    extra = {**geography, "not_a_target@2023": "national"}
    unknown = {**geography, "national_target@2023": "ward"}

    for invalid in (missing, extra):
        with pytest.raises(ValueError, match="must exactly cover"):
            uk_calibration_diagnostics_payload(
                result,
                frame,
                target_geography_levels=invalid,
                target_registry=registry,
            )
    with pytest.raises(ValueError, match="Unknown UK target geography level"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=unknown,
            target_registry=registry,
        )
    with pytest.raises(TypeError, match="must map declared target names"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=list(geography),
            target_registry=registry,
        )


def test_skipped_target_counts_as_a_geography_non_pass() -> None:
    result, frame, registry, geography = _diagnostics_case(with_skipped=True)

    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert len(payload["skipped"]) == 1
    assert payload["skipped"][0]["name"] == "skipped_national_target"
    assert "missing_measure" in payload["skipped"][0]["reason"]
    national = payload["uk_diagnostics"]["target_pass_rates_by_geography_level"][0]
    assert national == {
        "geography_level": "national",
        "n_targets": 2,
        "n_scored": 1,
        "n_skipped": 1,
        "n_within_10pct": 1,
        "pass_rate": 0.5,
    }
    rates = payload["uk_diagnostics"]["target_pass_rates_by_geography_level"]
    assert sum(row["n_targets"] for row in rates) == len(registry)
    assert sum(row["n_scored"] for row in rates) == len(result.diagnostics)
    assert sum(row["n_skipped"] for row in rates) == len(result.skipped)

    missing_skipped = dict(geography)
    missing_skipped.pop("skipped_national_target@2023")
    with pytest.raises(ValueError, match="must exactly cover"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=missing_skipped,
            target_registry=registry,
        )


def test_payload_requires_a_valid_matching_uk_registry() -> None:
    result, frame, registry, geography = _diagnostics_case()

    with pytest.raises(TypeError, match="require a TargetRegistry"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=object(),
        )
    with pytest.raises(ValueError, match="country == 'uk'"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=TargetRegistry(registry.specs, country="us"),
        )
    with pytest.raises(ValueError, match="non-empty registry"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=TargetRegistry((), country="uk"),
        )
    with pytest.raises(ValueError, match="does not contain compiled target row"):
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=TargetRegistry(registry.specs[:-1], country="uk"),
        )


def test_payload_requires_the_exact_shipped_weight_vector() -> None:
    result, frame, registry, geography = _diagnostics_case()
    bad_weights = frame.weights_for("household").values.copy()
    bad_weights[0] += 1.0
    mismatched = Frame(
        {
            "person": frame.table("person"),
            "household": frame.table("household"),
        },
        frame.schema,
        {
            "household": Weights(
                bad_weights,
                frame.weights_for("household").kind,
            )
        },
    )

    with pytest.raises(ValueError, match="must exactly match"):
        uk_calibration_diagnostics_payload(
            result,
            mismatched,
            target_geography_levels=geography,
            target_registry=registry,
        )


def test_writer_round_trips_strict_json(tmp_path: Path) -> None:
    result, frame, registry, geography = _diagnostics_case()
    path = tmp_path / "calibration_diagnostics.json"
    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    assert outcome.path == path
    assert outcome.schema_version == 9
    assembled = CalibrationDiagnosticsV8.model_validate(
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=registry,
        )
    )
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        catalog_calibration_diagnostics(
            assembled, label_choices=uk_hierarchy_label_choices()
        ).model_dump_json(exclude_defaults=True)
    )
    invalid = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        build={"not_json": object()},
    )
    assert isinstance(invalid, DiagnosticsWriteFailure)
    assert invalid.error_code == "validation_error"
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    ("with_skipped", "with_consumption"),
    [
        pytest.param(False, False, id="scored-targets-only"),
        pytest.param(True, True, id="skipped-target-and-consumption-drift"),
    ],
)
def test_written_uk_diagnostics_match_the_published_json_schema(
    tmp_path: Path,
    with_skipped: bool,
    with_consumption: bool,
) -> None:
    result, frame, registry, geography = _diagnostics_case(
        with_skipped=with_skipped,
        with_consumption=with_consumption,
    )
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        build={"release_id": "fixture"},
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    assert published_json_schema_errors(load_strict_json(path.read_bytes())) == []


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_writer_rejects_nested_non_finite_build_provenance(
    tmp_path: Path,
    value: float,
) -> None:
    result, frame, registry, geography = _diagnostics_case()
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text("stale")

    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        build={"nested": [{"value": value}]},
    )

    assert isinstance(outcome, DiagnosticsWriteFailure)
    assert outcome.error_code == "validation_error"
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_writer_reports_canonical_atomic_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, frame, registry, geography = _diagnostics_case()
    path = tmp_path / "calibration_diagnostics.json"
    prior = b'{"prior":true}\n'
    path.write_bytes(prior)

    def fail_replace(_temporary, _output):
        raise OSError("seeded replace failure")

    monkeypatch.setattr("microcosm.diagnostics.writer.os.replace", fail_replace)
    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert isinstance(outcome, DiagnosticsWriteFailure)
    assert outcome.error_code == "io_error"
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_writer_delegates_a_typed_payload_to_the_canonical_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, frame, registry, geography = _diagnostics_case()
    path = tmp_path / "calibration_diagnostics.json"
    captured: dict[str, object] = {}

    def canonical_writer(
        diagnostics: CalibrationDiagnosticsV8,
        destination: Path | str,
        *,
        label_choices: HierarchyLabelChoices | None = None,
    ) -> DiagnosticsWriteSuccess:
        captured["diagnostics"] = diagnostics
        captured["destination"] = Path(destination)
        captured["label_choices"] = label_choices
        return DiagnosticsWriteSuccess(
            status="available",
            path=Path(destination),
            schema_version=9,
            sha256="a" * 64,
        )

    monkeypatch.setattr(
        "microcosm.build.uk_runtime.diagnostics.write_typed_calibration_diagnostics",
        canonical_writer,
    )

    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    assert isinstance(captured["diagnostics"], CalibrationDiagnosticsV8)
    assert captured["destination"] == path
    assert captured["label_choices"] == uk_hierarchy_label_choices()


def test_uk_label_choices_name_every_area_microcosm_has_a_catalog_for() -> None:
    # UK publishers name one area several ways ("Hartlepool UA" from HMRC,
    # "Hartlepool" from ONS). Diagnostics schema 9 holds one label per area,
    # taken from Microcosm's own catalogs: UK_GEOGRAPHY_ID_TO_LABEL for the
    # UK, its nations and English regions, and the ONS April 2023 names in
    # local_authority_names.json for local authorities.
    choices = uk_hierarchy_label_choices()

    assert choices.geography_dimensions == frozenset({"geography"})
    assert dict(choices.dimension_values) == {}
    for code, label in UK_GEOGRAPHY_ID_TO_LABEL.items():
        assert choices.geographies[code] == label
    assert choices.geographies["E06000001"] == "Hartlepool"
    assert choices.geographies["E06000023"] == "Bristol, City of"
    assert choices.geographies["E08000013"] == "St. Helens"
    assert choices.geographies["W06000001"] == "Isle of Anglesey"
    assert len(choices.geographies) == len(UK_GEOGRAPHY_ID_TO_LABEL) + 361


def test_geography_level_vocabulary_includes_future_release_levels() -> None:
    assert UK_TARGET_GEOGRAPHY_LEVELS == (
        "national",
        "region",
        "country",
        "local_authority",
        "constituency",
        "brma",
    )


class TestSupportFamilyWeights:
    """Folding support-split families back to one weight (microcosm#1045)."""

    @staticmethod
    def _household(copies: bool) -> pd.DataFrame:
        from microcosm.build.uk_runtime.diagnostics import UK_SUPPORT_COPY_FLAG_COLUMN

        # Root 1 split into three (ids 1, 11, 21); root 2 whole; the clone
        # side (ids +100) repeats the pattern at half weight.
        ids = [1, 2, 11, 21, 101, 102, 111, 121]
        household = pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [10.0, 30.0, 10.0, 10.0, 5.0, 15.0, 5.0, 5.0],
                "source_household_id": [1, 2, 1, 1, 1, 2, 1, 1],
                "household_support_channel": ["frs"] * 8,
                "household_support_clone_index": [0] * 8,
                "household_is_capital_gains_clone": [False] * 4 + [True] * 4,
            }
        )
        if copies:
            household[UK_SUPPORT_COPY_FLAG_COLUMN] = [
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
            ]
        return household

    def test_families_sum_to_one_weight_per_root_on_each_clone_side(self) -> None:
        from microcosm.build.uk_runtime.diagnostics import uk_support_family_weights

        folded, receipt = uk_support_family_weights(self._household(copies=True))
        assert sorted(folded.tolist()) == [15.0, 15.0, 30.0, 30.0]
        assert receipt == {
            "basis": "support_family_fold",
            "key_columns": [
                "source_household_id",
                "household_support_channel",
                "household_support_clone_index",
                "household_is_capital_gains_clone",
            ],
            "geographic_clone_index_in_key": False,
            "families": 4,
            "support_copy_rows": 4,
            "rows_folded": 4,
        }

    def test_geographic_clones_stay_apart_on_an_expanded_frame(self) -> None:
        """Round 2 of the #1045 review: the dense line's K clones fold only
        their own copies, never each other."""
        from microcosm.build.uk_runtime.diagnostics import (
            uk_support_family_geographic_clone_column,
            uk_support_family_weights,
        )

        base = self._household(copies=True)
        k = 3
        clone_column = uk_support_family_geographic_clone_column()
        expanded = pd.concat(
            [
                base.assign(
                    household_id=base["household_id"] + 1_000 * clone,
                    household_weight=base["household_weight"] / k,
                    **{clone_column: clone},
                )
                for clone in range(k)
            ],
            ignore_index=True,
        )
        folded, receipt = uk_support_family_weights(expanded)
        copies = int(expanded["household_is_cgt_support_copy"].sum())
        assert receipt["geographic_clone_index_in_key"] is True
        assert receipt["key_columns"][-1] == clone_column
        assert receipt["families"] == len(expanded) - copies == 4 * k
        assert receipt["rows_folded"] == copies == 4 * k
        np.testing.assert_allclose(
            sorted(folded.tolist()),
            sorted([w / k for w in [15.0, 15.0, 30.0, 30.0] * k]),
        )

    def test_a_table_without_copies_folds_to_itself(self) -> None:
        from microcosm.build.uk_runtime.diagnostics import uk_support_family_weights

        household = self._household(copies=False)
        folded, receipt = uk_support_family_weights(household)
        assert folded.tolist() == household["household_weight"].tolist()
        assert receipt == {
            "basis": "rows",
            "families": 8,
            "support_copy_rows": 0,
            "rows_folded": 0,
        }

    def test_copies_without_the_lineage_key_refuse(self) -> None:
        from microcosm.build.uk_runtime.diagnostics import uk_support_family_weights

        household = self._household(copies=True).drop(columns=["source_household_id"])
        with pytest.raises(ValueError, match="source_household_id"):
            uk_support_family_weights(household)


def test_a_thin_row_keeps_its_model_figures() -> None:
    """María's ruling of 2026-10-26 dropped the withholding of microcosm#1262.

    One BRMA row rests on three source households, the other on eleven. Both
    keep their model figures and carry no withholding note or carrier
    verdict; both count in the pass rates and the family rollup, and the
    totals are the solve's own.
    """

    n = 14
    sources = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    household = pd.DataFrame(
        {
            "household_id": np.arange(n, dtype=np.int64),
            "source_household_id": np.asarray(sources, dtype=np.int64),
            _SPI_COLUMN: [False] * n,
            _CG_COLUMN: [False] * n,
            "thin_measure": [1.0] * 6 + [0.0] * 8,
            "wide_measure": [1.0] * n,
        }
    )
    specs = [
        TargetSpec(
            name=name,
            entity="household",
            value=10.0,
            measure=measure,
            period=2023,
            source="Synthetic UK diagnostics fixture",
            family="private_rent",
            hierarchy=_fixture_hierarchy(name, "brma"),
        )
        for name, measure in (
            ("thin_brma_target", "thin_measure"),
            ("wide_brma_target", "wide_measure"),
        )
    ]
    geography = {f"{spec.name}@2023": "brma" for spec in specs}
    weights = np.ones(n)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.arange(n, dtype=np.int64),
                "person_household_id": np.arange(n, dtype=np.int64),
            }
        ),
        "household": household,
    }
    frame = Frame(
        tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    registry = TargetRegistry(specs, country="uk")
    result = score_targets(frame, registry.to_target_set(), weights=weights)
    support = pd.DataFrame(
        columns=[
            "geography_level",
            "area_code",
            "nonzero_households",
            "nonzero_source_households",
            "effective_sample_size",
        ]
    )
    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        local_area_support=support,
    )
    rows = {row["target_name"]: row for row in payload["targets"]}
    thin, wide = rows["thin_brma_target"], rows["wide_brma_target"]
    assert thin["final_estimate"] == pytest.approx(6.0)
    assert thin["relative_error"] == pytest.approx(-0.4)
    assert wide["final_estimate"] == pytest.approx(14.0)
    for row in (thin, wide):
        metadata = row.get("metadata") or {}
        assert "model_figures_withheld" not in metadata
        assert "carrier_floor_met" not in metadata
    (brma,) = [
        row
        for row in payload["uk_diagnostics"]["target_pass_rates_by_geography_level"]
        if row["geography_level"] == "brma"
    ]
    assert (brma["n_targets"], brma["n_scored"], brma["n_within_10pct"]) == (2, 2, 0)
    (family,) = payload["uk_diagnostics"]["weakest_families"]
    assert family["n_targets"] == 2
    assert "model_figures_withheld" not in payload["uk_diagnostics"]
    shared = diagnostics_payload(result, target_registry=registry)
    for field in ("initial_loss", "final_loss", "loss_trajectory", "options"):
        assert payload[field] == shared[field]
