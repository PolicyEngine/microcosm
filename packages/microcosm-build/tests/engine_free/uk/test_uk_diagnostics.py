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
    uk_support_limited_misses,
    uk_weakest_areas_by_fit,
    uk_weakest_families,
    uk_weight_summary,
    uk_zero_weight_strata,
    write_uk_calibration_diagnostics,
)
from microcosm.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
    diagnostics_payload,
    score_targets,
)
from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    DiagnosticsWriteFailure,
    DiagnosticsWriteSuccess,
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

    # The rows ship in a release, so they carry the area's floor flag, never
    # its rows, ESS or sources (microcosm#1301).
    flagged = uk_weakest_areas_by_fit(
        rows, support.assign(support_below_floor=[True, False])
    )
    assert {
        row["area_code"]: row["support_below_floor"] for row in flagged["bottom_by_fit"]
    } == {"W07000041": True, "E09000001": False}
    assert all(
        type(row["support_below_floor"]) is bool for row in flagged["bottom_by_fit"]
    )
    for row in flagged["bottom_by_fit"]:
        assert not {
            "nonzero_households",
            "nonzero_source_households",
            "effective_sample_size",
        } & set(row)
    # Only the codes are required of the support table.
    codes_only = uk_weakest_areas_by_fit(
        rows, support.loc[:, ["geography_level", "area_code"]]
    )
    assert codes_only["bottom_by_fit"] == areas["bottom_by_fit"]


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
            {"area_code": "E14000001", "worst_abs_relative_error": 0.3},
            {"area_code": "E14000002", "worst_abs_relative_error": 0.05},
        ],
    }
    assert result["la"]["n_failing_cells"] == 1
    assert result["la"]["spearman_worst_abs_relative_error_vs_ess"] is None

    # With the floor flag in the support table, each worst area carries it in
    # place of its rows, ESS and sources (microcosm#1301).
    flagged = uk_support_limited_misses(
        diagnostics,
        {
            grain: frame.assign(support_below_floor=True)
            for grain, frame in support.items()
        },
        max_abs_relative_error=0.25,
    )
    assert flagged["constituency"]["worst_areas"] == [
        {
            "area_code": "E14000001",
            "worst_abs_relative_error": 0.3,
            "support_below_floor": True,
        },
        {
            "area_code": "E14000002",
            "worst_abs_relative_error": 0.05,
            "support_below_floor": True,
        },
    ]

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
    """One row per geography level, each carried by one household.

    Every row rests on fewer than ten source households, so a test that is not
    about withholding passes an empty withheld set; the withheld path has its
    own tests (María's rule of 2026-10-21 withholds a thin row's figures).
    """

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
        withheld=frozenset(),
        build={"release_id": "fixture"},
    )

    assert payload["schema_version"] == CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
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
        withheld=frozenset(),
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
        withheld=frozenset(),
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
        withheld=frozenset(),
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
        withheld=frozenset(),
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
            withheld=frozenset(),
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
        withheld=frozenset(),
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    assert outcome.path == path
    assert json.loads(path.read_text(encoding="utf-8")) == (
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=registry,
            withheld=frozenset(),
        )
    )
    invalid = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
        withheld=frozenset(),
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
        withheld=frozenset(),
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
        withheld=frozenset(),
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
        withheld=frozenset(),
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
    ) -> DiagnosticsWriteSuccess:
        captured["diagnostics"] = diagnostics
        captured["destination"] = Path(destination)
        return DiagnosticsWriteSuccess(
            status="available",
            path=Path(destination),
            schema_version=8,
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
        withheld=frozenset(),
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    assert isinstance(captured["diagnostics"], CalibrationDiagnosticsV8)
    assert captured["destination"] == path


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


def test_a_thin_brma_row_keeps_its_place_but_not_its_model_figures() -> None:
    """María's ruling of 2026-10-17 (#1305 review): ten source households per BRMA row.

    One BRMA row rests on three source households, the other on eleven. The
    thin row stays in the payload with its model figures null and a note;
    it counts as scored but never as passing, and stays out of the family
    rollup, so nothing published can be differenced to recover it.
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
    assert thin["final_estimate"] is None and thin["relative_error"] is None
    assert thin["within_tolerance"] is None
    assert "model_figures_withheld" in thin["metadata"]
    assert wide["final_estimate"] == pytest.approx(14.0)
    (brma,) = [
        row
        for row in payload["uk_diagnostics"]["target_pass_rates_by_geography_level"]
        if row["geography_level"] == "brma"
    ]
    assert (brma["n_targets"], brma["n_scored"], brma["n_within_10pct"]) == (2, 2, 0)
    (family,) = payload["uk_diagnostics"]["weakest_families"]
    assert family["n_targets"] == 1 and family["worst_target"] == wide["name"]


def test_a_thin_national_row_keeps_its_place_but_not_its_model_figures() -> None:
    """María's rule of 2026-10-21: a national row is bound however thin it is.

    The BRMA rule now covers national rows. One national row rests on three source
    households and stays bound, its model figures null and noted; the other,
    on eleven, publishes. The withheld block restates the totals over the
    published row, and no row is told it met a BRMA floor.
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
            family="dwp_universal_credit",
            hierarchy=_fixture_hierarchy(name, "national"),
        )
        for name, measure in (
            ("thin_payment_band", "thin_measure"),
            ("wide_payment_band", "wide_measure"),
        )
    ]
    geography = {f"{spec.name}@2023": "national" for spec in specs}
    weights = np.ones(n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    registry = TargetRegistry(specs, country="uk")
    result = score_targets(frame, registry.to_target_set(), weights=weights)
    payload = uk_calibration_diagnostics_payload(
        result,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )
    rows = {row["target_name"]: row for row in payload["targets"]}
    thin, wide = rows["thin_payment_band"], rows["wide_payment_band"]
    assert thin["final_estimate"] is None and thin["relative_error"] is None
    assert "model_figures_withheld" in thin["metadata"]
    assert "carrier_floor_met" not in (thin.get("metadata") or {})
    assert wide["final_estimate"] == pytest.approx(14.0)
    (national,) = [
        row
        for row in payload["uk_diagnostics"]["target_pass_rates_by_geography_level"]
        if row["geography_level"] == "national"
    ]
    assert (national["n_targets"], national["n_scored"]) == (2, 2)
    block = payload["uk_diagnostics"]["model_figures_withheld"]
    assert block["targets"] == 1 and block["totals_basis"] == "published_rows"
    assert payload["loss_trajectory"] == []


def test_a_thin_row_is_withheld_at_every_level() -> None:
    """María's rule of 2026-10-21, and her ruling of 2026-10-23 on the local line.

    A row resting on fewer than ten source households keeps its place without
    its figures at every level; the area-support rule still decides which
    constituency and authority rows bind.
    """

    from types import SimpleNamespace

    from scipy import sparse

    from microcosm.build.uk_runtime.diagnostics import uk_withheld_rows

    n = 12
    matrix = sparse.csr_array(
        (np.ones(3), (np.zeros(3, dtype=int), np.arange(3))), shape=(1, n)
    )
    problem = SimpleNamespace(matrix=matrix, names=["thin_row@E14000001@2025"])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(n, dtype=np.int64),
                    "source_household_id": np.arange(n, dtype=np.int64),
                    "region": ["LONDON"] * n,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    rows = [{"name": "thin_row@E14000001@2025"}]
    for level, withheld in (
        ("national", True),
        ("region", True),
        ("brma", True),
        ("statistical_scope", True),
        ("constituency", True),
        ("local_authority", True),
        ("la", True),
    ):
        found = uk_withheld_rows(
            problem, np.ones(n), frame, rows, {"thin_row@E14000001@2025": level}
        )
        assert (found == {"thin_row@E14000001@2025"}) is withheld, level


def _withheld_sets(
    rows: dict[str, list[int]],
    n: int,
    values: dict[str, float | list[float]] | None = None,
    sources: list[float] | None = None,
    levels: dict[str, str] | None = None,
    regions: list[str] | None = None,
    targets: dict[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    """``uk_withheld_row_sets`` over rows given by the households that carry them.

    Every household is its own source household (or ``sources``) and carries
    weight one. A row's entries are one, or its ``values`` (one for all its
    households, or one each). ``regions`` gives each household's region, and
    ``targets`` a row's contract target.
    """

    from types import SimpleNamespace

    from scipy import sparse

    from microcosm.build.uk_runtime.diagnostics import uk_withheld_row_sets

    names = list(rows)
    dense = np.zeros((len(names), n))
    for position, name in enumerate(names):
        dense[position, rows[name]] = (values or {}).get(name, 1.0)
    problem = SimpleNamespace(matrix=sparse.csr_array(dense), names=names)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(n, dtype=np.int64),
                    "source_household_id": (
                        np.arange(n, dtype=np.int64)
                        if sources is None
                        else np.asarray(sources)
                    ),
                    **({} if regions is None else {"region": regions}),
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    target_rows = [
        {"name": name}
        if name not in (targets or {})
        else {"name": name, "metadata": {"contract_target_id": targets[name]}}
        for name in names
    ]
    return uk_withheld_row_sets(problem, np.ones(n), frame, target_rows, levels)


def test_a_band_its_total_gives_back_withholds_the_next_smallest_band() -> None:
    """Vahid's second review of #1333, item 10, and María's hard rule.

    The total less the published bands gives the thin band back, so the
    next-smallest band is withheld beside it, not the total. The two together
    rest on fifteen source households, which the published rows may give back.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": [0, 1, 2],
            "band_b": list(range(3, 15)),
            "band_c": list(range(15, 30)),
        },
        30,
    )
    assert (thin, beside) == ({"band_a"}, {"band_b"})


def test_a_thin_top_band_no_row_carries_withholds_the_next_smallest_band() -> None:
    """Vahid's second review of #1344, item 17: a thin top band, no thin bound band.

    The top band (above GBP 2,500 a month) is no row, so no row is thin and
    no withheld row is given back, but the total less the bound bands gives
    the top band's three households back. The smaller bound band is withheld,
    so what the published rows give back rests on fifteen households.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(0, 15)),
            "band_b": list(range(15, 27)),
        },
        30,
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_a_top_band_resting_on_no_household_withholds_nothing() -> None:
    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(0, 15)),
            "band_b": list(range(15, 30)),
        },
        30,
    )
    assert (thin, beside) == (set(), set())


def test_a_household_in_two_bands_still_leaves_the_bands_one_partition() -> None:
    """Two benefit units of one household in two bands: half its total each.

    The bands still partition the total, so its thin rest (the top band's
    three households) is found and the smaller band withheld.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(0, 15)),
            "band_b": [0, *range(15, 27)],
        },
        30,
        values={"total": [2.0, *[1.0] * 29]},
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_bands_beside_regions_of_the_same_total_are_still_checked() -> None:
    """The regions partition the total exactly; the bands leave a thin top band.

    Each band crosses both regions, so a band and a region never stand in one
    partition, and the bands' rest is measured on its own.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "region_1": list(range(0, 15)),
            "region_2": list(range(15, 30)),
            "band_a": [*range(0, 10), *range(15, 20)],
            "band_b": [*range(10, 15), *range(20, 27)],
        },
        30,
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_a_constant_multiple_of_a_part_is_withheld_with_it() -> None:
    """A count of people beside the count of their households is that part again.

    The people row is four times the units row on the same households, so
    once the units row is withheld, the total less a quarter of the people row
    would give the three households back. Both are withheld (the adversarial
    review of #1344's item 17 fix, 22 October).
    """

    thin, beside = _withheld_sets(
        {"total": list(range(30)), "units": list(range(27)), "people": list(range(27))},
        30,
        values={"people": 4.0},
    )
    assert (thin, beside) == (set(), {"people", "units"})


def test_bands_in_another_unit_than_their_total_are_scaled() -> None:
    """Monthly bands under a yearly total: a twelfth of the total on each household."""

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(15)),
            "band_b": list(range(15, 27)),
        },
        30,
        values={"total": 1200.0, "band_a": 100.0, "band_b": 100.0},
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_a_band_of_two_unit_households_only_still_partitions_its_total() -> None:
    """Every household of band_b holds a second unit in band_a: half the total each."""

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(27)),
            "band_b": list(range(15, 27)),
        },
        30,
        values={"total": [1.0] * 15 + [2.0] * 12 + [1.0] * 3},
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_income_components_less_their_total_band_are_checked() -> None:
    """A total-income band less its published components gives back other income.

    The components hold a different share of the total on each household, and
    a count of the band's payers sits inside it too; the count is no member.
    """

    employment = [100.5 + h for h in range(40)]
    pension = [20.25 + h / 2 for h in range(10, 40)]
    total = [
        employment[h] + (pension[h - 10] if h >= 10 else 0.0) + (3.25 if h < 5 else 0.0)
        for h in range(40)
    ]
    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "employment": list(range(40)),
            "pension": list(range(10, 40)),
            "payers": list(range(40)),
        },
        40,
        values={"total": total, "employment": employment, "pension": pension},
    )
    assert (thin, beside) == (set(), {"pension"})


def test_two_overlapping_parts_less_their_total_give_back_the_overlap() -> None:
    """Claimants of either benefit less the claimants of each: five on both."""

    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "contrib": list(range(25)),
            "income": list(range(20, 40)),
        },
        40,
    )
    assert (thin, beside) == (set(), {"income"})


def test_a_set_smaller_than_the_largest_group_is_checked() -> None:
    """Bands a and b and a third part fit pairwise but pass one together.

    The total less bands a and b alone gives back the nine-household top band
    and household 0, which the larger set covers.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "band_a": list(range(15)),
            "band_b": [0, *range(15, 31)],
            "c_other": [0, *range(31, 40)],
        },
        40,
        values={"total": [2.0] + [1.0] * 39},
    )
    assert (thin, beside) == (set(), {"band_a"})


def test_bands_crossing_larger_regions_are_still_grouped() -> None:
    """Each band misses a region larger than it, which a greedy group takes first."""

    cells = iter(range(39))
    b0r0, b0r1, b1r1, b1r2, b2r0, b2r2 = (
        [next(cells) for _ in range(6)] for _ in range(6)
    )
    top = [next(cells) for _ in range(3)]
    thin, beside = _withheld_sets(
        {
            "total": list(range(39)),
            "band_0": b0r0 + b0r1,
            "band_1": b1r1 + b1r2,
            "band_2": b2r0 + b2r2,
            "region_0": b0r0 + b2r0 + top[:1],
            "region_1": b0r1 + b1r1 + top[1:2],
            "region_2": b1r2 + b2r2 + top[2:],
        },
        39,
    )
    assert (thin, beside) == (set(), {"band_0"})


def test_a_total_past_the_search_cap_is_grouped_greedily(monkeypatch) -> None:
    """With no work allowed, every total is grouped greedily and named in a warning."""

    from microcosm.build.uk_runtime import diagnostics

    monkeypatch.setattr(diagnostics, "_RESIDUAL_SEARCH_WORK", 0)
    with pytest.warns(UserWarning, match="passed its cap"):
        thin, beside = _withheld_sets(
            {
                "total": list(range(120)),
                "region_1": list(range(41)),
                "region_2": list(range(41, 82)),
                "region_3": list(range(82, 120)),
                "band_a": [*range(3, 22), *range(41, 53)],
                "band_b": [*range(22, 41), *range(82, 94)],
                "band_c": list(range(53, 82)),
                "band_d": list(range(94, 120)),
            },
            120,
        )
    assert (thin, beside) == (set(), {"band_d"})


def test_households_whose_shares_differ_slightly_stay_in_different_atoms() -> None:
    """A large share elsewhere must not round band B's last five households away.

    Band B holds 99.999 of the total's 100 on five households, so the total
    less bands A and B rests on those five; a part with a share of a million
    on another household beside them changes nothing.
    """

    total = [1.0] + [100.0] * 39
    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "band_a": list(range(20)),
            "band_b": list(range(20, 40)),
            "big": list(range(10)),
        },
        40,
        values={
            "total": total,
            "band_a": total[:20],
            "band_b": [100.0] * 15 + [99.999] * 5,
            "big": [1e6] + [100.0] * 9,
        },
    )
    assert (thin, beside) == (set(), {"band_a"})


def test_a_component_in_whole_pounds_is_still_searched() -> None:
    """An amount component of whole pounds is no count: three components less
    their total give back other income on five households."""

    whole = [100.0 + h for h in range(40)]
    first = [10.25 + h for h in range(20)]
    second = [20.5 + h for h in range(20)]
    other = {3: 7.31, 8: 7.31, 25: 7.31, 30: 7.31, 33: 7.31}
    total = [
        whole[h] + (first[h] if h < 20 else second[h - 20]) + other.get(h, 0.0)
        for h in range(40)
    ]
    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "whole": list(range(40)),
            "first": list(range(20)),
            "second": list(range(20, 40)),
        },
        40,
        values={"total": total, "whole": whole, "first": first, "second": second},
    )
    assert (thin, beside) == (set(), {"first"})


def test_a_part_of_constant_negative_share_is_scaled() -> None:
    """The total plus twice a part of share minus a half gives back three households."""

    thin, beside = _withheld_sets(
        {"total": list(range(30)), "loss": list(range(27))},
        30,
        values={"total": 100.0, "loss": -50.0},
    )
    assert (thin, beside) == (set(), {"loss"})


def test_three_components_with_no_thin_pair_are_searched_together() -> None:
    """Every component sits on every household, so no pair less the total is
    thin; all three less it give back other income on five households."""

    employment = [100.5 + h for h in range(40)]
    pension = [20.25 + h / 4 for h in range(40)]
    savings = [3.5 + h / 8 for h in range(40)]
    total = [
        employment[h] + pension[h] + savings[h] + (2.75 if h < 5 else 0.0)
        for h in range(40)
    ]
    thin, beside = _withheld_sets(
        {
            "total": list(range(40)),
            "employment": list(range(40)),
            "pension": list(range(40)),
            "savings": list(range(40)),
        },
        40,
        values={
            "total": total,
            "employment": employment,
            "pension": pension,
            "savings": savings,
        },
    )
    assert (thin, beside) == (set(), {"employment"})


def test_a_part_above_its_total_on_one_household_is_checked_alone() -> None:
    """Twice the total's entry on household 0, so no member; the total less it
    still rests on four households."""

    thin, beside = _withheld_sets(
        {"total": list(range(30)), "part": list(range(27))},
        30,
        values={"part": [2.0] + [1.0] * 26},
    )
    assert (thin, beside) == (set(), {"part"})


def test_a_missing_source_id_counts_as_a_source_of_its_own() -> None:
    sources = [float(h) for h in range(29)] + [float("nan")]
    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": list(range(15)),
            "band_b": list(range(15, 27)),
        },
        30,
        sources=sources,
    )
    assert (thin, beside) == (set(), {"band_b"})


def test_cells_of_ten_households_clear_a_total_without_a_search(monkeypatch) -> None:
    """Twelve regions, each split in two cells of ten: every atom rests on ten
    households, so no combination at any weights can be thin."""

    from microcosm.build.uk_runtime import diagnostics

    def refuse(*_args, **_kwargs):
        raise AssertionError("the search ran")

    monkeypatch.setattr(diagnostics._ResidualCheck, "_search", refuse)
    rows = {"total": list(range(240))}
    for g in range(12):
        rows[f"region_{g:02d}"] = list(range(20 * g, 20 * g + 20))
        rows[f"region_{g:02d}_a"] = list(range(20 * g, 20 * g + 10))
        rows[f"region_{g:02d}_b"] = list(range(20 * g + 10, 20 * g + 20))
    assert _withheld_sets(rows, 240) == (set(), set())


def test_a_small_atom_in_the_span_of_large_ones_clears_its_total(monkeypatch) -> None:
    """Household 20 holds two units, one in each band: its column is the mean of
    the two bands' columns, so nothing can isolate it, and no search runs."""

    from microcosm.build.uk_runtime import diagnostics

    def refuse(*_args, **_kwargs):
        raise AssertionError("the search ran")

    monkeypatch.setattr(diagnostics._ResidualCheck, "_search", refuse)
    thin, beside = _withheld_sets(
        {
            "total": list(range(21)),
            "band_a": [*range(10), 20],
            "band_b": [*range(10, 20), 20],
        },
        21,
        values={"total": [1.0] * 20 + [2.0]},
    )
    assert (thin, beside) == (set(), set())


def test_an_outlier_household_does_not_hide_a_gap_from_the_span_test() -> None:
    """Employment less a huge offsetting loss on one household of total one.

    The outlier squashes every other share once a row is scaled to its
    largest; two households where the parts fall short of the total by a
    hundredth still rest on two source households, and the total less both
    parts gives them back (the third adversarial review of item 17).
    """

    total, x, y = {}, {}, {}
    for h in range(12):
        total[h] = x[h] = 30000.0 + 101.0 * h
    for h in range(12, 24):
        total[h] = y[h] = 30000.0 + 97.0 * h
    for h in range(24, 44):
        total[h] = 25000.0 + 89.0 * h
        x[h], y[h] = 0.4 * total[h], 0.6 * total[h]
    total[44], x[44], y[44] = 1.0, 1e6, 1.0 - 1e6
    for h in (45, 46):
        total[h] = 40000.0 + h
        x[h], y[h] = 0.4 * total[h], 0.59 * total[h]
    thin, beside = _withheld_sets(
        {"total": sorted(total), "x": sorted(x), "y": sorted(y)},
        47,
        values={
            "total": [total[h] for h in sorted(total)],
            "x": [x[h] for h in sorted(x)],
            "y": [y[h] for h in sorted(y)],
        },
    )
    assert (thin, beside) == (set(), {"x"})


def test_a_large_atom_short_of_one_by_less_than_the_rounding_is_covered() -> None:
    """Twelve households hold 0.6 and 0.4 less five ten-millionths: the search
    counts that as covered, so the span test must not count it as a direction
    that hides the two-household atom."""

    delta = 5e-7
    shares = (
        [(0.6, 0.4 - delta)] * 12
        + [(0.3, 0.7)] * 12
        + [(1.0, 0.0)] * 12
        + [(0.2, 0.3)] * 2
    )
    first = [h for h, (a, _) in enumerate(shares) if a]
    second = [h for h, (_, b) in enumerate(shares) if b]
    thin, beside = _withheld_sets(
        {"total": list(range(38)), "p1": first, "p2": second},
        38,
        values={
            "total": 1000.0,
            "p1": [1000 * shares[h][0] for h in first],
            "p2": [1000 * shares[h][1] for h in second],
        },
    )
    assert (thin, beside) == (set(), {"p2"})


def test_a_share_exactly_one_rounding_short_of_one_counts_as_covered_everywhere() -> (
    None
):
    """A part holding one less a millionth on ten households and a half on one:
    the total less it rests on three, whether or not a copy of the total sits
    beside it."""

    almost = 1 - 1e-6
    for rows in (
        {"total": list(range(13)), "copy": list(range(13)), "part": list(range(11))},
        {"total": list(range(13)), "part": list(range(11))},
    ):
        thin, beside = _withheld_sets(rows, 13, values={"part": [almost] * 10 + [0.5]})
        assert (thin, beside) == (set(), {"part"})


def test_an_outlier_on_the_gap_household_itself_is_measured_as_stored() -> None:
    """Two households where employment is a thousand times the total and a
    matching loss falls short of it by a ten-thousandth: as unit columns the
    gap looks tiny, but the total less both parts rests on those two."""

    total, x, y = {}, {}, {}
    for h in range(12):
        total[h] = x[h] = 30000.0 + 101.0 * h
    for h in range(12, 24):
        total[h] = y[h] = 30000.0 + 97.0 * h
    for h in range(24, 44):
        total[h] = 25000.0 + 89.0 * h
        x[h], y[h] = 0.4 * total[h], 0.6 * total[h]
    for h in (44, 45):
        total[h] = 40000.0 + h
        x[h], y[h] = 1000.0 * total[h], (1 - 1000.0 - 1e-4) * total[h]
    thin, beside = _withheld_sets(
        {"total": sorted(total), "x": sorted(x), "y": sorted(y)},
        46,
        values={
            "total": [total[h] for h in sorted(total)],
            "x": [x[h] for h in sorted(x)],
            "y": [y[h] for h in sorted(y)],
        },
    )
    assert (thin, beside) == (set(), {"x"})


def test_a_small_shortfall_beside_many_bands_is_not_cleared() -> None:
    """Three hundred bands of ten; two more households in band 1 hold one less
    2.4 millionths of the total, above the rounding share, so the total less
    every band rests on those two."""

    rows = {"total": list(range(3002))}
    for k in range(300):
        rows[f"band_{k:03d}"] = list(range(10 * k, 10 * k + 10)) + (
            [3000, 3001] if k == 1 else []
        )
    thin, beside = _withheld_sets(
        rows, 3002, values={"band_001": [1.0] * 10 + [1 - 2.4e-6] * 2}
    )
    assert (thin, beside) == (set(), {"band_000"})


def test_a_row_given_back_two_ways_loses_a_giver_from_each() -> None:
    """Two relations give the thin band back, so one giver of each is withheld.

    While both stand, every giver has a stand-in and the smallest goes first
    (the other total's rest); then the first total's bands are the only way
    back, and the next-smallest of them goes.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a": [0, 1, 2],
            "band_b": list(range(3, 15)),
            "band_c": list(range(15, 30)),
            "other_total": [0, 1, 2, *range(30, 40)],
            "other_rest": list(range(30, 40)),
        },
        40,
    )
    assert (thin, beside) == ({"band_a"}, {"other_rest", "band_b"})


def test_proportional_thin_rows_are_no_recovery() -> None:
    """A thin count and an amount on the same households differ by nothing.

    The amount is twice the count everywhere, so their difference is zero and
    no published row gives anything back by it; the total's relation alone
    asks for one more band.
    """

    thin, beside = _withheld_sets(
        {
            "total": list(range(30)),
            "band_a_count": [0, 1, 2],
            "band_a_amount": [0, 1, 2],
            "band_b": list(range(3, 15)),
            "band_c": list(range(15, 30)),
        },
        30,
        values={"band_a_amount": 2.0},
    )
    assert (thin, beside) == ({"band_a_count", "band_a_amount"}, {"band_b"})


def test_a_thin_row_no_published_row_gives_back_withholds_nothing_more() -> None:
    thin, beside = _withheld_sets(
        {"band_a": [0, 1, 2], "band_b": list(range(3, 15))}, 15
    )
    assert (thin, beside) == ({"band_a"}, set())


def test_a_row_withheld_beside_a_thin_one_says_so_and_meets_the_schema(
    tmp_path: Path,
) -> None:
    """The written diagnostics tell a row withheld beside a thin one from it.

    (A row can also be withheld where no row is thin, when a total less its
    parts would give back a cell that is no row; it carries the same flag.)

    BRMA rows: the thin band fails the ten-household floor; the band withheld
    beside it meets the floor and is flagged ``complementary_withholding``, as
    the release contract reads it. The payload, with its null figures, its
    notes and its restated totals, is valid against the published schema.
    """

    n = 30
    members = {
        "total": range(30),
        "band_a": range(3),
        "band_b": range(3, 15),
        "band_c": range(15, 30),
    }
    household = pd.DataFrame(
        {
            "household_id": np.arange(n, dtype=np.int64),
            "source_household_id": np.arange(n, dtype=np.int64),
            _SPI_COLUMN: [False] * n,
            _CG_COLUMN: [False] * n,
        }
    )
    for name, carriers in members.items():
        household[f"{name}_measure"] = 0.0
        household.loc[list(carriers), f"{name}_measure"] = 1.0
    specs = [
        TargetSpec(
            name=name,
            entity="household",
            value=float(len(carriers)),
            measure=f"{name}_measure",
            period=2023,
            source="Synthetic UK diagnostics fixture",
            family="fixture",
            hierarchy=_fixture_hierarchy(name, "brma"),
        )
        for name, carriers in members.items()
    ]
    geography = {f"{spec.name}@2023": "brma" for spec in specs}
    weights = np.ones(n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    registry = TargetRegistry(specs, country="uk")
    result = score_targets(frame, registry.to_target_set(), weights=weights)
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_uk_calibration_diagnostics(
        result,
        path,
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert isinstance(outcome, DiagnosticsWriteSuccess)
    document = load_strict_json(path.read_bytes())
    assert published_json_schema_errors(document) == []
    rows = {row["target_name"]: row for row in document["targets"]}
    thin, beside = rows["band_a"]["metadata"], rows["band_b"]["metadata"]
    assert thin["carrier_floor_met"] is False
    assert "complementary_withholding" not in thin
    assert beside["carrier_floor_met"] is True
    assert beside["complementary_withholding"] is True
    assert beside["model_figures_withheld"].startswith(
        "model figures withheld: with the published rows"
    )
    assert "a row or a cell" in beside["model_figures_withheld"]
    assert rows["band_b"]["final_estimate"] is None
    assert rows["total"]["final_estimate"] == pytest.approx(30.0)
    assert document["uk_diagnostics"]["model_figures_withheld"]["targets"] == 2


def _thin_and_wide_brma_payload(
    thin_values=(1.0,) * 6 + (0.0,) * 8, extra_options=None, thin_target=10.0
):
    """The thin/wide BRMA fixture's diagnostics payload and its result.

    ``extra_options`` joins the result's options, as a solve records them.
    """

    n = 14
    sources = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    household = pd.DataFrame(
        {
            "household_id": np.arange(n, dtype=np.int64),
            "source_household_id": np.asarray(sources, dtype=np.int64),
            _SPI_COLUMN: [False] * n,
            _CG_COLUMN: [False] * n,
            "thin_measure": list(thin_values),
            "wide_measure": [1.0] * n,
            "other_measure": [0.5] * n,
        }
    )
    specs = [
        TargetSpec(
            name=name,
            entity="household",
            value=value,
            measure=measure,
            period=2023,
            source="Synthetic UK diagnostics fixture",
            family="private_rent",
            hierarchy=_fixture_hierarchy(name, "brma"),
        )
        for name, measure, value in (
            ("thin_brma_target", "thin_measure", thin_target),
            ("wide_brma_target", "wide_measure", 10.0),
            ("other_brma_target", "other_measure", 7.5),
        )
    ]
    geography = {f"{spec.name}@2023": "brma" for spec in specs}
    weights = np.ones(n)
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": household,
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    registry = TargetRegistry(specs, country="uk")
    result = score_targets(frame, registry.to_target_set(), weights=weights)
    if extra_options:
        from dataclasses import replace

        result = replace(result, options={**result.options, **extra_options})
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
    return payload, result


def test_no_published_total_gives_a_withheld_row_back() -> None:
    """María's ruling of 2026-10-18: the totals cover the published rows only.

    Try to recover the withheld row from what the payload publishes: the
    final and initial losses less the published rows' contributions leave
    nothing, the fraction within 10% is the published rows' own, and the
    trajectory, which no published row can be set against, is empty.
    """

    import math

    payload, result = _thin_and_wide_brma_payload()
    rows = {row["target_name"]: row for row in payload["targets"]}
    thin = rows.pop("thin_brma_target")
    assert thin["final_estimate"] is None
    assert thin.get("final_loss_contribution") is None
    published = list(rows.values())
    assert payload["final_loss"] == math.fsum(
        row["final_loss_contribution"] for row in published
    )
    # The initial loss is the published rows' too, from their own fields.
    assert payload["initial_loss"] == pytest.approx(
        math.fsum(
            row["target_loss_weight_share"]
            * min(
                abs(row["initial_estimate"] - row["target"]) / row["target_loss_scale"],
                payload["target_loss_basis"]["cap"],
            )
            for row in published
        ),
        rel=1e-12,
    )
    # Either total over every row would have left the thin row's share.
    assert payload["final_loss"] < result.final_loss
    assert payload["fraction_within_10pct"] == pytest.approx(
        sum(abs(row["relative_error"]) <= 0.10 for row in published) / len(published)
    )
    assert payload["loss_trajectory"] == []
    assert payload["past_cap_census"]["n_targets"] == len(published)
    block = payload["uk_diagnostics"]["model_figures_withheld"]
    assert block["targets"] == 1 and block["totals_basis"] == "published_rows"
    # A build with nothing withheld keeps the solve's own totals.
    payload, result = _thin_and_wide_brma_payload(thin_values=(1.0,) * 14)
    assert payload["final_loss"] == pytest.approx(result.final_loss, rel=1e-12)
    assert payload["loss_trajectory"] == pytest.approx(list(result.loss_trajectory))
    assert "model_figures_withheld" not in payload["uk_diagnostics"]


def _numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def test_a_withheld_row_inside_ten_percent_counts_nowhere() -> None:
    """The third review of #1305: no test had a withheld row inside 10%.

    The thin row now misses its target by about 3%. Of the published rows
    one is inside 10% and one is not, so the fraction is a half; counting the
    numerator over every row would make it one.
    """

    payload, _ = _thin_and_wide_brma_payload(thin_target=6.2)
    rows = {row["target_name"]: row for row in payload["targets"]}
    assert rows["thin_brma_target"]["final_estimate"] is None
    assert payload["fraction_within_10pct"] == 0.5
    (brma,) = [
        row
        for row in payload["uk_diagnostics"]["target_pass_rates_by_geography_level"]
        if row["geography_level"] == "brma"
    ]
    assert brma["n_within_10pct"] == 1


def test_every_brma_row_states_its_ten_household_verdict() -> None:
    """María's ruling of 2026-10-19 on the fourth review of #1305.

    Every BRMA row carries ``carrier_floor_met``, false exactly where its model
    figures are withheld, so the release contract can warn about a thin row
    the build never flagged.
    """

    payload, _ = _thin_and_wide_brma_payload()
    verdicts = {
        row["target_name"]: row["metadata"]["carrier_floor_met"]
        for row in payload["targets"]
    }
    assert verdicts == {
        "thin_brma_target": False,
        "wide_brma_target": True,
        "other_brma_target": True,
    }


def test_the_published_options_are_settings_only() -> None:
    """The fourth review of #1305: an outcome key added to the allow-list
    passed every test. Each key here is a setting the solve was given; a new
    one is added here, after checking that it holds no outcome."""

    from microcosm.diagnostics import (
        UK_WITHHELD_PUBLISHED_ITERATE_FIELDS,
        UK_WITHHELD_PUBLISHED_OPTIONS,
    )

    assert UK_WITHHELD_PUBLISHED_OPTIONS == {
        "budget_basis",
        "epochs",
        "feasible_draw_pi_hi",
        "gate_initialization_supplied",
        "iterate_selection",
        "l1_lambda",
        "l1_penalty",
        "l2_anchor",
        "l2_anchor_weights_supplied",
        "l2_basis",
        "l2_lambda",
        "l2_penalty",
        "learning_rate",
        "mass",
        "mass_parametrization",
        "mass_reason",
        "matrix_format",
        "max_weight_ratio",
        "method",
        "seed",
        "target_loss_cap",
        "target_loss_scales",
        "target_loss_weights",
        "target_records",
        "warm_start_weights",
    }
    assert UK_WITHHELD_PUBLISHED_ITERATE_FIELDS == {
        "rule",
        "selected_epoch",
        "epochs_executed",
        "epoch_convention",
    }


def test_no_loss_over_every_row_survives_in_the_payload() -> None:
    """The third review of #1305: the solve's options kept its iterate's loss.

    ``iterate_selection_receipt`` held the selected iterate's loss over every
    row, which less the restated final loss gave the withheld row back. While
    a row is withheld the options keep their settings only, and no number in
    the payload equals a loss over every row.
    """

    import math

    receipt = {
        "rule": "best_feasible_loss",
        "selected_epoch": 3,
        "epochs_executed": 5,
        "epoch_convention": "completed_optimizer_updates; zero is start",
        "selected_loss_float32": 0.2617,
        "closing_iterate_loss_float32": 0.2711,
    }
    extra = {
        "iterate_selection_receipt": receipt,
        "budget_search": {"probes": [{"loss": 0.3119}]},
        "an_outcome_nobody_listed": 0.2903,
    }
    payload, result = _thin_and_wide_brma_payload(extra_options=extra)
    options = payload["options"]
    assert options["iterate_selection_receipt"] == {
        **receipt,
        "selected_loss_float32": None,
        "closing_iterate_loss_float32": None,
    }
    assert options["budget_search"] is None
    assert options["an_outcome_nobody_listed"] is None
    assert options["method"] == "score_only"
    every_row = (result.final_loss, result.initial_loss, 0.2617, 0.2711, 0.3119)
    for number in _numbers(payload):
        assert not any(math.isclose(number, loss, rel_tol=1e-9) for loss in every_row)
    # With nothing withheld the options stay as the solve wrote them.
    payload, _ = _thin_and_wide_brma_payload(
        thin_values=(1.0,) * 14, extra_options=extra
    )
    assert payload["options"]["iterate_selection_receipt"] == receipt
    assert payload["options"]["an_outcome_nobody_listed"] == 0.2903


def test_the_restated_totals_match_the_solve_and_drop_a_pushed_out_row() -> None:
    """Over every row the restatement reproduces the solve's own totals.

    A withheld row pushed past the cap leaves the census's listing and count.
    """

    from types import SimpleNamespace

    from microcosm.build.uk_runtime.diagnostics import (
        _restate_totals_over_published_rows,
    )

    def row(name, target, initial, final):
        return SimpleNamespace(
            name=name,
            target=target,
            initial_estimate=initial,
            final_estimate=final,
            relative_error=(final - target) / target,
        )

    rows = [
        row("a@2025", 10.0, 11.0, 10.5),
        row("b@2025", 20.0, 19.0, 21.0),
        row("thin@2025", 10.0, 12.0, 400.0),
    ]
    result = SimpleNamespace(
        diagnostics=rows,
        target_loss_weights=np.array([1.0, 2.0, 1.0]),
        target_loss_scales=np.array([10.0, 20.0, 10.0]),
        target_loss_cap=10.0,
        options={"target_loss_cap": 10.0},
    )
    payload: dict[str, object] = {}
    _restate_totals_over_published_rows(payload, result, frozenset())
    shares = np.array([0.25, 0.5, 0.25])
    assert payload["final_loss"] == pytest.approx(
        0.25 * 0.05 + 0.5 * 0.05 + 0.25 * min(39.0, 10.0)
    )
    assert payload["initial_loss"] == pytest.approx(
        float(shares @ np.array([0.1, 0.05, 0.2]))
    )
    assert payload["past_cap_census"]["pushed_out"] == 1
    _restate_totals_over_published_rows(payload, result, frozenset({"thin@2025"}))
    assert payload["final_loss"] == pytest.approx(0.25 * 0.05 + 0.5 * 0.05)
    assert payload["fraction_within_10pct"] == 1.0
    census = payload["past_cap_census"]
    assert census["n_targets"] == 2
    assert census["pushed_out"] == 0 and census["pushed_out_rows"] == []


def test_written_rows_flag_every_row_while_any_is_withheld() -> None:
    from microcosm.build.uk_runtime.diagnostics import uk_withhold_brma_figures

    withheld = frozenset({"thin@2025"})
    rows = [
        {"name": "thin@2025", "target": 10.0, "estimate": 6.0, "relative_error": -0.4},
        {"name": "wide@2025", "target": 10.0, "estimate": 14.0, "relative_error": 0.4},
    ]
    thin, wide = uk_withhold_brma_figures(rows, withheld)
    assert thin["estimate"] is None and thin["relative_error"] is None
    assert thin["target"] == 10.0 and thin["model_figures_withheld"] is True
    assert wide == {**rows[1], "model_figures_withheld": False}
    assert rows[0]["estimate"] == 6.0
    # With nothing withheld the written table keeps its old columns.
    flagged = [{**rows[0], "model_figures_withheld": True}]
    assert uk_withhold_brma_figures(flagged, frozenset()) == [rows[0]]


def test_the_report_guard_refuses_any_mention_of_a_withheld_row(capsys) -> None:
    """María's ruling of 2026-10-19: no gate output names a withheld row.

    The third review of #1305 found more than fifteen shapes that slipped
    past a guard looking for figures beside a name. The guard now refuses the
    name, the gate key and the area code themselves, in any case, anywhere.
    The fourth found that its message named the place, and a failed receipt
    sends the message on: it is now constant, and the place stays on stderr.
    """

    from microcosm.build.uk_runtime.diagnostics import (
        UK_WITHHELD_ROW_NAMED_MESSAGE,
        UKWithheldRowNamedError,
        uk_refuse_named_withheld_rows,
    )

    names = frozenset({"ons.rent.private_rent_brma@S33999999@2025"})
    keys = frozenset({"private_rent/S33999999/rent/private_rent_brma"})
    (key,) = keys

    def guard(report):
        uk_refuse_named_withheld_rows(
            report, names=names, keys=keys, areas=frozenset({"S33999999"})
        )

    # Counts, published rows and other areas pass.
    guard(
        {
            "outcomes": [
                {
                    "failures": ["private_rent/E06000001/rent/private_rent +30.0%"],
                    "details": {
                        "withheld_targets": 1,
                        "family_sizes": {"private_rent": 7},
                        "failing_targets": {
                            "private_rent/S33999998/rent/private_rent_brma": 0.3
                        },
                    },
                }
            ]
        }
    )
    for report in (
        {"failing_targets": {key: -0.4}},
        {"rows": [{"name": key, "relative_error": -0.4}]},
        {"pairs": [[key, -0.4]]},
        {key: "-41.2%"},
        {"names": [key], "errors": [-0.4]},
        {"failures": [f"{key} is 40 per cent out"]},
        {"failures": [f"err=-4.0e-01 for {key.upper()}"]},
        {"failures": ["ons.rent.private_rent_brma@S33999999 failed"]},
        {"failures": ["S33999999 is beyond the bound"]},
        {"dormant_exclusions": [key]},
    ):
        with pytest.raises(UKWithheldRowNamedError) as excinfo:
            guard(report)
        message = str(excinfo.value)
        assert message == UK_WITHHELD_ROW_NAMED_MESSAGE
        for label in (key, "S33999999", "failing_targets", "failures", "40"):
            assert label.casefold() not in message.casefold()
        assert "names a withheld row" in capsys.readouterr().err
    # Nothing withheld: nothing to guard.
    uk_refuse_named_withheld_rows(
        {"failing_targets": {key: -0.4}}, names=frozenset(), keys=frozenset()
    )


def test_an_explicit_zero_in_a_brma_row_carries_nothing() -> None:
    """A stored zero is no entry: the household does not carry the row."""

    from types import SimpleNamespace

    from scipy import sparse

    from microcosm.build.uk_runtime.diagnostics import uk_withheld_brma_rows

    n = 12
    data = np.array([1.0] * 9 + [0.0] * 3)
    matrix = sparse.csr_array((data, (np.zeros(n, dtype=int), np.arange(n))), (1, n))
    assert matrix.nnz == n
    problem = SimpleNamespace(matrix=matrix, names=["brma_row@2025"])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.arange(n, dtype=np.int64),
                    "person_household_id": np.arange(n, dtype=np.int64),
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": np.arange(n, dtype=np.int64),
                    "source_household_id": np.arange(n, dtype=np.int64),
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    rows = [{"name": "brma_row@2025"}]
    levels = {"brma_row@2025": "brma"}
    assert uk_withheld_brma_rows(problem, np.ones(n), frame, rows, levels) == {
        "brma_row@2025"
    }
    data[9] = 1.0
    matrix = sparse.csr_array((data, (np.zeros(n, dtype=int), np.arange(n))), (1, n))
    problem = SimpleNamespace(matrix=matrix, names=["brma_row@2025"])
    assert uk_withheld_brma_rows(problem, np.ones(n), frame, rows, levels) == set()
    # A zero weight carries nothing either.
    weights = np.ones(n)
    weights[0] = 0.0
    assert uk_withheld_brma_rows(problem, weights, frame, rows, levels) == {
        "brma_row@2025"
    }


def test_the_local_line_withholds_thin_local_rows_by_block() -> None:
    """María's ruling of 2026-10-23: thin local rows are withheld, by block.

    In one constituency a family total less its published band gives a thin
    band back, so the next-smallest band is withheld beside it, as nationally.
    A thin row in another area is withheld on its own, and no row of one area
    is withheld for a difference with another's that gives nothing back.
    """

    rows = {
        "fam.total@E14000001@2025": list(range(30)),
        "fam.band_1@E14000001@2025": list(range(25)),
        "fam.band_2@E14000001@2025": list(range(25, 30)),
        "other.total@E14000002@2025": list(range(30, 35)),
        "other.total@E14000003@2025": list(range(35, 60)),
    }
    levels = {name: "constituency" for name in rows}
    thin, beside = _withheld_sets(rows, 60, levels=levels, regions=["LONDON"] * 60)
    assert thin == {"fam.band_2@E14000001@2025", "other.total@E14000002@2025"}
    assert beside == {"fam.band_1@E14000001@2025"}
    # The same rows as one national block would give the same verdict; the
    # block split changes only which differences are looked for.
    national = _withheld_sets(rows, 60)
    assert national == (thin, beside)


def test_a_region_row_less_its_published_local_cells_withholds_one_more() -> None:
    """Vahid's review of #1383, item 1: the region and its cells are one block.

    The region row binds beside its local-authority cells, so the region less
    the published cell gives the thin one back, and the next-smallest giver
    is withheld beside it. Per area, nothing was withheld beside it.
    """

    rows = {
        "fam.region@E12000007@2025": list(range(30)),
        "fam.cell@E09000001@2025": list(range(25, 30)),
        "fam.cell@E09000002@2025": list(range(25)),
        "fam.region@E12000008@2025": list(range(30, 60)),
    }
    levels = {
        "fam.region@E12000007@2025": "region",
        "fam.region@E12000008@2025": "region",
        "fam.cell@E09000001@2025": "local_authority",
        "fam.cell@E09000002@2025": "local_authority",
    }
    regions = ["LONDON"] * 30 + ["SOUTH_EAST"] * 30
    thin, beside = _withheld_sets(rows, 60, levels=levels, regions=regions)
    assert thin == {"fam.cell@E09000001@2025"}
    assert beside == {"fam.cell@E09000002@2025"}


def test_a_regions_constituencies_less_its_authorities_withhold_one_more() -> None:
    """Both local grains partition a region, so each gives the other back."""

    rows = {
        "fam.cell@E14000001@2025": list(range(20)),
        "fam.cell@E14000002@2025": list(range(20, 40)),
        "fam.cell@E06000001@2025": list(range(35)),
        "fam.cell@E06000002@2025": list(range(35, 40)),
    }
    levels = {
        "fam.cell@E14000001@2025": "constituency",
        "fam.cell@E14000002@2025": "constituency",
        "fam.cell@E06000001@2025": "local_authority",
        "fam.cell@E06000002@2025": "local_authority",
    }
    thin, beside = _withheld_sets(rows, 40, levels=levels, regions=["WALES"] * 40)
    assert thin == {"fam.cell@E06000002@2025"}
    # The constituencies less the larger authority give the thin one back;
    # the smaller constituency is withheld (ties go by name), and what is left
    # (the thin cell with it) rests on 25 households.
    assert beside == {"fam.cell@E14000001@2025"}


def _uc_family_rows() -> tuple[dict[str, list[int]], dict[str, str], dict[str, str]]:
    # The national UC caseload and its constituency cells share one
    # measurement, so they are one family with its parent (the contract).
    rows = {
        "dwp.uc.households@2025": list(range(60)),
        "dwp.uc.households_by_area@E14000001@2025": list(range(30)),
        "dwp.uc.households_by_area@E14000002@2025": list(range(30, 55)),
        "dwp.uc.households_by_area@E14000003@2025": list(range(55, 60)),
    }
    levels = {name: "constituency" for name in rows}
    levels["dwp.uc.households@2025"] = "country"
    targets = {
        name: "dwp.uc.households"
        if name.startswith("dwp.uc.households@")
        else ("dwp.uc.households_by_area")
        for name in rows
    }
    return rows, levels, targets


def test_a_national_row_less_a_familys_published_cells_withholds_one_more() -> None:
    """A national parent less its family's cells across regions gives one back.

    No region row binds, so no region block sees it: the family's own block
    with its parent, the contract's national UC caseload, does.
    """

    rows, levels, targets = _uc_family_rows()
    regions = ["NORTH_EAST"] * 30 + ["NORTH_WEST"] * 30
    thin, beside = _withheld_sets(
        rows, 60, levels=levels, regions=regions, targets=targets
    )
    assert thin == {"dwp.uc.households_by_area@E14000003@2025"}
    assert beside == {"dwp.uc.households_by_area@E14000002@2025"}
    # Without the contract targets no family block forms, and nothing is
    # withheld beside the thin cell: the gap this closes.
    assert _withheld_sets(rows, 60, levels=levels, regions=regions) == (thin, set())


def test_a_row_one_block_withholds_stays_hidden_in_another() -> None:
    """A row withheld beside a thin one is protected in every block holding it.

    The family block withholds the second cell beside the thin third. A row
    of the region's own, on exactly that cell's households, would give it
    back, and with it the thin cell, so the region block withholds that too.
    """

    rows, levels, targets = _uc_family_rows()
    rows["other.cell@E14000002@2025"] = list(range(30, 55))
    levels["other.cell@E14000002@2025"] = "constituency"
    regions = ["NORTH_EAST"] * 30 + ["NORTH_WEST"] * 30
    thin, beside = _withheld_sets(
        rows, 60, levels=levels, regions=regions, targets=targets
    )
    assert thin == {"dwp.uc.households_by_area@E14000003@2025"}
    assert beside == {
        "dwp.uc.households_by_area@E14000002@2025",
        "other.cell@E14000002@2025",
    }


def test_local_rows_without_household_regions_refuse() -> None:
    """A local row is checked against its region's rows, so it needs regions."""

    rows = {"fam.cell@E14000001@2025": list(range(20))}
    with pytest.raises(ValueError, match="household table needs a region column"):
        _withheld_sets(rows, 20, levels={"fam.cell@E14000001@2025": "constituency"})


def test_the_report_guard_matches_a_withheld_local_row_by_its_key(capsys) -> None:
    """A constituency binds many families, so its bare code names published rows."""

    from microcosm.build.uk_runtime.diagnostics import (
        UKWithheldRowNamedError,
        uk_refuse_named_withheld_rows,
        uk_withheld_gate_areas,
        uk_withheld_gate_keys,
    )

    rows = [
        {
            "family": "fam",
            "area_code": "E14000001",
            "metric": "band_2",
            "geography_level": "constituency",
            "model_figures_withheld": True,
        },
        {
            "family": "private_rent",
            "area_code": "S33999999",
            "metric": "private_rent_brma",
            "geography_level": "statistical_scope",
            "model_figures_withheld": True,
        },
        {
            "family": "other",
            "area_code": "E14000001",
            "metric": "total",
            "geography_level": "constituency",
        },
    ]
    keys = uk_withheld_gate_keys(rows)
    areas = uk_withheld_gate_areas(rows)
    assert keys == {"fam/E14000001/band_2", "private_rent/S33999999/private_rent_brma"}
    assert areas == {"S33999999"}

    def guard(report):
        uk_refuse_named_withheld_rows(report, names=frozenset(), keys=keys, areas=areas)

    # Another family's published row in the same constituency may be named.
    guard({"failures": ["other/E14000001/total +30.0%"], "areas": ["E14000001"]})
    for report in (
        {"failures": ["fam/E14000001/band_2 +30.0%"]},
        {"failures": ["S33999999 is beyond the bound"]},
    ):
        with pytest.raises(UKWithheldRowNamedError):
            guard(report)
        assert "names a withheld row" in capsys.readouterr().err
