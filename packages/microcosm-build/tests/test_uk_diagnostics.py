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
    uk_weakest_areas_by_fit,
    uk_weakest_families,
    uk_weight_summary,
    uk_zero_weight_strata,
    write_uk_calibration_diagnostics,
)
from microcosm.calibrate import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    TargetRegistry,
    TargetSpec,
    diagnostics_payload,
    score_targets,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

_SPI_COLUMN = "household_is_spi_synthetic"
_CG_COLUMN = "household_is_capital_gains_clone"


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


def _diagnostics_case(*, with_skipped: bool = False):
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
    ]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


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
    path = write_uk_calibration_diagnostics(
        result,
        tmp_path / "calibration_diagnostics.json",
        frame,
        target_geography_levels=geography,
        target_registry=registry,
    )

    assert json.loads(path.read_text(encoding="utf-8")) == (
        uk_calibration_diagnostics_payload(
            result,
            frame,
            target_geography_levels=geography,
            target_registry=registry,
        )
    )
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="Out of range float values"):
        write_uk_calibration_diagnostics(
            result,
            path,
            frame,
            target_geography_levels=geography,
            target_registry=registry,
            build={"not_json": float("nan")},
        )
    assert path.read_bytes() == previous
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_writer_preserves_prior_bytes_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, frame, registry, geography = _diagnostics_case()
    path = tmp_path / "calibration_diagnostics.json"
    prior = b'{"prior":true}\n'
    path.write_bytes(prior)

    def fail_replace(_temporary, _output):
        raise OSError("seeded replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="seeded replace failure"):
        write_uk_calibration_diagnostics(
            result,
            path,
            frame,
            target_geography_levels=geography,
            target_registry=registry,
        )

    assert path.read_bytes() == prior
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_geography_level_vocabulary_includes_future_release_levels() -> None:
    assert UK_TARGET_GEOGRAPHY_LEVELS == (
        "national",
        "region",
        "country",
        "local_authority",
        "constituency",
    )
