from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build import FitWeightRecord
from populace.build.us_runtime.acs_sources import (
    AcsSourceArtifact,
    AcsSourceManifest,
)
from populace.build.us_runtime.base_pool import spine_column
from populace.build.us_runtime.puf_support import support_channel_column
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_multispine_base",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    benefit_participation: bool = True,
    spines: tuple[str, str] | None = None,
) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "person_household_id": [1, 2],
            "person_tax_unit_id": [1, 2],
            "person_spm_unit_id": [1, 2],
            "person_family_id": [1, 2],
            "person_marital_unit_id": [1, 2],
            "age": [35, 67],
            "is_female": [False, True],
        }
    )
    if benefit_participation:
        person["takes_up_snap_if_eligible"] = [True, False]
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]}),
        "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
        "family": pd.DataFrame({"family_id": [1, 2]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
    }
    if spines is not None:
        for entity, table in tables.items():
            table[spine_column(entity)] = list(spines)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([40.0, 60.0]),
                WeightKind.CALIBRATED,
            )
        },
    )


def _manifest() -> AcsSourceManifest:
    source_directory = (
        "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/"
    )
    return AcsSourceManifest(
        version=1,
        spine="acs_2024_1yr",
        vintage=2024,
        verified_on="2026-07-10",
        source_directory=source_directory,
        artifacts=(
            AcsSourceArtifact(
                role="household",
                filename="csv_hus.zip",
                url=f"{source_directory}csv_hus.zip",
                sha256="1" * 64,
                size_bytes=123,
            ),
            AcsSourceArtifact(
                role="person",
                filename="csv_pus.zip",
                url=f"{source_directory}csv_pus.zip",
                sha256="2" * 64,
                size_bytes=456,
            ),
        ),
    )


def test_parser_exposes_production_defaults_and_transfer_controls() -> None:
    builder = _load_builder_module()

    args = builder._parse_args(
        [
            "--base-h5",
            "dense.h5",
            "--out-h5",
            "combined.h5",
        ]
    )

    assert args.inputs_dir == builder.DEFAULT_INPUTS_DIR
    assert args.inputs_dir.is_absolute()
    assert args.inputs_dir.parts[-2:] == ("inputs", "acs_2024_1yr")
    assert args.puma_ladder == builder.DEFAULT_PUMA_LADDER
    assert args.puma_ladder.is_absolute()
    assert args.n_estimators == 32
    assert args.max_targets_per_fit == 8
    assert args.period == 2024
    assert args.acs_share == 0.5
    assert args.seed == 0
    assert args.geography_seed == 0
    assert args.donor_channel == builder.ACS_DONOR_CHANNEL_AUTO

    custom = builder._parse_args(
        [
            "--base-h5",
            "dense.h5",
            "--out-h5",
            "combined.h5",
            "--chunksize",
            "1234",
            "--acs-share",
            "0.3",
            "--seed",
            "17",
            "--geography-seed",
            "23",
            "--puma-ladder",
            "ladder.npz",
            "--n-estimators",
            "9",
            "--max-targets-per-fit",
            "3",
            "--donor-channel",
            "benefit_support",
        ]
    )
    assert custom.chunksize == 1234
    assert custom.acs_share == 0.3
    assert custom.seed == 17
    assert custom.geography_seed == 23
    assert custom.puma_ladder == Path("ladder.npz")
    assert custom.n_estimators == 9
    assert custom.max_targets_per_fit == 3
    assert custom.donor_channel == "benefit_support"


def test_main_wires_verified_sources_transfer_audit_export_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    manifest = _manifest()
    base = _frame()
    combined = _frame(spines=("asec_puf", "acs_2024_1yr"))
    combined.table("household")["puma"] = ["0100100", "0200100"]
    combined.table("household")["congressional_district_geoid"] = [101, 200]
    combined.table("household")["county_fips"] = ["01001", "02020"]
    combined.table("household")["TYPEHUGQ"] = [1.0, 3.0]
    base_h5 = tmp_path / "dense.h5"
    base_h5.write_bytes(b"dense-base")
    manifest_path = tmp_path / "acs_sources.json"
    manifest_path.write_text("{}", encoding="utf-8")
    puma_ladder_path = tmp_path / "us_puma_ladder_2020.npz"
    puma_ladder_path.write_bytes(b"puma-ladder")
    puma_ladder = builder.UsPumaLadder(
        puma=np.asarray([100_100], dtype=np.int64),
        puma_population=np.asarray([100.0]),
        cd_overlap_puma=np.asarray([100_100], dtype=np.int64),
        cd_overlap_cd=np.asarray([101], dtype=np.int64),
        cd_overlap_population=np.asarray([100.0]),
        county_overlap_puma=np.asarray([100_100], dtype=np.int64),
        county_overlap_county=np.asarray([1_001], dtype=np.int32),
        county_overlap_population=np.asarray([100.0]),
        tract_overlap_puma=np.asarray([100_100], dtype=np.int64),
        tract_overlap_tract=np.asarray([1_001_000_100], dtype=np.int64),
        tract_overlap_population=np.asarray([100.0]),
        metadata={
            "schema_version": 1,
            "kind": "us_puma_ladder",
            "puma_vintage": "2020_puma",
            "sampling_basis": "population",
            "layers": {
                "congressional_district": {"vintage": "119th_congress"},
                "county": {"vintage": "2020_census"},
                "tract": {"vintage": "2020_census"},
            },
        },
    )
    inputs_dir = tmp_path / "inputs"
    household_zip = inputs_dir / "csv_hus.zip"
    person_zip = inputs_dir / "csv_pus.zip"
    source = builder.AcsPumsSource(household_zip, person_zip)
    output_h5 = tmp_path / "combined.h5"
    summary_path = tmp_path / "combined.summary.json"
    captured: dict[str, object] = {}
    transfer_plan = {
        "person": {
            "benefit_participation": ("takes_up_snap_if_eligible",),
        }
    }

    def fake_load_manifest(path):
        captured["manifest_path"] = path
        return manifest

    def fake_fetch(cache_dir, *, manifest):
        captured["fetch"] = (cache_dir, manifest)
        return source

    def fake_build(actual_base, actual_source, **kwargs):
        captured["build"] = (actual_base, actual_source, kwargs)
        return builder.AcsMultispineResult(
            frame=combined,
            fit_records=(FitWeightRecord("acs_transfer:person:benefits", "design"),),
            provenance={
                "enabled": True,
                "deferred_inputs": ["block_geoid", "tract_geoid"],
                "geography_ladder": {
                    "applied": True,
                    "household_rows": 2,
                    "ladder_pumas": 1,
                    "layer_vintages": puma_ladder.layer_vintages,
                    "sampling_basis": "population",
                    "seed": 19,
                    "resolved_model_inputs": [
                        "congressional_district_geoid",
                        "county_fips",
                    ],
                    "unresolved_sub_puma_inputs": [
                        "block_geoid",
                        "tract_geoid",
                    ],
                },
                "imputed_inputs": [
                    {
                        "column": "takes_up_snap_if_eligible",
                        "family": "benefit_participation",
                        "unmodeled_recipient_rows": 0,
                    }
                ],
            },
        )

    def fake_write(frame, path, *, period):
        captured["write"] = (frame, path, period)
        path.write_bytes(b"combined-output")

    def fake_load_puma_ladder(path):
        captured["puma_ladder_path"] = path
        return puma_ladder

    monkeypatch.setattr(builder, "_load_base_frame", lambda path: base)
    monkeypatch.setattr(
        builder,
        "_require_dense_donor_coverage",
        lambda frame, **kwargs: None,
    )
    monkeypatch.setattr(
        builder,
        "declared_acs_transfer_target_families",
        lambda: transfer_plan,
    )
    monkeypatch.setattr(
        builder.acs_sources,
        "load_acs_source_manifest",
        fake_load_manifest,
    )
    monkeypatch.setattr(
        builder.acs_sources,
        "fetch_acs_pums_sources",
        fake_fetch,
    )
    monkeypatch.setattr(builder, "build_optional_acs_multispine", fake_build)
    monkeypatch.setattr(
        builder,
        "load_us_puma_ladder",
        fake_load_puma_ladder,
    )
    monkeypatch.setattr(builder, "_engine_input_null_audit", lambda frame: [])
    monkeypatch.setattr(
        builder,
        "_preflight_staging_export",
        lambda frame: 123_456,
    )
    monkeypatch.setattr(builder, "_write_dataset", fake_write)

    exit_code = builder.main(
        [
            "--base-h5",
            str(base_h5),
            "--out-h5",
            str(output_h5),
            "--summary",
            str(summary_path),
            "--source-manifest",
            str(manifest_path),
            "--inputs-dir",
            str(inputs_dir),
            "--puma-ladder",
            str(puma_ladder_path),
            "--max-households",
            "7",
            "--chunksize",
            "2000",
            "--acs-share",
            "0.4",
            "--seed",
            "11",
            "--geography-seed",
            "19",
        ]
    )

    assert exit_code == 0
    assert captured["manifest_path"] == manifest_path
    assert captured["puma_ladder_path"] == puma_ladder_path
    assert captured["fetch"] == (inputs_dir, manifest)
    actual_base, actual_source, build_options = captured["build"]
    assert actual_base is base
    assert actual_source.max_households == 7
    assert build_options.pop("puma_ladder") is puma_ladder
    assert build_options == {
        "chunksize": 2000,
        "acs_share": 0.4,
        "target_families": transfer_plan,
        "donor_channel": builder.ACS_DONOR_CHANNEL_AUTO,
        "seed": 11,
        "n_estimators": 32,
        "max_targets_per_fit": 8,
        "geography_seed": 19,
    }
    assert captured["write"] == (combined, output_h5, 2024)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["artifact_kind"] == "nullable_precalibration_staging_h5"
    assert summary["calibration_applied"] is False
    assert summary["simulation_ready"] is False
    assert summary["simulation_ready_except_calibration"] is True
    assert summary["simulation_readiness_blockers"] == ["calibration_not_applied"]
    assert [item["id"] for item in summary["reviewed_limitations"]] == [
        "acs_group_quarters_housing_universe",
        "native_acs_source_universe_blanks",
        "sub_puma_geographic_precision",
    ]
    assert summary["reviewed_limitations"][2]["unavailable_exact_geography"] == [
        "block_geoid",
        "tract_geoid",
    ]
    assert summary["reviewed_engine_input_nulls"] == []
    assert "pending_engine_input_nulls" not in summary
    assert summary["staging_export_peak_estimate_bytes"] == 123_456
    assert summary["geography_ladder"] == {
        "path": str(puma_ladder_path.resolve()),
        "sha256": hashlib.sha256(b"puma-ladder").hexdigest(),
        "pumas": 1,
        "layer_vintages": {
            "congressional_district": "119th_congress",
            "county": "2020_census",
            "puma": "2020_puma",
            "tract": "2020_census",
        },
        "seed": 19,
        "assignment": summary["orchestration"]["provenance"]["geography_ladder"],
    }
    assert summary["acs_sources"]["manifest"] == str(manifest_path.resolve())
    assert (
        summary["acs_sources"]["manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()
    )
    assert summary["acs_sources"]["artifacts"] == [
        {
            "role": "household",
            "filename": "csv_hus.zip",
            "url": manifest.artifacts[0].url,
            "sha256": "1" * 64,
            "size_bytes": 123,
            "local_path": str(household_zip.resolve()),
        },
        {
            "role": "person",
            "filename": "csv_pus.zip",
            "url": manifest.artifacts[1].url,
            "sha256": "2" * 64,
            "size_bytes": 456,
            "local_path": str(person_zip.resolve()),
        },
    ]
    assert summary["weights_audit"]["details"]["resolved_weight_kinds"] == {
        "acs_transfer:person:benefits": "design"
    }
    assert summary["rows"]["combined"] == {entity: 2 for entity in US_SCHEMA.entities}
    assert summary["household_weight_totals"] == {
        "base": 100.0,
        "combined": 100.0,
    }
    assert summary["spine_totals"] == {
        "acs_2024_1yr": {
            "rows": {entity: 1 for entity in US_SCHEMA.entities},
            "household_weight_total": 60.0,
        },
        "asec_puf": {
            "rows": {entity: 1 for entity in US_SCHEMA.entities},
            "household_weight_total": 40.0,
        },
    }
    assert summary["output"]["sha256"] == hashlib.sha256(b"combined-output").hexdigest()


def test_weights_audit_failure_aborts_before_export() -> None:
    builder = _load_builder_module()
    result = builder.AcsMultispineResult(
        frame=_frame(),
        fit_records=(FitWeightRecord("acs_transfer:person:benefits", "none"),),
    )

    with pytest.raises(SystemExit) as exc:
        builder._audit_fits(result)

    assert "Weights audit failed" in str(exc.value)
    assert "acs_transfer:person:benefits" in str(exc.value)
    assert "unweighted" in str(exc.value)


def test_default_transfer_fails_closed_before_fetch_without_benefit_participation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        builder.acs_sources,
        "load_acs_source_manifest",
        lambda path: _manifest(),
    )
    monkeypatch.setattr(
        builder,
        "_load_base_frame",
        lambda path: _frame(benefit_participation=False),
    )
    monkeypatch.setattr(builder, "_sha256", lambda path: "0" * 64)

    def must_not_fetch(*_args, **_kwargs):
        raise AssertionError("source fetch must not run for an incomplete donor")

    monkeypatch.setattr(
        builder.acs_sources,
        "fetch_acs_pums_sources",
        must_not_fetch,
    )

    with pytest.raises(SystemExit) as exc:
        builder.main(
            [
                "--base-h5",
                "incomplete.h5",
                "--out-h5",
                "combined.h5",
            ]
        )

    assert "no takes_up_*" in str(exc.value)
    assert "must run after the benefit input-family stages" in str(exc.value)


def test_empty_transfer_audit_is_not_treated_as_success() -> None:
    builder = _load_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._audit_fits(
            SimpleNamespace(
                fit_records=(),
            )
        )

    assert "produced no fit records" in str(exc.value)


class _DefaultsEngine:
    _defaults = {
        "age": 0,
        "alimony_income": 0.0,
        "employment_income_before_lsr": 0.0,
        "has_esi": False,
        "is_female": False,
        "state_fips": 0,
        "takes_up_snap_if_eligible": True,
        "weekly_hours_worked_before_lsr": 40.0,
    }

    def default_values(self, names):
        return {name: self._defaults[name] for name in names if name in self._defaults}


def test_dense_donor_ignores_default_runtime_column_not_consumed_by_transfer() -> None:
    builder = _load_builder_module()
    base = _frame()
    base.person["weekly_hours_worked_before_lsr"] = [40.0, 40.0]

    builder._require_dense_donor_coverage(
        base,
        engine=_DefaultsEngine(),
        donor_channel=None,
        target_families={
            "person": {
                "benefit_participation": ("takes_up_snap_if_eligible",),
            }
        },
    )


def test_dense_donor_missing_transfer_consumed_column_fails_hard() -> None:
    builder = _load_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._require_dense_donor_coverage(
            _frame(),
            engine=_DefaultsEngine(),
            donor_channel=None,
            target_families={"person": {"health": ("has_esi",)}},
        )

    message = str(exc.value)
    assert "hard ACS transfer-consumption gate" in message
    assert "person.has_esi" in message
    assert "transfer-consumed column is absent" in message


def test_dense_donor_default_transfer_consumed_column_fails_hard() -> None:
    builder = _load_builder_module()
    base = _frame()
    base.person["has_esi"] = [False, False]

    with pytest.raises(SystemExit) as exc:
        builder._require_dense_donor_coverage(
            base,
            engine=_DefaultsEngine(),
            donor_channel=None,
            target_families={"person": {"health": ("has_esi",)}},
        )

    message = str(exc.value)
    assert "hard ACS transfer-consumption gate" in message
    assert "person.has_esi" in message
    assert "every observed donor value equals the engine default" in message


def test_dense_donor_default_consumed_optional_feature_fails_hard() -> None:
    builder = _load_builder_module()
    base = _frame()
    base.person["employment_income_before_lsr"] = [0.0, 0.0]

    with pytest.raises(SystemExit) as exc:
        builder._require_dense_donor_coverage(
            base,
            engine=_DefaultsEngine(),
            donor_channel=None,
            target_families={
                "person": {
                    "benefit_participation": ("takes_up_snap_if_eligible",),
                }
            },
        )

    message = str(exc.value)
    assert "person.employment_income_before_lsr" in message
    assert "every observed donor value equals the engine default" in message


def test_signal_bearing_release_exclusion_is_irrelevant_to_donor_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = _load_builder_module()
    base = _frame()
    base.person["alimony_income"] = [0.0, 100.0]

    builder._require_dense_donor_coverage(
        base,
        engine=_DefaultsEngine(),
        donor_channel=None,
        target_families={
            "person": {
                "benefit_participation": ("takes_up_snap_if_eligible",),
            }
        },
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_dense_donor_coverage_is_checked_on_resolved_puf_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    base = _frame()
    channel_column = support_channel_column("person")
    for entity in base.entities:
        base.table(entity)[support_channel_column(entity)] = [
            "asec",
            "puf_tax_detail",
        ]
    captured: dict[str, Frame] = {}

    def fake_requirements(frame, target_families):
        captured["frame"] = frame
        return {"person": ("age",)}

    monkeypatch.setattr(
        builder,
        "acs_transfer_donor_requirements",
        fake_requirements,
    )

    builder._require_dense_donor_coverage(
        base,
        engine=_DefaultsEngine(),
        target_families={"person": {"test": ("age",)}},
    )

    selected = captured["frame"]
    assert selected.n("person") == 1
    assert selected.person[channel_column].tolist() == ["puf_tax_detail"]


def test_partial_donor_channel_metadata_fails_before_coverage_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    base = _frame()
    base.table("household")[support_channel_column("household")] = [
        "asec",
        "puf_tax_detail",
    ]
    called = False

    def fake_requirements(frame, target_families):
        nonlocal called
        called = True
        return {"person": ("age",)}

    monkeypatch.setattr(
        builder,
        "acs_transfer_donor_requirements",
        fake_requirements,
    )

    with pytest.raises(SystemExit, match="partial support metadata"):
        builder._require_dense_donor_coverage(
            base,
            engine=_DefaultsEngine(),
            target_families={"person": {"test": ("age",)}},
        )

    assert not called


def test_default_transfer_cannot_report_success_without_benefit_imputation() -> None:
    builder = _load_builder_module()
    result = builder.AcsMultispineResult(
        frame=_frame(),
        provenance={
            "imputed_inputs": [
                {
                    "column": "qualified_dividend_income",
                    "family": "puf_tax_itemization",
                }
            ]
        },
    )

    with pytest.raises(SystemExit) as exc:
        builder._require_benefit_participation_transfer(result)

    assert "no takes_up_* benefit-participation input" in str(exc.value)


def test_default_transfer_must_register_every_planned_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    result = builder.AcsMultispineResult(
        frame=_frame(),
        provenance={
            "imputed_inputs": [
                {
                    "column": "takes_up_snap_if_eligible",
                    "family": "benefit_participation",
                }
            ]
        },
    )
    monkeypatch.setattr(
        builder,
        "default_acs_transfer_target_families",
        lambda donor: {
            "person": {
                "benefit_participation": ("takes_up_snap_if_eligible",),
                "model_required": ("has_esi",),
            }
        },
    )

    with pytest.raises(SystemExit) as exc:
        builder._require_default_transfer_coverage(result, _frame())

    assert "omitted donor-observed model input" in str(exc.value)
    assert "has_esi" in str(exc.value)


def test_staging_h5_round_trips_base_only_nullable_boolean(tmp_path: Path) -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.table("person")["is_snap_abawd_discretionary_exempt"] = pd.Series(
        [True, np.nan], dtype=object
    )
    output = tmp_path / "nullable-bool-staging.h5"

    builder._write_dataset(frame, output, period=2024)

    stored = pd.read_hdf(output, key="person")
    assert stored.loc[0, "is_snap_abawd_discretionary_exempt"] is True
    assert pd.isna(stored.loc[1, "is_snap_abawd_discretionary_exempt"])


def test_engine_input_null_audit_includes_float_bool_and_spine_counts() -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.person["age"] = [35.0, np.nan]
    frame.person["takes_up_snap_if_eligible"] = pd.Series([True, np.nan], dtype=object)

    class FakeEngine:
        def variables(self):
            return ["age", "takes_up_snap_if_eligible"]

        def variable_metadata(self, name):
            return SimpleNamespace(
                dtype={
                    "age": "float",
                    "takes_up_snap_if_eligible": "bool",
                }[name]
            )

    audit = builder._engine_input_null_audit(frame, FakeEngine())

    assert audit == [
        {
            "entity": "person",
            "column": "age",
            "dtype": "float",
            "missing_rows": 1,
            "rows": 2,
            "missing_rows_by_spine": {"acs_2024_1yr": 1},
        },
        {
            "entity": "person",
            "column": "takes_up_snap_if_eligible",
            "dtype": "bool",
            "missing_rows": 1,
            "rows": 2,
            "missing_rows_by_spine": {"acs_2024_1yr": 1},
        },
    ]


def test_nullable_staging_writer_round_trips_group_quarters_blanks(
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.person["real_estate_taxes"] = [1_000.0, np.nan]
    frame.person["pre_subsidy_rent"] = [1_200.0, np.nan]
    frame.table("household")["TYPEHUGQ"] = [1.0, 3.0]
    frame.table("household")["tenure_type"] = ["RENTED", np.nan]
    frame.table("spm_unit")["spm_unit_tenure_type"] = ["RENTER", np.nan]
    output = tmp_path / "staging.h5"

    builder._write_dataset(frame, output, period=2024)

    with pd.HDFStore(output, mode="r") as store:
        assert store["person"]["real_estate_taxes"].isna().sum() == 1
        assert store["person"]["pre_subsidy_rent"].isna().sum() == 1
        assert store["household"]["tenure_type"].isna().sum() == 1
        assert store["spm_unit"]["spm_unit_tenure_type"].isna().sum() == 1
        assert store["household"]["household_weight"].tolist() == [40.0, 60.0]
        metadata = json.loads(store["_populace_staging_metadata"].iloc[0])
    assert metadata == {
        "artifact_kind": "nullable_precalibration_staging_h5",
        "entity_hdf_format": "fixed_nullable",
        "household_weight_kind": "calibrated",
    }


def test_staging_export_preflight_enforces_decimal_30gb_limit() -> None:
    builder = _load_builder_module()
    frame = _frame()

    estimate = builder._preflight_staging_export(frame)
    assert estimate < builder.DEFAULT_STAGING_EXPORT_PEAK_LIMIT_BYTES
    with pytest.raises(MemoryError, match="above the 0.00 GB limit"):
        builder._preflight_staging_export(frame, max_peak_bytes=1)


def test_registered_transfer_with_unmodeled_required_rows_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.person["has_esi"] = pd.Series([True, np.nan], dtype=object)
    result = builder.AcsMultispineResult(
        frame=frame,
        provenance={
            "imputed_inputs": [
                {
                    "column": "has_esi",
                    "family": "model_required_boolean",
                    "unmodeled_recipient_rows": 1,
                }
            ]
        },
    )
    monkeypatch.setattr(
        builder,
        "default_acs_transfer_target_families",
        lambda donor: {"person": {"model_required_boolean": ("has_esi",)}},
    )

    with pytest.raises(SystemExit, match="left 1 unmodeled row"):
        builder._require_default_transfer_coverage(result, _frame())


def test_group_quarters_rent_gap_is_explicit_structural_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.person["pre_subsidy_rent"] = [1_000.0, np.nan]
    frame.table("household")["TYPEHUGQ"] = [np.nan, 3.0]
    result = builder.AcsMultispineResult(
        frame=frame,
        provenance={
            "imputed_inputs": [
                {
                    "column": "pre_subsidy_rent",
                    "family": "housing",
                    "unmodeled_recipient_rows": 1,
                }
            ]
        },
    )
    monkeypatch.setattr(
        builder,
        "default_acs_transfer_target_families",
        lambda donor: {"person": {"housing": ("pre_subsidy_rent",)}},
    )

    coverage = builder._require_default_transfer_coverage(result, _frame())

    assert coverage["structural_pending"] == [
        {
            "column": "pre_subsidy_rent",
            "entity": "person",
            "rows": 1,
            "reason": (
                "ACS group-quarters rows are outside the housing-tenure universe"
            ),
        }
    ]


def test_reviewed_limitations_close_gq_and_sub_puma_gaps() -> None:
    builder = _load_builder_module()
    frame = _frame(spines=("asec_puf", "acs_2024_1yr"))
    frame.table("household")["TYPEHUGQ"] = [np.nan, 3.0]
    result = builder.AcsMultispineResult(
        frame=frame,
        provenance={
            "geography_ladder": {
                "applied": True,
                "seed": 29,
                "layer_vintages": {
                    "puma": "2020_puma",
                    "congressional_district": "119th_congress",
                    "county": "2020_census",
                    "tract": "2020_census",
                },
                "unresolved_sub_puma_inputs": ["block_geoid", "tract_geoid"],
            }
        },
    )
    transfer_coverage = {
        "structural_pending": [
            {
                "column": "pre_subsidy_rent",
                "entity": "person",
                "rows": 1,
                "reason": (
                    "ACS group-quarters rows are outside the housing-tenure universe"
                ),
            }
        ]
    }
    input_null_audit = [
        {
            "entity": "person",
            "column": "pre_subsidy_rent",
            "missing_rows_by_spine": {"acs_2024_1yr": 1},
        },
        {
            "entity": "person",
            "column": "employment_income_before_lsr",
            "missing_rows_by_spine": {"acs_2024_1yr": 1},
        },
    ]

    limitations = builder._reviewed_limitations(
        result,
        transfer_coverage=transfer_coverage,
        input_null_audit=input_null_audit,
    )

    gq, native, geography = limitations
    assert gq["status"] == "reviewed_structural_absence"
    assert gq["affected_rows"] == {"household": 1, "person": 1, "spm_unit": 1}
    assert gq["engine_input_nulls"] == [input_null_audit[0]]
    assert gq["transfer_evidence"] == transfer_coverage["structural_pending"]
    assert native["engine_input_nulls_excluding_group_quarters_housing"] == [
        input_null_audit[1]
    ]
    assert geography["status"] == "reviewed_probabilistic_assignment"
    assert geography["unavailable_exact_geography"] == [
        "block_geoid",
        "tract_geoid",
    ]
    assert geography["assignment_seed"] == 29
    assert all(item["calibration_blocker"] is False for item in limitations)


def test_donor_release_identity_verifies_sha_and_pins_provenance(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    sha = "a" * 64
    manifest = tmp_path / "release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_role": "national_default",
                "is_default": True,
                "build": {"build_id": "populace-us-2024-buildo-sparse-x"},
                "artifacts": {
                    "populace_us_2024": {
                        "kind": "microdata",
                        "path": "populace_us_2024.h5",
                        "repo_id": "policyengine/populace-us",
                        "revision": "populace-us-2024-buildo-sparse-x",
                        "sha256": sha,
                    },
                    "calibration_diagnostics": {
                        "kind": "diagnostics",
                        "path": "calibration_diagnostics.json",
                        "sha256": "b" * 64,
                    },
                },
            }
        )
    )

    identity = module._donor_release_identity(manifest, sha)

    assert identity == {
        "manifest_path": str(manifest.resolve()),
        "artifact": "populace_us_2024",
        "release_id": "populace-us-2024-buildo-sparse-x",
        "revision": "populace-us-2024-buildo-sparse-x",
        "repo_id": "policyengine/populace-us",
        "sha256": sha,
        "dataset_role": "national_default",
        "is_default": True,
    }
    assert module._donor_release_identity(None, sha) is None


def test_donor_release_identity_rejects_sha_mismatch(tmp_path: Path) -> None:
    module = _load_builder_module()
    manifest = tmp_path / "release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": {
                    "populace_us_2024": {
                        "kind": "microdata",
                        "sha256": "a" * 64,
                    }
                }
            }
        )
    )

    with pytest.raises(SystemExit, match="does not match its release manifest"):
        module._donor_release_identity(manifest, "c" * 64)


def test_donor_release_identity_requires_one_microdata_artifact(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    manifest = tmp_path / "release_manifest.json"
    manifest.write_text(json.dumps({"artifacts": {}}))

    with pytest.raises(SystemExit, match="exactly one microdata artifact"):
        module._donor_release_identity(manifest, "a" * 64)
