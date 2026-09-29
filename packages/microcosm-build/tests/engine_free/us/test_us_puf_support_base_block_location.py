"""``--location-rule block_v1`` on the PUF-support base builder (microcosm #696).

The legacy rule must stay byte-identical: same stage list, descriptions and
pipeline sha, same run config and child argv. ``block_v1`` swaps the two
legacy location stages for one ``household_location_assignment`` stage that
draws each household's 2020 block from its CPS ASEC source geography and
records the rule and seed in stage metadata, the summary and H5 root attrs.
Everything here is engine-free: a synthetic block ladder, synthetic ASEC
household tables and a small US frame.
"""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import GateResult
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime.block_location import (
    US_BLOCK_LOCATION_METADATA_KEY,
    US_BLOCK_LOCATION_RULE_ID,
    load_us_location_ladder,
    location_geography_columns,
    us_block_location_gate,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    default_congressional_district_vintage_crosswalk_path,
)
from microcosm.build.us_runtime.cps_source_geography import (
    cps_source_geography,
    household_source_keys,
    load_cps_household_geography,
)
from microcosm.build.us_runtime.geography_ladder import (
    GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
    GEOGRAPHY_LADDER_VINTAGES_ATTR,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_block_location import (
    synthetic_blocks,
    write_location_ladder,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
LEGACY_PIPELINE_SHA256 = (
    "906c9c022de6d154771d8250897a77430b0f457371b3d76a6bd046bceaccd266"
)
LEGACY_LOCATION_STAGES = (
    "congressional_district_assignment",
    "block_ladder_assignment",
)

# (source_year, H_SEQ, GESTFIPS, GTCO, GTCBSA). The synthetic ladder's
# counties are 001, 003 and sometimes 005 in states 01, 02 and 36.
_ASEC_HOUSEHOLDS = (
    (2023, 11, 1, 1, 0),
    (2023, 12, 1, 0, 0),
    (2023, 13, 2, 0, 0),
    (2023, 14, 36, 3, 0),
    (2023, 15, 36, 0, 0),
    (2023, 16, 2, 1, 0),
    (2024, 11, 1, 0, 0),
    (2024, 21, 36, 0, 0),
    (2024, 22, 2, 3, 0),
)
# Frame households: every ASEC row once (the native copy), plus PUF-support
# style copies of three rows that share their native row's source key.
_COPIED_SOURCE_ROWS = (0, 3, 6)


def _load_builder():
    path = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def _write_asec_h5s(tmp_path: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for year in sorted({row[0] for row in _ASEC_HOUSEHOLDS}):
        rows = [row for row in _ASEC_HOUSEHOLDS if row[0] == year]
        household = pd.DataFrame(
            {
                "H_SEQ": [row[1] for row in rows],
                "GESTFIPS": [row[2] for row in rows],
                "GTCO": [row[3] for row in rows],
                "GTCBSA": [row[4] for row in rows],
                "HSUP_WGT": np.full(len(rows), 1500.0),
                "H_NUMPER": np.full(len(rows), 2.0),
            }
        )
        path = tmp_path / f"census_cps_{year}.h5"
        household.to_hdf(path, key="household", format="table")
        paths[year] = path
    return paths


def _located_source_frame() -> Frame:
    rows = [*_ASEC_HOUSEHOLDS, *(_ASEC_HOUSEHOLDS[i] for i in _COPIED_SOURCE_ROWS)]
    n = len(rows)
    household_ids = np.arange(1, n + 1, dtype=np.int64) * 10
    sizes = np.array([1 + (i % 2) for i in range(n)])
    person_household = np.repeat(household_ids, sizes)
    person_row = np.repeat(np.arange(n), sizes)
    person_ids = np.arange(1, len(person_household) + 1, dtype=np.int64)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": person_ids,
                "person_household_id": person_household,
                "person_tax_unit_id": person_household + 1,
                "person_spm_unit_id": person_household + 2,
                "person_family_id": person_household + 3,
                "person_marital_unit_id": person_ids + 10**6,
                "source_year": np.array([rows[i][0] for i in person_row]),
                "source_household_id": np.array([rows[i][1] for i in person_row]),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.array([row[2] for row in rows], dtype=np.int64),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2}),
        "family": pd.DataFrame({"family_id": household_ids + 3}),
        "marital_unit": pd.DataFrame({"marital_unit_id": person_ids + 10**6}),
    }
    weights = {
        "household": Weights(
            values=np.linspace(50.0, 900.0, n), kind=WeightKind.CALIBRATED
        )
    }
    strata = pd.Series(np.where(tables["person"]["source_year"] == 2023, "a", "b"))
    return Frame(tables, US_SCHEMA, weights, strata, metadata={"stage": "test"})


def _location_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    ladder = write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    asec = _write_asec_h5s(tmp_path)
    values = {
        "location_rule": "block_v1",
        "location_seed": 11,
        "location_clones": 1,
        "block_ladder_artifact": ladder,
        "congressional_district_vintage_crosswalk": (
            default_congressional_district_vintage_crosswalk_path()
        ),
        "asec_h5": [f"{year}={path}" for year, path in sorted(asec.items())],
        "allow_geography_ladder_gate_failures": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--asec-h5",
        f"2023={tmp_path / 'census_cps_2023.h5'}",
        "--asec-h5",
        f"2024={tmp_path / 'census_cps_2024.h5'}",
        "--puf-h5",
        str(tmp_path / "puf.h5"),
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def _block_v1_argv(tmp_path: Path, *extra: str) -> list[str]:
    return _argv(
        tmp_path,
        "--location-rule",
        "block_v1",
        "--block-ladder-artifact",
        str(tmp_path / "ladder.npz"),
        "--congressional-district-vintage-crosswalk",
        str(default_congressional_district_vintage_crosswalk_path()),
        *extra,
    )


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


def test_default_args_keep_the_legacy_pipeline(builder, tmp_path: Path) -> None:
    args = builder._parse_args(
        _argv(tmp_path, "--without-block-ladder", "--checkpoint-dir", "ck")
    )

    assert args.location_rule == "legacy"
    assert builder._pipeline_for(args) is builder.OUTER_STAGE_PIPELINE
    assert builder._pipeline_steps_for(args) == builder.PIPELINE_STEPS
    assert builder.OUTER_STAGE_PIPELINE.sha256 == LEGACY_PIPELINE_SHA256
    assert builder.OUTER_STAGE_PIPELINE.names == builder.PIPELINE_STEPS
    assert set(LEGACY_LOCATION_STAGES) <= set(builder.PIPELINE_STEPS)
    assert builder.HOUSEHOLD_LOCATION_STAGE_NAME not in builder.STAGE_NAMES
    # An args object without the new attribute (older callers) is legacy too.
    assert builder._pipeline_for(SimpleNamespace()) is builder.OUTER_STAGE_PIPELINE

    config = builder._stage_run_config(args)
    assert not {key for key in config if key.startswith("location_")}
    assert "block_ladder_artifact_sha256" not in config
    child = builder._stage_cli_args(args, "source_construction")
    assert not [token for token in child if token.startswith("--location")]


def test_block_v1_pipeline_replaces_only_the_location_stages(builder) -> None:
    legacy = builder.PIPELINE_STEPS
    block_v1 = builder.BLOCK_V1_PIPELINE_STEPS
    index = legacy.index(LEGACY_LOCATION_STAGES[0])

    assert block_v1 == (
        *legacy[:index],
        "household_location_assignment",
        *legacy[index + 2 :],
    )
    assert block_v1[-1] == "final_export"
    assert dict(builder.BLOCK_V1_STAGE_BOUNDARIES)["household_location_assignment"] == (
        "with_household_us_block_location[block_v1]",
    )
    assert builder.BLOCK_V1_OUTER_STAGE_PIPELINE.names == block_v1
    assert builder.BLOCK_V1_OUTER_STAGE_PIPELINE.sha256 != LEGACY_PIPELINE_SHA256
    shared = dict(builder.STAGE_BOUNDARIES)
    for name, boundaries in builder.BLOCK_V1_STAGE_BOUNDARIES:
        if name != "household_location_assignment":
            assert shared[name] == boundaries
    args = SimpleNamespace(location_rule="block_v1")
    assert builder._pipeline_for(args) is builder.BLOCK_V1_OUTER_STAGE_PIPELINE
    assert "household_location_assignment" in builder.ALL_STAGE_NAMES


# --------------------------------------------------------------------------
# Parse rules
# --------------------------------------------------------------------------


def _parse_error(builder, argv: list[str], capsys) -> str:
    with pytest.raises(SystemExit):
        builder._parse_args(argv)
    return capsys.readouterr().err


def test_legacy_rule_refuses_location_seed_and_clones(
    builder, tmp_path: Path, capsys
) -> None:
    for extra in (("--location-seed", "3"), ("--location-clones", "2")):
        err = _parse_error(
            builder, _argv(tmp_path, "--without-block-ladder", *extra), capsys
        )
        assert "apply only to --location-rule block_v1" in err


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (("--without-block-ladder",), "contradictory"),
        (("--assign-congressional-districts",), "replaces the SOI return-count"),
        (("--ledger-facts", "facts.jsonl"), "replaces the SOI return-count"),
        (("--congressional-district-seed", "4"), "--location-seed"),
        (("--geography-ladder-seed", "4"), "--location-seed"),
        (("--location-seed", "-1"), "non-negative"),
    ),
)
def test_block_v1_refuses_legacy_location_inputs(
    builder, tmp_path: Path, capsys, extra: tuple[str, ...], message: str
) -> None:
    assert message in _parse_error(builder, _block_v1_argv(tmp_path, *extra), capsys)


def test_block_v1_requires_ladder_crosswalk_and_asec_sources(
    builder, tmp_path: Path, capsys
) -> None:
    crosswalk = str(default_congressional_district_vintage_crosswalk_path())
    ladder = str(tmp_path / "ladder.npz")
    no_ladder = _argv(
        tmp_path,
        "--location-rule",
        "block_v1",
        "--congressional-district-vintage-crosswalk",
        crosswalk,
    )
    assert "requires --block-ladder-artifact" in _parse_error(
        builder, no_ladder, capsys
    )
    no_crosswalk = _argv(
        tmp_path, "--location-rule", "block_v1", "--block-ladder-artifact", ladder
    )
    assert "requires --congressional-district-vintage-crosswalk" in _parse_error(
        builder, no_crosswalk, capsys
    )
    base_h5 = [
        "--base-h5",
        str(tmp_path / "base.h5"),
        "--puf-h5",
        str(tmp_path / "puf.h5"),
        "--out",
        str(tmp_path / "out"),
        "--location-rule",
        "block_v1",
        "--block-ladder-artifact",
        ladder,
        "--congressional-district-vintage-crosswalk",
        crosswalk,
    ]
    assert "requires --asec-h5 sources" in _parse_error(builder, base_h5, capsys)


def test_block_v1_refuses_location_clones_naming_the_route_a_checks(
    builder, tmp_path: Path, capsys
) -> None:
    err = _parse_error(
        builder, _block_v1_argv(tmp_path, "--location-clones", "3"), capsys
    )

    assert "--location-clones must be 1 on the base-h5 line" in err
    for check in (
        "support_provenance.py:support_copy_rank_series",
        "sipp_head_start.py:_recipient_predictors",
        "voluntary_filing.py:_source_receiver_rows",
        "puf_capital_gains_tail.py:assert_puf_capital_gains_tail_survives_selection",
    ):
        assert check in " ".join(err.split())


def test_block_v1_run_config_round_trips_and_locks_the_ladder(
    builder, tmp_path: Path
) -> None:
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    parent = builder._parse_args(
        _block_v1_argv(tmp_path, "--location-seed", "9", "--checkpoint-dir", "ck")
    )
    child_argv = builder._stage_cli_args(parent, "household_location_assignment")
    child = builder._parse_args(child_argv)

    parent_config = builder._stage_run_config(parent)
    assert builder._stage_run_config(child) == parent_config
    assert parent_config["location_rule"] == "block_v1"
    assert parent_config["location_seed"] == 9
    assert parent_config["location_clones"] == 1
    assert parent_config["block_ladder_artifact_sha256"] == (
        load_us_location_ladder(tmp_path / "ladder.npz").sha256
    )
    assert builder._pipeline_for(child) is builder.BLOCK_V1_OUTER_STAGE_PIPELINE


# --------------------------------------------------------------------------
# The location stage
# --------------------------------------------------------------------------


def test_block_v1_location_writes_consistent_geography_from_the_asec_row(
    builder, tmp_path: Path
) -> None:
    args = _location_args(tmp_path)
    frame = _located_source_frame()

    located, record = builder._block_v1_household_location(args, frame)

    ladder = load_us_location_ladder(args.block_ladder_artifact)
    household = located.table("household")
    assert us_block_location_gate(household, ladder).passed
    assert set(location_geography_columns(ladder)) <= set(household.columns)
    # Adds columns only: same rows, ids, source keys and weights.
    assert frame_identity(located) == frame_identity(frame)
    np.testing.assert_array_equal(
        located.weights_for("household").values, frame.weights_for("household").values
    )
    # A household with an identified ASEC county lands in that county; one
    # without lands in its state outside the year's excluded counties.
    source_year, source_household_id = household_source_keys(frame)
    geography = cps_source_geography(
        ladder,
        load_cps_household_geography(
            {
                int(year): Path(path)
                for year, path in (value.split("=", 1) for value in args.asec_h5)
            }
        ),
        source_year=source_year,
        source_household_id=source_household_id,
        state_fips=household["state_fips"].to_numpy(np.int64),
    )
    county = household["county_fips"].astype(str).to_numpy()
    for row, identified in enumerate(geography.county_fips):
        if identified:
            assert county[row] == f"{identified:05d}"
        else:
            excluded = geography.identified_counties.get(int(source_year[row]), ())
            assert int(county[row]) not in excluded
            assert county[row][:2] == f"{household['state_fips'].iloc[row]:02d}"

    assert record["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert record["location_rule"] == "block_v1"
    assert record["seed"] == 11
    assert record["clones"] == 1
    assert record["households"] == len(household)
    assert record["gate"]["passed"] is True
    assert record["candidate_set_rule"] == geography.record
    assert record["source_geography_households"]["county"] == int(
        (geography.county_fips > 0).sum()
    )
    assert record["artifact_sha256"] == ladder.sha256
    assert record["congressional_district_vintage_target"] == "119th_congress"
    assert "geography_ladder_gate" in record
    json.dumps(record)  # the record is the summary's JSON
    receipt = located.metadata[US_BLOCK_LOCATION_METADATA_KEY]
    assert receipt["seed"] == 11
    assert receipt["rule"] == US_BLOCK_LOCATION_RULE_ID

    again, _ = builder._block_v1_household_location(args, frame)
    pd.testing.assert_frame_equal(again.table("household"), household)


def test_block_v1_location_gate_failure_aborts_unless_allowed(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = GateResult(
        name="us_block_location",
        passed=False,
        failures=("tract_geoid: 1 row",),
        details={},
    )
    monkeypatch.setattr(builder, "us_block_location_gate", lambda *_a, **_k: failing)
    frame = _located_source_frame()

    with pytest.raises(SystemExit, match="us_block_location: tract_geoid"):
        builder._block_v1_household_location(
            _location_args(tmp_path, allow_geography_ladder_gate_failures=False), frame
        )
    _, record = builder._block_v1_household_location(_location_args(tmp_path), frame)
    assert record["gate"] == {
        "passed": False,
        "failures": ["tract_geoid: 1 row"],
        "details": {},
    }


def test_block_v1_refuses_a_ladder_whose_primary_plan_is_not_current(
    builder, tmp_path: Path
) -> None:
    from test_support.microcosm_build.us_block_location import ladder_metadata

    metadata = ladder_metadata()
    metadata["layers"]["congressional_district"]["vintage"] = "118th_congress"
    ladder = write_location_ladder(
        tmp_path / "old.npz", synthetic_blocks(3), metadata=metadata
    )
    args = _location_args(tmp_path)
    args.block_ladder_artifact = ladder

    with pytest.raises(SystemExit, match="118th_congress"):
        builder._block_v1_household_location(args, _located_source_frame())


def test_staged_household_location_stage_checkpoints_the_record(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _located_source_frame()
    checkpoint_dir = tmp_path / "ck"
    runtime = builder.StageRuntime(
        checkpoint_dir, builder.BLOCK_V1_OUTER_STAGE_PIPELINE, run_config={}
    )
    for stage in builder.BLOCK_V1_PIPELINE_STEPS:
        if stage == "household_location_assignment":
            break
        if stage == "primary_qrf_chain":
            runtime.complete_without_frame(stage)
        else:
            runtime.complete(stage, frame)
    monkeypatch.setattr(builder, "_stage_run_config", lambda _args: {})
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )
    args = _location_args(
        tmp_path, stage="household_location_assignment", checkpoint_dir=checkpoint_dir
    )

    builder._run_outer_stage(args)

    completed = runtime.load("household_location_assignment")
    record = runtime.metadata["household_location_assignment"]["household_location"]
    assert record["location_rule"] == "block_v1"
    assert record["seed"] == 11
    ladder = load_us_location_ladder(args.block_ladder_artifact)
    assert us_block_location_gate(completed.frame.table("household"), ladder).passed
    assert frame_identity(completed.frame) == frame_identity(frame)


def test_stage_names_are_refused_outside_their_rule(builder, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a stage of the --location-rule legacy"):
        builder._run_outer_stage(
            SimpleNamespace(
                stage="household_location_assignment", checkpoint_dir=tmp_path
            )
        )
    for stage in LEGACY_LOCATION_STAGES:
        with pytest.raises(SystemExit, match="block_v1 pipeline"):
            builder._run_outer_stage(
                SimpleNamespace(
                    stage=stage, checkpoint_dir=tmp_path, location_rule="block_v1"
                )
            )


def test_staged_export_records_rule_and_seed_in_summary_and_attrs(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location_args = _location_args(tmp_path)
    frame, record = builder._block_v1_household_location(
        location_args, _located_source_frame()
    )

    class AllSignals(dict):
        def __contains__(self, _key: object) -> bool:
            return True

        def __getitem__(self, key: str) -> object:
            return {"name": key, "passed": True}

    def fake_write(_args, _frame, path: Path) -> None:
        with h5py.File(path, "w") as h5:
            h5.attrs["fixture"] = "h5"

    monkeypatch.setattr(builder, "_ensure_capital_gains_tail_manifest", lambda _m: None)
    monkeypatch.setattr(builder, "_write_policyengine_dataset", fake_write)
    monkeypatch.setattr(
        builder, "finalize_puf_e01000_reconciliation", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(builder, "_merged_stage_signals", lambda _m: AllSignals())
    monkeypatch.setattr(builder, "_channel_weight_totals", lambda _frame: {})
    monkeypatch.setattr(builder, "_channel_output_totals", lambda _frame: {})
    monkeypatch.setattr(builder, "us_immigration_composition_summary", lambda _f: {})
    args = SimpleNamespace(
        out=tmp_path / "out",
        target_year=2024,
        seed=7,
        n_estimators=4,
        **vars(location_args),
    )
    stage_metadata = {
        "source_construction": {
            "base_source": {"kind": "pooled_asec"},
            "base_rows": {"household": 12},
            "base_household_weight_total": 1.0,
        },
        "pre_clone_enrichment": {
            "acs_h5": None,
            "acs_sha256": None,
            "acs_rent_donor_rows": None,
            "weeks_unemployed_source": {},
        },
        "clone_feature_extraction": {
            "puf_h5": "puf.h5",
            "puf_sha256": "puf-sha",
            "puf_source_year_csv": None,
            "puf_source_year_csv_sha256": None,
            "puf_donor_rows": 0,
            "puf_donor_columns": [],
            "puf_donor_build_summary": {},
            "puf_e01000_reconciliation_basis": {},
        },
        "qrf_finalization": {
            "weights_audit": {"passed": True},
            "puf_tax_detail_tail_bounds": [],
        },
        builder.PUF_CAPITAL_GAINS_TAIL_STAGE_NAME: {},
        "household_location_assignment": {
            "household_location": json.loads(json.dumps(record))
        },
    }

    result = builder._export_staged_result(args, frame, stage_metadata)

    summary = json.loads(Path(result["summary_path"]).read_text())
    assert summary["household_location"]["seed"] == 11
    assert summary["household_location"]["rule"] == US_BLOCK_LOCATION_RULE_ID
    assert summary["congressional_district_assignment"]["applied"] is False
    assert summary["geography_ladder_assignment"]["applied"] is False
    ladder = load_us_location_ladder(location_args.block_ladder_artifact)
    with h5py.File(result["output_h5"], "r") as h5:
        attrs = dict(h5.attrs)
    assert attrs["populace_location_rule"] == "block_v1"
    assert attrs["populace_location_seed"] == "11"
    assert attrs["populace_location_clones"] == "1"
    assert attrs[GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR] == ladder.sha256
    assert json.loads(attrs[GEOGRAPHY_LADDER_VINTAGES_ATTR]) == ladder.layer_vintages
    # The block-ladder attrs every block_v1 line writes, under their own names.
    assert attrs["populace_block_ladder_sha256"] == ladder.sha256
    assert (
        json.loads(attrs["populace_block_ladder_vintages"]) == ladder.layer_vintages
    )
    assert attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] == "119th_congress"
    assert (
        attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR]
        == (record["congressional_district_vintage_crosswalk_sha256"])
    )
