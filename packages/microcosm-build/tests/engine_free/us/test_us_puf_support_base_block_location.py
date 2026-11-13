"""``--location-rule block_v1`` on the PUF-support base builder (microcosm #696).

The legacy rule must stay byte-identical: same stage list, descriptions and
pipeline sha, same run config and child argv. ``block_v1`` swaps the two
legacy location stages for one ``household_location_assignment`` stage that
draws each household's 2020 block from its CPS ASEC source geography and
records the rule and seed in stage metadata, the summary and H5 root attrs.
Everything here is engine-free: a synthetic block ladder, synthetic ASEC
household tables and a small US frame.

``--cd-plan-registry PATH`` (block_v1 only) attaches the pinned block ->
CD-plan registry to the ladder before the draw. Invariants, each executed
below (the Hypothesis properties hold for every generated input):

1. Legacy is byte-identical: the flag is refused at parse time, and a legacy
   args object carrying the attribute anyway adds no argv token, run-config
   key or pre-stage check.
2. block_v1 without the flag is byte-identical: child argv, run-config keys,
   the ``household_location`` record, household columns and H5 root attrs
   equal what the builder produced before the flag existed.
3. Only the pinned registry loads, and a wrong or missing one is refused in
   ``main`` before any stage (and again by the pinned loader at the stage).
4. Every ``congressional_district_geoid__<plan>`` is the registry's district
   of the drawn block and lies in the household's state; the gate passes.
5. Attaching never moves a drawn block or any other column: the registry
   draw is the plain draw plus the three plan columns.
6. The registry digest is the loaded file's wherever it is recorded: the
   run config, the manifest record, the summary and the H5 root attr (the
   attr from the record, never from args).
7. A checkpoint binds the setting: a run with the registry cannot resume a
   directory locked without it, nor the reverse, nor one locked with other
   registry bytes; a stage whose loaded registry differs from the lock
   refuses.
8. The consumer agrees with the producer: the ACS pool line
   (``tools/_legacy/build_us_acs_multispine_base.py``), which reads this
   base as its donor, reads back exactly the registry attr written here,
   admits the base only under the registry it records, and re-derives the
   plan columns written here with the same values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import GateResult
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime.block_location import (
    POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR,
    US_BLOCK_LOCATION_METADATA_KEY,
    US_BLOCK_LOCATION_RULE_ID,
    US_CD_PLAN_COLUMN_PREFIX,
    attach_pinned_cd_plan_registry,
    derive_us_block_geography,
    load_us_location_ladder,
    location_geography_columns,
    us_block_location_gate,
)
from microcosm.build.us_runtime.cd_plan_registry import (
    PRIMARY_CD_PLAN,
    load_us_cd_plan_registry,
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
from test_support.microcosm_build.us_cd_plan_registry import (
    REGISTRY_PLAN_APPORTIONMENT,
    file_sha256,
    pin_cd_plan_registry,
    registry_ready_blocks,
    write_cd_plan_registry,
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

    # --cd-plan-registry: absent by default, and a legacy args object that
    # carries it anyway (bypassing the parse-time refusal) attaches nothing,
    # emits no child token and runs no pre-stage check on it.
    assert args.cd_plan_registry is None
    assert not {key for key in config if key.startswith("cd_plan_registry")}
    assert "--cd-plan-registry" not in child
    carrying = argparse.Namespace(
        **{**vars(args), "cd_plan_registry": tmp_path / "missing.npz"}
    )
    assert builder._cd_plan_registry_path(carrying) is None
    assert builder._stage_cli_args(carrying, "source_construction") == child
    builder._verify_cd_plan_registry_pin(carrying)
    assert builder._cd_plan_registry_path(SimpleNamespace()) is None


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
    assert json.loads(attrs["populace_block_ladder_vintages"]) == ladder.layer_vintages
    assert attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] == "119th_congress"
    assert (
        attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR]
        == (record["congressional_district_vintage_crosswalk_sha256"])
    )


# --------------------------------------------------------------------------
# --cd-plan-registry: the pinned block -> CD-plan registry
# --------------------------------------------------------------------------

_NON_CURRENT_PLANS = tuple(
    sorted(plan for plan in REGISTRY_PLAN_APPORTIONMENT if plan != PRIMARY_CD_PLAN)
)
_PLAN_COLUMNS = tuple(
    f"{US_CD_PLAN_COLUMN_PREFIX}{plan}" for plan in _NON_CURRENT_PLANS
)
_REGISTRY_CONFIG_KEYS = frozenset({"cd_plan_registry", "cd_plan_registry_sha256"})

# What the builder produced for block_v1 before --cd-plan-registry existed,
# captured from the unmodified tool on this fixture (the record, household
# columns and attrs from _location_args; the argv and run config from
# _block_v1_argv with --location-seed 9). Without the flag they must not move.
_PRE_REGISTRY_BLOCK_V1_CONFIG_KEYS = frozenset(
    {
        "acs_h5",
        "allow_geography_ladder_gate_failures",
        "asec_2023_weeks_unemployed_source",
        "asec_education_source",
        "asec_h5",
        "asec_h5_sha256",
        "asec_max_households",
        "assign_congressional_districts",
        "base_h5",
        "block_ladder_artifact",
        "block_ladder_artifact_sha256",
        "builder_code_identity",
        "congressional_district_seed",
        "congressional_district_vintage_crosswalk",
        "equivalence_deterministic_h5_metadata",
        "geography_ladder_seed",
        "ledger_facts",
        "location_clones",
        "location_rule",
        "location_seed",
        "n_estimators",
        "out",
        "puf_h5",
        "puf_h5_sha256",
        "puf_source_year_csv",
        "puf_source_year_csv_sha256",
        "seed",
        "support_spine_spec",
        "target_year",
        "thread_environment",
        "without_block_ladder",
    }
)
_PRE_REGISTRY_RECORD_KEYS = frozenset(
    {
        "applied",
        "artifact",
        "artifact_sha256",
        "asec_h5",
        "block_ladder",
        "candidate_set_rule",
        "clones",
        "congressional_district_vintage_crosswalk",
        "congressional_district_vintage_crosswalk_sha256",
        "congressional_district_vintage_target",
        "coverage",
        "gate",
        "geography_columns",
        "geography_ladder_gate",
        "households",
        "location_rule",
        "location_weight_total",
        "rows",
        "rule",
        "sampling_basis",
        "seed",
        "source_geography_households",
        "uniforms",
    }
)
_PRE_REGISTRY_BLOCK_LADDER_KEYS = frozenset(
    {
        "blocks",
        "carries_puma",
        "congressional_district_plans",
        "layer_vintages",
        "primary_congressional_district_plan",
        "sha256",
    }
)
_PRE_REGISTRY_HOUSEHOLD_COLUMNS = [
    "household_id",
    "state_fips",
    "block_geoid",
    "tract_geoid",
    "county_fips",
    "place_fips",
    "sldu",
    "sldl",
    "cbsa_code",
    "puma",
    "congressional_district_geoid",
]
_PRE_REGISTRY_ATTR_KEYS = [
    "populace_congressional_district_vintage_crosswalk_sha256",
    "populace_congressional_district_vintage_target",
    "populace_geography_ladder_artifact_sha256",
    "populace_geography_ladder_vintages",
    "populace_block_ladder_sha256",
    "populace_block_ladder_vintages",
    "populace_location_rule",
    "populace_location_seed",
    "populace_location_clones",
]


def _pre_registry_block_v1_child_argv(tmp_path: Path) -> list[str]:
    """The pre-flag child argv of ``_block_v1_argv(tmp_path, --location-seed 9,
    --checkpoint-dir tmp_path/ck)`` for the location stage."""

    return [
        "--asec-h5",
        f"2023={tmp_path / 'census_cps_2023.h5'}",
        "--asec-h5",
        f"2024={tmp_path / 'census_cps_2024.h5'}",
        "--target-year",
        "2024",
        "--puf-h5",
        str(tmp_path / "puf.h5"),
        "--out",
        str(tmp_path / "out"),
        "--stage",
        "household_location_assignment",
        "--checkpoint-dir",
        str(tmp_path / "ck"),
        "--seed",
        "0",
        "--n-estimators",
        "32",
        "--congressional-district-vintage-crosswalk",
        str(default_congressional_district_vintage_crosswalk_path()),
        "--congressional-district-seed",
        "0",
        "--block-ladder-artifact",
        str(tmp_path / "ladder.npz"),
        "--geography-ladder-seed",
        "0",
        "--location-rule",
        "block_v1",
        "--location-seed",
        "9",
        "--location-clones",
        "1",
    ]


def _write_registry_ladder(directory: Path, seed: int = 3) -> tuple[Path, Path]:
    """A registry-compatible ladder and a registry covering it (unpinned).

    ``synthetic_blocks`` districts break the apportionment rosters, so the
    ladder's own (119th) districts are made valid first; its blocks, states
    and counties are unchanged, so the ASEC fixture households still land.
    """

    arrays = registry_ready_blocks(synthetic_blocks(seed))
    ladder = write_location_ladder(directory / "ready_ladder.npz", arrays)
    registry = write_cd_plan_registry(directory / "registry.npz", arrays)
    return ladder, registry


def _registry_location_args(
    tmp_path: Path, patch: pytest.MonkeyPatch, **overrides
) -> tuple[SimpleNamespace, str]:
    """Location args on a registry-ready ladder with the pinned registry."""

    ladder, registry = _write_registry_ladder(tmp_path)
    digest = pin_cd_plan_registry(patch, registry)
    args = _location_args(
        tmp_path, block_ladder_artifact=ladder, cd_plan_registry=registry, **overrides
    )
    return args, digest


def _registry_lookup(registry_path: Path, household: pd.DataFrame):
    """Each plan's registry district of every household's drawn block."""

    registry = load_us_cd_plan_registry(registry_path)
    blocks = household["block_geoid"].astype(np.int64).to_numpy()
    index = np.searchsorted(registry.block_geoid, blocks)
    assert (registry.block_geoid[index] == blocks).all()
    return {plan: registry.plans[plan][index] for plan in registry.plans}


def _assert_registry_draw_is_the_plain_draw_plus_lookups(
    builder,
    plain: object,
    plain_record: dict,
    located: object,
    record: dict,
    *,
    registry_path: Path,
    digest: str,
) -> None:
    """Invariants 4-6 for one draw with and one without the registry."""

    household = located.table("household")
    before = plain.table("household")
    # 5. No redraw: the plain draw's columns are unchanged; exactly the
    # non-current plan columns are added, after them.
    assert list(household.columns) == [*before.columns, *_PLAN_COLUMNS]
    pd.testing.assert_frame_equal(household[list(before.columns)], before)
    for entity in located.entities:
        if entity != "household":
            pd.testing.assert_frame_equal(located.table(entity), plain.table(entity))
    assert frame_identity(located) == frame_identity(plain)
    np.testing.assert_array_equal(
        located.weights_for("household").values, plain.weights_for("household").values
    )
    # 4. Lookup: each plan column is the registry's district of the drawn
    # block, in the household's state; the current plan is the ladder's.
    lookup = _registry_lookup(registry_path, household)
    state = household["state_fips"].to_numpy(np.int64)
    for plan, districts in lookup.items():
        column = record["cd_plan_registry"]["columns"][plan]
        written = household[column].to_numpy(np.int64)
        np.testing.assert_array_equal(written, districts, err_msg=plan)
        assert (written // 100 == state).all(), plan
    assert record["cd_plan_registry"]["columns"][PRIMARY_CD_PLAN] == (
        "congressional_district_geoid"
    )
    assert record["gate"]["passed"] is True
    assert record["gate"]["failures"] == plain_record["gate"]["failures"]
    # 6. The recorded digest is the loaded file's, and the attr is the record's.
    assert record["cd_plan_registry"]["sha256"] == digest == file_sha256(registry_path)
    attrs: dict[str, object] = {}
    plain_attrs: dict[str, object] = {}
    builder._write_block_v1_location_attrs(attrs, json.loads(json.dumps(record)))
    builder._write_block_v1_location_attrs(
        plain_attrs, json.loads(json.dumps(plain_record))
    )
    assert attrs == {**plain_attrs, POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: digest}
    # The record is the plain record plus the registry: the ladder's digest,
    # the draw's counts and the candidate sets do not move.
    assert "cd_plan_registry" not in plain_record
    assert set(record) == {*plain_record, "cd_plan_registry"}
    for key in set(plain_record) - {"block_ladder", "geography_columns", "gate"}:
        assert record[key] == plain_record[key], key
    assert record["geography_columns"] == [
        *plain_record["geography_columns"],
        *_PLAN_COLUMNS,
    ]
    ladder_record = dict(record["block_ladder"])
    plain_ladder_record = dict(plain_record["block_ladder"])
    assert set(ladder_record.pop("congressional_district_plans")) == set(
        REGISTRY_PLAN_APPORTIONMENT
    )
    plain_ladder_record.pop("congressional_district_plans")
    assert ladder_record == plain_ladder_record


def _stage_metadata_for_export(builder, record: dict) -> dict[str, dict[str, object]]:
    """The completed-stage metadata ``_export_staged_result`` reads."""

    return {
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


def _patch_staged_export(builder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub everything ``_export_staged_result`` needs besides location."""

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


def _staged_export(
    builder, tmp_path: Path, location_args: SimpleNamespace, frame, record: dict
) -> tuple[dict, dict, dict]:
    args = SimpleNamespace(
        out=tmp_path / "out",
        target_year=2024,
        seed=7,
        n_estimators=4,
        **vars(location_args),
    )
    result = builder._export_staged_result(
        args, frame, _stage_metadata_for_export(builder, record)
    )
    summary = json.loads(Path(result["summary_path"]).read_text())
    with h5py.File(result["output_h5"], "r") as h5:
        attrs = dict(h5.attrs)
    return result, summary, attrs


def test_legacy_rule_refuses_the_cd_plan_registry(
    builder, tmp_path: Path, capsys
) -> None:
    registry = ("--cd-plan-registry", str(tmp_path / "registry.npz"))
    err = _parse_error(
        builder, _argv(tmp_path, "--without-block-ladder", *registry), capsys
    )
    assert "--cd-plan-registry applies only to --location-rule block_v1" in err
    # The existing legacy refusals keep their texts and their precedence.
    err = _parse_error(
        builder,
        _argv(tmp_path, "--without-block-ladder", "--location-seed", "3", *registry),
        capsys,
    )
    error_line = err.strip().splitlines()[-1]
    assert "--location-seed and --location-clones apply only to" in error_line
    assert "--cd-plan-registry" not in error_line


def test_block_v1_without_the_registry_matches_the_pre_registry_contract(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    parent = builder._parse_args(
        _block_v1_argv(
            tmp_path, "--location-seed", "9", "--checkpoint-dir", str(tmp_path / "ck")
        )
    )
    assert parent.cd_plan_registry is None
    child_argv = builder._stage_cli_args(parent, "household_location_assignment")
    assert child_argv == _pre_registry_block_v1_child_argv(tmp_path)
    config = builder._stage_run_config(parent)
    assert set(config) == _PRE_REGISTRY_BLOCK_V1_CONFIG_KEYS

    frame = _located_source_frame()
    args = _location_args(tmp_path, cd_plan_registry=None)
    located, record = builder._block_v1_household_location(args, frame)
    assert set(record) == _PRE_REGISTRY_RECORD_KEYS
    assert set(record["block_ladder"]) == _PRE_REGISTRY_BLOCK_LADDER_KEYS
    assert set(record["block_ladder"]["congressional_district_plans"]) == {
        PRIMARY_CD_PLAN
    }
    household_columns = list(located.table("household").columns)
    assert household_columns == _PRE_REGISTRY_HOUSEHOLD_COLUMNS
    # An args object without the attribute (older callers) is the same run.
    older = _location_args(tmp_path)
    assert not hasattr(older, "cd_plan_registry")
    located_older, record_older = builder._block_v1_household_location(older, frame)
    assert record_older == record
    pd.testing.assert_frame_equal(
        located_older.table("household"), located.table("household")
    )
    attrs: dict[str, object] = {}
    builder._write_block_v1_location_attrs(attrs, json.loads(json.dumps(record)))
    assert list(attrs) == _PRE_REGISTRY_ATTR_KEYS

    _patch_staged_export(builder, monkeypatch)
    _, summary, h5_attrs = _staged_export(builder, tmp_path, args, located, record)
    assert "cd_plan_registry" not in summary["household_location"]
    assert set(h5_attrs) == {"fixture", *_PRE_REGISTRY_ATTR_KEYS}
    assert not any(
        str(column).startswith(US_CD_PLAN_COLUMN_PREFIX) for column in household_columns
    )


def test_block_v1_cd_plan_registry_round_trips_and_locks_its_digest(
    builder, tmp_path: Path
) -> None:
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    _, registry = _write_registry_ladder(tmp_path)
    common = ("--location-seed", "9", "--checkpoint-dir", str(tmp_path / "ck"))
    plain = builder._parse_args(_block_v1_argv(tmp_path, *common))
    parent = builder._parse_args(
        _block_v1_argv(tmp_path, *common, "--cd-plan-registry", str(registry))
    )
    child_argv = builder._stage_cli_args(parent, "household_location_assignment")
    child = builder._parse_args(child_argv)

    # The child re-derives the parent's locked config exactly.
    parent_config = builder._stage_run_config(parent)
    assert builder._stage_run_config(child) == parent_config
    assert child.cd_plan_registry == registry
    assert parent_config["cd_plan_registry"] == str(registry.resolve())
    assert parent_config["cd_plan_registry_sha256"] == file_sha256(registry)
    # The flag adds exactly its two tokens and its two locked keys.
    plain_argv = builder._stage_cli_args(plain, "household_location_assignment")
    flag = child_argv.index("--cd-plan-registry")
    assert child_argv[flag + 1] == str(registry)
    assert [*child_argv[:flag], *child_argv[flag + 2 :]] == plain_argv
    plain_config = builder._stage_run_config(plain)
    assert set(parent_config) == set(plain_config) | _REGISTRY_CONFIG_KEYS
    assert {
        key: value
        for key, value in parent_config.items()
        if key not in _REGISTRY_CONFIG_KEYS
    } == plain_config


def test_block_v1_location_attaches_the_pinned_registry(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, digest = _registry_location_args(tmp_path, monkeypatch)
    frame = _located_source_frame()

    located, record = builder._block_v1_household_location(args, frame)
    plain, plain_record = builder._block_v1_household_location(
        _location_args(tmp_path, block_ladder_artifact=args.block_ladder_artifact),
        frame,
    )

    _assert_registry_draw_is_the_plain_draw_plus_lookups(
        builder,
        plain,
        plain_record,
        located,
        record,
        registry_path=args.cd_plan_registry,
        digest=digest,
    )
    registry_record = record["cd_plan_registry"]
    assert registry_record["artifact"] == str(args.cd_plan_registry.resolve())
    assert registry_record["plans"] == sorted(REGISTRY_PLAN_APPORTIONMENT)
    assert registry_record["primary_plan_check"] == {
        "plan": PRIMARY_CD_PLAN,
        "ladder_blocks": len(load_us_location_ladder(args.block_ladder_artifact)),
        "blocks_disagreeing": 0,
    }
    assert registry_record["known_deviations"] == {
        "117th_congress": ["37"],
        "120th_congress": ["29"],
    }
    # The drawn frame carries the ladder's digest, not the registry's.
    assert (
        located.metadata[US_BLOCK_LOCATION_METADATA_KEY]["ladder_sha256"]
        == (record["artifact_sha256"])
    )
    json.dumps(record)


def test_the_location_stage_refuses_a_registry_it_cannot_attach(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _located_source_frame()
    ladder, registry = _write_registry_ladder(tmp_path)

    # Unpinned: the stage's own pinned load refuses it (main refuses earlier).
    args = _location_args(
        tmp_path, block_ladder_artifact=ladder, cd_plan_registry=registry
    )
    with pytest.raises(SystemExit, match="--cd-plan-registry: .*SHA-256"):
        builder._block_v1_household_location(args, frame)
    with pytest.raises(SystemExit, match="--cd-plan-registry: .*not found"):
        builder._block_v1_household_location(
            _location_args(
                tmp_path,
                block_ladder_artifact=ladder,
                cd_plan_registry=tmp_path / "missing.npz",
            ),
            frame,
        )

    # Pinned, but its current plan disagrees with the ladder on one block:
    # the differential refuses rather than overriding the ladder.
    pin_cd_plan_registry(monkeypatch, registry)
    arrays = registry_ready_blocks(synthetic_blocks(3))
    new_york = np.flatnonzero(arrays["block_geoid"] // 10**13 == 36)
    moved = dict(arrays)
    moved["congressional_district_geoid"] = arrays[
        "congressional_district_geoid"
    ].copy()
    current = int(moved["congressional_district_geoid"][new_york[0]])
    moved["congressional_district_geoid"][new_york[0]] = (
        3602 if current == 3601 else 3601
    )
    disagreeing = write_location_ladder(tmp_path / "disagreeing.npz", moved)
    with pytest.raises(SystemExit, match="disagrees with the ladder's own copy on 1"):
        builder._block_v1_household_location(
            _location_args(
                tmp_path,
                block_ladder_artifact=disagreeing,
                cd_plan_registry=registry,
            ),
            frame,
        )


@pytest.mark.parametrize(
    "extra",
    (
        (),
        ("--checkpoint-dir", "{ck}"),
        ("--stage", "household_location_assignment", "--checkpoint-dir", "{ck}"),
    ),
    ids=("monolithic", "staged", "named-stage"),
)
def test_main_refuses_an_unpinned_registry_before_any_stage(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra
) -> None:
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    _, registry = _write_registry_ladder(tmp_path)
    checkpoint_dir = tmp_path / "ck"
    extra = tuple(value.format(ck=checkpoint_dir) for value in extra)
    started: list[str] = []
    for name in ("_run_all", "_run_staged_all", "_run_configured_stage"):
        monkeypatch.setattr(
            builder, name, lambda *_a, _name=name, **_k: started.append(_name)
        )
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    def run(path: Path) -> None:
        builder.main(_block_v1_argv(tmp_path, *extra, "--cd-plan-registry", str(path)))

    with pytest.raises(SystemExit, match="expected the pinned"):
        run(registry)
    with pytest.raises(SystemExit, match="not found"):
        run(tmp_path / "missing.npz")
    assert started == []
    assert not checkpoint_dir.exists()

    # The packaged pin admits exactly its file, and the run proceeds.
    pin_cd_plan_registry(monkeypatch, registry)
    run(registry)
    assert len(started) == 1


def test_the_staged_location_stage_records_the_registry_and_checks_the_lock(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _located_source_frame()
    monkeypatch.setattr(
        builder, "profile_stage", lambda *_args, **_kwargs: nullcontext()
    )

    def prepared(checkpoint_dir: Path, locked: str) -> object:
        runtime = builder.StageRuntime(
            checkpoint_dir,
            builder.BLOCK_V1_OUTER_STAGE_PIPELINE,
            run_config={"cd_plan_registry_sha256": locked},
        )
        for stage in builder.BLOCK_V1_PIPELINE_STEPS:
            if stage == "household_location_assignment":
                break
            if stage == "primary_qrf_chain":
                runtime.complete_without_frame(stage)
            else:
                runtime.complete(stage, frame)
        return runtime

    args, digest = _registry_location_args(
        tmp_path, monkeypatch, stage="household_location_assignment"
    )

    # A lock naming other registry bytes refuses after the load, and the
    # stage is not completed.
    other = "f" * 64
    monkeypatch.setattr(
        builder, "_stage_run_config", lambda _args: {"cd_plan_registry_sha256": other}
    )
    stale = prepared(tmp_path / "stale", other)
    args.checkpoint_dir = tmp_path / "stale"
    with pytest.raises(SystemExit, match=f"sha256 {digest}, but the run config"):
        builder._run_outer_stage(args)
    assert "household_location_assignment" not in stale.context.completed

    monkeypatch.setattr(
        builder,
        "_stage_run_config",
        lambda _args: {"cd_plan_registry_sha256": digest},
    )
    runtime = prepared(tmp_path / "ck", digest)
    args.checkpoint_dir = tmp_path / "ck"
    builder._run_outer_stage(args)

    completed = runtime.load("household_location_assignment")
    record = runtime.metadata["household_location_assignment"]["household_location"]
    assert record["cd_plan_registry"]["sha256"] == digest
    assert set(_PLAN_COLUMNS) <= set(completed.frame.table("household").columns)
    assert frame_identity(completed.frame) == frame_identity(frame)


@pytest.mark.parametrize("loaded", (None, "a" * 64, "b" * 64))
@pytest.mark.parametrize("locked", ("absent", None, "a" * 64, "b" * 64))
def test_the_registry_lock_check_admits_only_equal_digests(
    builder, locked, loaded
) -> None:
    run_config = {} if locked == "absent" else {"cd_plan_registry_sha256": locked}
    record = {} if loaded is None else {"cd_plan_registry": {"sha256": loaded}}
    expected = None if locked == "absent" else locked

    if expected == loaded:
        builder._require_cd_plan_registry_matches_lock(record, run_config)
    else:
        with pytest.raises(SystemExit, match="the run config locked"):
            builder._require_cd_plan_registry_matches_lock(record, run_config)


def test_a_checkpoint_binds_its_registry_setting(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Code identity is orthogonal here (and the same for every config).
    monkeypatch.setattr(builder, "_builder_code_identity", lambda: {"fixture": 1})
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    arrays = registry_ready_blocks(synthetic_blocks(3))
    registry = write_cd_plan_registry(tmp_path / "registry.npz", arrays)
    common = ("--checkpoint-dir", str(tmp_path / "ck"))
    argv = {
        "none": _block_v1_argv(tmp_path, *common),
        "registry": _block_v1_argv(
            tmp_path, *common, "--cd-plan-registry", str(registry)
        ),
    }
    configs = {
        "none": builder._stage_run_config(builder._parse_args(argv["none"])),
        "registry": builder._stage_run_config(builder._parse_args(argv["registry"])),
    }
    # The same path rewritten in place with other bytes is another setting.
    write_cd_plan_registry(registry, arrays, known_deviations=False)
    configs["rewritten"] = builder._stage_run_config(
        builder._parse_args(argv["registry"])
    )
    assert (
        configs["rewritten"]["cd_plan_registry_sha256"]
        != configs["registry"]["cd_plan_registry_sha256"]
    )

    for first, first_config in configs.items():
        for second, second_config in configs.items():
            checkpoint_dir = tmp_path / f"{first}-{second}"
            builder.StageRuntime(
                checkpoint_dir,
                builder.BLOCK_V1_OUTER_STAGE_PIPELINE,
                run_config=first_config,
            )
            if first == second:
                builder.StageRuntime(
                    checkpoint_dir,
                    builder.BLOCK_V1_OUTER_STAGE_PIPELINE,
                    run_config=second_config,
                )
            else:
                with pytest.raises(ValueError, match="run_config differs"):
                    builder.StageRuntime(
                        checkpoint_dir,
                        builder.BLOCK_V1_OUTER_STAGE_PIPELINE,
                        run_config=second_config,
                    )

    # The staged driver refuses before starting any child, both ways.
    def no_child(*_args, **_kwargs):
        raise AssertionError("a stage child started")

    monkeypatch.setattr(builder.subprocess, "run", no_child)
    for locked, resumed in (("none", "registry"), ("rewritten", "none")):
        checkpoint_dir = tmp_path / f"driver-{locked}"
        builder.StageRuntime(
            checkpoint_dir,
            builder.BLOCK_V1_OUTER_STAGE_PIPELINE,
            run_config=configs[locked],
        )
        args = builder._parse_args(argv[resumed])
        args.checkpoint_dir = checkpoint_dir
        with pytest.raises(ValueError, match="run_config differs"):
            builder._run_staged_all(args)


def test_staged_export_records_the_registry_in_summary_and_attrs(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location_args, digest = _registry_location_args(tmp_path, monkeypatch)
    frame, record = builder._block_v1_household_location(
        location_args, _located_source_frame()
    )
    _patch_staged_export(builder, monkeypatch)

    result, summary, attrs = _staged_export(
        builder, tmp_path, location_args, frame, record
    )

    assert summary["household_location"]["cd_plan_registry"] == json.loads(
        json.dumps(record["cd_plan_registry"])
    )
    assert summary["household_location"]["cd_plan_registry"]["sha256"] == digest
    assert attrs[POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR] == digest
    assert set(attrs) == {
        "fixture",
        *_PRE_REGISTRY_ATTR_KEYS,
        POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR,
    }
    assert summary["congressional_district_assignment"]["applied"] is False

    # The attr and summary are functions of the stage record, never of args,
    # so the final-export repair regenerates them from stage metadata alone.
    for elsewhere in (None, tmp_path / "elsewhere.npz"):
        location_args.cd_plan_registry = elsewhere
        again, summary_again, attrs_again = _staged_export(
            builder, tmp_path, location_args, frame, record
        )
        assert again["summary_sha256"] == result["summary_sha256"]
        assert again["output_sha256"] == result["output_sha256"]
        assert summary_again == summary
        assert attrs_again == attrs


# --------------------------------------------------------------------------
# Producer -> consumer: the ACS pool line reads this base as its donor
# --------------------------------------------------------------------------


def _load_acs_pool_builder():
    """``tools/_legacy/build_us_acs_multispine_base.py``, which reads this
    line's base H5 as its donor (``--base-h5``)."""

    path = _TEST_PATHS.repository / "tools" / "_legacy"
    path = path / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "acs_pool_consumer_of_the_puf_support_base", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acs_pool_builder():
    pytest.importorskip("tables")  # the consumer reads root attrs with PyTables
    return _load_acs_pool_builder()


def _base_root_attrs(builder, path: Path, record: dict) -> Path:
    """An H5 whose root attrs are exactly what this line writes for ``record``."""

    with h5py.File(path, "w") as h5:
        builder._write_block_v1_location_attrs(h5.attrs, json.loads(json.dumps(record)))
    return path


def test_the_acs_pool_line_admits_this_base_only_under_its_registry(
    builder, acs_pool_builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The donor contract, run on both sides: this line writes the attr and
    the plan columns; the ACS pool line reads them back and decides."""

    args, digest = _registry_location_args(tmp_path, monkeypatch)
    frame = _located_source_frame()
    located, record = builder._block_v1_household_location(args, frame)
    plain, plain_record = builder._block_v1_household_location(
        _location_args(tmp_path, block_ladder_artifact=args.block_ladder_artifact),
        frame,
    )
    plain_ladder = load_us_location_ladder(args.block_ladder_artifact)
    attached, registry_record = attach_pinned_cd_plan_registry(
        plain_ladder, args.cd_plan_registry
    )
    other = {**registry_record, "sha256": "f" * 64}

    # The attr: read back through PyTables exactly as this line wrote it.
    read = acs_pool_builder._base_location_rule(
        _base_root_attrs(builder, tmp_path / "registry_base.h5", record)
    )
    assert read == {
        "location_rule": "block_v1",
        "recorded": True,
        "attributes": {
            "populace_location_rule": "block_v1",
            "populace_location_seed": str(args.location_seed),
            "populace_location_clones": "1",
            POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: digest,
        },
    }
    acs_pool_builder._require_base_cd_plan_registry(read, registry_record)
    for wrong in (None, other):
        with pytest.raises(SystemExit, match="located with CD-plan registry"):
            acs_pool_builder._require_base_cd_plan_registry(read, wrong)
    # A base built without the flag records nothing, so either run admits it.
    plain_read = acs_pool_builder._base_location_rule(
        _base_root_attrs(builder, tmp_path / "plain_base.h5", plain_record)
    )
    assert POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR not in plain_read["attributes"]
    assert read["attributes"] == {
        **plain_read["attributes"],
        POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR: digest,
    }
    for registry in (None, registry_record, other):
        acs_pool_builder._require_base_cd_plan_registry(plain_read, registry)

    # The columns: the pool re-derives every location column from the block;
    # under the same registry it rewrites the plan columns this line wrote
    # with the same values, and without it it refuses them by name.
    acs_pool_builder._refuse_unwritten_base_cd_plan_columns(located, attached)
    for ladder in (plain_ladder, None):
        with pytest.raises(SystemExit, match="congressional_district_geoid__"):
            acs_pool_builder._refuse_unwritten_base_cd_plan_columns(located, ladder)
        acs_pool_builder._refuse_unwritten_base_cd_plan_columns(plain, ladder)
    household = located.table("household")
    index = np.searchsorted(
        attached.block_geoid, household["block_geoid"].astype(np.int64).to_numpy()
    )
    derived = derive_us_block_geography(attached, index)
    assert set(location_geography_columns(attached)) <= set(household.columns)
    for column in location_geography_columns(attached):
        np.testing.assert_array_equal(
            household[column].to_numpy(), derived[column], err_msg=column
        )


# --------------------------------------------------------------------------
# Properties (Hypothesis is a workspace dependency; skipped without it)
# --------------------------------------------------------------------------


def test_property_the_registry_draw_is_the_plain_draw_plus_lookups(
    builder, tmp_path: Path
) -> None:
    """Invariants 4-6 for every ladder and location seed Hypothesis draws."""

    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")
    frame = _located_source_frame()
    asec = _write_asec_h5s(tmp_path)
    crosswalk = default_congressional_district_vintage_crosswalk_path()

    @hypothesis.settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[hypothesis.HealthCheck.too_slow],
    )
    @hypothesis.given(
        ladder_seed=st.integers(0, 10_000), location_seed=st.integers(0, 2**31)
    )
    def check(ladder_seed: int, location_seed: int) -> None:
        with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
            ladder, registry = _write_registry_ladder(Path(directory), ladder_seed)
            digest = pin_cd_plan_registry(patch, registry)
            values = {
                "location_rule": "block_v1",
                "location_seed": location_seed,
                "location_clones": 1,
                "block_ladder_artifact": ladder,
                "congressional_district_vintage_crosswalk": crosswalk,
                "asec_h5": [f"{year}={path}" for year, path in sorted(asec.items())],
                "allow_geography_ladder_gate_failures": True,
            }
            plain, plain_record = builder._block_v1_household_location(
                SimpleNamespace(**values), frame
            )
            located, record = builder._block_v1_household_location(
                SimpleNamespace(**values, cd_plan_registry=registry), frame
            )
            _assert_registry_draw_is_the_plain_draw_plus_lookups(
                builder,
                plain,
                plain_record,
                located,
                record,
                registry_path=registry,
                digest=digest,
            )

    check()


def test_property_argv_and_run_config_carry_the_registry_iff_set(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For any rule, seed, registry path and bytes: legacy refuses the flag;
    block_v1 forwards it to every child and locks its path and digest, and
    without it adds nothing; the child always re-derives the parent config."""

    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")
    monkeypatch.setattr(builder, "_builder_code_identity", lambda: {"fixture": 1})
    write_location_ladder(tmp_path / "ladder.npz", synthetic_blocks(3))
    plain_config = builder._stage_run_config(
        builder._parse_args(_block_v1_argv(tmp_path))
    )
    names = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_- .", min_size=1, max_size=12
    ).filter(lambda name: name.strip(" .") == name and name not in {".", ".."})

    @hypothesis.settings(max_examples=40, deadline=None)
    @hypothesis.given(
        block_v1=st.booleans(),
        with_registry=st.booleans(),
        location_seed=st.integers(0, 2**31),
        name=names,
        content=st.binary(min_size=0, max_size=64),
        stage=st.sampled_from(builder.BLOCK_V1_PIPELINE_STEPS),
    )
    def check(block_v1, with_registry, location_seed, name, content, stage):
        registry = tmp_path / "registries" / f"{name}.npz"
        registry.parent.mkdir(exist_ok=True)
        registry.write_bytes(content)
        extra = ("--cd-plan-registry", str(registry)) if with_registry else ()
        checkpoint = ("--checkpoint-dir", str(tmp_path / "ck"))
        if not block_v1:
            argv = _argv(tmp_path, "--without-block-ladder", *checkpoint, *extra)
            if with_registry:
                with pytest.raises(SystemExit):
                    builder._parse_args(argv)
                return
            args = builder._parse_args(argv)
            config = builder._stage_run_config(args)
            assert not set(config) & _REGISTRY_CONFIG_KEYS
            assert "--cd-plan-registry" not in builder._stage_cli_args(args, stage)
            return
        parent = builder._parse_args(
            _block_v1_argv(
                tmp_path, "--location-seed", str(location_seed), *checkpoint, *extra
            )
        )
        child_argv = builder._stage_cli_args(parent, stage)
        child = builder._parse_args(child_argv)
        config = builder._stage_run_config(parent)
        assert builder._stage_run_config(child) == config
        assert child_argv.count("--cd-plan-registry") == int(with_registry)
        if with_registry:
            assert set(config) == set(plain_config) | _REGISTRY_CONFIG_KEYS
            assert config["cd_plan_registry"] == str(registry.resolve())
            assert config["cd_plan_registry_sha256"] == file_sha256(registry)
        else:
            assert set(config) == set(plain_config)
        assert {
            key: value
            for key, value in config.items()
            if key not in _REGISTRY_CONFIG_KEYS | {"location_seed"}
        } == {
            key: value for key, value in plain_config.items() if key != "location_seed"
        }

    check()


def test_property_the_acs_pool_line_reads_back_the_digest_this_base_records(
    builder, acs_pool_builder, tmp_path: Path
) -> None:
    """For any recorded registry digest (or none), location seed and run
    registry (or none): the ACS pool line reads back exactly the attrs this
    line wrote from the record, and admits the base iff it recorded no
    registry or the run attaches the one it recorded."""

    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")
    _, plain_record = builder._block_v1_household_location(
        _location_args(tmp_path), _located_source_frame()
    )
    digests = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)
    path = tmp_path / "base.h5"

    @hypothesis.settings(max_examples=30, deadline=None)
    @hypothesis.given(
        recorded=st.none() | digests,
        attached=st.none() | digests,
        same_registry=st.booleans(),
        seed=st.integers(0, 2**31),
    )
    def check(recorded, attached, same_registry, seed):
        if same_registry:
            attached = recorded
        record = {**plain_record, "seed": seed}
        if recorded is not None:
            record["cd_plan_registry"] = {"sha256": recorded, "plans": ["x"]}
        written: dict[str, object] = {}
        builder._write_block_v1_location_attrs(written, record)
        read = acs_pool_builder._base_location_rule(
            _base_root_attrs(builder, path, record)
        )
        assert read["location_rule"] == "block_v1"
        assert read["attributes"] == {
            key: str(value)
            for key, value in written.items()
            if key
            in {
                "populace_location_rule",
                "populace_location_seed",
                "populace_location_clones",
                POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR,
            }
        }
        assert read["attributes"]["populace_location_seed"] == str(seed)
        assert read["attributes"].get(POPULACE_CD_PLAN_REGISTRY_SHA256_ATTR) == (
            recorded
        )
        run_record = None if attached is None else {"sha256": attached}
        if recorded is None or recorded == attached:
            acs_pool_builder._require_base_cd_plan_registry(read, run_record)
        else:
            with pytest.raises(SystemExit, match="located with CD-plan registry"):
                acs_pool_builder._require_base_cd_plan_registry(read, run_record)

    check()
