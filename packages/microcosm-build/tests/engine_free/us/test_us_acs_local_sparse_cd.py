"""The ACS local release's sparse target matrix and ``state_cd`` surface.

The central test is a differential: the fake-engine fixture is materialized
once the old way (every target a dense float32 household column, compiled by
the calibrate kernel from frame columns) and once through the sparse path
(chunked carrier split -> CSR checkpoint -> callable rows). Both must give
the kernel the identical constraint matrix and the identical calibrated
weights. The dense construction lives only here.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.calibrate import TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.calibrate.target import TargetSet
from microcosm.frame import US_SCHEMA, Frame
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
requires_pytables = pytest.mark.skipif(
    importlib.util.find_spec("tables") is None,
    reason="requires pytables (the build environment)",
)


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_fixtures():
    spec = importlib.util.spec_from_file_location(
        "batched_materialization_fixtures",
        Path(__file__).with_name("test_us_batched_target_materialization.py"),
    )
    fixtures = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixtures)
    return fixtures


def _district_hierarchy(name: str, district: str):
    """A production-shaped hierarchy: its target id is the spec name."""

    from microcosm.calibrate import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    return CalibrationHierarchy(
        provider=HierarchyNode(id="irs_soi", label="IRS SOI"),
        category=HierarchyCategory(
            id="irs_soi.fixture", label="Fixture", provider_id="irs_soi"
        ),
        geography=HierarchyGeography(
            id=f"5001900US{district}",
            label=f"District {district}",
            level="congressional_district",
        ),
        dimensions=(),
        target=HierarchyNode(id=name, label=name),
    )


def _cd_soi(fixtures, name, variable, state, district, **metadata):
    return dataclasses.replace(
        fixtures._soi(
            name,
            variable,
            state_fips=state,
            congressional_district_geoid=district,
            **metadata,
        ),
        hierarchy=_district_hierarchy(name, district),
    )


def _cd_surface_specs(fixtures) -> tuple:
    """The fixture's targets plus district SOI rows of several concepts.

    The fixture households sit in districts 0601 and 0602 (CA), 3601 and
    3602 (NY) and 2401 (MD). The concepts cover an amount, a count, an AGI
    band, a filing-status slice, an itemizer slice and the positive-EITC
    domain, so every mask the SOI slice builds is exercised.
    """

    rows = []
    for state, district in (
        ("06", "0601"),
        ("06", "0602"),
        ("36", "3601"),
        ("36", "3602"),
        ("24", "2401"),
    ):
        rows += [
            _cd_soi(
                fixtures, f"cd_{district}_agi", "adjusted_gross_income", state, district
            ),
            _cd_soi(fixtures, f"cd_{district}_returns", "count", state, district),
            _cd_soi(
                fixtures,
                f"cd_{district}_agi_band_count",
                "count",
                state,
                district,
                agi_lower_bound="-100",
                agi_upper_bound="900",
            ),
            _cd_soi(
                fixtures,
                f"cd_{district}_joint_wages",
                "employment_income",
                state,
                district,
                filing_status="Married Filing Jointly/Surviving Spouse",
            ),
            _cd_soi(
                fixtures,
                f"cd_{district}_itemized",
                "itemized_taxable_income_deductions",
                state,
                district,
                itemized_only="true",
            ),
            _cd_soi(
                fixtures,
                f"cd_{district}_eitc_returns",
                "count",
                state,
                district,
                ledger_domain=(
                    "individual_income_tax_returns_with_earned_income_credit"
                ),
            ),
        ]
    # A district row of a state no household lives in: an empty CSR row.
    rows.append(_cd_soi(fixtures, "cd_4801_agi", "adjusted_gross_income", "48", "4801"))
    return tuple(fixtures._TARGETS) + tuple(rows)


def _identity(digests: dict, *, households: int) -> dict:
    return {
        "households": households,
        "target_registry_sha256": digests["target_registry_sha256"],
        "target_roles_sha256": digests["target_roles_sha256"],
        "target_matrix": {"sha256": digests["target_matrix_sha256"]},
    }


def _materialize(module, fixtures, release, monkeypatch, tmp_path, specs, hh_chunk):
    fixtures._install_fake_engine(release, monkeypatch, reform_specs=fixtures._REFORMS)
    monkeypatch.setattr(module, "project_input_only", lambda frame, **kw: (frame, {}))
    monkeypatch.setattr(module, "fill_reviewed_nulls", lambda *args, **kw: None)
    return module.materialize_chunked(
        fixtures._nested_frame(),
        specs,
        hh_chunk=hh_chunk,
        batch=2,
        summary_path=tmp_path / "summary.json",
    )


def _dense_reference(fixtures, release, specs):
    """The pre-sparse construction: one float32 household column per target."""

    frame = fixtures._nested_frame()
    target_frame, registry, _ = release._materialize_target_frame(
        frame, tuple(specs), maximum_microsim_batch_size=2
    )
    households = target_frame.table("household")
    columns = {
        f"m{index:04d}": households[spec.measure].to_numpy(dtype=np.float32)
        for index, spec in enumerate(registry.specs)
    }
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"] = pd.concat(
        [
            tables["household"][["household_id"]].reset_index(drop=True),
            pd.DataFrame(columns),
        ],
        axis=1,
    )
    dense_frame = Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    # Exactly what TargetSpec.to_target() compiled on the dense checkpoint,
    # reading each target's float32 column.
    targets = TargetSet(
        [
            dataclasses.replace(spec.to_target(), measure=f"m{index:04d}")
            for index, spec in enumerate(registry.specs)
        ]
    )
    return dense_frame, targets, list(registry.specs)


def _sparse_records(module, materialized, fraction=0.0):
    records = [
        {
            "entity": "household",
            "measure": spec.name,
            "period": 2024,
            **module.cd_surface.target_record(spec),
        }
        for spec in materialized.compiled_specs
    ]
    module.cd_surface.assign_target_roles(records, fraction=fraction)
    return records


def _assert_same_problem(dense_problem, sparse_problem) -> None:
    assert dense_problem.names == sparse_problem.names
    assert dense_problem.matrix.shape == sparse_problem.matrix.shape
    np.testing.assert_array_equal(
        dense_problem.target_vector, sparse_problem.target_vector
    )
    dense_matrix = sparse.csr_array(dense_problem.matrix)
    sparse_matrix = sparse.csr_array(sparse_problem.matrix)
    np.testing.assert_array_equal(dense_matrix.indptr, sparse_matrix.indptr)
    np.testing.assert_array_equal(dense_matrix.indices, sparse_matrix.indices)
    np.testing.assert_array_equal(dense_matrix.data, sparse_matrix.data)


@pytest.mark.parametrize("hh_chunk", [1, 2, 5])
def test_sparse_materialization_equals_the_dense_columns_row_for_row(
    monkeypatch, tmp_path, hh_chunk
) -> None:
    """Every target's CSR row is its dense float32 column, zeros removed."""

    module = _load_tool_module()
    import build_us_fiscal_refresh_release as release

    fixtures = _load_fixtures()
    specs = _cd_surface_specs(fixtures)
    materialized = _materialize(
        module, fixtures, release, monkeypatch, tmp_path, specs, hh_chunk
    )
    dense_frame, dense_targets, dense_specs = _dense_reference(fixtures, release, specs)
    assert [spec.name for spec in materialized.compiled_specs] == [
        spec.name for spec in dense_specs
    ]
    assert materialized.matrix.data.dtype == np.float32
    dense_matrix = np.column_stack(
        [
            dense_frame.table("household")[f"m{index:04d}"].to_numpy()
            for index in range(len(dense_specs))
        ]
    ).T
    np.testing.assert_array_equal(materialized.matrix.toarray(), dense_matrix)
    # The district rows came from carriers, and the first chunk proved one
    # per carrier equal to its direct materialization.
    check = materialized.carrier_check
    assert check["district_rows"] == 32
    assert check["all_equal"] is True
    assert check["checked_district_rows"] == check["carriers"] >= 6


@requires_pytables
def test_dense_and_sparse_paths_give_identical_problems_and_weights(
    monkeypatch, tmp_path
) -> None:
    """The differential test of the sparse rewrite, end to end.

    Dense: float32 household columns compiled by the kernel from frame
    columns (the pre-sparse checkpoint). Sparse: the carrier-split CSR,
    written to and read back from the checkpoint, compiled by the kernel
    from callable rows. Same constraint values, same calibrated weights.
    """

    module = _load_tool_module()
    import build_us_fiscal_refresh_release as release

    from microcosm.calibrate import calibrate

    fixtures = _load_fixtures()
    specs = _cd_surface_specs(fixtures)
    materialized = _materialize(
        module, fixtures, release, monkeypatch, tmp_path, specs, hh_chunk=2
    )
    records = _sparse_records(module, materialized)
    struct = module.extract_struct_tables(fixtures._nested_frame())
    checkpoint = tmp_path / "checkpoint"
    _path, _registry, digests = module.write_lean_checkpoint(
        struct, materialized.matrix, materialized.compiled_specs, records, checkpoint
    )
    identity = _identity(digests, households=5)
    lean_frame, design, registry, loaded_records, loaded_matrix = (
        module.load_checkpoint_surface(checkpoint, identity)
    )
    dense_frame, dense_targets, _ = _dense_reference(fixtures, release, specs)
    np.testing.assert_array_equal(design, dense_frame.weights_for("household").values)

    sparse_targets = module.cd_surface.calibration_target_set(
        loaded_records, loaded_matrix, lean_frame.n("household"), specs=registry.specs
    )
    _assert_same_problem(
        build_constraint_matrix(dense_frame, dense_targets),
        build_constraint_matrix(lean_frame, sparse_targets),
    )

    settings = dict(max_weight_ratio=5.0, target_loss_cap=1.0, l2_lambda=0.0, seed=0)
    sparse_result, done = module.calibrate_surface(
        lean_frame,
        sparse_targets,
        epochs=40,
        epoch_batch=20,
        **settings,
    )
    assert done == 40
    warm = None
    for _batch in range(2):
        dense_result = calibrate(
            dense_frame,
            dense_targets,
            weight_entity="household",
            method="adam",
            epochs=20,
            learning_rate=0.02,
            mass="conserve",
            warm_start_weights=warm,
            **settings,
        )
        warm = dense_result.weights.copy()
    np.testing.assert_array_equal(sparse_result.weights, dense_result.weights)
    _assert_same_problem(dense_result.problem, sparse_result.problem)


def test_carriers_drop_the_district_hierarchy() -> None:
    """A carrier is renamed, so it cannot keep a hierarchy naming its row."""

    module = _load_tool_module()
    fixtures = _load_fixtures()
    plan = module.cd_surface.plan_carriers(_cd_surface_specs(fixtures))
    carriers = [
        spec
        for spec in plan.engine_specs
        if spec.name.startswith(module.cd_surface.CARRIER_PREFIX)
    ]
    assert len(carriers) == len(set(plan.carrier_of.values())) >= 6
    assert all(carrier.hierarchy is None for carrier in carriers)
    assert all(
        spec.hierarchy is not None
        for spec in plan.declared
        if module.cd_surface.is_cd_soi_spec(spec) and spec.name != "cd_0601_wages"
    )


def test_a_stored_district_row_that_disagrees_with_its_direct_row_is_refused(
    monkeypatch, tmp_path
) -> None:
    """The first-chunk check compares the rows the assembler stored."""

    module = _load_tool_module()
    import build_us_fiscal_refresh_release as release

    fixtures = _load_fixtures()
    assembler = module.cd_surface.SparseTargetAssembler
    real_add_masked = assembler.add_masked

    def corrupting_add_masked(self, row, low, carrier, positions, *, name):
        corrupted = np.asarray(carrier, dtype=np.float64).copy()
        corrupted[positions] += 1.0
        return real_add_masked(self, row, low, corrupted, positions, name=name)

    monkeypatch.setattr(assembler, "add_masked", corrupting_add_masked)
    with pytest.raises(RuntimeError, match="carrier split is not exact"):
        _materialize(
            module,
            fixtures,
            release,
            monkeypatch,
            tmp_path,
            _cd_surface_specs(fixtures),
            hh_chunk=5,
        )


def test_calibration_never_sees_a_held_out_target(monkeypatch, tmp_path) -> None:
    module = _load_tool_module()
    import build_us_fiscal_refresh_release as release

    import microcosm.calibrate as calibrate_package

    fixtures = _load_fixtures()
    materialized = _materialize(
        module,
        fixtures,
        release,
        monkeypatch,
        tmp_path,
        _cd_surface_specs(fixtures),
        hh_chunk=5,
    )
    records = _sparse_records(module, materialized)
    for record in records:
        if record.get("geography_level") is None and record["family"] == "irs_soi":
            if record.get("congressional_district_geoid"):
                record["geography_level"] = "congressional_district"
                record["state_cd_parent_target_name"] = "agi_amount"
    module.cd_surface.assign_target_roles(records, fraction=0.5)
    held = {record["name"] for record in records if record["role"] == "holdout"}
    assert held, "the fixture must hold something out at 0.5"
    seen: list[set[str]] = []
    real_calibrate = calibrate_package.calibrate

    def spy(frame, targets, **kwargs):
        seen.append({target.name for target in targets})
        return real_calibrate(frame, targets, **kwargs)

    monkeypatch.setattr(calibrate_package, "calibrate", spy)
    frame = fixtures._nested_frame()
    module.calibrate_surface(
        frame,
        module.cd_surface.calibration_target_set(records, materialized.matrix, 5),
        epochs=4,
        epoch_batch=2,
        max_weight_ratio=5.0,
        target_loss_cap=1.0,
        l2_lambda=0.0,
        seed=0,
    )
    assert len(seen) == 2
    trained = {record["name"] for record in records if record["role"] == "train"}
    for names in seen:
        assert names == trained
        assert not names & held


def test_population_rows_match_the_dense_household_size_indicator() -> None:
    module = _load_tool_module()
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "state_fips": [6, 6, 36, 6],
            "congressional_district_geoid": [601, 602, 3601, 601],
        }
    )
    size = np.asarray([2.0, 0.0, 3.0, 1.0], dtype=np.float32)
    populations = {
        "state": {6: 100.0, 36: 50.0, 24: 10.0},
        "cd": {601: 60.0, 602: 40.0, 3601: 50.0},
    }
    records, matrix, dropped = module.cd_surface.population_rows(
        households, size, populations, ["state", "cd"]
    )
    assert dropped == ["pop_state_24"]
    assert [record["name"] for record in records] == [
        "pop_state_06",
        "pop_state_36",
        "pop_cd_0601",
        "pop_cd_0602",
        "pop_cd_3601",
    ]
    # pop_cd_0602's only household has no persons: the row stays, empty.
    expected = np.asarray(
        [
            [2, 0, 0, 1],
            [0, 0, 3, 0],
            [2, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 3, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(matrix.toarray(), expected)
    assert records[2]["state_fips"] == "06"
    assert records[2]["congressional_district_geoid"] == "0601"


@requires_pytables
def test_checkpoint_refuses_changed_bytes_and_dense_predecessors(tmp_path) -> None:
    module = _load_tool_module()
    frame = _load_fixtures()._nested_frame()
    struct = module.extract_struct_tables(frame)
    matrix = sparse.csr_array(np.eye(2, 5, dtype=np.float32))
    specs = [
        TargetSpec(
            name=f"t{i}", entity="household", measure=f"t{i}", value=1.0, source="x"
        )
        for i in range(2)
    ]
    records = [
        {"name": f"t{i}", "value": 1.0, "source": "x", "family": "f", "role": "train"}
        for i in range(2)
    ]
    checkpoint = tmp_path / "ckpt"
    _path, _registry, digests = module.write_lean_checkpoint(
        struct, matrix, specs, records, checkpoint
    )
    identity = _identity(digests, households=5)
    module.load_checkpoint_surface(checkpoint, identity)
    for key, file_name in (
        ("target_registry_sha256", "target_registry.json"),
        ("target_roles_sha256", "target_roles.json"),
    ):
        with pytest.raises(SystemExit, match=f"{file_name} changed"):
            module.load_checkpoint_surface(checkpoint, {**identity, key: "0" * 64})
    with pytest.raises(SystemExit, match="target_matrix.npz changed"):
        module.load_checkpoint_surface(
            checkpoint, {**identity, "target_matrix": {"sha256": "0" * 64}}
        )
    with pytest.raises(ValueError, match="row-aligned"):
        module.write_lean_checkpoint(struct, matrix, specs, records[::-1], checkpoint)
    (checkpoint / module.TARGET_MATRIX_FILENAME).unlink()
    with pytest.raises(SystemExit, match="predates the sparse target matrix"):
        module.load_checkpoint_surface(checkpoint, identity)


def test_holdout_scoring_against_the_pro_rata_baseline() -> None:
    module = _load_tool_module()
    cd_surface = module.cd_surface
    records = [
        {"name": "state_agi", "value": 100.0, "family": "irs_soi", "role": "train"},
        *(
            {
                "name": f"cd_{d}",
                "value": value,
                "family": "irs_soi",
                "geography_level": "congressional_district",
                "state_fips": "06",
                "congressional_district_geoid": d,
                "source_measure_id": "adjusted_gross_income",
                "state_cd_parent_target_name": "state_agi",
                "cd_population": population,
                "state_population": 100.0,
            }
            for d, value, population in (("0601", 70.0, 50.0), ("0602", 30.0, 50.0))
        ),
    ]
    cd_surface.assign_target_roles(records, fraction=0.5)
    for record in records[1:]:
        record["role"] = "holdout"
    # Household 0 is in 0601, household 1 in 0602; each carries AGI 1.
    matrix = sparse.csr_array(np.asarray([[1, 1], [1, 0], [0, 1]], dtype=np.float32))
    baseline = cd_surface.pro_rata_baseline(records, [1, 2])
    np.testing.assert_allclose(baseline, [50.0, 50.0])
    assert baseline.sum() == pytest.approx(records[0]["value"])
    scored = cd_surface.score_cd_holdout(
        records,
        matrix,
        design_weights=np.asarray([50.0, 50.0]),
        final_weights=np.asarray([70.0, 30.0]),
        cap=1.0,
    )
    assert scored["n_targets"] == 2
    assert scored["calibrated"]["mean_abs_rel_error"] == pytest.approx(0.0)
    assert scored["pro_rata_baseline"]["mean_abs_rel_error"] == pytest.approx(
        (20 / 70 + 20 / 30) / 2
    )
    assert scored["calibrated_beats_pro_rata_share"] == 1.0
    assert scored["by_family"]["adjusted_gross_income"]["n_targets"] == 2


def test_weight_origin_counts_copies_of_one_household_once() -> None:
    cd_surface = _load_tool_module().cd_surface
    weights = np.asarray([1.0, 1.0, 2.0, 4.0])
    spine = np.asarray(["asec_puf", "asec_puf", "acs_2024_1yr", "asec_puf"])
    # Rows 0 and 1 are the native and PUF-detail copies of donor household 7;
    # ACS source id 7 is a different household.
    source = np.asarray([7, 7, 7, 9])
    summary = cd_surface.weight_origin_summary(weights, spine=spine, source_id=source)
    assert summary["distinct_households"] == 3
    assert summary["effective_sample_size_distinct_households"] == pytest.approx(
        8.0**2 / (2.0**2 + 2.0**2 + 4.0**2)
    )
    assert summary["effective_sample_size_rows"] == pytest.approx(
        8.0**2 / (1 + 1 + 4 + 16)
    )
    assert summary["weight_share_by_spine"] == {
        "acs_2024_1yr": 0.25,
        "asec_puf": 0.75,
    }


def test_sampling_is_a_rung_and_package_refuses_anything_below_full(
    tmp_path,
) -> None:
    module = _load_tool_module()
    argv = [
        "--stage",
        "materialize",
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--feed",
        str(tmp_path / "feed.jsonl"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ]
    assert module._parse_args(argv).sample_fraction == 1.0
    assert module._parse_args([*argv, "--sample-fraction", "0.1"]).sample_seed == 578
    with pytest.raises(SystemExit):
        module._parse_args([*argv, "--sample-fraction", "0.3"])
    with pytest.raises(SystemExit):
        module._parse_args([*argv, "--cd-holdout-fraction", "0.6"])
    module._require_full_rung({"sampling": {"sampled": False, "rung": "f100"}})
    with pytest.raises(SystemExit, match="f010 development rung"):
        module._require_full_rung({"sampling": {"sampled": True, "rung": "f010"}})
    with pytest.raises(SystemExit, match="no sampling block"):
        module._require_full_rung({})


def test_full_rung_returns_the_frame_unchanged() -> None:
    cd_surface = _load_tool_module().cd_surface
    frame = _load_fixtures()._nested_frame()
    same, receipt = cd_surface.sample_staging_frame(frame, fraction=1.0, seed=578)
    assert same is frame
    assert receipt["rung"] == "f100"
    assert receipt["sampled"] is False


def test_the_refresh_recipe_reproduces_the_recorded_cd_holdout() -> None:
    """Re-running a recorded recipe draws the recorded holdout, 0 included."""

    import shlex

    module = _load_tool_module()
    assert "--cd-holdout-fraction" not in module.release_refresh_recipe("state", 0.0)
    for fraction in (0.0, 0.1, 0.1234567):
        recipe = module.release_refresh_recipe("state_cd", fraction)
        assert f"--soi-mode state_cd --cd-holdout-fraction {fraction!r} " in recipe
        argv = [
            token.replace("<", "").replace(">", "") for token in shlex.split(recipe)[3:]
        ]
        args = module._parse_args(argv)
        assert module._cd_holdout_fraction(args) == fraction


def test_holdout_needs_the_state_cd_surface(tmp_path) -> None:
    module = _load_tool_module()
    args = SimpleNamespace(
        families="snap,medicaid,soi",
        geographies="state,cd",
        soi_mode="state",
        cd_holdout_fraction=0.1,
    )
    with pytest.raises(SystemExit, match="needs --soi-mode state_cd"):
        module.do_materialize(args)
    assert module._cd_holdout_fraction(
        SimpleNamespace(soi_mode="state_cd", cd_holdout_fraction=None)
    ) == pytest.approx(0.1)
    assert (
        module._cd_holdout_fraction(
            SimpleNamespace(soi_mode="state", cd_holdout_fraction=None)
        )
        == 0.0
    )


def test_district_exclusions_were_reviewed_against_the_packaged_crosswalk() -> None:
    """Regenerating the 117th->119th crosswalk must revisit the exclusions.

    North Carolina's district rows came back with #1043's registry-built
    crosswalk; a later crosswalk change needs the same review.
    """

    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
    )

    module = _load_tool_module()
    digest = module._sha256(default_congressional_district_vintage_crosswalk_path())
    assert digest == module.cd_surface.STATE_CD_REVIEWED_CROSSWALK_SHA256, (
        "The packaged CD crosswalk changed. Re-check which states' district "
        "rows it maps correctly, update STATE_CD_EXCLUDED_CD_STATES, and re-pin "
        "the reviewed digest."
    )
    assert module.cd_surface.STATE_CD_EXCLUDED_CD_STATES == {}


def _pinned_feed() -> Path:
    import os

    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS")
    if not feed or not Path(feed).exists():
        pytest.skip("MICROCOSM_US_CHRONICLE_FACTS is not set to the pinned feed")
    return Path(feed)


def test_pinned_feed_state_cd_surface_matches_its_contract() -> None:
    """On the pinned feed the ``state_cd`` surface has the documented shape.

    Counts are docs/us-acs-local-soi-target-surface.md's; every district
    block adds up to its state parent; each state concept has one vintage;
    every district of North Carolina is bound and no defective district-file
    column is.
    """

    feed = _pinned_feed()
    module = _load_tool_module()
    surface = module.state_admin_surface(
        feed, ["snap", "medicaid", "soi"], soi_mode="state_cd"
    )
    specs = list(surface.registry.specs)
    receipt = surface.soi_receipt
    soi = [spec for spec in specs if spec.family == "irs_soi"]
    assert receipt["counts"] == {
        "historic_table_2_state": 3819,
        "cd_file_state": 306,
        "congressional_district": 21777,
        "congressional_district_by_parent_basis": {
            "historic_table_2": 19215,
            "cd_file_state_total_bridged": 2562,
        },
        "total": 25902,
    }
    assert len(soi) == 25902
    assert len(specs) == 25902 + 102 + 51
    reconciliation = module.cd_surface.state_parent_reconciliation(soi)
    assert len(reconciliation) == 2193
    assert all(block["ok"] for block in reconciliation)
    state_rows = [
        spec for spec in soi if spec.metadata.get("ledger_geography_level") == "state"
    ]
    keys = [
        (spec.metadata["state_fips"], module.cd_surface.soi_concept_identity(spec))
        for spec in state_rows
    ]
    assert len(keys) == len(set(keys)), "a state concept is bound twice"
    districts = [
        spec
        for spec in soi
        if spec.metadata.get("ledger_geography_level") == "congressional_district"
    ]
    # North Carolina's 14 districts are bound since #1043's crosswalk.
    assert (
        len([spec for spec in districts if spec.metadata["state_fips"] == "37"])
        == 14 * 51
    )
    defective = set(module.cd_surface.STATE_CD_DEFECTIVE_CD_FILE_MEASURES)
    assert not [
        spec.name
        for spec in soi
        if module.cd_surface.is_cd_file_spec(spec)
        and spec.metadata["source_measure_id"] in defective
    ]
    assert all(spec.metadata.get("state_cd_parent_target_name") for spec in districts)
    assert receipt["sigma"]["targets_with_sigma"] == 0
    # 43 states carry district rows (51 minus the 8 at-large on the 117th plan).
    assert {entry["n"] for entry in receipt["rebase_factor_by_measure"].values()} == {
        43
    }
    # Every district-file-only state level is bridged, in all 51 states.
    bridges = receipt["level_bridge_factor_by_measure"]
    assert set(bridges) == set(module.cd_surface.STATE_CD_LEVEL_BRIDGES)
    assert {entry["n"] for entry in bridges.values()} == {51}
    assert json.dumps(receipt)  # the receipt is manifest-serializable


def test_the_cd_holdout_hash_is_pinned() -> None:
    """Golden values: a changed hash formula would silently redraw the holdout.

    The US holdout port reuses the same function, so both lines move together.
    """

    from microcosm.build.holdout import hash_holdout_uniform

    salt = _load_tool_module().cd_surface.CD_HOLDOUT_SALT
    assert hash_holdout_uniform("06|eitc", salt=salt) == 0.7980954941465199
    assert (
        hash_holdout_uniform("36|adjusted_gross_income", salt=salt)
        == 0.037229772944579555
    )


def _stratified_frame(donor_sizes: tuple[int, ...]) -> Frame:
    """Two spines x five districts; one weight per stratum (so sums are exact)."""

    from microcosm.frame import WeightKind, Weights

    districts = ["0601", "0602", "3601", "3602", "2401"]
    rows = []
    for spine, sizes in (("acs_2024_1yr", (40,) * 5), ("asec_puf", donor_sizes)):
        for index, (district, size) in enumerate(zip(districts, sizes, strict=True)):
            rows += [
                (spine, district, 10.0 + index + (50 if spine == "asec_puf" else 0))
            ] * size
    n = len(rows)
    ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        tables[group] = pd.DataFrame({f"{group}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": ids,
            "household_spine": [row[0] for row in rows],
            "congressional_district_geoid": [row[1] for row in rows],
        }
    )
    weights = np.asarray([row[2] for row in rows])
    return Frame(tables, US_SCHEMA, {"household": Weights(weights, WeightKind.DESIGN)})


def _stratum_mass(frame) -> pd.Series:
    table = frame.table("household").assign(
        weight=frame.weights_for("household").values
    )
    return table.groupby(
        ["household_spine", "congressional_district_geoid"]
    ).weight.sum()


def test_a_rung_keeps_every_drawn_stratum_at_full_mass() -> None:
    """Inverse-rate stratum weights: with one weight per stratum, exact."""

    cd_surface = _load_tool_module().cd_surface
    frame = _stratified_frame(donor_sizes=(8, 8, 8, 8, 8))
    sampled, receipt = cd_surface.sample_staging_frame(frame, fraction=0.25, seed=578)
    assert receipt["rung"] == "f025"
    assert receipt["zero_draw_strata"] == 0
    assert all(
        factor == pytest.approx(1.0, rel=1e-12)
        for factor in receipt["per_spine_normalization_factor"].values()
    )
    pd.testing.assert_series_equal(_stratum_mass(sampled), _stratum_mass(frame))


def test_zero_draw_strata_are_recorded_and_spread_over_their_spine() -> None:
    cd_surface = _load_tool_module().cd_surface
    # The first donor district (3 households) floors to zero draws at f025.
    frame = _stratified_frame(donor_sizes=(3, 8, 8, 8, 8))
    full = _stratum_mass(frame)
    sampled, receipt = cd_surface.sample_staging_frame(frame, fraction=0.25, seed=578)
    assert receipt["zero_draw_strata"] == 1
    lost = full[("asec_puf", "0601")]
    assert receipt["zero_draw_weight_share"] == pytest.approx(lost / full.sum())
    mass = _stratum_mass(sampled)
    assert ("asec_puf", "0601") not in mass.index
    donor_full = full.xs("asec_puf").sum()
    factor = donor_full / (donor_full - lost)
    for district in ("0602", "3601", "3602", "2401"):
        assert mass[("asec_puf", district)] == pytest.approx(
            full[("asec_puf", district)] * factor, rel=1e-12
        )
    assert mass.xs("asec_puf").sum() == pytest.approx(donor_full, rel=1e-12)
    # A spine none of whose strata draws cannot be represented at the rung.
    with pytest.raises(ValueError, match="drew no weight on spine 'asec_puf'"):
        cd_surface.sample_staging_frame(
            _stratified_frame(donor_sizes=(3, 3, 3, 3, 3)), fraction=0.25, seed=578
        )


@requires_pytables
def test_calibration_outputs_are_bound_to_the_materialization(tmp_path) -> None:
    """Resume, the complete shortcut, finalize and package refuse stale evidence."""

    module = _load_tool_module()
    identity = {
        "staging_sha256": "s",
        "target_roles_sha256": "r",
        "target_registry_sha256": "g",
        "target_matrix": {"sha256": "m"},
        "sampling": {"rung": "f100", "sampled": False},
    }
    stamp = module._run_identity_digest(identity)
    assert stamp == module._run_identity_digest(dict(reversed(identity.items())))
    changed = {**identity, "sampling": {"rung": "f010", "sampled": True}}
    assert module._run_identity_digest(changed) != stamp
    module._require_current_calibration(
        identity, {"run_identity_sha256": stamp}, stage="finalize"
    )
    for summary in ({}, {"run_identity_sha256": module._run_identity_digest(changed)}):
        with pytest.raises(SystemExit, match="does not belong to this checkpoint"):
            module._require_current_calibration(identity, summary, stage="package")
    # A pre-sparse identity (no roles digest) stays readable.
    module._require_current_calibration({"staging_sha256": "s"}, {}, stage="finalize")
    assert set(module.CALIBRATION_OUTPUT_FILENAMES) == {
        "weights_latest.npz",
        "calibration_summary.json",
        "calibration_diagnostics.json",
    }


def test_resume_refuses_weights_without_the_run_identity_stamp(
    tmp_path, monkeypatch
) -> None:
    module = _load_tool_module()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    identity = {"staging_sha256": "s", "target_roles_sha256": "r"}
    monkeypatch.setattr(module, "_verify_run_identity", lambda args: identity)
    frame = _load_fixtures()._nested_frame()
    registry = SimpleNamespace(specs=())
    monkeypatch.setattr(
        module,
        "load_checkpoint_surface",
        lambda *a, **k: (
            frame,
            np.ones(5),
            registry,
            [],
            sparse.csr_array((0, 5), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(module.cd_surface, "calibration_target_set", lambda *a, **k: [])
    args = SimpleNamespace(checkpoint_dir=checkpoint, resume=True, epochs=10)
    for saved in (
        {"weights": np.ones(5), "epochs_done": 5, "staging_sha256": "s"},
        {
            "weights": np.ones(5),
            "epochs_done": 5,
            "run_identity_sha256": module._run_identity_digest(
                {**identity, "target_roles_sha256": "other"}
            ),
        },
    ):
        np.savez(checkpoint / "weights_latest.npz", **saved)
        with pytest.raises(SystemExit, match="different materialization"):
            module.do_calibrate(args)
    # A matching stamp with every epoch done still needs a matching summary.
    np.savez(
        checkpoint / "weights_latest.npz",
        weights=np.ones(5),
        epochs_done=10,
        run_identity_sha256=module._run_identity_digest(identity),
    )
    (checkpoint / "calibration_summary.json").write_text(
        json.dumps({"run_identity_sha256": "stale"})
    )
    with pytest.raises(SystemExit, match="belongs to another materialization"):
        module.do_calibrate(args)
