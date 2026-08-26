"""Ladder-based rowwise clone path (#495 increment 6a).

The release route for the rowwise dataset: clone the national tables at
``n_clones``, assign geography through the ratified OA ladder
(:func:`assign_uk_geography_ladder`) instead of the crosswalk sampler, run
the release-blocking ladder gate, and carry the #501 weight-kind/mass-log
fence chain unchanged. Declared design delta vs the crosswalk route: no
cross-clone constituency collision avoidance — duplicate (source,
constituency) pairs are a reported diagnostic, not a prevented event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    assemble_uk_oa_ladder,
    clone_uk_dataset_with_ladder_geography,
    ladder_clone_index_column,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    uk_household_weight_kind,
    write_uk_rowwise_dataset,
)
from microcosm.frame import MassChangeRecord, WeightKind


def _ladder_metadata() -> dict[str, object]:
    def layer(vintage: str) -> dict[str, object]:
        return {"vintage": vintage, "source": "synthetic test source"}

    return {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "synthetic",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("synthetic"),
            "msoa": layer("synthetic"),
            "local_authority": layer("synthetic"),
            "ward": layer("synthetic"),
            "itl": layer("2021_itl"),
            "region": layer("synthetic"),
        },
    }


def _ladder_frame() -> pd.DataFrame:
    rows = [
        # London: two constituencies so draws can vary.
        ("E00000001", "E12000007", "E14000001", "E05014284", "E09000001", "TLI31"),
        ("E00000002", "E12000007", "E14000002", "E05014285", "E09000002", "TLI32"),
        # Wales, Scotland, NI: one constituency each.
        ("W00000001", "W99999999", "W07000041", "W05001517", "W06000001", "TLL11"),
        ("S00000001", "S99999999", "S14000001", "S13002835", "S12000033", "TLM50"),
        ("N20000001", "N99999999", "N05000001", "N10000104", "N09000001", "TLN0A"),
    ]
    return pd.DataFrame(
        [
            {
                "oa_code": oa,
                "population": 100.0,
                "households": 40.0,
                "constituency_code": constituency,
                "region_code": region_code,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": la,
                "ward_code": ward,
                "itl3_code": itl3,
            }
            for oa, region_code, constituency, ward, la, itl3 in rows
        ]
    )


@pytest.fixture()
def toy_ladder(tmp_path):
    payload = assemble_uk_oa_ladder(_ladder_frame(), _ladder_metadata())
    path = tmp_path / "toy_ladder.npz"
    np.savez_compressed(path, **payload)
    return load_uk_oa_ladder(path), path


def _household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "household_weight": [3.0, 10.0, 10.0, 10.0],
            "region": ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"],
        }
    )


def _person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [11, 21, 22, 31, 41],
            "person_household_id": [1, 2, 2, 3, 4],
            "person_benunit_id": [101, 201, 201, 301, 401],
        }
    )


def _benunit_frame() -> pd.DataFrame:
    return pd.DataFrame({"benunit_id": [101, 201, 301, 401]})


def _seam_record() -> MassChangeRecord:
    return MassChangeRecord(
        entity="household",
        old_total=33.0,
        new_total=33.0,
        declared_factor=1.0,
        reason="Toy reviewed record.",
    )


def _seam_frame(
    *,
    person: pd.DataFrame | None = None,
    benunit: pd.DataFrame | None = None,
    household: pd.DataFrame | None = None,
    mass_log: tuple[MassChangeRecord, ...] | None = None,
):
    from microcosm.build.uk_runtime import uk_national_frame

    return uk_national_frame(
        person=person if person is not None else _person_frame(),
        benunit=benunit if benunit is not None else _benunit_frame(),
        household=household if household is not None else _household_frame(),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(_seam_record(),) if mass_log is None else mass_log,
    )


def test_ladder_clone_assigns_gates_and_conserves(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    output = tmp_path / "rowwise_ladder.h5"
    result = clone_uk_dataset_with_ladder_geography(
        _seam_frame(),
        ladder,
        output_path=output,
        n_clones=2,
        seed=7,
        expected_constituency_vintage="2024_pcon",
    )

    frame_household = result.frame.table("household")
    frame_person = result.frame.table("person")
    frame_benunit = result.frame.table("benunit")
    assert len(frame_household) == 8
    assert len(frame_person) == 10
    assert len(frame_benunit) == 8
    assert result.frame.weights_for("household").total == pytest.approx(33.0)
    assert frame_household["household_id"].is_unique
    # In-memory carrier: per-entity clone-index names (Frame's flattening
    # rule forbids one shared name across entity tables).
    assert set(frame_household[ladder_clone_index_column("household")]) == {0, 1}
    assert ladder_clone_index_column("person") in frame_person.columns
    assert ladder_clone_index_column("benunit") in frame_benunit.columns

    # The full ladder column set rides on every clone, nonblank.
    for column in (
        "oa_code",
        "lsoa_code",
        "msoa_code",
        "local_authority_code",
        "ward_code",
        "constituency_code",
        "region_code",
        "itl3_code",
        "itl2_code",
        "itl1_code",
    ):
        values = frame_household[column].astype(str)
        assert (values.str.strip() != "").all(), column

    # Region marginals are preserved exactly (the ladder's core invariant).
    scotland = frame_household[frame_household["region"] == "SCOTLAND"]
    assert set(scotland["region_code"]) == {"S99999999"}
    assert set(scotland["constituency_code"]) == {"S14000001"}

    assert uk_household_weight_kind(result.frame) is WeightKind.IMPORTANCE
    assert len(result.frame.mass_log) == 2
    assert "n_clones=2" in result.frame.mass_log[-1].reason
    assert result.gate.passed

    # The written artifact keeps the legacy single-year schema: one
    # ``clone_index`` name per table (the writer renames the per-entity
    # in-memory columns at the export boundary), the weight-kind/mass-log
    # attrs, and the structural facts the rowwise seam's reader re-checks:
    # entity row counts, id uniqueness, and person->household linkage.
    stored_kind, stored_mass_log = read_uk_single_year_weight_metadata(output)
    assert stored_kind is WeightKind.IMPORTANCE
    assert stored_mass_log == result.frame.mass_log
    with pd.HDFStore(output, mode="r") as store:
        person = store["person"]
        benunit = store["benunit"]
        household = store["household"]
        assert len(person) == len(frame_person)
        assert len(benunit) == len(frame_benunit)
        assert len(household) == len(frame_household)
        assert household["household_id"].is_unique
        assert person["person_id"].is_unique
        assert set(person["person_household_id"]) <= set(household["household_id"])
        # The artifact clone column keeps its legacy name and position on all
        # three tables, no per-entity name leaks into the H5, and typed
        # household weights are materialized into the export payload.
        for table in (person, benunit, household):
            assert "clone_index" in table.columns
        assert not any(
            column.endswith("_clone_index")
            for table in (person, benunit, household)
            for column in table.columns
        )
        assert person.columns.tolist() == [
            "person_id",
            "person_household_id",
            "person_benunit_id",
            "clone_index",
        ]
        assert benunit.columns.tolist() == ["benunit_id", "clone_index"]
        assert household.columns.tolist()[:5] == [
            "household_id",
            "region",
            "source_household_id",
            "source_household_key",
            "clone_index",
        ]
        assert household["household_weight"].sum() == pytest.approx(33.0)


def test_ladder_clone_refuses_vintage_mismatch(toy_ladder) -> None:
    ladder, _ = toy_ladder
    with pytest.raises(ValueError, match="vintage"):
        clone_uk_dataset_with_ladder_geography(
            _seam_frame(),
            ladder,
            n_clones=1,
            expected_constituency_vintage="2005_pcon",
        )


def test_ladder_clone_refuses_uncovered_region(tmp_path) -> None:
    frame = _ladder_frame()
    england_only = frame[frame["region_code"].str.startswith("E")]
    payload = assemble_uk_oa_ladder(england_only, _ladder_metadata())
    path = tmp_path / "ew_ladder.npz"
    np.savez_compressed(path, **payload)
    ladder = load_uk_oa_ladder(path)
    with pytest.raises(ValueError, match="region"):
        clone_uk_dataset_with_ladder_geography(_seam_frame(), ladder, n_clones=1)


def test_ladder_clone_gate_failure_raises(toy_ladder) -> None:
    ladder, _ = toy_ladder

    all_london = _seam_frame(
        household=_household_frame().assign(region="LONDON"),
        mass_log=(),
    )

    # 100% London share breaches the gate's collapse bounds; the clone must
    # fail closed rather than write an artifact that fails its own gate.
    with pytest.raises(ValueError, match="[Ll]ondon"):
        clone_uk_dataset_with_ladder_geography(all_london, ladder, n_clones=1)


def test_ladder_clone_carries_pool_lineage(toy_ladder) -> None:
    ladder, _ = toy_ladder

    # A pooled input whose prior-year clone id (100000101) folds back
    # to source household 101 under the modulus. Group ids ascend —
    # every production producer is a Frame load, which guarantees it.
    pool_frame = _seam_frame(
        household=pd.DataFrame(
            {
                "household_id": [101, 102, 103, 100000101],
                "household_weight": [3.0, 10.0, 10.0, 10.0],
                "region": [
                    "LONDON",
                    "SCOTLAND",
                    "NORTHERN_IRELAND",
                    "WALES",
                ],
            }
        ),
        person=pd.DataFrame(
            {
                "person_id": [11, 31, 41, 21],
                "person_household_id": [101, 102, 103, 100000101],
                "person_benunit_id": [1, 3, 4, 2],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3, 4]}),
        mass_log=(),
    )

    result = clone_uk_dataset_with_ladder_geography(
        pool_frame,
        ladder,
        n_clones=1,
        source_lineage_modulus=100_000_000,
    )
    household = result.frame.table("household")
    lineage = dict(
        zip(
            household["household_id"],
            household["pool_source_household_id"],
            strict=True,
        )
    )
    assert lineage[101] == 101
    assert lineage[100000101] == 101


def test_write_round_trip_preserves_ladder_columns(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    result = clone_uk_dataset_with_ladder_geography(_seam_frame(), ladder, n_clones=1)
    path = write_uk_rowwise_dataset(result, tmp_path / "out.h5")
    with pd.HDFStore(path, mode="r") as store:
        household = store["household"]
    assert "ward_code" in household.columns
    assert "itl1_code" in household.columns


def _load_builder_module():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_rowwise_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_seam_h5(path) -> None:
    from microcosm.build.uk_runtime import write_uk_national_frame
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    dataset = uk_national_frame(
        person=_person_frame(),
        benunit=_benunit_frame(),
        household=_household_frame(),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=33.0,
                new_total=33.0,
                declared_factor=1.0,
                reason="Toy reviewed record.",
            ),
        ),
    )
    write_uk_national_frame(dataset, path)


def test_driver_ladder_route_builds_with_gate(monkeypatch, toy_ladder, tmp_path):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import json
    import sys

    _, ladder_path = toy_ladder
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_seam_h5(input_h5)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--out",
            str(output_dir),
            "--ladder",
            str(ladder_path),
            "--n-clones",
            "2",
            "--seed",
            "7",
        ],
    )
    assert builder.main() == 0
    manifest = json.loads((output_dir / builder.MANIFEST_FILENAME).read_text())
    assert manifest["parameters"]["assignment_route"] == "ladder"
    assert manifest["inputs"]["ladder"]["sha256"]
    summary = manifest["rowwise_dataset"]
    assert summary["gate"]["passed"] is True
    assert summary["missing_geography_rows"] == 0
    assert summary["assigned_constituencies"] >= 4
    assert summary["weights"]["household_weight_kind"] == "importance"
    assert summary["weights"]["mass_conservation"]["passed"] is True
    assert (output_dir / "staging_rowwise.h5").exists()


def test_driver_ladder_dry_run_matches_real_assignment(
    monkeypatch, toy_ladder, tmp_path
):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import json
    import sys

    ladder, ladder_path = toy_ladder
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_seam_h5(input_h5)
    plan_dir = tmp_path / "plan"
    build_dir = tmp_path / "build"

    base_argv = [
        "build_uk_rowwise_dataset.py",
        "--input-h5",
        str(input_h5),
        "--ladder",
        str(ladder_path),
        "--n-clones",
        "2",
        "--seed",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", [*base_argv, "--out", str(plan_dir), "--dry-run"])
    assert builder.main() == 0
    plan = json.loads((plan_dir / builder.DRY_RUN_PLAN_FILENAME).read_text())
    assert not (plan_dir / "staging_rowwise.h5").exists()

    monkeypatch.setattr(sys, "argv", [*base_argv, "--out", str(build_dir)])
    assert builder.main() == 0
    manifest = json.loads((build_dir / builder.MANIFEST_FILENAME).read_text())

    # The dry-run's realized support is exact: identical draws to the build.
    realized = {
        row["area_code"]: row["rows"]
        for row in plan["realized_support"]["constituency"]["bottom"]
    }
    with pd.HDFStore(build_dir / "staging_rowwise.h5", mode="r") as store:
        household = store["household"]
    built_counts = household["constituency_code"].value_counts()
    for code, rows in realized.items():
        assert built_counts.get(code, 0) == rows
    assert (
        plan["realized_support"]["constituency"]["n_areas"]
        >= manifest["rowwise_dataset"]["assigned_constituencies"]
    )


def test_driver_ladder_refuses_crosswalk_combo(monkeypatch, toy_ladder, tmp_path):
    pytest.importorskip("tables")
    import sys

    _, ladder_path = toy_ladder
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_seam_h5(input_h5)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--out",
            str(tmp_path / "out"),
            "--ladder",
            str(ladder_path),
            "--crosswalk",
            str(tmp_path / "crosswalk.csv"),
        ],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        builder.main()


def test_ladder_clone_pins_per_copy_weights_and_fk_alignment(toy_ladder) -> None:
    ladder, _ = toy_ladder
    result = clone_uk_dataset_with_ladder_geography(_seam_frame(), ladder, n_clones=2)
    household = result.frame.table("household")
    household_weights = result.frame.weights_for("household").values
    person = result.frame.table("person")
    household_clone_column = ladder_clone_index_column("household")
    # Every source copy carries exactly its divided weight.
    for clone_index in (0, 1):
        mask = household[household_clone_column] == clone_index
        copy = household[mask]
        weights = dict(
            zip(copy["source_household_id"], household_weights[mask], strict=True)
        )
        assert weights == {
            1: pytest.approx(1.5),
            2: pytest.approx(5.0),
            3: pytest.approx(5.0),
            4: pytest.approx(5.0),
        }
    # Person links never cross clone generations.
    household_clone = household.set_index("household_id")[household_clone_column]
    mapped = person["person_household_id"].map(household_clone)
    person_clones = person[ladder_clone_index_column("person")].to_numpy()
    assert (mapped.to_numpy() == person_clones).all()


def test_ladder_clone_refuses_negative_weights(toy_ladder) -> None:
    ladder, _ = toy_ladder

    from microcosm.build.uk_runtime import (
        clone_uk_dataset_tables_with_ladder_geography,
    )

    # A Frame input cannot carry a negative weight (the kernel refuses at
    # construction), so the clone's own guard is exercised at the raw-table
    # entry: a negative component hidden behind a positive aggregate.
    with pytest.raises(ValueError, match="non-negative"):
        clone_uk_dataset_tables_with_ladder_geography(
            person=_person_frame(),
            benunit=_benunit_frame(),
            household=_household_frame().assign(
                household_weight=[-1.0, 12.0, 11.0, 11.0]
            ),
            ladder=ladder,
            n_clones=1,
            time_period="2023",
            household_weight_kind=WeightKind.IMPORTANCE,
        )


def test_ladder_clone_rejects_unknown_weight_kind_h5(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import h5py

    from microcosm.build.uk_runtime.national_frame import (
        UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
    )

    ladder, _ = toy_ladder
    corrupted = tmp_path / "corrupted.h5"
    with pd.HDFStore(corrupted) as store:
        store.put("person", _person_frame(), format="table", data_columns=True)
        store.put("benunit", _benunit_frame(), format="table", data_columns=True)
        store.put("household", _household_frame(), format="table", data_columns=True)
        store.put("time_period", pd.Series(["2023"]), format="table", data_columns=True)
    with h5py.File(corrupted, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = "quantum"

    with pytest.raises(ValueError, match="weight kind"):
        clone_uk_dataset_with_ladder_geography(corrupted, ladder, n_clones=1)


def test_write_refuses_post_gate_geography_mutation(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    result = clone_uk_dataset_with_ladder_geography(_seam_frame(), ladder, n_clones=1)
    # Collapse every region code after the gate passed; Frame.table returns
    # the stored internals, so this models exactly the post-gate mutation
    # the writer must catch by re-gating the frame it actually writes.
    result.frame.table("household").loc[:, "region_code"] = "E12000007"
    with pytest.raises(ValueError, match="gate failed on the frame"):
        write_uk_rowwise_dataset(result, tmp_path / "mutated.h5")
    assert not (tmp_path / "mutated.h5").exists()


def test_dry_run_bottom_covers_every_toy_area(monkeypatch, toy_ladder, tmp_path):
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import json
    import sys

    _, ladder_path = toy_ladder
    builder = _load_builder_module()
    input_h5 = tmp_path / "staging.h5"
    _write_seam_h5(input_h5)
    plan_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_rowwise_dataset.py",
            "--input-h5",
            str(input_h5),
            "--ladder",
            str(ladder_path),
            "--out",
            str(plan_dir),
            "--n-clones",
            "2",
            "--dry-run",
        ],
    )
    assert builder.main() == 0
    plan = json.loads((plan_dir / builder.DRY_RUN_PLAN_FILENAME).read_text())
    constituency = plan["realized_support"]["constituency"]
    # The toy surface must stay within the bottom cap so the exactness test
    # keeps comparing every area; growing the fixture past the cap should
    # fail here loudly instead of silently weakening the comparison.
    assert constituency["n_areas"] <= builder.EXPECTED_SUPPORT_BOTTOM_AREAS
    assert len(constituency["bottom"]) == constituency["n_areas"]


def test_inherited_clone_index_is_replaced_like_the_pre_frame_writer(
    toy_ladder, tmp_path
) -> None:
    """The production staging input carries a candidate-tier clone_index.

    The rowwise artifact's clone_index names the rowwise clone dimension;
    the inherited SPI/pool-tier column is replaced exactly as the pre-Frame
    clone overwrote it in place — never left to collide with the per-entity
    in-memory names at the writer's rename (the adversarial-review blocker:
    pandas rename happily creates duplicate labels).
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    inherited = _seam_frame(
        household=_household_frame().assign(clone_index=[7, 8, 9, 3]),
    )
    output = tmp_path / "inherited.h5"
    result = clone_uk_dataset_with_ladder_geography(
        inherited,
        ladder,
        output_path=output,
        n_clones=2,
        seed=7,
        expected_constituency_vintage="2024_pcon",
    )

    household = result.frame.table("household")
    assert set(household[ladder_clone_index_column("household")]) == {0, 1}
    assert "clone_index" not in household.columns
    with pd.HDFStore(output, mode="r") as store:
        stored = store["household"]
        assert stored.columns.tolist().count("clone_index") == 1
        assert set(stored["clone_index"]) == {0, 1}


def test_reserved_in_memory_clone_names_fail_closed(toy_ladder) -> None:
    ladder, _ = toy_ladder
    poisoned = _seam_frame(
        household=_household_frame().assign(household_clone_index=[1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="reserved in-memory clone column"):
        clone_uk_dataset_with_ladder_geography(poisoned, ladder, n_clones=1)
