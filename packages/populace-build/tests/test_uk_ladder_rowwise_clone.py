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

from populace.build.uk_runtime import (
    assemble_uk_oa_ladder,
    clone_uk_dataset_with_ladder_geography,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    validate_uk_rowwise_dataset_tables,
    write_uk_rowwise_dataset,
)
from populace.frame import MassChangeRecord, WeightKind


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


class SeamLike:
    time_period = "2023"
    household_weight_kind = WeightKind.IMPORTANCE
    mass_log = (
        MassChangeRecord(
            entity="household",
            old_total=33.0,
            new_total=33.0,
            declared_factor=1.0,
            reason="Toy reviewed record.",
        ),
    )

    def __init__(self) -> None:
        self.person = _person_frame()
        self.benunit = _benunit_frame()
        self.household = _household_frame()


def test_ladder_clone_assigns_gates_and_conserves(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    output = tmp_path / "rowwise_ladder.h5"
    result = clone_uk_dataset_with_ladder_geography(
        SeamLike(),
        ladder,
        output_path=output,
        n_clones=2,
        seed=7,
        expected_constituency_vintage="2024_pcon",
    )

    assert len(result.household) == 8
    assert len(result.person) == 10
    assert len(result.benunit) == 8
    assert result.household["household_weight"].sum() == pytest.approx(33.0)
    assert result.household["household_id"].is_unique
    assert set(result.household["clone_index"]) == {0, 1}

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
        values = result.household[column].astype(str)
        assert (values.str.strip() != "").all(), column

    # Region marginals are preserved exactly (the ladder's core invariant).
    scotland = result.household[result.household["region"] == "SCOTLAND"]
    assert set(scotland["region_code"]) == {"S99999999"}
    assert set(scotland["constituency_code"]) == {"S14000001"}

    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert len(result.mass_log) == 2
    assert "n_clones=2" in result.mass_log[-1].reason
    assert result.gate.passed

    # The rowwise seam's own reader accepts the output with the fence
    # chain intact. (The output is not a national frame: clone_index lives
    # on every entity table, which Frame's flattening rule rejects.) The
    # written bytes must also satisfy the rowwise structural validator —
    # the reader-side teeth the retired national loader used to provide.
    stored_kind, stored_mass_log = read_uk_single_year_weight_metadata(output)
    assert stored_kind is WeightKind.IMPORTANCE
    assert stored_mass_log == result.mass_log
    with pd.HDFStore(output, mode="r") as store:
        validate_uk_rowwise_dataset_tables(
            store["person"], store["benunit"], store["household"]
        )
        assert len(store["person"]) == len(result.person)


def test_ladder_clone_refuses_vintage_mismatch(toy_ladder) -> None:
    ladder, _ = toy_ladder
    with pytest.raises(ValueError, match="vintage"):
        clone_uk_dataset_with_ladder_geography(
            SeamLike(),
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
        clone_uk_dataset_with_ladder_geography(SeamLike(), ladder, n_clones=1)


def test_ladder_clone_gate_failure_raises(toy_ladder) -> None:
    ladder, _ = toy_ladder

    class AllLondon(SeamLike):
        def __init__(self) -> None:
            super().__init__()
            self.household = self.household.assign(region="LONDON")
            self.mass_log = ()

    # 100% London share breaches the gate's collapse bounds; the clone must
    # fail closed rather than write an artifact that fails its own gate.
    with pytest.raises(ValueError, match="[Ll]ondon"):
        clone_uk_dataset_with_ladder_geography(AllLondon(), ladder, n_clones=1)


def test_ladder_clone_carries_pool_lineage(toy_ladder) -> None:
    ladder, _ = toy_ladder

    class PoolLike(SeamLike):
        mass_log = ()

        def __init__(self) -> None:
            super().__init__()
            self.household = pd.DataFrame(
                {
                    "household_id": [101, 100000101, 102, 103],
                    "household_weight": [3.0, 10.0, 10.0, 10.0],
                    "region": [
                        "LONDON",
                        "WALES",
                        "SCOTLAND",
                        "NORTHERN_IRELAND",
                    ],
                }
            )
            self.person = pd.DataFrame(
                {
                    "person_id": [11, 21, 31, 41],
                    "person_household_id": [101, 100000101, 102, 103],
                    "person_benunit_id": [1, 2, 3, 4],
                }
            )
            self.benunit = pd.DataFrame({"benunit_id": [1, 2, 3, 4]})

    result = clone_uk_dataset_with_ladder_geography(
        PoolLike(),
        ladder,
        n_clones=1,
        source_lineage_modulus=100_000_000,
    )
    lineage = dict(
        zip(
            result.household["household_id"],
            result.household["pool_source_household_id"],
            strict=True,
        )
    )
    assert lineage[101] == 101
    assert lineage[100000101] == 101


def test_write_round_trip_preserves_ladder_columns(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    ladder, _ = toy_ladder
    result = clone_uk_dataset_with_ladder_geography(SeamLike(), ladder, n_clones=1)
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
    from populace.build.uk_runtime import write_uk_national_frame
    from populace.build.uk_runtime.national_frame import uk_national_frame

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
    assert (output_dir / "populace_uk_2023_rowwise.h5").exists()


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
    assert not (plan_dir / "populace_uk_2023_rowwise.h5").exists()

    monkeypatch.setattr(sys, "argv", [*base_argv, "--out", str(build_dir)])
    assert builder.main() == 0
    manifest = json.loads((build_dir / builder.MANIFEST_FILENAME).read_text())

    # The dry-run's realized support is exact: identical draws to the build.
    realized = {
        row["area_code"]: row["rows"]
        for row in plan["realized_support"]["constituency"]["bottom"]
    }
    with pd.HDFStore(build_dir / "populace_uk_2023_rowwise.h5", mode="r") as store:
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
    result = clone_uk_dataset_with_ladder_geography(SeamLike(), ladder, n_clones=2)
    household = result.household
    # Every source copy carries exactly its divided weight.
    for clone_index in (0, 1):
        copy = household[household["clone_index"] == clone_index]
        weights = dict(
            zip(copy["source_household_id"], copy["household_weight"], strict=True)
        )
        assert weights == {
            1: pytest.approx(1.5),
            2: pytest.approx(5.0),
            3: pytest.approx(5.0),
            4: pytest.approx(5.0),
        }
    # Person links never cross clone generations.
    household_clone = household.set_index("household_id")["clone_index"]
    mapped = result.person["person_household_id"].map(household_clone)
    assert (mapped.to_numpy() == result.person["clone_index"].to_numpy()).all()


def test_ladder_clone_refuses_negative_weights(toy_ladder) -> None:
    ladder, _ = toy_ladder

    class NegativeWeights(SeamLike):
        mass_log = ()

        def __init__(self) -> None:
            super().__init__()
            # Negative component hidden behind a positive aggregate.
            self.household = self.household.assign(
                household_weight=[-1.0, 12.0, 11.0, 11.0]
            )

    with pytest.raises(ValueError, match="non-negative"):
        clone_uk_dataset_with_ladder_geography(NegativeWeights(), ladder, n_clones=1)


def test_ladder_clone_rejects_unknown_weight_kind_h5(toy_ladder, tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import h5py

    from populace.build.uk_runtime.national_build import (
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
    result = clone_uk_dataset_with_ladder_geography(SeamLike(), ladder, n_clones=1)
    # Collapse every region code after the gate passed; the writer must
    # re-gate the frame it actually writes.
    result.household.loc[:, "region_code"] = "E12000007"
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
