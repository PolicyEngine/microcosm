"""Weight-kind/mass-log carriage and mass conservation for the rowwise clone.

populace#495 increment 2. The rowwise clone consumes the national seam's
staging H5, whose ``populace_household_weight_kind`` and
``populace_mass_log_json`` attrs are part of the HMRC replay's fence chain
(IMPORTANCE weights, one reviewed mass-change record). Cloning through a
reader/writer that drops those attrs would silently launder IMPORTANCE
weights into the absence-default DESIGN kind — these tests pin the carriage,
the absence semantics, the fail-closed unknown-kind and stale-chain paths,
the atomicity of the write, and the exact mass-conservation bound of the
clone operator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    POOL_SOURCE_LINEAGE_COLUMN,
    apply_uk_source_lineage_modulus,
    clone_entity_frame,
    clone_uk_dataset_tables_with_rowwise_geography,
    clone_uk_dataset_with_rowwise_geography,
    load_uk_national_dataset,
    write_uk_national_dataset,
    write_uk_rowwise_dataset,
)
from populace.build.uk_runtime.national_build import (
    UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
    UKNationalDataset,
)
from populace.frame import MassChangeRecord, WeightKind


def household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": [1, 2],
            "household_weight": [10.0, 20.0],
            "region": ["LONDON", "WALES"],
        }
    )


def person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1001, 2001, 2002],
            "person_household_id": [1, 2, 2],
            "person_benunit_id": [101, 201, 201],
            "dividend_income": [5.0, 9.0, 11.0],
        }
    )


def benunit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benunit_id": [101, 201],
            "would_claim_uc": [True, False],
        }
    )


def crosswalk_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": "E0001",
                "lsoa_code": "E0101",
                "msoa_code": "E0201",
                "la_code": "E06000063",
                "constituency_code": "E14000001",
                "region_code": "E12000007",
                "country": "England",
                "population": 100,
            },
            {
                "oa_code": "W0001",
                "lsoa_code": "W0101",
                "msoa_code": "W0201",
                "la_code": "W06000001",
                "constituency_code": "W07000001",
                "region_code": "W99999999",
                "country": "Wales",
                "population": 50,
            },
        ]
    )


def _write_legacy_h5(path, household: pd.DataFrame | None = None) -> None:
    with pd.HDFStore(path) as store:
        store.put("person", person_frame(), format="table", data_columns=True)
        store.put("benunit", benunit_frame(), format="table", data_columns=True)
        store.put(
            "household",
            household_frame() if household is None else household,
            format="table",
            data_columns=True,
        )
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )


def _importance_dataset() -> UKNationalDataset:
    return UKNationalDataset(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        time_period="2023",
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=30.0,
                new_total=30.0,
                declared_factor=1.0,
                reason="Toy reviewed SPI-channel allocation record.",
            ),
        ),
    )


def test_clone_h5_carries_importance_weight_kind_and_mass_log(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    staging = tmp_path / "staging.h5"
    write_uk_national_dataset(_importance_dataset(), staging)

    output = tmp_path / "rowwise.h5"
    result = clone_uk_dataset_with_rowwise_geography(
        staging,
        crosswalk_frame(),
        output_path=output,
        n_clones=2,
        seed=7,
    )

    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert len(result.mass_log) == 2
    assert result.mass_log[0].reason == "Toy reviewed SPI-channel allocation record."
    clone_record = result.mass_log[-1]
    assert clone_record.entity == "household"
    assert clone_record.old_total == pytest.approx(30.0)
    assert clone_record.new_total == pytest.approx(30.0)
    assert clone_record.declared_factor == 1.0
    assert "n_clones=2" in clone_record.reason

    # The national loader must accept the rowwise output and see the same
    # weight-kind chain: carriage is proven by the seam's own reader.
    reloaded = load_uk_national_dataset(output)
    assert reloaded.household_weight_kind is WeightKind.IMPORTANCE
    assert reloaded.mass_log == result.mass_log
    assert reloaded.household["household_weight"].sum() == pytest.approx(30.0)


def test_clone_legacy_h5_defaults_design_and_writes_explicit_attrs(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import h5py

    legacy = tmp_path / "legacy.h5"
    _write_legacy_h5(legacy)

    output = tmp_path / "rowwise.h5"
    result = clone_uk_dataset_with_rowwise_geography(
        legacy,
        crosswalk_frame(),
        output_path=output,
        n_clones=1,
        seed=7,
    )
    assert result.household_weight_kind is WeightKind.DESIGN
    assert len(result.mass_log) == 1
    assert result.mass_log[0].entity == "household"

    with h5py.File(output, mode="r") as file:
        stored = file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR]
        if isinstance(stored, bytes):
            stored = stored.decode("utf-8")
        assert stored == WeightKind.DESIGN.value


def test_clone_h5_rejects_unknown_weight_kind_attr(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    import h5py

    corrupted = tmp_path / "corrupted.h5"
    _write_legacy_h5(corrupted)
    with h5py.File(corrupted, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = "quantum"

    with pytest.raises(ValueError, match="weight kind"):
        clone_uk_dataset_with_rowwise_geography(
            corrupted,
            crosswalk_frame(),
            n_clones=1,
            seed=7,
        )


def test_clone_float32_weights_conserve_within_declared_bound(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    household = household_frame()
    household["household_weight"] = household["household_weight"].astype(np.float32)
    legacy = tmp_path / "float32.h5"
    _write_legacy_h5(legacy, household=household)

    result = clone_uk_dataset_with_rowwise_geography(
        legacy,
        crosswalk_frame(),
        n_clones=3,
        seed=7,
    )
    assert result.household["household_weight"].sum() == pytest.approx(30.0)


def test_clone_tables_appends_conservation_record_and_conserves_mass() -> None:
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=3,
        seed=11,
        time_period="2023",
        household_weight_kind=WeightKind.IMPORTANCE,
        mass_log=(),
    )
    assert result.household_weight_kind is WeightKind.IMPORTANCE
    assert result.household["household_weight"].sum() == pytest.approx(30.0, rel=1e-12)
    assert len(result.mass_log) == 1
    record = result.mass_log[0]
    assert record.old_total == pytest.approx(30.0)
    assert record.new_total == pytest.approx(30.0)
    assert record.declared_factor == 1.0


def test_clone_tables_rejects_non_weightkind_and_none_mass_log() -> None:
    kwargs = dict(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=1,
        time_period="2023",
    )
    with pytest.raises(TypeError, match="WeightKind"):
        clone_uk_dataset_tables_with_rowwise_geography(
            **kwargs, household_weight_kind="importance"
        )
    with pytest.raises(TypeError, match="MassChangeRecord"):
        clone_uk_dataset_tables_with_rowwise_geography(
            **kwargs,
            household_weight_kind=WeightKind.DESIGN,
            mass_log=None,
        )


def test_clone_tables_rejects_stale_mass_log() -> None:
    stale = (
        MassChangeRecord(
            entity="household",
            old_total=99.0,
            new_total=99.0,
            declared_factor=1.0,
            reason="Stale toy record.",
        ),
    )
    with pytest.raises(ValueError, match="stale"):
        clone_uk_dataset_tables_with_rowwise_geography(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            crosswalk=crosswalk_frame(),
            n_clones=1,
            time_period="2023",
            mass_log=stale,
        )


def test_clone_tables_rejects_zero_total_pool() -> None:
    household = household_frame()
    household["household_weight"] = 0.0
    with pytest.raises(ValueError, match="positive total mass"):
        clone_uk_dataset_tables_with_rowwise_geography(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household,
            crosswalk=crosswalk_frame(),
            n_clones=1,
            time_period="2023",
        )


def test_mass_conservation_guard_trips_on_leaked_or_nonfinite_mass() -> None:
    from populace.build.uk_runtime.rowwise_dataset import (
        _assert_household_mass_conserved,
    )

    _assert_household_mass_conserved(30.0, 30.0 + 30.0 * 1e-12)
    with pytest.raises(ValueError, match="mass"):
        _assert_household_mass_conserved(30.0, 29.0)
    with pytest.raises(ValueError, match="finite"):
        _assert_household_mass_conserved(float("inf"), float("inf"))


def test_write_refuses_mutated_weights(tmp_path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=1,
        time_period="2023",
    )
    result.household.loc[:, "household_weight"] = (
        result.household["household_weight"] * 2.0
    )
    # The frozen dataclass cannot freeze DataFrames; the writer re-verifies
    # the chain so the mutation surfaces as a stale mass log.
    with pytest.raises(ValueError, match="stale"):
        write_uk_rowwise_dataset(result, tmp_path / "mutated.h5")
    assert not (tmp_path / "mutated.h5").exists()


def test_write_is_atomic_when_metadata_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("tables")
    from populace.build.uk_runtime import rowwise_dataset as module

    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=1,
        time_period="2023",
    )

    def broken_metadata_write(path, dataset) -> None:
        raise RuntimeError("simulated metadata failure")

    monkeypatch.setattr(module, "_write_weight_metadata", broken_metadata_write)
    output = tmp_path / "rowwise.h5"
    with pytest.raises(RuntimeError, match="simulated metadata failure"):
        write_uk_rowwise_dataset(result, output)
    # No complete-looking attr-less H5 and no temporary remnant may remain.
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_source_lineage_modulus_writes_distinct_pool_layer() -> None:
    household = pd.DataFrame(
        {
            "household_id": [101, 102, 100000101, 100000102],
            "household_weight": [10.0, 20.0, 10.0, 20.0],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
            "source_household_id": [901, 902, 903, 904],
        }
    )
    derived = apply_uk_source_lineage_modulus(household, modulus=100_000_000)
    assert list(derived["source_household_id"]) == [901, 902, 903, 904]
    lineage = dict(
        zip(
            derived["household_id"],
            derived[POOL_SOURCE_LINEAGE_COLUMN],
            strict=True,
        )
    )
    assert lineage[101] == 101
    assert lineage[100000101] == 101
    assert lineage[102] == 102
    assert lineage[100000102] == 102


def test_source_lineage_modulus_rejects_bad_inputs() -> None:
    household = household_frame()
    with pytest.raises(ValueError, match="source_lineage_modulus"):
        apply_uk_source_lineage_modulus(household, modulus=0)
    # A modulus larger than every household_id is an identity mapping — a
    # near-certain caller error, refused rather than silently accepted.
    with pytest.raises(ValueError, match="identity"):
        apply_uk_source_lineage_modulus(household, modulus=10_000)
    # An existing pool column makes the modulus ambiguous.
    with_pool = household_frame()
    with_pool[POOL_SOURCE_LINEAGE_COLUMN] = with_pool["household_id"]
    with pytest.raises(ValueError, match=POOL_SOURCE_LINEAGE_COLUMN):
        apply_uk_source_lineage_modulus(with_pool, modulus=100)
    # Validation happens on numeric values, not after a lossy integer cast.
    fractional = household_frame()
    fractional["household_id"] = [-0.5, 200.0]
    with pytest.raises(ValueError, match="non-negative|integral"):
        apply_uk_source_lineage_modulus(fractional, modulus=100)


def test_clone_entity_frame_refuses_int64_overflow() -> None:
    frame = pd.DataFrame({"benunit_id": [np.iinfo(np.int64).max - 5]})
    with pytest.raises(ValueError, match="overflow"):
        clone_entity_frame(
            frame,
            id_columns=("benunit_id",),
            n_clones=2,
            id_multiplier=10,
        )


def test_dataset_object_metadata_carried_and_defaulted() -> None:
    class SeamLike:
        time_period = "2023"
        household_weight_kind = WeightKind.IMPORTANCE
        mass_log = (
            MassChangeRecord(
                entity="household",
                old_total=30.0,
                new_total=30.0,
                declared_factor=1.0,
                reason="Toy reviewed record.",
            ),
        )

        def __init__(self) -> None:
            self.person = person_frame()
            self.benunit = benunit_frame()
            self.household = household_frame()

    class Legacy:
        time_period = "2023"

        def __init__(self) -> None:
            self.person = person_frame()
            self.benunit = benunit_frame()
            self.household = household_frame()

    class NoneLog:
        time_period = "2023"
        mass_log = None

        def __init__(self) -> None:
            self.person = person_frame()
            self.benunit = benunit_frame()
            self.household = household_frame()

    carried = clone_uk_dataset_with_rowwise_geography(
        SeamLike(), crosswalk_frame(), n_clones=1, seed=3
    )
    assert carried.household_weight_kind is WeightKind.IMPORTANCE
    assert len(carried.mass_log) == 2

    defaulted = clone_uk_dataset_with_rowwise_geography(
        Legacy(), crosswalk_frame(), n_clones=1, seed=3
    )
    assert defaulted.household_weight_kind is WeightKind.DESIGN
    assert len(defaulted.mass_log) == 1

    with pytest.raises(TypeError, match="mass_log"):
        clone_uk_dataset_with_rowwise_geography(
            NoneLog(), crosswalk_frame(), n_clones=1, seed=3
        )
