"""Weight-kind/mass-log carriage and mass conservation for the rowwise clone.

populace#495 increment 2. The rowwise clone consumes the national seam's
staging H5, whose ``populace_household_weight_kind`` and
``populace_mass_log_json`` attrs are part of the HMRC replay's fence chain
(IMPORTANCE weights, one reviewed mass-change record). Cloning through a
reader/writer that drops those attrs would silently launder IMPORTANCE
weights into the absence-default DESIGN kind — these tests pin the carriage,
the absence semantics, the fail-closed unknown-kind path, and the exact
mass-conservation bound of the clone operator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from populace.build.uk_runtime import (
    clone_uk_dataset_tables_with_rowwise_geography,
    clone_uk_dataset_with_rowwise_geography,
    load_uk_national_dataset,
    write_uk_national_dataset,
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
    with pd.HDFStore(legacy) as store:
        store.put("person", person_frame(), format="table", data_columns=True)
        store.put("benunit", benunit_frame(), format="table", data_columns=True)
        store.put("household", household_frame(), format="table", data_columns=True)
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )

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
    with pd.HDFStore(corrupted) as store:
        store.put("person", person_frame(), format="table", data_columns=True)
        store.put("benunit", benunit_frame(), format="table", data_columns=True)
        store.put("household", household_frame(), format="table", data_columns=True)
        store.put(
            "time_period",
            pd.Series(["2023"]),
            format="table",
            data_columns=True,
        )
    with h5py.File(corrupted, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = "quantum"

    with pytest.raises(ValueError, match="weight kind"):
        clone_uk_dataset_with_rowwise_geography(
            corrupted,
            crosswalk_frame(),
            n_clones=1,
            seed=7,
        )


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


def test_clone_tables_rejects_non_weightkind() -> None:
    with pytest.raises(TypeError, match="WeightKind"):
        clone_uk_dataset_tables_with_rowwise_geography(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household_frame(),
            crosswalk=crosswalk_frame(),
            n_clones=1,
            time_period="2023",
            household_weight_kind="importance",
        )


def test_mass_conservation_guard_trips_on_leaked_mass() -> None:
    from populace.build.uk_runtime.rowwise_dataset import (
        _assert_household_mass_conserved,
    )

    _assert_household_mass_conserved(30.0, 30.0 + 30.0 * 1e-12)
    with pytest.raises(ValueError, match="mass"):
        _assert_household_mass_conserved(30.0, 29.0)


def test_source_lineage_modulus_recovers_pool_lineage() -> None:
    household = pd.DataFrame(
        {
            "household_id": [101, 102, 100000101, 100000102],
            "household_weight": [10.0, 20.0, 10.0, 20.0],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [101, 102, 100000101, 100000102],
            "person_benunit_id": [11, 12, 13, 14],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [11, 12, 13, 14]})
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=person,
        benunit=benunit,
        household=household,
        crosswalk=crosswalk_frame(),
        n_clones=1,
        time_period="2023",
        source_lineage_modulus=100_000_000,
    )
    clone0 = result.household
    lineage = dict(
        zip(clone0["household_id"], clone0["source_household_id"], strict=True)
    )
    assert lineage[101] == 101
    assert lineage[100000101] == 101
    assert lineage[102] == 102
    assert lineage[100000102] == 102


def test_source_lineage_modulus_rejects_bad_inputs() -> None:
    kwargs = dict(
        person=person_frame(),
        benunit=benunit_frame(),
        household=household_frame(),
        crosswalk=crosswalk_frame(),
        n_clones=1,
        time_period="2023",
    )
    with pytest.raises(ValueError, match="source_lineage_modulus"):
        clone_uk_dataset_tables_with_rowwise_geography(
            **kwargs, source_lineage_modulus=0
        )
    # A modulus larger than every household_id is an identity mapping — a
    # near-certain caller error, refused rather than silently accepted.
    with pytest.raises(ValueError, match="identity"):
        clone_uk_dataset_tables_with_rowwise_geography(
            **kwargs, source_lineage_modulus=10_000
        )
    # An existing lineage column makes the modulus ambiguous.
    household = household_frame()
    household["source_household_id"] = household["household_id"]
    with pytest.raises(ValueError, match="source_household_id"):
        clone_uk_dataset_tables_with_rowwise_geography(
            person=person_frame(),
            benunit=benunit_frame(),
            household=household,
            crosswalk=crosswalk_frame(),
            n_clones=1,
            time_period="2023",
            source_lineage_modulus=100,
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
