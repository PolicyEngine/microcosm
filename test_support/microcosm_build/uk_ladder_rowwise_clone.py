"""Ladder-based rowwise clone path (#495 increment 6a).

The release route for the rowwise dataset: clone the national tables at
``n_clones``, assign geography through the ratified OA ladder
(:func:`assign_uk_geography_ladder`) instead of the crosswalk sampler, run
the release-blocking ladder gate, and carry the #501 weight-kind/mass-log
fence chain unchanged. Declared design delta vs the crosswalk route: no
cross-clone constituency collision avoidance — duplicate (source,
constituency) pairs are a reported diagnostic, not a prevented event.
"""

# ruff: noqa: F401

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
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


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
            # Real April 2023 codes, so the engine input resolves through the
            # names resource (the ladder refuses any other vintage here).
            "local_authority": layer("2023_april_lad"),
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


def _spine_household_frame() -> pd.DataFrame:
    return _household_frame().assign(
        source_household_id=[10, 20, 10, 30],
        household_support_channel=["frs", "spi", "frs", "spi"],
        household_support_clone_index=[0, 1, 0, 0],
        household_is_spi_synthetic=[False, True, False, True],
        household_is_capital_gains_clone=[False, False, True, False],
        household_is_cgt_band_donor=[False, False, False, True],
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


def _load_builder_module():
    import importlib.util
    from pathlib import Path

    root = _TEST_PATHS.repository
    path = root / "tools" / "build_uk_rowwise_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_rowwise_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_seam_h5(path, *, household: pd.DataFrame | None = None) -> None:
    from microcosm.build.uk_runtime import write_uk_national_frame
    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    dataset = uk_national_frame(
        person=_person_frame(),
        benunit=_benunit_frame(),
        household=_household_frame() if household is None else household,
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


__all__ = [name for name in globals() if not name.startswith("__")]
