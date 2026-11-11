"""Tests split from packages/microcosm-build/tests/test_uk_ladder_rowwise_clone.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_ladder_rowwise_clone import *


def test_cloned_local_authority_survives_the_engine_loader(
    toy_ladder, tmp_path
) -> None:
    """The engine decodes the written member names, never its MAIDSTONE default.

    Through the written H5, not only the in-memory tables: the artifact is
    read back with pandas and loaded through the multi-year dataset path,
    which is the loader's ``set_input`` surface without the single-year
    economic-assumption uprating (that path reads council tax and rent
    columns a toy frame does not carry).
    """

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    from policyengine_uk import Microsimulation
    from policyengine_uk.data import UKMultiYearDataset, UKSingleYearDataset

    ladder, _ = toy_ladder
    result = clone_uk_dataset_with_ladder_geography(_seam_frame(), ladder, n_clones=1)
    written = result.frame.table("household")["local_authority"].tolist()
    # Household order is the seam order (London, Wales, Scotland, Northern
    # Ireland); the London row draws one of the two London OAs.
    assert written[0] in {"CITY_OF_LONDON", "BARKING_AND_DAGENHAM"}
    assert written[1:] == [
        "ISLE_OF_ANGLESEY",
        "ABERDEEN_CITY",
        "ANTRIM_AND_NEWTOWNABBEY",
    ]

    path = write_uk_rowwise_dataset(result, tmp_path / "rowwise.h5")
    with pd.HDFStore(path, mode="r") as store:
        tables = {
            entity: store[entity] for entity in ("person", "benunit", "household")
        }
    assert tables["household"]["local_authority"].tolist() == written

    single_year = UKSingleYearDataset(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        fiscal_year=2023,
    )
    simulation = Microsimulation(dataset=UKMultiYearDataset(datasets=[single_year]))

    decoded = [
        str(value)
        for value in simulation.calculate("local_authority", 2023, decode_enums=True)
    ]

    assert decoded == written
    assert "MAIDSTONE" not in decoded
