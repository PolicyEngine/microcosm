"""The high-value home support stage (microcosm#1240, fix 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import high_value_home_support as hv
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    uk_household_clone_paths,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.property_localisation import SUPPORT_STRATUM_LOWER
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_clone_first_frames import (
    assert_copy_ids_follow_lineage,
)


def _frame(n: int = 40):
    """``n`` single-adult owner households in one region, values ascending."""
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype="int64"),
            "region": pd.array(["LONDON"] * n, dtype="string"),
            "tenure_type": pd.array(
                ["OWNED_OUTRIGHT"] * (n - 2) + ["RENT_PRIVATELY"] * 2, dtype="string"
            ),
            "main_residence_value": np.linspace(100_000.0, 2_000_000.0, n),
            "source_household_id": np.arange(1, n + 1, dtype="int64"),
            "household_support_channel": pd.array(["frs"] * n, dtype="string"),
            "household_support_clone_index": np.zeros(n, dtype="int64"),
            "household_is_cgt_support_copy": [False] * n,
            "cgt_support_copy_index": np.zeros(n, dtype="int64"),
            "household_is_capital_gains_clone": [False] * n,
            "household_is_cgt_residential_clone": [False] * n,
            "cgt_residential_clone_index": np.zeros(n, dtype="int64"),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(1, n + 1, dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_household_id": np.arange(1, n + 1, dtype="int64"),
            "person_benunit_id": np.arange(1, n + 1, dtype="int64"),
            "age": np.full(n, 45, dtype="int64"),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.full(n, 10.0),
    )


def test_selection_takes_the_top_stratum_of_each_region() -> None:
    frame = _frame()
    household = frame.table("household")
    selected, rows = hv.select_high_value_support_households(
        household_ids=household["household_id"].to_numpy(),
        weights=np.full(len(household), 10.0),
        region=household["region"].to_numpy(),
        tenure_type=household["tenure_type"].to_numpy(),
        value=household["main_residence_value"].to_numpy(),
        stratum_lower=0.9,
    )
    # 38 equal-weight owners: mid-ranks (i + 0.5) / 38 >= 0.9 for the top four.
    assert int(selected.sum()) == 4
    owners = household["tenure_type"] != "RENT_PRIVATELY"
    assert (
        household.loc[selected, "main_residence_value"].min()
        > household.loc[owners & ~selected, "main_residence_value"].max()
    )
    assert not selected[-2:].any()  # renters are never in the stratum
    (row,) = rows
    assert (
        row["region"] == "LONDON"
        and row["owner_households"] == 38
        and row["households_selected"] == 4
    )


def test_split_copies_whole_households_and_conserves_mass() -> None:
    frame = _frame()
    result = hv.split_high_value_support_households(frame, stratum_lower=0.9, copies=3)
    out = result.frame
    household = out.table("household")
    assert result.households_selected == 4 and result.copies_created == 8
    assert len(household) == 40 + 8 and result.households_after == 48
    assert out.weights_for("household").total == pytest.approx(
        frame.weights_for("household").total
    )
    copies = household[household[HIGH_VALUE_SUPPORT_COPY_COLUMN]]
    assert sorted(copies[HIGH_VALUE_SUPPORT_INDEX_COLUMN].unique()) == [1, 2]
    roots = household[~household[HIGH_VALUE_SUPPORT_COPY_COLUMN]]
    assert (roots[HIGH_VALUE_SUPPORT_INDEX_COLUMN] == 0).all()
    weights = pd.Series(
        out.weights_for("household").values, index=household["household_id"].to_numpy()
    )
    split_roots = roots[roots[hv.HIGH_VALUE_SUPPORT_COPIES_COLUMN] == 3]
    assert len(split_roots) == 4
    for root_id in split_roots["household_id"]:
        assert weights[root_id] == pytest.approx(10.0 / 3)
    # Every copy carries its person and benefit-unit rows under remapped ids.
    person = out.table("person")
    assert len(person) == 48 and person["person_id"].is_unique
    assert set(person["person_household_id"]) == set(household["household_id"])
    # The lineage keys read the copy index as the high_value_support branch.
    _, paths = uk_household_clone_paths(household)
    branches = {tuple(path) for path in paths if path}
    assert branches == {(("high_value_support", 1),), (("high_value_support", 2),)}
    with pytest.raises(ValueError, match="already ran"):
        hv.split_high_value_support_households(out)


def test_copies_take_their_roots_ids_plus_k_times_the_declared_radix() -> None:
    # Roots come from the input by lineage, never from the copy's id: every
    # id column of copy k sits at its root's plus k x 10**12 (microcosm#931).
    frame = _frame()
    result = hv.split_high_value_support_households(frame, stratum_lower=0.9)

    checked = assert_copy_ids_follow_lineage(
        frame,
        result.frame,
        field=hv.HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
        key_columns=("source_household_id",),
        ordinal_column=HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    )
    assert checked == result.copies_created == 4 * (hv.HIGH_VALUE_SUPPORT_COPIES - 1)


def _frame_with_spi_copies():
    """The owner frame in FRS id spaces, with SPI copies of its top owners.

    Households keep their sernum, benefit units take sernum x 100 + 1 and
    people sernum x 1000 + 1. Each of the four most valuable owners also has
    an SPI copy at every id plus 10**11, whose provenance names its root, so
    a copy's immediate parent and its provenance differ. ``marker`` names
    each row: lineage never leans on an id or on the provenance column.
    """
    base = _frame()
    household = base.table("household").copy()
    benunit = base.table("benunit").copy()
    person = base.table("person").copy()
    benunit["benunit_id"] = benunit["benunit_id"] * 100 + 1
    person["person_benunit_id"] = person["person_benunit_id"] * 100 + 1
    person["person_id"] = person["person_id"] * 1000 + 1
    household["marker"] = np.arange(1, len(household) + 1)
    spi = 10**11
    top = household.loc[household["tenure_type"] != "RENT_PRIVATELY"].tail(4)
    copies = top.assign(
        household_id=top["household_id"] + spi,
        household_support_channel=pd.array(["spi"] * len(top), dtype="string"),
        household_support_clone_index=1,
        marker=top["marker"] + 1_000,
    )
    members = person.loc[person["person_household_id"].isin(top["household_id"])]
    member_units = benunit.loc[benunit["benunit_id"].isin(members["person_benunit_id"])]
    return uk_national_frame(
        person=pd.concat(
            [
                person,
                members.assign(
                    person_id=members["person_id"] + spi,
                    person_household_id=members["person_household_id"] + spi,
                    person_benunit_id=members["person_benunit_id"] + spi,
                ),
            ],
            ignore_index=True,
        ),
        benunit=pd.concat(
            [benunit, member_units.assign(benunit_id=member_units["benunit_id"] + spi)],
            ignore_index=True,
        ),
        household=pd.concat([household, copies], ignore_index=True),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.full(len(household) + len(copies), 10.0),
    )


def test_copies_of_spi_copies_take_their_parents_ids_plus_k_times_the_radix() -> None:
    # Separate id spaces and SPI copies, whose provenance differs from their
    # immediate parent: a stage that minted from the provenance column, or one
    # id column from another, would miss here (microcosm#931).
    frame = _frame_with_spi_copies()
    result = hv.split_high_value_support_households(frame, stratum_lower=0.9)

    checked = assert_copy_ids_follow_lineage(
        frame,
        result.frame,
        field=hv.HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
        key_columns=("marker",),
        ordinal_column=HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    )
    assert checked == result.copies_created > 0
    household = result.frame.table("household")
    copied = household.loc[household[HIGH_VALUE_SUPPORT_COPY_COLUMN]]
    # Some copies are of SPI copies, and some of base households.
    assert (copied["marker"] > 1_000).any()
    assert (copied["marker"] < 1_000).any()
    # The identity tool's old-spine gate reads the stage's copies as on the
    # table (microcosm#931).
    assert _identity_tool()._copy_id_rows_off_the_declared_table(household) == 0


def _identity_tool():
    import importlib.util

    from test_support.paths import paths_for

    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "verify_uk_identity_stability.py"
    )
    spec = importlib.util.spec_from_file_location("verify_uk_identity_stability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_split_refuses_incoming_ids_at_its_copy_id_radix() -> None:
    frame = _frame()
    household = frame.table("household").copy()
    person = frame.table("person").copy()
    household.loc[household.index[-1], "household_id"] = 10**12 + 40
    person.loc[person.index[-1], "person_household_id"] = 10**12 + 40
    shifted = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match=r"^1 of the incoming person_household_id"):
        hv.split_high_value_support_households(shifted)


def test_evidence_matches_the_gate_and_the_committed_declaration() -> None:
    frame = _frame()
    result = hv.split_high_value_support_households(frame)
    evidence = result.evidence()
    assert evidence["parameters"] == {
        "stratum_lower": SUPPORT_STRATUM_LOWER,
        "copies": hv.HIGH_VALUE_SUPPORT_COPIES,
    }
    entry = {entry.id: entry for entry in load_country_spec("uk").gates.gates}[
        "uk_stage_high_value_home_support_mass"
    ]
    verdict = uk_stage_health_gate(
        evidence=evidence,
        stage=entry.parameters["stage"],
        check=entry.parameters["check"],
        parameters=entry.parameters,
    )
    assert verdict.passed, verdict.failures
    broken = {**evidence, "totals": {**evidence["totals"], "copies_created": 1}}
    assert not uk_stage_health_gate(
        evidence=broken,
        stage=entry.parameters["stage"],
        check=entry.parameters["check"],
        parameters=entry.parameters,
    ).passed


def test_stage_manifest_restates_the_code_dictionary() -> None:
    spec = load_country_spec("uk")
    stage = next(
        s for s in spec.sources.stages if s.stage == hv.HIGH_VALUE_SUPPORT_STAGE_NAME
    )
    transform = hv.UKHighValueHomeSupportStageTransform(stage=stage)
    assert transform.output_columns() == tuple(stage.outputs)
    names = [s.stage for s in spec.sources.stages]
    assert names.index("regional_property_uprating") + 1 == names.index(
        hv.HIGH_VALUE_SUPPORT_STAGE_NAME
    )
    out = transform(_frame())
    assert HIGH_VALUE_SUPPORT_COPY_COLUMN in out.table("household").columns
    assert (
        transform.checkpoint_metadata()["evidence"]["stage"]
        == hv.HIGH_VALUE_SUPPORT_STAGE_NAME
    )
