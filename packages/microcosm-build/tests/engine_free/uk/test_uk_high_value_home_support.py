"""The high-value home support stage (microcosm#1240, fix 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import (
    binomial_share_tolerance,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime import copy_id_table
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

#: The stratum's declared tie-break, which every direct call passes.
TIE = {"seed": hv.HIGH_VALUE_SUPPORT_TIE_SEED, "salt": hv.HIGH_VALUE_SUPPORT_TIE_SALT}


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
        **TIE,
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


def _owners(values, *, ids=None, region="LONDON"):
    """Equal-weight owners in one region, the selection's inputs."""

    n = len(values)
    return {
        "household_ids": np.arange(1, n + 1, dtype=np.int64) if ids is None else ids,
        "weights": np.full(n, 10.0),
        "region": np.full(n, region, dtype=object),
        "tenure_type": np.full(n, "OWNED_OUTRIGHT", dtype=object),
        "value": np.asarray(values, dtype=float),
    }


def _id_ranked_stratum(inputs, stratum_lower: float) -> np.ndarray:
    """The stratum as the rank took it before (microcosm#931): ties by id."""

    ids = np.asarray(inputs["household_ids"])
    weight = np.asarray(inputs["weights"], dtype=float)
    region = np.asarray(inputs["region"]).astype(str)
    values = np.asarray(inputs["value"], dtype=float)
    selected = np.zeros(len(ids), dtype=bool)
    for code in sorted(set(region)):
        pool = np.flatnonzero((region == code) & (values > 0) & (weight > 0))
        order = pool[np.lexsort((ids[pool], values[pool]))]
        cumulative = np.cumsum(weight[order])
        mid_rank = (cumulative - weight[order] / 2.0) / float(weight[order].sum())
        selected[order[mid_rank >= stratum_lower]] = True
    return selected


def test_a_value_tie_straddling_the_stratum_is_ranked_in_keyed_order() -> None:
    # Twenty equal-weight owners, the top three tied on value: at 0.9 the two
    # highest mid-ranks are selected, so the cut falls inside the tie group,
    # and the two of the three ranked last in keyed order are selected, not
    # the two highest ids (microcosm#931).
    values = [*(100_000.0 * (i + 1) for i in range(17)), *([3_000_000.0] * 3)]
    inputs = _owners(values)
    tied = np.asarray([18, 19, 20])
    keyed = tied[np.lexsort((tied, stable_identity_uniforms(tied, **TIE)))]
    assert sorted(keyed[1:].tolist()) != [19, 20]

    selected, (row,) = hv.select_high_value_support_households(
        **inputs, stratum_lower=0.9, **TIE
    )

    ids = inputs["household_ids"]
    assert sorted(ids[selected].tolist()) == sorted(keyed[1:].tolist())
    assert row["households_selected"] == 2 and row["selected_mass"] == 20.0
    assert ids[_id_ranked_stratum(inputs, 0.9)].tolist() == [19, 20]
    for seed in range(5):
        rows = np.random.default_rng(seed).permutation(len(values))
        permuted, (permuted_row,) = hv.select_high_value_support_households(
            **{key: np.asarray(value)[rows] for key, value in inputs.items()},
            stratum_lower=0.9,
            **TIE,
        )
        assert permuted.tolist() == selected[rows].tolist()
        assert permuted_row == row


def test_distinct_values_rank_the_stratum_as_the_id_rank_did_bit_for_bit() -> None:
    # The tie-break only reorders exact ties, so on distinct values every
    # region's stratum is the one the id rank took, down to the household
    # whose mid-rank sits nearest the cut, in each of forty regions
    # (microcosm#931).
    rng = np.random.default_rng(20261109)
    n = 2_000
    inputs = {
        "household_ids": rng.choice(10**8, n, replace=False).astype(np.int64),
        "weights": rng.uniform(1.0, 50.0, n),
        "region": rng.choice([f"R{code:02d}" for code in range(40)], n).astype(object),
        "tenure_type": np.full(n, "OWNED_OUTRIGHT", dtype=object),
        "value": rng.permutation(n) * 1_000.0 + 50_000.0,
    }

    selected, _ = hv.select_high_value_support_households(**inputs, **TIE)

    np.testing.assert_array_equal(
        selected, _id_ranked_stratum(inputs, hv.HIGH_VALUE_SUPPORT_STRATUM_LOWER)
    )
    assert 0 < int(selected.sum()) < n


def test_tied_clones_enter_the_stratum_at_their_share() -> None:
    # Sixty regions, each of seventeen owners below one source household's
    # three geographic clones, tied on value at the top: the stratum takes
    # one clone a region. The rank by id took the last clone every time;
    # keyed on each clone's own id, clone 0 enters at about a third of the
    # regions, inside the four-sigma binomial band (design verifier P10;
    # microcosm#931).
    families, clone_count = 60, 3
    parts: list[dict[str, np.ndarray]] = []
    clone_index: list[np.ndarray] = []
    for family in range(families):
        source = np.int64(10_000 + family)
        clones = np.asarray(
            [
                source,
                *(
                    copy_id_table.mint(np.asarray([source]), "geographic_support", k)[0]
                    for k in range(1, clone_count)
                ),
            ],
            dtype=np.int64,
        )
        below = np.arange(1, 18, dtype=np.int64) + 100 * family
        parts.append(
            _owners(
                [*(1_000.0 * (i + 1) for i in range(17)), *([90_000.0] * clone_count)],
                ids=np.concatenate([below, clones]),
                region=f"R{family:02d}",
            )
        )
        clone_index.append(np.asarray([-1] * 17 + list(range(clone_count))))
    inputs = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    clone = np.concatenate(clone_index)

    selected, rows = hv.select_high_value_support_households(**inputs, **TIE)

    assert [row["households_selected"] for row in rows] == [1] * families
    share = float((clone[selected] == 0).mean())
    band = binomial_share_tolerance(target=1 / clone_count, rows=families, sigma=4.0)
    assert abs(share - 1 / clone_count) <= band
    by_id = _id_ranked_stratum(inputs, hv.HIGH_VALUE_SUPPORT_STRATUM_LOWER)
    assert float((clone[by_id] == clone_count - 1).mean()) == 1.0


def test_the_stage_ranks_on_the_declared_tie_break(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    select = hv.select_high_value_support_households

    def recording(**kwargs):
        calls.append(kwargs)
        return select(**kwargs)

    monkeypatch.setattr(hv, "select_high_value_support_households", recording)
    hv.split_high_value_support_households(_frame(), stratum_lower=0.9)

    ((_, payload),) = hv.high_value_support_operation_parameters()
    assert hv.high_value_support_tie_break() == (payload["seed"], payload["salt"])
    assert hv.high_value_support_tie_break() == (0, "high_value_home_support:value_tie")
    (call,) = calls
    assert (call["seed"], call["salt"]) == hv.high_value_support_tie_break()
    assert payload["stratum_rule"] == (
        "within each FRS region, owner households (tenure owned outright or with "
        "a mortgage, main_residence_value > 0, positive weight) ordered by "
        "main_residence_value ascending, households tied on it in ascending "
        "order of a uniform keyed on their own household_id under seed and salt, "
        "then household_id ascending; a household whose weighted mid-rank "
        "(cumulative weight before it plus half its own, over the region's owner "
        "weight) is at or above stratum_lower is selected"
    )
    assert payload["values"] == (
        "every column copied unchanged; the stratum's tie-break is the stage's "
        "only draw"
    )
    with pytest.raises(TypeError, match="seed"):
        select(**_owners([1.0, 2.0]))


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


#: Owner weights of the correction frame (synthetic), lowest value first.
_CORRECTION_OWNERS = [
    305.47,
    148.26,
    219.59,
    328.65,
    319.35,
    254.19,
    293.18,
    154.34,
    263.21,
    108.39,
    324.55,
    358.98,
    155.78,
    332.99,
]


def _correction_frame():
    """63 heavy renters, a zero-weight one, a light one, then 14 owners.

    Renters are never in the stratum, so the stage leaves them whole; at
    stratum 0.3 the ten most valuable owners become six rows each, and the
    129-row vector the stage corrects is summed as two halves split at row
    64. The 63 heavy renters sit in the first half, which sums inside the
    total's binade, while the second, holding the light renter, the owners
    and the copies, carries exactly half an ulp of the total: round half to
    even keeps every change to a heavy renter's weight off the target, and
    the light one, the 64th whole household by weight, can carry it.
    """
    weights = [*(1_101.0 + i for i in range(63)), 0.0, 1_000.0, *_CORRECTION_OWNERS]
    n = len(weights)
    owners = len(_CORRECTION_OWNERS)
    base = _frame(n)
    household = base.table("household").copy()
    household["tenure_type"] = pd.array(
        ["RENT_PRIVATELY"] * (n - owners) + ["OWNED_OUTRIGHT"] * owners,
        dtype="string",
    )
    household["main_residence_value"] = [150_000.0] * (n - owners) + [
        100_000.0 * (i + 1) for i in range(owners)
    ]
    return uk_national_frame(
        person=base.table("person"),
        benunit=base.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.asarray(weights),
    )


def test_the_correction_passes_over_whole_households_that_cannot_carry_it(
    monkeypatch,
) -> None:
    # The stage corrects through the CGT split's helper (microcosm#931): the
    # 64 heaviest households it leaves whole, its unselected ones, heaviest
    # first. The rule before tried the heaviest only, and the shared fallback
    # then corrected the last copy; now the light renter carries it and every
    # root and copy stays bitwise w / 6.
    from microcosm.build.uk_runtime.spi_support import (
        _exact_total_correction,
        _importance_weights_with_exact_total,
    )

    calls = []
    correct = hv._exact_support_weights

    def recording(values, *, target, whole, **keywords):
        calls.append((np.array(values), target, np.array(whole)))
        return correct(values, target=target, whole=whole, **keywords)

    monkeypatch.setattr(hv, "_exact_support_weights", recording)
    result = hv.split_high_value_support_households(
        _correction_frame(), stratum_lower=0.3, copies=6
    )
    ((values, target, whole),) = calls

    def carries(row: int) -> bool:
        corrected = _exact_total_correction(values, target=target, correction_index=row)
        return corrected is not None

    assert result.households_selected == 10 and len(values) == 129
    assert whole.tolist() == _unselected_positive_rows(result, values)
    heaviest_first = whole[np.lexsort((whole, -values[whole]))]
    assert heaviest_first[:64].tolist() == [*range(62, -1, -1), 64]
    assert float(values.sum()) != target
    assert not any(carries(row) for row in range(63))
    assert carries(64)
    fallback = np.asarray(_importance_weights_with_exact_total(values, target).values)
    assert np.flatnonzero(fallback != values).tolist() == [128]

    weights = np.asarray(result.frame.weights_for("household").values)
    assert result.frame.weights_for("household").total == target
    assert np.flatnonzero(weights != values).tolist() == [64]
    assert weights[64] == pytest.approx(1_000.0, rel=1e-12)
    household = result.frame.table("household")
    copies = household[hv.HIGH_VALUE_SUPPORT_COPIES_COLUMN].to_numpy()
    sources = household["household_id"].to_numpy() % 10**12
    for source in np.unique(sources[copies == 6]):
        family = weights[sources == source]
        weight = _CORRECTION_OWNERS[int(source) - 66]
        assert family.tolist() == [weight / 6] * 6


def _unselected_positive_rows(result, values) -> list[int]:
    """The incumbent rows the stage left whole: unselected, positive weight."""

    household = result.frame.table("household")
    copies = household[hv.HIGH_VALUE_SUPPORT_COPIES_COLUMN].to_numpy()
    incumbent = ~household[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy(dtype=bool)
    return np.flatnonzero(incumbent & (copies == 1) & (values > 0.0)).tolist()


def test_a_selected_root_never_carries_the_correction(monkeypatch) -> None:
    # The stage's whole households are its unselected positive ones
    # (microcosm#931): every household it selects becomes six rows at w / 6,
    # roots included. Here the two most valuable owners are selected, and
    # the root of the 379.53 one, at 379.53 / 6, outweighs both unselected
    # households and could carry the correction, but only they are tried:
    # the heaviest, the 36.21 one, carries it, and every root and copy stays
    # bitwise w / 6.
    from microcosm.build.uk_runtime.spi_support import _exact_total_correction

    weights_in = [29.45, 36.21, 379.53, 37.78]
    base = _frame(len(weights_in))
    household = base.table("household").copy()
    household["tenure_type"] = pd.array(["OWNED_OUTRIGHT"] * 4, dtype="string")
    household["main_residence_value"] = [100_000.0 * (i + 1) for i in range(4)]
    frame = uk_national_frame(
        person=base.table("person"),
        benunit=base.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.asarray(weights_in),
    )
    calls = []
    correct = hv._exact_support_weights

    def recording(values, *, target, whole, **keywords):
        calls.append((np.array(values), target, np.array(whole)))
        return correct(values, target=target, whole=whole, **keywords)

    monkeypatch.setattr(hv, "_exact_support_weights", recording)
    result = hv.split_high_value_support_households(frame, stratum_lower=0.5, copies=6)
    ((values, target, whole),) = calls

    assert result.households_selected == 2
    assert whole.tolist() == _unselected_positive_rows(result, values) == [0, 1]
    assert float(values.sum()) != target
    assert values[2] > max(values[0], values[1])
    assert (
        _exact_total_correction(values, target=target, correction_index=2) is not None
    )
    weights = np.asarray(result.frame.weights_for("household").values)
    assert result.frame.weights_for("household").total == target
    assert np.flatnonzero(weights != values).tolist() == [1]
    sources = result.frame.table("household")["household_id"].to_numpy() % 10**12
    for source, weight in ((3, 379.53), (4, 37.78)):
        assert weights[sources == source].tolist() == [weight / 6] * 6


def test_the_copy_rule_names_the_one_exception() -> None:
    import json
    from pathlib import Path

    from microcosm.build.uk_runtime.cgt_support import (
        CGT_SUPPORT_CORRECTION_CANDIDATES,
        CGT_SUPPORT_EXACT_TOTAL_EXCEPTION,
    )

    ((_, payload),) = hv.high_value_support_operation_parameters()
    expected = (
        "every selected household becomes copies rows at weight / copies, "
        "except where none of the 64 heaviest households left whole can carry "
        "the exact-total correction, when the shared fallback can put it on a "
        "root or copy, off weight / copies by about one ulp of the total weight"
    )
    assert CGT_SUPPORT_CORRECTION_CANDIDATES == 64
    assert payload["copy_rule"] == expected
    assert expected.endswith(CGT_SUPPORT_EXACT_TOTAL_EXCEPTION)
    stage = load_country_spec("uk").sources.stage_map()[
        hv.HIGH_VALUE_SUPPORT_STAGE_NAME
    ]
    (operation,) = stage.operations
    assert operation.parameters["copy_rule"] == expected
    packaged = json.loads(
        (Path(hv.__file__).parents[1] / "uk" / "source_stages.json").read_text()
    )
    (entry,) = [row for row in packaged["stages"] if row["stage"] == stage.stage]
    assert entry["operations"][0]["copy_rule"] == expected


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
    # The manifest declares the stratum's tie-break the stage walks on
    # (microcosm#931).
    (operation,) = stage.operations
    assert (
        operation.parameters["seed"],
        operation.parameters["salt"],
    ) == hv.high_value_support_tie_break()
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
