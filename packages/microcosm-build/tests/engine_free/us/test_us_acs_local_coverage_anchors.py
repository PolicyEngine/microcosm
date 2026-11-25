"""ACS local lane: employer and direct-purchase coverage (microcosm#1720, A6)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import acs_multispine
from microcosm.build.us_runtime.acs_local_coverage_anchors import (
    ACS_DIRECT_PURCHASE_CHOICE_SOURCE,
    ACS_DIRECT_PURCHASE_TYPES,
    ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS,
    ACS_LOCAL_COVERAGE_ANCHORS_ISSUE,
    ACS_LOCAL_COVERAGE_ANCHORS_METHOD,
    acs_local_direct_purchase_donor_shares,
    choose_acs_direct_purchase_types,
    with_acs_local_coverage_anchors,
)
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE, AcsPumsSource
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    required_acs_transfer_inputs,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_ESI, _MARKETPLACE, _OTHER = ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS
TAG = spine_column("person")


def _support_donor(
    *,
    under_65: tuple[bool, bool] = (True, False),
    over_65: tuple[bool, bool] = (False, True),
    n_asec: int = 40,
) -> Frame:
    """An ASEC-by-PUF donor whose ASEC direct purchasers have one type per
    age band; the PUF clones carry the opposite type, which must not count."""

    rng = np.random.default_rng(720)
    n = 2 * n_asec
    ids = np.arange(1, n + 1)
    roles = np.asarray(["asec"] * n_asec + ["puf_tax_detail"] * n_asec, dtype=object)
    age = np.tile(np.where(np.arange(n_asec) % 2 == 0, 40.0, 70.0), 2)
    purchaser = np.tile(np.arange(n_asec) % 4 < 2, 2)
    young = age < 65
    asec = roles == "asec"
    marketplace = np.where(young, under_65[0], over_65[0])
    other = np.where(young, under_65[1], over_65[1])
    marketplace = np.where(asec, marketplace, ~marketplace) & purchaser
    other = np.where(asec, other, ~other) & purchaser
    person = pd.DataFrame(
        {
            "person_id": ids,
            "age": age,
            "is_female": np.tile(rng.integers(0, 2, n_asec).astype(bool), 2),
            "employment_income_before_lsr": np.tile(
                rng.integers(0, 80_000, n_asec).astype(float), 2
            ),
            _ESI: np.tile(rng.random(n_asec) < 0.5, 2),
            _MARKETPLACE: marketplace,
            _OTHER: other,
        }
    )
    for entity in US_SCHEMA.entities:
        if entity != "person":
            person[f"person_{entity}_id"] = ids
    tables = {"person": person}
    for entity in US_SCHEMA.entities:
        if entity != "person":
            tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"]["state_fips"] = np.resize([6, 36, 48], n)
    for entity, table in tables.items():
        table[support_channel_column(entity)] = roles
        table[support_source_id_column(entity)] = np.tile(np.arange(1, n_asec + 1), 2)
        table[support_clone_index_column(entity)] = (roles == "puf_tax_detail").astype(
            int
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(n, 100.0), WeightKind.DESIGN)},
        pd.Series(ASEC_PUF_DONOR_SPINE, index=person.index, dtype=object),
    )


#: Eight ACS persons after the transfer, one per household (SERIALNO H1-H8):
#: (age, HINS1, HINS2, drawn ESI, drawn Marketplace, drawn other direct)
_ROWS = [
    (40, 1, 2, False, True, False),  # ESI from HINS1; Marketplace off (HINS2 2)
    (40, 2, 2, True, False, True),  # ESI off; other direct off
    (40, 1, 1, True, True, False),  # HINS2 1 with a drawn type: kept
    (40, 2, 1, False, False, False),  # HINS2 1, nothing drawn: under-65 type
    (70, 2, 1, False, False, False),  # HINS2 1, nothing drawn: 65+ type
    (70, 1, 1, False, False, True),  # HINS2 1 with a drawn type: kept
    (10, np.nan, 2, True, False, False),  # HINS1 blank: the draw stands
    (30, 2, np.nan, False, True, True),  # HINS2 blank: the draws stand
]


def _acs_frame(rows=_ROWS, **person_overrides: Any) -> Frame:
    ids = np.arange(1, len(rows) + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                f"person_{entity}_id": ids
                for entity in US_SCHEMA.entities
                if entity != "person"
            },
            "SPORDER": 1,
            "age": [row[0] for row in rows],
            "HINS1": [row[1] for row in rows],
            "HINS2": [row[2] for row in rows],
            _ESI: [row[3] for row in rows],
            _MARKETPLACE: [row[4] for row in rows],
            _OTHER: [row[5] for row in rows],
            # HIMRKS 0 (categorically ineligible for a subsidy): no rule reads
            # it, so the earlier rule decides every row.
            "ACS_HIMRKS": 0,
            **person_overrides,
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.entities
            if entity != "person"
        },
    }
    tables["household"]["SERIALNO"] = [f"2024HU{i:07d}" for i in ids]
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.full(len(ids), 10.0), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def test_the_three_columns_are_shared_transfer_targets() -> None:
    """The step overrides the shared draws; it does not change the plan."""

    assert set(ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS) <= required_acs_transfer_inputs()


def test_the_donor_shares_come_from_the_asec_role_by_age_band() -> None:
    shares = acs_local_direct_purchase_donor_shares(_support_donor())
    assert shares["channel"] == "asec"
    assert shares["bands"]["under_65"]["shares"] == {
        "marketplace_only": 1.0,
        "non_marketplace_only": 0.0,
        "both": 0.0,
    }
    assert shares["bands"]["65_plus"]["shares"] == {
        "marketplace_only": 0.0,
        "non_marketplace_only": 1.0,
        "both": 0.0,
    }
    assert shares["all_ages"] == pytest.approx(
        {"marketplace_only": 0.5, "non_marketplace_only": 0.5, "both": 0.0}
    )
    for band in shares["bands"].values():
        assert band["fallback_to_all_ages"] is False


def test_the_acs_items_replace_the_draws() -> None:
    frame = _acs_frame()
    result, receipt = with_acs_local_coverage_anchors(frame, _support_donor(), seed=7)
    person = result.table("person")

    assert person[_ESI].tolist() == [True, False, True, False, False, True, True, False]
    assert person[_MARKETPLACE].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    assert person[_OTHER].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
    ]
    for column in ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS:
        assert person[column].dtype == bool, column
    # Every other column, and the input frame, are untouched.
    other_columns = [c for c in person if c not in ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS]
    pd.testing.assert_frame_equal(
        person[other_columns], frame.table("person")[other_columns]
    )
    assert frame.table("person")[_ESI].tolist()[1] is True

    assert receipt["issue"] == ACS_LOCAL_COVERAGE_ANCHORS_ISSUE == "microcosm#1720"
    assert receipt["method"] == ACS_LOCAL_COVERAGE_ANCHORS_METHOD
    assert receipt["acs_persons"] == 8
    assert receipt["hins1_rows"] == {"1": 3, "2": 4, "blank": 1}
    assert receipt["hins2_rows"] == {"1": 4, "2": 3, "blank": 1}
    esi = receipt[_ESI]
    # Rows with a HINS1 code: 3 yes (one drawn too), 4 no (one drawn yes).
    assert esi["agreement_with_draw"] == {
        "both_true_rows": 1,
        "native_only_rows": 2,
        "drawn_only_rows": 1,
        "both_false_rows": 3,
    }
    assert esi["changed_rows"] == 3
    assert esi["native_weighted_share"] == pytest.approx(4 / 8)
    assert esi["drawn_weighted_share"] == pytest.approx(3 / 8)
    direct = receipt["direct_purchase"]
    assert direct["hins2_no_rows"] == 3
    assert direct["set_false_from_draw"] == {_MARKETPLACE: 1, _OTHER: 1}
    assert direct["hins2_yes_rows"] == 4
    choice = direct["choice"]
    assert choice["source"] == ACS_DIRECT_PURCHASE_CHOICE_SOURCE
    assert choice["rows"] == 4
    assert choice["draw_kept_rows"] == 2
    assert choice["type_drawn_rows"] == 2
    assert choice["type_drawn_by_type"] == {
        "marketplace_only": 1,
        "non_marketplace_only": 1,
        "both": 0,
    }
    assert choice["unresolved_rows"] == 0
    assert choice["draw_key"] == "acs_2024_1yr:SERIALNO:SPORDER"
    assert choice["seed"] == 7
    assert choice["himrks"]["codes"] == {"0": 4, "blank": 0}
    assert choice["set_marketplace_by_code_1_rows"] == 0
    assert direct["himrks_conflicts"] == {
        "hins2_no_himrks_direct_rows": {"1": 0, "2": 0},
        "hins2_no_himrks_direct_weighted_share": 0.0,
        "hins2_blank_himrks_direct_rows": 0,
    }
    assert set(receipt["not_used"]) == {
        "HIMRKS code 0",
        "HIMRKS code 2",
        "subsidized-Marketplace anchor",
    }
    json.dumps(receipt, allow_nan=False)


def test_the_choice_is_one_function_over_the_direct_purchasers() -> None:
    """The Marketplace-versus-other choice, alone. Without a HIMRKS code 1,
    draws with a type stand; rows with none take the donor's type for their
    age band; a band without shares stays unresolved."""

    rows = pd.DataFrame(
        {
            "person_household_id": [1, 2, 3, 4, 5],
            "SPORDER": [1, 1, 1, 1, 2],
            "age": [40.0, 70.0, 30.0, 80.0, 50.0],
            # Codes 0 and 2 and a blank: the earlier rule decides every row.
            "ACS_HIMRKS": [0, 2, 0, np.nan, 0],
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2, 3, 4, 5], "SERIALNO": ["A", "B", "C", "D", "E"]}
    )
    shares = {
        "bands": {
            "under_65": {
                "shares": {
                    "marketplace_only": 0.0,
                    "non_marketplace_only": 0.0,
                    "both": 1.0,
                }
            },
            "65_plus": {"shares": None},
        }
    }
    marketplace, other, receipt = choose_acs_direct_purchase_types(
        rows,
        household,
        drawn_marketplace=np.asarray([True, False, False, False, False]),
        drawn_other=np.asarray([False, True, False, False, False]),
        donor_shares=shares,
        seed=11,
        weights=np.ones(5),
    )
    assert marketplace.tolist() == [True, False, True, False, True]
    assert other.tolist() == [False, True, True, False, True]
    assert receipt == {
        "source": ACS_DIRECT_PURCHASE_CHOICE_SOURCE,
        "rows": 5,
        "himrks": {
            "codes": {"0": 3, "2": 1, "blank": 1},
            "weighted_shares": {"0": 0.6, "1": 0.0, "2": 0.2, "blank": 0.2},
            "allocation_flag_loaded": False,
            "allocated_rows": None,
            "allocated_code_1_rows": None,
            "code_1": {
                "rows": 0,
                "drawn_marketplace_true_rows": 0,
                "drawn_marketplace_false_rows": 0,
                "drawn_marketplace_weighted_share": 0.0,
                "drawn_non_marketplace_kept_rows": 0,
                "drawn_non_marketplace_only_became_both_rows": 0,
            },
        },
        "set_marketplace_by_code_1_rows": 0,
        "draw_kept_rows": 2,
        "type_drawn_rows": 2,
        "type_drawn_by_type": {
            "marketplace_only": 0,
            "non_marketplace_only": 0,
            "both": 2,
        },
        "unresolved_rows": 1,
        "donor_shares": shares,
        "draw_key": "acs_2024_1yr:SERIALNO:SPORDER",
        "draw_salt": "acs_local_direct_purchase_type",
        "seed": 11,
    }
    # No row needs a draw: no key is built, so no household is needed.
    kept_marketplace, kept_other, kept = choose_acs_direct_purchase_types(
        rows.iloc[:2],
        household.iloc[:0],
        drawn_marketplace=np.asarray([True, False]),
        drawn_other=np.asarray([False, True]),
        donor_shares=shares,
        seed=11,
        weights=np.ones(2),
    )
    assert (kept_marketplace.tolist(), kept_other.tolist()) == (
        [True, False],
        [False, True],
    )
    assert kept["type_drawn_rows"] == 0


def test_himrks_code_1_sets_the_marketplace_flag_and_keeps_the_other_draw() -> None:
    """As ruled on 25 Nov 2026: HIMRKS 1 (direct purchase with a premium
    subsidy) sets the Marketplace flag; the drawn non-Marketplace flag stands.
    Codes 0 and 2 and a blank keep the earlier rule."""

    rows = pd.DataFrame(
        {
            "person_household_id": [1, 2, 3, 4, 5, 6],
            "SPORDER": 1,
            "age": [40.0, 40.0, 40.0, 40.0, 40.0, 70.0],
            "ACS_HIMRKS": [1, 1, 1, 2, 0, np.nan],
            "ACS_FHIMRKSP": [1, 0, 0, 0, 0, 0],
        }
    )
    household = pd.DataFrame(
        {"household_id": [1, 2, 3, 4, 5, 6], "SERIALNO": list("ABCDEF")}
    )
    shares = {
        "bands": {
            "under_65": {
                "shares": {
                    "marketplace_only": 0.0,
                    "non_marketplace_only": 0.0,
                    "both": 1.0,
                }
            },
            "65_plus": {"shares": None},
        }
    }
    marketplace, other, receipt = choose_acs_direct_purchase_types(
        rows,
        household,
        # Code 1 rows: nothing drawn; a non-Marketplace draw; a Marketplace draw.
        drawn_marketplace=np.asarray([False, False, True, False, False, False]),
        drawn_other=np.asarray([False, True, False, False, True, False]),
        donor_shares=shares,
        seed=11,
        weights=np.arange(1.0, 7.0),
    )
    assert marketplace.tolist() == [True, True, True, True, False, False]
    assert other.tolist() == [False, True, False, True, True, False]
    himrks = receipt["himrks"]
    assert himrks["codes"] == {"0": 1, "1": 3, "2": 1, "blank": 1}
    assert himrks["weighted_shares"] == pytest.approx(
        {"0": 5 / 21, "1": 6 / 21, "2": 4 / 21, "blank": 6 / 21}
    )
    assert himrks["allocation_flag_loaded"] is True
    assert (himrks["allocated_rows"], himrks["allocated_code_1_rows"]) == (1, 1)
    assert himrks["code_1"] == {
        "rows": 3,
        "drawn_marketplace_true_rows": 1,
        "drawn_marketplace_false_rows": 2,
        "drawn_marketplace_weighted_share": pytest.approx(3 / 6),
        "drawn_non_marketplace_kept_rows": 1,
        "drawn_non_marketplace_only_became_both_rows": 1,
    }
    assert receipt["set_marketplace_by_code_1_rows"] == 2
    # Code 0 with a drawn type: kept. Code 2 with nothing drawn: the donor's
    # under-65 type. The blank 70-year-old: no 65+ shares, so unresolved.
    assert receipt["draw_kept_rows"] == 1
    assert receipt["type_drawn_rows"] == 1
    assert receipt["type_drawn_by_type"]["both"] == 1
    assert receipt["unresolved_rows"] == 1
    json.dumps(receipt, allow_nan=False)

    with pytest.raises(ValueError, match=r"FHIMRKSP requires blank"):
        choose_acs_direct_purchase_types(
            rows.assign(ACS_FHIMRKSP=[2, 0, 0, 0, 0, 0]),
            household,
            drawn_marketplace=np.zeros(6, dtype=bool),
            drawn_other=np.zeros(6, dtype=bool),
            donor_shares=shares,
            seed=11,
            weights=np.ones(6),
        )


def test_hins2_no_wins_over_a_himrks_direct_purchase_code_and_is_counted() -> None:
    result, receipt = with_acs_local_coverage_anchors(
        _acs_frame(ACS_HIMRKS=[1, 2, 1, 1, 1, 0, 0, 1]), _support_donor(), seed=7
    )
    person = result.table("person")
    # Rows 1 and 2 have HINS2 2: both flags False, whatever HIMRKS says.
    assert not person[_MARKETPLACE].iloc[:2].any()
    assert not person[_OTHER].iloc[:2].any()
    # Rows 4 and 5 have HINS2 1, nothing drawn and code 1: Marketplace only
    # (row 5 is 70, where the donor's type would be non-Marketplace).
    assert person[_MARKETPLACE].iloc[3:5].tolist() == [True, True]
    assert person[_OTHER].iloc[3:5].tolist() == [False, False]
    # Row 6 has HINS2 1, a drawn non-Marketplace type and code 0: kept.
    assert (person[_MARKETPLACE].iloc[5], person[_OTHER].iloc[5]) == (False, True)
    conflicts = receipt["direct_purchase"]["himrks_conflicts"]
    assert conflicts["hins2_no_himrks_direct_rows"] == {"1": 1, "2": 1}
    assert conflicts["hins2_no_himrks_direct_weighted_share"] == pytest.approx(2 / 8)
    # Row 8 has a blank HINS2 and code 1: its draws stand, and it is counted.
    assert conflicts["hins2_blank_himrks_direct_rows"] == 1
    assert (person[_MARKETPLACE].iloc[7], person[_OTHER].iloc[7]) == (True, True)
    choice = receipt["direct_purchase"]["choice"]
    assert choice["himrks"]["codes"] == {"0": 1, "1": 3, "blank": 0}
    assert choice["set_marketplace_by_code_1_rows"] == 2
    assert "subsidized-Marketplace anchor" in receipt["not_used"]


def _mixed_rows(n: int = 40) -> list[tuple]:
    """HINS2 yes and nothing drawn, all under 65."""

    return [(30 + i % 20, 2, 1, False, False, False) for i in range(n)]


def test_the_type_draw_is_keyed_on_the_record_not_the_row_order() -> None:
    donor = _support_donor(under_65=(True, True), over_65=(True, False))
    # Make the under-65 types mixed: half Marketplace only, half both.
    tables = {entity: donor.table(entity).copy() for entity in donor.entities}
    person = tables["person"]
    young_buyers = person.index[
        (person["age"] < 65) & (person[_MARKETPLACE] | person[_OTHER])
    ]
    person.loc[young_buyers[::2], _OTHER] = False
    mixed = Frame(
        tables, US_SCHEMA, {"household": donor.weights_for("household")}, donor.strata
    )
    frame = _acs_frame(_mixed_rows())
    first, receipt = with_acs_local_coverage_anchors(frame, mixed, seed=3)
    again, _ = with_acs_local_coverage_anchors(frame, mixed, seed=3)
    types = receipt["direct_purchase"]["choice"]["type_drawn_by_type"]
    assert types["marketplace_only"] > 0 and types["both"] > 0
    assert types["non_marketplace_only"] == 0
    pd.testing.assert_frame_equal(first.table("person"), again.table("person"))

    # Half the records removed: each remaining person keeps their type, since
    # the draw depends only on the seed and their SERIALNO and SPORDER.
    kept = frame.table("household")["household_id"] % 2 == 1
    kept_ids = set(frame.table("household").loc[kept, "household_id"])
    subset_tables = {
        entity: frame.table(entity)
        .loc[lambda table, entity=entity: table[f"{entity}_id"].isin(kept_ids)]
        .reset_index(drop=True)
        for entity in frame.entities
    }
    subset = Frame(
        subset_tables,
        US_SCHEMA,
        {
            "household": Weights(
                frame.weights_for("household").values[kept.to_numpy()],
                WeightKind.DESIGN,
            )
        },
    )
    fewer, _ = with_acs_local_coverage_anchors(subset, mixed, seed=3)
    by_id = fewer.table("person").set_index("person_id")
    original = first.table("person").set_index("person_id").loc[by_id.index]
    for column in (_MARKETPLACE, _OTHER):
        pd.testing.assert_series_equal(by_id[column], original[column])


def test_a_band_without_donor_purchasers_falls_back_to_all_ages() -> None:
    donor = _support_donor(over_65=(False, False))
    shares = acs_local_direct_purchase_donor_shares(donor)
    assert shares["bands"]["65_plus"]["fallback_to_all_ages"] is True
    assert shares["bands"]["65_plus"]["shares"] == shares["all_ages"]
    result, receipt = with_acs_local_coverage_anchors(_acs_frame(), donor, seed=7)
    # The 65+ row with nothing drawn takes the all-ages type (Marketplace only).
    assert result.table("person")[_MARKETPLACE].iloc[4] == True  # noqa: E712
    assert receipt["direct_purchase"]["choice"]["unresolved_rows"] == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"HINS1": [1, 2, 1, 2, 2, 1, 3, 2]}, r"HINS1 requires blank"),
        ({"HINS2": [2, 2, 1, 1, 1, 1, 2, 0]}, r"HINS2 requires blank"),
        (
            {_ESI: [True, None, True, False, False, True, True, False]},
            "complete boolean",
        ),
        ({"ACS_HIMRKS": [0, 0, 0, 0, 0, 0, 0, 3]}, r"HIMRKS requires blank"),
    ],
)
def test_malformed_inputs_are_refused(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        with_acs_local_coverage_anchors(
            _acs_frame(**overrides), _support_donor(), seed=7
        )


@pytest.mark.parametrize(
    "column", ["HINS1", "HINS2", "ACS_HIMRKS", _MARKETPLACE, "age"]
)
def test_a_missing_column_is_refused(column) -> None:
    frame = _acs_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        with_acs_local_coverage_anchors(
            Frame(
                tables,
                US_SCHEMA,
                {"household": frame.weights_for("household")},
                frame.strata,
            ),
            _support_donor(),
            seed=7,
        )


def test_a_donor_without_the_asec_role_is_refused() -> None:
    donor = _support_donor()
    plain = {
        entity: donor.table(entity).drop(
            columns=[
                support_channel_column(entity),
                support_source_id_column(entity),
                support_clone_index_column(entity),
            ]
        )
        for entity in donor.entities
    }
    with pytest.raises(ValueError, match="ASEC observation role"):
        with_acs_local_coverage_anchors(
            _acs_frame(),
            Frame(
                plain,
                US_SCHEMA,
                {"household": donor.weights_for("household")},
                donor.strata,
            ),
            seed=7,
        )


# ---------------------------------------------------------------------------
# The multispine seam
# ---------------------------------------------------------------------------


def test_the_seam_runs_the_step_after_the_transfers_and_before_the_premiums(
    monkeypatch, tmp_path
) -> None:
    events: list[tuple[Any, ...]] = []
    transferred = SimpleNamespace(table=lambda entity: pd.DataFrame())
    anchored = object()
    attributed = object()
    base = SimpleNamespace(table=lambda entity: pd.DataFrame())

    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda source, *, chunksize: (object(), {"spine": ACS_2024_1YR_SPINE}),
    )
    monkeypatch.setattr(
        acs_multispine,
        "map_acs_native_inputs",
        lambda frame: SimpleNamespace(frame=object(), native_inputs={}),
    )
    monkeypatch.setattr(
        acs_multispine,
        "transfer_acs_inputs",
        lambda recipient, donor, **kwargs: (
            events.append(("transfer",))
            or SimpleNamespace(
                frame=transferred,
                imputed_inputs=(),
                fit_records=(),
                deferred_inputs=(),
                resolved_donor_channel="puf_support",
            )
        ),
    )

    def fake_anchor(frame, donor, *, seed):
        events.append(("coverage", frame, donor, seed))
        return anchored, {"issue": "microcosm#1720", "rows": np.int64(1)}

    def fake_attribute(frame):
        events.append(("se_health", frame))
        return attributed, {"issue": "microcosm#1720"}

    monkeypatch.setattr(acs_multispine, "with_acs_local_coverage_anchors", fake_anchor)
    monkeypatch.setattr(
        acs_multispine, "with_acs_local_se_health_premiums", fake_attribute
    )
    monkeypatch.setattr(
        acs_multispine,
        "with_optional_acs_spine",
        lambda actual_base, actual_acs, **kwargs: (
            events.append(("pool", actual_acs)) or object()
        ),
    )
    source = AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip")
    result = acs_multispine.build_optional_acs_multispine(
        cast(Frame, base),
        source,
        coverage_anchors=True,
        se_health_premiums=True,
        seed=9,
    )
    # The premium attribution reads the frame the coverage step returned.
    assert events == [
        ("transfer",),
        ("coverage", transferred, base, 9),
        ("se_health", anchored),
        ("pool", attributed),
    ]
    assert result.provenance["acs_local_coverage_anchors"] == {
        "issue": "microcosm#1720",
        "rows": 1,
    }

    events.clear()
    result = acs_multispine.build_optional_acs_multispine(cast(Frame, base), source)
    assert events == [("transfer",), ("pool", transferred)]
    assert "acs_local_coverage_anchors" not in result.provenance


def _raw_acs(n: int = 20) -> Frame:
    """A raw ACS spine of one-person households with HINS1, HINS2 and
    HIMRKS (as the loader renames it)."""

    rng = np.random.default_rng(2024)
    households = np.arange(600, 600 + n)
    person = pd.DataFrame(
        {
            "person_id": households,
            "person_household_id": households,
            "person_tax_unit_id": households,
            "person_spm_unit_id": households,
            "person_family_id": households,
            "person_marital_unit_id": households,
            "SPORDER": 1,
            "AGEP": np.where(np.arange(n) % 3 == 0, 70, 35),
            "SEX": rng.integers(1, 3, n),
            "RELSHIPP": 20,
            "ADJINC": 1_000_000,
            "WAGP": rng.integers(0, 90_000, n).astype(float),
            "SEMP": 0.0,
            "SSP": 0.0,
            "SSIP": 0.0,
            "RETP": 0.0,
            "INTP": 0.0,
            "HINS1": np.resize([1, 2, 2], n),
            "HINS2": np.resize([2, 1, 2, 1], n),
            "ACS_HIMRKS": np.resize([0, 1, 2, 1, 0], n),
            "ACS_FHIMRKSP": 0,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": households,
                    "SERIALNO": [f"2024HU{i:07d}" for i in households],
                    "state_fips": np.resize([6, 36, 48], n),
                    "puma": "0600100",
                    "ADJHSG": 1_000_000,
                    "TEN": 3,
                    "RNTP": 1_000.0,
                    "GRNTP": 1_200.0,
                    "TAXAMT": np.nan,
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": households}),
            "spm_unit": pd.DataFrame({"spm_unit_id": households}),
            "family": pd.DataFrame({"family_id": households}),
            "marital_unit": pd.DataFrame({"marital_unit_id": households}),
        },
        US_SCHEMA,
        {"household": Weights(np.full(n, 50.0), WeightKind.DESIGN)},
        pd.Series(ACS_2024_1YR_SPINE, index=person.index, dtype=object),
    )


def _real_seam(monkeypatch, tmp_path, *, coverage_anchors: bool):
    monkeypatch.setattr(
        acs_multispine, "build_acs_pums_unit_frame", lambda *_a, **_k: (_raw_acs(), {})
    )
    return acs_multispine.build_optional_acs_multispine(
        _support_donor(),
        AcsPumsSource(tmp_path / "csv_hus.zip", tmp_path / "csv_pus.zip"),
        target_families={
            "person": {"coverage": tuple(ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS)}
        },
        coverage_anchors=coverage_anchors,
        seed=5,
        n_estimators=5,
    )


def test_a_real_transfer_keeps_every_other_draw_and_takes_the_acs_items(
    monkeypatch, tmp_path
) -> None:
    """The shared transfer's draws (and every other column) are identical
    with and without the step; only the three columns follow HINS1/HINS2."""

    with_step = _real_seam(monkeypatch, tmp_path, coverage_anchors=True)
    without_step = _real_seam(monkeypatch, tmp_path, coverage_anchors=False)

    columns = list(ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS)
    pd.testing.assert_frame_equal(
        with_step.frame.table("person").drop(columns=columns),
        without_step.frame.table("person").drop(columns=columns),
        check_like=True,
    )
    assert (
        with_step.provenance["imputed_inputs"]
        == without_step.provenance["imputed_inputs"]
    )
    assert with_step.fit_records == without_step.fit_records

    person = with_step.frame.table("person")
    acs = person.loc[person[TAG].eq(ACS_2024_1YR_SPINE)].reset_index(drop=True)
    donor = person.loc[person[TAG].eq(ASEC_PUF_DONOR_SPINE)].reset_index(drop=True)
    before = without_step.frame.table("person")
    donor_before = before.loc[before[TAG].eq(ASEC_PUF_DONOR_SPINE)].reset_index(
        drop=True
    )
    assert acs[_ESI].tolist() == (acs["HINS1"] == 1).tolist()
    no_direct = acs["HINS2"].eq(2)
    assert not acs.loc[no_direct, _MARKETPLACE].any()
    assert not acs.loc[no_direct, _OTHER].any()
    assert (acs.loc[~no_direct, _MARKETPLACE] | acs.loc[~no_direct, _OTHER]).all()
    subsidized = ~no_direct & acs["ACS_HIMRKS"].eq(1)
    assert subsidized.any() and acs.loc[subsidized, _MARKETPLACE].all()
    # The raw item stays on the staged ACS rows, for a later stage.
    assert acs["ACS_HIMRKS"].notna().all()
    # Donor rows keep the donor release's values.
    for column in columns:
        assert donor[column].tolist() == donor_before[column].tolist(), column
    receipt = with_step.provenance["acs_local_coverage_anchors"]
    assert receipt["acs_persons"] == 20
    assert set(ACS_DIRECT_PURCHASE_TYPES) == set(
        receipt["direct_purchase"]["choice"]["type_drawn_by_type"]
    )
    json.dumps(with_step.provenance, allow_nan=False)
