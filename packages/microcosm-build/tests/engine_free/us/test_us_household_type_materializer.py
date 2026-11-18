"""Household roster classification, independent pandas oracle and partitions."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.household_type_targets import (
    HOUSEHOLD_TYPE_MATERIALIZER,
    HouseholdUniverseFallbackWarning,
    household_type_household_values,
    household_universe_receipt,
)
from test_support.microcosm_build.us_batched_target_materialization import (
    _install_fake_engine,
    _load_builder_module,
    _nested_frame,
    _spec,
)
from test_support.microcosm_build.us_release_head_to_head_scorer import (
    _load_head_to_head_module,
)

# Most rosters here carry neither TYPEHUGQ nor HRHTYPE on purpose. The tests
# about the fallback assert its warning with pytest.warns, and the ones that
# must not fall back turn it into an error.
pytestmark = pytest.mark.filterwarnings(
    "ignore::microcosm.build.us_runtime.household_type_targets."
    "HouseholdUniverseFallbackWarning"
)


@dataclass
class _Frame:
    tables: dict[str, pd.DataFrame]

    def table(self, entity: str) -> pd.DataFrame:
        return self.tables[entity]


def _frame(
    rosters: list[list[tuple[int, int, float, bool]]],
    *,
    kinds: list[float] | None = None,
) -> _Frame:
    households = pd.DataFrame(
        {
            "household_id": [10 + 11 * i for i in range(len(rosters))],
            "state_fips": [6 if i % 2 else 48 for i in range(len(rosters))],
        }
    )
    if kinds is not None:
        households["TYPEHUGQ"] = kinds
    people = []
    for i, roster in enumerate(rosters):
        for relationship, marital, age, female in roster:
            people.append(
                {
                    "person_household_id": int(households.iloc[i].household_id),
                    "A_EXPRRP": relationship,
                    "A_MARITL": marital,
                    "age": age,
                    "is_female": female,
                }
            )
    return _Frame({"person": pd.DataFrame(people), "household": households})


def _values(frame: _Frame, household_type: str, own_children: str = "all", **extra):
    return household_type_household_values(
        frame,
        {"household_type": household_type, "own_children": own_children, **extra},
    )


def test_classifies_each_family_type_and_living_alone() -> None:
    frame = _frame(
        [
            [(1, 1, 40, False), (4, 1, 40, True), (5, 7, 8, False)],
            [(1, 5, 40, False), (5, 7, 5, True)],
            [(1, 5, 40, True), (5, 7, 17, True)],
            [(2, 7, 70, True)],
            [(2, 7, 40, False), (12, 7, 35, True)],
            [(1, 1, 50, True), (3, 1, 50, False)],
        ]
    )
    assert HOUSEHOLD_TYPE_MATERIALIZER == "household_type"
    np.testing.assert_array_equal(_values(frame, "all"), [1, 1, 1, 1, 1, 1])
    for kind, expected in (
        ("married_couple_family", [1, 0, 0, 0, 0, 1]),
        ("male_householder_no_spouse", [0, 1, 0, 0, 0, 0]),
        ("female_householder_no_spouse", [0, 0, 1, 0, 0, 0]),
        ("householder_living_alone", [0, 0, 0, 1, 0, 0]),
        ("nonfamily", [0, 0, 0, 1, 1, 0]),
    ):
        np.testing.assert_array_equal(_values(frame, kind), expected)
    np.testing.assert_array_equal(
        _values(frame, "married_couple_family", "with_own_children_under_18"),
        [1, 0, 0, 0, 0, 0],
    )


def test_own_children_excludes_adults_ever_married_and_other_relations() -> None:
    frame = _frame(
        [
            [(1, 5, 40, False), (5, 7, 18, True)],
            [(1, 5, 40, False), (5, 5, 17, True)],
            [(1, 5, 40, False), (9, 7, 5, True)],
            [(1, 5, 40, False), (11, 7, 5, True)],
            [(1, 5, 40, False), (5, 7, 17.99, True)],
        ]
    )
    np.testing.assert_array_equal(
        _values(frame, "male_householder_no_spouse", "with_own_children_under_18"),
        [0, 0, 0, 0, 1],
    )


def test_group_quarters_stay_outside_every_household_indicator() -> None:
    frame = _frame(
        [[(2, 7, 30, False)], [(14, 7, 30, True)], [(14, 7, 70, False)]],
        kinds=[1, 2, 3],
    )
    np.testing.assert_array_equal(_values(frame, "all"), [1, 0, 0])
    np.testing.assert_array_equal(_values(frame, "householder_living_alone"), [1, 0, 0])
    np.testing.assert_array_equal(_values(frame, "nonfamily"), [1, 0, 0])
    # Mixed support leaves TYPEHUGQ absent on ASEC rows.
    frame.tables["household"].loc[0, "TYPEHUGQ"] = np.nan
    np.testing.assert_array_equal(_values(frame, "all"), [1, 0, 0])


@pytest.mark.parametrize("group_quarters_kind", [9, 10])
def test_native_asec_group_quarters_with_householders_are_not_households(
    group_quarters_kind: int,
) -> None:
    frame = _frame(
        [
            [(2, 7, 30, False)],
            [(2, 7, 30, False)],
            [(1, 1, 40, False), (4, 1, 40, True), (5, 7, 8, False)],
        ]
    )
    frame.tables["household"]["HRHTYPE"] = [6, group_quarters_kind, group_quarters_kind]
    for kind in (
        "all",
        "nonfamily",
        "householder_living_alone",
    ):
        np.testing.assert_array_equal(_values(frame, kind), [1, 0, 0])
    for kind in (
        "family",
        "married_couple_family",
        "male_householder_no_spouse",
        "female_householder_no_spouse",
        "householder_not_living_alone",
    ):
        np.testing.assert_array_equal(_values(frame, kind), [0, 0, 0])
        np.testing.assert_array_equal(
            _values(frame, kind, "with_own_children_under_18"), [0, 0, 0]
        )


def _no_fallback_values(frame: _Frame, household_type: str, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HouseholdUniverseFallbackWarning)
        return _values(frame, household_type, **extra)


def test_historical_asec_frame_without_hrhtype_still_uses_roster() -> None:
    # A code-2 reference person in native GQ is indistinguishable from a
    # housing householder on today's certified frame, which lacks HRHTYPE.
    # The fallback keeps its numbers, warns, and is recorded.
    frame = _frame([[(2, 7, 30, False)], [(2, 7, 30, True)]])
    frame.tables["household"]["household_spine"] = "asec_puf"
    indicators = {}
    for kind in ("all", "householder_living_alone", "nonfamily"):
        with pytest.warns(HouseholdUniverseFallbackWarning, match="roster alone"):
            indicators[kind] = _values(frame, kind)
        np.testing.assert_array_equal(indicators[kind], [1, 1])
    receipt = household_universe_receipt(
        frame, weights=[3.0, 5.0], indicators=indicators
    )
    assert receipt["households"] == {"TYPEHUGQ": 0, "HRHTYPE": 0, "roster_fallback": 2}
    assert receipt["weighted_households"] == {
        "TYPEHUGQ": 0.0,
        "HRHTYPE": 0.0,
        "roster_fallback": 8.0,
    }
    for kind in indicators:
        assert receipt["targets"][kind] == {
            "weighted_value": 8.0,
            "roster_fallback_weighted_value": 8.0,
        }


def test_mixed_native_acs_and_asec_group_quarters() -> None:
    frame = _frame(
        [[(14, 7, 30, False)], [(2, 7, 30, False)], [(2, 7, 30, False)]],
        kinds=[2, np.nan, np.nan],
    )
    frame.tables["household"]["HRHTYPE"] = [np.nan, 10, 6]
    frame.tables["household"]["household_spine"] = [
        "acs_2024_1yr",
        "asec_puf",
        "asec_puf",
    ]
    np.testing.assert_array_equal(_no_fallback_values(frame, "all"), [0, 0, 1])


@pytest.mark.parametrize(
    ("column", "channel"),
    [("household_spine", "asec_puf"), ("household_support_channel", "asec")],
)
@pytest.mark.parametrize("missing", [np.nan, None])
def test_asec_household_without_hrhtype_refuses_once_the_frame_carries_it(
    column: str, channel: str, missing
) -> None:
    frame = _frame([[(2, 7, 30, False)], [(2, 7, 30, False)]])
    frame.tables["household"]["HRHTYPE"] = pd.Series([6, missing], dtype=object)
    frame.tables["household"][column] = channel
    with pytest.raises(ValueError, match=r"observed HRHTYPE.*household ids \[21\]"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("column", "channel"),
    [("household_spine", "asec_puf"), ("household_support_channel", "asec")],
)
def test_asec_arm_cannot_borrow_typehugq_for_a_missing_hrhtype(
    column: str, channel: str
) -> None:
    # The second ASEC row carries a stray TYPEHUGQ; its arm still requires
    # HRHTYPE, so the stray code cannot classify it. Here the arm alone
    # decides, so each arm column must carry it: today's certified frame
    # names its arm only in household_support_channel.
    frame = _frame([[(2, 7, 30, False)], [(2, 7, 30, False)]], kinds=[np.nan, 1])
    frame.tables["household"]["HRHTYPE"] = [6, np.nan]
    frame.tables["household"][column] = channel
    with pytest.raises(ValueError, match=r"observed HRHTYPE.*household ids \[21\]"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("column", "channels"),
    [
        ("household_spine", ["acs_2024_1yr", "acs_2024_1yr", "asec_puf"]),
        ("household_support_channel", ["acs", "acs", "asec"]),
    ],
)
def test_acs_household_without_hrhtype_does_not_refuse(
    column: str, channels: list[str]
) -> None:
    frame = _frame(
        [[(2, 7, 30, False)], [(14, 7, 30, True)], [(2, 7, 30, True)]],
        kinds=[1, 2, np.nan],
    )
    frame.tables["household"]["HRHTYPE"] = [np.nan, np.nan, 9]
    frame.tables["household"][column] = channels
    np.testing.assert_array_equal(_no_fallback_values(frame, "all"), [1, 0, 0])


def test_the_spine_decides_the_asec_arm_over_the_support_channel() -> None:
    # An ACS row whose spine names it but whose support channel a clone
    # stamped "asec" keeps its TYPEHUGQ classification, as the SPM universe
    # producer reads the spine first; a real ASEC row still needs HRHTYPE.
    frame = _frame([[(2, 7, 30, False)], [(2, 7, 30, False)]], kinds=[1, np.nan])
    frame.tables["household"]["HRHTYPE"] = [np.nan, 6]
    frame.tables["household"]["household_spine"] = ["acs_2024_1yr", "asec_puf"]
    frame.tables["household"]["household_support_channel"] = ["asec", "asec"]
    np.testing.assert_array_equal(_no_fallback_values(frame, "all"), [1, 1])
    frame.tables["household"]["HRHTYPE"] = [np.nan, np.nan]
    with pytest.raises(ValueError, match=r"observed HRHTYPE.*household ids \[21\]"):
        _values(frame, "all")


def test_frame_carrying_hrhtype_without_provenance_classifies_each_household() -> None:
    # Without an arm column, TYPEHUGQ alone marks an ACS record; a household
    # with neither code has no classification left but the roster, which a
    # frame built with the HRHTYPE carry must not need.
    frame = _frame([[(2, 7, 30, False)], [(2, 7, 30, False)]], kinds=[1, np.nan])
    frame.tables["household"]["HRHTYPE"] = [np.nan, 6]
    np.testing.assert_array_equal(_no_fallback_values(frame, "all"), [1, 1])
    frame.tables["household"]["HRHTYPE"] = [np.nan, np.nan]
    with pytest.raises(ValueError, match=r"observed HRHTYPE.*household ids \[21\]"):
        _values(frame, "all")


def test_household_carrying_both_record_types_refuses() -> None:
    frame = _frame([[(2, 7, 30, False)]], kinds=[1])
    frame.tables["household"]["HRHTYPE"] = [6]
    with pytest.raises(ValueError, match="at most one of TYPEHUGQ and HRHTYPE"):
        _values(frame, "all")


def _universe_fixture(*, hrhtype: list[float] | None) -> tuple[_Frame, dict]:
    frame = _frame(
        [
            [(2, 7, 30, False)],
            [(14, 7, 30, True)],
            [(1, 1, 40, False), (4, 1, 40, True), (5, 7, 8, False)],
            [(2, 7, 70, True)],
            [(2, 7, 30, False)],
        ],
        kinds=[1, 2, np.nan, np.nan, np.nan],
    )
    frame.tables["household"]["household_spine"] = ["acs_2024_1yr"] * 2 + [
        "asec_puf"
    ] * 3
    if hrhtype is not None:
        frame.tables["household"]["HRHTYPE"] = hrhtype
    metadata = {
        "households": {"household_type": "all"},
        "living_alone": {"household_type": "householder_living_alone"},
        "married_with_children": {
            "household_type": "married_couple_family",
            "own_children": "with_own_children_under_18",
        },
    }
    return frame, metadata


def test_receipt_counts_each_universe_source_and_the_fallback_contribution() -> None:
    weights = [1.0, 2.0, 4.0, 8.0, 16.0]
    # Today's certified shape: the ASEC rows carry no HRHTYPE, so the last
    # (native group-quarters) record still counts as a household living alone.
    frame, metadata = _universe_fixture(hrhtype=None)
    with pytest.warns(HouseholdUniverseFallbackWarning):
        indicators = {
            name: household_type_household_values(frame, meta)
            for name, meta in metadata.items()
        }
    receipt = household_universe_receipt(frame, weights=weights, indicators=indicators)
    assert receipt == {
        "schema_version": 1,
        "rule": receipt["rule"],
        "households": {"TYPEHUGQ": 2, "HRHTYPE": 0, "roster_fallback": 3},
        "weighted_households": {
            "TYPEHUGQ": 3.0,
            "HRHTYPE": 0.0,
            "roster_fallback": 28.0,
        },
        "targets": {
            "households": {
                "weighted_value": 29.0,
                "roster_fallback_weighted_value": 28.0,
            },
            "living_alone": {
                "weighted_value": 25.0,
                "roster_fallback_weighted_value": 24.0,
            },
            "married_with_children": {
                "weighted_value": 4.0,
                "roster_fallback_weighted_value": 4.0,
            },
        },
    }
    assert "roster" in receipt["rule"]
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt

    # A fresh build's shape: HRHTYPE classifies every ASEC row, the native
    # group quarters leave every cell, and nothing falls back or warns.
    frame, metadata = _universe_fixture(hrhtype=[np.nan, np.nan, 1, 6, 10])
    indicators = {
        name: _no_fallback_values(frame, **meta) for name, meta in metadata.items()
    }
    receipt = household_universe_receipt(frame, weights=weights, indicators=indicators)
    assert receipt["households"] == {"TYPEHUGQ": 2, "HRHTYPE": 3, "roster_fallback": 0}
    assert receipt["weighted_households"] == {
        "TYPEHUGQ": 3.0,
        "HRHTYPE": 28.0,
        "roster_fallback": 0.0,
    }
    assert receipt["targets"] == {
        "households": {"weighted_value": 13.0, "roster_fallback_weighted_value": 0.0},
        "living_alone": {"weighted_value": 9.0, "roster_fallback_weighted_value": 0.0},
        "married_with_children": {
            "weighted_value": 4.0,
            "roster_fallback_weighted_value": 0.0,
        },
    }


@pytest.mark.parametrize(
    ("weights", "indicators"),
    [
        ([1.0, 2.0], {"households": [1.0, 1.0]}),
        ([1.0, np.nan, 1.0, 1.0, 1.0], {"households": [1.0] * 5}),
        ([1.0] * 5, {"households": [1.0] * 4}),
    ],
)
def test_receipt_refuses_misaligned_or_nonfinite_inputs(weights, indicators) -> None:
    frame, _ = _universe_fixture(hrhtype=None)
    with pytest.raises(ValueError, match="household-table order"):
        household_universe_receipt(frame, weights=weights, indicators=indicators)


@pytest.mark.parametrize("kind", [0, 11, 1.5, "unknown"])
def test_invalid_asec_household_type_refuses(kind) -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables["household"]["HRHTYPE"] = kind
    with pytest.raises(ValueError, match="HRHTYPE"):
        _values(frame, "all")


@pytest.mark.parametrize("heads", [0, 2])
def test_requires_exactly_one_householder(heads: int) -> None:
    frame = _frame([[(2 if i < heads else 12, 7, 30, False) for i in range(2)]])
    with pytest.raises(ValueError, match="exactly one householder"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("entity", "column"),
    [
        ("person", "A_EXPRRP"),
        ("person", "A_MARITL"),
        ("person", "age"),
        ("person", "is_female"),
        ("person", "person_household_id"),
        ("household", "household_id"),
    ],
)
def test_missing_required_column_refuses(entity: str, column: str) -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables[entity] = frame.tables[entity].drop(columns=column)
    with pytest.raises(ValueError, match="require.*column"):
        _values(frame, "all")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("A_EXPRRP", np.nan, "A_EXPRRP codes"),
        ("A_EXPRRP", 15, "A_EXPRRP codes"),
        ("A_MARITL", 8, "A_MARITL codes"),
        ("age", np.nan, "finite nonnegative age"),
        ("age", -1, "finite nonnegative age"),
        ("is_female", None, "observed boolean is_female"),
        ("person_household_id", 99, "household is missing"),
    ],
)
def test_invalid_person_values_refuse(column, value, message) -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables["person"][column] = value
    with pytest.raises(ValueError, match=message):
        _values(frame, "all")


@pytest.mark.parametrize("kind", [0, 4, "unknown"])
def test_invalid_group_quarters_kind_refuses(kind) -> None:
    frame = _frame([[(2, 7, 30, False)]], kinds=[kind])
    with pytest.raises(ValueError, match="TYPEHUGQ"):
        _values(frame, "all")


def test_acs_provenance_requires_group_quarters_classification() -> None:
    frame = _frame([[(2, 7, 30, False)]])
    frame.tables["household"]["household_spine"] = "acs_2024_1yr"
    with pytest.raises(ValueError, match="observed TYPEHUGQ"):
        _values(frame, "all")


def test_group_quarters_cannot_contain_multiple_people() -> None:
    frame = _frame([[(14, 7, 30, False), (14, 7, 40, True)]], kinds=[2])
    with pytest.raises(ValueError, match="exactly one person"):
        _values(frame, "all")


def test_unknown_metadata_and_missing_state_refuse() -> None:
    frame = _frame([[(2, 7, 30, False)]])
    with pytest.raises(ValueError, match="Unknown ACS household type"):
        _values(frame, "new_type")
    with pytest.raises(ValueError, match="Unknown ACS own-children filter"):
        _values(frame, "all", "new_children")
    frame.tables["household"] = frame.tables["household"].drop(columns="state_fips")
    with pytest.raises(ValueError, match="state_fips"):
        _values(frame, "all", state_fips="06")


@pytest.mark.parametrize("field", ["household_type", "own_children", "statistic"])
def test_release_filter_guard_checks_household_filter_attestation(field: str) -> None:
    builder = _load_builder_module()
    metadata = {
        "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
        "household_type": "married_couple_family",
        "own_children": "with_own_children_under_18",
        "statistic": "estimate",
    }
    metadata[f"ledger_filter_{field}"] = metadata[field]
    spec = SimpleNamespace(name="households", metadata=metadata)
    builder._assert_supported_ledger_filter_metadata((spec,))
    metadata[f"ledger_filter_{field}"] = "different"
    with pytest.raises(RuntimeError, match="disagrees with compiled"):
        builder._assert_supported_ledger_filter_metadata((spec,))
    metadata[f"ledger_filter_{field}"] = metadata[field]
    metadata["materializer"] = "policyengine_variable"
    with pytest.raises(RuntimeError, match=f"ledger_filter_{field}"):
        builder._assert_supported_ledger_filter_metadata((spec,))
    if field == "statistic":
        metadata["materializer"] = HOUSEHOLD_TYPE_MATERIALIZER
        metadata["statistic"] = "margin_of_error"
        metadata["ledger_filter_statistic"] = "margin_of_error"
        with pytest.raises(RuntimeError, match="not an ACS estimate"):
            builder._assert_supported_ledger_filter_metadata((spec,))


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_release_materializes_required_roster_indicators(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person")
    person["A_EXPRRP"] = [1, 4, 5, 2, 2, 12, 1, 5, 7, 12, 14]
    person["A_MARITL"] = 7
    person["age"] = [40, 40, 5, 40, 40, 40, 40, 5, 40, 40, 40]
    person["is_female"] = False
    frame.table("household")["TYPEHUGQ"] = [1, 1, 1, 1, 2]
    targets = (
        _spec(
            "households",
            "census_households",
            {
                "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
                "household_type": "all",
            },
        ),
        _spec(
            "married_with_children",
            "census_households",
            {
                "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
                "household_type": "married_couple_family",
                "own_children": "with_own_children_under_18",
            },
        ),
    )
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    target_frame, _, compilation = builder._materialize_target_frame(
        frame, targets, maximum_microsim_batch_size=batch_size
    )
    assert compilation["dropped_target_names"] == []
    np.testing.assert_array_equal(
        target_frame.table("household")["households"], [1, 1, 1, 1, 0]
    )
    np.testing.assert_array_equal(
        target_frame.table("household")["married_with_children"], [1, 0, 0, 0, 0]
    )


_RELEASE_HOUSEHOLD_TARGETS = (
    _spec(
        "households",
        "census_households",
        {"materializer": HOUSEHOLD_TYPE_MATERIALIZER, "household_type": "all"},
    ),
    _spec(
        "living_alone",
        "census_households",
        {
            "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
            "household_type": "householder_living_alone",
        },
    ),
    _spec(
        "married_with_children",
        "census_households",
        {
            "materializer": HOUSEHOLD_TYPE_MATERIALIZER,
            "household_type": "married_couple_family",
            "own_children": "with_own_children_under_18",
        },
    ),
)


def _release_roster_frame(*, hrhtype=None):
    """The nested frame's five households, weights 1 to 5: two ACS housing
    units, two ASEC households (ids 2 and 3) and an ACS group-quarters
    placeholder (id 5)."""
    frame = _nested_frame()
    person = frame.table("person")
    person["A_EXPRRP"] = [1, 4, 5, 2, 2, 12, 1, 5, 7, 12, 14]
    person["A_MARITL"] = 7
    person["age"] = [40, 40, 5, 40, 40, 40, 40, 5, 40, 40, 40]
    person["is_female"] = False
    household = frame.table("household")
    household["TYPEHUGQ"] = [1, np.nan, np.nan, 1, 2]
    household["household_spine"] = [
        "acs_2024_1yr",
        "asec_puf",
        "asec_puf",
        "acs_2024_1yr",
        "acs_2024_1yr",
    ]
    if hrhtype is not None:
        household["HRHTYPE"] = hrhtype
    return frame


def _diagnostics_build(builder, monkeypatch, tmp_path, compilation) -> dict:
    """The build block the release writes into calibration_diagnostics.json."""
    captured = {}

    def recording_write_calibration_diagnostics(
        result, path, *, target_registry, build, target_surface
    ):
        captured["path"] = path
        captured["build"] = build
        return SimpleNamespace(
            status="available", path=path, schema_version=9, sha256="d" * 64
        )

    monkeypatch.setattr(
        builder,
        "write_calibration_diagnostics",
        recording_write_calibration_diagnostics,
    )
    gate = SimpleNamespace(passed=True, failures=(), details={})
    builder._write_release_calibration_diagnostics(
        result=SimpleNamespace(),
        target_surface={"sha256": "b" * 64, "n_targets": 3},
        release_dir=tmp_path,
        registry=builder.TargetRegistry((), country="us"),
        base_dataset_sha256="base-sha",
        compilation=compilation,
        target_profile_gate=gate,
        health_input_gate=gate,
        base_population_gate=gate,
        support_value_repairs={},
        audit_export_targets=False,
        gate_failures=[],
    )
    assert captured["path"] == tmp_path / "calibration_diagnostics.json"
    return captured["build"]


@pytest.mark.parametrize("batch_size", [2, None])
def test_release_records_the_roster_fallback_in_calibration_diagnostics(
    monkeypatch, tmp_path, capsys, batch_size
) -> None:
    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    with pytest.warns(HouseholdUniverseFallbackWarning):
        target_frame, _, compilation = builder._load_or_materialize_target_frame(
            _release_roster_frame(),
            _RELEASE_HOUSEHOLD_TARGETS,
            maximum_microsim_batch_size=batch_size,
        )
    household = target_frame.table("household")
    # The fallback leaves the materialized numbers as they were.
    np.testing.assert_array_equal(household["households"], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(household["living_alone"], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(household["married_with_children"], [1, 0, 0, 0, 0])
    receipt = compilation["household_universe"]
    assert receipt == {
        "weight_basis": "initial_household_weights",
        "weight_kind": "design",
        "schema_version": 1,
        "rule": receipt["rule"],
        "households": {"TYPEHUGQ": 3, "HRHTYPE": 0, "roster_fallback": 2},
        "weighted_households": {
            "TYPEHUGQ": 10.0,
            "HRHTYPE": 0.0,
            "roster_fallback": 5.0,
        },
        "targets": {
            "households": {
                "weighted_value": 10.0,
                "roster_fallback_weighted_value": 5.0,
            },
            "living_alone": {
                "weighted_value": 2.0,
                "roster_fallback_weighted_value": 2.0,
            },
            "married_with_children": {
                "weighted_value": 1.0,
                "roster_fallback_weighted_value": 0.0,
            },
        },
    }
    stderr = capsys.readouterr().err
    # One warning per compile, however many batches the engine ran.
    assert stderr.count("WARNING: household-type targets classified") == 1
    assert "2 household(s) with initial weight 5 by the roster fallback" in stderr

    build = _diagnostics_build(builder, monkeypatch, tmp_path, compilation)
    written = build["target_compilation"]["household_universe"]
    assert written == receipt
    assert json.loads(builder._strict_json_text(written)) == receipt


def test_release_receipt_without_fallback_records_hrhtype_and_stays_quiet(
    monkeypatch, capsys
) -> None:
    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    # Household 2 is native ASEC group quarters; household 3 a housing unit.
    with warnings.catch_warnings():
        warnings.simplefilter("error", HouseholdUniverseFallbackWarning)
        target_frame, _, compilation = builder._load_or_materialize_target_frame(
            _release_roster_frame(hrhtype=[np.nan, 9, 6, np.nan, np.nan]),
            _RELEASE_HOUSEHOLD_TARGETS,
        )
    np.testing.assert_array_equal(
        target_frame.table("household")["households"], [1, 0, 1, 1, 0]
    )
    receipt = compilation["household_universe"]
    assert receipt["households"] == {"TYPEHUGQ": 3, "HRHTYPE": 2, "roster_fallback": 0}
    assert receipt["weighted_households"] == {
        "TYPEHUGQ": 10.0,
        "HRHTYPE": 5.0,
        "roster_fallback": 0.0,
    }
    assert receipt["targets"]["households"] == {
        "weighted_value": 8.0,
        "roster_fallback_weighted_value": 0.0,
    }
    assert "WARNING" not in capsys.readouterr().err


def test_release_receipt_follows_a_checkpoint_hit_and_needs_a_household_target(
    monkeypatch, tmp_path, capsys
) -> None:
    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    identity = {
        "kind": "fixture",
        "materializer_version": builder.TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
    }
    path = tmp_path / "target_frame_checkpoint.h5"

    def load(build_commit: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HouseholdUniverseFallbackWarning)
            return builder._load_or_materialize_target_frame(
                _release_roster_frame(),
                _RELEASE_HOUSEHOLD_TARGETS,
                target_frame_checkpoint_path=path,
                target_frame_checkpoint_identity=identity,
                target_frame_checkpoint_build_commit=build_commit,
            )

    _, _, written = load("a" * 40)
    _, _, reloaded = load("b" * 40)
    assert written["target_frame_checkpoint"]["status"] == "miss_written"
    assert reloaded["target_frame_checkpoint"]["status"] == "hit"
    # A re-solve that reuses the checkpoint runs no materializer, yet still
    # records and warns about the fallback in its own compilation.
    assert reloaded["household_universe"] == written["household_universe"]
    assert reloaded["household_universe"]["households"]["roster_fallback"] == 2
    assert capsys.readouterr().err.count("by the roster fallback") == 2
    # The receipt is attached after the checkpoint write, so the checkpoint's
    # stored compilation is what it was before the receipt existed.
    stored = reloaded["target_frame_checkpoint"]["stored_compilation"]
    assert stored["compiled_candidate_targets"] == 3
    assert "household_universe" not in stored

    # A compiled surface without a household-type target records nothing.
    frame = _release_roster_frame()
    frame.table("household")["other"] = 1.0
    other = _spec("other", "fixture", {"materializer": "policyengine_variable"})
    registry, compilation = builder._compile_materialized_target_registry(
        frame, (other,)
    )
    _, _, compilation = builder._with_household_universe(frame, registry, compilation)
    assert compilation["compiled_candidate_targets"] == 1
    assert "household_universe" not in compilation


@pytest.mark.parametrize("batch_size", [2, None])
def test_materializing_without_the_release_records_no_receipt(
    monkeypatch, capsys, batch_size
) -> None:
    # The scorers and the post-export audit run _materialize_target_frame on
    # their own frames and weights. The materializer still warns, but only
    # the release's own target frame carries the receipt and its stderr line.
    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=())
    with pytest.warns(HouseholdUniverseFallbackWarning):
        _, _, compilation = builder._materialize_target_frame(
            _release_roster_frame(),
            _RELEASE_HOUSEHOLD_TARGETS,
            maximum_microsim_batch_size=batch_size,
        )
    assert compilation["dropped_target_names"] == []
    assert "household_universe" not in compilation
    assert "WARNING" not in capsys.readouterr().err


def test_head_to_head_slices_compile_alike_on_a_roster_fallback_frame(
    monkeypatch, capsys
) -> None:
    # The head-to-head scorer digests each household slice's compilation and
    # keeps slice sizes out of it, so slices of 2, 2 and 1 households must
    # compile alike even when their households fall back to the roster.
    scorer = _load_head_to_head_module()
    _install_fake_engine(scorer.release, monkeypatch, reform_specs=())
    with pytest.warns(HouseholdUniverseFallbackWarning):
        chunk = scorer._score_chunk_household_sliced(
            _release_roster_frame(),
            _RELEASE_HOUSEHOLD_TARGETS,
            chunk_loss_weights=np.ones(len(_RELEASE_HOUSEHOLD_TARGETS)),
            artifact_name="fixture",
            chunk_label="household types",
            maximum_microsim_batch_size=2,
        )
    compilation = chunk.compilation
    assert compilation["household_slice_row_counts"] == [2, 2, 1]
    assert len(compilation["slice_compilation_sha256s"]) == 3
    assert len(set(compilation["slice_compilation_sha256s"])) == 1
    assert "household_universe" not in compilation
    assert "WARNING" not in capsys.readouterr().err


@st.composite
def _frames(draw):
    count = draw(st.integers(min_value=1, max_value=8))
    rosters = []
    kinds = []
    for _ in range(count):
        kind = draw(st.sampled_from(["married", "other", "nonfamily", "gq"]))
        female = draw(st.booleans())
        if kind == "gq":
            rosters.append([(14, 7, draw(st.integers(0, 105)), female)])
            kinds.append(draw(st.sampled_from([2, 3])))
            continue
        members = [(2 if kind == "nonfamily" else 1, 7, 40, female)]
        if kind == "married":
            members.append((3 if female else 4, 1, 40, not female))
        extras = draw(st.integers(min_value=0 if kind != "other" else 1, max_value=4))
        for _ in range(extras):
            relation = draw(
                st.sampled_from(
                    [12, 13, 14] if kind == "nonfamily" else [5, 7, 9, 11, 12]
                )
            )
            members.append(
                (
                    relation,
                    draw(st.integers(1, 7)),
                    draw(st.integers(0, 105)),
                    draw(st.booleans()),
                )
            )
        rosters.append(members)
        kinds.append(draw(st.sampled_from([1, np.nan])))
    frame = _frame(rosters, kinds=kinds)
    # Neither implementation can depend on sorted persons or household ids.
    frame.tables["household"] = frame.tables["household"].iloc[::-1]
    frame.tables["person"] = frame.tables["person"].iloc[::-1]
    return frame


@given(_frames())
@settings(max_examples=70, deadline=None)
def test_indicator_partitions_hold_for_every_valid_roster(frame: _Frame) -> None:
    household_types = (
        "all",
        "family",
        "married_couple_family",
        "male_householder_no_spouse",
        "female_householder_no_spouse",
        "nonfamily",
        "householder_living_alone",
        "householder_not_living_alone",
    )
    values = {kind: _values(frame, kind) for kind in household_types}
    for kind, national in values.items():
        assert np.isin(national, [0.0, 1.0]).all()
        np.testing.assert_array_equal(
            _values(frame, kind, state_fips="06")
            + _values(frame, kind, state_fips="48"),
            national,
        )
    np.testing.assert_array_equal(
        values["married_couple_family"]
        + values["male_householder_no_spouse"]
        + values["female_householder_no_spouse"]
        + values["nonfamily"],
        values["all"],
    )
    for kind in (
        "married_couple_family",
        "male_householder_no_spouse",
        "female_householder_no_spouse",
    ):
        with_children = _values(frame, kind, "with_own_children_under_18")
        without_children = _values(frame, kind, "no_own_children_under_18")
        assert np.isin(with_children, [0.0, 1.0]).all()
        assert np.isin(without_children, [0.0, 1.0]).all()
        np.testing.assert_array_equal(with_children + without_children, values[kind])
        for own_children in ("with_own_children_under_18", "no_own_children_under_18"):
            np.testing.assert_array_equal(
                _values(frame, kind, own_children, state_fips="06")
                + _values(frame, kind, own_children, state_fips="48"),
                _values(frame, kind, own_children),
            )


@given(_frames(), st.sampled_from([0, 2]))
@settings(max_examples=50, deadline=None)
def test_any_roster_with_zero_or_two_householders_refuses(
    frame: _Frame, heads: int
) -> None:
    person = frame.table("person").copy()
    references = person.loc[person.A_EXPRRP.isin([1, 2])]
    assume(not references.empty)
    household_id = references.iloc[0].person_household_id
    member_indices = person.index[person.person_household_id == household_id]
    person.loc[member_indices, "A_EXPRRP"] = 12
    if heads == 2:
        person.loc[member_indices[0], "A_EXPRRP"] = 2
        duplicate = person.loc[[member_indices[0]]].copy()
        person = pd.concat([person, duplicate], ignore_index=True)
    frame.tables["person"] = person
    with pytest.raises(ValueError, match="exactly one householder"):
        _values(frame, "all")


def _pandas_indicators(frame: _Frame) -> np.ndarray:
    """Independent row/group oracle for the eight bound national ACS cells."""
    rows = []
    people = frame.table("person")
    for _, household in frame.table("household").iterrows():
        if household.TYPEHUGQ in (2, 3):
            rows.append([0] * 8)
            continue
        members = people.loc[people.person_household_id == household.household_id]
        reference = members.loc[members.A_EXPRRP.isin([1, 2])].iloc[0]
        family = reference.A_EXPRRP == 1
        married = family and members.A_EXPRRP.isin([3, 4]).any()
        other_male = family and not married and not reference.is_female
        other_female = family and not married and reference.is_female
        child = not members.loc[
            members.A_EXPRRP.eq(5) & members.age.lt(18) & members.A_MARITL.eq(7)
        ].empty
        rows.append(
            [
                1,
                married,
                other_male,
                other_female,
                len(members) == 1,
                married and child,
                other_male and child,
                other_female and child,
            ]
        )
    return np.asarray(rows, dtype=float)


@given(_frames())
@settings(max_examples=70, deadline=None)
def test_eight_indicators_match_independent_pandas_oracle(frame: _Frame) -> None:
    actual = np.column_stack(
        [
            _values(frame, "all"),
            _values(frame, "married_couple_family"),
            _values(frame, "male_householder_no_spouse"),
            _values(frame, "female_householder_no_spouse"),
            _values(frame, "householder_living_alone"),
            _values(frame, "married_couple_family", "with_own_children_under_18"),
            _values(frame, "male_householder_no_spouse", "with_own_children_under_18"),
            _values(
                frame, "female_householder_no_spouse", "with_own_children_under_18"
            ),
        ]
    )
    np.testing.assert_array_equal(actual, _pandas_indicators(frame))
