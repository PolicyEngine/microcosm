"""The SPM measurement-universe producer: the rule, and every refusal.

Every fixture here is invented. Nothing in this file opens a build artifact,
a pinned source, or a release.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.spm_universe_source import (
    ACS_ARM,
    ACS_GROUP_QUARTERS_KINDS,
    ACS_HOUSEHOLD_KIND_COLUMN,
    ACS_HOUSING_UNIT_KIND,
    ASEC_ARM,
    ASEC_NATIVE_UNIT_COLUMN,
    HOUSEHOLD_SPINE_COLUMN,
    HOUSEHOLD_SUPPORT_CHANNEL_COLUMN,
    INCLUDED,
    OUTSIDE,
    SPM_UNIVERSE_CHANNEL_ARMS,
    SPM_UNIVERSE_REFUSALS,
    UNIVERSE_INPUT,
    UNRESOLVED,
    attach_spm_universe_status,
    classify_spm_universe,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_US_RUNTIME = (
    Path(__file__).resolve().parents[1] / "src" / "microcosm" / "build" / "us_runtime"
)
_ACS_CHANNEL = "acs_2024_1yr"
_ASEC_CHANNEL = "asec_puf"


def _tables(
    persons: list[dict[str, object]],
    *,
    channel_column: str = HOUSEHOLD_SPINE_COLUMN,
    household_extra: dict[int, dict[str, object]] | None = None,
    person_extra_columns: dict[str, list[object]] | None = None,
    extra_household_rows: list[dict[str, object]] | None = None,
    extra_unit_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Household, person and SPM unit ids from a list of invented persons.

    Each person dict carries ``hid`` (household), ``unit`` (SPM unit),
    ``chan`` (provenance tag), and optionally ``kind`` (``TYPEHUGQ``),
    ``spm_id`` (the ASEC native unit id) and ``clone`` (clone index).
    """
    prows = []
    households: dict[int, dict[str, object]] = {}
    for index, spec in enumerate(persons, start=1):
        hid = int(spec["hid"])
        row: dict[str, object] = {
            "person_id": index,
            "person_household_id": hid,
            "person_spm_unit_id": int(spec["unit"]),
            "person_tax_unit_id": hid,
            "person_family_id": hid,
            "person_marital_unit_id": index,
            "person_support_clone_index": int(spec.get("clone", 0)),
        }
        if "spm_id" in spec:
            row[ASEC_NATIVE_UNIT_COLUMN] = spec["spm_id"]
        prows.append(row)
        household = {"household_id": hid, channel_column: spec["chan"]}
        if "kind" in spec:
            household[ACS_HOUSEHOLD_KIND_COLUMN] = spec["kind"]
        households.setdefault(hid, household)
    person = pd.DataFrame(prows)
    if person_extra_columns:
        for name, values in person_extra_columns.items():
            person[name] = values
    household_rows = [households[hid] for hid in sorted(households)]
    if household_extra:
        for row in household_rows:
            row.update(household_extra.get(int(row["household_id"]), {}))
    if extra_household_rows:
        household_rows.extend(extra_household_rows)
    household = pd.DataFrame(household_rows)
    if ACS_HOUSEHOLD_KIND_COLUMN in household.columns:
        household[ACS_HOUSEHOLD_KIND_COLUMN] = household[
            ACS_HOUSEHOLD_KIND_COLUMN
        ].astype(object)
    unit_ids = np.asarray(
        sorted({int(spec["unit"]) for spec in persons} | set(extra_unit_ids or []))
    )
    return household, person, unit_ids


def _classify(persons: list[dict[str, object]], **kwargs) -> dict[int, str]:
    table_kwargs = {
        key: kwargs.pop(key)
        for key in (
            "channel_column",
            "household_extra",
            "person_extra_columns",
            "extra_household_rows",
            "extra_unit_ids",
        )
        if key in kwargs
    }
    household, person, unit_ids = _tables(persons, **table_kwargs)
    result = classify_spm_universe(
        household=household, person=person, unit_ids=unit_ids, **kwargs
    )
    return dict(zip(result.unit_ids.tolist(), result.status.tolist(), strict=True))


def _frame(persons: list[dict[str, object]], **kwargs) -> Frame:
    household, person, unit_ids = _tables(persons, **kwargs)
    household_ids = household["household_id"].to_numpy()
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": pd.DataFrame({"tax_unit_id": household_ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": unit_ids}),
            "family": pd.DataFrame({"family_id": household_ids}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.full(len(household_ids), 100.0), WeightKind.CALIBRATED
            )
        },
        metadata={"stage": "invented"},
    )


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_acs_housing_unit_is_included():
    statuses = _classify(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": ACS_HOUSING_UNIT_KIND},
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": ACS_HOUSING_UNIT_KIND},
        ]
    )
    assert statuses == {1: INCLUDED}


@pytest.mark.parametrize("kind", ACS_GROUP_QUARTERS_KINDS)
def test_acs_group_quarters_are_outside_for_each_code(kind):
    statuses = _classify([{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": kind}])
    assert statuses == {1: OUTSIDE}


def test_asec_records_are_included():
    """The ASEC ruling is spine-level: no record-type field is consulted."""
    statuses = _classify(
        [
            {"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 11},
            {"hid": 1, "unit": 2, "chan": _ASEC_CHANNEL, "spm_id": 12},
            {"hid": 2, "unit": 3, "chan": _ASEC_CHANNEL, "spm_id": 13},
        ]
    )
    assert statuses == {1: INCLUDED, 2: INCLUDED, 3: INCLUDED}


def test_a_zero_adult_housing_unit_is_included():
    """The anti-inference test.

    A housing unit whose only members are children is a household-unit data
    defect, not an out-of-universe record. Marking it ``OUTSIDE`` would hide
    the defect and make the universe move whenever the role column moved. Only
    the record type decides: the identical composition under ``TYPEHUGQ`` 2 is
    ``OUTSIDE``, and it is the record type that moved it.
    """
    children = [
        {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": ACS_HOUSING_UNIT_KIND},
        {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": ACS_HOUSING_UNIT_KIND},
    ]
    ages = {"age": [3, 7], "is_spm_independent_minor_role": [False, False]}
    assert _classify(children, person_extra_columns=ages) == {1: INCLUDED}

    group_quarters = [dict(spec, kind=2) for spec in children[:1]]
    assert _classify(group_quarters, person_extra_columns={"age": [3]}) == {1: OUTSIDE}


def test_the_declaration_is_invariant_to_age_and_role():
    """Composition cannot move the answer, in either direction."""
    persons = [
        {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": ACS_HOUSING_UNIT_KIND},
        {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 2},
    ]
    children = _classify(
        persons,
        person_extra_columns={
            "age": [2, 3],
            "is_spm_independent_minor_role": [False, False],
        },
    )
    adults = _classify(
        persons,
        person_extra_columns={
            "age": [44, 51],
            "is_spm_independent_minor_role": [True, True],
        },
    )
    assert children == adults == {1: INCLUDED, 2: OUTSIDE}


def test_a_clone_inherits_its_native_rows_status():
    """Cloning a record carries its record type, so it carries its universe."""
    statuses = _classify(
        [
            # Native group-quarters placeholder and its clone.
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2, "clone": 0},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 2, "clone": 1},
            # Native housing unit and its clone.
            {"hid": 3, "unit": 3, "chan": _ACS_CHANNEL, "kind": 1, "clone": 0},
            {"hid": 4, "unit": 4, "chan": _ACS_CHANNEL, "kind": 1, "clone": 1},
        ]
    )
    assert statuses == {1: OUTSIDE, 2: OUTSIDE, 3: INCLUDED, 4: INCLUDED}


def test_a_pooled_frame_declares_each_arm_from_its_own_rule():
    household, person, unit_ids = _tables(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 3},
            {"hid": 3, "unit": 3, "chan": _ASEC_CHANNEL, "spm_id": 31},
        ]
    )
    result = classify_spm_universe(
        household=household, person=person, unit_ids=unit_ids
    )
    assert result.status.tolist() == [INCLUDED, OUTSIDE, INCLUDED]
    assert result.provenance["households_by_arm"] == {ACS_ARM: 2, ASEC_ARM: 1}
    assert result.provenance["households_by_acs_record_type"] == {1: 1, 3: 1}
    assert result.provenance["acs_group_quarters_households"] == 1
    assert result.provenance["units_included"] == 2
    assert result.provenance["units_outside"] == 1
    assert result.provenance["units_unresolved"] == 0


def test_the_support_channel_column_is_read_when_no_spine_tag_is_present():
    statuses = _classify(
        [
            {"hid": 1, "unit": 1, "chan": "acs", "kind": 1},
            {"hid": 2, "unit": 2, "chan": "asec", "spm_id": 21},
        ],
        channel_column=HOUSEHOLD_SUPPORT_CHANNEL_COLUMN,
    )
    assert statuses == {1: INCLUDED, 2: INCLUDED}


def test_an_untagged_table_accepts_an_explicit_caller_declaration():
    household = pd.DataFrame(
        {"household_id": [1], ACS_HOUSEHOLD_KIND_COLUMN: [2]},
    )
    person = pd.DataFrame(
        {"person_id": [1], "person_household_id": [1], "person_spm_unit_id": [1]}
    )
    result = classify_spm_universe(
        household=household,
        person=person,
        unit_ids=np.asarray([1]),
        declared_channel=_ACS_CHANNEL,
    )
    assert result.status.tolist() == [OUTSIDE]
    assert result.provenance["channel_source"] == f"declared:{_ACS_CHANNEL}"


# --------------------------------------------------------------------------
# Stored encoding
# --------------------------------------------------------------------------


def test_the_status_is_stored_as_the_enum_member_name():
    household, person, unit_ids = _tables(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 2},
        ]
    )
    status = classify_spm_universe(
        household=household, person=person, unit_ids=unit_ids
    ).status
    assert status.dtype == np.dtype(object)
    assert status.tolist() == ["INCLUDED", "OUTSIDE"]
    # Negative controls: not the integer index, not repr(), not the label.
    assert 2 not in set(status.tolist())
    assert "SPMUniverseStatus.OUTSIDE" not in set(status.tolist())
    assert not any(" " in value for value in status.tolist())
    # UNRESOLVED is the engine's absence sentinel; a producer never emits it.
    assert UNRESOLVED not in set(status.tolist())


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def _refuses(code: str):
    return pytest.raises(ValueError, match=rf"^{code}: ")


def test_refuses_a_preexisting_declaration():
    frame = _frame([{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}])
    declared = attach_spm_universe_status(frame).frame
    with _refuses("SPM_UNIVERSE_COLUMN_EXISTS"):
        attach_spm_universe_status(declared)


def test_refuses_a_channel_with_no_universe_ruling():
    with _refuses("SPM_UNIVERSE_UNDECLARED_SPINE"):
        _classify([{"hid": 1, "unit": 1, "chan": "puf_tax_detail", "kind": 1}])


def test_refuses_a_frame_with_no_provenance_tag_and_no_declaration():
    household = pd.DataFrame({"household_id": [1], ACS_HOUSEHOLD_KIND_COLUMN: [1]})
    person = pd.DataFrame(
        {"person_id": [1], "person_household_id": [1], "person_spm_unit_id": [1]}
    )
    with _refuses("SPM_UNIVERSE_UNDECLARED_SPINE"):
        classify_spm_universe(
            household=household, person=person, unit_ids=np.asarray([1])
        )


def test_refuses_a_caller_declaration_over_a_tagged_frame():
    with _refuses("SPM_UNIVERSE_UNDECLARED_SPINE"):
        _classify(
            [{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}],
            declared_channel=_ASEC_CHANNEL,
        )


@pytest.mark.parametrize("kind", [4, 0, -1, None, float("nan")])
def test_refuses_an_unknown_acs_record_type(kind):
    with _refuses("SPM_UNIVERSE_UNKNOWN_HOUSEHOLD_KIND"):
        _classify([{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": kind}])


def test_refuses_an_acs_record_type_on_a_non_acs_row():
    """``TYPEHUGQ == 1`` means "ACS housing unit", never "not group quarters"."""
    with _refuses("SPM_UNIVERSE_OFF_ARM_HOUSEHOLD_KIND"):
        _classify(
            [{"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "kind": 1, "spm_id": 11}]
        )


def test_refuses_a_household_resolving_to_more_than_one_kind():
    household = pd.DataFrame(
        {
            "household_id": [1, 1],
            HOUSEHOLD_SPINE_COLUMN: [_ACS_CHANNEL, _ACS_CHANNEL],
            ACS_HOUSEHOLD_KIND_COLUMN: [1, 2],
        }
    )
    person = pd.DataFrame(
        {"person_id": [1], "person_household_id": [1], "person_spm_unit_id": [1]}
    )
    with _refuses("SPM_UNIVERSE_MIXED_HOUSEHOLD"):
        classify_spm_universe(
            household=household, person=person, unit_ids=np.asarray([1])
        )


def test_refuses_a_group_quarters_placeholder_with_two_native_persons():
    with _refuses("SPM_UNIVERSE_GQ_MULTI_PERSON"):
        _classify(
            [
                {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2, "clone": 0},
                {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2, "clone": 0},
            ]
        )


def test_a_group_quarters_placeholder_with_a_clone_is_accepted():
    """Negative control for the refusal above: clones are expected."""
    statuses = _classify(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2, "clone": 0},
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2, "clone": 1},
        ]
    )
    assert statuses == {1: OUTSIDE}


def test_refuses_a_member_of_no_declared_unit():
    with _refuses("SPM_UNIVERSE_ORPHAN_MEMBER"):
        household, person, _ = _tables(
            [{"hid": 1, "unit": 7, "chan": _ACS_CHANNEL, "kind": 1}]
        )
        classify_spm_universe(
            household=household, person=person, unit_ids=np.asarray([1])
        )


def test_refuses_a_member_of_no_declared_household():
    household, person, unit_ids = _tables(
        [{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}]
    )
    person["person_household_id"] = [9]
    with _refuses("SPM_UNIVERSE_ORPHAN_MEMBER"):
        classify_spm_universe(household=household, person=person, unit_ids=unit_ids)


def test_refuses_a_unit_with_no_members():
    with _refuses("SPM_UNIVERSE_EMPTY_UNIT"):
        _classify(
            [{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}],
            extra_unit_ids=[2],
        )


def test_refuses_a_unit_spanning_two_record_types():
    """Natively unreachable; a majority vote here would fabricate a universe."""
    with _refuses("SPM_UNIVERSE_UNIT_SPANS_KINDS"):
        _classify(
            [
                {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
                {"hid": 2, "unit": 1, "chan": _ACS_CHANNEL, "kind": 2},
            ]
        )


def test_refuses_a_unit_spanning_two_arms():
    with _refuses("SPM_UNIVERSE_UNIT_SPANS_KINDS"):
        _classify(
            [
                {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
                {"hid": 2, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 21},
            ]
        )


def test_refuses_an_asec_arm_with_no_native_unit_column():
    with _refuses("SPM_UNIVERSE_DEGRADED_PARTITION"):
        _classify([{"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL}])


def test_refuses_a_missing_native_unit_id_on_the_asec_arm():
    """One missing ``SPM_ID`` degrades every SPM unit in the frame."""
    with _refuses("SPM_UNIVERSE_DEGRADED_PARTITION"):
        _classify(
            [
                {"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 11},
                {"hid": 2, "unit": 2, "chan": _ASEC_CHANNEL, "spm_id": np.nan},
            ]
        )


def test_refuses_an_asec_partition_that_is_not_the_native_one():
    """Two native units collapsed into one frame unit: the fallback's shape."""
    with _refuses("SPM_UNIVERSE_DEGRADED_PARTITION"):
        _classify(
            [
                {"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 11},
                {"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 12},
            ]
        )


def test_the_degraded_partition_check_does_not_fire_on_the_acs_arm():
    """ACS PUMS supplies no ``SPM_ID``; one unit per household is native there."""
    assert _classify(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 1},
        ]
    ) == {1: INCLUDED, 2: INCLUDED}


@pytest.mark.parametrize("field", ["H_TYPE", "HRHTYPE", "H_HHTYPE", "HUNITS"])
def test_refuses_an_asec_vintage_carrying_a_record_type_field(field):
    with _refuses("SPM_UNIVERSE_ASEC_RECORD_TYPE_UNREVIEWED"):
        _classify(
            [{"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 11}],
            household_extra={1: {field: 1}},
        )


def test_an_acs_only_frame_is_unaffected_by_the_asec_record_type_guard():
    """Negative control: the guard is scoped to the arm whose ruling it guards."""
    assert _classify(
        [{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}],
        household_extra={1: {"H_LIVQRT": 1}},
    ) == {1: INCLUDED}


# --------------------------------------------------------------------------
# The frame wrapper
# --------------------------------------------------------------------------


def test_attach_declares_the_status_and_changes_nothing_else():
    frame = _frame(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 2},
            {"hid": 3, "unit": 3, "chan": _ACS_CHANNEL, "kind": 3},
        ]
    )
    result = attach_spm_universe_status(frame)
    declared = result.frame

    assert declared.table("spm_unit")[UNIVERSE_INPUT].tolist() == [
        INCLUDED,
        OUTSIDE,
        OUTSIDE,
    ]
    # The input frame is untouched: this is a new frame, not a mutation.
    assert UNIVERSE_INPUT not in frame.table("spm_unit").columns
    # No weight, no membership, no row moves.
    assert declared.weights_for("household").total == pytest.approx(
        frame.weights_for("household").total
    )
    for entity in frame.entities:
        assert declared.n(entity) == frame.n(entity)
        before = frame.table(entity)
        after = declared.table(entity)
        assert set(after.columns) - set(before.columns) <= {UNIVERSE_INPUT}
        pd.testing.assert_frame_equal(after[before.columns], before)
    assert declared.metadata == frame.metadata
    assert declared.mass_log == frame.mass_log


def test_attach_reports_the_classification_it_declared():
    frame = _frame(
        [
            {"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1},
            {"hid": 2, "unit": 2, "chan": _ACS_CHANNEL, "kind": 2},
        ]
    )
    result = attach_spm_universe_status(frame)
    assert result.status.tolist() == [INCLUDED, OUTSIDE]
    assert result.provenance["declared_variable"] == UNIVERSE_INPUT
    assert result.provenance["declared_entity"] == "spm_unit"
    assert result.provenance["stored_encoding"] == "enum member name string"


def test_the_provenance_states_what_the_producer_is_not():
    frame = _frame([{"hid": 1, "unit": 1, "chan": _ACS_CHANNEL, "kind": 1}])
    provenance = attach_spm_universe_status(frame).provenance
    assert provenance["weights_used"] is False
    assert provenance["ages_changed"] is False
    assert provenance["composition_read"] is False
    assert provenance["is_calibration_target"] is False
    assert provenance["is_spine_agreement_surface"] is False


# --------------------------------------------------------------------------
# Constants parity and refusal-code completeness
# --------------------------------------------------------------------------


def _module_constant(module: str, name: str) -> str:
    """Read one module-level string constant without importing the module.

    ``acs_pums`` and ``stacked_spine`` both pull the country engine and cost
    tens of seconds to import; parsing is exact and free.
    """
    tree = ast.parse((_US_RUNTIME / module).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Constant), (module, name)
            return node.value.value
    raise AssertionError(f"{module} declares no module-level {name}")


@pytest.mark.parametrize(
    ("module", "constant", "arm"),
    [
        ("acs_pums.py", "ACS_2024_1YR_SPINE", ACS_ARM),
        ("stacked_spine.py", "ACS_STACKED_SUPPORT_CHANNEL", ACS_ARM),
        ("base_pool.py", "ASEC_PUF_SPINE", ASEC_ARM),
        ("support_provenance.py", "BASE_ASEC_SUPPORT_CHANNEL", ASEC_ARM),
    ],
)
def test_every_ruled_channel_matches_its_canonical_definition(module, constant, arm):
    value = _module_constant(module, constant)
    assert SPM_UNIVERSE_CHANNEL_ARMS.get(value) == arm, (
        f"{module}:{constant} == {value!r}, which this producer rules as "
        f"{SPM_UNIVERSE_CHANNEL_ARMS.get(value)!r} rather than {arm!r}"
    )


def test_the_ruling_table_declares_nothing_beyond_the_canonical_channels():
    canonical = {
        _module_constant("acs_pums.py", "ACS_2024_1YR_SPINE"),
        _module_constant("stacked_spine.py", "ACS_STACKED_SUPPORT_CHANNEL"),
        _module_constant("base_pool.py", "ASEC_PUF_SPINE"),
        _module_constant("support_provenance.py", "BASE_ASEC_SUPPORT_CHANNEL"),
    }
    assert set(SPM_UNIVERSE_CHANNEL_ARMS) == canonical


def test_the_spine_column_name_matches_its_canonical_factory():
    tree = ast.parse((_US_RUNTIME / "base_pool.py").read_text())
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "spine_column"
    )
    template = next(
        node for node in ast.walk(factory) if isinstance(node, ast.JoinedStr)
    )
    suffix = "".join(
        part.value for part in template.values if isinstance(part, ast.Constant)
    )
    assert HOUSEHOLD_SPINE_COLUMN == f"household{suffix}"


def test_every_refusal_the_module_raises_is_declared():
    """No refusal code may exist only inside a message string."""
    source = (_US_RUNTIME / "spm_universe_source.py").read_text()
    tree = ast.parse(source)
    raised = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_require"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            raised.add(node.args[1].value)
    assert raised, "the producer raises no named refusal at all"
    assert raised <= set(SPM_UNIVERSE_REFUSALS), sorted(
        raised - set(SPM_UNIVERSE_REFUSALS)
    )
    assert len(SPM_UNIVERSE_REFUSALS) == len(set(SPM_UNIVERSE_REFUSALS))
    # SPM_UNIVERSE_COLUMN_EXISTS is raised by the frame wrapper, every other
    # declared code by the classifier; together they must cover the tuple.
    assert raised == set(SPM_UNIVERSE_REFUSALS)
