"""Assembler capability probe, its refusal, and the observed role label.

The capability branches no real runtime exhibits are driven by synthetic stub
assemblers installed in ``sys.modules``, so every branch is exercised wherever
this file runs. The tests that need the actual imported assembler carry the
same strict conditional marker as the partition suite.
"""

import dataclasses
import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.acs_spm_partition import (
    ACS_SPM_DEVELOPMENT_POLICY,
    AcsSpmAssemblerProbe,
    AcsSpmRoleDecision,
    UnsupportedAssembler,
    probe_acs_spm_assembler,
    reconstruct_acs_spm_partition,
)

REASONS = {
    "supported",
    "assembler_unavailable",
    "incompatible_parent_link_order",
    "incompatible_foster_boundary",
    "incompatible_call_contract",
}
# The probe's own roster, pinned here so a change to it has to be reviewed.
PROBE_IDS = [1, 2, 3, 4, 5, 6]
CORRECT = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
# The installed 1.0.0 ordering: the residual under-15 reaches the reference unit
# before the accepted parent link is applied, so household 1 collapses.
MERGED_PARENT = {1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 3}
# The foster boundary moved by one year: the 21-year-old leaves the reference
# unit and pools with the 22-year-old instead.
SPLIT_FOSTER = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3}
BOTH_WRONG = {1: 0, 2: 0, 3: 0, 4: 2, 5: 3, 6: 3}
GOOD_DIAGNOSTICS = {
    "method": "inferred",
    "fallback_rules_used": ["link_parent_child"],
}

requires_assembler = pytest.mark.xfail(
    not probe_acs_spm_assembler().supported,
    strict=True,
    raises=UnsupportedAssembler,
    reason="Assembled membership requires the reviewed canonical assembler",
)


def fake(mapping, *, info=GOOD_DIAGNOSTICS, calls=None):
    """A synthetic assembler: fixed units for the probe roster, singletons else."""

    def spm_unit_id(persons, *, diagnostics=False, **_):
        if calls is not None:
            calls.append(persons.copy(deep=True))
        ids = pd.Series(
            [mapping[pid] for pid in persons.person_id]
            if list(persons.person_id) == PROBE_IDS
            else range(len(persons)),
            dtype="int64",
        )
        return (ids, info) if diagnostics else ids

    return spm_unit_id


def install(monkeypatch, spm_unit_id):
    module = types.ModuleType("spm_calculator")
    module.spm_unit_id = spm_unit_id
    monkeypatch.setitem(sys.modules, "spm_calculator", module)


def roster(relationships, ages, *, marital=None, gq=False):
    n = len(ages)
    if marital is None:
        marital = [1 if rel in {20, 21, 23} else 5 for rel in relationships]
    return pd.DataFrame(
        {
            "person_id": range(101, 101 + n),
            "person_household_id": [10] * n,
            "person_spm_unit_id": [700] * n,
            "SPORDER": range(1, n + 1),
            "RELSHIPP": relationships,
            "AGEP": ages,
            "MAR": marital,
            "TYPEHUGQ": [2 if gq else 1] * n,
        }
    )


def assemble(persons, **kwargs):
    old_units = pd.DataFrame(
        {"spm_unit_id": persons.person_spm_unit_id.unique(), "snap": 123.0}
    )
    return reconstruct_acs_spm_partition(
        persons,
        old_units,
        household_person_counts=persons.groupby("person_household_id").size().to_dict(),
        **kwargs,
    )


def test_probe_verdict_is_frozen_typed_and_json_safe():
    probe = probe_acs_spm_assembler()
    assert isinstance(probe, AcsSpmAssemblerProbe)
    assert probe.reason in REASONS
    assert probe.supported is (probe.reason == "supported")
    with pytest.raises(dataclasses.FrozenInstanceError):
        probe.supported = True
    provenance = probe.as_provenance()
    assert set(provenance) == {
        "supported",
        "reason",
        "module_file",
        "module_sha256",
    }
    assert json.loads(json.dumps(provenance)) == provenance


def test_probe_binds_the_actual_assembler_file_hash():
    probe = probe_acs_spm_assembler()
    if probe.reason == "assembler_unavailable":
        assert probe.module_file is None and probe.module_sha256 is None
        return
    if probe.module_file is None:
        assert probe.module_sha256 is None
        return
    digest = hashlib.sha256(Path(probe.module_file).read_bytes()).hexdigest()
    assert probe.module_sha256 == digest


def test_probe_reports_an_absent_assembler_without_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "spm_calculator", None)
    probe = probe_acs_spm_assembler()
    assert probe == AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None)


def test_probe_is_re_evaluated_rather_than_cached(monkeypatch):
    baseline = probe_acs_spm_assembler()
    with monkeypatch.context() as patched:
        patched.setitem(sys.modules, "spm_calculator", None)
        assert probe_acs_spm_assembler().reason == "assembler_unavailable"
    assert probe_acs_spm_assembler() == baseline


def test_probe_records_the_stub_it_actually_exercised(monkeypatch):
    install(monkeypatch, fake(CORRECT))
    probe = probe_acs_spm_assembler()
    assert probe.supported and probe.reason == "supported"
    # ``fake`` is defined in this file, so the probe must name this file.
    assert Path(probe.module_file) == Path(__file__)
    assert (
        probe.module_sha256 == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mapping,reason",
    [
        (CORRECT, "supported"),
        (MERGED_PARENT, "incompatible_parent_link_order"),
        (SPLIT_FOSTER, "incompatible_foster_boundary"),
        # Both capabilities wrong: the parent-order verdict is reported first.
        (BOTH_WRONG, "incompatible_parent_link_order"),
    ],
)
def test_probe_names_the_capability_that_diverged(monkeypatch, mapping, reason):
    install(monkeypatch, fake(mapping))
    probe = probe_acs_spm_assembler()
    assert probe.reason == reason
    assert probe.supported is (reason == "supported")


def raises_on_call(persons, **_):
    raise RuntimeError("synthetic assembler failure")


@pytest.mark.parametrize(
    "spm_unit_id",
    [
        pytest.param(raises_on_call, id="raises"),
        pytest.param(len, id="unreadable-builtin"),
        pytest.param(
            lambda persons, **_: pd.Series(range(len(persons))), id="not-a-pair"
        ),
        pytest.param(
            lambda persons, **_: (pd.Series(range(len(persons))), None, None),
            id="three-tuple",
        ),
        pytest.param(
            lambda persons, **_: (pd.Series([0]), GOOD_DIAGNOSTICS),
            id="short-ids",
        ),
        pytest.param(
            lambda persons, **_: (
                pd.Series([pd.NA] * len(persons), dtype="Int64"),
                GOOD_DIAGNOSTICS,
            ),
            id="missing-ids",
        ),
        pytest.param(fake(CORRECT, info=["not", "a", "mapping"]), id="no-mapping"),
        pytest.param(fake(CORRECT, info={1: "not a string key"}), id="nonstring-key"),
        pytest.param(fake(CORRECT, info={"fallback_rules_used": []}), id="no-method"),
        pytest.param(
            fake(CORRECT, info={"method": 7, "fallback_rules_used": []}),
            id="nonstring-method",
        ),
        pytest.param(
            fake(
                CORRECT,
                info={"method": "native_spm_id", "fallback_rules_used": []},
            ),
            id="echoed-native-id",
        ),
        pytest.param(fake(CORRECT, info={"method": "inferred"}), id="no-rules"),
        pytest.param(
            fake(CORRECT, info={"method": "inferred", "fallback_rules_used": "link"}),
            id="rules-are-a-string",
        ),
        pytest.param(
            fake(CORRECT, info={"method": "inferred", "fallback_rules_used": [7]}),
            id="nonstring-rule",
        ),
    ],
)
def test_probe_reports_call_contract_failures_without_raising(monkeypatch, spm_unit_id):
    install(monkeypatch, spm_unit_id)
    probe = probe_acs_spm_assembler()
    assert probe == AcsSpmAssemblerProbe(
        False, "incompatible_call_contract", probe.module_file, probe.module_sha256
    )


def test_probe_identity_is_omitted_when_the_assembler_file_is_unreadable(
    monkeypatch,
):
    install(monkeypatch, len)
    probe = probe_acs_spm_assembler()
    assert probe.module_file is None and probe.module_sha256 is None


def test_probe_sends_the_same_view_shape_the_partition_sends(monkeypatch):
    calls = []
    install(monkeypatch, fake(CORRECT, calls=calls))
    assemble(roster([20, 21, 25], [45, 44, 10]))
    probed, actual = calls[0], calls[-1]
    assert list(probed.person_id) == PROBE_IDS
    assert list(probed.columns) == list(actual.columns)
    assert probed.dtypes.astype(str).tolist() == actual.dtypes.astype(str).tolist()
    # The probe must not hand the assembler evidence the adapter withholds.
    assert not {"person_spm_unit_id", "spm_unit_id", "SPM_ID", "A_EXPRRP"} & set(
        probed.columns
    )


@pytest.mark.parametrize(
    "reason",
    [
        "assembler_unavailable",
        "incompatible_parent_link_order",
        "incompatible_foster_boundary",
        "incompatible_call_contract",
    ],
)
def test_unsupported_probe_refuses_with_the_dedicated_exception(monkeypatch, reason):
    probe = AcsSpmAssemblerProbe(False, reason, None, None)
    monkeypatch.setattr(
        "microcosm.build.acs_spm_partition.probe_acs_spm_assembler", lambda: probe
    )
    with pytest.raises(UnsupportedAssembler) as excinfo:
        assemble(roster([20, 21, 25], [45, 44, 10]))
    assert excinfo.value.probe is probe
    assert reason in str(excinfo.value)


def test_unsupported_assembler_is_not_a_value_error():
    assert issubclass(UnsupportedAssembler, Exception)
    assert not issubclass(UnsupportedAssembler, ValueError)


def test_source_evidence_is_validated_before_the_assembler_refusal(monkeypatch):
    monkeypatch.setattr(
        "microcosm.build.acs_spm_partition.probe_acs_spm_assembler",
        lambda: AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None),
    )
    persons = roster([20, 21], [45, 44])
    persons["RELSHIPP"] = [20, 20]
    # A refusable roster must still refuse the same way on every runtime.
    with pytest.raises(ValueError, match="reference person"):
        assemble(persons)


@pytest.mark.parametrize(
    "decisions",
    [
        [AcsSpmRoleDecision(101, False, "head_override")],
        [AcsSpmRoleDecision(999, True, "absent_person")],
        [AcsSpmRoleDecision(102, 1, "not_bool")],
        [AcsSpmRoleDecision(102, True, "")],
        [AcsSpmRoleDecision(102, True, "one"), AcsSpmRoleDecision(102, False, "two")],
        [object()],
    ],
)
def test_invalid_role_decisions_refuse_before_any_assembler_probe(
    monkeypatch, decisions
):
    def must_not_probe():
        pytest.fail("Invalid source decisions reached the assembler probe")

    monkeypatch.setattr(
        "microcosm.build.acs_spm_partition.probe_acs_spm_assembler", must_not_probe
    )
    persons = roster([20, 25], [45, 16])
    with pytest.raises(ValueError):
        assemble(persons, role_decisions=decisions)


@pytest.mark.parametrize("age,gq", [(14, False), (18, False), (17, True)])
def test_inapplicable_role_decisions_refuse_without_assembler(monkeypatch, age, gq):
    def must_not_probe():
        pytest.fail("Inapplicable source decision reached the assembler probe")

    monkeypatch.setattr(
        "microcosm.build.acs_spm_partition.probe_acs_spm_assembler", must_not_probe
    )
    persons = roster([37, 37] if gq else [20, 25], [45, age], gq=gq)
    with pytest.raises(ValueError, match="15–17"):
        assemble(
            persons, role_decisions=[AcsSpmRoleDecision(102, True, "outside_scope")]
        )


@pytest.mark.parametrize("kind", ["group-quarters", "unresolved"])
def test_no_refusal_when_no_household_reaches_the_assembler(monkeypatch, kind):
    monkeypatch.setattr(
        "microcosm.build.acs_spm_partition.probe_acs_spm_assembler",
        lambda: AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None),
    )
    persons = (
        roster([37, 37], [17, 40], gq=True)
        if kind == "group-quarters"
        else roster([20, 34, 36], [45, 35, 8])
    )
    if kind == "group-quarters":
        persons["person_spm_unit_id"] = [700, 701]
    result = assemble(persons)
    assert result.provenance["assembler"] is None
    assert result.provenance["canonical_diagnostics"] is None


@requires_assembler
def test_partition_provenance_binds_the_actual_probe_identity():
    provenance = assemble(roster([20, 21, 25], [45, 44, 10])).provenance
    assert provenance["assembler"] == probe_acs_spm_assembler().as_provenance()
    assert provenance["assembler"]["supported"] is True
    assert json.loads(json.dumps(provenance["assembler"])) == provenance["assembler"]


@requires_assembler
def test_observed_head_and_spouse_carry_the_relationship_rule_role_label():
    membership = assemble(roster([20, 21, 25, 36], [45, 44, 10, 16])).membership
    assert membership.role_source.tolist() == [
        "observed_relationship_rule",
        "observed_relationship_rule",
        "age_not_role_sensitive",
        "unresolved",
    ]
    assert "source_observed" not in set(membership.role_source)
    assert membership.role_rule.iloc[0] == "acs_relshipp_reference_head_or_spouse"


@requires_assembler
def test_observed_link_source_label_survives_the_role_rename():
    result = assemble(roster([20, 21, 25], [45, 44, 10]))
    assert set(result.links.source) == {"source_observed"}
    assert result.membership.parent_link_status.iloc[2] == "observed_parent_link"
    assert set(result.membership.role_source) <= {
        "observed_relationship_rule",
        "approved_inference",
        "modeled_assumption",
        "unresolved",
        "age_not_role_sensitive",
        "outside_acs_household_universe",
    }


@requires_assembler
def test_development_modeled_roles_are_not_relabelled_as_observed():
    membership = assemble(
        roster([20, 36, 34], [45, 16, 17]), policy=ACS_SPM_DEVELOPMENT_POLICY
    ).membership
    assert membership.role_source.tolist() == [
        "observed_relationship_rule",
        "modeled_assumption",
        "modeled_assumption",
    ]


def test_group_quarters_roles_stay_null_and_unlabelled_on_any_runtime():
    persons = roster([37, 37], [17, 40], gq=True)
    persons["person_spm_unit_id"] = [700, 701]
    membership = assemble(persons).membership
    assert membership.independent_minor_role.isna().all()
    assert membership.role_source.eq("outside_acs_household_universe").all()


def test_probe_refuses_units_that_cross_households(monkeypatch):
    # Correct components within each household do not permit shared IDs across them.
    install(monkeypatch, fake({1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}))
    assert probe_acs_spm_assembler().reason == "incompatible_call_contract"


def test_probe_classifies_a_broken_import_separately_from_absence(monkeypatch):
    import builtins

    original = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "spm_calculator":
            raise RuntimeError("synthetic broken import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert probe_acs_spm_assembler().reason == "incompatible_call_contract"


def test_probe_returns_typed_refusal_if_roster_construction_fails(monkeypatch):
    install(monkeypatch, fake(CORRECT))

    def broken_frame(*args, **kwargs):
        raise ValueError("synthetic DataFrame failure")

    monkeypatch.setattr(pd, "DataFrame", broken_frame)
    assert probe_acs_spm_assembler().reason == "incompatible_call_contract"
