"""Invented source literals; pure checks grant no source or receiving authority."""

import copy
import csv
import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pandas as pd
import pytest
from test_us_acs_spm_native_construction import (
    OPTIONS,
    require_acs_spm_source_capability,
    requires_assembler,
)

from microcosm.build.us_runtime import current_survey_spm_source as owner


def row(number, *, head=False, age=14, unit="000007"):
    return {
        "PERIDNUM": str(number).zfill(22),
        "PH_SEQ": "7",
        "A_LINENO": str(number),
        "A_AGE": str(age),
        "SPM_ID": unit,
        "SPM_HEAD": str(int(head)),
        "A_FAMTYP": "1",
        "A_FAMREL": "1" if head else "3",
        "SPM_HAGE": "55",
        "SPM_NUMADULTS": "1",
        "SPM_NUMKIDS": "1",
        "SPM_NUMPER": "2",
    }


def stream(rows):
    value = io.StringIO(newline="")
    writer = csv.DictWriter(value, fieldnames=owner._ASEC_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return io.BytesIO(value.getvalue().encode())


def pure_fixture(rows=None):
    rows = [row(1, head=True, age=55), row(2)] if rows is None else rows
    keys = {
        ("asec", owner.original._key(r, "asec")): i + 10 for i, r in enumerate(rows)
    }
    selected, rosters = owner._scan_asec(stream(rows), {k for _, k in keys}, len(rows))
    frame = SimpleNamespace(
        person=pd.DataFrame(
            {
                "person_id": [10, 11],
                "person_spm_unit_id": [100, 100],
                "person_household_id": [200, 200],
                "A_AGE": [55, 14],
            }
        )
    )
    origins = pd.DataFrame(
        {"source": ["asec", "asec"]}, index=pd.Index([10, 11], name="person_id")
    )
    return frame, origins, keys, selected, rosters


def project(args, policy=owner.ASEC_2025_INCOME_2024_SPM_POLICY):
    return owner._asec_projection(*args, policy)


def test_pure_reconciliation_preserves_roles_and_requires_explicit_annual_policy():
    args = pure_fixture()
    raw, roles, units = project(args)
    assert str(roles.dtype) == "boolean" and roles.tolist() == [True, False]
    assert units[0]["status"] == owner.INCLUDED
    assert raw.SPM_ID.tolist() == ["000007", "000007"]
    assert units[0]["source_spm_unit_id"] == "000007"
    assert project(args, None)[2][0]["status"] == owner.UNRESOLVED
    assert project(args, None)[2][0]["reason"] == "annual_scope_policy_not_admitted"


@pytest.mark.parametrize("field", ["SPM_HEAD", "A_FAMTYP", "A_FAMREL"])
def test_missing_role_is_nullable_and_unresolved_even_with_membership_policy(field):
    args = pure_fixture()
    list(args[3].values())[1][field] = ""
    raw, roles, units = project(args)
    assert raw.iloc[1][field] == "" and pd.isna(roles.iloc[1])
    assert units[0]["status"] == owner.UNRESOLVED
    assert units[0]["reason"] == "missing_role_source"


def test_asec_count_consistency_contract_is_implementation_bound(monkeypatch):
    args = pure_fixture()
    second = list(args[3].values())[1]
    second["SPM_NUMADULTS"], second["SPM_NUMKIDS"] = "2", "0"
    with pytest.raises(ValueError, match="ASEC_COMPLETE_UNIT"):
        project(args)
    monkeypatch.setattr(owner.asec_roles, "_COUNTS", ())
    assert owner._live() != owner._LIVE
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        owner.qualify_current_survey_spm(
            SimpleNamespace(),
            acs_profile=owner.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True),
            asec_scope_policy=owner.ASEC_2025_INCOME_2024_SPM_POLICY,
        )


@pytest.mark.parametrize(
    "policy", [True, "included", {}, replace(owner.ASEC_2025_INCOME_2024_SPM_POLICY)]
)
def test_arbitrary_or_copied_scope_policy_is_not_admitted(policy):
    with pytest.raises(ValueError, match="ASEC_SCOPE_POLICY"):
        project(pure_fixture(), policy)


def test_source_extra_member_refuses_even_when_selected_counts_match():
    args = list(pure_fixture())
    rows = list(args[3].values()) + [row(3)]
    args[3], args[4] = owner._scan_asec(stream(rows), {k for _, k in args[2]}, 3)
    with pytest.raises(ValueError, match="EXACT_SOURCE_UNIT_ROSTER"):
        project(args)


def test_pure_source_coordinate_mismatch_refuses():
    args = pure_fixture()
    list(args[3].values())[1]["PERIDNUM"] = str(3).zfill(22)
    with pytest.raises(ValueError, match="SOURCE_COORDINATE_CHANGED"):
        project(args)


@pytest.mark.parametrize(
    "mutation", ["split", "mixed", "age", "count", "head", "adult_count"]
)
def test_asec_membership_and_source_reconciliation_refusals(mutation):
    args = pure_fixture()
    rows = list(args[3].values())
    if mutation == "split":
        args[0].person.loc[1, "person_spm_unit_id"] = 101
    elif mutation == "mixed":
        rows[1]["SPM_ID"] = "8"
    elif mutation == "age":
        args[0].person.loc[1, "A_AGE"] = 13
    elif mutation == "head":
        rows[0]["SPM_HEAD"] = "0"
    else:
        for r in rows:
            if mutation == "count":
                r["SPM_NUMPER"] = "3"
            else:
                r["SPM_NUMADULTS"], r["SPM_NUMKIDS"] = "2", "0"
    with pytest.raises(ValueError):
        project(args)


def test_scanner_joins_by_identity_and_exhausts_unselected_rows():
    args = pure_fixture()
    rows = list(args[3].values())
    extra = row(3, unit="8")
    selected, rosters = owner._scan_asec(
        stream([rows[1], extra, rows[0]]), {k for _, k in args[2]}, 3
    )
    assert set(selected) == {k for _, k in args[2]}
    assert set(rosters) == {"000007"}
    with pytest.raises(ValueError, match="ROW_SHAPE"):
        owner._scan_asec(stream([*rows, extra]), set(selected), 2)
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        owner._scan_asec(stream([*rows, rows[0]]), set(selected), 3)


def test_detached_preparation_and_qualification_cannot_enter_owner_boundary():
    profile = owner.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True)
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        owner.qualify_current_survey_spm(
            SimpleNamespace(), acs_profile=profile, asec_scope_policy=None
        )
    fake = owner.QualifiedNativeSpmInputs(None, None, None, None, None, None, b"{}")
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        fake.validate()
    with pytest.raises(ValueError, match="RECEIVING_OWNER"):
        owner.project_qualified_spm_inputs(
            object(), fake, year=2024, outside_role_placeholder=False
        )
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        owner.qualify_native_spm_inputs(
            SimpleNamespace(), acs_profile=profile, asec_scope_policy=None
        )


def _real_source_arguments(tmp_path, monkeypatch):
    """Extend invented source bytes before the genuine preparation is issued."""
    from test_us_acs_spm_native_construction import request
    from test_us_survey_population_preparation import fixture

    from microcosm.build.us_runtime import asec_person_income_source as restoration

    arguments = fixture(tmp_path, monkeypatch)
    folder = arguments["source_dir"] / "asec"
    paths, pins = {}, []
    for year, member, archive, *_ in owner.original.asec._MEMBER_PINS:
        path = folder / member
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        table["SPM_ID"] = table.PH_SEQ.str.zfill(6)
        table["SPM_HEAD"] = table.A_LINENO.eq("1").astype(int).astype(str)
        table["A_FAMTYP"] = "1"
        table["A_FAMREL"] = table.SPM_HEAD.map({"1": "1", "0": "3"})
        table["SPM_HAGE"] = "55"
        table["SPM_NUMADULTS"], table["SPM_NUMKIDS"], table["SPM_NUMPER"] = (
            "1",
            "1",
            "2",
        )
        table.to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(table),
                len(payload),
            )
        )
        paths[year] = path
    for module in (owner.original.asec, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = tmp_path / "spm-restored-money"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=restored,
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    request(arguments["source_dir"] / "selection-request.json")
    return arguments


@requires_assembler
def test_genuine_preparation_retains_source_roles_and_refuses_copies_and_mutations(
    tmp_path, monkeypatch
):
    require_acs_spm_source_capability(OPTIONS)
    arguments = _real_source_arguments(tmp_path, monkeypatch)
    preparation = owner.source.prepare_authenticated_survey_population(**arguments)
    profile = owner.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True)
    result = owner.qualify_current_survey_spm(
        preparation,
        acs_profile=profile,
        asec_scope_policy=owner.ASEC_2025_INCOME_2024_SPM_POLICY,
    )
    result.validate()
    with pytest.raises(ValueError, match="RECEIVING_OWNER"):
        owner.project_qualified_spm_inputs(
            object(), result, year=2024, outside_role_placeholder=False
        )
    for module, name, value in (
        (owner.projection, "OUTSIDE", "UNRESOLVED"),
        (owner.acs, "INCLUDED", "OUTSIDE"),
        (owner.projection.attachment, "attach_columns", lambda *a, **k: {}),
        (owner.projection.provenance, "support_source_id_column", lambda e: "wrong"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(module, name, value)
            with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
                result.validate()
        result.validate()
    assert result.roles.index.equals(result.origins.index)
    asec = result.unit_evidence.source.eq("asec")
    assert result.unit_status.loc[asec].eq(owner.INCLUDED).all()
    assert result.roles.isna().sum() == 2
    assert set(result.unit_status) == {"INCLUDED", "OUTSIDE"}
    assert json.loads(result.receipt)["release_eligible"] is False
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        copy.copy(result).validate()
    callback = result._revalidate
    cells = dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))
    for name in ("roles", "result", "table_seals"):
        previous_cell = cells[name].cell_contents
        cells[name].cell_contents = object()
        with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
            result.validate()
        cells[name].cell_contents = previous_cell
        result.validate()
    forged = replace(result, roles=~result.roles.copy())

    def cell(value):
        return (lambda: value).__closure__[0]

    changed = {name: value.cell_contents for name, value in cells.items()}
    changed.update(
        result=forged,
        roles=forged.roles,
        table_seals=tuple(
            owner.seals._table_seal(t) for t in owner._result_tables(forged)
        ),
    )
    forged_cells = {name: cell(value) for name, value in changed.items()}
    cloned = FunctionType(
        callback.__code__,
        callback.__globals__,
        closure=tuple(forged_cells[name] for name in callback.__code__.co_freevars),
    )
    forged_cells["revalidate"].cell_contents = cloned
    object.__setattr__(forged, "_revalidate", cloned)
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        forged.validate()
    object.__setattr__(result, "_revalidate", cloned)
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        result.validate()
    object.__setattr__(result, "_revalidate", callback)
    result.validate()
    pid = result.roles.index[result.roles.notna()][0]
    previous = result.roles.at[pid]
    assert not pd.isna(previous)
    result.roles.at[pid] = pd.NA
    with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
        result.validate()
    result.roles.at[pid] = previous
    result.validate()
    (arguments["source_dir"] / "asec" / "pppub25.csv").write_bytes(b"changed")
    with pytest.raises(ValueError):
        result.validate()


@requires_assembler
def test_genuine_source_refuses_closure_mutation_during_final_io(tmp_path, monkeypatch):
    require_acs_spm_source_capability(OPTIONS)
    arguments = _real_source_arguments(tmp_path, monkeypatch)
    preparation = owner.source.prepare_authenticated_survey_population(**arguments)
    result = owner.qualify_current_survey_spm(
        preparation,
        acs_profile=owner.acs.ACSAnalysisProfile(2024, "acs_2024_1yr", True, True),
        asec_scope_policy=owner.ASEC_2025_INCOME_2024_SPM_POLICY,
    )
    callback = result._revalidate
    cells = dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))
    issued = owner._ISSUED[result]
    read_bytes = Path.read_bytes
    changed = []

    def mutate_after_read(path):
        value = read_bytes(path)
        if path == Path(owner.__file__) and not changed:
            changed.append(True)
            pid = result.roles.index[result.roles.notna()][0]
            result.roles.at[pid] = not bool(result.roles.at[pid])
            cells["table_seals"].cell_contents = tuple(
                owner.seals._table_seal(table) for table in owner._result_tables(result)
            )
        return value

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED|PROJECTION_CHANGED"):
        result.validate()
    assert changed == [True]
    assert owner._ISSUED[result] is issued
    assert issued[3] != tuple(
        owner.seals._table_seal(table) for table in owner._result_tables(result)
    )
