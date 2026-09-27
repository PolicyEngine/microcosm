"""Invented original SS reports for the real survey preparation/qualification path.

This helper constructs raw source bytes and their matching parent checkpoints
before fresh issuance. It does not replace an issuer, qualification, or model.
The private source pins describe invented fixtures, never Census evidence.
"""

import copy
import hashlib
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from test_us_survey_population_preparation import fixture

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime import asec_coverage_authentication as coverage
from microcosm.build.us_runtime import asec_current_money_source as money
from microcosm.build.us_runtime import asec_household_observations as household_owner
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import native_household_origin as origin
from microcosm.frame import Frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ss_report_source_arguments(
    root: Path, monkeypatch, *, no_recipients: bool = False
) -> dict:
    """Return arguments for ``prepare_authenticated_survey_population``.

    Native ASEC household 7 contains 105 (retirement, age 55), 106 (NIU,
    age 14), 109 (disability, age 45), and 110 (dependents, age 18).
    Household 8 contains 107 (ambiguous reason 7, age 55), 108 (NIU,
    age 14), 111 (survivors, age 65), and 112 (known zero, age 40).
    The maintained fraction 2/3, seed 41 selection keeps household 8 and
    omits household 7; original DESIGN weights remain 2552.12 and 100.
    Every ACS source person reports positive SSP so any selected ACS person
    supplies an unresolved positive report. ``no_recipients`` instead encodes
    coherent zero reports in both surveys, retaining the under-15 NIU cases.

    The historical ASEC parent rows, raw members, and cohort files are
    preserved. New people join existing groups; raw source household IDs
    7/8 remain distinct from parent-internal PH_SEQ/group IDs 3/4.
    """
    arguments = fixture(
        root,
        monkeypatch,
        fraction=Fraction(2, 3),
        seed=41,
        zero=False,
        acs_ssp_values={
            key: 0 if no_recipients else 1200
            for key in (
                ("2024HU0000001", 1),
                ("2024HU0000001", 2),
                ("2024HU0000002", 1),
                ("2024GQ0000001", 1),
                ("2024GQ0000002", 1),
            )
        },
    )
    source_root = arguments["source_dir"] / "asec"
    parent_path = source_root / "parent.h5"
    household_path = source_root / "household-attachment.h5"
    loaded = load_frame_checkpoint(parent_path)
    tables = {e: loaded.frame.table(e).copy(deep=True) for e in loaded.frame.entities}
    people = tables["person"]
    strata = loaded.frame.strata.copy(deep=True)
    historical = people.loc[people.source_year.ne(2024)].copy(deep=True)

    # Copy the dependent record in each existing household, preserving every
    # group link. Only raw construction is changed, before any new issuance.
    additions, added_strata = [], []
    for pid, template_pid, line, age in (
        (109, 106, 3, 45),
        (110, 106, 4, 18),
        (111, 108, 3, 65),
        (112, 108, 4, 40),
    ):
        positions = np.flatnonzero(people.person_id.to_numpy() == template_pid)
        assert len(positions) == 1
        position = positions[0]
        person = people.iloc[[position]].copy(deep=True)
        person["person_id"] = np.array([pid], dtype=np.int64)
        person["PERIDNUM"] = pd.array([str(pid - 100).zfill(22)], dtype="string")
        person["A_LINENO"] = line
        person["A_AGE"] = age
        person["WSAL_VAL"] = 0.0
        person["SEMP_VAL"] = 0.0
        additions.append(person)
        added_strata.append(strata.iloc[[position]].copy(deep=True))
    people = pd.concat([people, *additions])
    strata = pd.concat([strata, *added_strata])
    # Report reason 7 is ambiguous; it is not the person's age.
    reports = {
        105: (12000, 1),
        106: (0, 0),
        107: (3000, 7),
        108: (0, 0),
        109: (6000, 2),
        110: (2400, 4),
        111: (9000, 3),
        112: (0, 0),
    }
    for pid, (amount, _) in reports.items():
        selected = people.person_id.eq(pid)
        assert selected.sum() == 1
        people.loc[selected, "SS_VAL"] = 0.0 if no_recipients else float(amount)
    pd.testing.assert_frame_equal(
        people.loc[people.source_year.ne(2024)], historical, check_exact=True
    )
    tables["person"] = people
    frame = Frame(
        tables,
        loaded.frame.schema,
        {e: loaded.frame.weights_for(e) for e in loaded.frame.weighted_entities},
        strata,
    )
    metadata = copy.deepcopy(loaded.metadata)
    for name in ("identity", "source_construction_identity"):
        metadata[name] = frame_identity(frame).to_payload()
    sources = tuple(
        household_owner.AsecHouseholdObservationSource(
            entry["year"], Path(entry["path"]), entry["sha256"]
        )
        for entry in metadata["source_receipt"]["sources"]
    )
    assert [s.year for s in sources] == [2022, 2023, 2024]
    assert all(_sha(s.path) == s.sha256 for s in sources)
    write_frame_checkpoint(parent_path, frame, metadata=metadata)
    parent_sha = _sha(parent_path)
    attachment = household_owner.with_asec_household_observations(
        frame,
        checkpoint_metadata=metadata,
        checkpoint_sha256=parent_sha,
        sources=sources,
    )
    write_frame_checkpoint(
        household_path,
        attachment.frame,
        metadata={
            "schema_version": 1,
            "artifact_kind": "microcosm.asec_household_observations_source",
            "parent_checkpoint_sha256": parent_sha,
            "household_observations": attachment.receipt,
        },
    )
    monkeypatch.setattr(
        money,
        "_SOURCE_PINS",
        (
            parent_sha,
            _sha(household_path),
            tuple((s.year, s.sha256) for s in sources),
        ),
    )

    members, pins = {}, []
    current = people.loc[people.source_year.eq(2024)].set_index("PERIDNUM")
    for year, member, archive, old_digest, old_rows, old_size in coverage._MEMBER_PINS:
        path = source_root / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if year == 2024:
            extra = []
            for pid, template_pid in ((109, 106), (110, 106), (111, 108), (112, 108)):
                record = raw.loc[
                    raw.PERIDNUM.eq(str(template_pid - 100).zfill(22))
                ].copy(deep=True)
                assert len(record) == 1
                record["PERIDNUM"] = str(pid - 100).zfill(22)
                extra.append(record)
            raw = pd.concat([raw, *extra], ignore_index=True)
            assert raw.PERIDNUM.is_unique and set(raw.PERIDNUM) == set(current.index)
            for field, parent_field in (
                ("PH_SEQ", "source_household_id"),
                ("A_LINENO", "A_LINENO"),
                ("A_AGE", "A_AGE"),
                ("SS_VAL", "SS_VAL"),
                ("WSAL_VAL", "WSAL_VAL"),
                ("SEMP_VAL", "SEMP_VAL"),
            ):
                values = current.loc[raw.PERIDNUM, parent_field].to_numpy(np.float64)
                assert np.isfinite(values).all() and (values == np.floor(values)).all()
                raw[field] = [str(int(v)) for v in values]
            raw["SS_YN"] = [
                "0" if int(age) < 15 else "1" if int(amount) > 0 else "2"
                for age, amount in zip(raw.A_AGE, raw.SS_VAL, strict=True)
            ]
            raw["RESNSS1"] = [
                "0" if no_recipients else str(reports[int(key) + 100][1])
                for key in raw.PERIDNUM
            ]
            for field in ("RESNSS2", "RESNSSA", "I_SSVAL", "I_SSYN"):
                raw[field] = "0"
            # Deliberately differ from the parent order: qualification must join
            # native source keys, not rely on row positions.
            raw.iloc[::-1].to_csv(path, index=False)
        else:
            assert (_sha(path), len(raw), path.stat().st_size) == (
                old_digest,
                old_rows,
                old_size,
            )
        pins.append((year, member, archive, _sha(path), len(raw), path.stat().st_size))
        members[year] = path
    for owner in (coverage, restoration):
        monkeypatch.setattr(owner, "_MEMBER_PINS", tuple(pins))

    household_member = source_root / "hhpub25.csv"
    raw_households = pd.read_csv(household_member, dtype=str, keep_default_na=False)
    assert raw_households.H_SEQ.map(int).tolist() == [7, 8]
    raw_households["H_NUMPER"] = "4"
    raw_households.to_csv(household_member, index=False)
    assert len(origin._ASEC_MEMBER_PINS) == 1
    monkeypatch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        (
            replace(
                origin._ASEC_MEMBER_PINS[0],
                member_sha256=_sha(household_member),
                size_bytes=household_member.stat().st_size,
                rows=len(raw_households),
            ),
        ),
    )
    restored = root / "ss-report-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path,
        household_path,
        member_paths=members,
        output_dir=restored,
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        source_root / "person-income-attachment.h5",
    )
    return arguments
