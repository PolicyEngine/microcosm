"""The UK reviewed-exclusion registers on the shared review core.

microcosm#1759 step D7 moved three pieces of ``uk_runtime.weighted_integrity``
onto ``microcosm.build.declared_registers``: the strict JSON reader of the
reviewed-exclusion files (``read_strict_json``), the date checks of
``UKReviewedExclusion`` (``canonical_date``) and its window
(``ReviewWindow``). ``test_support/microcosm_build/
uk_reviewed_exclusion_reference.py`` holds the old code verbatim. On the
packaged files, on changed texts and on the days around every packaged
entry's window, the two give the same value or the same refusal: the same
error class and the same text. A gate that fails closed records the class
name in its report, so the class is part of the output.
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from microcosm.build.uk_runtime import weighted_integrity
from test_support.microcosm_build import uk_reviewed_exclusion_reference as reference

#: The reviewed-exclusion files this reader serves.
RESOURCES = (
    "target_fit_reviewed_exclusions.json",
    "local_target_fit_reviewed_exclusions.json",
    "local_area_support_exclusions.json",
)


def _outcome(call):
    try:
        return ("value", call())
    except Exception as error:  # noqa: BLE001 - the refusal is the outcome
        return ("refused", type(error).__name__, str(error))


@pytest.mark.parametrize("resource", RESOURCES)
def test_the_packaged_files_read_the_same(resource) -> None:
    old = _outcome(lambda: reference._read_register_payload(None, resource=resource))
    new = _outcome(
        lambda: weighted_integrity._read_register_payload(None, resource=resource)
    )
    assert new == old
    assert new[0] == "value"


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1, "b": 1, "c": 2, "c": 3}}',
        '{"a": NaN}',
        '{"a": [Infinity]}',
        '{"a": -Infinity}',
        '{"a": }',
        "",
        "[1, 2]",
        '"text"',
        '{"exclusions": {}}',
    ],
)
def test_changed_texts_read_the_same(text, tmp_path) -> None:
    path = tmp_path / "register.json"
    path.write_text(text, encoding="utf-8")
    old = _outcome(lambda: reference._read_register_payload(path, resource="x"))
    new = _outcome(
        lambda: weighted_integrity._read_register_payload(path, resource="x")
    )
    assert new == old


@pytest.mark.parametrize(
    ("approved_on", "expires_on"),
    [
        ("2026-10-01", "2027-03-31"),
        ("20261001", "2027-03-31"),
        ("2026-10-01", "2027-W13-3"),
        ("2026-02-30", "2027-03-31"),
        ("2026-10-01", " 2027-03-31"),
        ("2026-10-01", "2026-10-01"),
        ("2027-03-31", "2026-10-01"),
        (None, "2027-03-31"),
        ("2026-10-01", 20270331),
        ("", "2027-03-31"),
    ],
)
def test_the_date_checks_refuse_the_same(approved_on, expires_on) -> None:
    def new() -> None:
        weighted_integrity.UKReviewedExclusion(
            reason="r",
            approved_by="a",
            adjudication="d",
            approved_on=approved_on,
            expires_on=expires_on,
        )

    old = _outcome(
        lambda: reference.check_dates(
            SimpleNamespace(approved_on=approved_on, expires_on=expires_on)
        )
    )
    outcome = _outcome(new)
    assert outcome[0] == old[0]
    if old[0] == "refused":
        assert outcome[1:] == old[1:]


def _packaged_entries():
    for resource in RESOURCES:
        payload, _label = reference._read_register_payload(None, resource=resource)
        for block in ("exclusions", "bound_despite_support_floor"):
            for name, entry in (payload.get(block) or {}).items():
                yield f"{resource}:{name}", entry


def test_every_packaged_window_reads_the_same() -> None:
    checked = 0
    for _name, entry in _packaged_entries():
        record = weighted_integrity.UKReviewedExclusion(
            reason=entry["reason"],
            approved_by=entry["approved_by"],
            adjudication=entry["adjudication"],
            approved_on=entry["approved_on"],
            expires_on=entry["expires_on"],
        )
        approved = date.fromisoformat(entry["approved_on"])
        expires = date.fromisoformat(entry["expires_on"])
        for on in (
            approved - timedelta(days=1),
            approved,
            expires,
            expires + timedelta(days=1),
        ):
            assert record.expired(on) == reference.expired(record, on)
            assert record.premature(on) == reference.premature(record, on)
            checked += 1
    assert checked > 0
