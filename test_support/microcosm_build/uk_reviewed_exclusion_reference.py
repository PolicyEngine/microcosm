# Verbatim copies, at c866b8951 (before microcosm#1759 step D7), of the UK
# reviewed-exclusion code that now uses microcosm.build.declared_registers:
# uk_runtime.weighted_integrity._read_register_payload, the date checks of
# UKReviewedExclusion.__post_init__ (as check_dates), and its expired and
# premature methods (as functions). Tests hold the new code to them; do not
# edit.
# ruff: noqa

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from importlib.resources import files
from pathlib import Path


def _read_register_payload(
    source: str | Path | None, *, resource: str
) -> tuple[Mapping[str, object], str]:
    if source is None:
        raw = files("microcosm.build.uk").joinpath(resource).read_text("utf-8")
        label = resource
    else:
        raw = Path(source).read_text(encoding="utf-8")
        label = str(source)

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicates: set[str] = set()
        for name, value in pairs:
            if name in result:
                duplicates.add(name)
            result[name] = value
        if duplicates:
            raise ValueError(f"{label}: duplicate JSON key(s): {sorted(duplicates)}.")
        return result

    def reject_nonfinite_json(value: str) -> object:
        raise ValueError(f"{label}: non-finite JSON value {value!r} is invalid.")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: malformed JSON: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: exclusion register must be a JSON object.")
    return payload, label


def check_dates(self) -> None:
        parsed: dict[str, date] = {}
        for name in ("approved_on", "expires_on"):
            value = getattr(self, name)
            try:
                parsed[name] = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"UK reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                ) from exc
            # fromisoformat also accepts compact (20270210) and week-date
            # forms; the receipt pins the canonical rendering because the
            # raw string is sealed into the policy digest — two spellings
            # of one date must not produce two digests.
            if parsed[name].isoformat() != value:
                raise ValueError(
                    f"UK reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                )
        if parsed["expires_on"] <= parsed["approved_on"]:
            raise ValueError(
                "UK reviewed exclusion expires_on must be after approved_on; "
                f"got approved_on={self.approved_on!r}, "
                f"expires_on={self.expires_on!r}."
            )


def expired(self, now: date) -> bool:
    """Honored through ``expires_on``; expired strictly after it."""

    return now > date.fromisoformat(self.expires_on)

def premature(self, now: date) -> bool:
    """In force from ``approved_on``; premature strictly before it."""

    return now < date.fromisoformat(self.approved_on)
