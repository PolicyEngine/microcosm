"""The committed register of adjudicated spine-vs-incumbent differences.

The whole-spine comparison (#686) has one rule: **anything differing that is
not in this register is a defect.**  Every intentional deviation of the
microcosm-built UK spine from the frozen enhanced-FRS incumbent — a mechanism
that was deliberately changed, a defect fixed on one side, a stochastic stream
that cannot align, a net-new column — is recorded here with its class, the
surface it shows up on, evidence, and who adjudicated it.

This register sits **above** the per-gate exclusion registers, and does not
replace them.  ``input_mass_reviewed_exclusions.json``,
``qrf_tail_reviewed_exclusions.json`` and ``degenerate_reviewed_exclusions.json``
keep their own roles: they suppress a specific gate for a specific column, they
are scoped per reference, and they **expire** so that a suppression cannot
outlive its reason.  A signed difference is the opposite kind of object — a
permanent adjudicated fact about how the two artifacts differ — so entries here
carry no expiry.  Where a gate exclusion descends from the same adjudication as
an entry here, that entry's ``evidence`` points at it.

Consumers: ``tools/verify_uk_spine_parity.py`` and
``tools/compare_uk_h5_payload.py --structure-only``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

__all__ = [
    "UK_SPINE_SWAP_SIGNED_DIFFERENCES_RESOURCE",
    "UKSignedDifference",
    "UKSignedDifferenceRegister",
    "load_uk_spine_swap_signed_differences",
]

UK_SPINE_SWAP_SIGNED_DIFFERENCES_RESOURCE = "spine_swap_signed_differences.json"

_UK_PACKAGE = "microcosm.build.uk"

#: Why the two artifacts differ.  Adding a class is a reviewed change: the
#: vocabulary is what makes "everything unexplained is a defect" enforceable.
SIGNED_DIFFERENCE_CLASSES = frozenset(
    {
        # A mechanism was deliberately re-designed in the port.
        "mechanism_change",
        # One side is wrong and the other fixes it; the entry says which.
        "defect_fix",
        # Identity-keyed or re-seeded draws that cannot align row-for-row.
        "rng_stream",
        # Same estimator family, different implementation.
        "qrf_implementation",
        # The spine produces a column the incumbent never had.
        "net_new_column",
        # The two artifacts were built from different source vintages.
        "vintage",
        # A gate threshold was re-measured on the candidate surface.
        "threshold_recut",
    }
)

#: Which comparison surface the difference shows up on.
SIGNED_DIFFERENCE_SURFACES = frozenset(
    {
        "nonzero_shares",
        "entity_counts",
        "weighted_totals",
        "payload_column",
        "root_attr",
    }
)

#: What the comparison should expect to see.
SIGNED_DIFFERENCE_EXPECTATIONS = frozenset(
    {
        "column_differs",
        "column_missing_in_reference",
        "column_missing_in_candidate",
        "count_differs",
    }
)

_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Surfaces whose ``column_differs`` adjudication also covers a payload-level
#: value difference on the same column. A share or weighted-total divergence
#: and a payload value mismatch are one fact measured by two instruments, so
#: an entry adjudicated on either value surface covers both readings. No other
#: cross-surface coverage exists: structural expectations (a column appearing
#: or vanishing, a count moving) never excuse a value difference, and vice
#: versa.
_PAYLOAD_BRIDGE_SURFACES = frozenset({"nonzero_shares", "weighted_totals"})


@dataclass(frozen=True)
class UKSignedDifference:
    """One adjudicated difference between the spine and the incumbent."""

    id: str
    difference_class: str
    surface: str
    expectation: str
    columns: tuple[str, ...]
    entities: tuple[str, ...]
    magnitude_evidence: str
    evidence: str
    adjudicator: str
    adjudicated_on: str

    def covers(self, *, surface: str, column: str, expectation: str) -> bool:
        """Whether this entry signs the observed difference.

        A difference is observed as ``(surface, column, expectation)`` and an
        entry signs it only when the expectation matches — an entry adjudicated
        for a column appearing (``column_missing_in_reference``) never excuses
        that column's *values* diverging, and vice versa. The expectation is
        therefore consulted at every lookup, not just validated at load.

        An empty ``columns`` tuple is a surface-wide entry (used by
        ``entity_counts``, where the "column" is an entity name).

        One deliberate cross-surface rule: a ``column_differs`` entry on a
        value-bearing surface (``nonzero_shares``, ``weighted_totals``) also
        covers a ``payload_column`` value mismatch on the same column, because
        both readings measure the same adjudicated fact.
        """

        if expectation != self.expectation:
            return False
        if surface == self.surface:
            return not self.columns or column in self.columns
        if (
            surface == "payload_column"
            and expectation == "column_differs"
            and self.surface in _PAYLOAD_BRIDGE_SURFACES
        ):
            return not self.columns or column in self.columns
        return False


@dataclass(frozen=True)
class UKSignedDifferenceRegister:
    """The committed register, indexed for comparison instruments."""

    differences: tuple[UKSignedDifference, ...]
    scope_note: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for difference in self.differences:
            if difference.id in seen:
                raise ValueError(
                    "UK signed-difference ids must be unique; "
                    f"{difference.id!r} appears more than once."
                )
            seen.add(difference.id)

    def by_id(self, identifier: str) -> UKSignedDifference | None:
        for difference in self.differences:
            if difference.id == identifier:
                return difference
        return None

    def matching(
        self, *, surface: str, column: str, expectation: str
    ) -> UKSignedDifference | None:
        """The entry signing the observed difference, if any."""

        for difference in self.differences:
            if difference.covers(
                surface=surface, column=column, expectation=expectation
            ):
                return difference
        return None


def _resource_text(resource: str) -> str:
    candidate = Path(resource)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return files(_UK_PACKAGE).joinpath(resource).read_text(encoding="utf-8")


def _require_str(value: object, *, field_name: str, resource: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{resource}: field {field_name!r} must be a non-empty string, "
            f"got {value!r}."
        )
    return value


def _require_member(
    value: object,
    *,
    allowed: frozenset[str],
    field_name: str,
    resource: str,
) -> str:
    text = _require_str(value, field_name=field_name, resource=resource)
    if text not in allowed:
        raise ValueError(
            f"{resource}: field {field_name!r} must be one of "
            f"{sorted(allowed)}, got {text!r}."
        )
    return text


def _require_str_tuple(
    value: object, *, field_name: str, resource: str
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{resource}: field {field_name!r} must be a list of strings.")
    return tuple(
        _require_str(item, field_name=f"{field_name}[{index}]", resource=resource)
        for index, item in enumerate(value)
    )


def load_uk_spine_swap_signed_differences(
    resource: str = UK_SPINE_SWAP_SIGNED_DIFFERENCES_RESOURCE,
) -> UKSignedDifferenceRegister:
    """Load and validate the committed signed-differences register."""

    payload = json.loads(_resource_text(resource))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resource}: expected a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"{resource}: unsupported schema_version {schema_version!r}; expected 1."
        )

    scope_note = _require_str(
        payload.get("scope_note"), field_name="scope_note", resource=resource
    )

    raw_differences = payload.get("differences")
    if not isinstance(raw_differences, Sequence) or isinstance(raw_differences, str):
        raise ValueError(f"{resource}: 'differences' must be a list.")

    differences: list[UKSignedDifference] = []
    for index, raw in enumerate(raw_differences):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{resource}: differences[{index}] must be an object.")
        where = f"differences[{index}]"

        identifier = _require_str(
            raw.get("id"), field_name=f"{where}.id", resource=resource
        )
        if not _ID.match(identifier):
            raise ValueError(
                f"{resource}: {where}.id must be lowercase kebab-case, "
                f"got {identifier!r}."
            )

        raw_scope = raw.get("scope")
        if not isinstance(raw_scope, Mapping):
            raise ValueError(f"{resource}: {where}.scope must be an object.")

        adjudicated_on = _require_str(
            raw.get("adjudicated_on"),
            field_name=f"{where}.adjudicated_on",
            resource=resource,
        )
        if not _ISO_DATE.match(adjudicated_on):
            raise ValueError(
                f"{resource}: {where}.adjudicated_on must be an ISO date "
                f"(YYYY-MM-DD), got {adjudicated_on!r}."
            )

        # Signed differences are permanent adjudications, unlike gate
        # exclusions. An expiry here would silently turn one back into a
        # defect on a date nobody is watching.
        if "expires_on" in raw:
            raise ValueError(
                f"{resource}: {where} carries 'expires_on'. Signed differences "
                "do not expire; an expiring suppression belongs in the "
                "per-gate reviewed-exclusion register instead."
            )

        differences.append(
            UKSignedDifference(
                id=identifier,
                difference_class=_require_member(
                    raw.get("class"),
                    allowed=SIGNED_DIFFERENCE_CLASSES,
                    field_name=f"{where}.class",
                    resource=resource,
                ),
                surface=_require_member(
                    raw_scope.get("surface"),
                    allowed=SIGNED_DIFFERENCE_SURFACES,
                    field_name=f"{where}.scope.surface",
                    resource=resource,
                ),
                expectation=_require_member(
                    raw.get("expectation"),
                    allowed=SIGNED_DIFFERENCE_EXPECTATIONS,
                    field_name=f"{where}.expectation",
                    resource=resource,
                ),
                columns=_require_str_tuple(
                    raw_scope.get("columns"),
                    field_name=f"{where}.scope.columns",
                    resource=resource,
                ),
                entities=_require_str_tuple(
                    raw_scope.get("entities"),
                    field_name=f"{where}.scope.entities",
                    resource=resource,
                ),
                magnitude_evidence=_require_str(
                    raw.get("magnitude_evidence"),
                    field_name=f"{where}.magnitude_evidence",
                    resource=resource,
                ),
                evidence=_require_str(
                    raw.get("evidence"),
                    field_name=f"{where}.evidence",
                    resource=resource,
                ),
                adjudicator=_require_str(
                    raw.get("adjudicator"),
                    field_name=f"{where}.adjudicator",
                    resource=resource,
                ),
                adjudicated_on=adjudicated_on,
            )
        )

    return UKSignedDifferenceRegister(
        differences=tuple(differences),
        scope_note=scope_note,
        schema_version=1,
    )
