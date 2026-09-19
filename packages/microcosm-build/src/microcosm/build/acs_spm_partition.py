"""Propose ACS SPM membership through the canonical, source-aware adapter seam.

This module does not mutate a Frame, assign tax units, allocate unit amounts, or
calculate poverty. It deliberately lives outside ``us_runtime``: importing that
package initializes the legacy country registry. See docs/acs-spm-partition.md
for the reconstruction policy, unresolved cases and canonical dependency pin.

The assembler this adapter imports is not declared as a workspace dependency, so
runtimes differ: ``probe_acs_spm_assembler`` reports what the imported assembler
actually does on fixed synthetic rosters, and a partition that would need an
unsupported one refuses with ``UnsupportedAssembler`` rather than guessing.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ACS_SPM_PARTITION_POLICY = "acs_spm_partition_v1"
ACS_SPM_DEVELOPMENT_POLICY = "acs_spm_development_reconstruction_v1"
_REQUIRED = (
    "person_id",
    "person_household_id",
    "person_spm_unit_id",
    "SPORDER",
    "RELSHIPP",
    "AGEP",
    "MAR",
    "TYPEHUGQ",
)
# Native ACS codes, not the lossy CPS-compatible A_EXPRRP recode. This is a
# partition recode; it does not assert parentage between non-reference people.
_REFERENCE_FAMILY = frozenset({20, 21, 23, *range(25, 34)})
_SPOUSES = frozenset({21, 23})
_PARTNERS = frozenset({22, 24})
_SECONDARY = frozenset({22, 24, 34, 35, 36})
_PROBE_POINTERS = ("mother_id", "spouse_id", "unmarried_partner_id")
# Fixed synthetic probe rosters, in the exact shape ``_assembly_view`` emits.
# Household 1 is an under-15 residual child with an accepted parent link, whose
# parent must resolve before the unrelated-under-15 residual attachment.
# Household 2 is the foster-age boundary, exercised from either side.
_PROBE_ROWS = (
    (1, 1, 1, 45, "head", 1, False, pd.NA),
    (2, 1, 2, 35, "other", pd.NA, False, pd.NA),
    (3, 1, 3, 8, "other", pd.NA, False, 2),
    (4, 2, 1, 45, "head", 1, False, pd.NA),
    (5, 2, 2, 21, "foster child", pd.NA, True, pd.NA),
    (6, 2, 3, 22, "foster child", pd.NA, True, pd.NA),
)
_PROBE_EXPECTED = {
    1: ({frozenset({1}), frozenset({2, 3})}, "incompatible_parent_link_order"),
    2: ({frozenset({4, 5}), frozenset({6})}, "incompatible_foster_boundary"),
}


@dataclass(frozen=True)
class AcsSpmAssemblerProbe:
    """What the imported assembler did on the fixed rosters, and which file it is.

    ``supported`` means only that this runtime's assembler reproduced the two
    membership capabilities and the diagnostics call contract the probe
    exercises, on those rosters. It is not a version, general-compatibility,
    empirical-accuracy, consumer-acceptance or release claim, and a supported
    probe does not qualify any partition for measurement use.
    """

    supported: bool
    reason: Literal[
        "supported",
        "assembler_unavailable",
        "incompatible_parent_link_order",
        "incompatible_foster_boundary",
        "incompatible_call_contract",
    ]
    module_file: str | None
    module_sha256: str | None

    def as_provenance(self) -> dict[str, Any]:
        """JSON-safe probe identity; never the assembler object or its output."""
        return {
            "supported": self.supported,
            "reason": self.reason,
            "module_file": self.module_file,
            "module_sha256": self.module_sha256,
        }


class UnsupportedAssembler(Exception):  # noqa: N818
    """This runtime's assembler cannot produce the membership this adapter needs.

    Deliberately not a ``ValueError``: refusals of the caller's source evidence
    stay distinguishable from a runtime whose assembler is absent or divergent.
    The name states the runtime condition callers branch on rather than carrying
    the ``Error`` suffix N818 wants, so the marker on each dependent test reads
    as the capability it needs.
    """

    def __init__(self, probe: AcsSpmAssemblerProbe) -> None:
        super().__init__(
            "Canonical SPM assembler is unsupported "
            f"({probe.reason}; module_file={probe.module_file})."
        )
        self.probe = probe


def _assembler_identity(assembler: Any) -> tuple[str | None, str | None]:
    """The assembler's defining file and its hash, when both are readable."""
    try:
        path = inspect.getsourcefile(assembler)
    except Exception:
        return None, None
    if not path:
        return None, None
    try:
        return path, hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return path, None


def probe_acs_spm_assembler() -> AcsSpmAssemblerProbe:
    """Exercise the imported assembler; report a typed verdict, never raise.

    Checks the two membership capabilities this adapter depends on — an accepted
    parent link resolving before the unrelated-under-15 residual attachment, and
    the foster-child age boundary — and the diagnostics call contract its
    provenance records: a ``(ids, diagnostics)`` pair, complete IDs aligned to
    the input, and a string-keyed mapping carrying a non-native ``method`` and a
    ``fallback_rules_used`` sequence of strings. It does not check rule names,
    any other roster, or anything outside those checks. Any failure to import
    the assembler — ``ImportError`` included — reports it unavailable; every
    other ordinary failure is reported as an unsupported reason rather than
    raised. See ``AcsSpmAssemblerProbe`` for what a supported verdict does and
    does not establish.
    """
    try:
        from spm_calculator import spm_unit_id
    except ImportError:
        return AcsSpmAssemblerProbe(False, "assembler_unavailable", None, None)
    except Exception:
        return AcsSpmAssemblerProbe(False, "incompatible_call_contract", None, None)
    module_file, module_sha256 = _assembler_identity(spm_unit_id)

    def verdict(reason: str) -> AcsSpmAssemblerProbe:
        return AcsSpmAssemblerProbe(
            reason == "supported", reason, module_file, module_sha256
        )

    try:
        roster = pd.DataFrame(
            list(_PROBE_ROWS),
            columns=[
                "person_id",
                "household_id",
                "line_number",
                "age",
                "relationship_to_head",
                "family_id",
                "is_foster_child",
                "parent_id",
            ],
        )
        roster["family_id"] = roster.family_id.astype("Int64")
        roster["parent_id"] = roster.parent_id.astype("Int64")
        for column in _PROBE_POINTERS:
            roster[column] = pd.Series(pd.NA, index=roster.index, dtype="Int64")
        returned = spm_unit_id(roster, diagnostics=True)
        if not (isinstance(returned, tuple) and len(returned) == 2):
            return verdict("incompatible_call_contract")
        ids, diagnostics = returned
        ids = pd.Series(ids).reset_index(drop=True)
        rules = (
            diagnostics.get("fallback_rules_used")
            if isinstance(diagnostics, Mapping)
            else None
        )
        if not (
            len(ids) == len(roster)
            and not ids.isna().any()
            and isinstance(diagnostics, Mapping)
            and all(isinstance(key, str) for key in diagnostics)
            and isinstance(diagnostics.get("method"), str)
            # The allowlisted view carries no old SPM ID; an assembler that
            # echoed one back would be reproducing the partition being replaced.
            and diagnostics["method"] != "native_spm_id"
            and isinstance(rules, Sequence)
            and not isinstance(rules, (str, bytes))
            and all(isinstance(rule, str) for rule in rules)
        ):
            return verdict("incompatible_call_contract")
        assigned = roster.assign(_unit=ids.to_numpy())
        if not assigned.groupby("_unit").household_id.nunique().le(1).all():
            return verdict("incompatible_call_contract")
        for household, (expected, reason) in _PROBE_EXPECTED.items():
            rows = assigned.loc[assigned.household_id.eq(household)]
            if {
                frozenset(unit.person_id) for _, unit in rows.groupby("_unit")
            } != expected:
                return verdict(reason)
    except Exception:
        return verdict("incompatible_call_contract")
    return verdict("supported")


@dataclass(frozen=True)
class AcsSpmLink:
    """An approved inferred link; parent links point from child to parent.

    IDs are global person IDs, never row positions or CPS line-number aliases.
    ``rule_id`` identifies the caller's reviewed inference, not a Census program.
    """

    person_id: int
    relative_id: int
    kind: Literal["parent", "spouse", "partner"]
    rule_id: str


@dataclass(frozen=True)
class AcsSpmLinkAssessment:
    """Caller review of secondary relationships, including rejected candidates.

    ``complete`` declares that the supplied links (possibly none) resolve this
    person's secondary relationships under ``rule_id``. ``ambiguous`` quarantines
    the household. A supplied link alone is not a complete candidate assessment.
    """

    person_id: int
    status: Literal["complete", "ambiguous"]
    rule_id: str


@dataclass(frozen=True)
class AcsSpmRoleDecision:
    """Explicit reviewed classification for a non-head/spouse aged 15–17."""

    person_id: int
    value: bool
    rule_id: str


@dataclass(frozen=True)
class AcsSpmPartitionResult:
    """Proposed person membership and audits; never a replacement data Frame."""

    membership: pd.DataFrame
    links: pd.DataFrame
    crosswalk: pd.DataFrame
    regrouping: pd.DataFrame
    provenance: dict[str, Any]

    def require_resolved(self) -> None:
        """Require no pending policy inputs, not observational completeness."""
        if self.membership.proposed_spm_unit_id.isna().any():
            raise ValueError("ACS SPM partition has unresolved household membership.")
        included = self.membership.measurement_status.eq("included_acs_household")
        if self.membership.loc[included, "independent_minor_role"].isna().any():
            raise ValueError("ACS SPM independent-minor role remains unresolved.")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rule(value: str) -> None:
    _require(
        isinstance(value, str) and bool(value.strip()),
        "A nonempty rule_id is required.",
    )


def _person_id(value: int) -> None:
    _require(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and value >= 0,
        "Relationship evidence requires exact nonnegative person IDs.",
    )


def _validate_inputs(
    persons: pd.DataFrame,
    spm_units: pd.DataFrame,
    household_person_counts: Mapping[int, int],
) -> pd.DataFrame:
    _require(
        persons.columns.is_unique and spm_units.columns.is_unique,
        "Columns must be unique.",
    )
    _require(
        set(_REQUIRED).issubset(persons),
        "ACS partition is missing required source columns.",
    )
    _require(
        len(persons) > 0, "ACS partition requires nonempty, household-complete input."
    )
    work = persons.loc[:, _REQUIRED].copy().reset_index(drop=True)
    for name in _REQUIRED[:4]:
        _require(
            work[name]
            .map(
                lambda x: (
                    isinstance(x, Integral)
                    and not isinstance(x, (bool, np.bool_))
                    and x >= 0
                )
            )
            .all(),
            f"{name} requires nonnegative exact integer IDs.",
        )
    _require(work.person_id.is_unique, "person_id must be unique.")
    _require(
        all(
            isinstance(key, Integral)
            and not isinstance(key, (bool, np.bool_))
            and isinstance(value, Integral)
            and not isinstance(value, (bool, np.bool_))
            and value > 0
            for key, value in household_person_counts.items()
        ),
        "Expected household counts require exact integer keys and positive counts.",
    )
    _require(
        work.groupby("person_household_id").size().to_dict()
        == dict(household_person_counts),
        "ACS input must contain complete households matching declared person counts.",
    )
    _require(work.SPORDER.gt(0).all(), "SPORDER must be positive.")
    _require(
        not work.duplicated(["person_household_id", "SPORDER"]).any(),
        "Household SPORDER pointer keys must be unique.",
    )
    for name, low, high in (
        ("RELSHIPP", 20, 38),
        ("AGEP", 0, 99),
        ("MAR", 1, 5),
        ("TYPEHUGQ", 1, 3),
    ):
        numeric = pd.to_numeric(work[name], errors="coerce")
        _require(
            not work[name].map(lambda value: isinstance(value, (bool, np.bool_))).any()
            and (
                numeric.notna()
                & numeric.between(low, high)
                & numeric.eq(numeric.round())
            ).all(),
            f"{name} requires complete valid ACS integer values.",
        )
        work[name] = numeric.astype("int64")
    _require(
        "spm_unit_id" in spm_units and spm_units.spm_unit_id.is_unique,
        "Old spm_unit_id table must be present and unique.",
    )
    _require(
        set(spm_units.spm_unit_id) == set(work.person_spm_unit_id),
        "Old SPM table must cover exactly the input memberships.",
    )
    _require(
        work.groupby("person_spm_unit_id").person_household_id.nunique().le(1).all(),
        "Old SPM membership must not cross households.",
    )
    for _, household in work.groupby("person_household_id", sort=False):
        _require(
            household.TYPEHUGQ.nunique() == 1, "Household TYPEHUGQ must be consistent."
        )
        kind = int(household.TYPEHUGQ.iloc[0])
        if kind == 1:
            _require(
                household.RELSHIPP.eq(20).sum() == 1,
                "Housing unit requires exactly one reference person.",
            )
            _require(
                household.RELSHIPP.le(36).all(),
                "Housing unit contains a GQ relationship.",
            )
            spouses = household.loc[household.RELSHIPP.isin(_SPOUSES)]
            if len(spouses):
                pair = household.loc[household.RELSHIPP.isin({20, *_SPOUSES})]
                _require(
                    len(spouses) == 1 and pair.MAR.eq(1).all(),
                    "Observed reference/spouse pair requires one spouse and MAR=1.",
                )
        else:
            _require(
                household.RELSHIPP.eq(35 + kind).all(),
                "GQ type and raw relationship disagree.",
            )
    return work


def _secondary_status(
    work: pd.DataFrame, assessments: Sequence[AcsSpmLinkAssessment]
) -> tuple[pd.Series, pd.Series]:
    statuses = pd.Series("not_required", index=work.index)
    rules = pd.Series("reference_relationship", index=work.index)
    for _, household in work.loc[work.TYPEHUGQ.eq(1)].groupby(
        "person_household_id", sort=False
    ):
        candidates = household.loc[household.RELSHIPP.isin(_SECONDARY)]
        for pos, person in candidates.iterrows():
            if person.RELSHIPP in _PARTNERS or (
                person.RELSHIPP == 35 and person.AGEP < 22
            ):
                continue
            others = candidates.drop(index=pos)
            # This is a conservative review screen, not inferred parentage.
            # Any minor pair, or a 15+ year age gap, needs caller assessment.
            # Marital status alone does not select a resident spouse. Adult
            # residual separation is an explicit, counted development policy.
            possible = (others.AGEP - person.AGEP).abs().ge(15) | (
                others.AGEP.lt(18) | (person.AGEP < 18)
            )
            statuses.loc[pos] = (
                "unassessed"
                if possible.any()
                else (
                    "modeled_residual_child_attachment"
                    if person.AGEP < 15
                    else "modeled_residual_separation"
                )
            )
            rules.loc[pos] = (
                "secondary_candidate_screen_v1"
                if possible.any()
                else (
                    "unrelated_child_attachment_v1"
                    if person.AGEP < 15
                    else "residual_separation_v1"
                )
            )
    by_id = work.set_index("person_id", drop=False)
    positions = dict(zip(work.person_id, work.index, strict=True))
    seen = set()
    for assessment in assessments:
        _rule(assessment.rule_id)
        _person_id(assessment.person_id)
        _require(
            assessment.person_id in by_id.index and assessment.person_id not in seen,
            "Assessment person IDs must be present and unique.",
        )
        _require(
            assessment.status in {"complete", "ambiguous"},
            "Invalid secondary assessment status.",
        )
        row = by_id.loc[assessment.person_id]
        _require(
            row.TYPEHUGQ == 1, "GQ secondary assessment is outside this reconstruction."
        )
        pos = positions[assessment.person_id]
        statuses.loc[pos] = assessment.status
        rules.loc[pos] = assessment.rule_id
        seen.add(assessment.person_id)
    return statuses, rules


def _assembly_view(
    work: pd.DataFrame, links: Sequence[AcsSpmLink]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Construct an allowlisted view; physically exclude every old SPM/family ID,
    # A_EXPRRP, PEPAR1/2 and Census merge flag. Canonical alias lookup must not
    # rediscover the old household fallback or an unqualified inferred pointer.
    view = pd.DataFrame(
        {
            "person_id": work.person_id,
            "household_id": work.person_household_id,
            "line_number": work.SPORDER,
            "age": work.AGEP,
            "relationship_to_head": "other",
            "family_id": pd.Series(pd.NA, index=work.index, dtype="Int64"),
            "is_foster_child": work.RELSHIPP.eq(35),
        }
    )
    for column in ("parent_id", "mother_id", "spouse_id", "unmarried_partner_id"):
        view[column] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    view.loc[work.RELSHIPP.eq(20), "relationship_to_head"] = "head"
    view.loc[work.RELSHIPP.isin(_SPOUSES), "relationship_to_head"] = "spouse"
    view.loc[work.RELSHIPP.isin({25, 26, 27}), "relationship_to_head"] = "child"
    view.loc[work.RELSHIPP.eq(35), "relationship_to_head"] = "foster child"
    evidence = []
    for _, household in work.loc[work.TYPEHUGQ.eq(1)].groupby(
        "person_household_id", sort=False
    ):
        head = household.loc[household.RELSHIPP.eq(20)].iloc[0]
        core = household.index[household.RELSHIPP.isin(_REFERENCE_FAMILY)]
        view.loc[core, "family_id"] = int(head.SPORDER)
        for pos, person in household.iterrows():
            if person.RELSHIPP in _PARTNERS | _SPOUSES:
                kind = "partner" if person.RELSHIPP in _PARTNERS else "spouse"
                column = "unmarried_partner_id" if kind == "partner" else "spouse_id"
                view.loc[pos, column] = int(head.SPORDER)
                evidence.append(
                    {
                        "person_id": int(person.person_id),
                        "relative_id": int(head.person_id),
                        "kind": kind,
                        "source": "source_observed",
                        "rule_id": "acs_relshipp_reference_pair",
                    }
                )
            elif person.RELSHIPP in {25, 26, 27}:
                view.loc[pos, "parent_id"] = int(head.SPORDER)
                evidence.append(
                    {
                        "person_id": int(person.person_id),
                        "relative_id": int(head.person_id),
                        "kind": "parent",
                        "source": "source_observed",
                        "rule_id": "acs_relshipp_reference_child",
                    }
                )
    by_id = work.set_index("person_id", drop=False)
    positions = dict(zip(work.person_id, work.index, strict=True))
    used = set()
    for link in links:
        _rule(link.rule_id)
        _person_id(link.person_id)
        _person_id(link.relative_id)
        _require(link.kind in {"parent", "spouse", "partner"}, "Invalid link kind.")
        _require(
            link.person_id in by_id.index
            and link.relative_id in by_id.index
            and link.person_id != link.relative_id,
            "Link endpoints must identify distinct present people.",
        )
        person, relative = by_id.loc[link.person_id], by_id.loc[link.relative_id]
        _require(
            person.person_household_id == relative.person_household_id,
            "Link endpoints must share a household.",
        )
        _require(person.TYPEHUGQ == 1, "GQ links are outside this reconstruction.")
        if link.kind == "spouse":
            _require(
                person.MAR == 1 and relative.MAR == 1,
                "Inferred spouse links require both raw MAR values to be 1.",
            )
        signature = (link.person_id, link.relative_id, link.kind)
        _require(signature not in used, "Duplicate link.")
        used.add(signature)
        pos = positions[link.person_id]
        columns = (
            ("parent_id", "mother_id")
            if link.kind == "parent"
            else (
                ("spouse_id",) if link.kind == "spouse" else ("unmarried_partner_id",)
            )
        )
        _require(
            not any(
                view.loc[pos, column] == int(relative.SPORDER)
                for column in columns
                if pd.notna(view.loc[pos, column])
            ),
            "Link duplicates observed or supplied relationship.",
        )
        empty = [column for column in columns if pd.isna(view.loc[pos, column])]
        _require(bool(empty), "Link exceeds supported parent or partner cardinality.")
        view.loc[pos, empty[0]] = int(relative.SPORDER)
        evidence.append(
            {
                "person_id": link.person_id,
                "relative_id": link.relative_id,
                "kind": link.kind,
                "source": "approved_inference",
                "rule_id": link.rule_id,
            }
        )
    # Cardinality applies to both endpoints of the directional source statements.
    for kind in ("spouse", "partner"):
        pairings: dict[int, set[int]] = {}
        for item in evidence:
            if item["kind"] != kind:
                continue
            pairings.setdefault(item["person_id"], set()).add(item["relative_id"])
            pairings.setdefault(item["relative_id"], set()).add(item["person_id"])
        _require(
            all(len(relatives) <= 1 for relatives in pairings.values()),
            f"Conflicting {kind} links at an observed or inferred endpoint.",
        )
    parent_graph: dict[int, list[int]] = {}
    for item in evidence:
        if item["kind"] == "parent":
            parent_graph.setdefault(item["person_id"], []).append(item["relative_id"])
    active, visited = set(), set()

    def visit(person_id: int) -> None:
        _require(person_id not in active, "Parent links contain a cycle.")
        if person_id in visited:
            return
        active.add(person_id)
        for parent_id in parent_graph.get(person_id, ()):
            visit(parent_id)
        active.remove(person_id)
        visited.add(person_id)

    for person_id in parent_graph:
        visit(person_id)
    return view, pd.DataFrame(
        evidence, columns=["person_id", "relative_id", "kind", "source", "rule_id"]
    )


def _roles(
    work: pd.DataFrame,
    decisions: Sequence[AcsSpmRoleDecision],
    proposed: pd.Series,
    links: pd.DataFrame,
    *,
    policy: str,
    minor_partner_role: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = pd.Series(False, index=work.index, dtype="boolean")
    source = pd.Series("age_not_role_sensitive", index=work.index)
    rules = pd.Series("canonical_role_age_scope", index=work.index)
    sensitive = work.AGEP.between(15, 17)
    values.loc[sensitive] = pd.NA
    source.loc[sensitive] = "unresolved"
    rules.loc[sensitive] = "classification_required"
    observed = work.RELSHIPP.isin({20, *_SPOUSES})
    values.loc[observed] = True
    # An observed ACS reference/spouse relationship, which is a strict subset of
    # the ASEC independence rule; never an observed financial-independence fact.
    source.loc[observed] = "observed_relationship_rule"
    rules.loc[observed] = "acs_relshipp_reference_head_or_spouse"
    outside = work.TYPEHUGQ.ne(1)
    values.loc[outside] = pd.NA
    source.loc[outside] = "outside_acs_household_universe"
    rules.loc[outside] = "not_classified"
    positions = dict(zip(work.person_id, work.index, strict=True))
    if policy == ACS_SPM_DEVELOPMENT_POLICY:
        modeled = sensitive & ~observed & ~outside
        values.loc[modeled] = False
        source.loc[modeled] = "modeled_assumption"
        rules.loc[modeled] = "nonreference_minor_child_role_v1"
        partners = modeled & work.RELSHIPP.isin(_PARTNERS)
        values.loc[partners] = minor_partner_role
        rules.loc[partners] = "minor_reference_partner_role_sensitivity_v1"
        parent_edges = links.loc[links.kind.eq("parent")]
        parent_lookup: dict[int, set[int]] = {}
        for child, parent in parent_edges[["person_id", "relative_id"]].itertuples(
            index=False, name=None
        ):
            parent_lookup.setdefault(child, set()).add(parent)
        sensitive_units = set(proposed.loc[modeled].dropna())
        role_work = work.assign(proposed=proposed).loc[proposed.isin(sensitive_units)]
        for _, unit in role_work.groupby("proposed"):
            if unit.RELSHIPP.eq(20).any() or unit.AGEP.ge(18).any():
                continue
            eligible = unit.index[unit.AGEP.between(15, 17)]
            if len(unit) == 1 and len(eligible) == 1:
                values.loc[eligible] = True
                rules.loc[eligible] = "minor_singleton_unit_reference_v1"
                continue
            children = {pid for pid in unit.person_id if pid in parent_lookup}
            roots = {
                parent for pid in children for parent in parent_lookup[pid]
            } - children
            if len(roots) == 1:
                root = positions[next(iter(roots))]
                if work.loc[root, "AGEP"] >= 15:
                    values.loc[root] = True
                    rules.loc[root] = "accepted_parent_component_reference_v1"
                    continue
            # Multiple minor parent roots or an accepted secondary couple with
            # no unambiguous reference require an explicit role decision.
            values.loc[eligible] = pd.NA
            source.loc[eligible] = "unresolved"
            rules.loc[eligible] = "ambiguous_secondary_minor_reference"
    seen = set()
    for decision in decisions:
        _rule(decision.rule_id)
        _person_id(decision.person_id)
        _require(
            decision.person_id in positions and decision.person_id not in seen,
            "Role person IDs must be present and unique.",
        )
        _require(
            type(decision.value) is bool, "Role decision must be an explicit bool."
        )
        pos = positions[decision.person_id]
        _require(
            not observed.loc[pos],
            "Role decision cannot replace an observed head/spouse role.",
        )
        _require(
            sensitive.loc[pos] and not outside.loc[pos],
            "Role decision applies only to included non-head/spouse ages 15–17.",
        )
        values.loc[pos] = decision.value
        source.loc[pos] = "approved_inference"
        rules.loc[pos] = decision.rule_id
        seen.add(decision.person_id)
    return values, source, rules


def reconstruct_acs_spm_partition(
    persons: pd.DataFrame,
    spm_units: pd.DataFrame,
    *,
    household_person_counts: Mapping[int, int],
    policy: str = ACS_SPM_PARTITION_POLICY,
    minor_partner_role: bool = True,
    links: Sequence[AcsSpmLink] = (),
    assessments: Sequence[AcsSpmLinkAssessment] = (),
    role_decisions: Sequence[AcsSpmRoleDecision] = (),
) -> AcsSpmPartitionResult:
    """Propose a partition for complete ACS households using canonical assembly.

    The original person and SPM tables remain untouched. Only old unit IDs/column
    names are read from ``spm_units``; amounts are never allocated or copied.
    Strict mode leaves ambiguous candidate households without proposed membership.
    The explicit development preset applies labeled, counted fallback assumptions.
    Passing ``require_resolved`` is not composition, empirical or release acceptance.

    Raises ``UnsupportedAssembler`` when a household would reach an assembler that
    ``probe_acs_spm_assembler`` reports unsupported. Source evidence is validated
    first, so a refusable input still refuses with ``ValueError`` on any runtime,
    and no fallback membership is ever guessed in the assembler's place.
    """
    _require(
        policy in {ACS_SPM_PARTITION_POLICY, ACS_SPM_DEVELOPMENT_POLICY},
        "Unknown ACS SPM construction policy.",
    )
    _require(
        type(minor_partner_role) is bool, "minor_partner_role must be an explicit bool."
    )
    work = _validate_inputs(persons, spm_units, household_person_counts)
    status, status_rule = _secondary_status(work, assessments)
    view, link_evidence = _assembly_view(work, links)
    source_unresolved_households = set(
        work.loc[status.isin({"unassessed", "ambiguous"}), "person_household_id"]
    )
    unresolved_households = (
        source_unresolved_households if policy == ACS_SPM_PARTITION_POLICY else set()
    )
    eligible = work.TYPEHUGQ.eq(1) & ~work.person_household_id.isin(
        unresolved_households
    )
    proposed = pd.Series(pd.NA, index=work.index, dtype="string")
    diagnostics = None
    probe = None
    if eligible.any():
        probe = probe_acs_spm_assembler()
        if not probe.supported:
            raise UnsupportedAssembler(probe)
        from spm_calculator import spm_unit_id

        # The canonical helper, not this adapter, assembles connected units.
        ids, diagnostics = spm_unit_id(
            view.loc[eligible].reset_index(drop=True), diagnostics=True
        )
        _require(
            len(ids) == int(eligible.sum()) and not ids.isna().any(),
            "Canonical assembler returned incomplete membership.",
        )
        assigned = work.loc[eligible, ["person_id", "person_household_id"]].copy()
        assigned["canonical_id"] = ids.to_numpy()
        _require(
            assigned.groupby("canonical_id").person_household_id.nunique().le(1).all(),
            "Canonical units cross households.",
        )
        for _, unit in assigned.groupby("canonical_id", sort=False):
            proposed.loc[unit.index] = f"acs-spm-v1:p{int(unit.person_id.min())}"
    for _, unit in work.loc[work.TYPEHUGQ.ne(1)].groupby(
        "person_spm_unit_id", sort=False
    ):
        proposed.loc[unit.index] = f"acs-spm-v1:p{int(unit.person_id.min())}"
    role, role_source, role_rule = _roles(
        work,
        role_decisions,
        proposed,
        link_evidence,
        policy=policy,
        minor_partner_role=minor_partner_role,
    )
    parent_status = pd.Series("unknown", index=work.index)
    positions = dict(zip(work.person_id, work.index, strict=True))
    parents = link_evidence.loc[link_evidence.kind.eq("parent")]
    for pid, edges in parents.groupby("person_id"):
        sources = set(edges.source)
        parent_status.loc[positions[pid]] = (
            "observed_and_inferred_parent_links"
            if len(sources) > 1
            else (
                "observed_parent_link"
                if "source_observed" in sources
                else "approved_inferred_parent_link"
            )
        )
    assumption = pd.Series("reference_relationship_rule", index=work.index)
    residual = work.RELSHIPP.isin({34, 36}) | (work.RELSHIPP.eq(35) & work.AGEP.ge(22))
    sharing = set(link_evidence.person_id) | set(link_evidence.relative_id)
    unlinked = residual & ~work.person_id.isin(sharing)
    assumption.loc[unlinked & work.AGEP.lt(15)] = "parent_unknown_reference_pooling"
    assumption.loc[unlinked & work.AGEP.ge(15)] = "parent_unknown_residual_separation"
    assumption.loc[residual & ~unlinked] = "accepted_sharing_links"
    unknown_own_parent = (
        residual & work.AGEP.lt(15) & ~work.person_id.isin(parents.person_id)
    )
    assumption.loc[unknown_own_parent & ~unlinked] = (
        "accepted_sharing_and_parent_unknown_reference_pooling"
    )
    assumption.loc[work.TYPEHUGQ.ne(1)] = "preserved_gq_membership"
    assumption.loc[work.person_household_id.isin(unresolved_households)] = (
        "pending_relationship_resolution"
    )
    membership = work.rename(columns={"person_spm_unit_id": "old_spm_unit_id"})
    membership["proposed_spm_unit_id"] = proposed
    membership["secondary_link_status"] = status
    membership["secondary_link_rule"] = status_rule
    membership["parent_link_status"] = parent_status
    membership["partition_assumption"] = assumption
    membership["independent_minor_role"] = role
    membership["role_source"] = role_source
    membership["role_rule"] = role_rule
    membership["measurement_status"] = np.where(
        work.TYPEHUGQ.eq(1), "included_acs_household", "outside_acs_household_universe"
    )
    crosswalk = (
        membership.groupby(
            ["person_household_id", "old_spm_unit_id", "proposed_spm_unit_id"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("person_count")
        .reset_index()
    )
    old_signatures = {
        frozenset(group.person_id) for _, group in work.groupby("person_spm_unit_id")
    }
    new_signatures = {
        frozenset(group.person_id)
        for _, group in membership.groupby("proposed_spm_unit_id")
    }
    changed = old_signatures != new_signatures
    action = (
        "unresolved_partition"
        if unresolved_households
        else ("requires_regrouping" if changed else "membership_unchanged")
    )
    fields = sorted(set(spm_units.columns) - {"spm_unit_id"})
    regrouping = pd.DataFrame({"field": fields, "action": [action] * len(fields)})
    return AcsSpmPartitionResult(
        membership=membership,
        links=link_evidence,
        crosswalk=crosswalk,
        regrouping=regrouping,
        provenance={
            "policy": policy,
            "minor_partner_role": minor_partner_role
            if policy == ACS_SPM_DEVELOPMENT_POLICY
            else None,
            "source_relationship": "ACS RELSHIPP; secondary links are approved inferences",
            "canonical_diagnostics": diagnostics,
            # Null when no household reached the assembler, so an unexercised
            # runtime never reads as an attested one.
            "assembler": None if probe is None else probe.as_provenance(),
            "person_count": len(work),
            "old_unit_count": len(old_signatures),
            "proposed_resolved_unit_count": len(new_signatures),
            "unresolved_households": len(unresolved_households),
            "source_unresolved_relationship_households": len(
                source_unresolved_households
            ),
            "partition_assumption_counts": assumption.value_counts().to_dict(),
            "role_rule_counts": role_rule.value_counts().to_dict(),
            "modeled_residual_separation_people": int(
                assumption.eq("parent_unknown_residual_separation").sum()
            ),
            "modeled_residual_child_attachment_people": int(
                (unknown_own_parent & proposed.notna()).sum()
            ),
            "unresolved_role_people": int((role.isna() & work.TYPEHUGQ.eq(1)).sum()),
            "gq_people_preserved": int(work.TYPEHUGQ.ne(1).sum()),
            "unit_amounts_allocated": False,
            "tax_units_modified": False,
            "certification": "development_reconstruction_only",
        },
    )
