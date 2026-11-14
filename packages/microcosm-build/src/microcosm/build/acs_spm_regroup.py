"""Explicit conservation-first regrouping of the twelve Build P SPM fields.

This pure table transformation imports no country runtime and performs no
benefit calculation. Its caller authenticates the parent and source-null
register, supplies complete membership and globally assigned numeric IDs, and
chooses the tenure sensitivity. The result is incomplete while any positive
childcare amount lacks an unambiguous reference-unit allocation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ACS_SPM_REGROUP_POLICY = "acs_spm_conservation_regroup_v1"
ACS_SPM_CHILDCARE_POLICY = "acs_spm_childcare_unique_successor_v1"
ACS_SPM_HOUSING_POLICY = "acs_spm_housing_common_imputation_v1"
_CHILDCARE = "spm_unit_pre_subsidy_childcare_expenses"
_ENERGY = "spm_unit_energy_subsidy"
_HOUSING = "takes_up_housing_assistance_if_eligible"
_TENURE = "spm_unit_tenure_type"
_DEFAULT_FIELDS = frozenset(
    {
        "receives_housing_assistance",
        _ENERGY,
        "takes_up_tanf_if_eligible",
        "takes_up_snap_if_eligible",
    }
)
_SPM_FIELDS = _DEFAULT_FIELDS | frozenset(
    {
        "spm_unit_id",
        _CHILDCARE,
        _HOUSING,
        _TENURE,
        "spm_unit_source_id",
        "spm_unit_support_channel",
        "spm_unit_support_clone_index",
        "spm_unit_spine",
    }
)
_TENURE_POLICIES = frozenset({"preserve_parent_tenure_v1", "acs_ten4_no_mortgage_v1"})
# The old mapping is the explicitly selected comparison, not the corrected
# Census SPM tenure definition. TEN=4 is rent-free; the corrected policy puts
# it in the no-mortgage threshold category without claiming property ownership.
_PARENT_TENURE = {
    1: "OWNER_WITH_MORTGAGE",
    2: "OWNER_WITHOUT_MORTGAGE",
    3: "RENTER",
    4: "RENTER",
}


@dataclass(frozen=True)
class AcsSpmLegacyDefaults:
    """Caller-authenticated provenance and values for the four ACS defaults.

    Hashes bind this declaration to evidence; this function does not open or
    verify those artifacts. Their verification belongs to the release caller.
    """

    acs_spine: str
    parent_sha256: str
    null_register_sha256: str
    values: Mapping[str, bool | float]

    def validate(self) -> None:
        """Reject incomplete declarations or a monetary default fanout."""
        if not isinstance(self.acs_spine, str) or not self.acs_spine:
            raise ValueError("An explicit ACS spine identity is required.")
        for name in ("parent_sha256", "null_register_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA256 identity.")
        if set(self.values) != _DEFAULT_FIELDS:
            raise ValueError(
                "legacy-default declaration must cover exactly four fields."
            )
        for field, value in self.values.items():
            if field == _ENERGY:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or value != 0
                ):
                    raise ValueError("The legacy-default energy amount must be zero.")
            elif not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"legacy-default {field} must be boolean.")


@dataclass(frozen=True)
class AcsSpmRegroupResult:
    """Proposed SPM fields, explicit lineage and an unresolved amount ledger."""

    spm_units: pd.DataFrame
    crosswalk: pd.DataFrame
    childcare_ledger: pd.DataFrame
    field_provenance: pd.DataFrame
    exceptions: pd.DataFrame
    metadata: Mapping[str, Any]

    def require_complete(self) -> None:
        """Refuse an unresolved allocation; this is not a release-quality gate."""
        if not self.exceptions.empty:
            raise ValueError(
                f"SPM regrouping has {len(self.exceptions)} unresolved childcare "
                "allocation(s); inspect the exception and amount ledgers."
            )


def _require_columns(table: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{name} lacks columns {missing}.")


def _ids(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        raise ValueError(f"{name} requires integer identities, not booleans.")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} requires numeric integer identities.") from error
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{name} has missing or nonfinite identities.")
    if ((numeric < 0) | (numeric > np.iinfo(np.int64).max)).any():
        raise ValueError(f"{name} must fit nonnegative int64.")
    if pd.api.types.is_float_dtype(numeric) and (numeric > 2**53).any():
        raise ValueError(f"{name} has unsafe floating-point integer identities.")
    integer = numeric.astype(np.int64)
    if not integer.eq(numeric).all():
        raise ValueError(f"{name} contains fractional identities.")
    return integer


def _bool_values(values: pd.Series, name: str, *, nullable: bool = False) -> pd.Series:
    present = values.dropna()
    valid = present.map(
        lambda value: (
            isinstance(value, (bool, np.bool_, int, np.integer)) and value in (0, 1)
        )
    )
    if not valid.all() or (not nullable and values.isna().any()):
        raise ValueError(f"{name} must contain boolean values.")
    return values.astype("boolean")


def _numeric(values: pd.Series, name: str) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains nonnumeric values.") from error
    if np.isinf(result.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError(f"{name} contains infinite values.")
    return result


def regroup_acs_spm_units(
    persons: pd.DataFrame,
    spm_units: pd.DataFrame,
    households: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    legacy_defaults: AcsSpmLegacyDefaults,
    tenure_policy: str,
    older_care_evidence: pd.Series | None = None,
) -> AcsSpmRegroupResult:
    """Regroup an authenticated parent without changing non-SPM inputs.

    ``membership`` covers exactly the input people and supplies globally
    allocated ``new_spm_unit_id`` and ``new_spm_unit_source_id``. Optional
    ``canonical_spm_unit_id`` labels are retained in the crosswalk. Unchanged
    member sets must retain their old numeric and source IDs. A new source ID
    identifies a generated component, not a Census-observed resource unit.

    ``older_care_evidence`` is a nullable boolean Series indexed by person ID.
    It records caller-provided affirmative/negative care evidence for ages
    16–17; omission leaves those minors unassessed. Ages 0–15 are potential
    recipients, not a declaration of tax-credit or benefit eligibility.
    """
    legacy_defaults.validate()
    if tenure_policy not in _TENURE_POLICIES:
        raise ValueError(f"Unsupported explicit tenure policy {tenure_policy!r}.")
    _require_columns(
        persons,
        {"person_id", "person_household_id", "person_spm_unit_id", "AGEP", "RELSHIPP"},
        "persons",
    )
    _require_columns(spm_units, set(_SPM_FIELDS), "SPM table")
    extra = sorted(set(spm_units.columns) - _SPM_FIELDS)
    if extra:
        raise ValueError(f"SPM table has unhandled fields: {extra}.")
    _require_columns(
        households, {"household_id", "TYPEHUGQ", "NP", "TEN"}, "households"
    )
    _require_columns(
        membership,
        {"person_id", "new_spm_unit_id", "new_spm_unit_source_id"},
        "membership",
    )
    person_ids = _ids(persons.person_id, "person_id")
    old_ids = _ids(spm_units.spm_unit_id, "spm_unit_id")
    household_ids = _ids(households.household_id, "household_id")
    map_ids = _ids(membership.person_id, "membership person_id")
    for ids, name in (
        (person_ids, "person"),
        (old_ids, "SPM"),
        (household_ids, "household"),
        (map_ids, "membership person"),
    ):
        if ids.duplicated().any():
            raise ValueError(f"Duplicate {name} identities.")
    if set(map_ids) != set(person_ids):
        raise ValueError("Membership must cover exactly the input person IDs.")
    mapped = membership.set_index(map_ids).reindex(person_ids)
    work = pd.DataFrame(
        {
            "person_id": person_ids.to_numpy(),
            "household_id": _ids(
                persons.person_household_id, "person_household_id"
            ).to_numpy(),
            "old_spm_unit_id": _ids(
                persons.person_spm_unit_id, "person_spm_unit_id"
            ).to_numpy(),
            "new_spm_unit_id": _ids(
                mapped.new_spm_unit_id, "new_spm_unit_id"
            ).to_numpy(),
            "new_spm_unit_source_id": _ids(
                mapped.new_spm_unit_source_id, "new_spm_unit_source_id"
            ).to_numpy(),
            "age": _numeric(persons.AGEP, "AGEP").to_numpy(),
            "relationship": _numeric(persons.RELSHIPP, "RELSHIPP").to_numpy(),
        }
    )
    if set(work.old_spm_unit_id) != set(old_ids):
        raise ValueError("Every old SPM unit must have its complete member set.")
    if not work.household_id.isin(household_ids).all():
        raise ValueError("Person references an absent household.")
    if (work.groupby("old_spm_unit_id").household_id.nunique() != 1).any():
        raise ValueError("An old SPM unit crosses households.")
    groups = work.groupby("new_spm_unit_id", sort=True)
    if (groups.old_spm_unit_id.nunique() != 1).any():
        raise ValueError("Regrouping does not authorize a merge of old SPM units.")
    if (groups.new_spm_unit_source_id.nunique() != 1).any():
        raise ValueError("A new SPM component has conflicting source identity values.")
    crosswalk = groups.agg(
        old_spm_unit_id=("old_spm_unit_id", "first"),
        new_spm_unit_source_id=("new_spm_unit_source_id", "first"),
        household_id=("household_id", "first"),
        person_count=("person_id", "size"),
    ).reset_index()
    if "canonical_spm_unit_id" in membership:
        work["canonical_spm_unit_id"] = mapped.canonical_spm_unit_id.to_numpy()
        labels = work.groupby("new_spm_unit_id").canonical_spm_unit_id
        if (
            labels.nunique(dropna=False) != 1
        ).any() or work.canonical_spm_unit_id.isna().any():
            raise ValueError(
                "Canonical membership labels must be complete and coherent."
            )
        crosswalk["canonical_spm_unit_id"] = crosswalk.new_spm_unit_id.map(
            labels.first()
        )
        if crosswalk.canonical_spm_unit_id.duplicated().any():
            raise ValueError(
                "A canonical SPM unit cannot have multiple numeric identities."
            )
    else:
        crosswalk["canonical_spm_unit_id"] = pd.NA
    old = spm_units.copy(deep=False).set_index(old_ids)
    old_source = _ids(spm_units.spm_unit_source_id, "spm_unit_source_id")
    source_by_old = pd.Series(old_source.to_numpy(), index=old_ids)
    crosswalk["old_spm_unit_source_id"] = crosswalk.old_spm_unit_id.map(source_by_old)
    successor_counts = crosswalk.groupby("old_spm_unit_id").size()
    crosswalk["member_set_unchanged"] = crosswalk.old_spm_unit_id.map(
        successor_counts
    ).eq(1)
    unchanged = crosswalk.member_set_unchanged
    if (
        crosswalk.loc[unchanged, "new_spm_unit_id"]
        .ne(crosswalk.loc[unchanged, "old_spm_unit_id"])
        .any()
        or crosswalk.loc[unchanged, "new_spm_unit_source_id"]
        .ne(crosswalk.loc[unchanged, "old_spm_unit_source_id"])
        .any()
    ):
        raise ValueError(
            "Unchanged member sets must retain old IDs and source identity."
        )
    changed = ~unchanged
    source_counts = crosswalk.new_spm_unit_source_id.value_counts()
    if crosswalk.loc[changed, "new_spm_unit_source_id"].map(source_counts).gt(1).any():
        raise ValueError("New sibling components cannot share a source identity.")
    if crosswalk.loc[changed, "new_spm_unit_source_id"].isin(old_source).any():
        raise ValueError(
            "A generated component source identity collides with an old origin."
        )
    crosswalk["source_identity_status"] = np.where(
        unchanged, "preserved_origin", "generated_component_identity"
    )
    acs_old = old.spm_unit_spine.eq(legacy_defaults.acs_spine)
    acs = crosswalk.old_spm_unit_id.map(acs_old).astype(bool)
    if (changed & ~acs).any():
        raise ValueError("This policy does not authorize splitting non-ACS units.")
    household = households.copy(deep=False).set_index(household_ids)
    household_kind = _numeric(household.TYPEHUGQ, "TYPEHUGQ")
    kind = crosswalk.household_id.map(household_kind)
    if (acs & ~kind.isin([1, 2, 3])).any():
        raise ValueError("ACS TYPEHUGQ must identify housing units or group quarters.")
    gq = acs & kind.isin([2, 3])
    if (changed & gq).any():
        raise ValueError("ACS group quarters must preserve their member sets.")
    acs_households = crosswalk.loc[acs, "household_id"].unique()
    counts = work.groupby("household_id").size().reindex(acs_households)
    np_source = _numeric(household.NP, "NP").reindex(acs_households)
    if np_source.isna().any() or not np_source.eq(counts).all():
        raise ValueError("ACS NP does not match the complete household roster.")
    for field, expected in legacy_defaults.values.items():
        values = old.loc[acs_old, field]
        if values.isna().any() or not values.eq(expected).all():
            raise ValueError(
                f"ACS {field} differs from authenticated legacy-default evidence."
            )
        if field != _ENERGY:
            _bool_values(values, f"legacy-default {field}")
    _bool_values(old[_HOUSING], _HOUSING)
    amount = _numeric(old[_CHILDCARE], "childcare")
    if (amount.dropna() < 0).any():
        raise ValueError("Negative childcare amounts are invalid.")
    hu = acs & kind.eq(1)
    ten = _numeric(household.TEN, "TEN")
    mapped_ten = crosswalk.household_id.map(ten)
    if (hu & ~mapped_ten.isin(_PARENT_TENURE)).any():
        raise ValueError("ACS housing units require native TEN 1–4.")
    if (gq & mapped_ten.notna()).any():
        raise ValueError("Group-quarters TEN must remain source-unavailable.")
    expected_tenure = mapped_ten.map(_PARENT_TENURE)
    old_tenure = crosswalk.old_spm_unit_id.map(old[_TENURE])
    if (hu & old_tenure.ne(expected_tenure)).any():
        raise ValueError("Parent SPM tenure does not match its native ACS mapping.")

    # Reindex only SPM fields. Never write back to persons, households, tax
    # memberships, geography, weights or the caller's membership proposal.
    output = old.reindex(crosswalk.old_spm_unit_id).reset_index(drop=True).copy()
    output["spm_unit_id"] = crosswalk.new_spm_unit_id.to_numpy()
    output["spm_unit_source_id"] = crosswalk.new_spm_unit_source_id.to_numpy()
    tenure_recode = hu & mapped_ten.eq(4) & (tenure_policy == "acs_ten4_no_mortgage_v1")
    output.loc[tenure_recode, _TENURE] = "OWNER_WITHOUT_MORTGAGE"
    crosswalk["acs_source_scope"] = acs.to_numpy()
    crosswalk["tenure_recode"] = tenure_recode.to_numpy()
    care = pd.Series(pd.NA, index=person_ids, dtype="boolean")
    if older_care_evidence is not None:
        if older_care_evidence.index.has_duplicates:
            raise ValueError("Older-care evidence has duplicate person IDs.")
        if not older_care_evidence.index.isin(person_ids).all():
            raise ValueError("Older-care evidence references an absent person.")
        care = _bool_values(older_care_evidence, "older care", nullable=True).reindex(
            person_ids
        )
    work["older_care"] = care.to_numpy()
    split_ids = successor_counts[successor_counts > 1].index
    split_people = work.loc[work.old_spm_unit_id.isin(split_ids)]
    split_groups = {
        key: positions
        for key, positions in split_people.groupby("old_spm_unit_id").groups.items()
    }
    successor_positions = crosswalk.groupby("old_spm_unit_id").groups
    ledger: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    field_provenance: list[dict[str, Any]] = []
    for old_id in sorted(split_ids):
        positions = successor_positions[old_id]
        people = work.loc[split_groups[old_id]]
        value = amount.loc[old_id]
        reason: str | None = None
        recipient_id: int | None = None
        if pd.isna(value):
            reason = "missing_childcare_amount"
        elif value == 0:
            status = "inherited_imputed_zero"
        else:
            age = people.age
            invalid_age = age.notna() & ((age < 0) | (age > 99) | (age % 1 != 0))
            if invalid_age.any():
                raise ValueError(
                    "ACS childcare screen requires valid integer AGEP 0–99."
                )
            older = age.between(16, 17)
            unknown = age.isna() | (older & people.older_care.isna())
            if unknown.any():
                reason = "unknown_age_or_care_evidence"
            else:
                potential = age.le(15) | (
                    older & people.older_care.eq(True).fillna(False)
                )
                candidates = people.loc[potential, "new_spm_unit_id"].unique()
                references = people.loc[people.relationship.eq(20), "new_spm_unit_id"]
                if len(references) != 1:
                    raise ValueError(
                        "Changed ACS household must have exactly one reference person."
                    )
                if len(candidates) == 0:
                    reason = "no_potential_childcare_recipient"
                elif len(candidates) > 1:
                    reason = "multiple_potential_childcare_successors"
                elif candidates[0] != references.iloc[0]:
                    reason = "sole_recipient_is_not_reference"
                else:
                    recipient_id = int(candidates[0])
                    status = "conserved_unique_reference_successor"
        if reason is not None:
            status = "unresolved"
            output.loc[positions, _CHILDCARE] = np.nan
            exceptions.append(
                {"old_spm_unit_id": old_id, "field": _CHILDCARE, "reason": reason}
            )
        else:
            output.loc[positions, _CHILDCARE] = 0
            if recipient_id is not None:
                recipient_position = positions[
                    crosswalk.loc[positions, "new_spm_unit_id"].eq(recipient_id)
                ]
                output.loc[recipient_position, _CHILDCARE] = value
        # Check the emitted cells, rather than reporting the intended amount.
        # Exactly one successor can carry a positive amount under this policy.
        emitted = output.loc[positions, _CHILDCARE]
        allocated = float(emitted.sum())
        if reason is None and (
            emitted.isna().any()
            or allocated != float(value)
            or (value > 0 and emitted.gt(0).sum() != 1)
        ):
            raise ValueError(
                "Childcare output failed exact single-allocation conservation."
            )
        if reason is not None and not emitted.isna().all():
            raise ValueError(
                "Unresolved childcare must remain unavailable in all successors."
            )
        unresolved = value if reason is not None else 0.0
        ledger.append(
            {
                "old_spm_unit_id": old_id,
                "old_spm_unit_source_id": source_by_old.loc[old_id],
                "old_amount": value,
                "allocated_amount": allocated,
                "unresolved_amount": unresolved,
                "recipient_spm_unit_id": recipient_id,
                "status": status,
            }
        )
        for position in positions:
            new_id = int(crosswalk.at[position, "new_spm_unit_id"])
            basis = status
            if (
                status == "conserved_unique_reference_successor"
                and new_id != recipient_id
            ):
                basis = "allocation_zero_nonrecipient_successor"
            field_provenance.append(
                {"new_spm_unit_id": new_id, "field": _CHILDCARE, "basis": basis}
            )
            field_provenance.append(
                {
                    "new_spm_unit_id": new_id,
                    "field": _HOUSING,
                    "basis": "inherited_common_imputation",
                }
            )
    for position in np.flatnonzero(tenure_recode):
        field_provenance.append(
            {
                "new_spm_unit_id": int(crosswalk.at[position, "new_spm_unit_id"]),
                "field": _TENURE,
                "basis": "native_rent_free_threshold_recode",
            }
        )
    ledger_frame = pd.DataFrame(
        ledger,
        columns=[
            "old_spm_unit_id",
            "old_spm_unit_source_id",
            "old_amount",
            "allocated_amount",
            "unresolved_amount",
            "recipient_spm_unit_id",
            "status",
        ],
    )
    for column in ("old_amount", "allocated_amount", "unresolved_amount"):
        ledger_frame[column] = ledger_frame[column].astype("Float64")
    ledger_frame["recipient_spm_unit_id"] = ledger_frame.recipient_spm_unit_id.astype(
        "Int64"
    )
    metadata = {
        "policy": ACS_SPM_REGROUP_POLICY,
        "childcare_policy": ACS_SPM_CHILDCARE_POLICY,
        "housing_policy": ACS_SPM_HOUSING_POLICY,
        "tenure_policy": tenure_policy,
        "tenure_recode_units": int(tenure_recode.sum()),
        "old_units": len(old),
        "new_units": len(output),
        "split_old_units": len(split_ids),
        "unchanged_member_sets": int(unchanged.sum()),
        "group_quarters_units_preserved": int(gq.sum()),
        "unresolved_childcare_units": len(exceptions),
        "potential_childcare_age_max": 15,
        "older_care_evidence_supplied": older_care_evidence is not None,
        "legacy_defaults": {
            "parent_sha256": legacy_defaults.parent_sha256,
            "null_register_sha256": legacy_defaults.null_register_sha256,
            "acs_spine": legacy_defaults.acs_spine,
            "values": dict(legacy_defaults.values),
            "scope": "crosswalk.acs_source_scope",
            "basis": "legacy_default_preserved",
            "source_observed": False,
            "artifact_verification": "caller_responsibility",
        },
        "non_spm_inputs_modified": False,
        "country_calculation_performed": False,
        "release_accepted": False,
    }
    return AcsSpmRegroupResult(
        output.loc[:, spm_units.columns],
        crosswalk,
        ledger_frame,
        pd.DataFrame(field_provenance, columns=["new_spm_unit_id", "field", "basis"]),
        pd.DataFrame(exceptions, columns=["old_spm_unit_id", "field", "reason"]),
        metadata,
    )
