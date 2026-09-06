"""Declared household-to-SPM housing receipt interpretation and quantity separation.

This consumer reads only the status, validity, quality, zero-origin, routing and
period evidence the frozen ``asec_housing_status`` artifact already carries, plus
the occupied-housing-unit evidence the frozen ``asec_housing_universe`` artifact
already carries. It derives no new source semantics, opens no source file, and
never rewrites either artifact.

Three bindings hold before any SPM unit can be assigned:

* the interpretation carries the artifact's own ``household_id`` and
  ``income_year``, and a caller whose household axis differs -- reordered or
  simply different -- is refused rather than positionally aligned;
* the attached-frame reader takes a validated ``HousingStatusAttachedAsec`` and
  derives its authentication, period kind, content digest and household axis from
  that attachment, so no caller string can label synthetic columns verified; and
* occupied-HU evidence is a required input bound to the same households, the same
  income-year cohort and the same authentication, and a household outside a
  housing unit, of an unestablished kind, or whose kind disagrees with the
  status's own group-quarters column keeps its raw observation but is withheld
  from assignment.

Five quantities stay separate here and are never fused:

1. the interview-time receipt status observed on the ASEC household,
2. the counterfactual full contract charge an unassisted comparable unit would
   command (not available in this slice),
3. the reported contract rent an occupant states they pay,
4. the modeled HUD housing assistance payment the engine computes, and
5. the capped SPM housing-subsidy valuation.

The consumer refuses to emit a complete engine boolean while any SPM unit's
assignment is unjustified: those units are named and counted, never silently
dropped and never coerced to observed ``False``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .asec_housing_status import (
    DERIVED_COLUMNS,
    AuthenticatedHousingStatus,
    SyntheticHousingStatus,
    _HousingStatus,
)
from .asec_housing_status_source import ATTACHED_COLUMNS, HousingStatusAttachedAsec
from .asec_housing_universe import (
    DERIVED_COLUMNS as UNIVERSE_DERIVED_COLUMNS,
)
from .asec_housing_universe import (
    AuthenticatedHousingUniverse,
    SyntheticHousingUniverse,
    _HousingUniverse,
)
from .asec_housing_universe_source import (
    ATTACHED_COLUMNS as UNIVERSE_ATTACHED_COLUMNS,
)
from .asec_housing_universe_source import (
    HousingUniverseAttachedAsec,
)

CONSUMER_VERSION = 2
CONSUMER_KIND = "microcosm.asec_housing_receipt_interpretation.v2"

#: Carried status code name -> this consumer's view name. The consumer renames;
#: it never renumbers, so the carried codebook must present exactly this map.
STATUS_VIEW_NAMES = (
    ("unknown", "unknown"),
    ("receipt", "observed_receipt"),
    ("nonreceipt", "observed_nonreceipt"),
    ("not_in_universe", "not_in_universe"),
    ("conflict", "conflict"),
)
RECEIPT_VIEWS = tuple(view for _, view in STATUS_VIEW_NAMES)

#: Carried ``asec_housing_status`` group-quarters states, in code order.
GROUP_QUARTERS_STATES = ("not_gq", "gq", "unavailable")

#: Carried ``asec_housing_universe`` states, in code order.
OCCUPIED_HU_STATES = (
    "not_established",
    "occupied_housing_unit",
    "outside_housing_unit",
)
HOUSEHOLD_KINDS = (
    "source_zero_or_not_established",
    "primary_household",
    "group_quarters",
)

#: The household kind each carried group-quarters state implies. Both columns
#: are HRHTYPE-derived by their own frozen classifiers, so a disagreement is a
#: conflict between two source artifacts, never something to average away.
_KIND_FOR_GROUP_QUARTERS = tuple(
    HOUSEHOLD_KINDS.index(kind)
    for kind in (
        "primary_household",
        "group_quarters",
        "source_zero_or_not_established",
    )
)

#: SPM-grain assignment outcomes. Only the ``assigned_`` members may reach an
#: engine input; every ``withheld_`` member is preserved with its own count.
ASSIGNMENTS = (
    "assigned_receipt_sole_spm_unit",
    "assigned_nonreceipt_sole_spm_unit",
    "assigned_nonreceipt_household_negative",
    "withheld_scope_ambiguous_receipt",
    "withheld_scope_ambiguous_nonreceipt",
    "withheld_unknown",
    "withheld_conflict",
    "withheld_not_in_universe",
    "withheld_outside_housing_unit",
    "withheld_housing_unit_not_established",
    "withheld_housing_unit_conflict",
)
ASSIGNED = tuple(name for name in ASSIGNMENTS if name.startswith("assigned_"))

#: ``receives_housing_assistance`` projection codes. ``withheld`` is a third
#: state on purpose: an int8 code cannot be mistaken for a Boolean column.
RECEIPT_INPUT_CODES = {"false": 0, "true": 1, "withheld": 2}

#: The only supported carried period kind. Interview status is not annual
#: income-year participation, and this module derives no bridge to one.
SUPPORTED_PERIOD_KIND = "interview_status"

#: The only supported carried occupied-HU period kind.
SUPPORTED_UNIVERSE_PERIOD_KIND = "interview_household_universe"

MULTI_UNIT_NEGATIVE_POLICIES = ("refuse", "distribute")

#: Portable in-repository requirement for the still-unbound counterfactual
#: charge. It is a required future measurement, not a completed one, and this
#: module implements none of it.
CHARGE_REQUIREMENTS_DOCUMENT = "docs/us-housing-counterfactual-charge-requirements.md"
CHARGE_REQUIREMENTS_DEPENDENCY = "microcosm.build.sipp_december_source"

TAKE_UP_NOTE = (
    "policyengine-us takes_up_housing_assistance_if_eligible defaults to True. "
    "Interview-time status establishes no take-up; leaving a projected unit at "
    "that default is a modelling choice this consumer does not make."
)

#: The five separated quantities. ``engine_binding`` names the actual installed
#: policyengine-us variable, and ``is_engine_input`` records whether Microcosm
#: may ever persist it as a dataset input.
HOUSING_QUANTITIES: dict[str, dict[str, Any]] = {
    "observed_receipt_status": {
        "grain": "asec_household_interview",
        "period_kind": SUPPORTED_PERIOD_KIND,
        "units": "categorical",
        "source": "ASEC HPUBLIC/HLORENT with allocation flags",
        "engine_binding": "receives_housing_assistance",
        "is_engine_input": True,
        "defines_receipt": True,
        "additive_to_tenant_rent": False,
    },
    "counterfactual_full_contract_charge": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "not established in this slice",
        "engine_binding": "pre_subsidy_rent",
        "is_engine_input": True,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "available": False,
        "unavailable_reason": (
            "No year-correct unassisted comparable-charge source is bound. A SIPP "
            "December projection with an explicit rent/mortgage, utility, weight and "
            "2022-to-2024 price treatment is required first."
        ),
    },
    "reported_contract_rent": {
        "grain": "household",
        "period_kind": "monthly_reported",
        "units": "usd",
        "source": "ACS RNTP/GRNTP; ASEC carries no tenant rent-dollar question",
        "engine_binding": None,
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "Reported contract rent excludes housing-authority subsidies, so for an "
            "assisted household it is a post-assistance amount, not the full charge."
        ),
    },
    "modeled_hap": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "policyengine-us formula output",
        "engine_binding": "hud_hap",
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "A model output cannot define its own pre_subsidy_rent input. HAP may "
            "also exceed rent to owner, reimbursing tenant-paid utilities the "
            "installed engine never pays out."
        ),
    },
    "capped_spm_valuation": {
        "grain": "spm_unit",
        "period_kind": "annual",
        "units": "usd",
        "source": "ASEC SPM_CAPHOUSESUB / spm_unit_capped_housing_subsidy",
        "engine_binding": "spm_unit_capped_housing_subsidy",
        "is_engine_input": False,
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": (
            "A censored SPM resource valuation. Zero is compatible with positive "
            "assistance, so it may corroborate but never define receipt."
        ),
    },
}

#: Named refusals for the fusions the reported-rent bridge assessment rejected.
QUANTITY_FUSIONS_REFUSED = {
    ("reported_contract_rent", "capped_spm_valuation"): (
        "CAPPED_VALUATION_NOT_ADDITIVE_TO_RENT"
    ),
    ("reported_contract_rent", "modeled_hap"): (
        "MODEL_OUTPUT_CANNOT_DEFINE_ITS_OWN_INPUT"
    ),
    ("observed_receipt_status", "capped_spm_valuation"): (
        "CAPPED_VALUATION_CANNOT_DEFINE_RECEIPT"
    ),
    ("counterfactual_full_contract_charge", "modeled_hap"): (
        "MODEL_OUTPUT_CANNOT_DEFINE_ITS_OWN_INPUT"
    ),
}


class HousingReceiptRefusalError(ValueError):
    """Value-free refusal reason for unsupported consumer evidence or fusion."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HousingReceiptRefusalError(reason)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_identity_sha256(header: dict[str, Any]) -> str | None:
    """Retain the authenticated source identity without exposing its contents."""
    if header["source_authentication"] == "synthetic_unverified":
        return None
    identity = header["source"].get("identity")
    _require(type(identity) is str and bool(identity), "SOURCE_IDENTITY")
    return _sha(identity.encode())


def _by_code(entry: tuple[str, int]) -> int:
    """Sort key for one carried ``name -> code`` codebook entry.

    Annotated so the module keeps no unannotated (and therefore opaque) binding
    under a subscript; a lambda parameter cannot carry an annotation.
    """
    return entry[1]


def _sealed(values: np.ndarray) -> np.ndarray:
    """Return an independent, non-writeable copy of a carried array."""
    owned = np.array(values, copy=True)
    owned.flags.writeable = False
    return owned


def _counts(codes: np.ndarray, names: tuple[str, ...]) -> dict[str, int]:
    tally = np.bincount(codes.astype("int64"), minlength=len(names))
    return {name: int(tally[index]) for index, name in enumerate(names)}


def _ordered_codebook_names(codebook: Any, reason: str) -> tuple[str, ...]:
    """Carried ordinal-code names in code order; a gapped map refuses."""
    _require(type(codebook) is dict and len(codebook) > 0, reason)
    entries = sorted(codebook.items(), key=_by_code)
    _require(
        all(type(name) is str and type(code) is int for name, code in entries), reason
    )
    _require(tuple(code for _, code in entries) == tuple(range(len(entries))), reason)
    return tuple(name for name, _ in entries)


def _bit_counts(values: np.ndarray, codebook: Any, reason: str) -> dict[str, int]:
    """Count each carried bit. The bit map is the source codebook's, not ours."""
    _require(type(codebook) is dict and len(codebook) > 0, reason)
    counts: dict[str, int] = {}
    for name, bit in sorted(codebook.items(), key=_by_code):
        _require(type(name) is str and type(bit) is int and bit > 0, reason)
        counts[name] = int(((values.astype("int64") & bit) > 0).sum())
    return counts


def _bit_domain(values: np.ndarray, codebook: Any, reason: str) -> None:
    """Refuse any set bit the carried codebook does not name."""
    _require(type(codebook) is dict and len(codebook) > 0, reason)
    allowed = 0
    for name, bit in sorted(codebook.items(), key=_by_code):
        _require(type(name) is str and type(bit) is int and bit > 0, reason)
        allowed |= bit
    _require(bool(((values.astype("int64") & (~allowed)) == 0).all()), reason)


def _int64_axis(values: Any, reason: str) -> np.ndarray:
    _require(type(values) is np.ndarray, reason)
    _require(values.dtype == np.dtype("int64") and values.ndim == 1, reason)
    return values


def receipt_columns_from_status(status: _HousingStatus) -> dict[str, Any]:
    """Project the carried derived columns, household axis and provenance.

    ``content_sha256`` validates the artifact, so a tampered buffer cannot reach
    the consumer and the validation runs exactly once. Synthetic and
    authenticated statuses both work; the projection carries
    ``source_authentication`` so a synthetic-derived interpretation stays
    visibly unauthenticated downstream.
    """
    _require(
        type(status) in (SyntheticHousingStatus, AuthenticatedHousingStatus),
        "STATUS_TYPE",
    )
    digest = status.content_sha256
    header = status.header_data
    return {
        "columns": {name: status.array(name) for name in DERIVED_COLUMNS},
        "household_id": status.array("household_id"),
        "income_year": status.array("income_year"),
        "source_period_kind": header["source_period_kind"],
        "source_authentication": header["source_authentication"],
        "source_identity_sha256": _source_identity_sha256(header),
        "zero_origin_policy": header["zero_origin_policy"],
        "non_housing_unit_scope": header["non_housing_unit_scope"],
        "codebook": header["codebook"],
        "status_content_sha256": digest,
        "attachment_receipt_sha256": None,
    }


def receipt_columns_from_attached_household(
    attached: HousingStatusAttachedAsec,
) -> dict[str, Any]:
    """Project a validated status attachment, deriving every label from it.

    There is no caller-supplied authentication, period kind or digest: the
    attachment validator has already bound the attached ``uint8`` columns to the
    status bytes and the frame's household axis to the status household axis, so
    those are the only honest sources for this projection's provenance.
    """
    _require(type(attached) is HousingStatusAttachedAsec, "ATTACHED_ASEC")
    attached.validate()
    status = attached.status
    digest = status.content_sha256
    header = status.header_data
    household = attached.frame.table("household")
    _require(set(ATTACHED_COLUMNS) <= set(household.columns), "ATTACHED_COLUMNS")
    columns: dict[str, np.ndarray] = {}
    for name, alias in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True):
        values = household[alias]
        # Never coerce: the attachment validator guarantees uint8 already, and a
        # silent uint8 cast of a wider column would wrap an out-of-domain value.
        _require(values.dtype == np.dtype("uint8"), "ATTACHED_DTYPE")
        columns[name] = values.to_numpy(copy=True)
    household_id = household["household_id"]
    _require(household_id.dtype == np.dtype("int64"), "ATTACHED_HOUSEHOLD_DTYPE")
    _require(
        np.array_equal(household_id.to_numpy(), status.array("household_id")),
        "ATTACHED_HOUSEHOLD_IDENTITY",
    )
    return {
        "columns": columns,
        "household_id": household_id.to_numpy(copy=True),
        "income_year": status.array("income_year"),
        "source_period_kind": header["source_period_kind"],
        "source_authentication": header["source_authentication"],
        "source_identity_sha256": _source_identity_sha256(header),
        "zero_origin_policy": header["zero_origin_policy"],
        "non_housing_unit_scope": header["non_housing_unit_scope"],
        "codebook": header["codebook"],
        "status_content_sha256": digest,
        "attachment_receipt_sha256": _sha(_json(attached.receipt)),
    }


@dataclass(frozen=True)
class HouseholdReceiptInterpretation:
    """Household-grain receipt view with every carried column preserved.

    The stored arrays are non-writeable, so a diagnostic or a caller cannot
    mutate an interpretation out of agreement with its own receipt.
    """

    household_id: np.ndarray
    income_year: np.ndarray
    columns: dict[str, np.ndarray]
    codebook: dict[str, Any]
    source_period_kind: str
    source_authentication: str
    source_identity_sha256: str | None
    non_housing_unit_scope: str
    zero_origin_policy: Any
    status_content_sha256: str | None
    attachment_receipt_sha256: str | None

    def column(self, name: str) -> np.ndarray:
        """Return one carried derived column by its own source name."""
        if name not in DERIVED_COLUMNS:
            raise KeyError(name)
        return self.columns[name]

    @property
    def receipt_view(self) -> np.ndarray:
        return self.column("status")

    @property
    def receipt_valid(self) -> np.ndarray:
        return self.column("receipt_valid")

    @property
    def route(self) -> np.ndarray:
        return self.column("route")

    @property
    def group_quarters(self) -> np.ndarray:
        return self.column("group_quarters")

    @property
    def conflicts(self) -> np.ndarray:
        return self.column("conflicts")

    @property
    def unknown_reasons(self) -> np.ndarray:
        return self.column("unknown_reasons")

    @property
    def public_quality(self) -> np.ndarray:
        return self.column("public_quality")

    @property
    def lower_quality(self) -> np.ndarray:
        return self.column("lower_quality")

    @property
    def zero_origin(self) -> dict[str, np.ndarray]:
        return {
            name: self.columns[name]
            for name in DERIVED_COLUMNS
            if name.endswith("_zero_origin")
        }

    @property
    def rows(self) -> int:
        return int(self.receipt_view.shape[0])

    def view_counts(self) -> dict[str, int]:
        return _counts(self.receipt_view, RECEIPT_VIEWS)


def interpret_household_receipt(
    projection: dict[str, Any],
) -> HouseholdReceiptInterpretation:
    """Rename the carried household status into an explicit receipt view.

    Every carried invariant is rechecked rather than trusted: the code domains,
    the ``receipt_valid`` mask's agreement with the status, the exclusivity of
    conflict, the household axis's uniqueness, and the carried codebook's
    agreement with the view names this consumer renames into. No unknown,
    conflict or not-in-universe household becomes observed nonreceipt, and no
    zero-origin, routing or group-quarters evidence is discarded.
    """
    _require(type(projection) is dict, "PROJECTION_TYPE")
    columns = projection.get("columns")
    _require(type(columns) is dict, "PROJECTION_COLUMNS")
    _require(set(columns) == set(DERIVED_COLUMNS), "PROJECTION_COLUMNS")
    period_kind = projection.get("source_period_kind")
    _require(period_kind == SUPPORTED_PERIOD_KIND, "UNSUPPORTED_PERIOD_KIND")
    authentication = projection.get("source_authentication")
    _require(type(authentication) is str and authentication, "SOURCE_AUTHENTICATION")
    source_identity = projection.get("source_identity_sha256")
    _require(
        source_identity is None
        if authentication == "synthetic_unverified"
        else type(source_identity) is str
        and len(source_identity) == 64
        and set(source_identity) <= set("0123456789abcdef"),
        "SOURCE_IDENTITY",
    )
    scope_declaration = projection.get("non_housing_unit_scope")
    _require(type(scope_declaration) is str and scope_declaration, "SCOPE_DECLARATION")
    codebook = projection.get("codebook")
    _require(type(codebook) is dict, "CODEBOOK_MISSING")
    _require(
        _ordered_codebook_names(codebook.get("status"), "CODEBOOK_STATUS")
        == tuple(name for name, _ in STATUS_VIEW_NAMES),
        "CODEBOOK_STATUS",
    )
    _require(
        _ordered_codebook_names(codebook.get("group_quarters"), "CODEBOOK_QUARTERS")
        == GROUP_QUARTERS_STATES,
        "CODEBOOK_QUARTERS",
    )
    route_names = _ordered_codebook_names(codebook.get("route"), "CODEBOOK_ROUTE")
    quality_names = _ordered_codebook_names(codebook.get("quality"), "CODEBOOK_QUALITY")
    rows = None
    for name in DERIVED_COLUMNS:
        value = columns[name]
        _require(type(value) is np.ndarray, "COLUMN_TYPE")
        _require(value.dtype == np.dtype("uint8"), "COLUMN_DTYPE")
        _require(value.ndim == 1, "COLUMN_SHAPE")
        rows = value.shape[0] if rows is None else rows
        _require(value.shape[0] == rows, "COLUMN_LENGTH")
    _require(rows is not None and rows > 0, "ROW_COUNT")

    household_id = _int64_axis(projection.get("household_id"), "HOUSEHOLD_AXIS")
    income_year = _int64_axis(projection.get("income_year"), "COHORT_AXIS")
    _require(household_id.shape[0] == rows, "HOUSEHOLD_AXIS")
    _require(income_year.shape[0] == rows, "COHORT_AXIS")
    _require(
        len(np.unique(household_id)) == household_id.shape[0], "DUPLICATE_HOUSEHOLD_ID"
    )

    status = columns["status"]
    _require(bool((status < len(RECEIPT_VIEWS)).all()), "STATUS_DOMAIN")
    _require(bool((columns["route"] < len(route_names)).all()), "ROUTE_DOMAIN")
    _require(
        bool((columns["group_quarters"] < len(GROUP_QUARTERS_STATES)).all()),
        "QUARTERS_DOMAIN",
    )
    _require(
        bool((columns["public_quality"] < len(quality_names)).all())
        and bool((columns["lower_quality"] < len(quality_names)).all()),
        "QUALITY_DOMAIN",
    )
    for name in ("HPUBLIC_valid", "HLORENT_valid"):
        _require(bool((columns[name] <= 1).all()), "ANSWER_VALIDITY_DOMAIN")
    for name in DERIVED_COLUMNS:
        if name.endswith("_zero_origin"):
            _require(bool((columns[name] <= 1).all()), "ZERO_ORIGIN_DOMAIN")
    receipt_valid = columns["receipt_valid"]
    _require(bool((receipt_valid <= 1).all()), "VALIDITY_DOMAIN")
    _require(
        bool((receipt_valid == np.isin(status, [1, 2]).astype("uint8")).all()),
        "VALIDITY_DISAGREES_WITH_STATUS",
    )
    conflicts = columns["conflicts"]
    _require(bool(((conflicts > 0) == (status == 4)).all()), "CONFLICT_DISAGREES")
    _bit_domain(conflicts, codebook.get("conflict_bits"), "CONFLICT_BIT_DOMAIN")
    unknown_reasons = columns["unknown_reasons"]
    _bit_domain(
        unknown_reasons,
        codebook.get("unknown_reason_bits"),
        "UNKNOWN_REASON_BIT_DOMAIN",
    )
    _require(
        bool((unknown_reasons[receipt_valid == 1] == 0).all()),
        "UNKNOWN_REASON_ON_VALID_RECEIPT",
    )
    return HouseholdReceiptInterpretation(
        household_id=_sealed(household_id),
        income_year=_sealed(income_year),
        columns={name: _sealed(columns[name]) for name in DERIVED_COLUMNS},
        codebook=codebook,
        source_period_kind=period_kind,
        source_authentication=authentication,
        source_identity_sha256=source_identity,
        non_housing_unit_scope=scope_declaration,
        zero_origin_policy=projection.get("zero_origin_policy"),
        status_content_sha256=projection.get("status_content_sha256"),
        attachment_receipt_sha256=projection.get("attachment_receipt_sha256"),
    )


@dataclass(frozen=True)
class HousingUnitEvidence:
    """Carried occupied-HU evidence with its own household axis and provenance."""

    household_id: np.ndarray
    income_year: np.ndarray
    occupied_hu: np.ndarray
    household_kind: np.ndarray
    codebook: dict[str, Any]
    source_authentication: str
    source_identity_sha256: str | None
    universe_content_sha256: str | None
    attachment_receipt_sha256: str | None
    parent_receipt_sha256: str | None

    @property
    def rows(self) -> int:
        return int(self.occupied_hu.shape[0])

    def occupied_hu_counts(self) -> dict[str, int]:
        return _counts(self.occupied_hu, OCCUPIED_HU_STATES)

    def household_kind_counts(self) -> dict[str, int]:
        return _counts(self.household_kind, HOUSEHOLD_KINDS)


def _housing_unit_evidence(
    *,
    household_id: np.ndarray,
    income_year: np.ndarray,
    occupied_hu: np.ndarray,
    household_kind: np.ndarray,
    header: dict[str, Any],
    universe_content_sha256: str,
    attachment_receipt_sha256: str | None,
    parent_receipt_sha256: str | None,
) -> HousingUnitEvidence:
    """Recheck the carried occupied-HU domains without rederiving any of them."""
    _require(
        header["source_period_kind"] == SUPPORTED_UNIVERSE_PERIOD_KIND,
        "UNSUPPORTED_UNIVERSE_PERIOD_KIND",
    )
    authentication = header["source_authentication"]
    _require(type(authentication) is str and authentication, "UNIVERSE_AUTHENTICATION")
    codebook = header["codebook"]
    _require(type(codebook) is dict, "UNIVERSE_CODEBOOK")
    _require(
        _ordered_codebook_names(codebook.get("occupied_hu"), "CODEBOOK_OCCUPIED_HU")
        == OCCUPIED_HU_STATES,
        "CODEBOOK_OCCUPIED_HU",
    )
    _require(
        _ordered_codebook_names(codebook.get("household_kind"), "CODEBOOK_KIND")
        == HOUSEHOLD_KINDS,
        "CODEBOOK_KIND",
    )
    rows = household_id.shape[0]
    _require(rows > 0, "UNIVERSE_ROW_COUNT")
    for values in (occupied_hu, household_kind):
        _require(type(values) is np.ndarray, "UNIVERSE_COLUMN_TYPE")
        _require(values.dtype == np.dtype("uint8"), "UNIVERSE_COLUMN_DTYPE")
        _require(values.ndim == 1 and values.shape[0] == rows, "UNIVERSE_COLUMN_LENGTH")
    _require(income_year.shape[0] == rows, "UNIVERSE_COHORT_AXIS")
    _require(bool((occupied_hu < len(OCCUPIED_HU_STATES)).all()), "OCCUPIED_HU_DOMAIN")
    _require(
        bool((household_kind < len(HOUSEHOLD_KINDS)).all()), "HOUSEHOLD_KIND_DOMAIN"
    )
    _require(len(np.unique(household_id)) == rows, "DUPLICATE_UNIVERSE_HOUSEHOLD_ID")
    return HousingUnitEvidence(
        household_id=_sealed(household_id),
        income_year=_sealed(income_year),
        occupied_hu=_sealed(occupied_hu),
        household_kind=_sealed(household_kind),
        codebook=codebook,
        source_authentication=authentication,
        source_identity_sha256=_source_identity_sha256(header),
        universe_content_sha256=universe_content_sha256,
        attachment_receipt_sha256=attachment_receipt_sha256,
        parent_receipt_sha256=parent_receipt_sha256,
    )


def housing_unit_evidence_from_universe(
    universe: _HousingUniverse,
) -> HousingUnitEvidence:
    """Read occupied-HU evidence out of a validated universe artifact."""
    _require(
        type(universe) in (SyntheticHousingUniverse, AuthenticatedHousingUniverse),
        "UNIVERSE_TYPE",
    )
    digest = universe.content_sha256
    return _housing_unit_evidence(
        household_id=_int64_axis(
            universe.array("household_id"), "UNIVERSE_HOUSEHOLD_AXIS"
        ),
        income_year=_int64_axis(universe.array("income_year"), "UNIVERSE_COHORT_AXIS"),
        occupied_hu=universe.array("occupied_hu"),
        household_kind=universe.array("household_kind"),
        header=universe.header_data,
        universe_content_sha256=digest,
        attachment_receipt_sha256=None,
        parent_receipt_sha256=None,
    )


def housing_unit_evidence_from_attached_universe(
    attached: HousingUniverseAttachedAsec,
) -> HousingUnitEvidence:
    """Read occupied-HU evidence off a validated universe attachment.

    As with the status attachment, nothing here is caller-asserted: the
    attachment validator binds the attached ``uint8`` columns to the universe
    bytes and the frame's household axis to the universe household axis.
    """
    _require(type(attached) is HousingUniverseAttachedAsec, "ATTACHED_UNIVERSE_ASEC")
    attached.validate()
    universe = attached.universe
    digest = universe.content_sha256
    household = attached.frame.table("household")
    _require(
        set(UNIVERSE_ATTACHED_COLUMNS) <= set(household.columns),
        "UNIVERSE_ATTACHED_COLUMNS",
    )
    carried: dict[str, np.ndarray] = {}
    for name, alias in zip(
        UNIVERSE_DERIVED_COLUMNS, UNIVERSE_ATTACHED_COLUMNS, strict=True
    ):
        if name not in ("occupied_hu", "household_kind"):
            continue
        values = household[alias]
        # Never coerce: the attachment validator guarantees uint8 already.
        _require(values.dtype == np.dtype("uint8"), "UNIVERSE_ATTACHED_DTYPE")
        carried[name] = values.to_numpy(copy=True)
    _require(
        set(carried) == {"occupied_hu", "household_kind"}, "UNIVERSE_COLUMN_ROSTER"
    )
    household_id = household["household_id"]
    _require(
        household_id.dtype == np.dtype("int64")
        and np.array_equal(household_id.to_numpy(), universe.array("household_id")),
        "UNIVERSE_ATTACHED_HOUSEHOLD_IDENTITY",
    )
    return _housing_unit_evidence(
        household_id=household_id.to_numpy(copy=True),
        income_year=_int64_axis(universe.array("income_year"), "UNIVERSE_COHORT_AXIS"),
        occupied_hu=carried["occupied_hu"],
        household_kind=carried["household_kind"],
        header=universe.header_data,
        universe_content_sha256=digest,
        attachment_receipt_sha256=_sha(_json(attached.receipt)),
        parent_receipt_sha256=attached.receipt["parent_receipt_sha256"],
    )


@dataclass(frozen=True)
class SpmReceiptScope:
    """SPM-grain assignment of a household-grain observation, ambiguity kept."""

    spm_unit_id: np.ndarray
    household_row: np.ndarray
    receipt_view: np.ndarray
    spm_units_in_household: np.ndarray
    occupied_hu: np.ndarray
    household_kind: np.ndarray
    housing_unit_valid: np.ndarray
    assignment: np.ndarray
    interpretation: HouseholdReceiptInterpretation
    housing_unit: HousingUnitEvidence

    @property
    def rows(self) -> int:
        return int(self.spm_unit_id.shape[0])

    def assignment_counts(self) -> dict[str, int]:
        return _counts(self.assignment, ASSIGNMENTS)


def _membership(
    *,
    household_id: np.ndarray,
    spm_unit_id: np.ndarray,
    person_household_id: np.ndarray,
    person_spm_unit_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each SPM unit to its single household row and count units per household."""
    for name, value in (
        ("household_id", household_id),
        ("spm_unit_id", spm_unit_id),
        ("person_household_id", person_household_id),
        ("person_spm_unit_id", person_spm_unit_id),
    ):
        _require(type(value) is np.ndarray and value.ndim == 1, "MEMBERSHIP_SHAPE")
        _require(value.dtype.kind == "i", "MEMBERSHIP_DTYPE")
        _require(value.shape[0] > 0, f"MEMBERSHIP_EMPTY:{name}")
    _require(
        len(np.unique(household_id)) == household_id.shape[0], "DUPLICATE_HOUSEHOLD"
    )
    _require(len(np.unique(spm_unit_id)) == spm_unit_id.shape[0], "DUPLICATE_SPM_UNIT")
    _require(
        person_household_id.shape[0] == person_spm_unit_id.shape[0], "PERSON_LENGTH"
    )
    _require(
        bool(np.isin(person_household_id, household_id).all()), "UNKNOWN_HOUSEHOLD"
    )
    _require(bool(np.isin(person_spm_unit_id, spm_unit_id).all()), "UNKNOWN_SPM_UNIT")
    _require(bool(np.isin(spm_unit_id, person_spm_unit_id).all()), "EMPTY_SPM_UNIT")
    _require(bool(np.isin(household_id, person_household_id).all()), "EMPTY_HOUSEHOLD")

    order = np.argsort(household_id, kind="stable")
    household_row_of_person = order[
        np.searchsorted(household_id[order], person_household_id)
    ]
    spm_order = np.argsort(spm_unit_id, kind="stable")
    spm_row_of_person = spm_order[
        np.searchsorted(spm_unit_id[spm_order], person_spm_unit_id)
    ]
    household_row = np.full(spm_unit_id.shape[0], -1, dtype="int64")
    household_row[spm_row_of_person] = household_row_of_person
    spans = np.zeros(spm_unit_id.shape[0], dtype=bool)
    np.logical_or.at(
        spans,
        spm_row_of_person,
        household_row[spm_row_of_person] != household_row_of_person,
    )
    _require(not bool(spans.any()), "SPM_UNIT_SPANS_HOUSEHOLDS")
    units_per_household = np.bincount(household_row, minlength=household_id.shape[0])
    return household_row, units_per_household


def _bind_housing_unit(
    interpretation: HouseholdReceiptInterpretation, housing_unit: HousingUnitEvidence
) -> np.ndarray:
    """Bind occupied-HU evidence to the same households, cohort and authority.

    Returns the household-grain mask of kinds that disagree with the status's own
    group-quarters column. Both columns are HRHTYPE-derived by their own frozen
    classifier, so a disagreement is a real conflict between two artifacts.
    """
    _require(type(housing_unit) is HousingUnitEvidence, "HOUSING_UNIT_TYPE")
    _require(housing_unit.rows == interpretation.rows, "HOUSING_UNIT_ALIGNMENT")
    _require(
        np.array_equal(housing_unit.household_id, interpretation.household_id),
        "HOUSING_UNIT_HOUSEHOLD_IDENTITY",
    )
    _require(
        np.array_equal(housing_unit.income_year, interpretation.income_year),
        "HOUSING_UNIT_COHORT",
    )
    _require(
        housing_unit.source_authentication == interpretation.source_authentication,
        "HOUSING_UNIT_AUTHENTICATION",
    )
    _require(
        housing_unit.source_identity_sha256 == interpretation.source_identity_sha256,
        "HOUSING_UNIT_SOURCE_IDENTITY",
    )
    if (
        interpretation.attachment_receipt_sha256 is not None
        and housing_unit.attachment_receipt_sha256 is not None
    ):
        _require(
            housing_unit.parent_receipt_sha256
            == interpretation.attachment_receipt_sha256,
            "HOUSING_UNIT_ATTACHMENT_PARENT",
        )
    implied = np.asarray(_KIND_FOR_GROUP_QUARTERS, dtype="uint8")[
        interpretation.group_quarters
    ]
    return housing_unit.household_kind != implied


def resolve_spm_receipt_scope(
    interpretation: HouseholdReceiptInterpretation,
    *,
    housing_unit: HousingUnitEvidence,
    household_id: np.ndarray,
    spm_unit_id: np.ndarray,
    person_household_id: np.ndarray,
    person_spm_unit_id: np.ndarray,
) -> SpmReceiptScope:
    """Carry the household observation to SPM units without inventing a recipient.

    The caller's household axis must be the artifact's own household axis, in the
    artifact's own order. Nothing here re-derives an identity, so a reordered or
    unrelated axis is refused rather than aligned by position; person and SPM-unit
    order remain free and are resolved by identity.

    ``housing_unit`` has no default. An engine assignment asserts something about
    an occupied housing unit, and the ASEC status alone does not establish one:
    the frozen classifier declares its non-housing-unit scope "not_derived" and
    classifies group-quarters households as ordinary receipt. A household outside
    a housing unit, of an unestablished kind, or whose kind disagrees with the
    carried group-quarters column keeps its observed view and takes its own
    withheld class. Missing evidence is never a default ``True`` or ``False``.

    A household-level positive answer names the housing unit, not one of several
    SPM units sharing it, so a multi-SPM-unit household's receipt stays
    ``withheld_scope_ambiguous_receipt``. Replicating it would hand every SPM unit
    its own voucher budget: the installed engine gives each unit the household's
    full payment standard, so the replication is not conservative.

    A household-level negative answer is a statement about the housing unit
    itself and is separated as ``withheld_scope_ambiguous_nonreceipt`` so the
    caller decides explicitly whether to distribute it.
    """
    _require(
        type(interpretation) is HouseholdReceiptInterpretation, "INTERPRETATION_TYPE"
    )
    _int64_axis(household_id, "MEMBERSHIP_DTYPE")
    _require(household_id.shape[0] == interpretation.rows, "HOUSEHOLD_ALIGNMENT")
    _require(
        np.array_equal(household_id, interpretation.household_id),
        "CALLER_HOUSEHOLD_IDENTITY",
    )
    kind_conflict = _bind_housing_unit(interpretation, housing_unit)
    household_row, units_per_household = _membership(
        household_id=household_id,
        spm_unit_id=spm_unit_id,
        person_household_id=person_household_id,
        person_spm_unit_id=person_spm_unit_id,
    )
    view = interpretation.receipt_view[household_row]
    occupied = housing_unit.occupied_hu[household_row]
    kind = housing_unit.household_kind[household_row]
    conflicting = kind_conflict[household_row]
    units = units_per_household[household_row]
    sole = units == 1
    receipt = RECEIPT_VIEWS.index("observed_receipt")
    nonreceipt = RECEIPT_VIEWS.index("observed_nonreceipt")
    observed = (view == receipt) | (view == nonreceipt)
    inside = occupied == OCCUPIED_HU_STATES.index("occupied_housing_unit")
    primary = kind == HOUSEHOLD_KINDS.index("primary_household")
    bound = inside & primary & ~conflicting
    assignment = np.full(view.shape[0], ASSIGNMENTS.index("withheld_unknown"), "uint8")
    for mask, name in (
        (view == RECEIPT_VIEWS.index("conflict"), "withheld_conflict"),
        (view == RECEIPT_VIEWS.index("not_in_universe"), "withheld_not_in_universe"),
        (
            observed & (occupied == OCCUPIED_HU_STATES.index("outside_housing_unit")),
            "withheld_outside_housing_unit",
        ),
        (
            observed & (occupied == OCCUPIED_HU_STATES.index("not_established")),
            "withheld_housing_unit_not_established",
        ),
        (
            observed & ((inside & ~primary) | conflicting),
            "withheld_housing_unit_conflict",
        ),
        ((view == receipt) & bound & sole, "assigned_receipt_sole_spm_unit"),
        ((view == receipt) & bound & ~sole, "withheld_scope_ambiguous_receipt"),
        ((view == nonreceipt) & bound & sole, "assigned_nonreceipt_sole_spm_unit"),
        (
            (view == nonreceipt) & bound & ~sole,
            "withheld_scope_ambiguous_nonreceipt",
        ),
    ):
        assignment[mask] = ASSIGNMENTS.index(name)
    return SpmReceiptScope(
        spm_unit_id=_sealed(spm_unit_id),
        household_row=_sealed(household_row),
        receipt_view=_sealed(view),
        spm_units_in_household=_sealed(units),
        occupied_hu=_sealed(occupied),
        household_kind=_sealed(kind),
        housing_unit_valid=_sealed(bound),
        assignment=_sealed(assignment),
        interpretation=interpretation,
        housing_unit=housing_unit,
    )


@dataclass(frozen=True)
class EngineReceiptProjection:
    """Three-state projection: no withheld SPM unit ever becomes a Boolean."""

    spm_unit_id: np.ndarray
    code: np.ndarray
    assignment: np.ndarray
    multi_unit_negative: str
    _receipt: bytes

    @property
    def receipt(self) -> dict:
        return json.loads(self._receipt)

    @property
    def withheld(self) -> np.ndarray:
        return self.code == RECEIPT_INPUT_CODES["withheld"]

    def code_counts(self) -> dict[str, int]:
        names = tuple(
            name for name, _ in sorted(RECEIPT_INPUT_CODES.items(), key=_by_code)
        )
        return _counts(self.code, names)


def project_engine_receipt_inputs(
    scope: SpmReceiptScope, *, multi_unit_negative: str
) -> EngineReceiptProjection:
    """Emit ``receives_housing_assistance`` only where the assignment is justified.

    ``multi_unit_negative`` has no default: the caller must declare whether a
    multi-SPM-unit household's negative answer distributes to each unit
    (``"distribute"``) or stays withheld (``"refuse"``). Only units with bound
    occupied-HU evidence may distribute, and every other withheld class -- unknown,
    conflict, not-in-universe, ambiguous receipt, and each occupied-HU class --
    stays withheld under both policies.
    """
    _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
    _require(
        multi_unit_negative in MULTI_UNIT_NEGATIVE_POLICIES,
        "MULTI_UNIT_NEGATIVE_POLICY",
    )
    assignment = np.array(scope.assignment, copy=True)
    if multi_unit_negative == "distribute":
        distributable = (
            assignment == ASSIGNMENTS.index("withheld_scope_ambiguous_nonreceipt")
        ) & scope.housing_unit_valid
        assignment[distributable] = ASSIGNMENTS.index(
            "assigned_nonreceipt_household_negative"
        )
    code = np.full(assignment.shape[0], RECEIPT_INPUT_CODES["withheld"], dtype="uint8")
    code[assignment == ASSIGNMENTS.index("assigned_receipt_sole_spm_unit")] = (
        RECEIPT_INPUT_CODES["true"]
    )
    for name in (
        "assigned_nonreceipt_sole_spm_unit",
        "assigned_nonreceipt_household_negative",
    ):
        code[assignment == ASSIGNMENTS.index(name)] = RECEIPT_INPUT_CODES["false"]
    interpretation = scope.interpretation
    housing_unit = scope.housing_unit
    receipt = {
        "schema_version": CONSUMER_VERSION,
        "artifact_kind": CONSUMER_KIND,
        "release_eligible": False,
        "source_authentication": interpretation.source_authentication,
        "source_identity_sha256": interpretation.source_identity_sha256,
        "source_period_kind": interpretation.source_period_kind,
        "status_content_sha256": interpretation.status_content_sha256,
        "status_attachment_receipt_sha256": interpretation.attachment_receipt_sha256,
        "non_housing_unit_scope": interpretation.non_housing_unit_scope,
        "housing_unit_authentication": housing_unit.source_authentication,
        "universe_content_sha256": housing_unit.universe_content_sha256,
        "universe_attachment_receipt_sha256": housing_unit.attachment_receipt_sha256,
        "universe_parent_receipt_sha256": housing_unit.parent_receipt_sha256,
        "annual_income_year_participation": "not_established",
        "multi_unit_negative": multi_unit_negative,
        "engine_input": "receives_housing_assistance",
        "engine_take_up_input": "not_derived",
        "take_up_note": TAKE_UP_NOTE,
        "household_view_counts": interpretation.view_counts(),
        "occupied_hu_counts": housing_unit.occupied_hu_counts(),
        "assignment_counts": _counts(assignment, ASSIGNMENTS),
        "count_grains": {
            "household_view_counts": "household",
            "occupied_hu_counts": "household",
            "assignment_counts": "spm_unit",
        },
        "spm_unit_rows": int(assignment.shape[0]),
        "withheld_rows": int((code == RECEIPT_INPUT_CODES["withheld"]).sum()),
    }
    return EngineReceiptProjection(
        spm_unit_id=_sealed(scope.spm_unit_id),
        code=_sealed(code),
        assignment=_sealed(assignment),
        multi_unit_negative=multi_unit_negative,
        _receipt=_json(receipt),
    )


def engine_receives_housing_assistance(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Return the Boolean engine input, or refuse while any SPM unit is withheld.

    This is the blanket-Boolean refusal: an incomplete observation cannot be
    handed to ``receives_housing_assistance``, because the engine's default
    ``False`` would silently assert measured nonreceipt for withheld units.
    """
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    withheld = int(projection.withheld.sum())
    _require(withheld == 0, f"WITHHELD_SPM_UNITS:{withheld}")
    return projection.code.astype(bool)


def annual_income_year_participation(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Always refuse: an interview-time answer is not income-year participation."""
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    raise HousingReceiptRefusalError("ANNUAL_PARTICIPATION_NOT_OBSERVED_AT_INTERVIEW")


def engine_takes_up_housing_assistance_if_eligible(
    projection: EngineReceiptProjection,
) -> np.ndarray:
    """Always refuse: interview status establishes no take-up behaviour."""
    _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
    raise HousingReceiptRefusalError("TAKE_UP_NOT_DERIVABLE_FROM_INTERVIEW_STATUS")


def corroborate_with_capped_valuation(
    scope: SpmReceiptScope, capped_valuation: np.ndarray
) -> dict[str, Any]:
    """Cross-tabulate the capped SPM valuation against the view, changing nothing.

    A zero cap is compatible with positive assistance and the valuation is
    censored, so disagreement is diagnostic evidence, never a status correction.
    """
    _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
    _require(type(capped_valuation) is np.ndarray, "VALUATION_TYPE")
    _require(capped_valuation.ndim == 1, "VALUATION_SHAPE")
    _require(capped_valuation.shape[0] == scope.rows, "VALUATION_LENGTH")
    _require(capped_valuation.dtype.kind in "fi", "VALUATION_DTYPE")
    _require(bool(np.isfinite(capped_valuation).all()), "VALUATION_FINITE")
    _require(bool((capped_valuation >= 0).all()), "VALUATION_NEGATIVE")
    positive = capped_valuation > 0
    table = {
        view: {
            "positive_valuation": int((positive & (scope.receipt_view == index)).sum()),
            "zero_valuation": int((~positive & (scope.receipt_view == index)).sum()),
        }
        for index, view in enumerate(RECEIPT_VIEWS)
    }
    return {
        "role": "corroboration_only",
        "defines_receipt": False,
        "additive_to_tenant_rent": False,
        "note": HOUSING_QUANTITIES["capped_spm_valuation"]["note"],
        "table": table,
        "receipt_with_zero_valuation": table["observed_receipt"]["zero_valuation"],
        "nonreceipt_with_positive_valuation": table["observed_nonreceipt"][
            "positive_valuation"
        ],
    }


def compose_full_contract_charge(**quantities: np.ndarray) -> np.ndarray:
    """Refuse every fusion the reported-rent bridge assessment rejected.

    No supported combination of the currently bound quantities identifies the
    counterfactual full contract charge, so this always refuses. The refusal
    reason names the specific rejected shortcut where one applies.
    """
    names = set(quantities)
    _require(names <= set(HOUSING_QUANTITIES), "UNKNOWN_QUANTITY")
    for pair, reason in QUANTITY_FUSIONS_REFUSED.items():
        if set(pair) <= names:
            raise HousingReceiptRefusalError(reason)
    raise HousingReceiptRefusalError("FULL_CONTRACT_CHARGE_SOURCE_UNAVAILABLE")


def counterfactual_charge_availability() -> dict[str, Any]:
    """Report the unavailable counterfactual charge and its blocking defects.

    ``requirements`` is a repository-relative document and
    ``requirements_dependency`` names the producer that would have to grow the
    fields. Both describe a required future measurement. This consumer produces
    no charge, and nothing in this module implements any part of that lane.
    """
    quantity = HOUSING_QUANTITIES["counterfactual_full_contract_charge"]
    return {
        "available": quantity["available"],
        "reason": quantity["unavailable_reason"],
        "blocking_defects": (
            "installed FMR table falls forward to FY2025 for a 2024 request",
            "utility allowance is encoded for selected PHAs only and is zero elsewhere",
            "bedrooms are not observed for the assisted universe",
            "pha_payment_standard carries an undated LA branch",
        ),
        "refused_shortcut": "FMR_WITHOUT_YEAR_CORRECT_SOURCE_AND_UTILITY_CONTRACT",
        "requirements": CHARGE_REQUIREMENTS_DOCUMENT,
        "requirements_kind": "required_future_measurement",
        "requirements_dependency": CHARGE_REQUIREMENTS_DEPENDENCY,
        "measured_here": False,
    }


def housing_receipt_diagnostics(
    interpretation: HouseholdReceiptInterpretation,
    scope: SpmReceiptScope | None = None,
    projection: EngineReceiptProjection | None = None,
) -> dict[str, Any]:
    """Raw status, routing, quarters, quality, zero-origin and occupied-HU counts.

    Every code and bit name is read out of the carried source codebook, so a
    codebook change in a frozen artifact moves these labels instead of silently
    desynchronizing them from a copy kept here.
    """
    _require(
        type(interpretation) is HouseholdReceiptInterpretation, "INTERPRETATION_TYPE"
    )
    codebook = interpretation.codebook
    quality_names = _ordered_codebook_names(codebook.get("quality"), "CODEBOOK_QUALITY")
    route_names = _ordered_codebook_names(codebook.get("route"), "CODEBOOK_ROUTE")
    quarters_names = _ordered_codebook_names(
        codebook.get("group_quarters"), "CODEBOOK_QUARTERS"
    )
    payload: dict[str, Any] = {
        "schema_version": CONSUMER_VERSION,
        "source_authentication": interpretation.source_authentication,
        "source_period_kind": interpretation.source_period_kind,
        "non_housing_unit_scope": interpretation.non_housing_unit_scope,
        "zero_origin_policy": interpretation.zero_origin_policy,
        "household_rows": interpretation.rows,
        "household_view_counts": interpretation.view_counts(),
        "receipt_valid_rows": int(interpretation.receipt_valid.sum()),
        "route_counts": _counts(interpretation.route, route_names),
        "group_quarters_counts": _counts(interpretation.group_quarters, quarters_names),
        "observed_answer_valid_counts": {
            name: int(interpretation.column(name).sum())
            for name in ("HPUBLIC_valid", "HLORENT_valid")
        },
        "unknown_reason_counts": _bit_counts(
            interpretation.unknown_reasons,
            codebook.get("unknown_reason_bits"),
            "CODEBOOK_UNKNOWN_REASON_BITS",
        ),
        "conflict_counts": _bit_counts(
            interpretation.conflicts,
            codebook.get("conflict_bits"),
            "CODEBOOK_CONFLICT_BITS",
        ),
        "public_quality_counts": _counts(interpretation.public_quality, quality_names),
        "lower_quality_counts": _counts(interpretation.lower_quality, quality_names),
        "zero_origin_counts": {
            name: int(values.sum())
            for name, values in sorted(interpretation.zero_origin.items())
        },
    }
    if scope is not None:
        _require(type(scope) is SpmReceiptScope, "SCOPE_TYPE")
        payload["spm_unit_rows"] = scope.rows
        payload["assignment_counts"] = scope.assignment_counts()
        payload["multi_spm_unit_households"] = int(
            len(np.unique(scope.household_row[scope.spm_units_in_household > 1]))
        )
        payload["housing_unit_authentication"] = (
            scope.housing_unit.source_authentication
        )
        payload["occupied_hu_counts"] = scope.housing_unit.occupied_hu_counts()
        payload["household_kind_counts"] = scope.housing_unit.household_kind_counts()
        payload["spm_units_with_bound_housing_unit"] = int(
            scope.housing_unit_valid.sum()
        )
    if projection is not None:
        _require(type(projection) is EngineReceiptProjection, "PROJECTION_TYPE")
        payload["engine_input_code_counts"] = projection.code_counts()
        payload["multi_unit_negative"] = projection.multi_unit_negative
        payload["engine_boolean_complete"] = bool(projection.withheld.sum() == 0)
    payload["diagnostics_sha256"] = _sha(_json(payload))
    return payload


def describe_integration_seam() -> dict[str, Any]:
    """The proposed attachment point, its inputs, and what it refuses to export."""
    return {
        "schema_version": CONSUMER_VERSION,
        "consumer_module": "microcosm.build.us_runtime.asec_housing_receipt_consumer",
        "reads": {
            "artifact": "microcosm.asec_housing_status_observations.v1",
            "attachment": "microcosm.asec_housing_status_attachment.v1",
            "attached_columns": list(ATTACHED_COLUMNS),
            "housing_unit_artifact": "microcosm.asec_housing_universe.v1",
            "housing_unit_attachment": (
                "microcosm.asec_housing_universe_attachment.v1"
            ),
            "housing_unit_attached_columns": list(UNIVERSE_ATTACHED_COLUMNS),
        },
        "attaches_after": (
            "microcosm.build.us_runtime.asec_housing_universe_source"
            ".attach_housing_universe"
        ),
        "membership_source": (
            "the attached Frame's household_id, spm_unit_id and person memberships"
        ),
        "produces": {
            "receives_housing_assistance": "three-state code, Boolean only when complete",
            "takes_up_housing_assistance_if_eligible": "not derived",
            "pre_subsidy_rent": "not derived; needs the counterfactual charge source",
        },
        "refuses_to_export": {
            "rent": "formula-owned",
            "housing_assistance": "formula-owned",
            "hud_hap": "formula-owned",
            "spm_unit_capped_housing_subsidy": "formula-owned",
        },
        "export_gate": "microcosm.frame.adapters.policyengine_us.PolicyEngineUSEngine",
        "quantities": HOUSING_QUANTITIES,
        "unresolved_before_launch": counterfactual_charge_availability(),
    }
