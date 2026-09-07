"""Occupied-HU membership and tenure, through the adapters the contract pins.

Nothing here reimplements a source classification. The ACS side calls
:func:`microcosm.build.us_runtime.acs_housing_universe.classify_acs_housing_universe`
and the ASEC side calls
:func:`microcosm.build.us_runtime.asec_housing_universe.classify_synthetic_housing_universe`
— the same pure adapters ``/reference_authority/HU_adapter_clarification`` and
``/candidate_semantics`` pin by digest. Both imports are deferred to call time
so importing this package's pure core never loads the eager ``us_runtime``
package. :func:`verify_pinned_adapters` checks the installed files against the
contract's pins, because a changed adapter is a scope question rather than a
silent pin refresh.

Three refusals stay distinct here, exactly as the contract separates them:

* An out-of-domain source code makes the authenticated adapter refuse the whole
  batch. That is ``source_domain_refused``, and no classification artifact or
  unclassified mass may be fabricated from it.
* Membership or a denominator that cannot be established is
  ``denominator_unknown``, regardless of the estimated mass. An ACS
  ``TYPEHUGQ=1`` row with ``NP=0`` **retained** without an approved bridge is
  this case: the pure adapter leaves occupancy unresolved and asserts no
  vacancy.
* An absent or unapproved required mapping is ``mapping_unapproved``, again
  regardless of mass. Both may be retained together.

Excluding NP=0 rows as known vacant is a fourth, different thing:
``vacancy_bridge_missing``, which applies to a *drop*, not to a retention. An
unknown drop history is ``unresolved_source_lineage`` and cannot imply a zero
drop count; a known zero drop count waives none of the other requirements; and
a count that is not a nonnegative integer is refused outright rather than read
as a known nonpositive drop.

The GQ weight bridge is likewise an external required predicate. The shipped
ACS loader's policy of substituting ``PWGTP`` for a zero ``WGTP`` on a GQ
household placeholder is a representation policy, not observed occupied-HU
weighting, and this package does not adopt it: with no bridge injected the
population and age blocks report ``gq_weight_bridge_missing``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from types import MappingProxyType

from microcosm.build.cd_benchmark.canonical import is_sha256_hex
from microcosm.build.cd_benchmark.protocol import CompiledProtocol
from microcosm.build.cd_benchmark.reasons import Reason, dedupe

__all__ = [
    "AcsHousingUniverse",
    "AsecHousingUniverse",
    "GqWeightBridge",
    "PINNED_ADAPTER_MODULES",
    "UniverseError",
    "VacancyBridge",
    "acs_housing_universe",
    "asec_housing_universe",
    "gq_weight_bridge_reasons",
    "vacant_drop_count",
    "verify_pinned_adapters",
]

#: Adapter modules this package binds to, as ``us_runtime`` file names. Their
#: expected digests come from the contract, not from a constant retyped here.
PINNED_ADAPTER_MODULES = (
    "acs_housing_universe.py",
    "asec_housing_universe.py",
    "cd_reference.py",
)


class UniverseError(ValueError):
    """A source-universe input is not what the pinned adapters accept."""


@dataclass(frozen=True)
class VacancyBridge:
    """The separately approved authenticated vacancy-status bridge.

    Constructing one asserts that the caller resolved the artifacts
    ``/candidate_semantics/.../required_future_bridge`` lists. This package
    authenticates none of them and ships no default: the parameter is ``None``
    everywhere unless a caller passes one in.
    """

    definition_id: str
    definition_sha256: str
    classifier_implementation_sha256: str
    vacancy_artifact_id: str
    vacancy_artifact_sha256: str
    source_custody_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.definition_sha256,
            self.classifier_implementation_sha256,
            self.vacancy_artifact_sha256,
            self.source_custody_sha256,
        ):
            if not is_sha256_hex(value):
                raise UniverseError("A vacancy bridge needs lowercase digests.")
        for value in (self.definition_id, self.vacancy_artifact_id):
            if not isinstance(value, str) or not value.strip():
                raise UniverseError("A vacancy bridge needs nonempty artifact ids.")


@dataclass(frozen=True)
class GqWeightBridge:
    """The approved authenticated GQ source person weight and reducer binding.

    Same rule as the vacancy bridge: an explicit caller-supplied predicate with
    no default, and never inferred from the existing loader's policy.
    """

    weight_definition_id: str
    weight_definition_sha256: str
    person_to_placeholder_mapping_sha256: str
    custody_manifest_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.weight_definition_sha256,
            self.person_to_placeholder_mapping_sha256,
            self.custody_manifest_sha256,
        ):
            if not is_sha256_hex(value):
                raise UniverseError("A GQ weight bridge needs lowercase digests.")
        if not isinstance(self.weight_definition_id, str) or not (
            self.weight_definition_id.strip()
        ):
            raise UniverseError("A GQ weight bridge needs a nonempty definition id.")


def vacant_drop_count(value: object) -> int | None:
    """Read a caller's vacant-row drop count, or refuse a value that is not one.

    ``None`` is the unknown drop history the module docstring separates from a
    known zero, and it is returned unchanged. Anything else must be a
    nonnegative JSON-style integer. A negative number is not a known nonpositive
    drop: read as one it would waive the vacancy-bridge obligation that a
    positive drop carries, so it is refused rather than compared.

    Raises:
        UniverseError: If the count is a bool, a float, a string or negative.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise UniverseError("vacant_household_rows_dropped must be an integer or None.")
    if value < 0:
        raise UniverseError(
            "vacant_household_rows_dropped must be nonnegative; a negative count "
            "is not a known nonpositive drop."
        )
    return value


def gq_weight_bridge_reasons(bridge: GqWeightBridge | None) -> tuple[Reason, ...]:
    """Return ``gq_weight_bridge_missing`` unless an approved bridge is supplied."""
    return (
        () if isinstance(bridge, GqWeightBridge) else (Reason.GQ_WEIGHT_BRIDGE_MISSING,)
    )


def verify_pinned_adapters(protocol: CompiledProtocol) -> Mapping[str, bool]:
    """Compare the installed adapter modules with the contract's pinned digests.

    Returns one boolean per module in :data:`PINNED_ADAPTER_MODULES`. A ``False``
    means the adapter this contract inspected has changed; that is a scope
    review, not a pin to refresh in passing.
    """
    package = "microcosm.build.us_runtime"
    result = {}
    for name in PINNED_ADAPTER_MODULES:
        expected = protocol.pinned_code_sha256.get(name)
        if expected is None:
            raise UniverseError(f"The contract pins no digest for {name}.")
        data = resources.files(package).joinpath(name).read_bytes()
        result[name] = hashlib.sha256(data).hexdigest() == expected
    return MappingProxyType(result)


@dataclass(frozen=True)
class AcsHousingUniverse:
    """ACS occupied-HU membership and tenure counts, with the reasons they carry."""

    occupied: int
    outside: int
    occupancy_unresolved: int
    owner: int
    renter: int
    tenure_unclassified: int
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AsecHousingUniverse:
    """ASEC occupied-HU membership and tenure counts, with the reasons they carry."""

    occupied: int
    outside: int
    occupancy_unresolved: int
    owner: int
    renter: int
    tenure_unclassified: int
    unresolved_reason_bits: Mapping[int, int]
    reasons: tuple[Reason, ...]
    evidence: Mapping[str, object] = field(default_factory=dict)


def _counts(codes, values: Sequence[int]) -> dict[int, int]:
    return {value: int((codes == value).sum()) for value in values}


def acs_housing_universe(
    households,
    *,
    tenure_valid,
    vacant_household_rows_dropped: int | None,
    vacancy_bridge: VacancyBridge | None = None,
) -> AcsHousingUniverse:
    """Classify ACS housing rows through the pinned pure adapter.

    Args:
        households: An exact ``pandas`` frame with int64 ``TYPEHUGQ``, ``NP``
            and ``TEN``, as the adapter requires.
        tenure_valid: The adapter's positional bool validity vector for ``TEN``.
        vacant_household_rows_dropped: How many NP=0 housing rows an upstream
            loader dropped before these rows were produced, or ``None`` when
            that history is unknown. ``None`` is not zero.
        vacancy_bridge: The separately approved bridge, or ``None``.

    Returns:
        Counts plus the reasons they carry. A retained NP=0 row without a bridge
        is ``denominator_unknown``; a dropped one is ``vacancy_bridge_missing``.

    Raises:
        UniverseError: If ``vacant_household_rows_dropped`` is not a
            nonnegative integer or ``None``. This is checked before the adapter
            is imported, so a miscounted drop never reaches a classification.
    """
    vacant_household_rows_dropped = vacant_drop_count(vacant_household_rows_dropped)

    from microcosm.build.us_runtime.acs_housing_universe import (
        ACSHousingUniverseRefusalError,
        classify_acs_housing_universe,
    )

    try:
        classified, header = classify_acs_housing_universe(
            households, tenure_valid=tenure_valid
        )
    except ACSHousingUniverseRefusalError as error:
        return AcsHousingUniverse(
            occupied=0,
            outside=0,
            occupancy_unresolved=0,
            owner=0,
            renter=0,
            tenure_unclassified=0,
            reasons=(Reason.SOURCE_DOMAIN_REFUSED,),
            evidence=MappingProxyType(
                {"adapter_refusal": str(error), "no_fabricated_classification": True}
            ),
        )
    occupancy = _counts(classified["occupied_hu"].to_numpy(), (0, 1, 2))
    tenure = _counts(classified["hu_tenure_class"].to_numpy(), (1, 2, 3))
    problems: list[Reason] = []
    notes: dict[str, object] = {
        "adapter_source_authentication": header["source_authentication"],
        "np_zero_asserts_no_vacancy": True,
        "vacancy_bridge_supplied": vacancy_bridge is not None,
        "vacant_household_rows_dropped": vacant_household_rows_dropped,
    }
    if occupancy[0] and vacancy_bridge is None:
        problems.append(Reason.DENOMINATOR_UNKNOWN)
    if vacant_household_rows_dropped is None:
        problems.append(Reason.UNRESOLVED_SOURCE_LINEAGE)
    elif vacant_household_rows_dropped > 0 and vacancy_bridge is None:
        problems.append(Reason.VACANCY_BRIDGE_MISSING)
    return AcsHousingUniverse(
        occupied=occupancy[1],
        outside=occupancy[2],
        occupancy_unresolved=occupancy[0],
        owner=tenure[1],
        renter=tenure[2],
        tenure_unclassified=tenure[3],
        reasons=dedupe(problems),
        evidence=MappingProxyType(notes),
    )


def asec_housing_universe(observations) -> AsecHousingUniverse:
    """Classify invented ASEC household rows through the pinned pure adapter.

    The adapter refuses a whole batch when any context code leaves its domain —
    an ``H_TENURE`` outside 0..3, for instance. That refusal is
    ``source_domain_refused``: no unclassified mass is manufactured from it, and
    no partial classification is returned.
    """
    from microcosm.build.us_runtime.asec_housing_universe import (
        HousingUniverseRefusalError,
        classify_synthetic_housing_universe,
    )

    try:
        universe = classify_synthetic_housing_universe(observations)
    except HousingUniverseRefusalError as error:
        reason = (
            Reason.SOURCE_DOMAIN_REFUSED
            if str(error) == "CONTEXT_DOMAIN"
            else Reason.INVALID_REDUCER_INPUT
        )
        return AsecHousingUniverse(
            occupied=0,
            outside=0,
            occupancy_unresolved=0,
            owner=0,
            renter=0,
            tenure_unclassified=0,
            unresolved_reason_bits=MappingProxyType({}),
            reasons=(reason,),
            evidence=MappingProxyType(
                {"adapter_refusal": str(error), "no_fabricated_classification": True}
            ),
        )
    occupancy = _counts(universe.array("occupied_hu"), (0, 1, 2))
    tenure = _counts(universe.array("hu_tenure_class"), (1, 2, 3))
    reasons_array = universe.array("unresolved_reasons")
    bits = {
        bit: int((reasons_array & bit).astype(bool).sum())
        for bit in (1, 2, 4, 8, 16, 32, 64, 128)
    }
    problems: list[Reason] = []
    if occupancy[0]:
        problems.append(Reason.DENOMINATOR_UNKNOWN)
    return AsecHousingUniverse(
        occupied=occupancy[1],
        outside=occupancy[2],
        occupancy_unresolved=occupancy[0],
        owner=tenure[1],
        renter=tenure[2],
        tenure_unclassified=tenure[3],
        unresolved_reason_bits=MappingProxyType(bits),
        reasons=dedupe(problems),
        evidence=MappingProxyType(
            {
                "adapter_source_authentication": universe.header_data[
                    "source_authentication"
                ],
                "release_eligible": universe.header_data["release_eligible"],
            }
        ),
    )
