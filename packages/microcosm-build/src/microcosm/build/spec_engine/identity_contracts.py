"""Resolution of pipeline-owned stochastic draw-site bindings.

The immutable seed protocol describes *how* each draw is consumed.  Country
configuration separately binds every selected protocol site to the concrete
producer node, source stage, or outer pipeline operation that owns the draw.
Keeping this relation explicit prevents the compiler from guessing ownership
from kernel names when it constructs node slices and stream maps.
"""

from __future__ import annotations

from collections.abc import Mapping

from .model import SeedSiteBinding, SeedSiteOwner, SeedSiteOwnerKind
from .seeds import SeedProtocol


class IdentityContractError(ValueError):
    """A pipeline identity or seed-site owner binding is not closed."""


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IdentityContractError(f"{location}: object required")
    return value


def _array(value: object, *, location: str) -> list[object]:
    if not isinstance(value, list):
        raise IdentityContractError(f"{location}: array required")
    return value


def _unique_strings(value: object, *, location: str) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(_array(value, location=location)):
        if not isinstance(item, str) or not item:
            raise IdentityContractError(
                f"{location}/{index}: non-empty string required"
            )
        if item in result:
            raise IdentityContractError(f"{location}/{index}: duplicate value {item!r}")
        result.append(item)
    return tuple(result)


def _pipeline_operation_ids(spine: Mapping[str, object]) -> frozenset[str]:
    contract = _mapping(
        spine.get("pipeline_contract"), location="spine/pipeline_contract"
    )
    fields = (
        "stacked_operator_order",
        "pre_clone_source_operator_order",
        "post_clone_source_operator_order",
        "derive_operator_order",
        "auxiliary_operations",
    )
    operations: set[str] = set()
    for field in fields:
        operations.update(
            _unique_strings(
                contract.get(field), location=f"spine/pipeline_contract/{field}"
            )
        )
    return frozenset(operations)


def resolve_seed_site_bindings(
    spine: Mapping[str, object],
    *,
    protocol: SeedProtocol,
    source_stage_ids: frozenset[str],
    producer_node_ids: frozenset[str],
) -> tuple[SeedSiteBinding, ...]:
    """Resolve an exhaustive site-to-owner map for the selected protocol.

    Minimal shared-core bundles may omit the US pipeline extension entirely.
    Once either US identity field is present, both are required and the map
    must cover the selected protocol exactly once.
    """

    has_contract = "pipeline_contract" in spine
    has_bindings = "seed_site_bindings" in spine
    if not has_contract and not has_bindings:
        return ()
    if not has_contract or not has_bindings:
        raise IdentityContractError(
            "spine: pipeline_contract and seed_site_bindings must be declared together"
        )

    valid_ids = {
        SeedSiteOwnerKind.PRODUCER_NODE: producer_node_ids,
        SeedSiteOwnerKind.SOURCE_STAGE: source_stage_ids,
        SeedSiteOwnerKind.PIPELINE_OPERATION: _pipeline_operation_ids(spine),
    }
    expected_sites = {site.id for site in protocol.sites}
    seen_sites: set[str] = set()
    bindings: list[SeedSiteBinding] = []
    for index, raw_binding in enumerate(
        _array(spine.get("seed_site_bindings"), location="spine/seed_site_bindings")
    ):
        location = f"spine/seed_site_bindings/{index}"
        binding = _mapping(raw_binding, location=location)
        site = binding.get("site")
        if not isinstance(site, str) or not site:
            raise IdentityContractError(f"{location}/site: identifier required")
        if site in seen_sites:
            raise IdentityContractError(
                f"{location}/site: duplicate seed site {site!r}"
            )
        if site not in expected_sites:
            raise IdentityContractError(
                f"{location}/site: unknown {protocol.id} seed site {site!r}"
            )
        seen_sites.add(site)

        owners: list[SeedSiteOwner] = []
        seen_owners: set[tuple[SeedSiteOwnerKind, str]] = set()
        for owner_index, raw_owner in enumerate(
            _array(binding.get("owners"), location=f"{location}/owners")
        ):
            owner_location = f"{location}/owners/{owner_index}"
            owner = _mapping(raw_owner, location=owner_location)
            try:
                kind = SeedSiteOwnerKind(str(owner.get("kind")))
            except ValueError as error:
                raise IdentityContractError(
                    f"{owner_location}/kind: unknown owner namespace"
                ) from error
            owner_id = owner.get("id")
            if not isinstance(owner_id, str) or not owner_id:
                raise IdentityContractError(
                    f"{owner_location}/id: non-empty string required"
                )
            if owner_id not in valid_ids[kind]:
                raise IdentityContractError(
                    f"{owner_location}/id: dangling {kind.value} {owner_id!r}"
                )
            owner_key = (kind, owner_id)
            if owner_key in seen_owners:
                raise IdentityContractError(
                    f"{owner_location}: duplicate owner {kind.value}:{owner_id}"
                )
            seen_owners.add(owner_key)
            owners.append(SeedSiteOwner(kind=kind, id=owner_id))
        if not owners:
            raise IdentityContractError(f"{location}/owners: must not be empty")
        bindings.append(SeedSiteBinding(site=site, owners=tuple(owners)))

    missing = sorted(expected_sites - seen_sites)
    if missing:
        raise IdentityContractError(
            "spine/seed_site_bindings: selected protocol sites are unbound; "
            f"missing={missing!r}"
        )
    return tuple(bindings)


__all__ = [
    "IdentityContractError",
    "resolve_seed_site_bindings",
]
