"""Invented-only native AGI selection and unique household support matching.

This pure preparation contract does not issue source authority, place donor
amounts, or validate a receiving host. Its result can declare the separately
checked real structural EXPAND. Donor support weights never become household
mass. CG-only transfer remains outside this slice and refuses explicitly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from microcosm.frame import US_SCHEMA, Frame, WeightKind
from microcosm.graph.canonical import canonical_json

from . import graph_native_puf_tail_expand as expansion
from . import native_puf_tail as selection
from .acs_income_universe import ACS_PUMS_EARNINGS_MINIMUM_AGE
from .puf_capital_gains_tail import _RECIPIENT_AGI_PROXY_COLUMNS
from .puf_interest_components import US_PUF_E19200_AGI_BANDS
from .support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from .survey_population_domains import Domain, Source
from .survey_population_domains import declaration as domain_declaration

PROTOCOL = "microcosm.us.native-tail-invented-matching/1"
REFERENCE_COMMIT = "f7df78b2a00421f9b90305a9b7db192444075ae0"
RECIPIENT_PROXY_COMPONENTS = _RECIPIENT_AGI_PROXY_COLUMNS
_MAX_ID = np.iinfo(np.int64).max


def _require(condition, reason):
    if not condition:
        raise ValueError("NATIVE_TAIL_MATCH_" + reason)


def _id(value):
    return type(value) is int and 0 <= value <= _MAX_ID


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class InventedIncomeBasis:
    """Declared comparison basis, not evidence of source-window equivalence."""

    income_year: int
    price_year: int
    currency: str = "USD"
    period: str = "calendar_year"


@dataclass(frozen=True, slots=True)
class InventedDonorSourceAgi:
    donor_recid: int
    source_adjusted_gross_income: float


@dataclass(frozen=True, slots=True)
class InventedRecipientPerson:
    """Every required literal is known; missing/Boolean numeric cells refuse."""

    person_id: int
    role: str
    age: int
    proxy_components: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class InventedRecipientTaxUnit:
    tax_unit_id: int
    filing_status_code: int


@dataclass(frozen=True, slots=True)
class MatchedRoleCoordinates:
    """Clone1 parent coordinates; resolve clone2 IDs through EXPAND lineage.

    None of these IDs names a newly minted tail record. Monetary placement
    must use the actual EXPAND lineage and a separately qualified receiving
    boundary; this descriptive record does not issue that authority.
    """

    donor_recid: int
    household_id: int
    tax_unit_id: int
    head_person_id: int
    spouse_person_id: int | None


@dataclass(frozen=True, slots=True)
class InventedTailMatch:
    """Immutable descriptive result, not a lease or a matched-payload issuer."""

    selection: selection.NativeTailSelection
    assignments: tuple[expansion.InventedTailAssignment, ...]
    roles: tuple[MatchedRoleCoordinates, ...]
    expansion: expansion.InventedTailExpansion | None
    receipt: bytes

    @property
    def source_admission_issued(self):
        return False

    @property
    def matching_qualified(self):
        return False

    @property
    def release_eligible(self):
        return False

    @property
    def sha256(self):
        return hashlib.sha256(self.receipt).hexdigest()


def _basis(value):
    _require(type(value) is InventedIncomeBasis, "BASIS_TYPE")
    _require(
        all(
            type(year) is int and 1850 <= year <= 2200
            for year in (value.income_year, value.price_year)
        )
        and type(value.currency) is str
        and value.currency == "USD"
        and type(value.period) is str
        and value.period == "calendar_year",
        "BASIS_VALUES",
    )


def _band(amount):
    bounds = [
        b.upper_bound for b in US_PUF_E19200_AGI_BANDS if b.upper_bound is not None
    ]
    return int(np.searchsorted(bounds, amount, side="right"))


def _frame_inputs(frame, domains):
    _require(type(frame) is Frame and frame.schema == US_SCHEMA, "FRAME_SCHEMA")
    _require(frame.weighted_entities == ("household",), "SOLE_HOUSEHOLD_WEIGHT")
    weights = frame.weights_for("household")
    _require(weights.kind is WeightKind.IMPORTANCE, "HOUSEHOLD_WEIGHT_KIND")
    _require(
        type(domains) is tuple and 0 < len(domains) <= expansion.MAX_FIXTURE_HOUSEHOLDS,
        "DOMAIN_ROSTER",
    )
    tables = {entity: frame.table(entity) for entity in US_SCHEMA.entities}
    snapshots = {}
    for entity, table in tables.items():
        names = (
            US_SCHEMA.entity_id_column(entity),
            *expansion.provenance_columns(entity),
        )
        _require(0 < len(table) <= expansion.MAX_FIXTURE_ENTITY_ROWS, "ENTITY_SIZE")
        for name in names:
            _require(name in table and not table[name].isna().any(), "ORIGIN_KNOWN")
            if name == support_channel_column(entity):
                _require(
                    table[name].isin(tuple(s.value for s in Source)).all(), "CHANNEL"
                )
            else:
                _require(
                    table[name].dtype == np.dtype("int64") and (table[name] >= 0).all(),
                    "ORIGIN_INTEGER",
                )
        _require(table[support_clone_index_column(entity)].isin((0, 1)).all(), "CLONE")
        _require(table[names[0]].is_unique, "DUPLICATE_ENTITY_ID")
        _require(
            not table.duplicated(
                [support_source_id_column(entity), support_clone_index_column(entity)]
            ).any(),
            "DUPLICATE_ORIGIN",
        )
        snapshots[entity] = (
            table.loc[:, list(names)].sort_values(names[0]).to_dict("records")
        )
    hh = tables["household"].set_index("household_id")
    domain_by_id = {}
    origin_domains = {}
    units = {row.domain: row.statistical_unit for row in domain_declaration()}
    for row in domains:
        _require(type(row) is expansion.InventedHouseholdDomain, "DOMAIN_TYPE")
        _require(
            all(
                _id(i)
                for i in (row.household_id, row.support_source_id, row.spine_source_id)
            ),
            "DOMAIN_IDS",
        )
        _require(
            type(row.source) is Source
            and type(row.domain) is Domain
            and type(row.clone_index) is int
            and row.clone_index in (0, 1)
            and type(row.statistical_unit) is str
            and row.statistical_unit == units[row.domain],
            "DOMAIN_VALUES",
        )
        _require(
            row.household_id not in domain_by_id and row.household_id in hh.index,
            "DOMAIN_IDS",
        )
        live = hh.loc[row.household_id]
        _require(
            tuple(live[n] for n in expansion.provenance_columns("household"))
            == (
                row.support_source_id,
                row.spine_source_id,
                row.source.value,
                row.clone_index,
            ),
            "DOMAIN_ORIGIN",
        )
        origin = (row.source, row.support_source_id, row.spine_source_id)
        value = (row.domain, row.statistical_unit)
        _require(
            origin not in origin_domains or origin_domains[origin] == value,
            "DOMAIN_ORIGIN_DISAGREEMENT",
        )
        origin_domains[origin] = value
        domain_by_id[row.household_id] = row
    _require(set(domain_by_id) == set(hh.index), "DOMAIN_COVERAGE")
    people = tables["person"]
    _require(set(people.person_household_id) == set(hh.index), "HOUSEHOLD_COVERAGE")
    for group in US_SCHEMA.group_entities:
        link = US_SCHEMA.membership_column(group)
        group_table = tables[group].set_index(US_SCHEMA.entity_id_column(group))
        _require(
            people[link].dtype == np.dtype("int64")
            and set(people[link]) == set(group_table.index),
            "MEMBERSHIP_COVERAGE",
        )
        joined = group_table.loc[people[link]]
        for suffix in (support_channel_column, support_clone_index_column):
            _require(
                np.array_equal(
                    joined[suffix(group)].to_numpy(),
                    people[suffix("person")].to_numpy(),
                ),
                "MEMBERSHIP_ORIGIN",
            )
        _require(
            (people.groupby(link).person_household_id.nunique() == 1).all(),
            "GROUP_NOT_HOUSEHOLD_CLOSED",
        )
    weight_values = weights.values
    _require(
        len(weight_values) == len(hh)
        and np.isfinite(weight_values).all()
        and (weight_values >= 0).all(),
        "HOUSEHOLD_MASS",
    )
    weight_by_id = dict(
        zip(tables["household"].household_id, weight_values, strict=True)
    )
    snapshots["memberships"] = (
        people.loc[
            :,
            [
                "person_id",
                *(US_SCHEMA.membership_column(g) for g in US_SCHEMA.group_entities),
            ],
        ]
        .sort_values("person_id")
        .to_dict("records")
    )
    snapshots["importance"] = sorted(
        (int(i), float(w)) for i, w in weight_by_id.items()
    )
    snapshots["domains"] = [
        (
            r.household_id,
            r.source.value,
            r.support_source_id,
            r.spine_source_id,
            r.clone_index,
            r.domain.value,
            r.statistical_unit,
        )
        for r in sorted(domains, key=lambda r: r.household_id)
    ]
    return tables, domain_by_id, weight_by_id, snapshots


def _recipient_rows(tables, domains, weights, persons, tax_units):
    _require(type(persons) is tuple and type(tax_units) is tuple, "RECIPIENT_ROSTERS")
    people = tables["person"]
    clone_people = people.loc[people[support_clone_index_column("person")] == 1]
    by_person, by_tax_unit = {}, {}
    statuses = selection._filing_status_domain()
    for row in persons:
        _require(
            type(row) is InventedRecipientPerson and _id(row.person_id), "PERSON_TYPE"
        )
        _require(row.person_id not in by_person, "DUPLICATE_PERSON")
        _require(
            type(row.role) is str and row.role in ("head", "spouse", "dependent"),
            "ROLE_KNOWN",
        )
        _require(type(row.age) is int and 0 <= row.age <= 130, "AGE_KNOWN")
        _require(
            type(row.proxy_components) is tuple
            and all(
                type(cell) is tuple and len(cell) == 2 for cell in row.proxy_components
            )
            and tuple(cell[0] for cell in row.proxy_components)
            == RECIPIENT_PROXY_COMPONENTS
            and all(_finite(cell[1]) for cell in row.proxy_components),
            "PROXY_KNOWN",
        )
        by_person[row.person_id] = row
    for row in tax_units:
        _require(
            type(row) is InventedRecipientTaxUnit and _id(row.tax_unit_id),
            "TAX_UNIT_TYPE",
        )
        _require(row.tax_unit_id not in by_tax_unit, "DUPLICATE_TAX_UNIT")
        _require(
            type(row.filing_status_code) is int and row.filing_status_code in statuses,
            "FILING_STATUS_KNOWN",
        )
        by_tax_unit[row.tax_unit_id] = row
    _require(set(by_person) == set(clone_people.person_id), "PERSON_COVERAGE")
    _require(
        set(by_tax_unit) == set(clone_people.person_tax_unit_id), "TAX_UNIT_COVERAGE"
    )
    candidates, exclusions = [], []
    tus = tables["tax_unit"].set_index("tax_unit_id")
    for household_id, members in clone_people.groupby("person_household_id", sort=True):
        household_id = int(household_id)
        domain, weight = domains[household_id], float(weights[household_id])
        reason = None
        if domain.domain not in (Domain.SHARED_HOUSING, Domain.RESIDUAL_HOUSING):
            reason = "group_quarters"
        elif weight == 0:
            reason = "zero_household_importance"
        elif weight / 2 <= 0 or weight / 2 + weight / 2 != weight:
            reason = "unrepresentable_equal_halves"
        elif members.person_tax_unit_id.nunique() != 1:
            reason = "multiple_tax_units"
        roles = [by_person[int(i)] for i in members.person_id]
        heads = [r for r in roles if r.role == "head"]
        spouses = [r for r in roles if r.role == "spouse"]
        if reason is None and (len(heads) != 1 or len(spouses) > 1):
            reason = "incompatible_role_structure"
        # Inherited reference matching method: use the ACS earnings-universe
        # age floor for head/spouse compatibility. This is a declared support
        # choice, not a tax-unit definition or an independently observed role.
        if reason is None and any(
            r.age < ACS_PUMS_EARNINGS_MINIMUM_AGE for r in (*heads, *spouses)
        ):
            reason = "underage_head_or_spouse"
        if reason is not None:
            exclusions.append(dict(household_id=household_id, reason=reason))
            continue
        tax_unit_id = int(members.person_tax_unit_id.iloc[0])
        try:
            proxy = math.fsum(
                value
                for r in sorted(roles, key=lambda r: r.person_id)
                for _, value in r.proxy_components
            )
        except OverflowError:
            raise ValueError("NATIVE_TAIL_MATCH_PROXY_FINITE") from None
        _require(math.isfinite(proxy), "PROXY_FINITE")
        candidates.append(
            dict(
                household_id=household_id,
                tax_unit_id=tax_unit_id,
                household_source_id=domain.support_source_id,
                tax_unit_source_id=int(
                    tus.loc[tax_unit_id, support_source_id_column("tax_unit")]
                ),
                filing_status_code=by_tax_unit[tax_unit_id].filing_status_code,
                proxy_agi=proxy,
                band=_band(proxy),
                household_importance=weight,
                head_person_id=heads[0].person_id,
                spouse_person_id=spouses[0].person_id if spouses else None,
            )
        )
    return candidates, exclusions


def match_invented_native_puf_tail(
    projection: selection.DeclaredPufTailRoleProjection,
    *,
    capital_gains_mask: np.ndarray,
    source_eligible: np.ndarray,
    donor_source_agi: tuple[InventedDonorSourceAgi, ...],
    donor_basis: InventedIncomeBasis,
    recipient_basis: InventedIncomeBasis,
    frame: Frame,
    domains: tuple[expansion.InventedHouseholdDomain, ...],
    persons: tuple[InventedRecipientPerson, ...],
    tax_units: tuple[InventedRecipientTaxUnit, ...],
    seed: int,
) -> InventedTailMatch:
    """Select with maintained algebra, then match unique compatible fixture parents.

    Per-status shortages are explicit nonassignment; CG-only selection refuses
    the entire request. Known declarations are never inferred from order, zero
    filled, or promoted to source observations.
    """
    _require(type(seed) is int and 0 <= seed <= _MAX_ID, "SEED")
    _basis(donor_basis)
    _basis(recipient_basis)
    _require(donor_basis == recipient_basis, "COMPARISON_BASIS")
    selected = selection.select_native_puf_tail(
        projection,
        capital_gains_mask=capital_gains_mask,
        source_eligible=source_eligible,
    )
    _require(all(d.arm in (2, 3) for d in selected.donors), "CG_ONLY_UNSUPPORTED")
    returns, _ = selection.projection_tables(projection)
    status_by_id = dict(
        zip(returns.donor_recid, returns.filing_status_code, strict=True)
    )
    _require(type(donor_source_agi) is tuple, "DONOR_AGI_ROSTER")
    agi = {}
    for row in donor_source_agi:
        _require(
            type(row) is InventedDonorSourceAgi
            and _id(row.donor_recid)
            and row.donor_recid > 0,
            "DONOR_AGI_IDS",
        )
        _require(row.donor_recid not in agi, "DONOR_AGI_DUPLICATE")
        _require(_finite(row.source_adjusted_gross_income), "DONOR_AGI_KNOWN")
        agi[row.donor_recid] = float(row.source_adjusted_gross_income)
    _require(set(agi) == {d.donor_recid for d in selected.donors}, "DONOR_AGI_COVERAGE")
    tables, domain_by_id, weights, frame_inputs = _frame_inputs(frame, domains)
    candidates, exclusions = _recipient_rows(
        tables, domain_by_id, weights, persons, tax_units
    )
    candidates.sort(key=lambda r: (r["household_source_id"], r["tax_unit_source_id"]))
    for row, priority in zip(
        candidates, np.random.default_rng(seed).random(len(candidates)), strict=True
    ):
        row["priority"] = float(priority)
    assignments, roles, support = [], [], []
    for status in sorted({int(status_by_id[d.donor_recid]) for d in selected.donors}):
        donors = sorted(
            (d for d in selected.donors if status_by_id[d.donor_recid] == status),
            key=lambda d: (not d.spouse_payload_nonzero, d.donor_recid),
        )
        pool = [r for r in candidates if r["filing_status_code"] == status]
        spouse_demand = sum(d.spouse_payload_nonzero is True for d in donors)
        spouse_capacity = sum(r["spouse_person_id"] is not None for r in pool)
        sufficient = len(pool) >= len(donors) and spouse_capacity >= spouse_demand
        support.append(
            dict(
                filing_status_code=status,
                donors=len(donors),
                compatible_parents=len(pool),
                spouse_demand=spouse_demand,
                spouse_capacity=spouse_capacity,
                status="assigned" if sufficient else "insufficient_unique_support",
                donor_ids=[d.donor_recid for d in donors],
            )
        )
        if not sufficient:
            continue
        for donor in donors:
            donor_band = _band(agi[donor.donor_recid])
            compatible = [
                r
                for r in pool
                if not donor.spouse_payload_nonzero or r["spouse_person_id"] is not None
            ]
            _require(bool(compatible), "CAPACITY_INVARIANT")
            parent = min(
                compatible,
                key=lambda r: (
                    abs(r["band"] - donor_band),
                    r["band"],
                    r["priority"],
                    r["household_source_id"],
                    r["tax_unit_source_id"],
                ),
            )
            pool.remove(parent)
            assignments.append(
                expansion.InventedTailAssignment(
                    donor.donor_recid, parent["household_id"], donor.weight
                )
            )
            roles.append(
                MatchedRoleCoordinates(
                    donor.donor_recid,
                    parent["household_id"],
                    parent["tax_unit_id"],
                    parent["head_person_id"],
                    parent["spouse_person_id"],
                )
            )
    assignments = tuple(sorted(assignments, key=lambda r: r.donor_id))
    roles = tuple(sorted(roles, key=lambda r: r.donor_recid))
    declared = (
        expansion.declare_invented_tail_expansion(domains, assignments)
        if assignments
        else None
    )
    receipt = canonical_json(
        dict(
            protocol=PROTOCOL,
            reference_commit=REFERENCE_COMMIT,
            seed=seed,
            policy=expansion.POLICY,
            input_basis=asdict(donor_basis),
            selection_sha256=hashlib.sha256(selected.receipt).hexdigest(),
            selection_input_sha256=selected.selection_input_sha256,
            projection_sha256=projection.sha256,
            frame_input_sha256=_digest(frame_inputs),
            donor_source_agi=sorted(agi.items()),
            recipient_person_sha256=_digest(
                [asdict(r) for r in sorted(persons, key=lambda r: r.person_id)]
            ),
            recipient_tax_unit_sha256=_digest(
                [asdict(r) for r in sorted(tax_units, key=lambda r: r.tax_unit_id)]
            ),
            proxy_components=RECIPIENT_PROXY_COMPONENTS,
            candidates=candidates,
            exclusions=exclusions,
            support=support,
            assignments=[asdict(r) for r in assignments],
            roles=[asdict(r) for r in roles],
            expansion_sha256=declared.sha256 if declared else None,
            source_admission_issued=False,
            matching_qualified=False,
            release_eligible=False,
            money_placement=False,
            geography_assigned=False,
        )
    )
    return InventedTailMatch(selected, assignments, roles, declared, receipt)
