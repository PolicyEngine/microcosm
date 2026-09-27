"""Plan one draw over complete supplied survey records, before native Frames.

This is declaration/arithmetic only. Closed source owners must authenticate the
complete inputs and bind the result to the selected population. This module
reads no source, creates no Frame and never totals publisher weights. Original
anchors remain separate from the one-time share/inclusion multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from microcosm.build.survey_domain_sample import select_domain_households

from . import survey_population_domains as domains

PROTOCOL = "microcosm.us.survey-catalogue-selection.v1"
_BATCH_HOUSEHOLDS = 10_000
_BATCH_PEOPLE = 100_000


class CatalogueSelectionError(ValueError):
    """Static refusal, without raw source identities or observations."""


def _require(condition, code):
    if not condition:
        raise CatalogueSelectionError(code)


@dataclass(frozen=True, slots=True)
class SelectedHousehold:
    key: domains.HouseholdKey
    domain: domains.Domain
    statistical_unit: str
    original_design_weight: Fraction
    share: Fraction
    inclusion_probability: Fraction

    @property
    def importance_multiplier(self) -> Fraction:
        return self.share / self.inclusion_probability


@dataclass(frozen=True, slots=True)
class ExcludedHousehold:
    key: domains.HouseholdKey
    reason: str


@dataclass(frozen=True, slots=True)
class SelectionCell:
    source: domains.Source
    domain: domains.Domain
    statistical_unit: str
    share: Fraction
    eligible_households: int
    selected_households: int
    inclusion_probability: Fraction | None


@dataclass(frozen=True, slots=True)
class CatalogueSelectionPlan:
    selected: tuple[SelectedHousehold, ...]
    excluded: tuple[ExcludedHousehold, ...]
    cells: tuple[SelectionCell, ...]
    supplied_households: int
    fraction: Fraction
    seed: int

    @property
    def source_authenticated(self) -> bool:
        return False

    @property
    def population_binding_authenticated(self) -> bool:
        return False

    @property
    def release_eligible(self) -> bool:
        return False


def _sampling_key(key):
    # Keep the original H_SEQ token in the result, but order ASEC keys
    # numerically. All validated values fit five digits; padded strings allow
    # both survey channels to use the shared selector's one literal-ID axis.
    return (
        key.native_id
        if key.source is domains.Source.ACS
        else f"{int(key.native_id):05d}"
    )


def _ordered_key(key):
    return key.source.value, _sampling_key(key)


def _original_anchor(decision):
    row = decision.original
    if row.key.source is domains.Source.ASEC:
        return Fraction(int(row.hsup_wgt), 100)
    if decision.statistical_unit == "person":
        return Fraction(int(row.persons[0].pwgtp))
    return Fraction(int(row.wgtp))


def plan_catalogue_selection(*, acs_households, asec_households, fraction, seed):
    """Classify all supplied records and draw once in each positive-share cell.

    ``N`` counts complete eligible households in the supplied catalogue, never
    selected Frame rows. Unknown classifications refuse before any draw, even
    if a tentative share is zero. Known structural exclusions have a separate
    ledger. Original zero ASEC anchors remain eligible; an all-zero selected
    cell refuses without redrawing. No common-total normalization is performed.

    Full source completeness, immutable source authority, and original files
    are obligations of the calling source orchestration, not this value API.
    """
    _require(
        type(acs_households) is tuple and type(asec_households) is tuple,
        "CATALOGUE_TUPLES",
    )
    _require(type(fraction) is Fraction and 0 < fraction <= 1, "FRACTION")
    _require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    _require(
        type(_BATCH_HOUSEHOLDS) is int
        and 0 < _BATCH_HOUSEHOLDS <= min(10_000, domains.MAX_HOUSEHOLDS)
        and type(_BATCH_PEOPLE) is int
        and 0 < _BATCH_PEOPLE <= min(100_000, domains.MAX_TOTAL_MEMBERS),
        "BATCH_LIMITS",
    )
    rows, exclusions = [], []
    household_keys, person_keys = set(), set()
    batch, batch_people = [], 0

    def consume():
        for decision in domains.classify_households(tuple(batch)):
            _require(
                decision.status is not domains.Status.REVIEW_REQUIRED,
                "SOURCE_CLASSIFICATION_REVIEW_REQUIRED",
            )
            if decision.status is domains.Status.EXCLUDED:
                _require(decision.share == 0, "EXCLUSION_SHARE")
                exclusions.append(
                    ExcludedHousehold(decision.original.key, decision.reason)
                )
            else:
                _require(
                    decision.status is domains.Status.ELIGIBLE
                    and decision.share is not None
                    and decision.share > 0
                    and decision.domain is not None,
                    "ELIGIBLE_CLASSIFICATION",
                )
                # Retain only the compact household decision, not another full
                # population of per-person decisions and observation objects.
                rows.append(
                    (
                        decision.original.key,
                        decision.domain,
                        decision.statistical_unit,
                        _original_anchor(decision),
                        decision.share,
                    )
                )

    for source, expected, catalogue in (
        (domains.Source.ACS, domains.AcsHousehold, acs_households),
        (domains.Source.ASEC, domains.AsecHousehold, asec_households),
    ):
        for row in catalogue:
            _require(type(row) is expected, "CATALOGUE_SOURCE_TYPE")
            canonical = domains._key(row.key)
            _require(
                canonical[0] is source
                and row.key.source_year == 2024
                and row.key.survey_year
                == (2024 if source is domains.Source.ACS else 2025),
                "CATALOGUE_PERIOD_SOURCE",
            )
            _require(canonical not in household_keys, "GLOBAL_HOUSEHOLD_COLLISION")
            household_keys.add(canonical)
            _require(type(row.persons) is tuple, "MEMBER_TUPLE")
            _require(
                len(row.persons) <= min(domains.MAX_MEMBERS, _BATCH_PEOPLE),
                "HOUSEHOLD_MEMBER_BOUND",
            )
            if source is domains.Source.ASEC:
                for person in row.persons:
                    _require(type(person) is domains.AsecPerson, "PERSON_TYPE")
                    identity = (2024, person.peridnum)
                    _require(identity not in person_keys, "GLOBAL_PERSON_COLLISION")
                    person_keys.add(identity)
            if batch and (
                len(batch) == _BATCH_HOUSEHOLDS
                or batch_people + len(row.persons) > _BATCH_PEOPLE
            ):
                consume()
                batch, batch_people = [], 0
            batch.append(row)
            batch_people += len(row.persons)
    if batch:
        consume()

    declarations = {
        (item.domain.value, source.value): (item.statistical_unit, share)
        for item in domains.declaration()
        for source, share in (
            (domains.Source.ACS, item.acs_share),
            (domains.Source.ASEC, item.asec_share),
        )
        if share > 0
    }
    selection = select_domain_households(
        row_ids=tuple(_sampling_key(row[0]) for row in rows),
        source_channels=tuple(row[0].source.value for row in rows),
        domain_keys=tuple(row[1].value for row in rows),
        cells=tuple(declarations),
        fraction=fraction,
        seed=seed,
    )
    chosen, cells = [], []
    for cell in selection:
        unit, share = declarations[cell.domain, cell.source]
        selected = [rows[position] for position in cell.positions]
        _require(
            not selected or any(row[3] > 0 for row in selected),
            "SELECTED_CELL_HAS_NO_POSITIVE_ANCHOR",
        )
        for key, domain, observed_unit, anchor, observed_share in selected:
            _require(
                (observed_unit, observed_share) == (unit, share), "DECLARATION_DRIFT"
            )
            chosen.append(
                SelectedHousehold(
                    key, domain, unit, anchor, share, cell.inclusion_probability
                )
            )
        cells.append(
            SelectionCell(
                domains.Source(cell.source),
                domains.Domain(cell.domain),
                unit,
                share,
                cell.eligible_households,
                len(selected),
                cell.inclusion_probability,
            )
        )
    return CatalogueSelectionPlan(
        tuple(sorted(chosen, key=lambda row: _ordered_key(row.key))),
        tuple(sorted(exclusions, key=lambda row: _ordered_key(row.key))),
        tuple(cells),
        len(household_keys),
        fraction,
        seed,
    )
