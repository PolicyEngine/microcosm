"""US local-jurisdiction flags derived from household geography.

policyengine-us 2.2.1 declares a few local-tax jurisdictions as plain inputs
with no formula, so they default to ``False`` unless the dataset writes them:

- ``in_denver`` (household): read by the Denver property-tax-relief
  eligibility variables;
- ``in_yonkers`` (household): read by ``ny_yonkers_income_tax`` (the resident
  surcharge);
- ``in_wilmington`` (household): read by ``de_wilmington_earned_income_tax``;
- ``pa_philadelphia_wage_tax_resident`` (person): read by
  ``pa_philadelphia_wage_tax`` to pick the resident rate.

Local flags with a ``county_str`` formula (``in_nyc``, ``in_la`` and the
rest) need nothing: the engine computes them from ``county_fips``.

The packaged registry ``us_local_jurisdictions.json`` maps each input-only
flag to one 2020 geography, and its SHA-256 is pinned here:

- rule ``county``: the flag is true exactly when ``county_fips`` equals the
  registry county (Denver ``08031``; Philadelphia ``42101``);
- rule ``place``: the flag is true exactly when the household's state (the
  first two digits of ``county_fips``) and ``place_fips`` equal the registry
  state and 2020 place code (Yonkers city ``36``/``84000``; Wilmington city
  ``10``/``77580``). Place codes are unique only within a state, so the state
  is part of the match.

A geography line derives the flags from the rungs it actually writes. The
block ladder (and the block-location draw) writes county and place from one
block, so every flag is derived. The PUMA ladder writes a county but no
place, so the place-rule flags are *unavailable* there: the line never writes
them (any stale column is removed) and its receipt records them as
unavailable with the reason, rather than writing ``False``.

An artifact built before the flags existed carries none, so a step that
exports or recalibrates an existing artifact instead of running a geography
line calls :func:`rederive_us_local_jurisdiction_flags`. It derives every
flag the artifact's own geography decides and refuses a carried flag that
disagrees with that geography. Both ladder gates fail a table that lacks a
flag its geography decides.

A residence flag alone moves no tax: ``pa_philadelphia_wage_tax`` returns 0
unless the person input ``pa_philadelphia_wage_tax_taxable_wages`` is
nonzero. The second packaged registry, ``us_local_tax_inputs.json`` (its
SHA-256 also pinned here), records how each local-tax input that depends on
work, wages or a claim is provisioned:

- provision ``resident_wages``: a resident's input is
  ``max(employment_income_before_lsr, 0)``, since a resident is taxed on all
  compensation wherever earned; every other person holds 0, the engine
  default, because the nonresident (work-location) component has no source;
- provision ``unavailable``: no source decides the input at microcosm's
  geography (a nonresident's earnings in Yonkers or Wilmington, Denver
  occupational-privilege-tax months, the petition-claimed Philadelphia
  reduced rate). It is never written, and the receipt says why.

A line writes these with :func:`with_us_local_tax_inputs` once both the
residence flag and the wages are final, which is after the wage transfers
rather than at the geography step on the stacked pool.
:func:`rederive_us_local_jurisdiction_flags` re-derives them with the flags.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame

US_LOCAL_JURISDICTION_REGISTRY_RESOURCE = "us_local_jurisdictions.json"
#: SHA-256 of the packaged registry; :func:`packaged_us_local_jurisdiction_registry`
#: refuses any other bytes.
US_LOCAL_JURISDICTION_REGISTRY_SHA256 = (
    "45035eee30beddc0bccceadc7a00363635b0c9012662bf722435ee83243a053c"
)
US_LOCAL_JURISDICTION_REGISTRY_KIND = "us_local_jurisdiction_registry"
US_LOCAL_JURISDICTION_REGISTRY_SCHEMA_VERSION = 1

#: Geography rungs a line can supply. ``county`` needs ``county_fips``;
#: ``place`` needs ``place_fips`` (and the county's state prefix).
JURISDICTION_RUNG_COUNTY = "county"
JURISDICTION_RUNG_PLACE = "place"
JURISDICTION_RUNGS = (JURISDICTION_RUNG_COUNTY, JURISDICTION_RUNG_PLACE)
#: What the block ladder and the block-location draw write.
BLOCK_GEOGRAPHY_RUNGS = JURISDICTION_RUNGS
#: What the PUMA ladder writes.
PUMA_GEOGRAPHY_RUNGS = (JURISDICTION_RUNG_COUNTY,)

#: Receipt statuses.
JURISDICTION_FLAG_DERIVED = "derived"
JURISDICTION_FLAG_UNAVAILABLE = "unavailable"

US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE = "us_local_tax_inputs.json"
#: SHA-256 of the packaged local-tax input registry;
#: :func:`packaged_us_local_tax_input_registry` refuses any other bytes.
US_LOCAL_TAX_INPUT_REGISTRY_SHA256 = (
    "e5592932856d9de3815d97890b7839e28a106a451c544c7f503b3a72da2871d6"
)
US_LOCAL_TAX_INPUT_REGISTRY_KIND = "us_local_tax_input_registry"
US_LOCAL_TAX_INPUT_REGISTRY_SCHEMA_VERSION = 1

#: Local-tax input provisions.
LOCAL_TAX_INPUT_RESIDENT_WAGES = "resident_wages"
LOCAL_TAX_INPUT_UNAVAILABLE = "unavailable"
LOCAL_TAX_INPUT_PROVISIONS = (
    LOCAL_TAX_INPUT_RESIDENT_WAGES,
    LOCAL_TAX_INPUT_UNAVAILABLE,
)
#: Why an unavailable input has no source.
LOCAL_TAX_INPUT_UNAVAILABLE_BASES = ("refund_claim", "work_location")
#: What a resident's null wage yields: refuse it (every finished line), or
#: leave the input null so a nullable staging artifact's reviewed-null fill
#: gives it the engine default along with the wage itself.
NULL_WAGES_REFUSE = "refuse"
NULL_WAGES_PROPAGATE = "propagate"
NULL_WAGE_MODES = (NULL_WAGES_REFUSE, NULL_WAGES_PROPAGATE)

_ENTITIES = ("household", "person")
_REQUIRED_JURISDICTION_KEYS = (
    "flag",
    "entity",
    "jurisdiction",
    "rule",
    "state_fips",
    "population_2020",
)


@dataclass(frozen=True)
class UsLocalJurisdiction:
    """One input-only engine flag and the 2020 geography that decides it.

    Attributes:
        flag: The policyengine-us input variable.
        entity: ``household`` or ``person`` (a person flag is the person's
            household's value).
        jurisdiction: Human-readable name.
        rule: ``county`` or ``place``; the rung the flag needs.
        state_fips: Two-digit state FIPS.
        county_fips: Five-digit county FIPS (rule ``county``).
        place_fips: Five-digit 2020 place code within ``state_fips``
            (rule ``place``).
        coextensive_place_fips: For a county-rule jurisdiction whose city is
            the same area as the county, that city's place code; the
            block-ladder check requires the two block sets to be identical.
        population_2020: 2020 P.L. 94-171 population (``P1_001N``) of the
            jurisdiction; the block-ladder check requires the ladder's blocks
            to sum to it.
    """

    flag: str
    entity: str
    jurisdiction: str
    rule: str
    state_fips: str
    county_fips: str | None
    place_fips: str | None
    coextensive_place_fips: str | None
    population_2020: int

    @property
    def geoid(self) -> str:
        """The deciding geography's 2020 GEOID (county 5 digits, place 7)."""

        if self.rule == JURISDICTION_RUNG_COUNTY:
            assert self.county_fips is not None
            return self.county_fips
        assert self.place_fips is not None
        return self.state_fips + self.place_fips


@dataclass(frozen=True)
class UsLocalJurisdictionRegistry:
    """The parsed registry plus the SHA-256 of the bytes it came from."""

    jurisdictions: tuple[UsLocalJurisdiction, ...]
    sha256: str
    metadata: Mapping[str, Any]

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(item.flag for item in self.jurisdictions)

    def for_entity(self, entity: str) -> tuple[UsLocalJurisdiction, ...]:
        return tuple(item for item in self.jurisdictions if item.entity == entity)


def load_us_local_jurisdiction_registry(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> UsLocalJurisdictionRegistry:
    """Load and validate a registry file, optionally pinning its SHA-256."""

    payload = Path(path).read_bytes()
    return _registry_from_bytes(payload, expected_sha256=expected_sha256)


@cache
def packaged_us_local_jurisdiction_registry() -> UsLocalJurisdictionRegistry:
    """The packaged registry, refused unless its bytes match the pinned SHA-256."""

    payload = (
        files("microcosm.build.us_runtime")
        .joinpath(US_LOCAL_JURISDICTION_REGISTRY_RESOURCE)
        .read_bytes()
    )
    return _registry_from_bytes(
        payload, expected_sha256=US_LOCAL_JURISDICTION_REGISTRY_SHA256
    )


def _registry_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str | None,
) -> UsLocalJurisdictionRegistry:
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "US local-jurisdiction registry bytes do not match the pinned "
            f"SHA-256: expected {expected_sha256}, observed {digest}."
        )
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise ValueError("US local-jurisdiction registry must be a JSON object.")
    if document.get("kind") != US_LOCAL_JURISDICTION_REGISTRY_KIND:
        raise ValueError(
            "US local-jurisdiction registry kind must be "
            f"{US_LOCAL_JURISDICTION_REGISTRY_KIND!r}, got {document.get('kind')!r}."
        )
    if document.get("schema_version") != US_LOCAL_JURISDICTION_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "US local-jurisdiction registry schema_version must be "
            f"{US_LOCAL_JURISDICTION_REGISTRY_SCHEMA_VERSION}, got "
            f"{document.get('schema_version')!r}."
        )
    raw = document.get("jurisdictions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("US local-jurisdiction registry has no jurisdictions.")
    jurisdictions = tuple(_jurisdiction(item, index) for index, item in enumerate(raw))
    flags = [item.flag for item in jurisdictions]
    if len(set(flags)) != len(flags):
        raise ValueError(f"US local-jurisdiction registry repeats a flag: {flags}.")
    if flags != sorted(flags):
        raise ValueError(
            f"US local-jurisdiction registry flags must be sorted: {flags}."
        )
    geoids = [item.geoid for item in jurisdictions]
    if len(set(geoids)) != len(geoids):
        raise ValueError(
            f"US local-jurisdiction registry repeats a geography: {geoids}."
        )
    metadata = {key: value for key, value in document.items() if key != "jurisdictions"}
    return UsLocalJurisdictionRegistry(
        jurisdictions=jurisdictions, sha256=digest, metadata=metadata
    )


def _jurisdiction(item: object, index: int) -> UsLocalJurisdiction:
    label = f"jurisdictions[{index}]"
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} must be an object.")
    missing = [key for key in _REQUIRED_JURISDICTION_KEYS if key not in item]
    if missing:
        raise ValueError(f"{label} is missing {missing}.")
    flag = item["flag"]
    if not isinstance(flag, str) or not flag.isidentifier() or flag != flag.lower():
        raise ValueError(f"{label}.flag must be a lowercase identifier: {flag!r}.")
    entity = item["entity"]
    if entity not in _ENTITIES:
        raise ValueError(f"{label}.entity must be one of {_ENTITIES}: {entity!r}.")
    rule = item["rule"]
    if rule not in JURISDICTION_RUNGS:
        raise ValueError(f"{label}.rule must be one of {JURISDICTION_RUNGS}.")
    state = _digits(item["state_fips"], 2, label=f"{label}.state_fips")
    county = _optional_digits(item.get("county_fips"), 5, label=f"{label}.county_fips")
    place = _optional_digits(item.get("place_fips"), 5, label=f"{label}.place_fips")
    coextensive = _optional_digits(
        item.get("coextensive_place_fips"),
        5,
        label=f"{label}.coextensive_place_fips",
    )
    if rule == JURISDICTION_RUNG_COUNTY:
        if county is None or place is not None:
            raise ValueError(
                f"{label}: a county rule names county_fips and no place_fips."
            )
        if county[:2] != state:
            raise ValueError(f"{label}: county_fips {county} is not in state {state}.")
    else:
        if place is None or county is not None or coextensive is not None:
            raise ValueError(
                f"{label}: a place rule names place_fips and no county fields."
            )
    population = item["population_2020"]
    if isinstance(population, bool) or not isinstance(population, int):
        raise ValueError(f"{label}.population_2020 must be an integer.")
    if population <= 0:
        raise ValueError(f"{label}.population_2020 must be positive.")
    name = item["jurisdiction"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.jurisdiction must be a non-empty string.")
    return UsLocalJurisdiction(
        flag=flag,
        entity=entity,
        jurisdiction=name,
        rule=rule,
        state_fips=state,
        county_fips=county,
        place_fips=place,
        coextensive_place_fips=coextensive,
        population_2020=population,
    )


def _digits(value: object, width: int, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != width or not value.isdigit():
        raise ValueError(f"{label} must be a {width}-digit string, got {value!r}.")
    return value


def _optional_digits(value: object, width: int, *, label: str) -> str | None:
    return None if value is None else _digits(value, width, label=label)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def _normalized_rungs(rungs: Collection[str]) -> tuple[str, ...]:
    unknown = sorted(set(rungs) - set(JURISDICTION_RUNGS))
    if unknown:
        raise ValueError(
            f"Unknown local-jurisdiction rung(s) {unknown}; expected a subset of "
            f"{JURISDICTION_RUNGS}."
        )
    return tuple(rung for rung in JURISDICTION_RUNGS if rung in set(rungs))


def _text_codes(
    values: pd.Series, *, column: str, widths: tuple[int, ...]
) -> pd.Series:
    """Return ``values`` as text codes, refusing nulls, non-text and bad widths.

    An integer code is refused rather than zero-filled: the geography lines
    write text, and a numeric column here means something upstream changed.
    """

    series = pd.Series(values.to_numpy(dtype=object), dtype=object)
    if series.empty:
        return series
    kind = pd.api.types.infer_dtype(series, skipna=False)
    if kind == "bytes":
        series = series.str.decode("ascii")
        kind = pd.api.types.infer_dtype(series, skipna=False)
    allowed = " or ".join(
        "empty" if width == 0 else f"{width}-digit" for width in widths
    )
    if kind != "string":
        examples = (
            series[~series.map(lambda value: isinstance(value, str))].head(5).tolist()
        )
        raise ValueError(
            f"{column} must hold {allowed} text codes on every row to derive local "
            f"jurisdiction flags; it holds {kind} values, e.g. {examples}."
        )
    pattern = "|".join(rf"\d{{{width}}}" if width else "" for width in widths)
    valid = series.str.fullmatch(f"(?:{pattern})").astype(bool)
    if not valid.all():
        raise ValueError(
            f"{column} must hold {allowed} text codes on every row to derive local "
            f"jurisdiction flags; invalid value(s) {series[~valid].head(5).tolist()}."
        )
    return series


def _holds_text_codes(
    values: pd.Series, *, column: str, widths: tuple[int, ...]
) -> bool:
    try:
        _text_codes(values, column=column, widths=widths)
    except ValueError:
        return False
    return True


def us_local_jurisdiction_rungs_present(household: pd.DataFrame) -> tuple[str, ...]:
    """The rungs a finished household table's own geography supplies.

    For a step that starts from an artifact rather than a geography line. The
    county rung needs ``county_fips`` on every row. The place rung also needs
    every row to sit on a block: ``place_fips`` with no null row and a
    15-digit ``block_geoid`` on every row. A pooled table whose ACS rows have
    no block therefore supplies only the county rung, even where a writer has
    stored their missing place as an empty string. A malformed
    ``county_fips`` is not judged here; deriving the flags raises on it.
    """

    if "county_fips" not in household or household["county_fips"].isna().any():
        return ()
    block_anchored = (
        "place_fips" in household
        and not household["place_fips"].isna().any()
        and "block_geoid" in household
        and _holds_text_codes(
            household["block_geoid"], column="block_geoid", widths=(15,)
        )
    )
    return BLOCK_GEOGRAPHY_RUNGS if block_anchored else PUMA_GEOGRAPHY_RUNGS


def us_local_jurisdiction_flag_values(
    household: pd.DataFrame,
    *,
    rungs: Collection[str],
    registry: UsLocalJurisdictionRegistry | None = None,
) -> dict[str, np.ndarray]:
    """Per-household flag arrays for every registry flag ``rungs`` can decide.

    Returns one boolean array per derivable flag, aligned to ``household``'s
    rows, whatever the flag's entity (a person flag is its household's value).
    A flag whose rule is not in ``rungs`` is left out. Missing, null or
    malformed geography on a supplied rung raises: a household whose county or
    place is unknown cannot be decided, and writing ``False`` for it would be
    a silent guess.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    supplied = _normalized_rungs(rungs)
    needed = {item.rule for item in resolved.jurisdictions} & set(supplied)
    if not needed:
        return {}
    for column in (
        "county_fips",
        *(("place_fips",) if JURISDICTION_RUNG_PLACE in needed else ()),
    ):
        if column not in household:
            raise ValueError(
                f"household table has no {column!r}; it cannot supply the "
                f"local-jurisdiction rung(s) {sorted(needed)}."
            )
    county = _text_codes(household["county_fips"], column="county_fips", widths=(5,))
    place = (
        _text_codes(household["place_fips"], column="place_fips", widths=(0, 5))
        if JURISDICTION_RUNG_PLACE in needed
        else None
    )
    flags: dict[str, np.ndarray] = {}
    for item in resolved.jurisdictions:
        if item.rule not in needed:
            continue
        if item.rule == JURISDICTION_RUNG_COUNTY:
            mask = county.eq(item.county_fips)
        else:
            assert place is not None
            mask = county.str[:2].eq(item.state_fips) & place.eq(item.place_fips)
        flags[item.flag] = mask.to_numpy(dtype=bool)
    return flags


def apply_us_local_jurisdiction_flags(
    tables: Mapping[str, pd.DataFrame],
    *,
    rungs: Collection[str],
    registry: UsLocalJurisdictionRegistry | None = None,
    household_weights: np.ndarray | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Write every derivable flag and return the tables with a receipt.

    Household flags go on the household table; person flags on the person
    table, each person taking its household's value through
    ``person_household_id``. A flag ``rungs`` cannot decide is removed from
    its table if a stale or partially populated column is present, and the
    receipt records it as unavailable: the engine default then applies to
    every row, and the receipt says so, instead of a written ``False``.

    The household and person tables are returned as new (shallow-copied)
    frames; the input tables are never modified, and the other tables are
    passed through as-is.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    supplied = _normalized_rungs(rungs)
    household = tables["household"]
    values = us_local_jurisdiction_flag_values(
        household, rungs=supplied, registry=resolved
    )
    out = dict(tables)
    for entity in {item.entity for item in resolved.jurisdictions}:
        out[entity] = tables[entity].copy(deep=False)
    person_rows: np.ndarray | None = None
    if any(
        item.entity == "person" and item.flag in values
        for item in resolved.jurisdictions
    ):
        person_rows = _person_household_positions(out["household"], out["person"])
    for item in resolved.jurisdictions:
        table = out[item.entity]
        if item.flag not in values:
            if item.flag in table:
                out[item.entity] = table.drop(columns=[item.flag])
            continue
        household_values = values[item.flag]
        if item.entity == "household":
            table[item.flag] = household_values
        else:
            assert person_rows is not None
            table[item.flag] = household_values[person_rows]
    receipt = us_local_jurisdiction_flag_receipt(
        values,
        rungs=supplied,
        registry=resolved,
        household_weights=household_weights,
    )
    return out, receipt


def with_us_local_jurisdiction_flags(
    frame: Frame,
    *,
    rungs: Collection[str],
    registry: UsLocalJurisdictionRegistry | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """Return ``frame`` with its local-jurisdiction flags and the receipt.

    The one call every geography line makes once its county (and, where it
    has one, place) is final: ``rungs`` is :data:`BLOCK_GEOGRAPHY_RUNGS` for
    a block-anchored line and :data:`PUMA_GEOGRAPHY_RUNGS` for the PUMA
    ladder.
    """

    tables = {entity: frame.table(entity) for entity in frame.entities}
    written, receipt = apply_us_local_jurisdiction_flags(
        tables,
        rungs=rungs,
        registry=registry,
        household_weights=frame.weights_for("household").values,
    )
    for link_name in frame.links:
        written[link_name] = frame.link(link_name).copy()
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return (
        Frame(
            written,
            frame.schema,
            weights,
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        ),
        receipt,
    )


def rederive_us_local_jurisdiction_flags(
    frame: Frame,
    *,
    registry: UsLocalJurisdictionRegistry | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """Derive an existing artifact's flags from the geography it carries.

    For a step that exports or recalibrates an artifact rather than running a
    geography line. A base H5 built before the flags existed carries none,
    and exporting it as-is would leave the engine default (``False``) on
    every resident. The rungs come from the artifact's own geography
    (:func:`us_local_jurisdiction_rungs_present`). A flag the artifact
    already carries must agree with that geography, or this raises:
    re-deriving it would hide a corrupt artifact.

    The local-tax inputs are then derived from the re-derived flags and the
    artifact's wages (:func:`with_us_local_tax_inputs`); a carried input that
    disagrees with the artifact's own flags and wages is refused the same way.

    The receipt adds ``carried_flags``, the flags the artifact already held,
    ``local_tax_inputs``, the local-tax input receipt, and
    ``carried_local_tax_inputs``, the inputs the artifact already held.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    household = frame.table("household")
    failures = us_local_jurisdiction_flag_failures(
        household, person=frame.table("person"), registry=resolved
    )
    if failures:
        raise ValueError(
            "The artifact carries local-jurisdiction flags that disagree with "
            "its own geography: " + "; ".join(failures)
        )
    failures = us_local_tax_input_failures(
        frame.table("person"), jurisdictions=resolved
    )
    if failures:
        raise ValueError(
            "The artifact carries local-tax inputs that disagree with its own "
            "residence flags and wages: " + "; ".join(failures)
        )
    carried = sorted(
        item.flag
        for item in resolved.jurisdictions
        if item.flag in frame.table(item.entity)
    )
    derived, receipt = with_us_local_jurisdiction_flags(
        frame,
        rungs=us_local_jurisdiction_rungs_present(household),
        registry=resolved,
    )
    receipt["carried_flags"] = carried
    carried_tax_inputs = sorted(
        item.variable
        for item in packaged_us_local_tax_input_registry().inputs
        if item.variable in frame.table(item.entity)
    )
    derived, tax_input_receipt = with_us_local_tax_inputs(
        derived, jurisdictions=resolved
    )
    receipt["local_tax_inputs"] = tax_input_receipt
    receipt["carried_local_tax_inputs"] = carried_tax_inputs
    return derived, receipt


def _person_household_positions(
    household: pd.DataFrame, person: pd.DataFrame
) -> np.ndarray:
    id_column = US_SCHEMA.entity_id_column("household")
    membership = US_SCHEMA.membership_column("household")
    for table, column in ((household, id_column), (person, membership)):
        if column not in table:
            raise ValueError(
                f"Person local-jurisdiction flags need {column!r}; it is missing."
            )
    ids = pd.Index(household[id_column].to_numpy())
    if not ids.is_unique:
        raise ValueError("household_id must be unique to broadcast person flags.")
    positions = ids.get_indexer(person[membership].to_numpy())
    if (positions < 0).any():
        raise ValueError(
            "Every person must belong to a household in the table to receive "
            "its local-jurisdiction flags."
        )
    return positions


def us_local_jurisdiction_flag_receipt(
    values: Mapping[str, np.ndarray],
    *,
    rungs: Collection[str],
    registry: UsLocalJurisdictionRegistry | None = None,
    household_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """JSON-ready provenance: registry digest, rungs, and each flag's status."""

    resolved = registry or packaged_us_local_jurisdiction_registry()
    supplied = _normalized_rungs(rungs)
    weights = (
        None
        if household_weights is None
        else np.asarray(household_weights, dtype=np.float64)
    )
    flags: dict[str, Any] = {}
    for item in resolved.jurisdictions:
        entry: dict[str, Any] = {
            "entity": item.entity,
            "rule": item.rule,
            "geoid": item.geoid,
        }
        if item.flag in values:
            mask = np.asarray(values[item.flag], dtype=bool)
            entry["status"] = JURISDICTION_FLAG_DERIVED
            entry["households_in_jurisdiction"] = int(mask.sum())
            if weights is not None:
                if len(weights) != len(mask):
                    raise ValueError(
                        "household_weights must align with the household table."
                    )
                entry["weighted_households_in_jurisdiction"] = float(
                    weights[mask].sum()
                )
        else:
            entry["status"] = JURISDICTION_FLAG_UNAVAILABLE
            entry["reason"] = (
                f"the line supplies no {item.rule} rung; the flag is not written, "
                "so the engine default (False) applies to every row"
            )
        flags[item.flag] = entry
    return {
        "registry": US_LOCAL_JURISDICTION_REGISTRY_RESOURCE,
        "registry_sha256": resolved.sha256,
        "rungs": list(supplied),
        "flags": flags,
    }


def us_local_jurisdiction_flag_failures(
    household: pd.DataFrame,
    *,
    person: pd.DataFrame | None = None,
    required_rungs: Collection[str] = (),
    registry: UsLocalJurisdictionRegistry | None = None,
) -> list[str]:
    """Gate check: every flag column present agrees with the table's geography.

    A flag present must equal its rule applied to the table's own
    ``county_fips``/``place_fips``. A place-rule flag on a table with no
    complete ``place_fips`` fails: nothing could have decided it. A flag
    whose rule is in ``required_rungs`` must also be present: a gate whose
    line supplies that rung passes ``required_rungs``, so an artifact built
    before the flags existed fails rather than leaving the engine default on
    every resident.

    Person flags are checked only when ``person`` is given; each person must
    carry its household's value.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    required = set(_normalized_rungs(required_rungs))
    tables = {"household": household, "person": person}
    failures: list[str] = []
    for item in resolved.jurisdictions:
        table = tables[item.entity]
        if table is None:
            continue
        if item.flag not in table:
            if item.rule in required:
                failures.append(
                    f"{item.flag}: absent, but this table's {item.rule} rung "
                    "decides it; the engine would default every row to False"
                )
            continue
        try:
            expected = us_local_jurisdiction_flag_values(
                household, rungs=(item.rule,), registry=resolved
            )[item.flag]
            if item.entity == "person":
                expected = expected[_person_household_positions(household, table)]
        except ValueError as error:
            failures.append(
                f"{item.flag}: present but its {item.rule} rung cannot decide it ({error})"
            )
            continue
        observed = table[item.flag]
        if observed.isna().any():
            failures.append(f"{item.flag}: {int(observed.isna().sum())} null row(s)")
            continue
        if (
            not pd.api.types.is_bool_dtype(observed)
            and not observed.isin([True, False]).all()
        ):
            failures.append(f"{item.flag}: holds non-boolean values")
            continue
        disagree = observed.astype(bool).to_numpy() != expected
        if disagree.any():
            failures.append(
                f"{item.flag}: {int(disagree.sum())}/{len(table)} {item.entity} "
                f"row(s) disagree with the {item.rule} rule for {item.geoid}"
            )
    return failures


# ---------------------------------------------------------------------------
# Local-tax inputs
# ---------------------------------------------------------------------------

_REQUIRED_TAX_INPUT_KEYS = (
    "variable",
    "entity",
    "jurisdiction",
    "provision",
    "engine_consumers",
    "requires",
    "sources",
)
_RESIDENT_WAGES_KEYS = (
    "residence_flag",
    "wage_input",
    "resident_rule",
    "nonresident_reason",
)
_UNAVAILABLE_KEYS = ("basis", "reason")


@dataclass(frozen=True)
class UsLocalTaxInput:
    """One policyengine-us local-tax input and how microcosm provisions it.

    Attributes:
        variable: The policyengine-us input variable.
        entity: ``person`` or ``household``; a ``resident_wages`` input is a
            person input.
        jurisdiction: Human-readable name.
        provision: ``resident_wages`` or ``unavailable``.
        basis: For ``unavailable``, what no source supplies
            (``work_location`` or ``refund_claim``).
        residence_flag: For ``resident_wages``, the person flag in
            ``us_local_jurisdictions.json`` that marks residents.
        wage_input: For ``resident_wages``, the person wage input.
        reason: For ``unavailable``, why no source decides the input; for
            ``resident_wages``, why its nonresident component is unavailable.
    """

    variable: str
    entity: str
    jurisdiction: str
    provision: str
    basis: str | None
    residence_flag: str | None
    wage_input: str | None
    reason: str


@dataclass(frozen=True)
class UsLocalTaxInputRegistry:
    """The parsed local-tax input registry plus the SHA-256 of its bytes."""

    inputs: tuple[UsLocalTaxInput, ...]
    sha256: str
    metadata: Mapping[str, Any]

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(item.variable for item in self.inputs)

    def with_provision(self, provision: str) -> tuple[UsLocalTaxInput, ...]:
        return tuple(item for item in self.inputs if item.provision == provision)


def load_us_local_tax_input_registry(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> UsLocalTaxInputRegistry:
    """Load and validate a local-tax input registry, optionally pinning its SHA-256."""

    payload = Path(path).read_bytes()
    return _tax_input_registry_from_bytes(payload, expected_sha256=expected_sha256)


@cache
def packaged_us_local_tax_input_registry() -> UsLocalTaxInputRegistry:
    """The packaged local-tax input registry, refused unless its bytes match the pin."""

    payload = (
        files("microcosm.build.us_runtime")
        .joinpath(US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE)
        .read_bytes()
    )
    return _tax_input_registry_from_bytes(
        payload, expected_sha256=US_LOCAL_TAX_INPUT_REGISTRY_SHA256
    )


def us_local_tax_input_output_columns(
    registry: UsLocalTaxInputRegistry | None = None,
) -> dict[str, frozenset[str]]:
    """The columns a local-tax input step writes, by entity.

    Only ``resident_wages`` inputs are ever written; an unavailable input
    never is.
    """

    resolved = registry or packaged_us_local_tax_input_registry()
    columns: dict[str, set[str]] = {}
    for item in resolved.with_provision(LOCAL_TAX_INPUT_RESIDENT_WAGES):
        columns.setdefault(item.entity, set()).add(item.variable)
    return {entity: frozenset(names) for entity, names in columns.items()}


def _tax_input_registry_from_bytes(
    payload: bytes,
    *,
    expected_sha256: str | None,
) -> UsLocalTaxInputRegistry:
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "US local-tax input registry bytes do not match the pinned SHA-256: "
            f"expected {expected_sha256}, observed {digest}."
        )
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise ValueError("US local-tax input registry must be a JSON object.")
    if document.get("kind") != US_LOCAL_TAX_INPUT_REGISTRY_KIND:
        raise ValueError(
            "US local-tax input registry kind must be "
            f"{US_LOCAL_TAX_INPUT_REGISTRY_KIND!r}, got {document.get('kind')!r}."
        )
    if document.get("schema_version") != US_LOCAL_TAX_INPUT_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "US local-tax input registry schema_version must be "
            f"{US_LOCAL_TAX_INPUT_REGISTRY_SCHEMA_VERSION}, got "
            f"{document.get('schema_version')!r}."
        )
    sources = document.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("US local-tax input registry has no sources.")
    raw = document.get("inputs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("US local-tax input registry has no inputs.")
    inputs = tuple(
        _tax_input(item, index, sources=sources) for index, item in enumerate(raw)
    )
    variables = [item.variable for item in inputs]
    if len(set(variables)) != len(variables):
        raise ValueError(
            f"US local-tax input registry repeats a variable: {variables}."
        )
    if variables != sorted(variables):
        raise ValueError(
            f"US local-tax input registry variables must be sorted: {variables}."
        )
    metadata = {key: value for key, value in document.items() if key != "inputs"}
    return UsLocalTaxInputRegistry(inputs=inputs, sha256=digest, metadata=metadata)


def _tax_input(
    item: object, index: int, *, sources: Mapping[str, Any]
) -> UsLocalTaxInput:
    label = f"inputs[{index}]"
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} must be an object.")
    missing = [key for key in _REQUIRED_TAX_INPUT_KEYS if key not in item]
    if missing:
        raise ValueError(f"{label} is missing {missing}.")
    variable = _identifier(item["variable"], label=f"{label}.variable")
    entity = item["entity"]
    if entity not in _ENTITIES:
        raise ValueError(f"{label}.entity must be one of {_ENTITIES}: {entity!r}.")
    provision = item["provision"]
    if provision not in LOCAL_TAX_INPUT_PROVISIONS:
        raise ValueError(
            f"{label}.provision must be one of {LOCAL_TAX_INPUT_PROVISIONS}: "
            f"{provision!r}."
        )
    for key in ("jurisdiction", "requires"):
        if not isinstance(item[key], str) or not item[key]:
            raise ValueError(f"{label}.{key} must be a non-empty string.")
    consumers = item["engine_consumers"]
    if (
        not isinstance(consumers, list)
        or not consumers
        or not all(isinstance(name, str) and name.isidentifier() for name in consumers)
    ):
        raise ValueError(f"{label}.engine_consumers must list engine variables.")
    cited = item["sources"]
    if not isinstance(cited, list) or not cited:
        raise ValueError(f"{label}.sources must name at least one source.")
    unknown = sorted(set(cited) - set(sources))
    if unknown:
        raise ValueError(f"{label}.sources names unknown source(s) {unknown}.")
    own, other = (
        (_RESIDENT_WAGES_KEYS, _UNAVAILABLE_KEYS)
        if provision == LOCAL_TAX_INPUT_RESIDENT_WAGES
        else (_UNAVAILABLE_KEYS, _RESIDENT_WAGES_KEYS)
    )
    missing = [key for key in own if key not in item]
    if missing:
        raise ValueError(f"{label}: a {provision} input is missing {missing}.")
    mixed = [key for key in other if key in item]
    if mixed:
        raise ValueError(f"{label}: a {provision} input may not name {mixed}.")
    for key in own:
        if not isinstance(item[key], str) or not item[key]:
            raise ValueError(f"{label}.{key} must be a non-empty string.")
    if provision == LOCAL_TAX_INPUT_RESIDENT_WAGES:
        if entity != "person":
            raise ValueError(f"{label}: a resident_wages input is a person input.")
        return UsLocalTaxInput(
            variable=variable,
            entity=entity,
            jurisdiction=item["jurisdiction"],
            provision=provision,
            basis=None,
            residence_flag=_identifier(
                item["residence_flag"], label=f"{label}.residence_flag"
            ),
            wage_input=_identifier(item["wage_input"], label=f"{label}.wage_input"),
            reason=item["nonresident_reason"],
        )
    basis = item["basis"]
    if basis not in LOCAL_TAX_INPUT_UNAVAILABLE_BASES:
        raise ValueError(
            f"{label}.basis must be one of {LOCAL_TAX_INPUT_UNAVAILABLE_BASES}: "
            f"{basis!r}."
        )
    return UsLocalTaxInput(
        variable=variable,
        entity=entity,
        jurisdiction=item["jurisdiction"],
        provision=provision,
        basis=basis,
        residence_flag=None,
        wage_input=None,
        reason=item["reason"],
    )


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or value != value.lower():
        raise ValueError(f"{label} must be a lowercase identifier: {value!r}.")
    return value


def _resident_wage_inputs(
    registry: UsLocalTaxInputRegistry,
    jurisdictions: UsLocalJurisdictionRegistry,
) -> tuple[UsLocalTaxInput, ...]:
    """The registry's resident_wages inputs, each tied to a person residence flag."""

    person_flags = {item.flag for item in jurisdictions.for_entity("person")}
    inputs = registry.with_provision(LOCAL_TAX_INPUT_RESIDENT_WAGES)
    unknown = sorted(
        item.residence_flag
        for item in inputs
        if item.residence_flag not in person_flags
    )
    if unknown:
        raise ValueError(
            "Local-tax residence flag(s) "
            f"{unknown} are not person flags in the local-jurisdiction registry."
        )
    return inputs


def _boolean_values(values: pd.Series, *, column: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{column} has {int(values.isna().sum())} null row(s).")
    if not pd.api.types.is_bool_dtype(values) and not values.isin([True, False]).all():
        raise ValueError(f"{column} holds non-boolean values.")
    return values.astype(bool).to_numpy()


def us_resident_wage_tax_values(
    person: pd.DataFrame,
    item: UsLocalTaxInput,
    *,
    null_wages: str = NULL_WAGES_REFUSE,
) -> np.ndarray:
    """A ``resident_wages`` input's values, aligned to ``person``'s rows.

    A resident (``item.residence_flag`` true) gets ``max(wages, 0)``: a
    resident owes the tax on all compensation wherever earned, and a negative
    wage is no compensation. Everyone else gets 0, the engine default, since
    the nonresident component has no source. A missing column, a null or
    non-boolean flag, or an infinite wage raises: writing 0 for a row nothing
    could decide would be a silent guess. A resident's null wage raises too,
    unless ``null_wages`` is ``propagate``: then the resident's value is null,
    to be filled with the engine default exactly where the wage is.
    """

    if item.provision != LOCAL_TAX_INPUT_RESIDENT_WAGES:
        raise ValueError(f"{item.variable} is not a resident_wages input.")
    if null_wages not in NULL_WAGE_MODES:
        raise ValueError(f"null_wages must be one of {NULL_WAGE_MODES}.")
    assert item.residence_flag is not None and item.wage_input is not None
    for column in (item.residence_flag, item.wage_input):
        if column not in person:
            raise ValueError(
                f"person table has no {column!r}; it cannot decide {item.variable}."
            )
    resident = _boolean_values(person[item.residence_flag], column=item.residence_flag)
    wages = pd.to_numeric(person[item.wage_input], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    infinite = np.isinf(wages)
    if infinite.any():
        raise ValueError(
            f"{item.wage_input} has {int(infinite.sum())} infinite row(s); "
            f"{item.variable} cannot be derived for them."
        )
    null_residents = resident & np.isnan(wages)
    if null_residents.any() and null_wages == NULL_WAGES_REFUSE:
        raise ValueError(
            f"{item.wage_input} is null on {int(null_residents.sum())} resident "
            f"row(s); {item.variable} cannot be derived for them."
        )
    return np.where(resident, np.maximum(wages, 0.0), 0.0)


def apply_us_local_tax_inputs(
    tables: Mapping[str, pd.DataFrame],
    *,
    registry: UsLocalTaxInputRegistry | None = None,
    jurisdictions: UsLocalJurisdictionRegistry | None = None,
    household_weights: np.ndarray | None = None,
    null_wages: str = NULL_WAGES_REFUSE,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Write every derivable local-tax input and return the tables with a receipt.

    A ``resident_wages`` input is written whenever its residence flag is on
    the person table. Without the flag the line's geography could not decide
    residence, so the input is not written (a stale column is removed) and
    the receipt records it as unavailable. An ``unavailable`` input is never
    written; a stale column is removed. The engine default then applies to
    every row the receipt marks unavailable. ``null_wages`` is passed to
    :func:`us_resident_wage_tax_values`.

    Tables an input lives on are returned as new (shallow-copied) frames; the
    input tables are never modified, and the other tables pass through.
    """

    resolved = registry or packaged_us_local_tax_input_registry()
    flags = jurisdictions or packaged_us_local_jurisdiction_registry()
    _resident_wage_inputs(resolved, flags)
    out = dict(tables)
    for entity in {item.entity for item in resolved.inputs}:
        out[entity] = tables[entity].copy(deep=False)
    person_weights: np.ndarray | None = None
    entries: dict[str, Any] = {}
    for item in resolved.inputs:
        table = out[item.entity]
        entry: dict[str, Any] = {"entity": item.entity, "provision": item.provision}
        if item.provision == LOCAL_TAX_INPUT_UNAVAILABLE:
            if item.variable in table:
                out[item.entity] = table.drop(columns=[item.variable])
            entry.update(
                {
                    "status": JURISDICTION_FLAG_UNAVAILABLE,
                    "basis": item.basis,
                    "reason": item.reason,
                }
            )
            entries[item.variable] = entry
            continue
        assert item.residence_flag is not None
        entry.update(
            {"residence_flag": item.residence_flag, "wage_input": item.wage_input}
        )
        if item.residence_flag not in table:
            if item.variable in table:
                out[item.entity] = table.drop(columns=[item.variable])
            entry.update(
                {
                    "status": JURISDICTION_FLAG_UNAVAILABLE,
                    "reason": (
                        f"the line wrote no {item.residence_flag}, so residence is "
                        "undecided; the input is not written and the engine "
                        "default (0) applies to every row"
                    ),
                }
            )
            entries[item.variable] = entry
            continue
        values = us_resident_wage_tax_values(table, item, null_wages=null_wages)
        resident = table[item.residence_flag].astype(bool).to_numpy()
        assert item.wage_input is not None
        wages = pd.to_numeric(table[item.wage_input], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        table[item.variable] = values
        entry.update(
            {
                "status": JURISDICTION_FLAG_DERIVED,
                "null_wages": null_wages,
                "residents": int(resident.sum()),
                "residents_with_taxable_wages": int((values > 0).sum()),
                "residents_with_negative_wages_set_to_zero": int(
                    (resident & (wages < 0)).sum()
                ),
                "residents_with_null_wages_left_null": int(np.isnan(values).sum()),
                "nonresident_component": {
                    "status": JURISDICTION_FLAG_UNAVAILABLE,
                    "reason": item.reason,
                },
            }
        )
        if household_weights is not None:
            if person_weights is None:
                weights = np.asarray(household_weights, dtype=np.float64)
                if len(weights) != len(out["household"]):
                    raise ValueError(
                        "household_weights must align with the household table."
                    )
                person_weights = weights[
                    _person_household_positions(out["household"], table)
                ]
            entry["weighted_residents"] = float(person_weights[resident].sum())
            entry["weighted_taxable_wages"] = float(
                person_weights @ np.nan_to_num(values, nan=0.0)
            )
        entries[item.variable] = entry
    receipt = {
        "registry": US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE,
        "registry_sha256": resolved.sha256,
        "jurisdiction_registry_sha256": flags.sha256,
        "inputs": entries,
    }
    return out, receipt


def with_us_local_tax_inputs(
    frame: Frame,
    *,
    registry: UsLocalTaxInputRegistry | None = None,
    jurisdictions: UsLocalJurisdictionRegistry | None = None,
) -> tuple[Frame, dict[str, Any]]:
    """Return ``frame`` with its local-tax inputs and the receipt.

    A line calls this once both the residence flags and the wages are final:
    after the geography step on a line whose wages are already in place, and
    after the wage transfers on a line that draws geography first.
    """

    tables = {entity: frame.table(entity) for entity in frame.entities}
    written, receipt = apply_us_local_tax_inputs(
        tables,
        registry=registry,
        jurisdictions=jurisdictions,
        household_weights=frame.weights_for("household").values,
    )
    for link_name in frame.links:
        written[link_name] = frame.link(link_name).copy()
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return (
        Frame(
            written,
            frame.schema,
            weights,
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        ),
        receipt,
    )


def us_local_tax_input_failures(
    person: pd.DataFrame,
    *,
    require_derivable: bool = False,
    registry: UsLocalTaxInputRegistry | None = None,
    jurisdictions: UsLocalJurisdictionRegistry | None = None,
    null_wages: str = NULL_WAGES_REFUSE,
) -> list[str]:
    """Gate check: every local-tax input present agrees with the table.

    An ``unavailable`` input must be absent: nothing could have decided it.
    A ``resident_wages`` input present must equal its rule applied to the
    table's own residence flag and wages, which must both be present. With
    ``require_derivable``, a ``resident_wages`` input whose residence flag is
    present must also be present, so an artifact built before the input
    existed fails rather than leaving the engine default on every resident.
    With ``null_wages`` ``propagate``, a resident's null wage must carry a
    null input; otherwise any null fails.
    """

    resolved = registry or packaged_us_local_tax_input_registry()
    flags = jurisdictions or packaged_us_local_jurisdiction_registry()
    _resident_wage_inputs(resolved, flags)
    failures: list[str] = []
    for item in resolved.inputs:
        if item.entity != "person":
            continue
        present = item.variable in person
        if item.provision == LOCAL_TAX_INPUT_UNAVAILABLE:
            if present:
                failures.append(
                    f"{item.variable}: present, but no source decides it ({item.basis})"
                )
            continue
        if not present:
            if require_derivable and item.residence_flag in person:
                failures.append(
                    f"{item.variable}: absent, but {item.residence_flag} is present "
                    "and decides it; the engine would default every resident to 0"
                )
            continue
        if item.residence_flag not in person:
            failures.append(
                f"{item.variable}: present, but {item.residence_flag} is absent, so "
                "nothing decided residence"
            )
            continue
        try:
            expected = us_resident_wage_tax_values(person, item, null_wages=null_wages)
        except ValueError as error:
            failures.append(f"{item.variable}: present but cannot be checked ({error})")
            continue
        observed = pd.to_numeric(person[item.variable], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        unexpected = ~np.isfinite(observed) & ~np.isnan(expected)
        if unexpected.any():
            failures.append(
                f"{item.variable}: {int(unexpected.sum())} missing or non-finite "
                "row(s) where the residence flag and wages decide a value"
            )
            continue
        disagree = (observed != expected) & ~(np.isnan(observed) & np.isnan(expected))
        if disagree.any():
            failures.append(
                f"{item.variable}: {int(disagree.sum())}/{len(person)} person row(s) "
                f"disagree with max({item.wage_input}, 0) for residents and 0 "
                "otherwise"
            )
    return failures


# ---------------------------------------------------------------------------
# Block-ladder verification
# ---------------------------------------------------------------------------


def check_us_local_jurisdictions_against_block_ladder(
    block_geoid: np.ndarray,
    population: np.ndarray,
    place_fips: np.ndarray,
    *,
    registry: UsLocalJurisdictionRegistry | None = None,
) -> dict[str, dict[str, Any]]:
    """Check every registry geography against a 2020 block ladder.

    Derives each flag for one row per block exactly as a household on that
    block gets it (county = the block geoid's first five digits, place = the
    ladder's place code), then requires, per jurisdiction:

    - at least one block;
    - flag-weighted block population equal to ``population_2020``; and
    - for a county rule with a coextensive city, the flagged blocks equal
      that city's blocks.

    Returns the per-jurisdiction block count and population; raises on the
    first disagreement.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    blocks = np.asarray(block_geoid, dtype=np.int64)
    pop = np.asarray(population, dtype=np.float64)
    places = np.asarray(place_fips, dtype=np.int64)
    if not (len(blocks) == len(pop) == len(places)):
        raise ValueError("block_geoid, population and place_fips must align.")
    block_table = pd.DataFrame(
        {
            "county_fips": [f"{value // 10**10:05d}" for value in blocks.tolist()],
            "place_fips": [
                "" if value == 0 else f"{value:05d}" for value in places.tolist()
            ],
        }
    )
    flags = us_local_jurisdiction_flag_values(
        block_table, rungs=JURISDICTION_RUNGS, registry=resolved
    )
    state = blocks // 10**13
    report: dict[str, dict[str, Any]] = {}
    for item in resolved.jurisdictions:
        mask = flags[item.flag]
        blocks_in = int(mask.sum())
        population_in = float(pop[mask].sum())
        if blocks_in == 0:
            raise ValueError(
                f"{item.flag}: no block in the ladder matches {item.geoid}."
            )
        if population_in != float(item.population_2020):
            raise ValueError(
                f"{item.flag}: ladder population {population_in:,.0f} differs from "
                f"the registry's 2020 population {item.population_2020:,}."
            )
        entry: dict[str, Any] = {"blocks": blocks_in, "population": int(population_in)}
        if item.coextensive_place_fips is not None:
            city = (state == int(item.state_fips)) & (
                places == int(item.coextensive_place_fips)
            )
            if not np.array_equal(city, mask):
                raise ValueError(
                    f"{item.flag}: county {item.county_fips} and place "
                    f"{item.state_fips}{item.coextensive_place_fips} are not the "
                    "same blocks in this ladder."
                )
            entry["coextensive_place_blocks"] = int(city.sum())
        report[item.flag] = entry
    return report


__all__ = [
    "BLOCK_GEOGRAPHY_RUNGS",
    "JURISDICTION_FLAG_DERIVED",
    "JURISDICTION_FLAG_UNAVAILABLE",
    "JURISDICTION_RUNGS",
    "JURISDICTION_RUNG_COUNTY",
    "JURISDICTION_RUNG_PLACE",
    "LOCAL_TAX_INPUT_PROVISIONS",
    "LOCAL_TAX_INPUT_RESIDENT_WAGES",
    "LOCAL_TAX_INPUT_UNAVAILABLE",
    "LOCAL_TAX_INPUT_UNAVAILABLE_BASES",
    "NULL_WAGES_PROPAGATE",
    "NULL_WAGES_REFUSE",
    "NULL_WAGE_MODES",
    "PUMA_GEOGRAPHY_RUNGS",
    "US_LOCAL_JURISDICTION_REGISTRY_KIND",
    "US_LOCAL_JURISDICTION_REGISTRY_RESOURCE",
    "US_LOCAL_JURISDICTION_REGISTRY_SCHEMA_VERSION",
    "US_LOCAL_JURISDICTION_REGISTRY_SHA256",
    "US_LOCAL_TAX_INPUT_REGISTRY_KIND",
    "US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE",
    "US_LOCAL_TAX_INPUT_REGISTRY_SCHEMA_VERSION",
    "US_LOCAL_TAX_INPUT_REGISTRY_SHA256",
    "UsLocalJurisdiction",
    "UsLocalJurisdictionRegistry",
    "UsLocalTaxInput",
    "UsLocalTaxInputRegistry",
    "apply_us_local_jurisdiction_flags",
    "apply_us_local_tax_inputs",
    "check_us_local_jurisdictions_against_block_ladder",
    "load_us_local_jurisdiction_registry",
    "load_us_local_tax_input_registry",
    "packaged_us_local_jurisdiction_registry",
    "packaged_us_local_tax_input_registry",
    "rederive_us_local_jurisdiction_flags",
    "us_local_jurisdiction_flag_failures",
    "us_local_jurisdiction_flag_receipt",
    "us_local_jurisdiction_flag_values",
    "us_local_jurisdiction_rungs_present",
    "us_local_tax_input_failures",
    "us_local_tax_input_output_columns",
    "us_resident_wage_tax_values",
    "with_us_local_jurisdiction_flags",
    "with_us_local_tax_inputs",
]
