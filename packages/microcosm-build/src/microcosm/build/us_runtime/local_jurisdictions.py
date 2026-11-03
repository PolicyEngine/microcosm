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

The rungs belong to the whole household table: one row without a block
withholds the place rung from every row. A step that removes rows (a
frozen-support or L0 selection) can therefore make a place-rule flag
decidable, so it derives the flags again on the rows it keeps; deriving a
support whose flags are already complete returns it unchanged. A step that
must keep an artifact's columns and bytes, and so cannot write the flags,
calls :func:`us_local_jurisdiction_artifact_receipt` instead, which refuses
an artifact that lacks a flag its geography decides.
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
    geography line, and for every step that then removes rows from it. A base
    H5 built before the flags existed carries none, and exporting it as-is
    would leave the engine default (``False``) on every resident. The rungs
    come from the artifact's own geography
    (:func:`us_local_jurisdiction_rungs_present`), so a selection that drops
    every row without a block makes the place-rule flags decidable on the
    rows it keeps. A flag the artifact already carries must agree with that
    geography, or this raises: re-deriving it would hide a corrupt artifact.

    An artifact that already carries exactly the flags its rungs decide, as
    boolean columns, comes back as the same object with the same receipt
    that writing them again would give; writing them would only copy every
    table.

    The receipt adds ``carried_flags``, the flags the artifact already held.
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
    carried = _carried_flags(frame, resolved)
    rungs = us_local_jurisdiction_rungs_present(household)
    values = us_local_jurisdiction_flag_values(
        household, rungs=rungs, registry=resolved
    )
    complete = set(carried) == set(values) and all(
        frame.table(item.entity)[item.flag].dtype == np.dtype(bool)
        for item in resolved.jurisdictions
        if item.flag in values
    )
    if complete:
        # Every carried flag agrees with the geography (checked above), and
        # the carried set is the derivable set: writing them changes nothing.
        derived = frame
        receipt = us_local_jurisdiction_flag_receipt(
            values,
            rungs=rungs,
            registry=resolved,
            household_weights=frame.weights_for("household").values,
        )
    else:
        derived, receipt = with_us_local_jurisdiction_flags(
            frame, rungs=rungs, registry=resolved
        )
    receipt["carried_flags"] = carried
    return derived, receipt


def us_local_jurisdiction_artifact_receipt(
    frame: Frame,
    *,
    registry: UsLocalJurisdictionRegistry | None = None,
) -> dict[str, Any]:
    """Require an artifact to carry its flags already, and return its receipt.

    For a step that must keep an artifact's columns and bytes, so it cannot
    write the flags itself (the annual static-aging lane copies its parent's
    source year and keeps its columns in every projected year). The artifact
    must carry every flag its own geography decides
    (:func:`us_local_jurisdiction_rungs_present`), each agreeing with that
    geography, and no flag its rungs cannot decide. Otherwise this raises:
    the engine would default a missing flag to ``False`` on every resident.
    The receipt has the shape :func:`rederive_us_local_jurisdiction_flags`
    returns.
    """

    resolved = registry or packaged_us_local_jurisdiction_registry()
    household = frame.table("household")
    rungs = us_local_jurisdiction_rungs_present(household)
    failures = us_local_jurisdiction_flag_failures(
        household,
        person=frame.table("person"),
        required_rungs=rungs,
        registry=resolved,
    )
    values = us_local_jurisdiction_flag_values(
        household, rungs=rungs, registry=resolved
    )
    carried = _carried_flags(frame, resolved)
    failures.extend(
        f"{flag}: present, but this artifact's rungs {list(rungs)} do not decide it"
        for flag in carried
        if flag not in values
    )
    if failures:
        raise ValueError(
            "The artifact lacks or contradicts local-jurisdiction flags its own "
            "geography decides: " + "; ".join(failures)
        )
    receipt = us_local_jurisdiction_flag_receipt(
        values,
        rungs=rungs,
        registry=resolved,
        household_weights=frame.weights_for("household").values,
    )
    receipt["carried_flags"] = carried
    return receipt


def _carried_flags(frame: Frame, registry: UsLocalJurisdictionRegistry) -> list[str]:
    return sorted(
        item.flag
        for item in registry.jurisdictions
        if item.flag in frame.table(item.entity)
    )


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
    "PUMA_GEOGRAPHY_RUNGS",
    "US_LOCAL_JURISDICTION_REGISTRY_KIND",
    "US_LOCAL_JURISDICTION_REGISTRY_RESOURCE",
    "US_LOCAL_JURISDICTION_REGISTRY_SCHEMA_VERSION",
    "US_LOCAL_JURISDICTION_REGISTRY_SHA256",
    "UsLocalJurisdiction",
    "UsLocalJurisdictionRegistry",
    "apply_us_local_jurisdiction_flags",
    "check_us_local_jurisdictions_against_block_ladder",
    "load_us_local_jurisdiction_registry",
    "packaged_us_local_jurisdiction_registry",
    "rederive_us_local_jurisdiction_flags",
    "us_local_jurisdiction_artifact_receipt",
    "us_local_jurisdiction_flag_failures",
    "us_local_jurisdiction_flag_receipt",
    "us_local_jurisdiction_flag_values",
    "us_local_jurisdiction_rungs_present",
    "with_us_local_jurisdiction_flags",
]
