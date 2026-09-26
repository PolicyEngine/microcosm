"""Explicit activation of the direct-national ACS 2024 published age counts.

A captured reference bundle is evidence, not permission. The accepted
`national-derived-inventory.json` says so about itself: `target_activation` is
false, `calibration_target_registry` is `not_created`, and every table carries
`targets_allowed: false`. This module never reads that flag as authority and
never edits the bundle. Authority comes from
:data:`NATIONAL_AGE_ACTIVATION` — an immutable declaration, checked into this
repository and reviewed as code, that names the source digest, the exact
publisher cells, the target role, the population universe, the target period
and the age convention. :func:`activate_national_age_targets` refuses unless
the pinned bytes reproduce that declaration, and it then builds a **new**
:class:`~microcosm.calibrate.registry.TargetRegistry` artifact.

What is authenticated here: the inventory bytes against the declared digest;
each selected raw Census response against the digest recorded for it *and*
against its request descriptor; the publisher's own labels, group, concept and
integer predicate for every selected cell; the single national row's
`GEO_ID`/`us`/`NAME` identity; and a from-bytes re-derivation of every selected
cell through the same :func:`classify_acs_value` precedence the district
inventory uses. Cells are parsed from exactly the bytes that were hashed.

What is *not* established here: that the resulting calibration is nationally or
congressional-district valid, that any modelled person was observed in 2024, or
that this registry may be released. B19001 and B25003 are reserved same-source
holdouts and are refused outright — this module cannot activate them, and it
cannot mark them fresh or unseen.

Published margins of error are converted to standard errors with the Census
90 percent factor and travel on the specs. The current Adam loss does not
consume them. A cell whose margin is a controlled-estimate sentinel carries a
**null** standard error, never zero: "no published variance" is not "no
variance".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from microcosm.calibrate.hierarchy import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
)
from microcosm.calibrate.provider_labels import calibration_provider_label
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.calibrate.variable_labels import calibration_variable_label

from ..cd_benchmark.protocol import RESERVED_FAMILIES
from .cd_reference import AUTHORITIES, classify_acs_value
from .cd_reference_sources import strict_json

__all__ = [
    "ACTIVATION_SCHEMA_VERSION",
    "ActivationError",
    "AgeBand",
    "NationalAgeActivation",
    "NATIONAL_AGE_ACTIVATION",
    "activate_national_age_targets",
    "activation_digest",
    "band_columns",
]

#: Revision of the activation declaration's canonical form. Bump it when the
#: meaning of a field changes; the digest already moves when a value changes.
ACTIVATION_SCHEMA_VERSION = 1

#: Census publishes ACS margins of error at 90 percent confidence; this is the
#: documented divisor that turns one into a standard error. The authority is
#: the same sample-size-and-data-quality note the district inventory cites.
_MOE_90_TO_SE = 1.645
_MOE_AUTHORITY = AUTHORITIES["acs_confidence"]

#: The published label prefix of the disjoint AGE distribution in S0101. Cells
#: under `SELECTED AGE CATEGORIES` overlap it and are refused by construction.
_AGE_LABEL_PREFIX = "Estimate!!Total!!Total population!!AGE!!"

_HEX64 = re.compile(r"[0-9a-f]{64}")


class ActivationError(ValueError):
    """The pinned evidence does not reproduce the activation declaration."""


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise ActivationError(reason)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True, slots=True)
class AgeBand:
    """One activated publisher cell and the household column that answers it.

    Attributes:
        variable: The publisher cell, e.g. ``"S0101_C01_002"``.
        label: The publisher's exact label for that cell's estimate. Binding
            it here means an edited cell list cannot silently pull a
            differently-defined published quantity.
        low: Inclusive lower bound in completed years.
        high: Inclusive upper bound, or ``None`` for the open top band.
        column: The household count column the population operator owns.
    """

    variable: str
    label: str
    low: int
    high: int | None
    column: str

    def __post_init__(self) -> None:
        _require(
            isinstance(self.variable, str)
            and isinstance(self.label, str)
            and isinstance(self.column, str)
            and self.variable
            and self.label
            and self.column,
            "AGE_BAND_TEXT",
        )
        _require(
            type(self.low) is int
            and self.low >= 0
            and (
                self.high is None or (type(self.high) is int and self.high >= self.low)
            ),
            f"AGE_BAND_BOUNDS:{self.variable}",
        )


#: Diagnostics schema 8 hierarchy vocabulary for activated national age cells.
#: The provider and category labels come from the reviewed US label catalogs;
#: the geography is the activation's own national publisher row; the single
#: dimension is the publisher cell whose published label the band binds.
HIERARCHY_PROVIDER = "census_acs"
HIERARCHY_DIMENSION_ID = "publisher_cell"
HIERARCHY_DIMENSION_LABEL = "Publisher cell"
NATIONAL_GEOGRAPHY_ID = "0100000US"
NATIONAL_GEOGRAPHY_LABEL = "United States"
NATIONAL_GEOGRAPHY_LEVEL = "country"


def _hierarchy_category_id(table: str) -> str:
    if table == "S0101":
        return "population_by_age"
    if table == "B01001":
        return "population_by_sex_and_age"
    raise ActivationError(f"UNSUPPORTED_TABLE:{table}")


def expected_demographic_hierarchy(
    table: str, *, variable: str, label: str, geography: str
) -> CalibrationHierarchy:
    """The only hierarchy a national demographic count cell may carry.

    Everything but the published cell label is fixed by the table and the
    national geography, so a registry validator can rebuild it from a spec and
    refuse any other provider, category, geography, dimension, or target.
    """
    _require(geography == NATIONAL_GEOGRAPHY_ID, f"UNSUPPORTED_GEOGRAPHY:{geography}")
    category_id = _hierarchy_category_id(table)
    provider_label = calibration_provider_label("us", HIERARCHY_PROVIDER)
    category_label = calibration_variable_label("us", HIERARCHY_PROVIDER, category_id)
    _require(bool(provider_label) and bool(category_label), "HIERARCHY_LABELS")
    return CalibrationHierarchy(
        provider=HierarchyNode(HIERARCHY_PROVIDER, provider_label),
        category=HierarchyCategory(
            f"{HIERARCHY_PROVIDER}.{category_id}", category_label, HIERARCHY_PROVIDER
        ),
        geography=HierarchyGeography(
            NATIONAL_GEOGRAPHY_ID, NATIONAL_GEOGRAPHY_LABEL, NATIONAL_GEOGRAPHY_LEVEL
        ),
        dimensions=(
            HierarchyDimension(
                HIERARCHY_DIMENSION_ID, HIERARCHY_DIMENSION_LABEL, variable, label
            ),
        ),
        target=HierarchyNode(variable, label),
    )


def demographic_target_hierarchy(
    table: str, band: AgeBand, *, geography: str
) -> CalibrationHierarchy:
    """The complete schema 8 hierarchy for one activated band's target spec."""
    _require(type(band) is AgeBand, "AGE_BAND_TYPE")
    return expected_demographic_hierarchy(
        table, variable=band.variable, label=band.label, geography=geography
    )


def _bands() -> tuple[AgeBand, ...]:
    quinary = [(low, low + 4) for low in range(0, 85, 5)]
    bands = []
    for index, (low, high) in enumerate(quinary):
        if low == 0:
            label = "Under 5 years"
        else:
            label = f"{low} to {high} years"
        bands.append(
            AgeBand(
                variable=f"S0101_C01_{index + 2:03d}",
                label=_AGE_LABEL_PREFIX + label,
                low=low,
                high=high,
                column=f"people_age_{low}_{high}",
            )
        )
    bands.append(
        AgeBand(
            variable="S0101_C01_019",
            label=_AGE_LABEL_PREFIX + "85 years and over",
            low=85,
            high=None,
            column="people_age_85_plus",
        )
    )
    return tuple(bands)


@dataclass(frozen=True, slots=True)
class NationalAgeActivation:
    """The immutable authority for one calibration connection.

    Every field is normative: :func:`activation_digest` covers all of them, and
    the digest travels into each produced :class:`TargetSpec`, so a changed
    declaration produces a different registry version and a different node key.

    Attributes:
        inventory_sha256: Digest of the accepted derived inventory's bytes.
        table: The activated publisher table.
        dataset: The Census API dataset the selected responses come from.
        acs_year: The publisher's data year.
        acs_release: The publisher's release.
        geography: The direct national geography id.
        geography_query: The publisher query that produced that single row.
        published_name: The publisher's own name for that row.
        confidence_level: The confidence level of the published margins.
        role: What the activated cells are for.
        universe: The population universe the cells count, stated in full.
        entity: The entity whose weights the targets constrain.
        period: The target period tag.
        age_convention_id: The closed token naming the convention. The
            population operator implements exactly this one and refuses any
            other, so the declaration and the operator cannot drift apart.
        age_convention: How model ages are read against the published bands.
        allowed_reservation_status: The only inventory reservation status this
            activation may act on. The same-source holdouts carry a different
            one and can never satisfy it.
        bands: The activated cells, in publisher order.
    """

    inventory_sha256: str
    table: str
    dataset: str
    acs_year: int
    acs_release: str
    geography: str
    geography_query: str
    published_name: str
    confidence_level: float
    role: str
    universe: str
    entity: str
    period: str
    age_convention_id: str
    age_convention: str
    allowed_reservation_status: str
    bands: tuple[AgeBand, ...]

    def __post_init__(self) -> None:
        _require(
            _HEX64.fullmatch(self.inventory_sha256) is not None,
            "ACTIVATION_INVENTORY_DIGEST",
        )
        _require(self.table not in RESERVED_FAMILIES, f"RESERVED_TABLE:{self.table}")
        _require(self.table == "S0101", f"UNSUPPORTED_TABLE:{self.table}")
        _require(self.role == "calibration", f"UNSUPPORTED_ROLE:{self.role}")
        _require(self.universe.startswith("population"), "UNSUPPORTED_UNIVERSE")
        _require(self.entity == "household", f"UNSUPPORTED_ENTITY:{self.entity}")
        _require(self.period == "2024", f"UNSUPPORTED_PERIOD:{self.period}")
        _require(
            self.age_convention_id == "observed_interview_age_completed_years",
            f"UNSUPPORTED_AGE_CONVENTION:{self.age_convention_id}",
        )
        _require(self.acs_year == 2024, "UNSUPPORTED_ACS_YEAR")
        _require(self.geography == "0100000US", "UNSUPPORTED_GEOGRAPHY")
        _require(
            self.allowed_reservation_status == "inventory_only",
            "UNSUPPORTED_RESERVATION_STATUS",
        )
        _require(
            isinstance(self.bands, tuple)
            and all(isinstance(band, AgeBand) for band in self.bands),
            "ACTIVATION_BANDS_TYPE",
        )
        _require(len(self.bands) == 18, f"ACTIVATION_BAND_COUNT:{len(self.bands)}")
        _require(
            len({band.variable for band in self.bands}) == len(self.bands)
            and len({band.column for band in self.bands}) == len(self.bands),
            "ACTIVATION_BANDS_NOT_DISTINCT",
        )
        expected_low = 0
        for index, band in enumerate(self.bands):
            _require(band.low == expected_low, f"ACTIVATION_BANDS_GAP:{band.variable}")
            _require(
                band.label.startswith(_AGE_LABEL_PREFIX),
                f"ACTIVATION_BAND_NOT_AGE_DISTRIBUTION:{band.variable}",
            )
            last = index == len(self.bands) - 1
            _require(
                (band.high is None) is last,
                f"ACTIVATION_BAND_OPEN_INTERVAL:{band.variable}",
            )
            if band.high is not None:
                expected_low = band.high + 1
        # Publisher cell order is the band order; nothing may be reordered or
        # substituted without changing the digest.
        _require(
            [band.variable for band in self.bands]
            == [f"S0101_C01_{index:03d}" for index in range(2, 20)],
            "ACTIVATION_CELL_SELECTOR",
        )


#: The reviewed authority. Root's first engineering rung: observed interview
#: ages from pooled source cohorts, calibrated to the 2024 published national
#: distribution. No birth dates, no invented aging, and no claim that any
#: modelled person was interviewed in 2024.
NATIONAL_AGE_ACTIVATION = NationalAgeActivation(
    inventory_sha256=(
        "750b624e89557944c442600befb5557847dd75be6ce439c20daa9dfea5a0baf7"
    ),
    table="S0101",
    dataset="acs/acs1/subject",
    acs_year=2024,
    acs_release="1-year",
    geography="0100000US",
    geography_query="us:*",
    published_name="United States",
    confidence_level=0.9,
    role="calibration",
    universe=(
        "population: every resident person the publisher counts, household "
        "and group-quarters alike; no household-population subsetting"
    ),
    entity="household",
    period="2024",
    age_convention_id="observed_interview_age_completed_years",
    age_convention=(
        "age in completed years as observed at each person's own source "
        "interview, used verbatim; source cohorts are pooled and their "
        "observed age distribution is calibrated to the 2024 published "
        "counts; no birth dates, no aging, and no assertion that any person "
        "was observed in 2024"
    ),
    allowed_reservation_status="inventory_only",
    bands=_bands(),
)


def activation_digest(declaration: NationalAgeActivation) -> str:
    """The content digest of a declaration's canonical form."""

    # asdict recurses into the frozen AgeBand rows, so every bound, label and
    # column name is inside the digest.
    return hashlib.sha256(
        _canonical({"schema_version": ACTIVATION_SCHEMA_VERSION, **asdict(declaration)})
    ).hexdigest()


def band_columns(
    declaration: NationalAgeActivation = NATIONAL_AGE_ACTIVATION,
) -> tuple[str, ...]:
    """The household count columns the activation expects, in band order."""

    return tuple(band.column for band in declaration.bands)


def _read_pinned(path: Path, digest: str, *, what: str) -> tuple[bytes, object]:
    """Hash the bytes, then parse *those* bytes. Never re-read from disk."""

    _require(_HEX64.fullmatch(digest) is not None, f"PINNED_DIGEST_SHAPE:{what}")
    _require(path.is_file(), f"PINNED_FILE_MISSING:{what}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    _require(actual == digest, f"PINNED_DIGEST_MISMATCH:{what}:{actual}")
    return payload, strict_json(payload)


def _selected_source(inventory: dict, declaration: NationalAgeActivation, kind: str):
    sources = inventory.get("sources")
    _require(isinstance(sources, dict), "INVENTORY_SOURCES")
    census = sources.get("census")
    _require(isinstance(census, list), "INVENTORY_CENSUS_SOURCES")
    entries = [
        entry
        for entry in census
        if isinstance(entry, dict)
        and entry.get("table") == declaration.table
        and entry.get("kind") == kind
    ]
    _require(len(entries) == 1, f"SOURCE_ENTRY_NOT_UNIQUE:{kind}:{len(entries)}")
    entry = entries[0]
    digest = entry.get("sha256")
    _require(
        isinstance(digest, str) and _HEX64.fullmatch(digest) is not None,
        f"SOURCE_DIGEST:{kind}",
    )
    _require(entry.get("dataset") == declaration.dataset, f"SOURCE_DATASET:{kind}")
    _require(entry.get("year") == declaration.acs_year, f"SOURCE_YEAR:{kind}")
    _require(
        entry.get("geography") == declaration.geography_query,
        f"SOURCE_GEOGRAPHY:{kind}",
    )
    url = entry.get("url")
    _require(isinstance(url, str) and url.startswith("https://"), f"SOURCE_URL:{kind}")
    _require(
        f"/{declaration.acs_year}/{declaration.dataset}" in url,
        f"SOURCE_URL_SCOPE:{kind}",
    )
    _require(
        entry.get("path") == f"raw/{entry.get('sha256')}.json", f"SOURCE_PATH:{kind}"
    )
    return entry


def _authenticated_response(
    root: Path, entry: dict, *, kind: str
) -> tuple[bytes, object]:
    """Authenticate the request descriptor and the response it names."""

    descriptor_path = (
        root / "requests" / f"{hashlib.sha256(entry['url'].encode()).hexdigest()}.json"
    )
    _require(descriptor_path.is_file(), f"REQUEST_DESCRIPTOR_MISSING:{kind}")
    descriptor = strict_json(descriptor_path.read_bytes())
    _require(descriptor == entry, f"REQUEST_DESCRIPTOR_MISMATCH:{kind}")
    payload, document = _read_pinned(
        root / entry["path"], entry["sha256"], what=f"response:{kind}"
    )
    _require(len(payload) == entry.get("size_bytes"), f"RESPONSE_SIZE:{kind}")
    return payload, document


def _national_row(document: object, declaration: NationalAgeActivation) -> dict:
    _require(
        isinstance(document, list) and len(document) == 2,
        "RESPONSE_NOT_A_SINGLE_NATIONAL_ROW",
    )
    header, values = document
    _require(
        isinstance(header, list) and all(isinstance(name, str) for name in header),
        "RESPONSE_HEADER_SHAPE",
    )
    _require(len(set(header)) == len(header), "RESPONSE_DUPLICATE_HEADER")
    _require(isinstance(values, list) and len(values) == len(header), "RESPONSE_WIDTH")
    row = dict(zip(header, values, strict=True))
    _require(row.get("GEO_ID") == declaration.geography, "RESPONSE_GEO_ID")
    _require(row.get("us") == "1", "RESPONSE_NOT_THE_NATION")
    _require(row.get("NAME") == declaration.published_name, "RESPONSE_PUBLISHED_NAME")
    return row


def _published_variable(metadata: object, name: str, declaration) -> dict:
    _require(isinstance(metadata, dict), "METADATA_SHAPE")
    variables = metadata.get("variables")
    _require(isinstance(variables, dict), "METADATA_VARIABLES")
    entry = variables.get(name)
    _require(isinstance(entry, dict), f"METADATA_VARIABLE_MISSING:{name}")
    _require(entry.get("group") == declaration.table, f"METADATA_GROUP:{name}")
    _require(entry.get("concept") == "Age and Sex", f"METADATA_CONCEPT:{name}")
    return entry


def _count_estimate(cell: dict, variable: str) -> int:
    estimate = cell["estimate"]
    _require(
        estimate["class"] == "numeric" and not estimate["annotation_conflict"],
        f"NON_COUNT_ESTIMATE:{variable}:{estimate['class']}",
    )
    value = estimate["numeric_value"]
    _require(
        value is not None
        and math.isfinite(value)
        and value >= 0
        and float(value).is_integer(),
        f"ESTIMATE_NOT_A_NONNEGATIVE_COUNT:{variable}",
    )
    return int(value)


def _standard_error(
    cell: dict, variable: str, *, confidence_level: float
) -> float | None:
    """A published margin becomes a standard error; a sentinel becomes null."""

    moe = cell["moe"]
    _require(not moe["annotation_conflict"], f"MOE_ANNOTATION_CONFLICT:{variable}")
    if moe["class"] != "numeric":
        # Controlled, suppressed and open-interval margins carry no published
        # variance. Null says "unknown"; zero would claim certainty.
        return None
    _require(confidence_level == 0.9, "UNSUPPORTED_CONFIDENCE_LEVEL")
    value = moe["numeric_value"]
    _require(
        value is not None and math.isfinite(value) and value >= 0,
        f"MOE_NOT_NONNEGATIVE:{variable}",
    )
    if value == 0:
        # A zero margin is a statement about a controlled cell, not a claim
        # that the count is certain. Null again, never a zero standard error.
        return None
    return float(value) / _MOE_90_TO_SE


def activate_national_age_targets(
    bundle_dir: str | Path,
    *,
    declaration: NationalAgeActivation = NATIONAL_AGE_ACTIVATION,
) -> TargetRegistry:
    """Authenticate the pinned bundle and mint a new age-count registry.

    Args:
        bundle_dir: The captured reference bundle root, holding
            ``national-derived-inventory.json``, ``raw/`` and ``requests/``.
        declaration: The activation authority. The default is the reviewed
            one; a caller-supplied declaration is validated identically and
            has no privileges of its own.

    Returns:
        A new :class:`TargetRegistry` of 18 disjoint national age counts.

    Raises:
        ActivationError: On any digest, identity, universe, label, schema,
            reservation, selector or value violation. The bundle is never
            written to, and a reserved same-source holdout can never be
            activated.
    """

    root = Path(bundle_dir)
    _, inventory = _read_pinned(
        root / "national-derived-inventory.json",
        declaration.inventory_sha256,
        what="inventory",
    )
    _require(isinstance(inventory, dict), "INVENTORY_SHAPE")
    _require(
        inventory.get("kind") == "us_national_acs_reference_inventory",
        "INVENTORY_KIND",
    )
    # The bundle states it is not an activation. Assert that rather than
    # letting an edited descriptive field pass for authority.
    _require(inventory.get("target_activation") is False, "INVENTORY_CLAIMS_ACTIVATION")
    _require(
        inventory.get("calibration_target_registry") == "not_created",
        "INVENTORY_CLAIMS_A_REGISTRY",
    )
    scope = inventory.get("scope")
    _require(isinstance(scope, dict), "INVENTORY_SCOPE")
    _require(scope.get("country") == "us", "INVENTORY_COUNTRY")
    _require(scope.get("acs_year") == declaration.acs_year, "INVENTORY_ACS_YEAR")
    _require(scope.get("acs_release") == declaration.acs_release, "INVENTORY_RELEASE")
    _require(
        scope.get("geography") == f"nation (for={declaration.geography_query})",
        "INVENTORY_GEOGRAPHY",
    )

    tables = inventory.get("acs")
    _require(isinstance(tables, dict), "INVENTORY_TABLES")
    for reserved in sorted(RESERVED_FAMILIES):
        held = tables.get(reserved)
        _require(isinstance(held, dict), f"RESERVED_TABLE_ABSENT:{reserved}")
        held_reservation = held.get("reservation")
        _require(
            isinstance(held_reservation, dict),
            f"RESERVED_RESERVATION_SHAPE:{reserved}",
        )
        _require(
            held_reservation.get("status") == "reserved_same_source_holdout",
            f"RESERVED_TABLE_RELABELLED:{reserved}",
        )

    entry = tables.get(declaration.table)
    _require(isinstance(entry, dict), f"TABLE_ABSENT:{declaration.table}")
    _require(entry.get("lineage") == "published", "TABLE_LINEAGE")
    _require(entry.get("district_derived") is False, "TABLE_IS_DISTRICT_DERIVED")
    _require(entry.get("geography_level") == "nation", "TABLE_GEOGRAPHY_LEVEL")
    _require(entry.get("geography_id") == declaration.geography, "TABLE_GEOGRAPHY_ID")
    _require(
        entry.get("source_geography_query") == declaration.geography_query,
        "TABLE_GEOGRAPHY_QUERY",
    )
    _require(entry.get("published_name") == declaration.published_name, "TABLE_NAME")
    _require(entry.get("year") == declaration.acs_year, "TABLE_YEAR")
    _require(
        entry.get("confidence_level") == declaration.confidence_level,
        "TABLE_CONFIDENCE_LEVEL",
    )
    _require(
        entry.get("grouping") == "subject_age_and_sex_published_groups",
        "TABLE_GROUPING",
    )
    reservation = entry.get("reservation")
    _require(isinstance(reservation, dict), "TABLE_RESERVATION")
    _require(
        reservation.get("status") == declaration.allowed_reservation_status,
        f"TABLE_RESERVATION_STATUS:{reservation.get('status')}",
    )
    # The bundle's own permission flags stay false and stay unread as
    # authority; an inventory edited to claim permission is refused, and the
    # digest above would already have rejected it.
    _require(reservation.get("targets_allowed") is False, "INVENTORY_GRANTS_TARGETS")
    _require(reservation.get("tuning_allowed") is False, "INVENTORY_GRANTS_TUNING")

    metadata_entry = _selected_source(inventory, declaration, "metadata")
    data_entry = _selected_source(inventory, declaration, "data")
    _, metadata = _authenticated_response(root, metadata_entry, kind="metadata")
    _, data = _authenticated_response(root, data_entry, kind="data")
    row = _national_row(data, declaration)

    inventory_cells = entry.get("cells")
    _require(isinstance(inventory_cells, list), "TABLE_CELLS")
    by_variable: dict[str, dict] = {}
    for cell in inventory_cells:
        _require(isinstance(cell, dict), "CELL_SHAPE")
        variable = cell.get("variable")
        _require(isinstance(variable, str), "CELL_VARIABLE")
        _require(variable not in by_variable, f"DUPLICATE_CELL:{variable}")
        by_variable[variable] = cell

    def derived_cell(variable: str) -> dict:
        """Re-derive one cell from the authenticated bytes and cross-check it.

        The same annotation-precedence reader the district inventory uses runs
        again over exactly the response that was hashed; the accepted inventory
        must then agree with it cell for cell.
        """

        for suffix in ("E", "M", "EA", "MA"):
            _require(
                f"{variable}{suffix}" in row,
                f"RESPONSE_CELL_MISSING:{variable}{suffix}",
            )
        derived = {
            "variable": variable,
            "estimate": classify_acs_value(row[f"{variable}E"], row[f"{variable}EA"]),
            "moe": classify_acs_value(row[f"{variable}M"], row[f"{variable}MA"]),
        }
        recorded = by_variable.get(variable)
        _require(recorded is not None, f"CELL_ABSENT_FROM_INVENTORY:{variable}")
        _require(derived == recorded, f"CELL_DISAGREES_WITH_INVENTORY:{variable}")
        return derived

    specs: list[TargetSpec] = []
    digest = activation_digest(declaration)
    for band in declaration.bands:
        variable = band.variable
        published = _published_variable(metadata, f"{variable}E", declaration)
        _require(published.get("label") == band.label, f"PUBLISHED_LABEL:{variable}")
        _require(
            published.get("predicateType") == "int", f"PUBLISHED_NOT_A_COUNT:{variable}"
        )
        derived = derived_cell(variable)
        value = _count_estimate(derived, variable)
        specs.append(
            TargetSpec(
                name=variable,
                entity=declaration.entity,
                measure=band.column,
                value=float(value),
                period=declaration.period,
                se=_standard_error(
                    derived, variable, confidence_level=declaration.confidence_level
                ),
                source=f"{data_entry['url']} ({declaration.table} {variable}E)",
                family=f"acs.{declaration.table}",
                hierarchy=demographic_target_hierarchy(
                    declaration.table, band, geography=declaration.geography
                ),
                notes=(
                    f"activation={digest}; band={band.low}-"
                    f"{'+' if band.high is None else band.high}; "
                    f"label={band.label}; "
                    f"age_convention={declaration.age_convention}; "
                    f"uncertainty=published 90% MOE converted at "
                    f"{_MOE_90_TO_SE} ({_MOE_AUTHORITY}), not consumed "
                    f"by the current loss"
                ),
                metadata={
                    "table": declaration.table,
                    "reference_sha256": declaration.inventory_sha256,
                    "geography": declaration.geography,
                    "universe": "population",
                    "role": declaration.role,
                    "evidence_scope": "source_documented",
                },
            )
        )

    # Coherence with the publisher's own total, checked and then discarded:
    # the total is deliberately not activated as a nineteenth loss term.
    total_variable = f"{declaration.table}_C01_001"
    total = _count_estimate(derived_cell(total_variable), total_variable)
    _require(
        sum(int(spec.value) for spec in specs) == total,
        "ACTIVATED_BANDS_DO_NOT_PARTITION_THE_PUBLISHED_TOTAL",
    )
    return TargetRegistry(specs, country="us")
