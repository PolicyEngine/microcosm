"""The activation declaration, against invented-shaped and pinned evidence.

Most of this file runs on an *invented* bundle built in a temporary directory:
its geography ids and labels imitate the publisher's shape, but every count is
made up. That keeps the refusal surface under test everywhere, including CI,
where the accepted local bundle is absent.

The remaining tests read the accepted public national aggregate bundle when it
is present. Nothing here opens a genuine population, parses a full source, fits
or scores anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.build.cd_benchmark.protocol import RESERVED_FAMILIES
from microcosm.build.us_runtime.cd_reference import classify_acs_value
from microcosm.build.us_runtime.demographic_calibration_graph import (
    DemographicCalibrationKernel,
    demographic_calibration_node,
)
from microcosm.build.us_runtime.national_age_activation import (
    NATIONAL_AGE_ACTIVATION,
    ActivationError,
    AgeBand,
    activate_national_age_targets,
    activation_digest,
    band_columns,
)
from microcosm.calibrate.registry import TargetRegistry

#: The accepted capture is a local review artifact, not repository content.
#: Point ``MICROCOSM_US_NATIONAL_ACS_BUNDLE`` at it to run the pinned tests;
#: everywhere else, including CI, they skip and the invented bundle carries the
#: behavioural coverage.
_bundle_path = os.environ.get("MICROCOSM_US_NATIONAL_ACS_BUNDLE")
BUNDLE = Path(_bundle_path) if _bundle_path else None
INVENTORY = "national-derived-inventory.json"
ACCEPTED_INVENTORY_SHA256 = (
    "750b624e89557944c442600befb5557847dd75be6ce439c20daa9dfea5a0baf7"
)

pinned_bundle = pytest.mark.skipif(
    BUNDLE is None or not (BUNDLE / INVENTORY).is_file(),
    reason="the accepted national reference bundle is not present locally",
)


# --------------------------------------------------------------------------
# An invented bundle with the publisher's shape and none of its numbers
# --------------------------------------------------------------------------

#: Invented counts, one per activated band. These are not Census values.
INVENTED_COUNTS = tuple(1_000 + 7 * ordinal for ordinal in range(18))
#: Invented margins. The band at this index gets a controlled sentinel instead.
CONTROLLED_BAND = 3
INVENTED_MARGINS = tuple(10 + ordinal for ordinal in range(18))

_SUBJECT = "acs/acs1/subject"
_DATA_URL = (
    "https://api.census.gov/data/2024/acs/acs1/subject"
    "?get=group%28S0101%29&for=us%3A%2A"
)
_METADATA_URL = "https://api.census.gov/data/2024/acs/acs1/subject/groups/S0101.json"


def _invented_documents(*, geo_id="0100000US", name="United States", us="1"):
    """The metadata and single-row data response an invented capture would hold."""

    labels = {"S0101_C01_001": "Estimate!!Total!!Total population"}
    row = {"GEO_ID": geo_id, "NAME": name, "us": us}
    total = sum(INVENTED_COUNTS)
    row["S0101_C01_001E"] = str(total)
    row["S0101_C01_001EA"] = None
    row["S0101_C01_001M"] = "-555555555"
    row["S0101_C01_001MA"] = "*****"
    for ordinal, band in enumerate(NATIONAL_AGE_ACTIVATION.bands):
        labels[band.variable] = band.label
        row[f"{band.variable}E"] = str(INVENTED_COUNTS[ordinal])
        row[f"{band.variable}EA"] = None
        if ordinal == CONTROLLED_BAND:
            row[f"{band.variable}M"] = "-555555555"
            row[f"{band.variable}MA"] = "*****"
        else:
            row[f"{band.variable}M"] = str(INVENTED_MARGINS[ordinal])
            row[f"{band.variable}MA"] = None
    metadata = {
        "variables": {
            f"{variable}{suffix}": {
                "label": ("" if suffix == "E" else f"{suffix} of ") + label,
                "concept": "Age and Sex",
                "predicateType": "int" if suffix in {"E", "M"} else "string",
                "group": "S0101",
                "limit": 0,
                "predicateOnly": True,
            }
            for variable, label in labels.items()
            for suffix in ("E", "EA", "M", "MA")
        }
    }
    header = list(row)
    data = [header, [row[column] for column in header]]
    return metadata, data


def _cells_from(data) -> list[dict]:
    header, values = data[0], data[1]
    row = dict(zip(header, values, strict=True))
    variables = sorted(
        key[:-1] for key in row if key.startswith("S0101_") and key.endswith("E")
    )
    return [
        {
            "variable": variable,
            "estimate": classify_acs_value(row[f"{variable}E"], row[f"{variable}EA"]),
            "moe": classify_acs_value(row[f"{variable}M"], row[f"{variable}MA"]),
        }
        for variable in variables
    ]


def _write_source(root: Path, document, *, url, kind) -> dict:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "raw" / f"{digest}.json").write_bytes(payload)
    entry = {
        "dataset": _SUBJECT,
        "geography": "us:*",
        "kind": kind,
        "path": f"raw/{digest}.json",
        "retrieved_at": "2026-09-06T00:00:00+00:00",
        "sha256": digest,
        "size_bytes": len(payload),
        "table": "S0101",
        "url": url,
        "year": 2024,
    }
    return entry


def _write_descriptor(root: Path, entry: dict) -> None:
    descriptor = (
        root / "requests" / f"{hashlib.sha256(entry['url'].encode()).hexdigest()}.json"
    )
    descriptor.write_bytes(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
    )


def _reserved_stub(table: str, status: str) -> dict:
    return {
        "table": table,
        "year": 2024,
        "geography_level": "nation",
        "geography_id": "0100000US",
        "source_geography_query": "us:*",
        "published_name": "United States",
        "confidence_level": 0.9,
        "grouping": "published_detailed_table_cells",
        "reservation": {
            "status": status,
            "targets_allowed": False,
            "tuning_allowed": False,
            "reason": "invented fixture",
        },
        "lineage": "published",
        "district_derived": False,
        "cells": [],
        "variables": {},
    }


def invented_bundle(root: Path, *, mutate=None, documents=None) -> str:
    """Write an invented-shaped bundle; return its inventory digest."""

    root.mkdir(parents=True, exist_ok=True)
    metadata, data = documents if documents is not None else _invented_documents()
    metadata_entry = _write_source(root, metadata, url=_METADATA_URL, kind="metadata")
    data_entry = _write_source(root, data, url=_DATA_URL, kind="data")
    inventory = {
        "kind": "us_national_acs_reference_inventory",
        "schema_version": 1,
        "purpose": "invented fixture; no published Census value",
        "target_activation": False,
        "calibration_target_registry": "not_created",
        "national_acceptance_protocol": "not_created",
        "candidate_evaluation": "not_performed",
        "district_derived_check": "not_performed_out_of_scope",
        "scope": {
            "acs_release": "1-year",
            "acs_year": 2024,
            "country": "us",
            "geography": "nation (for=us:*)",
        },
        "sources": {"census": [metadata_entry, data_entry]},
        "acs": {
            "S0101": {
                "table": "S0101",
                "year": 2024,
                "geography_level": "nation",
                "geography_id": "0100000US",
                "source_geography_query": "us:*",
                "published_name": "United States",
                "confidence_level": 0.9,
                "grouping": "subject_age_and_sex_published_groups",
                "reservation": {
                    "status": "inventory_only",
                    "targets_allowed": False,
                    "tuning_allowed": False,
                    "reason": "invented fixture",
                },
                "lineage": "published",
                "district_derived": False,
                "cells": _cells_from(data),
                "variables": metadata["variables"],
            },
            "B01001": _reserved_stub("B01001", "inventory_only"),
            "B19001": _reserved_stub("B19001", "reserved_same_source_holdout"),
            "B25003": _reserved_stub("B25003", "reserved_same_source_holdout"),
        },
    }
    if mutate is not None:
        mutate(inventory)
    # Descriptors are written last so a mutated source entry stays consistent
    # with its own request record; a descriptor is tampered with explicitly.
    for entry in inventory["sources"]["census"]:
        _write_descriptor(root, entry)
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    (root / INVENTORY).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def invented_declaration(digest: str):
    return replace(NATIONAL_AGE_ACTIVATION, inventory_sha256=digest)


def activate_invented(root: Path, **kwargs) -> TargetRegistry:
    digest = invented_bundle(root, **kwargs)
    return activate_national_age_targets(root, declaration=invented_declaration(digest))


# --------------------------------------------------------------------------
# The declaration itself
# --------------------------------------------------------------------------


def test_declaration_names_the_accepted_inventory_and_eighteen_disjoint_bands():
    declaration = NATIONAL_AGE_ACTIVATION
    assert declaration.inventory_sha256 == ACCEPTED_INVENTORY_SHA256
    assert declaration.table == "S0101"
    assert declaration.table not in RESERVED_FAMILIES
    assert declaration.role == "calibration"
    assert declaration.period == "2024"
    assert declaration.geography == "0100000US"
    assert declaration.allowed_reservation_status == "inventory_only"
    assert "group-quarters" in declaration.universe
    assert declaration.age_convention_id == "observed_interview_age_completed_years"

    variables = [band.variable for band in declaration.bands]
    assert variables == [f"S0101_C01_{cell:03d}" for cell in range(2, 20)]
    assert len(variables) == 18
    covered = 0
    for band in declaration.bands[:-1]:
        assert band.low == covered
        assert band.high == band.low + 4
        covered = band.high + 1
    assert declaration.bands[-1].low == 85
    assert declaration.bands[-1].high is None
    assert band_columns()[0] == "people_age_0_4"
    assert band_columns()[-1] == "people_age_85_plus"


def test_declaration_is_immutable_and_its_digest_covers_every_field():
    declaration = NATIONAL_AGE_ACTIVATION
    with pytest.raises(AttributeError):
        declaration.role = "tuning"
    with pytest.raises(AttributeError):
        declaration.bands[0].low = 1

    baseline = activation_digest(declaration)
    moved = replace(
        declaration,
        bands=(
            replace(declaration.bands[0], column="people_age_0_to_4"),
            *declaration.bands[1:],
        ),
    )
    assert activation_digest(moved) != baseline
    assert (
        activation_digest(replace(declaration, universe="population, households only"))
        != baseline
    )
    assert activation_digest(replace(declaration, inventory_sha256="0" * 64)) != (
        baseline
    )
    # Prose that looks descriptive is normative here: a changed convention is a
    # changed activation.
    assert activation_digest(replace(declaration, age_convention="aged to 2024")) != (
        baseline
    )


@pytest.mark.parametrize(
    "change",
    [
        {"table": "B19001"},
        {"table": "B25003"},
        {"table": "B01001"},
        {"role": "tuning"},
        {"period": "2023"},
        {"acs_year": 2023},
        {"entity": "person"},
        {"geography": "0400000US06"},
        {"universe": "household population only"},
        {"allowed_reservation_status": "reserved_same_source_holdout"},
        {"age_convention_id": "aged_to_target_period"},
        {"inventory_sha256": "not-a-digest"},
    ],
)
def test_declaration_refuses_an_unapproved_scope(change):
    with pytest.raises(ActivationError):
        replace(NATIONAL_AGE_ACTIVATION, **change)


@pytest.mark.parametrize(
    "bands",
    [
        # The published total is not part of the disjoint AGE distribution.
        lambda base: (replace(base[0], variable="S0101_C01_001"), *base[1:]),
        # A SELECTED AGE CATEGORIES cell overlaps the distribution.
        lambda base: (*base[:-1], replace(base[-1], variable="S0101_C01_020")),
        # Dropping a band leaves the selector short of the partition.
        lambda base: base[:-1],
        # Reordering breaks contiguity.
        lambda base: (base[1], base[0], *base[2:]),
        # A duplicated column would double-count one band.
        lambda base: (replace(base[0], column=base[1].column), *base[1:]),
        # A gap between bands would lose people.
        lambda base: (replace(base[0], high=3), *base[1:]),
        # A second open interval cannot be disjoint.
        lambda base: (replace(base[0], high=None), *base[1:]),
        # A label outside the AGE distribution is not an approved selector.
        lambda base: (
            replace(base[0], label="Estimate!!Total!!Total population"),
            *base[1:],
        ),
    ],
)
def test_declaration_refuses_an_unapproved_selector(bands):
    with pytest.raises(ActivationError):
        replace(NATIONAL_AGE_ACTIVATION, bands=bands(NATIONAL_AGE_ACTIVATION.bands))


def test_age_band_refuses_impossible_bounds():
    with pytest.raises(ActivationError):
        AgeBand(variable="v", label="l", low=-1, high=4, column="c")
    with pytest.raises(ActivationError):
        AgeBand(variable="v", label="l", low=10, high=4, column="c")
    with pytest.raises(ActivationError):
        AgeBand(variable="v", label="l", low=0, high=4, column="")


# --------------------------------------------------------------------------
# Activation over the invented-shaped bundle
# --------------------------------------------------------------------------


def test_invented_activation_creates_a_new_registry(tmp_path):
    registry = activate_invented(tmp_path / "bundle")
    assert isinstance(registry, TargetRegistry)
    assert registry.country == "us"
    assert len(registry) == 18
    assert registry.families() == ("acs.S0101",)
    assert tuple(spec.measure for spec in registry) == band_columns()
    assert [spec.value for spec in registry] == [
        float(count) for count in INVENTED_COUNTS
    ]
    for spec in registry:
        assert spec.entity == "household"
        assert spec.period == "2024"
        assert spec.metadata["role"] == "calibration"
        assert spec.metadata["universe"] == "population"
        assert spec.metadata["geography"] == "0100000US"
        assert spec.metadata["evidence_scope"] == "source_documented"
    # The unchanged calibration node accepts it as-is.
    node = demographic_calibration_node(
        registry,
        base="assembled",
        epochs=4,
        learning_rate=0.01,
        max_weight_ratio=4.0,
        max_initial_weight_ratio=4.0,
    )
    assert node.inputs[0].columns == band_columns()


def test_a_controlled_margin_carries_a_null_standard_error(tmp_path):
    registry = activate_invented(tmp_path / "bundle")
    specs = list(registry)
    assert specs[CONTROLLED_BAND].se is None, "controlled uncertainty is null, not zero"
    for ordinal, spec in enumerate(specs):
        if ordinal == CONTROLLED_BAND:
            continue
        assert spec.se == pytest.approx(INVENTED_MARGINS[ordinal] / 1.645)
    assert DemographicCalibrationKernel.capabilities.consumes_se is False


def test_a_registry_version_moves_with_the_activation(tmp_path):
    baseline = activate_invented(tmp_path / "a").version
    shifted = activate_national_age_targets(
        tmp_path / "a",
        declaration=replace(
            invented_declaration(
                hashlib.sha256((tmp_path / "a" / INVENTORY).read_bytes()).hexdigest()
            ),
            age_convention="aged forward to 2024",
        ),
    ).version
    assert shifted != baseline


def test_the_published_total_is_a_coherence_check_not_a_target(tmp_path):
    registry = activate_invented(tmp_path / "bundle")
    assert "S0101_C01_001" not in {spec.name for spec in registry}
    assert sum(spec.value for spec in registry) == float(sum(INVENTED_COUNTS))

    metadata, data = _invented_documents()
    header, values = data
    row = dict(zip(header, values, strict=True))
    row["S0101_C01_001E"] = str(sum(INVENTED_COUNTS) + 1)
    with pytest.raises(
        ActivationError, match="ACTIVATED_BANDS_DO_NOT_PARTITION_THE_PUBLISHED_TOTAL"
    ):
        activate_invented(
            tmp_path / "broken",
            documents=(metadata, [header, [row[column] for column in header]]),
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda d: d["acs"]["S0101"].__setitem__("district_derived", True),
            "TABLE_IS_DISTRICT_DERIVED",
        ),
        (
            lambda d: d["acs"]["S0101"].__setitem__(
                "geography_level", "congressional district"
            ),
            "TABLE_GEOGRAPHY_LEVEL",
        ),
        (
            lambda d: d["acs"]["S0101"].__setitem__("geography_id", "5001900US0601"),
            "TABLE_GEOGRAPHY_ID",
        ),
        (
            lambda d: d["acs"]["S0101"].__setitem__(
                "source_geography_query", "congressional district:*"
            ),
            "TABLE_GEOGRAPHY_QUERY",
        ),
        (lambda d: d["acs"]["S0101"].__setitem__("year", 2023), "TABLE_YEAR"),
        (
            lambda d: d["acs"]["S0101"].__setitem__("lineage", "derived"),
            "TABLE_LINEAGE",
        ),
        (
            lambda d: d["acs"]["S0101"].__setitem__("confidence_level", 0.95),
            "TABLE_CONFIDENCE_LEVEL",
        ),
        (
            lambda d: d["acs"]["S0101"].__setitem__(
                "grouping", "published_detailed_table_cells"
            ),
            "TABLE_GROUPING",
        ),
        (lambda d: d["scope"].__setitem__("acs_year", 2023), "INVENTORY_ACS_YEAR"),
        (
            lambda d: d["scope"].__setitem__("acs_release", "5-year"),
            "INVENTORY_RELEASE",
        ),
        (
            lambda d: d["scope"].__setitem__("geography", "nation (for=state:*)"),
            "INVENTORY_GEOGRAPHY",
        ),
        (lambda d: d["scope"].__setitem__("country", "uk"), "INVENTORY_COUNTRY"),
        (lambda d: d.__setitem__("kind", "something_else"), "INVENTORY_KIND"),
        (
            lambda d: d.__setitem__("calibration_target_registry", "created"),
            "INVENTORY_CLAIMS_A_REGISTRY",
        ),
        (
            lambda d: d["acs"]["S0101"]["cells"].append(
                dict(d["acs"]["S0101"]["cells"][1])
            ),
            "DUPLICATE_CELL",
        ),
        (
            lambda d: d["acs"]["S0101"]["cells"].__setitem__(
                1, {"variable": "S0101_C01_002"}
            ),
            "CELL_DISAGREES_WITH_INVENTORY",
        ),
        (lambda d: d["acs"].pop("S0101"), "TABLE_ABSENT:S0101"),
        (lambda d: d["acs"].pop("B19001"), "RESERVED_TABLE_ABSENT:B19001"),
        (
            lambda d: d["sources"]["census"].append(dict(d["sources"]["census"][1])),
            "SOURCE_ENTRY_NOT_UNIQUE:data",
        ),
        (
            lambda d: d["sources"]["census"][1].__setitem__("dataset", "acs/acs1"),
            "SOURCE_DATASET:data",
        ),
        (
            lambda d: d["sources"]["census"][1].__setitem__("year", 2023),
            "SOURCE_YEAR:data",
        ),
        (
            lambda d: d["sources"]["census"][1].__setitem__(
                "geography", "congressional district:*"
            ),
            "SOURCE_GEOGRAPHY:data",
        ),
        (
            lambda d: d["sources"]["census"][1].__setitem__("size_bytes", 1),
            "RESPONSE_SIZE:data",
        ),
    ],
)
def test_reference_tampering_is_refused(tmp_path, mutate, reason):
    with pytest.raises(ActivationError, match=reason):
        activate_invented(tmp_path / "bundle", mutate=mutate)


def test_an_edited_permission_flag_is_not_activation_authority(tmp_path):
    """The bundle's own flags cannot grant what only the declaration can."""

    def allow(inventory):
        inventory["acs"]["S0101"]["reservation"]["targets_allowed"] = True

    with pytest.raises(ActivationError, match="INVENTORY_GRANTS_TARGETS"):
        activate_invented(tmp_path / "targets", mutate=allow)

    def tune(inventory):
        inventory["acs"]["S0101"]["reservation"]["tuning_allowed"] = True

    with pytest.raises(ActivationError, match="INVENTORY_GRANTS_TUNING"):
        activate_invented(tmp_path / "tuning", mutate=tune)

    def claim(inventory):
        inventory["target_activation"] = True

    with pytest.raises(ActivationError, match="INVENTORY_CLAIMS_ACTIVATION"):
        activate_invented(tmp_path / "claim", mutate=claim)


@pytest.mark.parametrize("table", sorted(RESERVED_FAMILIES))
def test_a_reserved_holdout_can_never_be_activated(tmp_path, table):
    with pytest.raises(ActivationError, match=f"RESERVED_TABLE:{table}"):
        replace(NATIONAL_AGE_ACTIVATION, table=table)

    def relabel(inventory):
        inventory["acs"][table]["reservation"]["status"] = "inventory_only"

    with pytest.raises(ActivationError, match=f"RESERVED_TABLE_RELABELLED:{table}"):
        activate_invented(tmp_path / "bundle", mutate=relabel)


@pytest.mark.parametrize("reservation", [None, []])
def test_malformed_holdout_reservation_raises_activation_error(tmp_path, reservation):
    def malformed(inventory):
        inventory["acs"]["B19001"]["reservation"] = reservation

    with pytest.raises(ActivationError, match="RESERVED_RESERVATION_SHAPE:B19001"):
        activate_invented(tmp_path / "bundle", mutate=malformed)


def test_missing_source_digest_raises_activation_error(tmp_path):
    def malformed(inventory):
        for entry in inventory["sources"]["census"]:
            if entry["kind"] == "metadata":
                entry.pop("sha256")

    with pytest.raises(ActivationError, match="SOURCE_DIGEST:metadata"):
        activate_invented(tmp_path / "bundle", mutate=malformed)


def test_the_activated_registry_never_mentions_a_reserved_family(tmp_path):
    registry = activate_invented(tmp_path / "bundle")
    text = json.dumps(
        [
            {
                "name": spec.name,
                "measure": spec.measure,
                "source": spec.source,
                "family": spec.family,
                "notes": spec.notes,
                "metadata": dict(spec.metadata),
            }
            for spec in registry
        ]
    )
    for reserved in RESERVED_FAMILIES:
        assert reserved not in text


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda row: row.__setitem__("GEO_ID", "5001900US0601"), "RESPONSE_GEO_ID"),
        (lambda row: row.__setitem__("us", "6"), "RESPONSE_NOT_THE_NATION"),
        (
            lambda row: row.__setitem__("NAME", "California"),
            "RESPONSE_PUBLISHED_NAME",
        ),
    ],
)
def test_the_national_row_identity_is_verified(tmp_path, edit, reason):
    metadata, data = _invented_documents()
    header, values = data
    row = dict(zip(header, values, strict=True))
    edit(row)
    with pytest.raises(ActivationError, match=reason):
        activate_invented(
            tmp_path / "bundle",
            documents=(metadata, [header, [row[column] for column in header]]),
        )


def test_more_than_one_geography_row_is_refused(tmp_path):
    metadata, data = _invented_documents()
    with pytest.raises(ActivationError, match="RESPONSE_NOT_A_SINGLE_NATIONAL_ROW"):
        activate_invented(
            tmp_path / "bundle", documents=(metadata, [*data, list(data[1])])
        )


def test_a_duplicate_response_header_is_refused(tmp_path):
    metadata, (header, values) = _invented_documents()
    with pytest.raises(ActivationError, match="RESPONSE_DUPLICATE_HEADER"):
        activate_invented(
            tmp_path / "bundle",
            documents=(metadata, [[*header, header[0]], [*values, values[0]]]),
        )


@pytest.mark.parametrize(
    ("raw", "annotation", "reason"),
    [
        ("-666666666", "-", "NON_COUNT_ESTIMATE"),  # insufficient sample
        ("-888888888", "(X)", "NON_COUNT_ESTIMATE"),  # not applicable
        ("-999999999", "N", "NON_COUNT_ESTIMATE"),  # suppressed geography
        (None, None, "NON_COUNT_ESTIMATE"),  # missing
        ("-4", None, "ESTIMATE_NOT_A_NONNEGATIVE_COUNT"),
        ("1234.5", None, "ESTIMATE_NOT_A_NONNEGATIVE_COUNT"),
    ],
)
def test_a_non_count_or_suppressed_estimate_is_refused(
    tmp_path, raw, annotation, reason
):
    metadata, data = _invented_documents()
    header, values = data
    row = dict(zip(header, values, strict=True))
    row["S0101_C01_005E"] = raw
    row["S0101_C01_005EA"] = annotation
    with pytest.raises(ActivationError, match=reason):
        activate_invented(
            tmp_path / "bundle",
            documents=(metadata, [header, [row[column] for column in header]]),
        )


def test_a_conflicting_annotation_is_refused(tmp_path):
    """A sentinel and an annotation that disagree are never silently resolved."""

    metadata, data = _invented_documents()
    header, values = data
    row = dict(zip(header, values, strict=True))
    row["S0101_C01_006M"] = "-555555555"
    row["S0101_C01_006MA"] = "(X)"
    with pytest.raises(ActivationError, match="MOE_ANNOTATION_CONFLICT"):
        activate_invented(
            tmp_path / "bundle",
            documents=(metadata, [header, [row[column] for column in header]]),
        )


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (
            lambda variables: variables["S0101_C01_004E"].__setitem__(
                "label", "Estimate!!Total!!Total population!!AGE!!10 to 15 years"
            ),
            "PUBLISHED_LABEL:S0101_C01_004",
        ),
        (
            lambda variables: variables["S0101_C01_004E"].__setitem__(
                "predicateType", "float"
            ),
            "PUBLISHED_NOT_A_COUNT:S0101_C01_004",
        ),
        (
            lambda variables: variables["S0101_C01_004E"].__setitem__("group", "S0102"),
            "METADATA_GROUP:S0101_C01_004E",
        ),
        (
            lambda variables: variables["S0101_C01_004E"].__setitem__(
                "concept", "Selected Housing Characteristics"
            ),
            "METADATA_CONCEPT:S0101_C01_004E",
        ),
        (
            lambda variables: variables.pop("S0101_C01_004E"),
            "METADATA_VARIABLE_MISSING:S0101_C01_004E",
        ),
    ],
)
def test_publisher_labels_groups_and_units_are_verified(tmp_path, edit, reason):
    metadata, data = _invented_documents()
    edit(metadata["variables"])
    with pytest.raises(ActivationError, match=reason):
        activate_invented(tmp_path / "bundle", documents=(metadata, data))


def test_a_swapped_raw_response_is_refused(tmp_path):
    root = tmp_path / "bundle"
    digest = invented_bundle(root)
    inventory = json.loads((root / INVENTORY).read_text())
    data = next(
        entry for entry in inventory["sources"]["census"] if entry["kind"] == "data"
    )
    payload = json.loads((root / data["path"]).read_bytes())
    payload[1][payload[0].index("S0101_C01_002E")] = "1"
    (root / data["path"]).write_text(json.dumps(payload))
    with pytest.raises(ActivationError, match="PINNED_DIGEST_MISMATCH:response:data"):
        activate_national_age_targets(root, declaration=invented_declaration(digest))


def test_a_tampered_request_descriptor_is_refused(tmp_path):
    root = tmp_path / "bundle"
    digest = invented_bundle(root)
    descriptor = (
        root / "requests" / f"{hashlib.sha256(_DATA_URL.encode()).hexdigest()}.json"
    )
    record = json.loads(descriptor.read_text())
    record["retrieved_at"] = "2020-01-01T00:00:00+00:00"
    descriptor.write_text(json.dumps(record))
    with pytest.raises(ActivationError, match="REQUEST_DESCRIPTOR_MISMATCH:data"):
        activate_national_age_targets(root, declaration=invented_declaration(digest))


def test_a_missing_request_descriptor_is_refused(tmp_path):
    root = tmp_path / "bundle"
    digest = invented_bundle(root)
    (
        root / "requests" / f"{hashlib.sha256(_DATA_URL.encode()).hexdigest()}.json"
    ).unlink()
    with pytest.raises(ActivationError, match="REQUEST_DESCRIPTOR_MISSING:data"):
        activate_national_age_targets(root, declaration=invented_declaration(digest))


def test_a_tampered_inventory_is_refused_by_digest(tmp_path):
    root = tmp_path / "bundle"
    digest = invented_bundle(root)
    document = json.loads((root / INVENTORY).read_text())
    document["purpose"] = "edited"
    (root / INVENTORY).write_text(json.dumps(document))
    with pytest.raises(ActivationError, match="PINNED_DIGEST_MISMATCH:inventory"):
        activate_national_age_targets(root, declaration=invented_declaration(digest))


def test_a_missing_bundle_is_refused(tmp_path):
    with pytest.raises(ActivationError, match="PINNED_FILE_MISSING:inventory"):
        activate_national_age_targets(tmp_path / "absent")


# --------------------------------------------------------------------------
# The accepted local bundle
# --------------------------------------------------------------------------


@pinned_bundle
def test_pinned_bundle_activates_to_the_published_age_distribution():
    registry = activate_national_age_targets(BUNDLE)
    assert len(registry) == 18
    assert tuple(spec.name for spec in registry) == tuple(
        band.variable for band in NATIONAL_AGE_ACTIVATION.bands
    )
    assert tuple(spec.measure for spec in registry) == band_columns()

    inventory = json.loads((BUNDLE / INVENTORY).read_text())
    cells = {cell["variable"]: cell for cell in inventory["acs"]["S0101"]["cells"]}
    for spec in registry:
        assert spec.value == float(cells[spec.name]["estimate"]["raw"])
        assert spec.se == pytest.approx(float(cells[spec.name]["moe"]["raw"]) / 1.645)
        assert spec.metadata["reference_sha256"] == ACCEPTED_INVENTORY_SHA256
        assert activation_digest(NATIONAL_AGE_ACTIVATION) in spec.notes
        assert "api.census.gov" in spec.source
    assert sum(spec.value for spec in registry) == float(
        cells["S0101_C01_001"]["estimate"]["raw"]
    )


@pinned_bundle
def test_activation_does_not_write_to_the_pinned_bundle(tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(BUNDLE, root)
    before = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    activate_national_age_targets(root)
    after = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert before == after
    assert before[Path(INVENTORY)] == ACCEPTED_INVENTORY_SHA256
