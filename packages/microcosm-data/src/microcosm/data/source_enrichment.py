"""Additive native-input releases that inherit an immutable calibration.

This release type does not certify a new calibration or upgrade its schema.
Its authority is the reviewed parent byte identity, an exhaustive H5 comparison,
Census source reconciliation, and separately measured native-loader compatibility.
Publication replays the latter checks before the Hub client is constructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path

from microcosm.data.contract import (
    COMPATIBILITY_CLAIM_DECLARER_MAX_CHARS,
    PUBLISHER_CLAIM_BASIS,
    ReleaseContractError,
    compatibility_claim_declarer_error,
)

SOURCE_ENRICHMENT_RELEASE_TYPE = "source_enrichment"
SOURCE_ENRICHMENT_FILE = "source_enrichment.json"
ROLE_VARIABLE = "is_spm_independent_minor_role"
EVIDENCE_COLUMN = "is_spm_independence_role"
SOURCE_EVIDENCE_FILE = "source_spm_person_independence.csv"
SOURCE_EVIDENCE_SHA256 = (
    "22b5968d90fecfeef7614583e493fe10cc16bda8b5be82e6f49a5bc2102d3ce5"
)
SOURCE_PROVENANCE_FILE = "source_spm_independence_provenance.json"
COMPATIBILITY_FILE = "source_enrichment_compatibility.json"
#: The compatibility field a publisher may widen. Core stays pinned exactly to
#: the tested version, as it always has: no producer can declare a Core range,
#: so the validator must not honour one either.
CLAIM_FIELD = "model"
#: Second boundedness probe for a publisher claim. A claim may exclude the next
#: major version by name, so the guard also asks whether it still admits a
#: version no release will reach.
_FAR_FUTURE_PROBE_MAJOR = 99999
COMPATIBILITY_PACKAGES = (
    "policyengine-us",
    "policyengine-core",
    "policyengine",
    "spm-calculator",
)
LOADED_SOURCE_PACKAGES = {
    "country_loader": "policyengine-us",
    "wrapper_loader": "policyengine",
    "native_role": "spm-calculator",
}
PARENT_BUILD_ID = "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
PARENT_DATASET_SHA256 = (
    "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"
)
# Reviewed immutable BuildP evidence. Callers cannot grant another schema-5
# calibration permission to use the inheritance lane by supplying their own pins.
PARENT_FILES = {
    "parent_release_manifest.json": (
        "dd949ba3c4c7a56aff8442c6db5a031d2c84c3da59f5d14832f468c358604506"
    ),
    "parent_build_manifest.json": (
        "1990e8fd37ce66f499b8e6600c5896701bb07015be132a12de64e321a2edc67e"
    ),
    "calibration_diagnostics.json": (
        "870449b44e86b13b25bcea1a57f0e7af37f4d4db18be815eea3acdf9fe6eb40e"
    ),
    "us_source_coverage.json": (
        "6406c8686c292015a5bc7265a42402f89935f9395a8b63510efdd4d9562e7ff5"
    ),
}
CENSUS_PERSON_PINS = {
    2023: "19b56537e50e7663f954361ef2bb5ce9cef8d9d45f156fe1a69a99b654198ffe",
    2024: "21a2b9e0e4b08534563578a45acad77868af4ae9a7d46f23776b707d4a559aa7",
    2025: "06921fe83fc66c907e6c7b86b82255dc70458ee7d76258fc48297cb34f0c06b5",
}
# Match the producer's education_assistance_source archive pins without making
# the data/consumer shard depend on the build shard. A build test checks parity.
CENSUS_ARCHIVE_PINS = {
    2023: {
        "income_year": 2022,
        "official_archive_url": (
            "https://www2.census.gov/programs-surveys/cps/datasets/2023/"
            "march/asecpub23csv.zip"
        ),
        "archive_sha256": "d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11",
        "member": "pppub23.csv",
    },
    2024: {
        "income_year": 2023,
        "official_archive_url": (
            "https://www2.census.gov/programs-surveys/cps/datasets/2024/"
            "march/asecpub24csv.zip"
        ),
        "archive_sha256": "cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09",
        "member": "pppub24.csv",
    },
    2025: {
        "income_year": 2024,
        "official_archive_url": (
            "https://www2.census.gov/programs-surveys/cps/datasets/2025/"
            "march/asecpub25csv.zip"
        ),
        "archive_sha256": "318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b",
        "member": "pppub25.csv",
    },
}
EXPECTED_COUNTS = {
    "persons_joined": 166321,
    "native_spm_units": 59900,
    "total_source_people": 432523,
    "total_source_units": 176039,
    "minor_only_units_resolved": 28,
    "classification_changed_units_vs_age_only": 132,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRODUCER_SOURCE_FILES = (
    "tools/build_us_spm_role_enrichment.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/spm_role_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
)
#: The native SPM role lane's operation; everything above is its lineage.
SPM_ROLE_OPERATION = "add_native_spm_independent_minor_role"

# Reported-receipt enrichment of the national default (microcosm#978). A second
# operation of this release type, not a second release type: the same manifest,
# inheritance, H5 replay, compatibility and producer-identity gates, pinned to
# its own reviewed parent. Like the Build P pins above, these are constants a
# caller cannot supply.

#: The operation the donor receipt qualification records
#: (``tools/build_us_acs_donor_receipt_qualification.py``), reused so the
#: release names exactly what its receipt proves.
RECEIPT_OPERATION = "add_reported_receipt_inputs"
#: The national default ``latest.json`` names from 2026-09-15. The Hub also
#: carries these H5 bytes as ``populace-us-2024-spm-20260909``; this id is the
#: one whose release manifest bytes are pinned below.
RECEIPT_PARENT_BUILD_ID = "populace-us-2024-spm-20260915"
RECEIPT_PARENT_DATASET_SHA256 = (
    "6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84"
)
#: Reviewed immutable evidence of that parent, copied unchanged into the child.
#: The parent release manifest hash-binds the parent H5 and every other parent
#: file (its build manifest, source-enrichment report, compatibility receipt and
#: role evidence). The diagnostics and coverage bytes are Build P's own
#: (``PARENT_FILES``), inherited unchanged through the SPM-role release, so
#: this lane grants no calibration an inheritance it did not already have.
RECEIPT_PARENT_FILES = {
    "parent_release_manifest.json": (
        "d5c9e2a33d097294af60a816fafb1bdcfbbd81cdda70ae4637daceb829dedc01"
    ),
    "calibration_diagnostics.json": (
        "870449b44e86b13b25bcea1a57f0e7af37f4d4db18be815eea3acdf9fe6eb40e"
    ),
    "us_source_coverage.json": (
        "6406c8686c292015a5bc7265a42402f89935f9395a8b63510efdd4d9562e7ff5"
    ),
}
#: Added input -> owning entity, in the order the qualification appends them.
RECEIPT_COLUMNS = {
    "receives_wic": "person",
    "receives_snap": "spm_unit",
    "receives_tanf": "spm_unit",
}
#: The published root path. Never the national default's ``populace_us_2024.h5``,
#: so no publication of this donor can overwrite the default's root file.
RECEIPT_DATASET_FILENAME = "populace_us_2024_receipt_qualified.h5"
#: The qualification's aggregate receipt, carried verbatim as source evidence.
RECEIPT_QUALIFICATION_FILE = "donor_receipt_qualification.json"
#: The qualification tool's ``_PRODUCER_FILES``: every file whose code decides a
#: receipt value. Declared here, not imported, so the data shard does not depend
#: on a tool; a build test checks parity.
RECEIPT_QUALIFICATION_SOURCE_FILES = (
    "tools/build_us_acs_donor_receipt_qualification.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/public_assistance_type_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/education_assistance_source.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
    "packages/microcosm-frame/src/microcosm/frame/bundle.py",
    "packages/microcosm-frame/src/microcosm/frame/schema.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/h5_boolean_append.py",
)
#: The release assembler's producer inventory, authenticated at publication.
RECEIPT_RELEASE_PRODUCER_FILES = (
    "tools/build_us_receipt_enrichment_release.py",
    "packages/microcosm-data/src/microcosm/data/h5_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/h5_boolean_append.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
)
#: Native inputs the loader qualification checks for a receipt child: the role
#: it inherits from its parent, then the three receipts.
RECEIPT_NATIVE_INPUTS = {ROLE_VARIABLE: "person", **RECEIPT_COLUMNS}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, failures: list[str]) -> dict:
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        return payload
    except (OSError, ValueError) as exc:
        failures.append(f"{path.name}: {exc}")
        return {}


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _check_person_evidence(path: Path, candidate: Path) -> dict:
    """Independently verify a complete, ordered, one-to-one native input join."""
    import h5py
    import numpy as np

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["person_id", "person_spm_unit_id", EVIDENCE_COLUMN]
        if reader.fieldnames != expected:
            raise ValueError(f"{path.name} must have exactly {expected}")
        rows = list(reader)
    if any(row[EVIDENCE_COLUMN] not in {"False", "True"} for row in rows):
        raise ValueError("source person evidence roles must be literal False or True")
    ids = np.array([int(row["person_id"]) for row in rows], dtype=np.int64)
    units = np.array([int(row["person_spm_unit_id"]) for row in rows], dtype=np.int64)
    roles = np.array([row[EVIDENCE_COLUMN] == "True" for row in rows], dtype=bool)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("source person evidence has duplicate person IDs")
    with h5py.File(candidate, "r") as h5:
        table = h5["person/table"]
        for field, values in (
            ("person_id", ids),
            ("person_spm_unit_id", units),
            (ROLE_VARIABLE, roles),
        ):
            if not np.array_equal(table[field], values):
                raise ValueError(
                    f"source person evidence does not match native {field}"
                )
        role_index = table.dtype.names.index(ROLE_VARIABLE)
        if table.id.get_type().get_member_type(role_index) != h5py.h5t.NATIVE_B8 or (
            table.attrs.get(f"{ROLE_VARIABLE}_dtype") != b"bool"
        ):
            raise ValueError("native person role must have PyTables bool bitfield type")
    return {"persons_joined": len(ids), "native_spm_units": len(np.unique(units))}


def _check_source_provenance(provenance: Mapping, failures: list[str]) -> None:
    if provenance.get("dataset_sha256") != PARENT_DATASET_SHA256:
        failures.append(
            "source provenance dataset_sha256 must identify reviewed BuildP"
        )
    for key, expected in EXPECTED_COUNTS.items():
        if type(provenance.get(key)) is not int or provenance[key] != expected:
            failures.append(f"source provenance {key} must equal {expected}")
    for key in ("unmatched_persons", "adult_child_person_count_mismatch_units"):
        if type(provenance.get(key)) is not int or provenance[key] != 0:
            failures.append(f"source provenance {key} must equal zero")
    if (
        provenance.get("complete_source_membership_units")
        != EXPECTED_COUNTS["native_spm_units"]
    ):
        failures.append("source provenance must reconcile every native SPM unit")
    for key in ("weights_used", "ages_changed"):
        if provenance.get(key) is not False:
            failures.append(f"source provenance {key} must be false")
    if provenance.get("primitive_column") != ROLE_VARIABLE:
        failures.append("source provenance primitive_column must name the native role")
    if provenance.get("evidence_column") != EVIDENCE_COLUMN:
        failures.append("source provenance evidence_column must name the source role")
    checks = provenance.get("source_checks")
    if not isinstance(checks, list) or len(checks) != len(CENSUS_PERSON_PINS):
        failures.append("source provenance must carry every pinned Census source check")
        return
    seen = set()
    for raw in checks:
        row = _mapping(raw)
        year = row.get("survey_year")
        if type(year) is not int or year not in CENSUS_PERSON_PINS or year in seen:
            failures.append("source provenance has an unknown or duplicate Census year")
            continue
        seen.add(year)
        if row.get("csv_sha256") != CENSUS_PERSON_PINS[year]:
            failures.append(f"source provenance Census {year} CSV hash differs")
        if row.get("adult_child_person_count_mismatch_units") != 0:
            failures.append(
                f"source provenance Census {year} count reconciliation failed"
            )
        for field, expected in CENSUS_ARCHIVE_PINS[year].items():
            if row.get(field) != expected:
                failures.append(
                    f"source provenance Census {year} pinned archive {field} differs"
                )


def _receipt_added_variables() -> list[dict]:
    """The qualification's ``added_columns`` for the three receipts, in order."""
    return [
        {"name": name, "entity": entity, "dtype": "bool"}
        for name, entity in RECEIPT_COLUMNS.items()
    ]


def _receipt_plan() -> dict[str, tuple[str, ...]]:
    """The Boolean-append plan the qualification wrote: group -> ordered names."""
    plan: dict[str, tuple[str, ...]] = {}
    for name, entity in RECEIPT_COLUMNS.items():
        plan[entity] = (*plan.get(entity, ()), name)
    return plan


def _receipt_column_counts(candidate: Path) -> dict[str, dict[str, int]]:
    """Rows and true values of each appended receipt column, read from the H5."""
    import h5py
    import numpy as np

    counts = {}
    with h5py.File(candidate, "r") as h5:
        for name, entity in RECEIPT_COLUMNS.items():
            values = np.asarray(h5[f"{entity}/table"][name])
            counts[name] = {
                "rows": int(values.size),
                "true": int(np.count_nonzero(values)),
            }
    return counts


def _check_receipt_qualification(
    release_dir: Path, report: Mapping, candidate: Path | None, failures: list[str]
) -> None:
    """Bind the qualification receipt to this release, its parent and its H5.

    The receipt is the qualification tool's own aggregate record, carried
    verbatim. Its preservation report must equal the release report's, which
    the caller has just compared with a fresh exhaustive comparison of the
    actual parent and candidate H5 files; its per-column totals are recounted
    here from the candidate. The receipt's producer identity is authenticated
    at certification and publication (:func:`_check_receipt_producers`).
    """
    source = _mapping(report.get("source"))
    path = release_dir / RECEIPT_QUALIFICATION_FILE
    if (
        source.get("qualification_filename") != RECEIPT_QUALIFICATION_FILE
        or not path.is_file()
        or sha256_file(path) != source.get("qualification_sha256")
    ):
        failures.append(
            "source enrichment qualification must bind the immutable "
            f"{RECEIPT_QUALIFICATION_FILE}"
        )
    if not path.is_file():
        return
    receipt = _json(path, failures)
    if receipt.get("schema_version") != 1 or receipt.get("operation") != (
        RECEIPT_OPERATION
    ):
        failures.append(
            f"{RECEIPT_QUALIFICATION_FILE} must be a schema-1 {RECEIPT_OPERATION} receipt"
        )
    if _mapping(receipt.get("parent")).get("sha256") != RECEIPT_PARENT_DATASET_SHA256:
        failures.append(
            f"{RECEIPT_QUALIFICATION_FILE} must name the pinned national default "
            "as its parent"
        )
    if _mapping(receipt.get("dataset")).get("sha256") != _mapping(
        report.get("dataset")
    ).get("sha256"):
        failures.append(f"{RECEIPT_QUALIFICATION_FILE} must describe the enriched H5")
    if receipt.get("added_columns") != _receipt_added_variables():
        failures.append(
            f"{RECEIPT_QUALIFICATION_FILE} must add exactly the three "
            "reported-receipt bool inputs"
        )
    if receipt.get("preservation") != report.get("preservation"):
        failures.append(
            f"{RECEIPT_QUALIFICATION_FILE} preservation must equal the release's "
            "replayed H5 comparison"
        )
    counts = _mapping(receipt.get("counts"))
    actual = None
    if candidate is not None and candidate.is_file():
        try:
            actual = _receipt_column_counts(candidate)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"receipt columns could not be counted in the H5: {exc}")
    for name, entity in RECEIPT_COLUMNS.items():
        recorded = _mapping(counts.get(name))
        true = recorded.get("true")
        channel_true = _mapping(recorded.get("gate_selected_channel")).get("true")
        if (
            recorded.get("entity") != entity
            or type(recorded.get("rows")) is not int
            or type(true) is not int
            or recorded.get("false") != recorded["rows"] - true
        ):
            failures.append(
                f"{RECEIPT_QUALIFICATION_FILE} counts for {name} are malformed"
            )
            continue
        if true <= 0 or type(channel_true) is not int or channel_true <= 0:
            failures.append(
                f"{RECEIPT_QUALIFICATION_FILE} records no true {name} in the donor "
                "or in its gate-selected channel"
            )
        if actual is not None and (
            recorded["rows"] != actual[name]["rows"] or true != actual[name]["true"]
        ):
            failures.append(
                f"{RECEIPT_QUALIFICATION_FILE} counts for {name} differ from the "
                "enriched H5"
            )


def _check_receipt_producers(
    build: Mapping, release_dir: Path, failures: list[str]
) -> None:
    """Authenticate both producers of a receipt child against this checkout.

    The release assembler's build identity covers the files that packaged the
    release; the qualification receipt's covers every file that decided a
    receipt value, including the verifier the contract has just replayed.
    """
    try:
        _check_producer_source_identity(
            _mapping(build.get("code")), inventory=RECEIPT_RELEASE_PRODUCER_FILES
        )
    except (OSError, ValueError) as exc:
        failures.append(f"producer source identity failed: {exc}")
    try:
        receipt = json.loads((release_dir / RECEIPT_QUALIFICATION_FILE).read_text())
        _check_producer_source_identity(
            _mapping(_mapping(receipt).get("code")),
            inventory=RECEIPT_QUALIFICATION_SOURCE_FILES,
        )
    except (OSError, ValueError) as exc:
        failures.append(f"qualification producer source identity failed: {exc}")


def is_receipt_enrichment(release_dir: Path | str) -> bool:
    """Whether ``release_dir`` declares the reported-receipt operation.

    A reporting probe for the publisher's latest-pointer guard; it validates
    nothing, and an unreadable report reads as ``False``.
    """
    try:
        report = json.loads((Path(release_dir) / SOURCE_ENRICHMENT_FILE).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(report, Mapping) and report.get("operation") == (
        RECEIPT_OPERATION
    )


def validate_source_enrichment_candidate(
    release_dir: Path | str,
    *,
    parent_h5: Path | str | None,
    artifact_root: Path | str | None,
    require_compatibility: bool = False,
    compatibility_wheels: tuple[Path | str, ...] = (),
) -> dict:
    """Check the inheritance contract; pending candidates cannot be published.

    Unlike the standard release contract, this never treats parent diagnostics
    as measurements of the enriched model. Both actual H5 files are mandatory:
    a hand-written preservation receipt is not sufficient evidence.

    ``source_enrichment.json``'s ``operation`` selects the reviewed lineage:
    the native SPM role added to Build P (every other value, so an unknown
    operation is still judged, and refused, as the role lane), or
    :data:`RECEIPT_OPERATION`, the reported-receipt inputs added to the pinned
    national default. The gates are the same; only the pinned parent, the
    evidence that proves the addition and the native inputs the loader
    qualification checks differ.
    """
    from microcosm.data.annual_projections import validate_annual_projection_extension
    from microcosm.data.h5_boolean_append import compare_boolean_append
    from microcosm.data.h5_enrichment import compare_h5_enrichment

    release_dir = Path(release_dir)
    failures: list[str] = []
    manifest = _json(release_dir / "release_manifest.json", failures)
    try:
        annual_extension = validate_annual_projection_extension(
            release_dir, manifest, artifact_root=artifact_root
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReleaseContractError(
            release_dir, [f"annual projection extension: {exc}"]
        ) from exc
    build = _json(release_dir / "build_manifest.json", failures)
    report = _json(release_dir / SOURCE_ENRICHMENT_FILE, failures)
    receipts = report.get("operation") == RECEIPT_OPERATION
    # Read at call time, so each lineage is exactly its module-level pins.
    lineage_build_id = RECEIPT_PARENT_BUILD_ID if receipts else PARENT_BUILD_ID
    lineage_dataset_sha256 = (
        RECEIPT_PARENT_DATASET_SHA256 if receipts else PARENT_DATASET_SHA256
    )
    lineage_files = RECEIPT_PARENT_FILES if receipts else PARENT_FILES
    reused_release_ids = (
        {PARENT_BUILD_ID, RECEIPT_PARENT_BUILD_ID} if receipts else {PARENT_BUILD_ID}
    )
    if manifest.get("release_type") != SOURCE_ENRICHMENT_RELEASE_TYPE:
        failures.append(
            "release_manifest.json must declare release_type=source_enrichment"
        )
    if manifest.get("schema_version") != 1:
        failures.append(
            "source enrichment release manifest schema_version must equal 1"
        )
    if _mapping(manifest.get("build")).get("build_id") != release_dir.name:
        failures.append(
            "release manifest build_id must match its new release directory"
        )
    if release_dir.name in reused_release_ids or not release_dir.name.startswith(
        "populace-us-2024-"
    ):
        failures.append("source enrichment requires a NEW US 2024 release id")
    if manifest.get("dataset_role", "national_default") != "national_default":
        failures.append("source enrichment supports only the national default dataset")
    if build.get("build_id") != release_dir.name:
        failures.append("build manifest build_id must match the new release directory")
    if build.get("release_type") != SOURCE_ENRICHMENT_RELEASE_TYPE:
        failures.append("build manifest must declare source_enrichment")
    if report.get("schema_version") != 1 or report.get("release_type") != (
        SOURCE_ENRICHMENT_RELEASE_TYPE
    ):
        failures.append(
            "source_enrichment.json must declare source_enrichment schema 1"
        )
    if not receipts and report.get("operation") != SPM_ROLE_OPERATION:
        failures.append("source enrichment operation must add only the native SPM role")
    parent = _mapping(report.get("parent"))
    expected_parent = {
        "build_id": lineage_build_id,
        "repo_id": "policyengine/populace-us",
        "revision": lineage_build_id,
        "dataset_sha256": lineage_dataset_sha256,
        "calibration_diagnostics_schema_version": 5,
        "files": lineage_files,
    }
    if parent != expected_parent:
        failures.append(
            "source enrichment parent must match the pinned national default identity"
            if receipts
            else "source enrichment parent must match the reviewed BuildP identity"
        )
    for name, expected in lineage_files.items():
        path = release_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(f"inherited {name} must preserve the reviewed parent bytes")
    diagnostics = _json(release_dir / "calibration_diagnostics.json", failures)
    if (
        type(diagnostics.get("schema_version")) is not int
        or diagnostics.get("schema_version") != 5
    ):
        failures.append("inherited calibration_diagnostics must retain schema 5")
    if receipts:
        if report.get("added_variables") != _receipt_added_variables():
            failures.append(
                "added_variables must declare exactly the three reported-receipt "
                "bool inputs, in their append order"
            )
    elif report.get("added_variable") != {
        "name": ROLE_VARIABLE,
        "entity": "person",
        "dtype": "bool",
        "age_gate_applied": False,
    }:
        failures.append(
            "added_variable must declare the person bool primitive before age gate"
        )
    dataset = _mapping(report.get("dataset"))
    filename = dataset.get("filename")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not (filename.endswith(".h5"))
    ):
        failures.append("source enrichment dataset.filename must be a bare H5 filename")
        filename = None
    if receipts and filename is not None and filename != RECEIPT_DATASET_FILENAME:
        failures.append(
            f"a reported-receipt enrichment publishes its H5 as "
            f"{RECEIPT_DATASET_FILENAME}, never over the national default's root file"
        )
    if filename:
        release_local_h5 = release_dir / filename
        if release_local_h5.exists() or release_local_h5.is_symlink():
            # The publisher treats release-local files as release-dir uploads
            # and suppresses their root uploads, even when the bytes match.
            # Reject ambiguity before compatibility imports or Hub activity.
            failures.append(
                f"source enrichment rejects release-local H5 {filename}; "
                "the native dataset must exist only in artifact_root"
            )
            raise ReleaseContractError(release_dir, failures)
    candidate = Path(artifact_root) / filename if artifact_root and filename else None
    if parent_h5 is None or candidate is None:
        failures.append(
            "source enrichment requires parent_h5 and artifact_root for exact H5 verification"
        )
    elif not Path(parent_h5).is_file() or not candidate.is_file():
        failures.append("source enrichment parent and candidate H5 files must exist")
    else:
        if sha256_file(parent_h5) != lineage_dataset_sha256:
            failures.append(
                "parent H5 SHA256 differs from the pinned national default"
                if receipts
                else "parent H5 SHA256 differs from reviewed BuildP"
            )
        if candidate.resolve() == Path(parent_h5).resolve():
            failures.append("source enrichment must create a NEW H5")
        if sha256_file(candidate) != dataset.get("sha256"):
            failures.append("candidate H5 SHA256 differs from source enrichment report")
        try:
            comparison = (
                compare_boolean_append(Path(parent_h5), candidate, _receipt_plan())
                if receipts
                else compare_h5_enrichment(Path(parent_h5), candidate)
            )
            if report.get("preservation") != comparison:
                failures.append(
                    "preservation report differs from actual exhaustive H5 comparison"
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"exact H5 preservation failed: {exc}")
    if _mapping(build.get("dataset")) != dataset:
        failures.append("build manifest dataset must match the new H5 identity")
    calibration = _mapping(build.get("calibration"))
    if calibration != {
        "mode": "inherited",
        "parent_build_id": lineage_build_id,
        "diagnostics_sha256": lineage_files["calibration_diagnostics.json"],
        "diagnostics_schema_version": 5,
    }:
        failures.append(
            "build calibration must explicitly inherit schema-5 parent evidence"
        )
    if receipts:
        _check_receipt_qualification(release_dir, report, candidate, failures)
        required_artifacts = {
            *lineage_files,
            SOURCE_ENRICHMENT_FILE,
            RECEIPT_QUALIFICATION_FILE,
        }
    else:
        source = _mapping(report.get("source"))
        for field, expected_name in (
            ("person_evidence", SOURCE_EVIDENCE_FILE),
            ("provenance", SOURCE_PROVENANCE_FILE),
        ):
            name = source.get(f"{field}_filename")
            path = release_dir / expected_name
            if (
                name != expected_name
                or not path.is_file()
                or sha256_file(path) != source.get(f"{field}_sha256")
            ):
                failures.append(
                    f"source enrichment {field} must bind the immutable {expected_name}"
                )
        evidence_path = release_dir / SOURCE_EVIDENCE_FILE
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != SOURCE_EVIDENCE_SHA256
        ):
            failures.append(
                "source evidence must match the independently reviewed Census-derived person table"
            )
        provenance = _json(release_dir / SOURCE_PROVENANCE_FILE, failures)
        _check_source_provenance(provenance, failures)
        if report.get("reconciliation") != provenance:
            failures.append(
                "source enrichment reconciliation must equal source provenance"
            )
        if (
            candidate
            and candidate.is_file()
            and (release_dir / SOURCE_EVIDENCE_FILE).is_file()
        ):
            try:
                counts = _check_person_evidence(
                    release_dir / SOURCE_EVIDENCE_FILE, candidate
                )
                if any(provenance.get(key) != value for key, value in counts.items()):
                    failures.append(
                        "source provenance coverage differs from actual person evidence"
                    )
            except (ValueError, KeyError, OSError, TypeError) as exc:
                failures.append(f"native source evidence reconciliation failed: {exc}")
        required_artifacts = {
            *PARENT_FILES,
            SOURCE_ENRICHMENT_FILE,
            SOURCE_EVIDENCE_FILE,
            SOURCE_PROVENANCE_FILE,
        }
    artifacts = _mapping(manifest.get("artifacts"))
    annual_artifacts = annual_extension.additional_artifacts if annual_extension else {}
    revision = annual_extension.revision if annual_extension else release_dir.name
    by_path = {}
    for key, raw in artifacts.items():
        entry = _mapping(raw)
        path = entry.get("path")
        if not isinstance(path, str) or (
            key not in annual_artifacts and Path(path).name != path
        ):
            failures.append(
                f"source enrichment artifact {key} must use a bare filename"
            )
            continue
        if path in by_path:
            failures.append(f"source enrichment duplicate artifact path {path}")
        by_path[path] = entry
        if (
            entry.get("repo_id") != "policyengine/populace-us"
            or entry.get("revision") != revision
        ):
            failures.append(
                f"source enrichment artifact {key} must pin the new repo/tag"
            )
        local = (
            annual_artifacts[key]
            if key in annual_artifacts
            else candidate
            if path == filename
            else release_dir / path
        )
        if (
            local is None
            or not local.is_file()
            or sha256_file(local) != entry.get("sha256")
        ):
            failures.append(f"source enrichment artifact {key} hash/file mismatch")
    if not required_artifacts.issubset(by_path):
        failures.append(
            "release manifest must list every inherited and source enrichment artifact"
        )
    national = _mapping(manifest.get("default_datasets")).get("national")
    native_artifact = (
        _mapping(artifacts.get(national)) if isinstance(national, str) else {}
    )
    if (
        native_artifact.get("path") != filename
        or native_artifact.get("kind") != "microdata"
    ):
        failures.append("default_datasets.national must select the enriched native H5")
    if (
        len(
            [
                entry
                for key, entry in artifacts.items()
                if _mapping(entry).get("kind") == "microdata"
                and (
                    annual_extension is None
                    or key not in annual_extension.projected_artifacts
                )
            ]
        )
        != 1
    ):
        failures.append(
            "source enrichment must deliver exactly one native H5 microdata artifact"
        )
    compatibility = _mapping(report.get("compatibility"))
    if compatibility.get("status") == "pending":
        if require_compatibility:
            failures.append(
                "source enrichment compatibility is pending; run real candidate-wheel native loader qualification"
            )
        if manifest.get("compatible_core_packages") or manifest.get(
            "compatible_model_packages"
        ):
            failures.append(
                "pending source enrichment must not claim model/Core compatibility"
            )
        if any(
            key in _mapping(manifest.get("build"))
            for key in ("built_with_core_package", "built_with_model_package")
        ):
            failures.append(
                "pending source enrichment must not copy built-with package claims"
            )
        if compatibility.get("publisher_claims") is not None:
            failures.append(
                "pending source enrichment must not declare a publisher "
                "compatibility claim; a claim is made at certification, "
                "against the measured runtime"
            )
    elif compatibility.get("status") == "passed":
        if COMPATIBILITY_FILE not in by_path:
            failures.append(
                "release manifest must deliver the native compatibility receipt"
            )
        _check_compatibility(
            release_dir,
            manifest,
            compatibility,
            candidate,
            require_compatibility,
            compatibility_wheels,
            failures,
            annual_revision=annual_extension.revision if annual_extension else None,
            native_inputs=RECEIPT_NATIVE_INPUTS if receipts else None,
        )
    else:
        failures.append(
            "source enrichment compatibility status must be pending or passed"
        )
    if require_compatibility and receipts:
        _check_receipt_producers(build, release_dir, failures)
    elif require_compatibility:
        try:
            _check_producer_source_identity(_mapping(build.get("code")))
        except (OSError, ValueError) as exc:
            failures.append(f"producer source identity failed: {exc}")
    if failures:
        raise ReleaseContractError(release_dir, failures)
    return report


def _check_producer_source_identity(
    code: Mapping, *, inventory: tuple[str, ...] | None = None
) -> None:
    """Bind clean build provenance to committed and currently executed sources.

    Publication/certification runs from a Microcosm checkout. A later docs-only
    commit is fine; each reviewed producer file must still have the exact bytes
    recorded in the build and in its immutable producer commit.

    ``inventory`` defaults to the native SPM role lane's
    :data:`PRODUCER_SOURCE_FILES`. Whichever data-contract modules the
    inventory names must also be the modules executing this check.
    """
    import subprocess

    from microcosm.data import contract, h5_boolean_append, h5_enrichment

    if inventory is None:
        inventory = PRODUCER_SOURCE_FILES

    commit = code.get("git_commit")
    if (
        code.get("git_dirty") is not False
        or not isinstance(commit, str)
        or not (re.fullmatch(r"[0-9a-f]{40}", commit))
    ):
        raise ValueError("publication requires a recorded clean producer commit")
    hashes = _mapping(code.get("source_files_sha256"))
    if set(hashes) != set(inventory) or any(
        not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
        for value in hashes.values()
    ):
        raise ValueError("producer must hash the exact reviewed source-file inventory")

    def git(*args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args], check=True, capture_output=True
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                "producer commit/source cannot be authenticated in the current Git checkout"
            ) from exc

    checkout = Path(git("rev-parse", "--show-toplevel").decode().strip())
    verified_commit = (
        git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    )
    if verified_commit != commit:
        raise ValueError("producer git_commit does not resolve to the recorded commit")
    for relative in inventory:
        committed = git("show", f"{commit}:{relative}")
        if hashlib.sha256(committed).hexdigest() != hashes[relative]:
            raise ValueError(
                f"producer committed source differs from recorded hash: {relative}"
            )
        current = checkout / relative
        if not current.is_file() or sha256_file(current) != hashes[relative]:
            raise ValueError(
                f"producer checkout source differs from recorded hash: {relative}"
            )
    executing = {
        "packages/microcosm-data/src/microcosm/data/h5_enrichment.py": h5_enrichment.__file__,
        "packages/microcosm-data/src/microcosm/data/h5_boolean_append.py": (
            h5_boolean_append.__file__
        ),
        "packages/microcosm-data/src/microcosm/data/source_enrichment.py": __file__,
        "packages/microcosm-data/src/microcosm/data/contract.py": contract.__file__,
    }
    for relative, actual in executing.items():
        if relative not in inventory:
            continue
        if actual is None or sha256_file(actual) != hashes[relative]:
            raise ValueError(
                f"executing producer contract differs from recorded source: {relative}"
            )


def _claim_specifier_set(specifier: object):
    """Return the PEP 440 set ``specifier`` denotes, or raise ``ValueError``.

    One copy of the specifier rules for both callers, so the requirement parser
    refuses at the door exactly what the entry builder would refuse after a
    qualification run.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    if not isinstance(specifier, str) or not specifier.strip():
        raise ValueError("publisher compatibility claim needs a PEP 440 specifier")
    try:
        specifier_set = SpecifierSet(specifier)
    except InvalidSpecifier as exc:
        raise ValueError(
            f"publisher compatibility claim {specifier!r} is not a valid PEP 440 "
            "specifier set"
        ) from exc
    if not tuple(specifier_set):
        raise ValueError(
            "publisher compatibility claim must constrain the version; an empty "
            "specifier claims every release"
        )
    return specifier_set


def parse_compatibility_claim_requirement(requirement: str, *, package: str) -> str:
    """Return the specifier of ``<package><specifier>`` as written, or raise.

    Spelling the package name into the claim is deliberate: the operator states
    which package the range is about, and a claim naming the wrong one is a
    typo the tooling must refuse rather than silently retarget. The specifier
    is returned as declared, not re-rendered, so the published claim reads back
    as the publisher wrote it.

    A bare name, or anything else that does not constrain the version, is
    refused here rather than after qualification: ``policyengine-us`` used to
    parse to ``""`` and cost the caller a whole native-loader run before the
    entry builder rejected it.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("publisher compatibility claim must not be empty")
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement as exc:
        raise ValueError(
            f"publisher compatibility claim {requirement!r} is not a PEP 508 "
            "requirement such as 'policyengine-us>=2.0.1,<2.1'"
        ) from exc
    if canonicalize_name(parsed.name) != canonicalize_name(package):
        raise ValueError(
            f"publisher compatibility claim names {parsed.name!r}; this claim "
            f"declares compatibility for the built-with package {package!r}"
        )
    if parsed.url or parsed.extras or parsed.marker:
        raise ValueError(
            "publisher compatibility claim must be a bare name and specifier, "
            "with no URL, extras or environment marker"
        )
    specifier = requirement.strip()[len(parsed.name) :].strip()
    # PEP 508 also allows ``name (>=1,<2)``; the parentheses are grammar, not
    # part of the specifier the manifest records.
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1].strip()
    _claim_specifier_set(specifier)
    return specifier


def check_compatibility_claim_declarer(declared_by: object) -> None:
    """Raise unless a claim names someone accountable for it.

    The rule itself lives in :func:`microcosm.data.contract` beside the release
    contract that re-checks it, so the producer cannot drift from the layer that
    reads a published bundle.
    """
    reason = compatibility_claim_declarer_error(declared_by)
    if reason is not None:
        raise ValueError(
            "publisher compatibility claim must record who declared it, as "
            f"trimmed printable text of at most "
            f"{COMPATIBILITY_CLAIM_DECLARER_MAX_CHARS} characters: declared_by "
            f"{reason}"
        )


def compatibility_claim_entry(
    specifier: object, *, package: str, version: str, declared_by: object
) -> dict:
    """Validate one publisher claim and return its manifest entry, or raise.

    Containment uses the same PEP 440 semantics the consumers apply
    (``microcosm.data.loader._package_certification`` and the wrapper's
    ``policyengine.provenance.manifest._specifier_matches``), so a claim this
    function accepts is a claim they will honour, and one they would refuse for
    the tested version is refused here instead of at load time.
    """
    from packaging.version import InvalidVersion, Version

    specifier_set = _claim_specifier_set(specifier)
    check_compatibility_claim_declarer(declared_by)
    try:
        tested = Version(version)
    except InvalidVersion as exc:
        raise ValueError(
            f"tested {package} version {version!r} is not a valid PEP 440 version"
        ) from exc
    if tested not in specifier_set:
        raise ValueError(
            f"publisher compatibility claim {specifier!r} excludes the tested "
            f"{package} version {version}; a claim must cover what was measured"
        )
    next_major = Version(f"{tested.epoch}!{tested.major + 1}.0.0")
    if next_major in specifier_set:
        raise ValueError(
            f"publisher compatibility claim {specifier!r} reaches "
            f"{next_major} and beyond; a claim measured against {package} "
            f"{version} must stop below the next major version, as "
            "'>=2.0.1,<2.1' or '~=2.0.1' do"
        )
    # A range that punches a hole at exactly the next major ('>=2.0.1,!=3.0.0')
    # passes the probe above while still certifying 4.x and 5.x, so probe far
    # past any version this package will plausibly reach as well. Two probes are
    # a bound check, not a proof of boundedness: a claim contrived to exclude
    # both while admitting versions between them would still pass.
    far_future = Version(f"{tested.epoch}!{_FAR_FUTURE_PROBE_MAJOR}.0.0")
    if far_future in specifier_set:
        raise ValueError(
            f"publisher compatibility claim {specifier!r} still admits "
            f"{far_future}; a claim measured against {package} {version} must "
            f"stop below {next_major} with an upper bound, not by excluding "
            "single versions from an open range"
        )
    # Bounding a claim above says nothing about how far below it reaches. A
    # bare '<2.1' contains the tested version and neither probe above, yet
    # certifies every release the package ever made — including ones predating
    # the native-input loader path this qualification measures. The probe sits
    # at the floor of the tested version's own epoch because a claim may mix
    # epochs: '>=2.0.1,<1!2.1' over a 1!2.0.1 build is open below within epoch
    # 1, admitting 1!0, while excluding the epoch-0 floor — which sorts under
    # every epoch-1 release, so a probe at a bare Version('0') would pass it.
    no_lower_bound = Version(f"{tested.epoch}!0")
    if no_lower_bound in specifier_set:
        raise ValueError(
            f"publisher compatibility claim {specifier!r} admits "
            f"{no_lower_bound}, far below the tested {package} version "
            f"{version}; a claim must also state a lower bound, as "
            "'>=2.0.1,<2.1' does"
        )
    return {
        "name": package,
        "specifier": specifier,
        "basis": PUBLISHER_CLAIM_BASIS,
        "declared_by": declared_by,
    }


def _claim_probe_versions(specifier_sets, tested):
    """Return versions to compare two claims over, lowest first.

    A finite search set, not an enumeration of PEP 440: every version the
    specifiers name, each of those and the tested version bumped by one micro,
    one minor and one major, plus one far-future release. A version found in
    here that one claim covers and the other does not is a real difference; not
    finding one is not proof that none exists.
    """
    from packaging.version import InvalidVersion, Version

    named = [tested]
    for specifier_set in specifier_sets:
        for specifier in specifier_set:
            try:
                named.append(Version(specifier.version.split("*")[0].rstrip(".")))
            except InvalidVersion:
                continue
    probes = {Version(f"{tested.epoch}!{_FAR_FUTURE_PROBE_MAJOR}.0.0")}
    for version in named:
        epoch, major, minor, micro = (
            version.epoch,
            version.major,
            version.minor,
            version.micro,
        )
        probes.update(
            {
                version,
                Version(f"{epoch}!{major}.{minor}.{micro + 1}"),
                Version(f"{epoch}!{major}.{minor + 1}.0"),
                Version(f"{epoch}!{major + 1}.0.0"),
            }
        )
    return sorted(probes)


def _claim_coverage_lost(previous, *, package, specifier, version):
    """Return what re-emitting ``specifier`` stops covering, or ``None``.

    ``previous`` is the ``compatible_*_packages`` list the bundle being
    certified already carried. Re-certifying without the claim flags rewrites
    that list from the flags, which turns a declared range back into an exact
    pin; this names the lowest probe version the old entries covered and the new
    one does not, so the caller can say what is being given up.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    if not isinstance(previous, list):
        return None
    declared = []
    for entry in previous:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or canonicalize_name(name) != canonicalize_name(
            package
        ):
            continue
        text = entry.get("specifier")
        if not isinstance(text, str):
            continue
        try:
            declared.append((text, SpecifierSet(text)))
        except InvalidSpecifier:
            continue
    if not declared:
        return None
    try:
        emitted = SpecifierSet(specifier)
        tested = Version(version)
    except (InvalidSpecifier, InvalidVersion):
        return None
    sets = [candidate for _, candidate in declared] + [emitted]
    for probe in _claim_probe_versions(sets, tested):
        if probe not in emitted and any(
            probe in candidate for _, candidate in declared
        ):
            return {
                "previous_specifiers": [text for text, _ in declared],
                "emitted_specifier": specifier,
                "first_version_no_longer_covered": str(probe),
            }
    return None


def _narrowing_notice(field, package, lost, *, offer_flags):
    """Say what re-emitting a compatibility entry takes away.

    ``model`` is the only field a publisher can declare a range for (see
    :data:`CLAIM_FIELD`), so it is the only one with a claim to narrow. Core's
    entry is always the exact tested pin; if it ever moved, the bundle would
    drop every consumer on the old Core, which is worth saying — but saying it
    narrows a Core *claim* would name a thing no producer can declare.

    Core's wording is defence in depth rather than a path anyone walks today:
    re-certification validates the input bundle first, and that gate requires
    its recorded receipt to equal the current runtime, so a moved Core version
    is refused before the emitted pin could differ from the carried one.
    """
    change = (
        f"narrows the {package} compatibility claim"
        if field == CLAIM_FIELD
        else f"moves the {package} compatibility pin"
    )
    notice = (
        f"certification {change} this bundle already carried: "
        f"{', '.join(lost['previous_specifiers'])} covered "
        f"{lost['first_version_no_longer_covered']} and the "
        f"{lost['emitted_specifier']} this run emits does not."
    )
    if offer_flags:
        notice += (
            " Pass --compatible-model-specifier with "
            "--compatibility-claim-declared-by to keep a declared range."
        )
    return notice


def _narrowed_claims(report) -> dict:
    """Return the narrowing ``report`` records, or ``{}``.

    One tolerance for every verdict that reports the record: certification and
    validation have the report in hand, publication reads it off disk, and a
    bundle they disagreed about would be reported by some of them and not
    others. Anything missing or unexpected reads as no record.
    """
    compatibility = report.get("compatibility") if isinstance(report, Mapping) else None
    narrowed = (
        compatibility.get("narrowed_claims")
        if isinstance(compatibility, Mapping)
        else None
    )
    return dict(narrowed) if isinstance(narrowed, Mapping) else {}


def recorded_narrowed_claims(release_dir) -> dict:
    """Return the compatibility narrowing ``release_dir`` records, or ``{}``.

    Certification warns when re-emitting a bundle takes coverage away, and
    records the same under ``compatibility.narrowed_claims``. That warning
    reaches one terminal; this is how a later gate reads the record back and
    reports it to whoever is standing in front of the next one.

    This surfaces something the validators have already accepted or refused on
    their own terms; it is a reporting path, not a gate, and must not turn a
    valid bundle into an error — an unreadable bundle reads as no record.
    """
    try:
        report = json.loads((Path(release_dir) / SOURCE_ENRICHMENT_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return _narrowed_claims(report)


def _check_compatibility(
    release_dir,
    manifest,
    compatibility,
    candidate,
    require_wheel_proof,
    wheels,
    failures,
    *,
    annual_revision: str | None = None,
    native_inputs: Mapping[str, str] | None = None,
):
    receipt_path = release_dir / COMPATIBILITY_FILE
    receipt = _json(receipt_path, failures)
    if (
        compatibility.get("filename") != COMPATIBILITY_FILE
        or not receipt_path.is_file()
        or (compatibility.get("sha256") != sha256_file(receipt_path))
    ):
        failures.append(
            "source enrichment compatibility must hash-bind the actual test receipt"
        )
    if candidate is None or not candidate.is_file():
        return
    try:
        actual = run_native_loader_compatibility(
            candidate,
            require_wheels=require_wheel_proof,
            compatibility_wheels=wheels,
            **_native_input_arguments(native_inputs),
        )
        if receipt != actual:
            failures.append(
                "compatibility receipt differs from actual native loader tests/runtime"
            )
        from microcosm.data.contract import _check_release_manifest

        _check_release_manifest(
            manifest, release_dir.name, failures, annual_revision=annual_revision
        )
        raw_claims = compatibility.get("publisher_claims")
        claims = _mapping(raw_claims)
        malformed_claims = raw_claims is not None and (
            not isinstance(raw_claims, Mapping) or set(claims) != {CLAIM_FIELD}
        )
        if malformed_claims:
            # Say only this. Falling through to the per-package branch below
            # would add "no publisher compatibility claim was declared", which
            # is the opposite of what happened.
            failures.append(
                f"publisher_claims must map {CLAIM_FIELD!r} to one declared "
                "compatibility claim; Core stays pinned to the tested version"
            )
        for package, field in (
            ("policyengine-us", "model"),
            ("policyengine-core", "core"),
        ):
            version = _mapping(actual.get("packages")).get(package, {}).get("version")
            if _mapping(
                _mapping(manifest.get("build")).get(f"built_with_{field}_package")
            ) != {"name": package, "version": version}:
                failures.append(
                    f"compatibility built-with {package} must match tested runtime"
                )
            if malformed_claims:
                continue
            declared = claims.get(field) if field == CLAIM_FIELD else None
            if declared is None:
                if manifest.get(f"compatible_{field}_packages") != [
                    {"name": package, "specifier": f"=={version}"}
                ]:
                    failures.append(
                        f"compatibility {package} must pin exactly the tested "
                        "version unless the certified report declares a "
                        "publisher compatibility claim"
                    )
                continue
            # A claim widens the binding, so the report must carry it and the
            # manifest must say exactly what the report says. The report is
            # hash-bound by the manifest's own artifact entry, so a manifest
            # widened after certification has no declaration to stand on.
            try:
                entry = compatibility_claim_entry(
                    _mapping(declared).get("specifier"),
                    package=package,
                    version=version,
                    declared_by=_mapping(declared).get("declared_by"),
                )
            except ValueError as exc:
                failures.append(f"declared {package} compatibility claim: {exc}")
                continue
            if declared != entry:
                failures.append(
                    f"declared {package} compatibility claim must record only "
                    "the validated name, specifier, basis and declarer"
                )
            if manifest.get(f"compatible_{field}_packages") != [entry]:
                failures.append(
                    f"compatibility {package} must match the declared publisher "
                    "compatibility claim"
                )
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        failures.append(f"native loader compatibility failed: {exc}")


def _wheel_files(path: Path) -> tuple[str, str, dict[str, bytes]]:
    from email.parser import BytesParser
    from zipfile import ZipFile

    with ZipFile(path) as wheel:
        metadata_files = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{path.name} must contain one wheel METADATA file")
        metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
        return (
            metadata["Name"],
            metadata["Version"],
            {
                name: wheel.read(name)
                for name in wheel.namelist()
                if name.endswith(".py")
            },
        )


def _runtime_package_identities(compatibility_wheels, *, require_wheels: bool) -> dict:
    """Hash actual installed source; optional wheel proof is offline, not PyPI proof."""
    from importlib import metadata

    names = COMPATIBILITY_PACKAGES
    wheels = {}
    for raw_path in compatibility_wheels:
        path = Path(raw_path)
        name, version, files = _wheel_files(path)
        name = name.lower().replace("_", "-")
        if name not in names or name in wheels:
            raise ValueError(
                "compatibility wheels must uniquely name country, Core, wrapper, and calculator"
            )
        wheels[name] = (path, version, files)
    if require_wheels and set(wheels) != set(names):
        raise ValueError(
            "compatibility requires exact installed policyengine-us, policyengine-core, policyengine, and spm-calculator wheels; candidate wheels may be tested before publication, whose external proof remains the publisher's gate"
        )
    result = {}
    for name in names:
        dist = metadata.distribution(name)
        direct_url = dist.read_text("direct_url.json")
        source_files = sorted(
            str(file) for file in (dist.files or ()) if str(file).endswith(".py")
        )
        if not source_files:
            raise ValueError(
                f"cannot attest actual installed Python sources for {name}"
            )
        digest = hashlib.sha256()
        for relative in source_files:
            path = Path(dist.locate_file(relative))
            digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
        identity = {
            "version": dist.version,
            "source_sha256": digest.hexdigest(),
            "direct_url": json.loads(direct_url) if direct_url else None,
            "wheel_sha256": None,
        }
        if name in wheels:
            path, version, files = wheels[name]
            if version != dist.version:
                raise ValueError(
                    f"{name} wheel version differs from the tested install"
                )
            for relative, expected in files.items():
                installed = Path(dist.locate_file(relative))
                if not installed.is_file() or installed.read_bytes() != expected:
                    raise ValueError(
                        f"{name} installed source differs from wheel: {relative}"
                    )
            if set(source_files) != set(files):
                raise ValueError(
                    f"{name} installed source inventory differs from wheel"
                )
            if direct_url and json.loads(direct_url).get("dir_info", {}).get(
                "editable"
            ):
                raise ValueError(
                    f"{name} editable installation is not a publishable wheel identity"
                )
            identity["wheel_sha256"] = sha256_file(path)
        result[name] = identity
    return result


def _native_input_arguments(native_inputs: Mapping[str, str] | None) -> dict:
    """Runner keywords for a lineage: none for the role lane, which predates them."""
    return {} if native_inputs is None else {"native_inputs": native_inputs}


def _native_input_label(name: str) -> str:
    """The loaded-source label of a native input; the role keeps its own."""
    return "native_role" if name == ROLE_VARIABLE else name


def _loaded_source_packages(inputs: Mapping[str, str]) -> dict[str, str]:
    """Label -> owning distribution for every source the probe binds.

    The role variable is registered from the calculator wheel; any other native
    input must be the country model's own variable.
    """
    packages = LOADED_SOURCE_PACKAGES.copy()
    for name in inputs:
        if name != ROLE_VARIABLE:
            packages[_native_input_label(name)] = "policyengine-us"
    return packages


def run_native_loader_compatibility(
    candidate_h5: Path | str,
    *,
    require_wheels: bool = False,
    compatibility_wheels: tuple[Path | str, ...] = (),
    native_inputs: Mapping[str, str] | None = None,
) -> dict:
    """Run fixed native-loader tests and one complete-household input-precedence probe.

    This proves native delivery, not canonical SPM numerical-model acceptance or
    publication of package versions. Those remain the root release gates. An
    installed wheel is checked against its source bytes, never a claimed bool.

    ``native_inputs`` maps each native Boolean input the H5 supplies to its
    entity. It defaults to the native SPM role alone, and the role lane's
    receipt is unchanged by this parameter. A reported-receipt child passes the
    role it inherits plus its three receipts (:data:`RECEIPT_NATIVE_INPUTS`);
    each must be registered by the tested country as a Boolean of that entity,
    reach both loaders byte-identical, and override its default in Core.
    """
    import inspect

    import h5py
    import numpy as np
    from policyengine.tax_benefit_models.us.datasets import PolicyEngineUSDataset
    from policyengine_us.data import USSingleYearDataset
    from policyengine_us.system import system

    candidate_h5 = Path(candidate_h5)
    inputs = {ROLE_VARIABLE: "person"} if native_inputs is None else dict(native_inputs)
    packages = _runtime_package_identities(
        compatibility_wheels, require_wheels=require_wheels
    )
    source_paths = {
        "country_loader": inspect.getsourcefile(USSingleYearDataset),
        "wrapper_loader": inspect.getsourcefile(PolicyEngineUSDataset),
    }
    for name, entity in inputs.items():
        variable = system.variables.get(name)
        if (
            variable is None
            or variable.entity.key != entity
            or variable.value_type is not bool
        ):
            raise ValueError(
                f"tested country model must register {name} as a native {entity} bool"
            )
        source_paths[_native_input_label(name)] = inspect.getsourcefile(type(variable))
    loaded_source_packages = _loaded_source_packages(inputs)
    if require_wheels:
        _check_loaded_source_ownership(source_paths, loaded_source_packages)
    country = USSingleYearDataset(file_path=str(candidate_h5))
    wrapper = PolicyEngineUSDataset(
        name="native_spm_role_compatibility",
        description="Read-only native SPM role compatibility probe",
        filepath=str(candidate_h5),
        year=2024,
    )
    checked = []
    with h5py.File(candidate_h5, "r") as h5:
        for entity in (
            "person",
            "household",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        ):
            table = h5[f"{entity}/table"]
            columns = [f"{entity}_id"]
            weight_column = f"{entity}_weight"
            if weight_column in table.dtype.names:
                columns.append(weight_column)
            columns += [name for name, owner in inputs.items() if owner == entity]
            if entity == "person":
                columns += [
                    name
                    for name in table.dtype.names
                    if name.startswith("person_") and name.endswith("_id")
                ]
            for column in dict.fromkeys(columns):
                expected = table[column]
                if column in inputs:
                    expected = expected.astype(bool)
                for loader, frame in (
                    ("country", getattr(country, entity)),
                    ("wrapper", getattr(wrapper.data, entity)),
                ):
                    observed = frame[column].to_numpy()
                    if (
                        observed.dtype != expected.dtype
                        or observed.shape != expected.shape
                        or observed.tobytes() != expected.tobytes()
                    ):
                        raise ValueError(
                            f"{loader} native loader changed {entity}.{column}"
                        )
                    checked.append(f"{loader}:{entity}.{column}")
        for name, entity in inputs.items():
            if not np.asarray(getattr(country, entity)[name]).dtype == np.dtype(bool):
                raise ValueError(
                    "country native role dtype must remain bool"
                    if name == ROLE_VARIABLE
                    else f"country native {entity} input {name} dtype must remain bool"
                )
    # The country engine may provide a household-role fallback for surveys
    # without this primitive. Opposite supplied values on one complete native
    # household must both reach Core unchanged, overriding any fixed fallback.
    _check_native_input_precedence(
        country, **({} if native_inputs is None else {"native_inputs": inputs})
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "native_input_loading_only",
        "external_package_publication": "not_attested",
        "canonical_spm_model_acceptance": "not_attested",
        "dataset_sha256": sha256_file(candidate_h5),
        "runner_sha256": sha256_file(__file__),
        "packages": packages,
        "loaded_source_packages": loaded_source_packages,
        "loaded_source_sha256": {
            key: sha256_file(path) for key, path in source_paths.items()
        },
        "checks": [
            *(
                f"country:registered_{entity}_bool_input"
                + ("" if name == ROLE_VARIABLE else f":{name}")
                for name, entity in inputs.items()
            ),
            "country:complete_household_native_input_overrides_default",
            *checked,
        ],
    }


def _check_loaded_source_ownership(
    source_paths: Mapping, loaded_source_packages: Mapping | None = None
) -> None:
    """The country registers the native role supplied by the calculator wheel.

    ``loaded_source_packages`` (label -> distribution) defaults to the role
    lane's :data:`LOADED_SOURCE_PACKAGES`; a receipt child adds its country
    receipt variables, each owned by the country wheel.
    """
    from importlib import import_module, metadata

    if loaded_source_packages is None:
        loaded_source_packages = LOADED_SOURCE_PACKAGES

    for package in COMPATIBILITY_PACKAGES:
        module_name = package.replace("-", "_")
        module = import_module(module_name)
        expected_root = Path(
            metadata.distribution(package).locate_file(module_name)
        ).resolve()
        if not Path(module.__file__).resolve().is_relative_to(expected_root):
            raise ValueError(
                f"tested {package} import does not come from the verified installed wheel"
            )
    for label, package in loaded_source_packages.items():
        expected_root = Path(
            metadata.distribution(package).locate_file(package.replace("-", "_"))
        ).resolve()
        source = source_paths.get(label)
        if source is None or not Path(source).resolve().is_relative_to(expected_root):
            raise ValueError(
                f"tested {label} source does not come from the verified {package} wheel"
            )


def _check_native_input_precedence(
    country, native_inputs: Mapping[str, str] | None = None
) -> None:
    """Opposite supplied values on one complete household must reach Core.

    ``native_inputs`` (name -> entity) defaults to the native SPM role alone.
    Every input is supplied on the household's rows of its own entity, all at
    once, and each must read back unchanged.
    """
    import numpy as np
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    inputs = {ROLE_VARIABLE: "person"} if native_inputs is None else native_inputs

    household_id = country.household["household_id"].iloc[0]
    person = country.person.loc[
        country.person["person_household_id"] == household_id
    ].copy()
    if person.empty:
        raise ValueError("native compatibility fixture has no complete household")
    tables = {"person": person, "time_period": 2024}
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        ids = person[f"person_{entity}_id"].unique()
        table = getattr(country, entity)
        tables[entity] = table.loc[table[f"{entity}_id"].isin(ids)].copy()
        if entity == "spm_unit":
            if (
                not country.person.loc[
                    country.person["person_spm_unit_id"].isin(ids),
                    "person_household_id",
                ]
                .eq(household_id)
                .all()
            ):
                raise ValueError("native compatibility household cuts an SPM unit")
    unsupplied = dict(tables)
    names_by_entity: dict[str, list[str]] = {}
    for name, entity in inputs.items():
        names_by_entity.setdefault(entity, []).append(name)
    for supplied_value in (False, True):
        for entity, names in names_by_entity.items():
            frame = unsupplied[entity]
            supplied = np.full(len(frame), supplied_value, dtype=bool)
            tables[entity] = frame.assign(**dict.fromkeys(names, supplied))
        simulation = Microsimulation(dataset=USSingleYearDataset(**tables))
        for name, entity in inputs.items():
            expected = np.full(len(tables[entity]), supplied_value, dtype=bool)
            observed = np.asarray(simulation.calculate(name, 2024))
            if observed.dtype != expected.dtype or not np.array_equal(
                observed, expected
            ):
                raise ValueError(
                    "country Core discarded the supplied native person role"
                    if name == ROLE_VARIABLE
                    else f"country Core discarded the supplied native {entity} "
                    f"input {name}"
                )


def certify_source_enrichment(
    release_dir: Path | str,
    output_dir: Path | str,
    *,
    parent_h5: Path | str,
    artifact_root: Path | str,
    compatibility_wheels: tuple[Path | str, ...],
    compatible_model_specifier: str | None = None,
    compatibility_claim_declared_by: str | None = None,
) -> Path:
    """Create a separate certified bundle only after measured loader checks pass.

    H5 and source evidence are never modified. The caller still owns canonical
    model acceptance, package publication proof, and publisher authorization.

    By default the bundle pins the exact model and Core versions the loader
    checks ran against. ``compatible_model_specifier`` lets the publisher
    declare a wider model range instead — ``"policyengine-us>=2.0.1,<2.1"`` —
    which the bundle records as a publisher claim attributed to
    ``compatibility_claim_declared_by``. The claim never replaces the measured
    runtime: ``build.built_with_model_package`` still names the exact version
    certification tested, and the range must contain it.

    Certifying a bundle that already declares a wider range, without passing
    the flags again, warns and records ``compatibility.narrowed_claims`` rather
    than quietly reverting it to the exact pin.
    """
    import shutil
    import tempfile
    from importlib import metadata

    release_dir, output_dir = Path(release_dir), Path(output_dir)
    if output_dir.exists() or output_dir.name != release_dir.name:
        raise ValueError(
            "output_dir must be new and keep the candidate release id as its basename"
        )
    if (compatible_model_specifier is None) != (
        compatibility_claim_declared_by is None
    ):
        raise ValueError(
            "a publisher compatibility claim needs both its specifier and the "
            "declarer who is accountable for it"
        )
    claim_specifier = None
    if compatible_model_specifier is not None:
        # Everything checkable without the tested version is checked here, so a
        # malformed claim costs nothing rather than a whole qualification run.
        claim_specifier = parse_compatibility_claim_requirement(
            compatible_model_specifier, package="policyengine-us"
        )
        check_compatibility_claim_declarer(compatibility_claim_declared_by)
    report = validate_source_enrichment_candidate(
        release_dir, parent_h5=parent_h5, artifact_root=artifact_root
    )
    candidate = Path(artifact_root) / report["dataset"]["filename"]
    receipt = run_native_loader_compatibility(
        candidate,
        require_wheels=True,
        compatibility_wheels=compatibility_wheels,
        **_native_input_arguments(
            RECEIPT_NATIVE_INPUTS
            if report.get("operation") == RECEIPT_OPERATION
            else None
        ),
    )
    claims = (
        {
            CLAIM_FIELD: compatibility_claim_entry(
                claim_specifier,
                package="policyengine-us",
                version=receipt["packages"]["policyengine-us"]["version"],
                declared_by=compatibility_claim_declared_by,
            )
        }
        if claim_specifier is not None
        else {}
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".source-enrichment-", dir=output_dir.parent
    ) as staging:
        staged = Path(staging) / release_dir.name
        shutil.copytree(release_dir, staged)

        def write(name, value):
            (staged / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )

        manifest = json.loads((staged / "release_manifest.json").read_text())
        emitted, narrowed = {}, {}
        for package, field in (
            ("policyengine-us", "model"),
            ("policyengine-core", "core"),
        ):
            version = receipt["packages"][package]["version"]
            entry = claims.get(field, {"name": package, "specifier": f"=={version}"})
            emitted[field] = (package, version, entry)
            # The compatible list below is rewritten from the flags, so
            # re-certifying a declared bundle without them silently reverts it
            # to an exact pin. Say so, in the terminal and in the bundle.
            lost = _claim_coverage_lost(
                manifest.get(f"compatible_{field}_packages"),
                package=package,
                specifier=entry["specifier"],
                version=version,
            )
            if lost is None:
                continue
            narrowed[field] = lost
            warnings.warn(
                _narrowing_notice(
                    field,
                    package,
                    lost,
                    # Only the run that forgot them needs telling. Re-certifying
                    # with a tighter range is a deliberate narrowing, still
                    # worth naming, but its operator already passed the flags.
                    offer_flags=field == CLAIM_FIELD and claim_specifier is None,
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        write(COMPATIBILITY_FILE, receipt)
        report["compatibility"] = {
            "status": "passed",
            "filename": COMPATIBILITY_FILE,
            "sha256": sha256_file(staged / COMPATIBILITY_FILE),
        }
        if claims:
            # Absent by default, so an undeclared bundle keeps the bytes it
            # has always had.
            report["compatibility"]["publisher_claims"] = claims
        if narrowed:
            report["compatibility"]["narrowed_claims"] = narrowed
        write(SOURCE_ENRICHMENT_FILE, report)
        for field, (package, version, entry) in emitted.items():
            manifest["build"][f"built_with_{field}_package"] = {
                "name": package,
                "version": version,
            }
            manifest[f"compatible_{field}_packages"] = [entry]
        manifest["data_package"] = {
            "name": "microcosm-data",
            "version": metadata.version("microcosm-data"),
        }
        for entry in manifest["artifacts"].values():
            if entry["path"] == SOURCE_ENRICHMENT_FILE:
                entry["sha256"] = sha256_file(staged / SOURCE_ENRICHMENT_FILE)
        manifest["artifacts"]["source_enrichment_compatibility"] = {
            "kind": "diagnostics",
            "path": COMPATIBILITY_FILE,
            "repo_id": "policyengine/populace-us",
            "revision": release_dir.name,
            "sha256": sha256_file(staged / COMPATIBILITY_FILE),
        }
        write("release_manifest.json", manifest)
        validate_source_enrichment_candidate(
            staged,
            parent_h5=parent_h5,
            artifact_root=artifact_root,
            require_compatibility=True,
            compatibility_wheels=compatibility_wheels,
        )
        staged.rename(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--parent-h5", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-compatibility", action="store_true")
    parser.add_argument("--compatibility-wheel", action="append", type=Path, default=[])
    parser.add_argument(
        "--compatible-model-specifier",
        help=(
            "declare a publisher model compatibility range instead of the "
            "default exact pin, as a PEP 508 requirement naming the built-with "
            "package, e.g. 'policyengine-us>=2.0.1,<2.1'. The range must "
            "contain the version certification tested and be bounded above "
            "and below. "
            "Requires --compatibility-claim-declared-by."
        ),
    )
    parser.add_argument(
        "--compatibility-claim-declared-by",
        help=(
            "who declares the compatibility range, recorded in the bundle "
            "(e.g. 'PolicyEngine data release owner, microcosm#912')"
        ),
    )
    args = parser.parse_args(argv)
    if args.certify and args.output_dir is None:
        parser.error(
            "--certify requires a new --output-dir ending in the same release id"
        )
    if (args.compatible_model_specifier is None) != (
        args.compatibility_claim_declared_by is None
    ):
        parser.error(
            "--compatible-model-specifier and --compatibility-claim-declared-by "
            "are declared together; a claim records who is accountable for it"
        )
    if args.compatible_model_specifier is not None and not args.certify:
        parser.error(
            "--compatible-model-specifier applies to --certify; validation "
            "reads the claim the certified bundle already records"
        )
    try:
        if args.certify:
            result = certify_source_enrichment(
                args.release_dir,
                args.output_dir,
                parent_h5=args.parent_h5,
                artifact_root=args.artifact_root,
                compatibility_wheels=tuple(args.compatibility_wheel),
                compatible_model_specifier=args.compatible_model_specifier,
                compatibility_claim_declared_by=(args.compatibility_claim_declared_by),
            )
            certified = {"certified_bundle": str(result), "published": False}
            # The run that narrowed a claim already warned about it. Say it in
            # the verdict too: the warning is the signal this reporting path
            # exists because it cannot be relied on.
            narrowed = recorded_narrowed_claims(result)
            if narrowed:
                certified["narrowed_claims"] = narrowed
            print(json.dumps(certified))
        else:
            report = validate_source_enrichment_candidate(
                args.release_dir,
                parent_h5=args.parent_h5,
                artifact_root=args.artifact_root,
                require_compatibility=args.require_compatibility,
                compatibility_wheels=tuple(args.compatibility_wheel),
            )
            validated = {
                "valid": True,
                "compatibility": report["compatibility"]["status"],
            }
            # Certification's narrowing warning reaches one terminal. The
            # record it leaves behind reaches every later gate, so say so here
            # rather than reporting only that the bundle is valid.
            narrowed = _narrowed_claims(report)
            if narrowed:
                validated["narrowed_claims"] = narrowed
            print(json.dumps(validated))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
