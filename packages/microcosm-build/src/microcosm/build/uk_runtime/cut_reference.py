"""The current build a UK cut is checked against (microcosm#1095).

A cut compares its candidate with the current build of its own release line,
as the Hub's ``releases.json`` names it when the cut fetches its inputs: the
national line with the national, the dense line with the dense. Comparing
the dense file with the national would mix the lines' differences (clones,
local targets, area columns) into checks meant to catch change between cuts.

The reference is resolved once per cut. The fetch writes its record, and
every later step reads that record and re-checks the file's SHA-256 against
it; none re-resolves, so a promotion while a cut runs cannot split its
reference. Each check that uses the reference names it in its own output.

The fetch runs without the gate signing key, so it reads a build's
certification by its fields, not its signature: a national build's
``release_certification.json`` must be shippable, name this file's SHA-256
and carry a registered key's fingerprint, and a dense build's
``gate_summary.json`` must report every gate passed. The publication that
made the build current verified the signatures. ``releases.json`` itself
never names a withdrawn build as current.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.release_identity import (
    UK_DENSE_RELEASE_ID,
    UK_NATIONAL_RELEASE_ID,
)

__all__ = [
    "UK_CUT_REFERENCE_KIND",
    "UK_CUT_REFERENCE_RELEASES",
    "UK_CUT_REFERENCE_SCHEMA_VERSION",
    "UKCutReference",
    "UK_CUT_REFERENCE_INPUT_MASS_NAME",
    "UK_CUT_REFERENCE_RECORD_NAME",
    "UK_CUT_REFERENCE_SURFACE_NAME",
    "check_uk_cut_reference",
    "fetch_uk_cut_reference",
    "read_uk_cut_reference",
    "resolve_uk_cut_reference",
    "uk_cut_reference_evidence_done",
    "uk_input_mass_reference_payload",
    "verify_uk_cut_reference_file",
    "write_uk_cut_reference",
    "write_uk_cut_reference_evidence",
]

UK_CUT_REFERENCE_KIND = "uk_cut_reference"
UK_CUT_REFERENCE_SCHEMA_VERSION = 1
#: The record's file name beside the reference's dataset in a cut.
UK_CUT_REFERENCE_RECORD_NAME = "reference.json"
#: The reference's input-mass totals and carriers, written beside it: the
#: input-mass gate's reference (licensed-data aggregates, never committed).
UK_CUT_REFERENCE_INPUT_MASS_NAME = "input_mass_reference.json"
#: The reference's column surface (names and flags): the export-surface and
#: input-coverage gates' reference (:mod:`.reference_surface`).
UK_CUT_REFERENCE_SURFACE_NAME = "reference_surface.json"

#: Each line's registry variant and the build evidence its fetch reads.
UK_CUT_REFERENCE_RELEASES: Mapping[str, tuple[str, str]] = {
    UK_NATIONAL_RELEASE_ID: ("compact", "release_certification"),
    UK_DENSE_RELEASE_ID: ("dense", "gate_summary"),
}
_REGISTRY_YEAR = 2025

_HubDownload = Callable[..., str]

_RECORD_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "release",
        "build",
        "repo_id",
        "path_in_repo",
        "filename",
        "sha256",
        "size_bytes",
        "built_with_model_package",
        "built_with_core_package",
        "evidence",
        "resolved_at",
    }
)


@dataclass(frozen=True)
class UKCutReference:
    """One build of a UK release line, resolved for a cut to compare with."""

    release: str
    build: str
    repo_id: str
    path_in_repo: str
    filename: str
    sha256: str
    size_bytes: int
    built_with_model_package: Mapping[str, str]
    built_with_core_package: Mapping[str, str]
    evidence: Mapping[str, str]
    resolved_at: str

    @property
    def label(self) -> str:
        """``<release>@<build>``, the name every check that uses it records."""

        return f"{self.release}@{self.build}"

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": UK_CUT_REFERENCE_KIND,
            "schema_version": UK_CUT_REFERENCE_SCHEMA_VERSION,
            "release": self.release,
            "build": self.build,
            "repo_id": self.repo_id,
            "path_in_repo": self.path_in_repo,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "built_with_model_package": dict(self.built_with_model_package),
            "built_with_core_package": dict(self.built_with_core_package),
            "evidence": dict(self.evidence),
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_record(cls, payload: object) -> UKCutReference:
        if not isinstance(payload, Mapping) or set(payload) != _RECORD_KEYS:
            raise ValueError(
                "A UK cut reference record must have exactly the keys "
                f"{sorted(_RECORD_KEYS)}."
            )
        if payload["kind"] != UK_CUT_REFERENCE_KIND:
            raise ValueError(
                f"A UK cut reference record has kind {UK_CUT_REFERENCE_KIND!r}, "
                f"not {payload['kind']!r}."
            )
        if payload["schema_version"] != UK_CUT_REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                "A UK cut reference record has schema_version "
                f"{UK_CUT_REFERENCE_SCHEMA_VERSION}, not "
                f"{payload['schema_version']!r}."
            )
        if payload["release"] not in UK_CUT_REFERENCE_RELEASES:
            raise ValueError(
                f"A UK cut reference names release {payload['release']!r}; a "
                f"cut compares with one of {sorted(UK_CUT_REFERENCE_RELEASES)}."
            )
        for key in ("build", "repo_id", "path_in_repo", "filename", "resolved_at"):
            if not isinstance(payload[key], str) or not payload[key]:
                raise ValueError(f"A UK cut reference record has no {key}.")
        sha256 = payload["sha256"]
        if not _is_sha256(sha256):
            raise ValueError("A UK cut reference record has no valid sha256.")
        size = payload["size_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("A UK cut reference record has no positive size_bytes.")
        for key in ("built_with_model_package", "built_with_core_package"):
            package = payload[key]
            if (
                not isinstance(package, Mapping)
                or set(package) != {"name", "version"}
                or not all(
                    isinstance(value, str) and value for value in package.values()
                )
            ):
                raise ValueError(
                    f"A UK cut reference record's {key} must name a package and "
                    "its version."
                )
        evidence = payload["evidence"]
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != {"key", "filename", "sha256"}
            or not _is_sha256(evidence["sha256"])
        ):
            raise ValueError(
                "A UK cut reference record's evidence must name the build file "
                "its fetch read, with its sha256."
            )
        return cls(
            release=payload["release"],
            build=payload["build"],
            repo_id=payload["repo_id"],
            path_in_repo=payload["path_in_repo"],
            filename=payload["filename"],
            sha256=sha256,
            size_bytes=size,
            built_with_model_package=dict(payload["built_with_model_package"]),
            built_with_core_package=dict(payload["built_with_core_package"]),
            evidence=dict(evidence),
            resolved_at=payload["resolved_at"],
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line(release: str) -> tuple[str, str]:
    try:
        return UK_CUT_REFERENCE_RELEASES[release]
    except KeyError:
        raise ValueError(
            f"A UK cut compares with the current build of {release!r} only if "
            f"it is one of {sorted(UK_CUT_REFERENCE_RELEASES)}."
        ) from None


def _check_national_certification(
    payload: object, *, release: str, build: str, dataset_sha256: str
) -> None:
    from microcosm.data.gate_signing_keys import registered_gate_signing_key

    where = f"release_certification.json of {release}@{build}"
    if not isinstance(payload, Mapping):
        raise ValueError(f"The {where} is not a JSON object.")
    if payload.get("kind") != "uk_release_certification":
        raise ValueError(f"The {where} is not a UK release certification.")
    if payload.get("release_id") != release:
        raise ValueError(f"The {where} certifies another release.")
    if payload.get("shippable") is not True:
        raise ValueError(
            f"The current build {release}@{build} is not certified shippable; a "
            "cut compares only with a certified build."
        )
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("sha256") != dataset_sha256:
        raise ValueError(
            f"The {where} certifies another file than the build's dataset."
        )
    attestation = payload.get("attestation")
    fingerprint = (
        attestation.get("signing_key_sha256")
        if isinstance(attestation, Mapping)
        else None
    )
    if not isinstance(fingerprint, str) or (
        registered_gate_signing_key("uk", fingerprint) is None
    ):
        raise ValueError(f"The {where} was not signed by a registered UK gate key.")


def _check_dense_gate_summary(payload: object, *, release: str, build: str) -> None:
    where = f"gate_summary.json of {release}@{build}"
    if not isinstance(payload, Mapping):
        raise ValueError(f"The {where} is not a JSON object.")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError(f"The {where} reports no gates.")
    failing = sorted(
        name
        for name, gate in gates.items()
        if not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or gate.get("status") != "passed"
    )
    if failing:
        raise ValueError(
            f"The current build {release}@{build} did not pass "
            f"{', '.join(failing)}; a cut compares only with a certified build."
        )


def _checked_evidence(
    certified: Any,
    *,
    release: str,
    evidence_key: str,
    hub_download: _HubDownload | None,
) -> dict[str, str]:
    from microcosm.data.loader import download_build_file

    path = download_build_file(certified, evidence_key, hub_download=hub_download)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read {path.name} of {release}@{certified.build}: {exc}."
        ) from exc
    if evidence_key == "release_certification":
        _check_national_certification(
            payload,
            release=release,
            build=certified.build,
            dataset_sha256=certified.artifact_sha256,
        )
    else:
        _check_dense_gate_summary(payload, release=release, build=certified.build)
    return {"key": evidence_key, "filename": path.name, "sha256": _sha256(path)}


def check_uk_cut_reference(
    release: str, *, hub_download: _HubDownload | None = None
) -> dict[str, str]:
    """Which build a cut of ``release`` would compare with now, read-only.

    Reads the release record, the build's manifest and its certification
    evidence, with every check the fetch makes, but downloads no dataset.
    """

    from microcosm.data.loader import resolve_certified

    variant, evidence_key = _line(release)
    certified = resolve_certified(
        "uk", _REGISTRY_YEAR, variant=variant, hub_download=hub_download
    )
    _require_build_folder(certified, release=release)
    evidence = _checked_evidence(
        certified, release=release, evidence_key=evidence_key, hub_download=hub_download
    )
    return {
        "release": release,
        "build": certified.build,
        "path_in_repo": certified.artifact_path,
        "sha256": certified.artifact_sha256,
        "built_with_model_package": (
            f"{certified.model.name} {certified.model.built_version}"
        ),
        "evidence": evidence["filename"],
    }


def _require_build_folder(certified: Any, *, release: str) -> None:
    if certified.release_id != release or certified.build is None:
        raise ValueError(
            f"{release} is not recorded in the Hub's releases.json, so it has no "
            "current build for a cut to compare with."
        )


def resolve_uk_cut_reference(
    release: str,
    *,
    hub_download: _HubDownload | None = None,
    now: Callable[[], datetime] | None = None,
) -> tuple[Path, UKCutReference]:
    """Download and check the current build of ``release`` for a cut.

    Returns the verified dataset's local path and the record a cut keeps.
    Refuses a release the Hub does not record in ``releases.json`` and a
    current build whose certification evidence does not pass.
    """

    from microcosm.data.loader import download_certified

    variant, evidence_key = _line(release)
    path, certified = download_certified(
        "uk", _REGISTRY_YEAR, variant=variant, hub_download=hub_download
    )
    _require_build_folder(certified, release=release)
    evidence = _checked_evidence(
        certified, release=release, evidence_key=evidence_key, hub_download=hub_download
    )
    resolved_at = (now or (lambda: datetime.now(UTC)))().strftime("%Y-%m-%dT%H:%M:%SZ")
    reference = UKCutReference(
        release=release,
        build=certified.build,
        repo_id=certified.artifact_repo_id,
        path_in_repo=certified.artifact_path,
        filename=Path(certified.artifact_path).name,
        sha256=certified.artifact_sha256,
        size_bytes=path.stat().st_size,
        built_with_model_package={
            "name": certified.model.name,
            "version": certified.model.built_version,
        },
        built_with_core_package={
            "name": certified.core.name,
            "version": certified.core.built_version,
        },
        evidence=evidence,
        resolved_at=resolved_at,
    )
    return path, reference


def write_uk_cut_reference(path: str | Path, reference: UKCutReference) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reference.to_record(), indent=2, sort_keys=True))
    return destination


def read_uk_cut_reference(path: str | Path) -> UKCutReference:
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read the cut's reference record {source}: {exc}."
        ) from exc
    return UKCutReference.from_record(payload)


def verify_uk_cut_reference_file(reference: UKCutReference, path: str | Path) -> Path:
    """Refuse ``path`` unless it is the file ``reference`` records."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(
            f"The cut's reference {reference.label} is not at {source}; fetch it again."
        )
    size = source.stat().st_size
    if size != reference.size_bytes:
        raise ValueError(
            f"{source} is {size} bytes, but the cut's reference {reference.label} "
            f"is {reference.size_bytes}."
        )
    observed = _sha256(source)
    if observed != reference.sha256:
        raise ValueError(
            f"{source} has sha256 {observed}, but the cut's reference "
            f"{reference.label} has {reference.sha256}."
        )
    return source


def fetch_uk_cut_reference(
    release: str,
    output_dir: str | Path,
    *,
    hub_download: _HubDownload | None = None,
    now: Callable[[], datetime] | None = None,
) -> UKCutReference:
    """Resolve the current build of ``release`` into ``output_dir``, once.

    Copies the verified dataset there and writes its record beside it
    (:data:`UK_CUT_REFERENCE_RECORD_NAME`). Refuses a directory that already
    holds a record, so a run keeps the reference it started with.
    """

    directory = Path(output_dir)
    record_path = directory / UK_CUT_REFERENCE_RECORD_NAME
    if record_path.exists():
        raise ValueError(
            f"{directory} already holds a reference ({record_path}); a run keeps "
            "the reference it started with. Use a new directory to resolve again."
        )
    source, reference = resolve_uk_cut_reference(
        release, hub_download=hub_download, now=now
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / reference.filename
    partial = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, partial)
    try:
        verify_uk_cut_reference_file(reference, partial)
    except ValueError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)
    write_uk_cut_reference(record_path, reference)
    return reference


def uk_input_mass_reference_payload(
    h5_path: str | Path,
    *,
    label: str,
    filename: str,
    revision: str,
    sha256: str,
    vintage: str = "2024_25",
) -> dict[str, object]:
    """The schema-2 input-mass reference a cut writes from its reference build.

    Totals and carriers come from the build's export frame at its own solved
    weights (:func:`~.weight_year.uk_recorded_solve_year_frame`), so a build
    an earlier policyengine-uk wrote compares on the weights it was solved
    with, as the candidate does.
    """

    # Here, not in weighted_integrity: the target kernels reach that module,
    # and the dataset readers below are no part of any target's value.
    from microcosm.build.uk_runtime.national_frame import uk_published_release_frame
    from microcosm.build.uk_runtime.rowwise_dataset import load_uk_rowwise_dataset
    from microcosm.build.uk_runtime.weight_year import uk_recorded_solve_year_frame
    from microcosm.build.uk_runtime.weighted_integrity import (
        UK_INPUT_MASS_REFERENCE_KIND,
        UK_INPUT_MASS_REFERENCE_SCHEMA_VERSION,
        uk_input_mass_carriers,
        uk_input_mass_totals,
    )

    # The row-wise reader opens both lines' files: the dense line's carries one
    # ``clone_index`` name on every table, which the compact national loader
    # refuses, and a national file carries none.
    frame, _provenance = load_uk_rowwise_dataset(h5_path)
    export = uk_published_release_frame(uk_recorded_solve_year_frame(frame))
    return {
        "schema_version": UK_INPUT_MASS_REFERENCE_SCHEMA_VERSION,
        "kind": UK_INPUT_MASS_REFERENCE_KIND,
        "label": label,
        "identity": {
            "filename": filename,
            "revision": revision,
            "sha256": sha256,
            "vintage": vintage,
        },
        "totals": uk_input_mass_totals(export),
        "carriers": uk_input_mass_carriers(export),
    }


def write_uk_cut_reference_evidence(
    reference: UKCutReference, h5_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    """Write the evidence the gates read from the reference, beside it.

    The input-mass totals and carriers of the build's export frame at its own
    solved weights (:func:`uk_input_mass_reference_payload`),
    which hold licensed-data aggregates and stay with the cut as the
    reference's dataset does, and the build's column surface, names and
    flags only (:func:`~.reference_surface.uk_reference_surface_payload`).
    """

    from microcosm.build.uk_runtime.reference_surface import (
        uk_reference_surface_payload,
    )

    source = verify_uk_cut_reference_file(reference, h5_path)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = uk_input_mass_reference_payload(
        source,
        label=reference.label,
        filename=reference.filename,
        revision=reference.build,
        sha256=reference.sha256,
    )
    surface = uk_reference_surface_payload(
        source, label=reference.label, sha256=reference.sha256
    )
    written: dict[str, Path] = {}
    for key, name, content in (
        ("input_mass_reference", UK_CUT_REFERENCE_INPUT_MASS_NAME, payload),
        ("reference_surface", UK_CUT_REFERENCE_SURFACE_NAME, surface),
    ):
        destination = directory / name
        partial = destination.with_name(destination.name + ".partial")
        partial.write_text(json.dumps(content, sort_keys=True, allow_nan=False))
        partial.replace(destination)
        written[key] = destination
    return written


def uk_cut_reference_evidence_done(
    reference: UKCutReference, output_dir: str | Path
) -> tuple[bool, str]:
    """Whether the reference's evidence is beside it and names this build."""

    for name in (UK_CUT_REFERENCE_INPUT_MASS_NAME, UK_CUT_REFERENCE_SURFACE_NAME):
        try:
            payload = json.loads((Path(output_dir) / name).read_text())
        except (OSError, ValueError):
            return False, f"no {name} beside the reference"
        if not isinstance(payload, Mapping):
            return False, f"{name} is not a JSON object"
        identity = payload.get("identity")
        sha256 = (
            identity.get("sha256")
            if isinstance(identity, Mapping)
            else payload.get("sha256")
        )
        if payload.get("label") != reference.label or sha256 != reference.sha256:
            return False, f"{name} names another build"
    return True, f"evidence for {reference.label}"
