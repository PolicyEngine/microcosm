"""TRACE Transparent Research Object (TRO) emission for populace builds.

Emits JSON-LD that conforms to the TRACE TROv vocabulary
(https://w3id.org/trace/trov/0.1#) for a populace dataset build. A build TRO
pins, by sha256, every artifact the build *produced* (the calibrated dataset
H5 and the calibration weights) **and every restricted input it consumed**
(the IRS PUF, the Fed SCF, the Census SIPP) — at the inputs' canonical access
locations, even though a third party cannot download them. That asymmetry is
the whole point, in Lars Vilhuber's framing: a re-runner cannot fetch the IRS
PUF, but anyone can check that *this* build consumed the same checksummed PUF,
with this pinned code, and passed these gates. A provenance certificate is
worth more than a re-run nobody can perform.

This module is deliberately **self-contained and stdlib-only** (``json``,
``hashlib``, ``os``, plus :mod:`importlib.metadata` for the package version).
It does NOT import :mod:`policyengine` — the two ecosystems agree on the hash
algorithms by *replicating* their definitions, not by depending on each other,
so a verifier on either side can reproduce the bytes. The canonical-JSON
encoding and the composition fingerprint here are byte-identical to
``policyengine.provenance.trace.canonical_json_bytes`` and
``compute_trace_composition_fingerprint`` (documented as such on each function
so the agreement is auditable). populace-specific metadata lives under the
``pop:`` namespace so generated TROs can still be validated against TROv SHACL
shapes when tooling is available.

CLI::

    python -m populace.build.trace <release_manifest.json> [--out PATH] \\
        [--self-url URL]

reads a release manifest (the shape of
``packages/populace-data/build/us/release_manifest.json``) and writes the TRO
to ``--out`` or stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "TRACE_TROV_NAMESPACE",
    "POPULACE_TRACE_NAMESPACE",
    "canonical_json_bytes",
    "compute_composition_fingerprint",
    "sha256_file",
    "build_build_tro",
    "tro_from_release_manifest",
    "serialize_tro",
    "main",
]

#: The TRACE TROv vocabulary namespace. Identical to the value used by
#: ``policyengine.provenance.trace`` — both ecosystems target the same
#: vocabulary so a single SHACL validator covers both.
TRACE_TROV_NAMESPACE = "https://w3id.org/trace/trov/0.1#"

#: The populace-specific extension namespace. Mirrors the role of ``pe:`` in
#: policyengine.py (which uses ``https://policyengine.org/trace/0.1#``);
#: populace gets its own so the two never collide.
POPULACE_TRACE_NAMESPACE = "https://populace.dev/trace/0.1#"

TRACE_CONTEXT: list[dict[str, str]] = [
    {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "trov": TRACE_TROV_NAMESPACE,
        "schema": "https://schema.org/",
        "pop": POPULACE_TRACE_NAMESPACE,
    }
]

POLICYENGINE_ORGANIZATION: dict[str, str] = {
    "@type": "schema:Organization",
    "schema:name": "PolicyEngine",
    "schema:url": "https://policyengine.org",
}

_COMPOSITION_ID = "composition/1"
_ARRANGEMENT_ID = "arrangement/1"

_MIME_TYPES = {
    "h5": "application/x-hdf5",
    "npz": "application/x-numpy-npz",
    "json": "application/json",
    "jsonld": "application/ld+json",
    "parquet": "application/vnd.apache.parquet",
    "csv": "text/csv",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON serialization used for every content hash in the TRO.

    Defined as ``json.dumps(value, indent=2, sort_keys=True) + "\\n"`` encoded
    as UTF-8. This is **byte-identical** to
    ``policyengine.provenance.trace.canonical_json_bytes`` — both definitions
    must agree so a third-party verifier can recompute the hashes that the
    composition fingerprint binds together, and so a payload hashed in one
    ecosystem reproduces in the other. Do not change one without the other.
    """
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compute_composition_fingerprint(hashes: Iterable[str]) -> str:
    """Fingerprint a composition by the sorted set of its artifact hashes.

    Defined as ``sha256`` of the artifact hashes **sorted** then joined with a
    single ``"\\n"`` (so concatenation is unambiguous regardless of hash
    length). This is **byte-identical** to
    ``policyengine.provenance.trace.compute_trace_composition_fingerprint``;
    the two ecosystems share the algorithm by replication, not dependency, so
    a fingerprint computed here reproduces there. Do not change one without
    the other.
    """
    sorted_hashes = sorted(hashes)
    digest = hashlib.sha256()
    digest.update("\n".join(sorted_hashes).encode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file's bytes (default 1 MiB chunks).

    Used to hash the build's output artifacts (multi-hundred-MB H5 files)
    without reading them whole into memory.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _populace_build_version() -> str | None:
    """The installed populace-build package version, for ``trov:createdWith``.

    Falls back to the in-tree ``__version__`` (and then ``None``) when the
    package metadata is unavailable — e.g. running from a source checkout that
    was never installed.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib >=3.8
        return None
    try:
        return version("populace-build")
    except PackageNotFoundError:
        try:
            from populace.build import __version__

            return __version__
        except Exception:  # pragma: no cover - defensive
            return None


def _mime_type_for(name: str, explicit: str | None) -> str | None:
    """An artifact's MIME type: an explicit value wins, else inferred by suffix."""
    if explicit is not None:
        return explicit
    lowered = name.lower()
    suffix = lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    return _MIME_TYPES.get(suffix)


def _emission_context() -> dict[str, str]:
    """Attestation metadata about where and how the TRO was emitted.

    Mirrors ``policyengine.provenance.trace._emission_context`` field-for-field
    under the ``pop:`` namespace. Always sets ``pop:emittedIn`` so a verifier
    distinguishes a CI build from a laptop build without inferring from the
    absence of optional fields.
    """
    context: dict[str, str] = {}
    if os.environ.get("GITHUB_ACTIONS") == "true":
        context["pop:emittedIn"] = "github-actions"
        server = os.environ.get("GITHUB_SERVER_URL")
        repo = os.environ.get("GITHUB_REPOSITORY")
        run_id = os.environ.get("GITHUB_RUN_ID")
        if server and repo and run_id:
            context["pop:ciRunUrl"] = f"{server}/{repo}/actions/runs/{run_id}"
        sha = os.environ.get("GITHUB_SHA")
        if sha:
            context["pop:ciGitSha"] = sha
        ref = os.environ.get("GITHUB_REF")
        if ref:
            context["pop:ciGitRef"] = ref
    else:
        context["pop:emittedIn"] = "local"
    return context


def _artifact_spec_from_mapping(
    artifact: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Normalize an input/output artifact mapping into an internal spec.

    Raises:
        ValueError: If the artifact has no ``sha256`` — provenance without a
            hash is not provenance; callers must hash the bytes or omit the
            artifact entirely.
    """
    sha256 = artifact.get("sha256")
    if not sha256:
        name = artifact.get("name") or artifact.get("id") or "<unnamed>"
        raise ValueError(
            f"{role} artifact {name!r} has no sha256; provenance without a "
            "hash is not provenance — hash the bytes or omit the artifact."
        )
    artifact_id = artifact.get("id") or artifact.get("name") or sha256
    return {
        "id": str(artifact_id),
        "hash": str(sha256),
        "location": artifact.get("location"),
        "name": artifact.get("name"),
        "mime_type": _mime_type_for(
            str(artifact.get("name") or artifact.get("location") or ""),
            artifact.get("mime_type"),
        ),
        "access_restricted": bool(artifact.get("access_restricted", False)),
    }


def _payload_spec(payload: Any, *, spec_id: str, name: str) -> dict[str, Any]:
    """A hashed-payload artifact spec for an inline JSON payload.

    The payload (a config manifest, gate report, or the stage records) is
    hashed by its :func:`canonical_json_bytes` so the TRO binds the exact
    bytes a verifier would recompute. The location is a content-addressed
    ``pop:`` URN rather than a fetchable URL — the payload is embedded in the
    build record, not published as a standalone file.
    """
    payload_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {
        "id": spec_id,
        "hash": payload_hash,
        "location": f"urn:pop:payload:{name}:{payload_hash}",
        "name": name,
        "mime_type": "application/json",
        "access_restricted": False,
    }


def _make_artifact(spec: Mapping[str, Any]) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "@id": f"{_COMPOSITION_ID}/artifact/{spec['id']}",
        "@type": "trov:ResearchArtifact",
        "trov:sha256": spec["hash"],
    }
    if spec.get("mime_type") is not None:
        artifact["trov:mimeType"] = spec["mime_type"]
    if spec.get("name") is not None:
        artifact["schema:name"] = spec["name"]
    return artifact


def _make_location(spec: Mapping[str, Any]) -> dict[str, Any]:
    location: dict[str, Any] = {
        "@id": f"{_ARRANGEMENT_ID}/location/{spec['id']}",
        "@type": "trov:ArtifactLocation",
        "trov:hasArtifact": {"@id": f"{_COMPOSITION_ID}/artifact/{spec['id']}"},
        "trov:hasLocation": spec["location"],
    }
    if spec.get("access_restricted"):
        location["pop:accessRestricted"] = True
    return location


def _assemble_composition_and_arrangement(
    specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = [_make_artifact(spec) for spec in specs]
    locations = [_make_location(spec) for spec in specs]
    hashes = [spec["hash"] for spec in specs]
    composition = {
        "@id": _COMPOSITION_ID,
        "@type": "trov:ArtifactComposition",
        "trov:hasFingerprint": {
            "@id": f"{_COMPOSITION_ID}/fingerprint",
            "@type": "trov:CompositionFingerprint",
            "trov:sha256": compute_composition_fingerprint(hashes),
        },
        "trov:hasArtifact": artifacts,
    }
    arrangement = {
        "@id": _ARRANGEMENT_ID,
        "@type": "trov:ArtifactArrangement",
        "trov:hasArtifactLocation": locations,
    }
    return composition, arrangement


def _research_system() -> dict[str, Any]:
    return {
        "@id": "trs",
        "@type": "trov:TransparentResearchSystem",
        "schema:name": "populace build pipeline",
        "rdfs:comment": (
            "The populace dataset build: a typed stage plan over primary "
            "survey donors, calibrated to an administrative target surface, "
            "gated before publication."
        ),
    }


def build_build_tro(
    *,
    build_id: str,
    builder: str,
    build_sha: str,
    build_date: str,
    config_manifest: Mapping[str, Any] | None = None,
    stage_records: Sequence[Mapping[str, Any]] | None = None,
    gate_manifest: Mapping[str, Any] | None = None,
    input_artifacts: Sequence[Mapping[str, Any]] = (),
    output_artifacts: Sequence[Mapping[str, Any]] = (),
    created_at: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    self_url: str | None = None,
) -> dict:
    """Build a TRACE TRO for a populace dataset build.

    Keyword-only. The composition binds, by sha256: every output artifact
    (the dataset H5 and calibration weights), every input artifact (restricted
    survey inputs the build consumed), and — when supplied — content-hashed
    payloads for the build config (``build_config``), the gate report
    (``gate_report``), and the stage records (``stage_records``). Output
    artifacts are arranged at their published (HF/HTTPS) locations; inputs are
    arranged at their canonical access location *even when restricted*, with
    ``pop:accessRestricted`` set on the location node so a reader knows the
    bytes are unreachable to them but the hash is still verifiable by someone
    with access.

    Args:
        build_id: The build identifier (``populace-us-2024-<sha>-<date>``).
        builder: Who/what ran the build (``"populace"``).
        build_sha: The short git sha of the build code.
        build_date: The build date (``YYYY-MM-DD``).
        config_manifest: ``BuildConfig.to_manifest()`` output, or ``None``.
        stage_records: A sequence of ``StageRecord`` manifest dicts (NOT
            ``StageRecord`` objects — the module stays decoupled from
            :mod:`populace.build.plan`), or ``None``.
        gate_manifest: ``GateReport.to_manifest()`` output, or ``None``.
        input_artifacts: Mappings ``{id, name, sha256, location,
            mime_type?, access_restricted?}`` for restricted inputs.
        output_artifacts: The same shape for the dataset H5 + calibration.
        created_at: The TRO's ``schema:dateCreated`` (ISO-8601).
        started_at: ``trov:startedAtTime`` of the build performance.
        ended_at: ``trov:endedAtTime`` of the build performance.
        self_url: Canonical URL this TRO is published at; recorded as
            ``pop:selfUrl`` so a verifier holding only the bytes can find the
            published copy.

    Raises:
        ValueError: If any input or output artifact has no ``sha256``.
    """
    specs: list[dict[str, Any]] = [
        _artifact_spec_from_mapping(artifact, role="output")
        for artifact in output_artifacts
    ]
    specs.extend(
        _artifact_spec_from_mapping(artifact, role="input")
        for artifact in input_artifacts
    )
    if config_manifest is not None:
        specs.append(
            _payload_spec(config_manifest, spec_id="build_config", name="build_config")
        )
    if gate_manifest is not None:
        specs.append(
            _payload_spec(gate_manifest, spec_id="gate_report", name="gate_report")
        )
    if stage_records is not None:
        specs.append(
            _payload_spec(
                list(stage_records),
                spec_id="stage_records",
                name="stage_records",
            )
        )

    composition, arrangement = _assemble_composition_and_arrangement(specs)
    performance = _build_performance(
        build_id=build_id,
        builder=builder,
        build_sha=build_sha,
        gate_manifest=gate_manifest,
        started_at=started_at,
        ended_at=ended_at,
    )

    node: dict[str, Any] = {
        "@id": "tro",
        "@type": "trov:TransparentResearchObject",
        "schema:name": f"populace build TRO ({build_id})",
        "schema:description": (
            "TRACE TRO for a populace dataset build. The composition pins the "
            "build's output artifacts (calibrated dataset and weights) and the "
            "restricted survey inputs it consumed, by sha256, so a build can "
            "be credibly attested even where the inputs cannot be re-accessed."
        ),
        "schema:creator": dict(POLICYENGINE_ORGANIZATION),
        "trov:wasAssembledBy": _research_system(),
        "trov:createdWith": {
            "@type": "schema:SoftwareApplication",
            "schema:name": "populace-build",
            "schema:softwareVersion": _populace_build_version(),
        },
        "trov:hasComposition": composition,
        "trov:hasArrangement": [arrangement],
        "trov:hasPerformance": performance,
        "pop:buildDate": build_date,
    }
    if created_at is not None:
        node["schema:dateCreated"] = created_at
    if self_url is not None:
        node["pop:selfUrl"] = self_url
    return {"@context": TRACE_CONTEXT, "@graph": [node]}


def _build_performance(
    *,
    build_id: str,
    builder: str,
    build_sha: str,
    gate_manifest: Mapping[str, Any] | None,
    started_at: str | None,
    ended_at: str | None,
) -> dict[str, Any]:
    performance: dict[str, Any] = {
        "@id": "trp/1",
        "@type": "trov:TransparentResearchPerformance",
        "rdfs:comment": f"populace build {build_id}.",
        "trov:wasConductedBy": {"@id": "trs"},
        "trov:accessedArrangement": {"@id": _ARRANGEMENT_ID},
        "pop:buildId": build_id,
        "pop:buildSha": build_sha,
        "pop:builder": builder,
    }
    if started_at is not None:
        performance["trov:startedAtTime"] = started_at
    if ended_at is not None:
        performance["trov:endedAtTime"] = ended_at
    if gate_manifest is not None:
        gates_passed = _gates_passed(gate_manifest)
        if gates_passed is not None:
            performance["pop:gatesPassed"] = gates_passed
        performance["pop:gateNames"] = sorted(_gate_names(gate_manifest))
    performance.update(_emission_context())
    return performance


def _gate_names(gate_manifest: Mapping[str, Any]) -> list[str]:
    """Gate names from a manifest, tolerating both shapes the build emits.

    Two gate-manifest shapes appear in the repo: the structured
    ``GateReport.to_manifest()`` output ``{"passed", "gates": {name: ...}}``,
    and the flat ``release_manifest.json`` ``gates`` block whose keys are the
    gate names directly (``parity_gaps``, ``exported_nonzero``,
    ``calibration``, ``smoke``). Either way the keys under ``gates`` (or the
    top level, minus the reserved ``passed``) are the names.
    """
    inner = gate_manifest.get("gates")
    if isinstance(inner, Mapping):
        return list(inner.keys())
    return [key for key in gate_manifest.keys() if key != "passed"]


def _gates_passed(gate_manifest: Mapping[str, Any]) -> bool | None:
    """An honest overall gate verdict, or ``None`` when none is recorded.

    ``pop:gatesPassed`` is a claim, so it is only emitted when the manifest
    actually records verdicts. The derivation, in order:

    1. An explicit top-level ``passed`` bool (the ``GateReport.to_manifest()``
       shape) is used as-is.
    2. Otherwise (the flat ``release_manifest.json`` shape) every recorded
       per-gate verdict is collected: a gate whose value is a bool, a gate
       whose value is a mapping with a ``passed`` bool, and a ``*_gaps``
       integer counter (zero gaps is that gate's pass condition). The overall
       verdict is the conjunction of recorded verdicts.
    3. Evidence-only gates (e.g. ``smoke`` aggregates, calibration fit
       metrics) record no pass criterion and contribute nothing; with no
       recorded verdicts at all, no overall claim is made and the gate block
       stays bound in the composition as evidence only.
    """
    explicit = gate_manifest.get("passed")
    if isinstance(explicit, bool):
        return explicit
    verdicts: list[bool] = []
    inner = gate_manifest.get("gates")
    entries = inner if isinstance(inner, Mapping) else gate_manifest
    for name, value in entries.items():
        if name == "passed":
            continue
        if isinstance(value, bool):
            verdicts.append(value)
        elif isinstance(value, Mapping) and isinstance(value.get("passed"), bool):
            verdicts.append(value["passed"])
        elif name.endswith("_gaps") and isinstance(value, int):
            verdicts.append(value == 0)
    if not verdicts:
        return None
    return all(verdicts)


#: The published dataset repo populace-us artifacts live in (see
#: ``populace.data.registry``). The release manifest does not record its
#: own repo yet, so converters take it as a parameter with this default.
DEFAULT_HF_REPO = "policyengine/populace-us"


def tro_from_release_manifest(
    manifest: Mapping[str, Any],
    *,
    input_artifacts: Sequence[Mapping[str, Any]] = (),
    self_url: str | None = None,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_revision: str = "main",
) -> dict:
    """Convert a populace release manifest into a TRACE TRO.

    Maps the release manifest fields:

    - schema-v1 ``build`` / ``artifacts`` fields, or the earlier top-level
      ``build_id`` / ``dataset`` / ``calibration`` smoke-manifest fields;
    - output artifacts, located at their canonical Hugging Face URLs;
    - the whole ``gates`` block -> the ``gate_report`` payload, with
      ``pop:gatesPassed``/``pop:gateNames`` derived from its keys;
    - ``construction`` -> the build-config payload;
    - optional ``stage_records`` -> the stage-records payload slot.

    Restricted inputs (the IRS PUF, Fed SCF, Census SIPP) are not in the
    release manifest — pass them via ``input_artifacts`` to bind them too.
    This makes emission mechanical for the already-shipped build.

    ``hf_repo`` / ``hf_revision`` locate the output artifacts on the Hub.
    Pass the pinned revision of the release when known — ``main`` is a
    moving ref, so a revision-pinned location is the immutable one.

    Raises:
        ValueError: If a declared output artifact lacks a ``sha256``.
    """
    build_info = manifest.get("build")
    if isinstance(build_info, Mapping):
        build_id = str(build_info.get("build_id", ""))
        builder = str(build_info.get("builder", manifest.get("builder", "populace")))
        build_sha = str(build_info.get("build_sha", manifest.get("build_sha", "")))
        build_date = str(build_info.get("build_date", manifest.get("build_date", "")))
    else:
        build_id = str(manifest["build_id"])
        builder = str(manifest.get("builder", "populace"))
        build_sha = str(manifest.get("build_sha", ""))
        build_date = str(manifest.get("build_date", ""))

    outputs: list[dict[str, Any]] = []
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        for artifact_id, artifact in artifacts.items():
            if not isinstance(artifact, Mapping):
                continue
            outputs.append(
                _output_from_manifest_artifact(
                    artifact,
                    spec_id=str(artifact_id),
                    hf_repo=hf_repo,
                    hf_revision=hf_revision,
                )
            )
    else:
        dataset = manifest.get("dataset")
        if isinstance(dataset, Mapping):
            outputs.append(
                _output_from_manifest_artifact(
                    dataset,
                    spec_id="dataset",
                    hf_repo=hf_repo,
                    hf_revision=hf_revision,
                )
            )
        calibration = manifest.get("calibration")
        if isinstance(calibration, Mapping):
            outputs.append(
                _output_from_manifest_artifact(
                    calibration,
                    spec_id="calibration",
                    hf_repo=hf_repo,
                    hf_revision=hf_revision,
                )
            )

    config_manifest = (
        {"construction": manifest["construction"]}
        if "construction" in manifest
        else None
    )
    stage_records_value = manifest.get("stage_records")
    stage_records = (
        list(stage_records_value)
        if isinstance(stage_records_value, Sequence)
        and not isinstance(stage_records_value, (str, bytes))
        else None
    )

    return build_build_tro(
        build_id=build_id,
        builder=builder,
        build_sha=build_sha,
        build_date=build_date,
        config_manifest=config_manifest,
        stage_records=stage_records,
        gate_manifest=manifest.get("gates"),
        input_artifacts=input_artifacts,
        output_artifacts=outputs,
        created_at=_created_at_from_manifest(manifest),
        self_url=self_url,
    )


def _created_at_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    """``schema:dateCreated`` for a release-manifest TRO.

    Uses the build date as-is (it is a calendar date, not an instant — the
    manifest records no finer timestamp).
    """
    build_info = manifest.get("build")
    build_date = (
        build_info.get("build_date")
        if isinstance(build_info, Mapping)
        else manifest.get("build_date")
    )
    return str(build_date) if build_date else None


def _output_from_manifest_artifact(
    artifact: Mapping[str, Any],
    *,
    spec_id: str,
    hf_repo: str,
    hf_revision: str,
) -> dict[str, Any]:
    """Build an output-artifact mapping from a manifest artifact entry.

    Locates the file at its canonical Hugging Face dataset URL. The sha256
    is carried through as-is; :func:`build_build_tro` raises when it is
    missing so the failure is legible.
    """
    filename = artifact.get("filename") or artifact.get("path")
    artifact_repo = artifact.get("repo_id")
    if not isinstance(artifact_repo, str) or not artifact_repo:
        artifact_repo = hf_repo
    return {
        "id": spec_id,
        "name": filename,
        "sha256": artifact.get("sha256"),
        "location": (
            _huggingface_location(
                filename, repo=artifact_repo, revision=hf_revision
            )
            if filename
            else None
        ),
        "mime_type": _mime_type_for(str(filename or ""), None),
    }


def _huggingface_location(filename: str, *, repo: str, revision: str) -> str:
    """Canonical HTTPS ``resolve`` URL for a dataset artifact on the Hub.

    ``https://huggingface.co/datasets/<repo>/resolve/<revision>/<filename>``
    mirrors policyengine.py's ``https_dataset_uri`` convention so a location
    is a real fetchable URL, not an opaque ``hf://`` scheme.
    """
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"


def serialize_tro(tro: Mapping[str, Any]) -> bytes:
    """Serialize a TRO with the same canonical JSON used for hashing.

    Deterministic: the same TRO dict always serializes to identical bytes, so
    a published ``.trace.tro.jsonld`` is reproducible.
    """
    return canonical_json_bytes(tro)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: emit a TRO from a release manifest.

    ``python -m populace.build.trace <release_manifest.json> [--out PATH]
    [--self-url URL]`` reads the manifest JSON and writes the serialized TRO
    to ``--out`` (or stdout when omitted).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="populace.build.trace",
        description=(
            "Emit a TRACE Transparent Research Object (TRO) from a populace "
            "release manifest."
        ),
    )
    parser.add_argument(
        "manifest", help="Path to the release_manifest.json to convert."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the TRO here (default: stdout).",
    )
    parser.add_argument(
        "--self-url",
        default=None,
        help="Canonical URL the TRO is published at (recorded as pop:selfUrl).",
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help="Hugging Face dataset repo holding the output artifacts.",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help=(
            "Hub revision for artifact locations. Pass the release's pinned "
            "revision when known; main is a moving ref."
        ),
    )
    args = parser.parse_args(argv)

    with open(args.manifest, "rb") as handle:
        manifest = json.load(handle)

    tro = tro_from_release_manifest(
        manifest,
        self_url=args.self_url,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
    )
    payload = serialize_tro(tro)

    if args.out is None:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    else:
        with open(args.out, "wb") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the module CLI
    raise SystemExit(main())
