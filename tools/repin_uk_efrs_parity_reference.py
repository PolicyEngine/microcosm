#!/usr/bin/env python
"""Re-pin the UK enhanced-FRS parity reference to another uk-data release.

The reference artifact's identity (Hugging Face revision, sha256, size) is
mirrored in eight committed files and attested by the totals digest and four
gate-battery digests. #723 moved it by hand across those files; this tool
makes the move one reviewed command and refuses to leave a half-moved tree.

Steps, in order:

1. resolve the target release's artifact identity — an HF tag resolved to its
   commit and the artifact's LFS sha256/size through the HF API, or explicit
   ``--revision/--sha256/--size-bytes`` pins — and the licensed local bytes
   (HF cache, ``--input-h5``, or ``--download``), always sha-verified;
2. rewrite every committed mirror of the old identity in lockstep and refuse
   if any old literal survives;
3. regenerate the two parity instruments from the artifact with the committed
   tools (engine pinned by uv.lock) and emit the gate's weighted-totals
   sidecar OUTSIDE the repository (#609);
4. move ``totals_sha256`` (the sidecar's canonical evidence digest) and re-cut
   the gate-battery digests that attest the moved spec — policy, gates
   manifest, spec fingerprint, and the input-mass evidence digest;
5. write a disclosure-safe receipt (aggregates only, no unit records, no
   totals) and run the lockstep tests.

``--dry-run`` performs every computation into a scratch directory and writes
the receipt without touching the repository — the evidence a maintainer needs
to decide which release to pin. Nothing here publishes anything: the committed
instruments remain the reviewed artifacts and the licensed sidecar stays
uncommitted.

Examples::

    # evidence only (nothing in the repository changes)
    uv run python tools/repin_uk_efrs_parity_reference.py \
        --release 1.56.16 --scratch /tmp/repin --dry-run

    # the reviewed move
    uv run python tools/repin_uk_efrs_parity_reference.py \
        --release 1.56.16 --totals-out ~/ukds/uk_input_mass_reference_2024_25.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SRC = REPO_ROOT / "packages" / "microcosm-build" / "src" / "microcosm" / "build"
UK_DIR = BUILD_SRC / "uk"
UK_RUNTIME_DIR = BUILD_SRC / "uk_runtime"
DATA_SRC = REPO_ROOT / "packages" / "microcosm-data" / "src" / "microcosm" / "data"
BUILD_TESTS = REPO_ROOT / "packages" / "microcosm-build" / "tests"
DATA_TESTS = REPO_ROOT / "packages" / "microcosm-data" / "tests"

REFERENCE_PATH = UK_DIR / "efrs_parity_reference.json"
MANIFEST_PATH = UK_DIR / "release_input_coverage_manifest.json"
GATES_PATH = UK_DIR / "gates.json"
FRS_RELEASE_PATH = UK_DIR / "frs_release.json"
WEIGHTED_INTEGRITY_PATH = UK_RUNTIME_DIR / "weighted_integrity.py"
CONTRACT_PATH = DATA_SRC / "contract.py"
PARITY_TOOL_PATH = REPO_ROOT / "tools" / "build_uk_efrs_parity_reference.py"
COVERAGE_TOOL_PATH = REPO_ROOT / "tools" / "build_uk_release_input_coverage_manifest.py"
TEST_CONTRACT_PATH = DATA_TESTS / "test_contract.py"

#: Committed mirrors of the artifact identity (revision / sha256 / size). The
#: two instruments are regenerated rather than edited, so they are not here.
IDENTITY_MIRRORS: tuple[Path, ...] = (
    PARITY_TOOL_PATH,
    WEIGHTED_INTEGRITY_PATH,
    CONTRACT_PATH,
    GATES_PATH,
    BUILD_TESTS / "test_uk_parity_reference.py",
    BUILD_TESTS / "test_uk_terminal_gates.py",
    BUILD_TESTS / "test_uk_weighted_integrity.py",
    TEST_CONTRACT_PATH,
)
#: Committed mirrors of the totals digest (``totals_sha256``).
TOTALS_DIGEST_MIRRORS: tuple[Path, ...] = (
    GATES_PATH,
    WEIGHTED_INTEGRITY_PATH,
    CONTRACT_PATH,
    TEST_CONTRACT_PATH,
)
#: Committed mirrors of the four gate-battery digests.
BATTERY_DIGEST_MIRRORS: tuple[Path, ...] = (CONTRACT_PATH, TEST_CONTRACT_PATH)

#: contract.py constant -> report_payload key (or the evidence digest).
BATTERY_DIGEST_CONSTANTS = {
    "_UK_GATE_BATTERY_POLICY_SHA256": "policy_sha256",
    "_UK_GATE_BATTERY_GATES_MANIFEST_SHA256": "gates_manifest_sha256",
    "_UK_GATE_BATTERY_SPEC_FINGERPRINT": "spec_fingerprint",
    "_UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256": "input_mass_evidence_sha256",
}

INPUT_MASS_GATE_ID = "uk_input_mass_parity"
#: release_id used only to instantiate the battery for its digests (it does
#: not enter policy/manifest/fingerprint); mirrors the contract-pins test.
BATTERY_DIGEST_RELEASE_ID = "populace-uk-2023-frs-k535080"

LOCKSTEP_TESTS: tuple[str, ...] = (
    "packages/microcosm-build/tests/test_uk_parity_reference.py",
    "packages/microcosm-build/tests/test_uk_release_input_coverage_manifest.py",
    "packages/microcosm-build/tests/test_uk_release_input_coverage.py",
    "packages/microcosm-build/tests/test_uk_efrs_weighted_totals.py",
    "packages/microcosm-build/tests/test_uk_weighted_integrity.py",
    "packages/microcosm-build/tests/test_uk_terminal_gates.py",
    "packages/microcosm-build/tests/test_uk_battery_bindings.py",
    "packages/microcosm-build/tests/test_gate_battery_contract_pins.py",
    "packages/microcosm-build/tests/test_spec_engine_country_bundles.py",
    "packages/microcosm-build/tests/test_country_spec.py",
    "packages/microcosm-build/tests/test_uk_frs_release.py",
    "packages/microcosm-build/tests/test_uk_hmrc_replay_artifacts.py",
    "packages/microcosm-build/tests/test_uk_efrs_repin_tool.py",
    "packages/microcosm-data/tests/test_contract.py",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ArtifactIdentity:
    """The pinned reference artifact: one HF revision, one byte identity."""

    revision: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _HEX40.match(self.revision):
            raise ValueError(
                f"revision must be a 40-hex HF commit; got {self.revision!r}"
            )
        if not _HEX64.match(self.sha256):
            raise ValueError(f"sha256 must be lowercase 64-hex; got {self.sha256!r}")
        if int(self.size_bytes) <= 0:
            raise ValueError("size_bytes must be positive")


# --------------------------------------------------------------------------- #
# Reading the committed state
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return payload


def committed_identity(reference_path: Path = REFERENCE_PATH) -> ArtifactIdentity:
    """The identity the committed parity reference was extracted from."""

    source = _load_json(reference_path)["source"]
    return ArtifactIdentity(
        revision=str(source["revision"]),
        sha256=str(source["sha256"]),
        size_bytes=int(source["size_bytes"]),
    )


def committed_totals_digest(gates_path: Path = GATES_PATH) -> tuple[str, str]:
    """(active reference name, totals_sha256) from the committed gates.json."""

    entry = _input_mass_entry(_load_json(gates_path))
    parameters = entry["parameters"]
    active = str(parameters["reference"])
    registry = parameters["reference_registry"]
    return active, str(registry[active]["totals_sha256"])


def _input_mass_entry(gates_payload: dict[str, Any]) -> dict[str, Any]:
    for entry in gates_payload["gates"]:
        if entry.get("id") == INPUT_MASS_GATE_ID:
            return entry
    raise ValueError(f"{GATES_PATH}: no gate entry {INPUT_MASS_GATE_ID!r}.")


def committed_battery_digests(contract_path: Path = CONTRACT_PATH) -> dict[str, str]:
    """The four gate-battery digest literals pinned in the data-shard contract."""

    text = contract_path.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for constant in BATTERY_DIGEST_CONSTANTS:
        match = re.search(rf'^{constant} = \(\n\s+"([0-9a-f]{{64}})"\n\)', text, re.M)
        if match is None:
            raise ValueError(f"{contract_path}: could not locate {constant}.")
        found[constant] = match.group(1)
    return found


# --------------------------------------------------------------------------- #
# Resolving the target release
# --------------------------------------------------------------------------- #


def _hf_token(token_file: Path | None) -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    if token_file is not None and token_file.is_file():
        return token_file.read_text(encoding="utf-8").strip() or None
    return None


def resolve_release_identity(
    release: str,
    *,
    repo_id: str,
    repo_type: str,
    filename: str,
    token: str | None,
) -> ArtifactIdentity:
    """HF tag -> commit; the artifact's LFS sha256/size at that commit."""

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    refs = api.list_repo_refs(repo_id, repo_type=repo_type)
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    if release not in tags:
        raise SystemExit(
            f"HF repo {repo_id} has no tag {release!r}; known 1.56.x tags: "
            f"{sorted(name for name in tags if name.startswith('1.56.'))}"
        )
    commit = str(tags[release])
    infos = api.get_paths_info(
        repo_id, [filename], revision=commit, repo_type=repo_type
    )
    if not infos:
        raise SystemExit(f"{repo_id}@{commit} has no {filename}.")
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if not sha:
        raise SystemExit(
            f"{repo_id}@{commit}/{filename} is not an LFS file; no sha256."
        )
    return ArtifactIdentity(revision=commit, sha256=str(sha), size_bytes=int(info.size))


def resolve_zip_identity(
    revision: str,
    *,
    repo_id: str,
    repo_type: str,
    zip_filename: str,
    token: str | None,
) -> tuple[str, int]:
    """The raw-FRS zip's LFS sha256/size at an HF revision (frs_release.json)."""

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    infos = api.get_paths_info(
        repo_id, [zip_filename], revision=revision, repo_type=repo_type
    )
    if not infos:
        raise SystemExit(f"{repo_id}@{revision} has no {zip_filename}.")
    lfs = getattr(infos[0], "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if not sha:
        raise SystemExit(f"{repo_id}@{revision}/{zip_filename} is not an LFS file.")
    return str(sha), int(infos[0].size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(
    identity: ArtifactIdentity,
    *,
    repo_id: str,
    repo_type: str,
    filename: str,
    explicit: Path | None,
    download: bool,
    token: str | None,
) -> Path:
    """Locate and sha-verify the licensed bytes for ``identity``."""

    candidate: Path | None = None
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
    else:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            revision=identity.revision,
            repo_type=repo_type,
        )
        if isinstance(cached, str):
            candidate = Path(cached)
        elif download:
            candidate = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=identity.revision,
                    repo_type=repo_type,
                    token=token,
                )
            )
        else:
            raise SystemExit(
                f"{filename}@{identity.revision[:12]} is not in the HF cache; pass "
                "--input-h5 <path> or --download."
            )
    if not candidate.is_file():
        raise SystemExit(f"artifact not found: {candidate}")
    size = candidate.stat().st_size
    if size != identity.size_bytes:
        raise SystemExit(
            f"{candidate}: size {size} != pinned {identity.size_bytes} for "
            f"revision {identity.revision[:12]}."
        )
    digest = _sha256_file(candidate)
    if digest != identity.sha256:
        raise SystemExit(f"{candidate}: sha256 {digest} != pinned {identity.sha256}.")
    return candidate


# --------------------------------------------------------------------------- #
# Lockstep literal moves
# --------------------------------------------------------------------------- #


def _size_literals(size: int) -> tuple[str, str]:
    """Python (underscored) and JSON (plain) renderings of a size literal."""

    return f"{size:_}", str(size)


def move_literals(
    files: tuple[Path, ...],
    replacements: dict[str, str],
    *,
    label: str,
    write: bool,
) -> dict[str, dict[str, int]]:
    """Replace ``old -> new`` literals across ``files``; refuse leftovers.

    Every file must contain at least one of the old literals (a mirror that
    has already drifted is a defect to surface, not to paper over), and after
    the pass no old literal may survive in the set.
    """

    counts: dict[str, dict[str, int]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        per_file: dict[str, int] = {}
        for old, new in replacements.items():
            if old == new:
                continue
            n = text.count(old)
            if n:
                per_file[old] = n
                text = text.replace(old, new)
        if not per_file:
            raise SystemExit(
                f"{label}: {_display(path)} carries none of the old literals — the "
                "mirror set has drifted; inspect before re-pinning."
            )
        counts[_display(path)] = per_file
        if write:
            path.write_text(text, encoding="utf-8")
    if write:
        for path in files:
            survivors = [
                old
                for old, new in replacements.items()
                if old != new and old in path.read_text(encoding="utf-8")
            ]
            if survivors:
                raise SystemExit(
                    f"{label}: {_display(path)} still carries {survivors}."
                )
    return counts


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def identity_replacements(
    old: ArtifactIdentity, new: ArtifactIdentity
) -> dict[str, str]:
    old_py, old_json = _size_literals(old.size_bytes)
    new_py, new_json = _size_literals(new.size_bytes)
    return {
        old.revision: new.revision,
        old.sha256: new.sha256,
        old_py: new_py,
        old_json: new_json,
    }


# --------------------------------------------------------------------------- #
# Instrument regeneration (in-process, pins patched in memory)
# --------------------------------------------------------------------------- #


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parity_tool(identity: ArtifactIdentity):
    """The committed extraction tool with its SOURCE_* pins set to ``identity``.

    The in-memory patch keeps the regeneration on the committed code path
    (engine classification, share convention, totals schema) while letting
    --dry-run extract a release the repository does not pin yet.
    """

    tool = _load_module(PARITY_TOOL_PATH, "build_uk_efrs_parity_reference")
    tool.SOURCE_REVISION = identity.revision
    tool.SOURCE_SHA256 = identity.sha256
    tool.SOURCE_SIZE_BYTES = identity.size_bytes
    tool.SOURCE_URL = (
        f"https://huggingface.co/{tool.SOURCE_REPO_ID}/resolve/"
        f"{identity.revision}/{tool.SOURCE_FILENAME}"
    )
    return tool


def render_json(payload: dict[str, Any]) -> str:
    """The committed tools' rendering (indent=1, sorted keys, trailing newline)."""

    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def canonical_totals_digest(totals_payload: dict[str, Any]) -> str:
    """The gate's canonical evidence digest of a schema-1 totals sidecar."""

    from microcosm.build.uk_runtime.weighted_integrity import (
        UKInputMassReference,
        _input_mass_reference_evidence_sha256,
    )

    identity = totals_payload["identity"]
    reference = UKInputMassReference(
        totals=dict(totals_payload["totals"]),
        filename=str(identity["filename"]),
        revision=str(identity["revision"]),
        sha256=str(identity["sha256"]),
        vintage=str(identity["vintage"]),
    )
    return _input_mass_reference_evidence_sha256(reference)


def build_coverage_manifest(reference: dict[str, Any]) -> dict[str, Any]:
    """The coverage manifest the committed tool would write for ``reference``."""

    tool = _load_module(COVERAGE_TOOL_PATH, "build_uk_release_input_coverage_manifest")
    known_gaps = _load_json(tool.KNOWN_GAPS_PATH)
    # The checked-in candidate evidence was measured against the reference's
    # populated-column surface; a release that moves that surface needs the
    # candidate evidence refreshed (--candidate-h5 on the coverage tool) before
    # the manifest can be rebuilt honestly.
    evidence_columns = set(known_gaps["candidate_evidence"]["nonzero_shares"])
    reference_columns = set(reference["nonzero_shares"])
    if evidence_columns != reference_columns:
        raise SystemExit(
            "the new reference's populated-column surface differs from the "
            "checked-in candidate evidence (only in reference: "
            f"{sorted(reference_columns - evidence_columns)}; only in evidence: "
            f"{sorted(evidence_columns - reference_columns)}); refresh "
            f"{tool.KNOWN_GAPS_PATH.name} with the certified candidate first."
        )
    return tool.build_manifest(reference=reference, known_gaps_payload=known_gaps)


def recut_battery_digests(
    gates_payload: dict[str, Any], totals_digest: str
) -> dict[str, str]:
    """Policy / manifest / fingerprint / input-mass evidence digests for a gates payload."""

    from microcosm.build.country_spec import GatesManifest
    from microcosm.build.gate_battery import GateBatteryRun, _canonical_sha256
    from microcosm.build.uk_runtime.battery_bindings import (
        UK_GATE_REGISTRY,
        _exclusion_payload,
    )
    from microcosm.build.uk_runtime.weighted_integrity import (
        uk_default_input_mass_reviewed_exclusions,
    )

    gates = GatesManifest.from_mapping(gates_payload, country="uk")
    with tempfile.TemporaryDirectory() as scratch:
        run = GateBatteryRun(
            gates,
            release_id=BATTERY_DIGEST_RELEASE_ID,
            report_path=Path(scratch) / "terminal_gates.json",
            release_candidate=False,
            registry=UK_GATE_REGISTRY,
        )
        payload = run.report_payload()
    active = str(_input_mass_entry(gates_payload)["parameters"]["reference"])
    committed = uk_default_input_mass_reviewed_exclusions()
    evidence = _canonical_sha256(
        {
            "reference": active,
            "reference_evidence_sha256": totals_digest,
            "exclusions_policy": "committed",
            "reviewed_exclusions": _exclusion_payload(dict(committed.get(active, {}))),
        }
    )
    return {
        "policy_sha256": str(payload["policy_sha256"]),
        "gates_manifest_sha256": str(payload["gates_manifest_sha256"]),
        "spec_fingerprint": str(payload["spec_fingerprint"]),
        "input_mass_evidence_sha256": evidence,
    }


def _gates_with_identity(
    gates_payload: dict[str, Any],
    *,
    new: ArtifactIdentity,
    totals_digest: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(gates_payload))
    entry = _input_mass_entry(payload)
    parameters = entry["parameters"]
    active = str(parameters["reference"])
    registry_entry = parameters["reference_registry"][active]
    registry_entry["identity"]["revision"] = new.revision
    registry_entry["identity"]["sha256"] = new.sha256
    registry_entry["totals_sha256"] = totals_digest
    return payload


# --------------------------------------------------------------------------- #
# Receipt (aggregates only)
# --------------------------------------------------------------------------- #


def _share_deltas(old_ref: dict[str, Any], new_ref: dict[str, Any]) -> dict[str, Any]:
    old_shares = old_ref["nonzero_shares"]
    new_shares = new_ref["nonzero_shares"]
    common = sorted(set(old_shares) & set(new_shares))
    deltas = sorted(
        ((new_shares[k] - old_shares[k], k) for k in common), key=lambda t: -abs(t[0])
    )
    return {
        "columns_old": len(old_shares),
        "columns_new": len(new_shares),
        "columns_only_old": sorted(set(old_shares) - set(new_shares)),
        "columns_only_new": sorted(set(new_shares) - set(old_shares)),
        "changed_columns": sum(1 for d, _ in deltas if d != 0),
        "max_abs_delta": {"column": deltas[0][1], "delta": round(deltas[0][0], 6)}
        if deltas
        else None,
        "beyond_0_001": [
            {
                "column": k,
                "delta": round(d, 6),
                "entity": new_ref["input_entities"].get(k),
            }
            for d, k in deltas
            if abs(d) > 0.001
        ],
        "entity_stats_old": old_ref["entity_stats"],
        "entity_stats_new": new_ref["entity_stats"],
        "input_entities_equal": old_ref["input_entities"] == new_ref["input_entities"],
    }


def _totals_deltas(
    old_totals: dict[str, float] | None,
    new_totals: dict[str, float],
    entities: dict[str, str],
) -> dict[str, Any]:
    if old_totals is None:
        return {"columns": len(new_totals), "comparison": "old sidecar not supplied"}
    rel: list[tuple[float, str]] = []
    for key in sorted(set(old_totals) & set(new_totals)):
        a, b = float(old_totals[key]), float(new_totals[key])
        if a == 0.0 and b == 0.0:
            continue
        rel.append((abs(b - a) / abs(a) if a else float("inf"), key))
    rel.sort(reverse=True)
    values = [r for r, _ in rel]
    by_entity: dict[str, list[float]] = {}
    for r, key in rel:
        by_entity.setdefault(str(entities.get(key)), []).append(r)
    return {
        "columns_old": len(old_totals),
        "columns_new": len(new_totals),
        "same_keys": set(old_totals) == set(new_totals),
        "relative_delta_median": round(statistics.median(values), 6)
        if values
        else None,
        "relative_delta_p90": round(sorted(values)[int(0.9 * (len(values) - 1))], 6)
        if values
        else None,
        "relative_delta_max": {
            "column": rel[0][1],
            "relative_delta": round(rel[0][0], 6),
        }
        if rel
        else None,
        "beyond_5pct": [
            {"column": key, "relative_delta": round(r, 4), "entity": entities.get(key)}
            for r, key in rel
            if r > 0.05
        ],
        "by_entity": {
            entity: {
                "columns": len(vals),
                "median": round(statistics.median(vals), 6),
                "max": round(max(vals), 6),
            }
            for entity, vals in sorted(by_entity.items())
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_argument_group("target release")
    target.add_argument(
        "--release",
        help="uk-data release tag on the HF repo (e.g. 1.56.16); resolved through the HF API.",
    )
    target.add_argument(
        "--revision", help="explicit HF commit (40 hex) instead of --release"
    )
    target.add_argument("--sha256", help="explicit artifact sha256 (with --revision)")
    target.add_argument(
        "--size-bytes", type=int, help="explicit artifact size (with --revision)"
    )
    source = parser.add_argument_group("licensed bytes")
    source.add_argument("--input-h5", type=Path, help="local artifact (sha-verified)")
    source.add_argument(
        "--download",
        action="store_true",
        help="download the artifact at the resolved revision if it is not cached",
    )
    source.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="file holding an HF token (env HF_TOKEN / HUGGINGFACE_TOKEN / HUGGING_FACE_TOKEN win)",
    )
    source.add_argument(
        "--old-totals",
        type=Path,
        help=(
            "the current reference's uncommitted totals sidecar, for the receipt's "
            "totals comparison; regenerated from the cached old artifact when omitted "
            "and cached, otherwise the comparison is skipped"
        ),
    )
    out = parser.add_argument_group("outputs")
    out.add_argument(
        "--totals-out",
        type=Path,
        help="where to write the gate's totals sidecar (must be OUTSIDE the repository)",
    )
    out.add_argument(
        "--scratch",
        type=Path,
        help="dry-run / receipt directory (default: a temporary directory)",
    )
    out.add_argument(
        "--receipt",
        type=Path,
        help="receipt JSON path (default: <scratch>/repin_receipt.json)",
    )
    mode = parser.add_argument_group("mode")
    mode.add_argument(
        "--dry-run", action="store_true", help="compute everything; edit nothing"
    )
    mode.add_argument(
        "--move-frs-release-acquisition",
        action="store_true",
        help=(
            "also move uk/frs_release.json acquisition.huggingface_revision to the new "
            "revision, after verifying the raw FRS zip's LFS sha256 there equals the "
            "pinned zip_sha256 (the bytes are what the pin attests; the revision is "
            "where they were fetched from)"
        ),
    )
    mode.add_argument(
        "--skip-tests", action="store_true", help="do not run the lockstep tests"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tool_defaults = _load_module(
        PARITY_TOOL_PATH, "build_uk_efrs_parity_reference_defaults"
    )
    repo_id = tool_defaults.SOURCE_REPO_ID
    repo_type = tool_defaults.SOURCE_REPO_TYPE
    filename = tool_defaults.SOURCE_FILENAME
    token = _hf_token(args.hf_token_file)

    old = committed_identity()
    active_reference, old_totals_digest = committed_totals_digest()
    old_battery = committed_battery_digests()

    if args.release:
        new = resolve_release_identity(
            args.release,
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            token=token,
        )
    elif args.revision and args.sha256 and args.size_bytes:
        new = ArtifactIdentity(args.revision, args.sha256, int(args.size_bytes))
    else:
        raise SystemExit(
            "pass --release TAG, or --revision/--sha256/--size-bytes together."
        )

    if not args.dry_run:
        if args.totals_out is None:
            raise SystemExit("--totals-out is required unless --dry-run.")
        totals_out = args.totals_out.expanduser().resolve()
        if totals_out.is_relative_to(REPO_ROOT):
            raise SystemExit(
                f"{totals_out} is inside the repository; the totals sidecar is "
                "UKDS-derived gate input and stays uncommitted (#609)."
            )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if dirty:
            raise SystemExit(
                "working tree has uncommitted tracked changes; commit or stash them "
                "so the re-pin is one reviewable diff:\n" + dirty
            )

    scratch = (
        args.scratch.expanduser().resolve()
        if args.scratch
        else Path(tempfile.mkdtemp(prefix="uk-efrs-repin-"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    if scratch.is_relative_to(REPO_ROOT):
        raise SystemExit(f"--scratch {scratch} must be outside the repository.")
    receipt_path = (
        args.receipt.expanduser().resolve()
        if args.receipt
        else scratch / "repin_receipt.json"
    )

    artifact = resolve_artifact(
        new,
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        explicit=args.input_h5,
        download=args.download,
        token=token,
    )

    # --- regenerate the instruments in memory -------------------------------
    tool = _parity_tool(new)
    new_reference = tool.build_reference(artifact)
    new_totals = tool.build_weighted_totals(artifact)
    new_totals_digest = canonical_totals_digest(new_totals)
    old_reference = _load_json(REFERENCE_PATH)
    new_manifest = build_coverage_manifest(new_reference)
    old_manifest = _load_json(MANIFEST_PATH)

    old_totals_payload: dict[str, Any] | None = None
    if args.old_totals is not None:
        old_totals_payload = _load_json(args.old_totals.expanduser().resolve())
    else:
        try:
            old_artifact = resolve_artifact(
                old,
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                explicit=None,
                download=False,
                token=token,
            )
        except SystemExit:
            old_artifact = None
        if old_artifact is not None:
            old_totals_payload = _parity_tool(old).build_weighted_totals(old_artifact)
    if old_totals_payload is not None:
        observed = canonical_totals_digest(old_totals_payload)
        if observed != old_totals_digest:
            raise SystemExit(
                "the old totals sidecar does not reproduce the committed totals_sha256 "
                f"({observed} != {old_totals_digest}); the regeneration path or the "
                "supplied sidecar is not the reviewed one — stop."
            )

    committed_gates = _load_json(GATES_PATH)
    # The recompute path must reproduce the committed pins before it is
    # trusted to mint new ones.
    reproduced = recut_battery_digests(committed_gates, old_totals_digest)
    mismatched = {
        constant: (old_battery[constant], reproduced[key])
        for constant, key in BATTERY_DIGEST_CONSTANTS.items()
        if old_battery[constant] != reproduced[key]
    }
    if mismatched:
        raise SystemExit(
            "the digest recompute does not reproduce the committed gate-battery pins "
            f"(constant: (pinned, recomputed)) {mismatched}; the spec or the "
            "recompute path has drifted — stop."
        )
    new_gates = _gates_with_identity(
        committed_gates, new=new, totals_digest=new_totals_digest
    )
    new_battery = recut_battery_digests(new_gates, new_totals_digest)

    frs_release_move: dict[str, Any] | None = None
    if args.move_frs_release_acquisition:
        release_payload = _load_json(FRS_RELEASE_PATH)
        acquisition = release_payload["acquisition"]
        zip_sha, zip_size = resolve_zip_identity(
            new.revision,
            repo_id=str(acquisition["huggingface_repo"]),
            repo_type=repo_type,
            zip_filename=str(acquisition["zip_filename"]),
            token=token,
        )
        if zip_sha != acquisition["zip_sha256"] or zip_size != int(
            acquisition["zip_size_bytes"]
        ):
            raise SystemExit(
                f"{acquisition['zip_filename']} at {new.revision[:12]} is not the pinned zip "
                f"(sha {zip_sha[:12]}… vs {str(acquisition['zip_sha256'])[:12]}…); refusing to move "
                "frs_release.json's acquisition revision."
            )
        frs_release_move = {
            "from": str(acquisition["huggingface_revision"]),
            "to": new.revision,
            "zip_sha256": zip_sha,
            "zip_size_bytes": zip_size,
        }

    # --- receipt ---------------------------------------------------------------
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "tool": "tools/repin_uk_efrs_parity_reference.py",
        "mode": "dry-run" if args.dry_run else "apply",
        "repo_id": repo_id,
        "filename": filename,
        "engine": new_reference["engine"]["version"],
        "identity": {"from": asdict(old), "to": asdict(new)},
        "reference": _share_deltas(old_reference, new_reference),
        "coverage_manifest": {
            "reference_block_from": old_manifest.get("reference"),
            "reference_block_to": new_manifest.get("reference"),
            "counts_from": old_manifest.get("counts"),
            "counts_to": new_manifest.get("counts"),
        },
        "totals": {
            "sha256_from": old_totals_digest,
            "sha256_to": new_totals_digest,
            **_totals_deltas(
                None if old_totals_payload is None else old_totals_payload["totals"],
                new_totals["totals"],
                new_reference["input_entities"],
            ),
        },
        "battery_digests": {
            key: {"from": old_battery[constant], "to": new_battery[key]}
            for constant, key in BATTERY_DIGEST_CONSTANTS.items()
        },
        "frs_release_acquisition": frs_release_move,
        "active_reference": active_reference,
    }

    if args.dry_run:
        (scratch / "efrs_parity_reference.json").write_text(
            render_json(new_reference), encoding="utf-8"
        )
        (scratch / "release_input_coverage_manifest.json").write_text(
            render_json(new_manifest), encoding="utf-8"
        )
        (scratch / "uk_input_mass_reference_2024_25.json").write_text(
            render_json(new_totals), encoding="utf-8"
        )
        (scratch / "gates.json").write_text(
            json.dumps(new_gates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        receipt["scratch"] = str(scratch)
        receipt_path.write_text(
            json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(_receipt_summary(receipt), indent=1))
        print(
            f"dry-run: nothing in {REPO_ROOT} changed; outputs under {scratch}; receipt {receipt_path}"
        )
        return 0

    # --- apply -----------------------------------------------------------------
    touched: dict[str, Any] = {}
    touched["identity"] = move_literals(
        IDENTITY_MIRRORS, identity_replacements(old, new), label="identity", write=True
    )
    REFERENCE_PATH.write_text(render_json(new_reference), encoding="utf-8")
    MANIFEST_PATH.write_text(render_json(new_manifest), encoding="utf-8")
    totals_out.parent.mkdir(parents=True, exist_ok=True)
    totals_out.write_text(render_json(new_totals), encoding="utf-8")
    touched["totals_digest"] = move_literals(
        TOTALS_DIGEST_MIRRORS,
        {old_totals_digest: new_totals_digest},
        label="totals_sha256",
        write=True,
    )
    touched["battery_digests"] = move_literals(
        BATTERY_DIGEST_MIRRORS,
        {
            old_battery[constant]: new_battery[key]
            for constant, key in BATTERY_DIGEST_CONSTANTS.items()
        },
        label="gate-battery digests",
        write=True,
    )
    if frs_release_move is not None:
        touched["frs_release_acquisition"] = move_literals(
            (FRS_RELEASE_PATH,),
            {frs_release_move["from"]: frs_release_move["to"]},
            label="frs_release",
            write=True,
        )
    # The committed tools must agree with what was written: re-run them in
    # --check mode from a fresh interpreter (constants now on disk).
    for command in (
        [sys.executable, str(PARITY_TOOL_PATH), "--input-h5", str(artifact), "--check"],
        [sys.executable, str(COVERAGE_TOOL_PATH), "--check"],
    ):
        result = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise SystemExit(
                f"post-write check failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
            )
    receipt["touched"] = touched
    receipt["totals_out"] = str(totals_out)
    tests: dict[str, Any] = {"ran": not args.skip_tests}
    if not args.skip_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *LOCKSTEP_TESTS],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        tests["returncode"] = result.returncode
        tests["tail"] = "\n".join(result.stdout.splitlines()[-15:])
        if result.returncode != 0:
            sys.stderr.write(
                result.stdout[-6000:] + "\n" + result.stderr[-3000:] + "\n"
            )
    receipt["tests"] = tests
    receipt_path.write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_receipt_summary(receipt), indent=1))
    print(
        f"re-pinned {old.revision[:12]} -> {new.revision[:12]}; receipt {receipt_path}; totals {totals_out}"
    )
    return 0 if tests.get("returncode", 0) == 0 else 1


def _receipt_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": receipt["mode"],
        "identity": receipt["identity"],
        "reference_columns": (
            receipt["reference"]["columns_old"],
            receipt["reference"]["columns_new"],
        ),
        "reference_changed_columns": receipt["reference"]["changed_columns"],
        "reference_max_abs_delta": receipt["reference"]["max_abs_delta"],
        "totals_sha256": (
            receipt["totals"]["sha256_from"],
            receipt["totals"]["sha256_to"],
        ),
        "totals_relative_delta": {
            k: receipt["totals"].get(k)
            for k in (
                "relative_delta_median",
                "relative_delta_p90",
                "relative_delta_max",
            )
        },
        "totals_beyond_5pct": len(receipt["totals"].get("beyond_5pct", [])),
        "battery_digests": receipt["battery_digests"],
        "frs_release_acquisition": receipt["frs_release_acquisition"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
