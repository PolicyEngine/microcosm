"""Build, sign and assemble a Microcosm UK national cut in one resumable command.

A cut used to be five commands run by hand on one machine (spine, national
calibration, certify, assemble, publish; ``docs/uk-national-release-assembly-runbook-806.md``),
with inputs gathered from wherever they happened to be. This command runs
them in order from a clean checkout, with every input fetched and checked
against the pins the build itself enforces, so anyone with the credentials
can cut at the policyengine-uk version their checkout locks::

    uv run --no-sync python tools/run_uk_cut.py status  --label sat24
    uv run --no-sync python tools/run_uk_cut.py fetch
    uv run --no-sync python tools/run_uk_cut.py preflight --label sat24
    uv run --no-sync python tools/run_uk_cut.py all --label sat24
    uv run --no-sync python tools/run_uk_cut.py all --label sat24 --publish

``all`` is resumable: each step first checks whether its outputs already
exist and verify (the done-check), and skips if so. A step that fails leaves
its outputs for inspection; nothing here deletes a spine, a calibration run
or a release directory. Re-run ``all`` after fixing the cause; a failed
national run is moved aside only with ``--retry-national``.

Outward-facing steps are opt-in. ``--publish`` publishes for inspection
only, always with ``--no-latest --tag-only``, so the repository's main
branch and its ``latest-*.json`` pointers never move. Promotion and the
policyengine.py certification are separate, deliberate calls; this command
prints them and runs neither.

Credentials come from the environment: ``HF_TOKEN`` (write access to
``policyengine/populace-uk-private`` and ``policyengine/populace-uk-staging``,
read access to the input repositories) and
``MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY``, which must be registered in
``microcosm.data.gate_signing_keys``. See ``docs/uk-cut-anyone.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
UK_DIR = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
CUT_INPUTS = REPO_ROOT / "tools/uk_cut_inputs.json"
DEFAULT_WORK_DIR = Path("~/microcosm-uk-work").expanduser()
RELEASE_ID = "microcosm-uk-2024-25-national"
CANDIDATE_NAME = "microcosm_uk_2024_25"
SPINE_NAME = "uk_frs_spine_2024_25.h5"
PUBLISH_REPO = "policyengine/populace-uk-private"
TELEMETRY_REPO = "policyengine/populace-uk-staging"
STAGING_BRANCH = f"release-staging/{RELEASE_ID}"


class CutError(RuntimeError):
    """A refusal the operator must act on; the message says how."""


# --------------------------------------------------------------------------- #
# Pins and input sources                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PinnedFile:
    """One input file and the digest the build checks it against."""

    name: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class InputSource:
    """Where one group of input files comes from (``tools/uk_cut_inputs.json``)."""

    key: str
    kind: str
    directory: str
    files: tuple[PinnedFile, ...]
    repo_id: str | None = None
    repo_type: str | None = None
    revision: str | None = None
    filename: str | None = None
    path_in_repo: str | None = None
    archive_sha256: str | None = None
    url: str | None = None
    upload_note: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_artifacts(node: object):
    if isinstance(node, Mapping):
        if "sha256" in node and (node.get("filename") or node.get("locator")):
            yield node
        for value in node.values():
            yield from _walk_artifacts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_artifacts(value)


def committed_source_pins(uk_dir: Path = UK_DIR) -> dict[str, dict[str, Any]]:
    """Every file pin in ``source_stages.json``, by file name.

    Tabs are named by ``filename`` or ``locator``; the HMRC workbook by its
    URL, whose last segment is its file name. A name pinned twice must carry
    the same digest.
    """

    stages = json.loads((uk_dir / "source_stages.json").read_text())
    pins: dict[str, dict[str, Any]] = {}
    for artifact in _walk_artifacts(stages):
        raw = str(artifact.get("filename") or artifact.get("locator"))
        name = raw.rsplit("/", 1)[-1]
        pin = {
            "sha256": str(artifact["sha256"]),
            "size_bytes": artifact.get("size_bytes"),
            "url": raw if raw.startswith("https://") else None,
        }
        previous = pins.get(name)
        if previous is not None and previous["sha256"] != pin["sha256"]:
            raise CutError(f"source_stages.json pins {name} to two digests.")
        pins[name] = previous or pin
    return pins


def _gate_reference_identity(uk_dir: Path) -> Mapping[str, str]:
    gates = json.loads((uk_dir / "gates.json").read_text())
    entries = gates["gates"] if isinstance(gates, Mapping) else gates
    for entry in entries:
        if entry.get("gate") == "input_mass_parity":
            params = entry["parameters"]
            return params["reference_registry"][params["reference"]]["identity"]
    raise CutError("gates.json declares no input_mass_parity reference identity.")


#: The ``*_pin`` selectors a source may carry, each naming the committed pin
#: a field is read from. Any other ``*_pin`` is refused, never ignored.
PIN_SELECTORS: Mapping[str, frozenset[str]] = {
    "archive_sha256_pin": frozenset({"frs_release"}),
    "revision_pin": frozenset({"gates:input_mass_parity"}),
    "path_in_repo_pin": frozenset({"chronicle_feed"}),
}


def _check_pin_selectors(raw: Mapping[str, Any]) -> None:
    for name, value in raw.items():
        if not name.endswith("_pin"):
            continue
        if name == "url_pin":
            known = value in {f"source_stages:{member}" for member in raw["members"]}
        else:
            known = value in PIN_SELECTORS.get(name, frozenset())
        if not known:
            raise CutError(f"{raw['key']}: unknown pin selector {name}={value!r}.")


def load_input_sources(
    manifest_path: Path = CUT_INPUTS, uk_dir: Path = UK_DIR
) -> list[InputSource]:
    """Resolve each declared source to files and the digests the build checks."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise CutError(f"{manifest_path}: version must be 1.")
    stage_pins = committed_source_pins(uk_dir)
    feed = json.loads((uk_dir / "chronicle_feed.json").read_text())
    frs_release = json.loads((uk_dir / "frs_release.json").read_text())
    reference = _gate_reference_identity(uk_dir)
    extra = manifest.get("file_pins", {})
    sources: list[InputSource] = []
    for raw in manifest["sources"]:
        _check_pin_selectors(raw)
        key = raw["key"]
        revision = raw.get("revision")
        archive_sha256 = raw.get("archive_sha256")
        path_in_repo = raw.get("path_in_repo")
        if raw.get("path_in_repo_pin") == "chronicle_feed":
            # The hosted artifact sits under its Chronicle commit, so the
            # path follows every re-pin of chronicle_feed.json.
            path_in_repo = f"{path_in_repo}/{feed['source_commit']}"
        url = None
        files: list[PinnedFile] = []
        for member in raw["members"]:
            if key == "chronicle_uk":
                digest = {
                    "manifest.json": feed["manifest_sha256"],
                    "consumer_facts.jsonl": feed["facts_sha256"],
                }[member]
                files.append(PinnedFile(member, digest))
            elif key == "incumbent_efrs":
                if member != reference["filename"]:
                    raise CutError(
                        f"incumbent member {member} is not the gate's "
                        f"reference {reference['filename']}."
                    )
                files.append(PinnedFile(member, reference["sha256"]))
            elif member in extra:
                files.append(PinnedFile(member, extra[member]["sha256"]))
            elif member in stage_pins:
                pin = stage_pins[member]
                files.append(PinnedFile(member, pin["sha256"], pin["size_bytes"]))
                url = url or pin["url"]
            else:
                raise CutError(
                    f"{key}: no committed pin for {member}; every input must be "
                    "pinned in source_stages.json, chronicle_feed.json or gates.json."
                )
        if raw.get("archive_sha256_pin") == "frs_release":
            acquisition = frs_release["acquisition"]
            archive_sha256 = acquisition["zip_sha256"]
            if acquisition["huggingface_revision"] != revision:
                raise CutError(
                    "frs_2024_25 revision disagrees with uk/frs_release.json."
                )
        if raw.get("revision_pin") == "gates:input_mass_parity":
            revision = reference["revision"]
        if raw["kind"] == "url" and not url:
            raise CutError(f"{key}: the pin carries no https URL.")
        sources.append(
            InputSource(
                key=key,
                kind=raw["kind"],
                directory=raw["directory"],
                files=tuple(files),
                repo_id=raw.get("repo_id"),
                repo_type=raw.get("repo_type"),
                revision=revision,
                filename=raw.get("filename"),
                path_in_repo=path_in_repo,
                archive_sha256=archive_sha256,
                url=url if raw["kind"] == "url" else None,
                upload_note=raw.get("upload_note"),
            )
        )
    return sources


# --------------------------------------------------------------------------- #
# Fetching                                                                      #
# --------------------------------------------------------------------------- #


class Downloader(Protocol):
    def hf_file(
        self,
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        revision: str | None,
        local_dir: Path | None = None,
    ) -> Path: ...

    def url(self, url: str, dest: Path) -> Path: ...


class HubDownloader:
    """Download through the Hugging Face cache (resumable, deduplicated)."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def hf_file(
        self,
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        revision: str | None,
        local_dir: Path | None = None,
    ) -> Path:
        """Through the cache, or straight into ``local_dir`` for a file that
        will be moved into place (no second copy of a multi-gigabyte input)."""

        from huggingface_hub import hf_hub_download

        location = (
            {"local_dir": str(local_dir)}
            if local_dir is not None
            else {"cache_dir": str(self.cache_dir)}
        )
        return Path(
            hf_hub_download(
                repo_id, filename, repo_type=repo_type, revision=revision, **location
            )
        )

    def url(self, url: str, dest: Path) -> Path:
        import urllib.request

        tmp = dest.with_name(dest.name + ".partial")
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            tmp.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        tmp.replace(dest)
        return dest


class VerifiedFiles:
    """Remembers files already hashed, keyed by inode, size, mtime and ctime.

    A file is re-hashed when any of these changes, or always with
    ``reverify``. ``ctime`` moves on every write and on every ``utime`` call
    and cannot be set back from user space, so replacing a file's bytes and
    restoring its mtime still invalidates the entry: a cache entry never
    vouches for bytes it did not hash.
    """

    def __init__(self, path: Path, *, reverify: bool = False) -> None:
        self.path = path
        self.reverify = reverify
        try:
            self.entries: dict[str, dict[str, Any]] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.entries = {}

    @staticmethod
    def _identity(stat: os.stat_result) -> dict[str, int]:
        return {
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }

    def matches(self, file: Path, pin: PinnedFile) -> bool:
        if not file.is_file():
            return False
        stat = file.stat()
        if pin.size_bytes is not None and stat.st_size != pin.size_bytes:
            return False
        key = str(file.resolve())
        identity = self._identity(stat)
        cached = self.entries.get(key)
        if (
            not self.reverify
            and cached is not None
            and all(cached.get(field) == value for field, value in identity.items())
        ):
            return cached.get("sha256") == pin.sha256
        digest = _sha256_file(file)
        self.entries[key] = {**identity, "sha256": digest}
        return digest == pin.sha256

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
        tmp.replace(self.path)


def _place(source: Path, dest: Path, pin: PinnedFile, *, move: bool = False) -> None:
    """Verify, then install at ``dest`` by an atomic rename.

    ``move`` installs ``source`` itself (a download staged beside ``dest``);
    otherwise a copy is verified, so a shared cache entry is never moved.
    A file that misses its pin is deleted, never installed.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if move:
        try:
            os.replace(source, tmp)
        except OSError:  # another filesystem: fall back to a copy
            shutil.copyfile(source, tmp)
            source.unlink(missing_ok=True)
    else:
        shutil.copyfile(source, tmp)
    digest = _sha256_file(tmp)
    if digest != pin.sha256:
        tmp.unlink()
        raise CutError(
            f"{source} hashes to {digest}, not the pinned {pin.sha256} for "
            f"{pin.name}; it was not installed."
        )
    tmp.replace(dest)


def _extract_member(archive: Path, pin: PinnedFile, dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        matches = [
            info
            for info in bundle.infolist()
            if Path(info.filename).name == pin.name
            and not info.filename.startswith("__MACOSX")
            and not info.is_dir()
        ]
        if len(matches) != 1:
            raise CutError(
                f"{archive.name} holds {len(matches)} members named {pin.name}; "
                "expected exactly one."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        digest = hashlib.sha256()
        with bundle.open(matches[0]) as member, tmp.open("wb") as handle:
            for chunk in iter(lambda: member.read(1 << 22), b""):
                digest.update(chunk)
                handle.write(chunk)
    if digest.hexdigest() != pin.sha256:
        tmp.unlink()
        raise CutError(
            f"{archive.name}:{pin.name} hashes to {digest.hexdigest()}, not the "
            f"pinned {pin.sha256}; it was not installed."
        )
    tmp.replace(dest)


@dataclass
class FetchResult:
    key: str
    status: str  # present | fetched | missing | refused
    detail: str = ""
    files: dict[str, str] = field(default_factory=dict)


def fetch_inputs(
    sources: Sequence[InputSource],
    inputs_dir: Path,
    *,
    downloader: Downloader,
    reverify: bool = False,
    only: set[str] | None = None,
) -> list[FetchResult]:
    """Install every pinned input under ``inputs_dir`` and verify it.

    Idempotent: a file already present with its pinned digest is left alone.
    Nothing unverified is ever left at an input's final path.
    """

    verified = VerifiedFiles(inputs_dir / ".verified.json", reverify=reverify)
    results: list[FetchResult] = []
    try:
        for source in sources:
            if only and source.key not in only:
                continue
            target = inputs_dir / source.directory
            paths = {pin.name: target / pin.name for pin in source.files}
            if all(verified.matches(paths[pin.name], pin) for pin in source.files):
                results.append(
                    FetchResult(
                        source.key,
                        "present",
                        files={n: str(p) for n, p in paths.items()},
                    )
                )
                continue
            try:
                _fetch_source(source, paths, downloader=downloader, verified=verified)
            except CutError as error:
                results.append(FetchResult(source.key, "refused", str(error)))
                continue
            except Exception as error:  # the Hub's 404/401, a dropped connection
                status = "missing" if _looks_missing(error) else "refused"
                detail = f"{type(error).__name__}: {error}"
                if status == "missing" and source.upload_note:
                    detail += f" | needs upload: {source.upload_note}"
                results.append(FetchResult(source.key, status, detail))
                continue
            results.append(
                FetchResult(
                    source.key, "fetched", files={n: str(p) for n, p in paths.items()}
                )
            )
    finally:
        verified.save()
    return results


def _looks_missing(error: Exception) -> bool:
    name = type(error).__name__
    return name in {"EntryNotFoundError", "RemoteEntryNotFoundError"} or (
        "404" in str(error)
    )


def _fetch_source(
    source: InputSource,
    paths: Mapping[str, Path],
    *,
    downloader: Downloader,
    verified: VerifiedFiles,
) -> None:
    pending = [
        pin for pin in source.files if not verified.matches(paths[pin.name], pin)
    ]
    if source.kind == "hf_zip_members":
        archive = downloader.hf_file(
            repo_id=source.repo_id,
            repo_type=source.repo_type,
            filename=source.filename,
            revision=source.revision,
        )
        if source.archive_sha256 is not None:
            digest = _sha256_file(archive)
            if digest != source.archive_sha256:
                raise CutError(
                    f"{source.filename} hashes to {digest}, not the pinned "
                    f"{source.archive_sha256}."
                )
        for pin in pending:
            _extract_member(archive, pin, paths[pin.name])
    elif source.kind == "hf_files":
        for pin in pending:
            name = (
                f"{source.path_in_repo}/{pin.name}" if source.path_in_repo else pin.name
            )
            local = downloader.hf_file(
                repo_id=source.repo_id,
                repo_type=source.repo_type,
                filename=name,
                revision=source.revision,
                local_dir=paths[pin.name].parent / ".download",
            )
            _place(local, paths[pin.name], pin, move=True)
    elif source.kind == "url":
        for pin in pending:
            with tempfile.TemporaryDirectory() as scratch:
                local = downloader.url(source.url, Path(scratch) / pin.name)
                _place(local, paths[pin.name], pin, move=True)
    else:
        raise CutError(f"{source.key}: unknown source kind {source.kind!r}.")
    for pin in pending:
        if not verified.matches(paths[pin.name], pin):
            raise CutError(f"{paths[pin.name]} did not verify after install.")


# --------------------------------------------------------------------------- #
# The cut's layout                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CutPaths:
    """Every path a cut reads or writes, derived from the work dir, label and commit."""

    work_dir: Path
    label: str
    commit: str

    @property
    def inputs(self) -> Path:
        return self.work_dir / "inputs"

    @property
    def cut(self) -> Path:
        return self.work_dir / "cuts" / f"{self.label}-{self.commit[:9]}"

    @property
    def spine_dir(self) -> Path:
        return self.cut / "spine"

    @property
    def spine_h5(self) -> Path:
        return self.spine_dir / SPINE_NAME

    @property
    def run_dir(self) -> Path:
        return self.cut / "national"

    @property
    def candidate_h5(self) -> Path:
        return self.run_dir / f"{CANDIDATE_NAME}.h5"

    @property
    def release_parent(self) -> Path:
        return self.cut / "release"

    @property
    def release_dir(self) -> Path:
        return self.release_parent / "releases" / RELEASE_ID

    @property
    def logs(self) -> Path:
        return self.cut / "logs"

    @property
    def diagnostics(self) -> Path:
        return self.cut / "diagnostics"

    @property
    def state(self) -> Path:
        return self.cut / "state.json"

    @property
    def receipt(self) -> Path:
        return self.cut / "cut_receipt.json"

    def input_file(self, directory: str, name: str) -> Path:
        return self.inputs / directory / name


def spine_argv(paths: CutPaths, *, python: str, staging_local_only: bool) -> list[str]:
    """The spine command, every input taken from the verified inputs dir."""

    def tab(directory: str, name: str) -> str:
        return str(paths.input_file(directory, name))

    argv = [
        python,
        str(REPO_ROOT / "tools/build_uk_frs_spine.py"),
        "--frs-raw-dir", str(paths.inputs / "frs_2024_25"),
        "--spi-tab", tab("spi_2022_23", "put2223uk.tab"),
        "--hmrc-ods", tab("hmrc", "Collated_Tables_3_1_to_3_11_2324.ods"),
        "--was-tab", tab("was_round_8", "was_round_8_hhold_eul_may_2025_230525.tab"),
        "--was-person-tab",
        tab("was_round_8", "was_round_8_person_eul_may_2025_230525.tab"),
        "--nts-household-tab", tab("nts_2002_2024", "household_eul_2002-2024.tab"),
        "--nts-individual-tab", tab("nts_2002_2024", "individual_eul_2002-2024.tab"),
        "--nts-trip-tab", tab("nts_2002_2024", "trip_eul_2002-2024.tab"),
        "--nts-stage-tab", tab("nts_2002_2024", "stage_eul_2002-2024.tab"),
        "--nts-ticket-tab", tab("nts_2002_2024", "ticket_eul_2002-2024.tab"),
        "--lcfs-hh-tab", tab("lcfs_2024_25", "dvhh_uk_anon_v3_2024.tab"),
        "--lcfs-person-tab", tab("lcfs_2024_25", "dvper_uk_anon_2024.tab"),
        "--etb-tab", tab("etb_1977_2024", "householdv2_1977-2024.tab"),
        "--release-candidate",
        "--spine-h5", str(paths.spine_h5),
    ]  # fmt: skip
    if staging_local_only:
        argv.append("--staging-local-only")
    return argv


def national_argv(
    paths: CutPaths,
    *,
    python: str,
    spine_sha256: str,
    sources: Mapping[str, InputSource],
    dry_run: bool,
    logbook_prev_row_digest: str | None = None,
) -> list[str]:
    chronicle = {pin.name: pin.sha256 for pin in sources["chronicle_uk"].files}
    incumbent = sources["incumbent_efrs"].files[0]
    argv = [
        python,
        str(REPO_ROOT / "tools/build_uk_full.py"),
        "--release-role", "national",
        "--input-h5", str(paths.spine_h5),
        "--input-sha256", spine_sha256,
        "--ledger-facts", str(paths.inputs / "chronicle_uk"),
        "--ledger-facts-sha256", chronicle["consumer_facts.jsonl"],
        "--ledger-manifest-sha256", chronicle["manifest.json"],
        "--incumbent-h5", str(paths.input_file("incumbent", incumbent.name)),
        "--incumbent-sha256", incumbent.sha256,
        # The build otherwise submits its graph to the public Orrery whenever
        # staging is remote; the graph files stay in the run directory.
        "--no-publish-orrery",
        "--out", str(paths.run_dir),
    ]  # fmt: skip
    if logbook_prev_row_digest:
        argv += ["--logbook-prev-row-digest", logbook_prev_row_digest]
    if dry_run:
        argv.append("--dry-run")
    return argv


def score_argv(
    paths: CutPaths, *, python: str, candidate_sha256: str, sources: Mapping
) -> list[str]:
    incumbent = sources["incumbent_efrs"].files[0]
    return [
        python,
        str(REPO_ROOT / "tools/score_uk_national_candidate.py"),
        "--candidate-h5", str(paths.candidate_h5),
        "--candidate-sha256", candidate_sha256,
        "--incumbent-h5", str(paths.input_file("incumbent", incumbent.name)),
        "--incumbent-sha256", incumbent.sha256,
        "--registry-json", str(paths.run_dir / "national_target_registry.json"),
        "--band-edge-registry-json",
        str(paths.run_dir / "national_contract_registry.json"),
        "--output-json", str(paths.run_dir / "score_vs_incumbent.json"),
    ]  # fmt: skip


def certify_argv(
    paths: CutPaths,
    *,
    python: str,
    candidate_sha256: str,
    spine_sha256: str,
    sources: Mapping,
    logbook_prev_row_digest: str | None = None,
) -> list[str]:
    chronicle = {pin.name: pin.sha256 for pin in sources["chronicle_uk"].files}
    reference = sources["input_mass_reference"].files[0]
    argv = [
        python,
        str(REPO_ROOT / "tools/certify_uk_release_cut.py"),
        "--candidate-h5", str(paths.candidate_h5),
        "--candidate-sha256", candidate_sha256,
        "--candidate-name", CANDIDATE_NAME,
        "--spine-h5", str(paths.spine_h5),
        "--spine-sha256", spine_sha256,
        "--diagnostics-json", str(paths.run_dir / "calibration_diagnostics.json"),
        "--build-record-json", str(paths.run_dir / "build_record.json"),
        "--seam-gate-report",
        str(paths.run_dir / f"{CANDIDATE_NAME}.terminal_gates.json"),
        "--ledger-facts", str(paths.inputs / "chronicle_uk"),
        "--ledger-facts-sha256", chronicle["consumer_facts.jsonl"],
        "--ledger-manifest-sha256", chronicle["manifest.json"],
        "--input-mass-reference",
        str(paths.input_file("input_mass_reference", reference.name)),
        "--score-receipt", str(paths.run_dir / "score_vs_incumbent.json"),
        "--release-id", RELEASE_ID,
    ]  # fmt: skip
    if logbook_prev_row_digest:
        argv += ["--logbook-prev-row-digest", logbook_prev_row_digest]
    return argv


def assemble_argv(paths: CutPaths, *, python: str) -> list[str]:
    return [
        python,
        str(REPO_ROOT / "tools/assemble_uk_release_dir.py"),
        "--candidate-h5", str(paths.candidate_h5),
        "--spine-h5", str(paths.spine_h5),
        "--certification-json",
        str(paths.run_dir / f"{CANDIDATE_NAME}.release_certification.json"),
        "--build-record-json", str(paths.run_dir / "build_record.json"),
        "--diagnostics-json", str(paths.run_dir / "calibration_diagnostics.json"),
        "--seam-gate-report",
        str(paths.run_dir / f"{CANDIDATE_NAME}.terminal_gates.json"),
        "--release-cut-gate-json",
        str(paths.run_dir / f"{CANDIDATE_NAME}.release_cut_gates.json"),
        "--score-receipt", str(paths.run_dir / "score_vs_incumbent.json"),
        "--out-dir", "releases",
    ]  # fmt: skip


def publish_argv(
    paths: CutPaths, *, python: str, tag: str, preflight_only: bool
) -> list[str]:
    """Inspection publication: never the default line, never main's files."""

    argv = [
        python, "-m", "microcosm.data.publish_cli", str(paths.release_dir),
        "--repo-id", PUBLISH_REPO,
        "--artifact-root", str(paths.run_dir),
        "--no-latest", "--tag-only",
        "--tag-name", tag,
    ]  # fmt: skip
    if preflight_only:
        argv.append("--preflight-only")
    return argv


def promote_command(paths: CutPaths, tag: str) -> str:
    return shlex.join(
        [
            "uv", "run", "--no-sync", "python", "-m", "microcosm.data.publish_cli",
            str(paths.release_dir), "--repo-id", PUBLISH_REPO,
            "--artifact-root", str(paths.run_dir),
            "--promote-line", "national", "--tag-name", tag,
        ]
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# Done-checks                                                                   #
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def spine_sha256(paths: CutPaths) -> str | None:
    sidecar = paths.spine_h5.with_name(paths.spine_h5.name + ".sha256")
    try:
        return sidecar.read_text().split()[0]
    except (FileNotFoundError, IndexError):
        return None


def spine_done(paths: CutPaths) -> tuple[bool, str]:
    """A complete spine: H5, sidecars, matching digest, shippable gate report."""

    h5 = paths.spine_h5
    if not h5.exists():
        return False, "no spine yet"
    recorded = spine_sha256(paths)
    for suffix in (".build.json", ".spine_gates.json"):
        if not h5.with_suffix(suffix).exists():
            return False, f"spine sidecar {h5.with_suffix(suffix).name} is missing"
    if recorded is None:
        return False, "spine .sha256 sidecar is missing"
    if _sha256_file(h5) != recorded:
        return False, "spine H5 does not match its .sha256 sidecar"
    gates = _read_json(h5.with_suffix(".spine_gates.json")) or {}
    if gates.get("shippable") is not True:
        return False, "spine gate report is not shippable"
    return True, f"spine {recorded}"


def national_done(paths: CutPaths) -> tuple[bool, str]:
    record = _read_json(paths.run_dir / "build_record.json")
    if record is None:
        return False, "no build_record.json"
    for name in (
        f"{CANDIDATE_NAME}.h5",
        "calibration_diagnostics.json",
        f"{CANDIDATE_NAME}.terminal_gates.json",
        "national_target_registry.json",
        "national_contract_registry.json",
    ):
        if not (paths.run_dir / name).exists():
            return False, f"{name} is missing"
    expected = (record.get("artifacts") or {}).get("staging_h5", {}).get("sha256")
    if not expected or _sha256_file(paths.candidate_h5) != expected:
        return False, "candidate H5 does not match build_record.json"
    return True, f"candidate {expected}"


def candidate_sha256(paths: CutPaths) -> str:
    record = _read_json(paths.run_dir / "build_record.json") or {}
    return str(record["artifacts"]["staging_h5"]["sha256"])


def score_done(paths: CutPaths) -> tuple[bool, str]:
    """The fields the certifier reads: the scored candidate and the verdict."""

    receipt = _read_json(paths.run_dir / "score_vs_incumbent.json")
    if receipt is None:
        return False, "no score receipt"
    scored = ((receipt.get("artifacts") or {}).get("candidate") or {}).get("sha256")
    if scored != candidate_sha256(paths):
        return False, "score receipt names another candidate"
    verdict = (receipt.get("evaluation") or {}).get("verdict")
    if verdict != "passed":
        return False, f"score verdict is {verdict!r}"
    return True, "score passed"


def certify_done(paths: CutPaths) -> tuple[bool, str]:
    cert = _read_json(paths.run_dir / f"{CANDIDATE_NAME}.release_certification.json")
    if cert is None:
        return False, "no release certification"
    if cert.get("shippable") is not True:
        return False, "certification is not shippable"
    if (cert.get("candidate") or {}).get("sha256") != candidate_sha256(paths):
        return False, "certification names another candidate"
    return True, "certified shippable"


def assemble_done(paths: CutPaths) -> tuple[bool, str]:
    manifest = paths.release_dir / "release_manifest.json"
    summary = _read_json(paths.cut / "assemble_summary.json")
    if not manifest.exists():
        return False, "no release directory"
    if summary is None or not summary.get("tag"):
        return False, "release directory exists but its assembly summary is missing"
    return True, f"assembled {summary['tag']}"


# --------------------------------------------------------------------------- #
# Preflight                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


def _git(*argv: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(["git", *argv], cwd=cwd, text=True).strip()


def git_checks(*, allow_off_main: str | None) -> list[Check]:
    checks = []
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    checks.append(Check("clean checkout", not dirty, dirty or "clean"))
    head = _git("rev-parse", "HEAD")
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"], cwd=REPO_ROOT, check=False
    )
    on_main = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, "origin/main"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if on_main:
        checks.append(Check("commit on origin/main", True, head))
    else:
        pushed = bool(_git("branch", "-r", "--contains", head))
        checks.append(
            Check(
                "commit on origin/main",
                bool(allow_off_main) and pushed,
                (
                    f"{head} is not on origin/main"
                    + (
                        f"; allowed ({allow_off_main})"
                        if allow_off_main and pushed
                        else "; build from main, or pass --allow-off-main "
                        "'<who agreed and why>' for a pushed commit"
                    )
                ),
            )
        )
    return checks


def environment_checks() -> list[Check]:
    checks = []
    import microcosm.build as build_package

    location = Path(build_package.__file__).resolve()
    checks.append(
        Check(
            "env built from this checkout",
            REPO_ROOT in location.parents,
            f"microcosm.build from {location}",
        )
    )
    try:
        from microcosm.build.uk_runtime.hmrc_uprating import (
            assert_installed_engine_is_pinned,
        )

        version = assert_installed_engine_is_pinned()
        checks.append(Check("policyengine-uk pinned", True, str(version)))
    except Exception as error:
        checks.append(Check("policyengine-uk pinned", False, str(error)))
    return checks


def cut_signed_reports(paths: CutPaths) -> list[Path]:
    """The signed reports of the steps this cut has finished, in step order.

    A step's report counts only once the step's done-check passes, so an
    unfinished attempt about to be moved aside (``--retry-spine``,
    ``--retry-national``) never binds the key. These are the reports the
    assembler collects.
    """

    reports: list[Path] = []
    if not spine_done(paths)[0]:
        return reports
    reports.append(paths.spine_h5.with_suffix(".spine_gates.json"))
    if not national_done(paths)[0]:
        return reports
    reports.append(paths.run_dir / f"{CANDIDATE_NAME}.terminal_gates.json")
    if certify_done(paths)[0]:
        reports += [
            paths.run_dir / f"{CANDIDATE_NAME}.release_cut_gates.json",
            paths.run_dir / f"{CANDIDATE_NAME}.release_certification.json",
        ]
    return reports


def signed_report_fingerprints(reports: Sequence[Path]) -> dict[str, str]:
    """Each signed report's ``attestation.signing_key_sha256``, by file name.

    An unsigned report (no attestation, or a null fingerprint) is left out.
    """

    found: dict[str, str] = {}
    for report in reports:
        payload = _read_json(report)
        if payload is None:
            raise CutError(f"{report} is not a readable JSON report.")
        fingerprint = (payload.get("attestation") or {}).get("signing_key_sha256")
        if fingerprint:
            found[report.name] = str(fingerprint)
    return found


def reports_signing_key(reports: Sequence[Path]) -> str | None:
    """The fingerprint of the one key that signed these reports, if any."""

    fingerprints = signed_report_fingerprints(reports)
    signers = set(fingerprints.values())
    if len(signers) > 1:
        raise CutError(
            "this cut's reports were signed by more than one key "
            f"({fingerprints}); certification verifies every part with one "
            "key, so start a fresh cut with a new --label."
        )
    return signers.pop() if signers else None


def cut_signing_key(paths: CutPaths) -> str | None:
    """The fingerprint of the one key that signed this cut so far, if any."""

    return reports_signing_key(cut_signed_reports(paths))


def signing_key_check(
    *, paths: CutPaths | None = None, environ: Mapping[str, str] | None = None
) -> Check:
    """The key in the environment is registered and is this cut's signer.

    HMAC verification needs the secret that signed, and certification
    verifies the spine and seam reports with the one key in the environment,
    so a cut resumes only under the key its reports carry. Once certified,
    the remaining steps only verify, so a signer retired since then still
    finishes the cut.
    """

    from microcosm.data.gate_signing_keys import (
        UK_GATE_SIGNING_KEY_ENV,
        GateSigningKeyError,
        decode_gate_signing_key,
        gate_signing_key_fingerprint,
        registered_gate_signing_key,
        require_registered_gate_signing_key,
    )

    environ = os.environ if environ is None else environ
    try:
        reports = cut_signed_reports(paths) if paths is not None else []
        signer = reports_signing_key(reports)
    except CutError as error:
        return Check("signing key registered", False, str(error))
    certified = any(r.name.endswith(".release_certification.json") for r in reports)
    try:
        key = decode_gate_signing_key(
            environ.get(UK_GATE_SIGNING_KEY_ENV), env=UK_GATE_SIGNING_KEY_ENV
        )
        require_registered_gate_signing_key(
            "uk", key, purpose="verify" if certified else "sign"
        )
    except GateSigningKeyError as error:
        return Check("signing key registered", False, str(error))
    fingerprint = gate_signing_key_fingerprint(key)
    if signer is not None and signer != fingerprint:
        entry = registered_gate_signing_key("uk", signer)
        holder = f"{entry.label}, held by {entry.custodian}" if entry else "a key"
        return Check(
            "signing key registered",
            False,
            f"this cut was signed by {holder} ({signer}), not the key in "
            f"{UK_GATE_SIGNING_KEY_ENV} ({fingerprint}); resume it with that "
            "key, or start a fresh cut with a new --label.",
        )
    detail = fingerprint + ("; this cut's signer" if signer else "")
    return Check("signing key registered", True, detail)


def hub_checks() -> list[Check]:
    checks = []
    if not os.environ.get("HF_TOKEN"):
        return [
            Check("HF_TOKEN", False, "export HF_TOKEN (write to the release repos)")
        ]
    try:
        from huggingface_hub import HfApi

        from microcosm.build.staging_storage import HuggingFaceDatasetStorage

        api = HfApi()
        for repo in (PUBLISH_REPO, TELEMETRY_REPO):
            storage = HuggingFaceDatasetStorage(repo, api=api)
            storage.head_revision()
            can_write = storage.credential_can_write()
            checks.append(
                Check(
                    f"write {repo}",
                    can_write is not False,
                    "write scope confirmed"
                    if can_write
                    else "scope unreadable; the upload will prove it",
                )
            )
        refs = api.list_repo_refs(PUBLISH_REPO, repo_type="dataset")
        stale = [b.name for b in refs.branches if b.name == STAGING_BRANCH]
        checks.append(
            Check(
                "no leftover release-staging branch",
                not stale,
                f"delete {STAGING_BRANCH} after checking no publish is running"
                if stale
                else "none",
            )
        )
    except Exception as error:
        checks.append(Check("Hugging Face reachable", False, f"{type(error).__name__}"))
    return checks


def register_expiry_check(today: date, *, warn_days: int = 7) -> Check:
    earliest: tuple[str, str] | None = None
    for path in sorted(UK_DIR.glob("*.json")):
        text = path.read_text()
        if "expires_on" not in text:
            continue
        for match in _expiries(json.loads(text)):
            if earliest is None or match < earliest[0]:
                earliest = (match, path.name)
    if earliest is None:
        return Check("register expiries", True, "none")
    days = (date.fromisoformat(earliest[0]) - today).days
    detail = f"earliest {earliest[0]} in {earliest[1]} ({days} days)"
    if days < 0:
        return Check("register expiries", False, detail + ": expired")
    # An entry expiring soon still builds today; say so without blocking.
    return Check("register expiries", days >= warn_days, detail, blocking=False)


def _expiries(node: object):
    if isinstance(node, Mapping):
        value = node.get("expires_on")
        if isinstance(value, str) and len(value) == 10:
            yield value
        for child in node.values():
            yield from _expiries(child)
    elif isinstance(node, list):
        for child in node:
            yield from _expiries(child)


def host_checks(
    work_dir: Path,
    *,
    min_free_gb: float,
    reservation_tool: Path | None = None,
    reservation_caller: str | None = None,
) -> list[Check]:
    """Disk, power and load, and a host reservation only where one is configured.

    A reservation tool belongs to one machine, so the check runs only when the
    operator names the tool and the caller; a named tool that is missing fails
    rather than being skipped.
    """

    if (reservation_tool is None) != (reservation_caller is None):
        raise CutError("a host reservation needs both its tool and its caller.")
    checks = []
    probe = next(
        (path for path in (work_dir, *work_dir.parents) if path.exists()), Path.home()
    )
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 1e9
    checks.append(
        Check(
            "free disk",
            free_gb >= min_free_gb,
            f"{free_gb:.1f} GB free; floor {min_free_gb:.0f} GB",
        )
    )
    if platform.system() == "Darwin":
        power = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        ).stdout
        on_ac = "AC Power" in power.splitlines()[0] if power else True
        checks.append(
            Check(
                "on AC power",
                on_ac,
                "AC" if on_ac else "on battery: a closed lid stops long runs",
                blocking=False,
            )
        )
    load5 = os.getloadavg()[1]
    checks.append(
        Check(
            "load",
            True,
            f"load5 {load5:.1f} on {os.cpu_count()} CPUs",
            blocking=False,
        )
    )
    if reservation_tool is not None:
        if not reservation_tool.is_file():
            checks.append(
                Check("host reservation", False, f"{reservation_tool} not found")
            )
            return checks
        allowed = subprocess.run(
            [sys.executable, str(reservation_tool), "allowed", reservation_caller],
            capture_output=True,
            text=True,
            check=False,
        )
        checks.append(
            Check(
                "host reservation",
                allowed.returncode == 0,
                (allowed.stdout or allowed.stderr).strip()[:200] or "allowed",
            )
        )
    return checks


def contract_pin_check(python: str) -> Check:
    tests = [
        "packages/microcosm-build/tests/engine_free/shared/test_gate_battery_contract_pins.py",
        "packages/microcosm-data/tests/engine_free/shared/test_contract.py",
    ]
    run = subprocess.run(
        [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return Check(
        "contract pins agree with the gate spec",
        run.returncode == 0,
        _pytest_detail(run),
    )


def surfaces_check(python: str, chronicle_dir: Path) -> Check:
    """The feed-gated surface tests that skip in CI (runbook: stale receipts)."""

    env = {**os.environ, "CHRONICLE_UK_FACTS": str(chronicle_dir)}
    run = subprocess.run(
        [
            python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "packages/microcosm-build/tests/engine_free/uk/test_uk_target_references.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )  # fmt: skip
    return Check(
        "target surfaces current against the pinned feed",
        run.returncode == 0,
        _pytest_detail(run),
    )


def _pytest_detail(run: subprocess.CompletedProcess) -> str:
    """The repo's addopts already pass -q, so the summary line may be absent:
    report the exit code and any failing test ids instead."""

    failed = [
        line.split()[1]
        for line in (run.stdout or "").splitlines()
        if line.startswith(("FAILED ", "ERROR "))
    ]
    if run.returncode == 0:
        return "passed"
    return f"exit {run.returncode}" + (f": {', '.join(failed[:5])}" if failed else "")


# --------------------------------------------------------------------------- #
# Running steps                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Runner:
    paths: CutPaths
    sources: Mapping[str, InputSource]
    python: str = sys.executable
    execute: Callable[[list[str], Path, Path], int] | None = None
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        if self.execute is None:
            self.execute = _execute

    def state(self) -> dict[str, Any]:
        return _read_json(self.paths.state) or {
            "label": self.paths.label,
            "commit": self.paths.commit,
            "steps": {},
        }

    def record(self, step: str, **fields: Any) -> None:
        state = self.state()
        state["steps"].setdefault(step, {}).update(fields)
        self.paths.cut.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.state.with_name("state.json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        tmp.replace(self.paths.state)

    def run(self, step: str, argv: list[str], *, cwd: Path = REPO_ROOT) -> None:
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.paths.logs / f"{step}.log"
        started = time.monotonic()
        self.record(step, status="running", started_at=_now(), argv=argv)
        self.log(f"[{step}] {shlex.join(argv)}")
        code = self.execute(argv, cwd, log_path)
        seconds = round(time.monotonic() - started, 1)
        if code != 0:
            self.record(step, status="failed", exit_code=code, seconds=seconds)
            raise CutError(f"{step} exited {code}; see {log_path}")
        self.record(step, status="done", finished_at=_now(), seconds=seconds)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _execute(argv: list[str], cwd: Path, log_path: Path) -> int:
    """Run one step, teeing its output to the console and its log."""

    with log_path.open("a") as log:
        log.write(f"\n# {_now()} $ {shlex.join(argv)}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def _superseded(path: Path) -> Path:
    """Where a failed attempt's evidence goes when a step is retried."""

    return path.with_name(
        f"{path.name}.superseded-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    )


def _set_aside(runner: Runner, step: str, *paths: Path) -> None:
    """Keep a failed attempt's outputs, signed failure reports included."""

    for path in paths:
        if path.exists():
            aside = _superseded(path)
            path.rename(aside)
            runner.log(
                f"[{step}] kept the earlier attempt's {path.name} as {aside.name}"
            )


def run_spine(runner: Runner, *, staging_local_only: bool, retry: bool = False) -> None:
    paths = runner.paths
    done, detail = spine_done(paths)
    if done:
        runner.log(f"[spine] done: {detail}")
        return
    if paths.spine_dir.exists() and any(paths.spine_dir.iterdir()):
        # A gate-blocked spine writes its signed .spine_gates.json before any
        # H5, and the builder deletes its sidecars on start: never rerun over
        # an attempt, move the whole attempt aside instead.
        if not retry:
            raise CutError(
                f"{paths.spine_dir} holds an unfinished spine attempt ({detail}); "
                "pass --retry-spine to move it aside and build a fresh one."
            )
        aside = paths.spine_dir.with_name(
            f"spine.failed-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        paths.spine_dir.rename(aside)
        runner.log(f"[spine] moved the unfinished attempt to {aside}")
    paths.spine_dir.mkdir(parents=True, exist_ok=True)
    runner.run(
        "spine",
        spine_argv(paths, python=runner.python, staging_local_only=staging_local_only),
    )
    done, detail = spine_done(paths)
    if not done:
        raise CutError(f"the spine finished but is not usable: {detail}")


def run_national(
    runner: Runner, *, retry: bool, logbook_prev_row_digest: str | None
) -> None:
    paths = runner.paths
    done, detail = national_done(paths)
    if done:
        runner.log(f"[national] done: {detail}")
        return
    spine = spine_sha256(paths)
    if spine is None:
        raise CutError("the national step needs a finished spine.")
    if paths.run_dir.exists():
        if not retry:
            raise CutError(
                f"{paths.run_dir} holds an unfinished national run ({detail}); "
                "pass --retry-national to move it aside and start a fresh one."
            )
        aside = paths.run_dir.with_name(
            f"national.failed-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        )
        paths.run_dir.rename(aside)
        runner.log(f"[national] moved the unfinished run to {aside}")
    common = dict(
        python=runner.python,
        spine_sha256=spine,
        sources=runner.sources,
        logbook_prev_row_digest=logbook_prev_row_digest,
    )
    dry_run = runner.state()["steps"].get("national-dry-run", {})
    if dry_run.get("status") != "done":
        runner.run("national-dry-run", national_argv(paths, dry_run=True, **common))
    runner.run("national", national_argv(paths, dry_run=False, **common))
    done, detail = national_done(paths)
    if not done:
        raise CutError(f"the national run finished but is not usable: {detail}")


def run_score(runner: Runner) -> None:
    paths = runner.paths
    done, detail = score_done(paths)
    if done:
        runner.log(f"[score] done: {detail}")
        return
    runner.log(f"[score] re-scoring the candidate ({detail})")
    _set_aside(runner, "score", paths.run_dir / "score_vs_incumbent.json")
    runner.run(
        "score",
        score_argv(
            paths,
            python=runner.python,
            candidate_sha256=candidate_sha256(paths),
            sources=runner.sources,
        ),
    )
    done, detail = score_done(paths)
    if not done:
        raise CutError(f"the candidate does not pass the incumbent score: {detail}")


def national_logbook_row(paths: CutPaths) -> str:
    """The digest of the national build's own Logbook row.

    The build spools exactly one row under ``logbook-spool/``, named by its
    digest; the certification's row chains on it, as the hand-run launchers
    chain it, so the two never share a predecessor.
    """

    rows = sorted((paths.run_dir / "logbook-spool").glob("*.json"))
    if len(rows) != 1:
        raise CutError(
            f"{paths.run_dir / 'logbook-spool'} holds {len(rows)} Logbook rows; "
            "certification chains on the national build's single row."
        )
    return rows[0].stem


def run_certify(runner: Runner) -> None:
    paths = runner.paths
    done, detail = certify_done(paths)
    if done:
        runner.log(f"[certify] done: {detail}")
        return
    _set_aside(
        runner,
        "certify",
        paths.run_dir / f"{CANDIDATE_NAME}.release_cut_gates.json",
        paths.run_dir / f"{CANDIDATE_NAME}.release_certification.json",
    )
    runner.run(
        "certify",
        certify_argv(
            paths,
            python=runner.python,
            candidate_sha256=candidate_sha256(paths),
            spine_sha256=spine_sha256(paths),
            sources=runner.sources,
            logbook_prev_row_digest=national_logbook_row(paths),
        ),
    )
    done, detail = certify_done(paths)
    if not done:
        raise CutError(f"the release cut is not certified: {detail}")


def stage1_done(paths: CutPaths, previous_h5: Path) -> tuple[bool, str]:
    """The comparison exists and measured exactly these two files."""

    report = _read_json(paths.diagnostics / "stage1_comparison.json")
    if report is None:
        return False, "no stage-1 comparison"
    measured = report.get("artifacts") or {}
    expected = {
        "candidate": candidate_sha256(paths),
        "previous": _sha256_file(previous_h5),
    }
    for label, sha256 in expected.items():
        if (measured.get(label) or {}).get("sha256") != sha256:
            return False, f"the comparison measured another {label} file"
    return True, "stage-1 comparison current"


def run_stage1(runner: Runner, previous_h5: Path | None) -> None:
    paths = runner.paths
    output = paths.diagnostics / "stage1_comparison.json"
    if previous_h5 is not None:
        done, detail = stage1_done(paths, previous_h5)
        if done:
            runner.log(f"[stage1] done: {detail}")
            return
        _set_aside(runner, "stage1", output)
    if previous_h5 is None:
        runner.record("stage1", status="skipped", reason="no --previous-h5")
        runner.log(
            "[stage1] skipped: pass --previous-h5 <promoted cut's H5> for the "
            "interim stage-1 comparison (diagnostic; microcosm#1276)"
        )
        return
    paths.diagnostics.mkdir(parents=True, exist_ok=True)
    runner.run(
        "stage1",
        [
            runner.python,
            str(REPO_ROOT / "tools/measure_uk_weighted_integrity_baselines.py"),
            "--h5", f"candidate={paths.candidate_h5}",
            "--h5", f"previous={previous_h5}",
            "--output", str(output),
        ],
    )  # fmt: skip


def run_assemble(runner: Runner) -> dict[str, Any]:
    paths = runner.paths
    done, detail = assemble_done(paths)
    if done:
        runner.log(f"[assemble] done: {detail}")
        return _read_json(paths.cut / "assemble_summary.json") or {}
    if paths.release_dir.exists():
        raise CutError(
            f"{paths.release_dir} exists without an assembly summary; inspect it. "
            "Assembly refuses an existing release directory."
        )
    paths.release_parent.mkdir(parents=True, exist_ok=True)
    runner.run(
        "assemble", assemble_argv(paths, python=runner.python), cwd=paths.release_parent
    )
    summary = _last_json_object((paths.logs / "assemble.log").read_text())
    tag = summary.get("tag") or summary.get("cut_tag") or summary.get("tag_name")
    if not tag:
        raise CutError("assembly printed no cut tag; see logs/assemble.log")
    summary["tag"] = tag
    (paths.cut / "assemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def _last_json_object(text: str) -> dict[str, Any]:
    """The last top-level JSON object a tool printed (its summary)."""

    decoder = json.JSONDecoder()
    found: dict[str, Any] = {}
    index = 0
    while (start := text.find("{", index)) != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            found = value
        index = end
    return found


def tag_published(tag: str) -> bool:
    from huggingface_hub import HfApi

    refs = HfApi().list_repo_refs(PUBLISH_REPO, repo_type="dataset")
    return any(ref.name == tag for ref in refs.tags)


def run_publish(runner: Runner, tag: str, *, publish: bool) -> None:
    paths = runner.paths
    runner.run(
        "publish-preflight",
        publish_argv(paths, python=runner.python, tag=tag, preflight_only=True),
        cwd=paths.release_parent,
    )
    if not publish:
        runner.log(
            "[publish] not published: pass --publish to publish this cut for "
            "inspection (tag only; main and latest-*.json do not move)"
        )
        return
    if tag_published(tag):
        runner.log(f"[publish] done: tag {tag} exists")
        return
    runner.run(
        "publish",
        publish_argv(paths, python=runner.python, tag=tag, preflight_only=False),
        cwd=paths.release_parent,
    )


def write_receipt(runner: Runner, extra: Mapping[str, Any]) -> None:
    paths = runner.paths
    receipt = {
        "schema_version": 1,
        "kind": "microcosm.uk_cut_receipt",
        "release_id": RELEASE_ID,
        "label": paths.label,
        "code": {"repository": "PolicyEngine/microcosm", "commit": paths.commit},
        "engine": _engine_version(),
        # The key that signed the cut's reports, whoever resumed it.
        "signing_key_sha256": cut_signing_key(paths),
        "inputs": {
            source.key: {
                "repo_id": source.repo_id,
                "revision": source.revision,
                "archive": source.filename,
                "url": source.url,
                "files": {pin.name: pin.sha256 for pin in source.files},
            }
            for source in runner.sources.values()
        },
        "spine_sha256": spine_sha256(paths),
        "candidate_sha256": candidate_sha256(paths)
        if (paths.run_dir / "build_record.json").exists()
        else None,
        "steps": runner.state()["steps"],
        **extra,
    }
    paths.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))


def _engine_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("policyengine-uk")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Command line                                                                  #
# --------------------------------------------------------------------------- #


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_uk_cut.py", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "command",
        choices=("status", "fetch", "preflight", "all", "inputs"),
        help=(
            "status: where each step stands; fetch: install and verify every "
            "input; preflight: every check, nothing built; all: fetch, "
            "preflight, then each step not already done; inputs: print the "
            "resolved input pins"
        ),
    )
    parser.add_argument("--label", help="names the cut's directory, e.g. sat24")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    parser.add_argument(
        "--reservation-tool",
        type=Path,
        help=(
            "a host reservation tool to ask before building (run as "
            "`<tool> allowed <caller>`; exit 0 means allowed); off by default"
        ),
    )
    parser.add_argument(
        "--reservation-caller",
        help="the caller name the reservation tool is asked about",
    )
    parser.add_argument(
        "--allow-off-main",
        metavar="REASON",
        help="build a pushed commit that is not on origin/main (say who agreed)",
    )
    parser.add_argument("--only", action="append", help="fetch only these sources")
    parser.add_argument("--reverify", action="store_true", help="re-hash every input")
    parser.add_argument("--skip-surface-check", action="store_true")
    parser.add_argument(
        "--retry-spine",
        action="store_true",
        help="move an unfinished spine attempt aside and build a fresh one",
    )
    parser.add_argument("--retry-national", action="store_true")
    parser.add_argument("--previous-h5", type=Path, help="for the stage-1 comparison")
    parser.add_argument(
        "--logbook-prev-row-digest",
        help=(
            "the national build's Logbook predecessor (default: a new chain); "
            "the certification always chains on the build's own row"
        ),
    )
    parser.add_argument(
        "--spine-telemetry",
        action="store_true",
        help="send the spine's telemetry to the staging repo (default: local only)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish for inspection (--no-latest --tag-only); outward-facing",
    )
    args = parser.parse_args(argv)
    if (args.reservation_tool is None) != (args.reservation_caller is None):
        parser.error("--reservation-tool and --reservation-caller go together.")
    # Steps run from different working directories (assembly and publication
    # from the release directory), so every path must be absolute.
    args.work_dir = args.work_dir.expanduser().resolve()
    if args.reservation_tool is not None:
        args.reservation_tool = args.reservation_tool.expanduser().resolve()
    if args.previous_h5 is not None:
        args.previous_h5 = args.previous_h5.expanduser().resolve()
    return args


def _cut_paths(args: argparse.Namespace) -> CutPaths:
    if not args.label:
        raise CutError("--label is required for this command.")
    return CutPaths(args.work_dir, args.label, _git("rev-parse", "HEAD"))


def _print_checks(checks: Sequence[Check]) -> bool:
    ok = True
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.blocking else "warn")
        print(f"  {mark} {check.name}: {check.detail}")
        ok = ok and (check.ok or not check.blocking)
    return ok


def _print_fetch(results: Sequence[FetchResult]) -> bool:
    ok = True
    for result in results:
        good = result.status in {"present", "fetched"}
        ok = ok and good
        print(
            f"  {'ok  ' if good else 'FAIL'} {result.key}: {result.status} {result.detail}"
        )
    return ok


def preflight(args: argparse.Namespace, paths: CutPaths) -> list[Check]:
    checks = git_checks(allow_off_main=args.allow_off_main)
    checks += environment_checks()
    checks.append(signing_key_check(paths=paths))
    checks += hub_checks()
    checks.append(register_expiry_check(date.today()))
    checks += host_checks(
        paths.work_dir,
        min_free_gb=args.min_free_gb,
        reservation_tool=args.reservation_tool,
        reservation_caller=args.reservation_caller,
    )
    checks.append(contract_pin_check(sys.executable))
    if not args.skip_surface_check:
        chronicle = paths.inputs / "chronicle_uk"
        if (chronicle / "consumer_facts.jsonl").exists():
            checks.append(surfaces_check(sys.executable, chronicle))
        else:
            checks.append(
                Check("target surfaces", False, "fetch the Chronicle artifact first")
            )
    return checks


def status(paths: CutPaths) -> None:
    print(f"cut {paths.cut}")
    for name, check in (
        ("spine", spine_done),
        ("national", national_done),
        ("score", score_done),
        ("certify", certify_done),
        ("assemble", assemble_done),
    ):
        try:
            done, detail = check(paths)
        except Exception as error:  # an earlier step is missing
            done, detail = False, f"not reached ({type(error).__name__})"
        print(f"  {'done' if done else 'todo'} {name}: {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        sources = {source.key: source for source in load_input_sources()}
        if args.command == "inputs":
            print(
                json.dumps(
                    {
                        key: {
                            "kind": s.kind,
                            "repo_id": s.repo_id,
                            "revision": s.revision,
                            "files": {p.name: p.sha256 for p in s.files},
                        }
                        for key, s in sources.items()
                    },
                    indent=2,
                )
            )
            return 0
        work = args.work_dir
        if args.command in {"fetch", "all"}:
            print("inputs:")
            results = fetch_inputs(
                list(sources.values()),
                work / "inputs",
                downloader=HubDownloader(work / "cache" / "hf"),
                reverify=args.reverify,
                only=set(args.only) if args.only else None,
            )
            if not _print_fetch(results):
                print("Some inputs are not available; see docs/uk-cut-anyone.md.")
                return 1
            if args.command == "fetch":
                return 0
        paths = _cut_paths(args)
        if args.command == "status":
            status(paths)
            return 0
        print("preflight:")
        if not _print_checks(preflight(args, paths)):
            print("Preflight failed; nothing was built.")
            return 1
        if args.command == "preflight":
            return 0
        runner = Runner(paths=paths, sources=sources)
        run_spine(
            runner, staging_local_only=not args.spine_telemetry, retry=args.retry_spine
        )
        run_national(
            runner,
            retry=args.retry_national,
            logbook_prev_row_digest=args.logbook_prev_row_digest,
        )
        run_score(runner)
        run_certify(runner)
        run_stage1(runner, args.previous_h5)
        summary = run_assemble(runner)
        tag = summary["tag"]
        run_publish(runner, tag, publish=args.publish)
        write_receipt(runner, {"tag": tag, "published": args.publish})
        print(f"\ncut {tag} assembled; receipt {paths.receipt}")
        print(
            "Next (each a separate, deliberate call):\n"
            f"  promote: {promote_command(paths, tag)}\n"
            "  certify in policyengine.py: tools/certify_uk_cut_in_policyengine_py.py "
            f"--national-tag {tag} ..."
        )
        return 0
    except CutError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
