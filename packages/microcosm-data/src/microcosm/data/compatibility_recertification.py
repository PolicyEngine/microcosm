"""Re-certify a published release for a newer country model, bytes unchanged.

A ``compatibility_recertification`` release is a child of one published,
certified US calibration release. It republishes the parent's microdata and
every other parent artifact byte for byte under a new id, and adds what the
parent cannot say for itself: that the same bytes were loaded and scored under
a newer policyengine-us, what that engine reads and ignores differently, and
how far the headline results moved. It transforms no data and never
recalibrates.

**Parent identity comes from the parent's own published manifest.** The child
carries the parent's ``release_manifest.json`` and ``build_manifest.json``
byte for byte (``parent_release_manifest.json``, ``parent_build_manifest.json``)
and records their SHA-256 in ``compatibility_recertification.json``. That
manifest pins every parent artifact's digest, so it is the whole lineage; no
parent is a code constant. Validation is offline: it checks the child against
the carried manifest and replays the current release contract on the parent
reconstructed from the carried files. Publication adds the online anchor
(:func:`require_parent_publication`): before any upload, the parent tag on the
Hub must serve exactly the carried manifest bytes.

**What is measured, and how it is bound.**

- *Loader replay* (:func:`run_recertification_loader_replay`). Under the
  tested country, Core and calculator wheels, the country loader reads the H5
  (the wrapper's loader is the downstream policyengine.py certification's to
  prove; :data:`WRAPPER_LOADER_STATUS`). Every stored column is classified against the tested engine and the
  stored-input register (:mod:`microcosm.data.stored_inputs`). A column the
  built-with engine read and the tested engine ignores, or one the register
  says no engine reads and the tested engine now reads, must have an entry in
  :data:`RECERTIFICATION_COLUMN_CHANGES`, the reviewed register of such
  changes, or the replay refuses. The receipt is replayed at every later gate.
- *Drift* (:func:`combine_drift`). The publisher's tool scores the H5 once
  under the built-with engine and once under the tested engine (and once more
  with the newly read inputs withheld), with the same household-batched scorer
  and the same metric definitions (:func:`drift_metric_definitions`): the 18
  exact US critical rows, every scorable national SOI Table 1.4 dollar row and
  the headline aggregates in :data:`HEADLINE_METRICS`. The drift document
  embeds all score documents, so every row recomputes from them; the tested
  side's runtime must be the exact runtime the loader replay certified.
- *Claim*. ``compatible_model_packages`` keeps every parent entry and gains one
  ``publisher_claim`` entry for the tested version, attributed to a named
  declarer. ``compatibility.claim_measurement`` records the one version
  measured, the baseline it was compared with and the drift summary. Whether
  the drift is acceptable is the declarer's call, not this module's.

**Scope.** National-default parents only: :data:`SUPPORTED_PARENT_ROLES`. The
identity mechanism does not depend on the role, so a local-area parent is an
additive follow-up. A parent must itself be a calibration release; a
recertification of a recertification or of a source enrichment is refused, so
every child is one hop from the build its calibration evidence describes.

See ``docs/us-compatibility-recertification.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from microcosm.data.contract import (
    PUBLISHER_CLAIM_BASIS,
    ReleaseContractError,
    required_release_files,
)

RECERTIFICATION_RELEASE_TYPE = "compatibility_recertification"
REPORT_FILE = "compatibility_recertification.json"
RECEIPT_FILE = "compatibility_recertification_loader.json"
DRIFT_FILE = "compatibility_drift.json"
PARENT_RELEASE_MANIFEST_FILE = "parent_release_manifest.json"
PARENT_BUILD_MANIFEST_FILE = "parent_build_manifest.json"
REPORT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
SCORES_SCHEMA_VERSION = 1
DRIFT_SCHEMA_VERSION = 1
SCORES_KIND = "compatibility_recertification_scores"
DRIFT_KIND = "compatibility_drift"
REPO_ID = "policyengine/populace-us"
US_RELEASE_PREFIX = "populace-us-"
NATIONAL_DEFAULT_ROLE = "national_default"
#: Roles a parent may declare. The identity checks do not depend on the role;
#: the local-area contract's own replay is what a second role needs.
SUPPORTED_PARENT_ROLES = (NATIONAL_DEFAULT_ROLE,)
#: The parent's ``release_type``: absent or ``calibration``, never a child.
PARENT_RELEASE_TYPES = (None, "calibration")
#: Files the child writes itself; a parent release file by one of these names
#: is refused rather than shadowed.
CHILD_OWNED_FILES = (
    "release_manifest.json",
    "build_manifest.json",
    REPORT_FILE,
    RECEIPT_FILE,
    DRIFT_FILE,
    PARENT_RELEASE_MANIFEST_FILE,
    PARENT_BUILD_MANIFEST_FILE,
)
#: The child's own artifacts, by manifest key, besides the carried ones.
CHILD_ARTIFACT_KEYS = {
    "build_manifest": "build_manifest.json",
    "compatibility_recertification": REPORT_FILE,
    "parent_release_manifest": PARENT_RELEASE_MANIFEST_FILE,
    "parent_build_manifest": PARENT_BUILD_MANIFEST_FILE,
}
CERTIFIED_ARTIFACT_KEYS = {
    "compatibility_recertification_loader": RECEIPT_FILE,
    "compatibility_drift": DRIFT_FILE,
}
#: The runtime a recertification proves: the country model, its Core and the
#: calculator that registers the native SPM role, by import name.
IMPORT_NAMES = MappingProxyType(
    {
        "policyengine-us": "policyengine_us",
        "policyengine-core": "policyengine_core",
        "spm-calculator": "spm_calculator",
    }
)
COMPATIBILITY_PACKAGES = tuple(IMPORT_NAMES)
#: The wrapper is not part of it. policyengine.py imports its bundled country
#: model with its US package (``import policyengine`` does so unless
#: ``POLICYENGINE_SKIP_COUNTRY_IMPORTS`` is set, and the US dataset loader
#: always does) and certifies the bundle's data release against the installed
#: country version there: policyengine 6.10.1 under
#: policyengine-us 2.60.4 raises "Data release manifest is not certified for
#: the runtime model version 2.60.4" before any loader exists. A wrapper that
#: loads this child is one whose bundle certifies this child, which is the
#: downstream policyengine.py certification step; that step runs the
#: wrapper's own suite on the pinned pair. The receipt records this.
WRAPPER_LOADER_STATUS = {
    "status": "not_attested",
    "package": "policyengine",
    "reason": (
        "policyengine.py certifies its bundled data release against the "
        "installed country model when its US package is imported, so no "
        "released wrapper loads US data under a country version its bundle "
        "does not certify; the wrapper loader is proven when a policyengine.py "
        "bundle certifies this child"
    ),
}
#: The producer inventory the release assembler, scorer and certifier run
#: from. Publication authenticates every file against the recorded commit and
#: the checkout (``source_enrichment._check_producer_source_identity``).
PRODUCER_SOURCE_FILES = (
    "tools/build_us_compatibility_recertification.py",
    "packages/microcosm-data/src/microcosm/data/compatibility_recertification.py",
    "packages/microcosm-data/src/microcosm/data/source_enrichment.py",
    "packages/microcosm-data/src/microcosm/data/stored_inputs.py",
    "packages/microcosm-data/src/microcosm/data/contract.py",
)
SCORING_PERIOD = 2024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")


# ---------------------------------------------------------------------------
# The reviewed register of what a newer engine reads and ignores differently.
# ---------------------------------------------------------------------------

#: The column the built-with engine read is no longer a variable.
CHANGE_IGNORED = "ignored"
#: The register says no engine reads the column; the tested engine now does.
CHANGE_READ = "read"


@dataclass(frozen=True)
class ReviewedColumnChange:
    """One reviewed change in how an engine treats a stored column."""

    change: str
    reason: str


_REMOVED_MORTGAGE_INPUT = (
    "A tax-unit float input of policyengine-us 2.2.1, defined in "
    "variables/household/expense/tax_unit/mortgage_interest_structure.py. "
    "PolicyEngine/policyengine-us#9605 (merged 2026-10-12) removed it; "
    "policyengine-us 2.60.4 defines no variable of this name, and its own "
    "tests/microsimulation/data/test_removed_mortgage_interest_inputs.py "
    "states that a dataset carrying it loads with core's unknown-column "
    "warning and computes mortgage interest from the person-level "
    "home_mortgage_interest input alone. Releases store that input too, so "
    "the interest is not lost; the {which} split of it is."
)

#: Reviewed changes, keyed by stored column. A recertification refuses any
#: change the tested engine makes that has no entry here, and any entry whose
#: direction disagrees with what the engine does. An entry is evidence about
#: the engines named in its reason; adding one is a reviewed change.
RECERTIFICATION_COLUMN_CHANGES: Mapping[str, ReviewedColumnChange] = MappingProxyType(
    {
        "first_home_mortgage_interest": ReviewedColumnChange(
            CHANGE_IGNORED, _REMOVED_MORTGAGE_INPUT.format(which="first-lien")
        ),
        "second_home_mortgage_interest": ReviewedColumnChange(
            CHANGE_IGNORED, _REMOVED_MORTGAGE_INPUT.format(which="second-lien")
        ),
        "tax_unit_role_input": ReviewedColumnChange(
            CHANGE_READ,
            "The HEAD / SPOUSE / DEPENDENT role Microcosm's tax-unit "
            "construction assigns each person (the stored-input register's "
            "entry). policyengine-us 2.2.1 defines no variable of this name and "
            "infers heads and spouses by age. PolicyEngine/policyengine-us#9617 "
            '("Honor dataset-supplied tax-unit roles") made it a person Enum '
            "input: in 2.60.4, tax_unit_roles_supplied is true when every member "
            "of a tax unit has a role, and is_tax_unit_head and "
            "is_tax_unit_spouse then follow the stored role instead of age. It "
            "raises on a unit with some members' roles missing, or without "
            "exactly one HEAD and at most one SPOUSE.",
        ),
        **{
            column: ReviewedColumnChange(
                CHANGE_READ,
                f"The person_id of the co-resident parent the CPS ASEC {pointer} "
                "pointer names, or 0 (the stored-input register's entry). "
                "policyengine-us 2.2.1 defines no variable of this name. "
                "PolicyEngine/policyengine-us#9406 (merged 2026-10-07) made it a "
                "person int input; in 2.60.4 Medicaid's MAGI household "
                "composition (variables/gov/hhs/medicaid/income/"
                "_non_filer_household.py and _claiming_tax_unit.py) and the "
                "parent-link helpers read it.",
            )
            for column, pointer in (
                ("parent_1_id", "PEPAR1"),
                ("parent_2_id", "PEPAR2"),
            )
        },
    }
)


def column_change_register_sha256(
    register: Mapping[str, ReviewedColumnChange] = RECERTIFICATION_COLUMN_CHANGES,
) -> str:
    """SHA-256 of the register as canonical JSON, recorded in each receipt."""

    payload = json.dumps(
        {column: asdict(entry) for column, entry in register.items()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def stored_column_changes(
    stored_tables: Mapping[str, Iterable[str]],
    *,
    engine_variables: Iterable[str],
    stored_input_register: Mapping[str, str],
) -> dict[str, list[dict]]:
    """How the tested engine's reading of the stored columns differs.

    Pure: the stored tables, the tested engine's variable names and the
    stored-input register in, two sorted lists out.

    - ``ignored``: model-named columns that are neither tested-engine variables
      nor register entries. The stored-input rule refuses exactly these; a
      parent that passed that rule under its built-with engine stored them as
      inputs that engine read.
    - ``read``: register entries (columns the register says the engine
      ignores) that the tested engine defines, so it now reads them.

    Each item names the column and every table that stores it.
    """

    from microcosm.data.stored_inputs import is_model_named

    variables = frozenset(engine_variables)
    tables_by_column: dict[str, set[str]] = {}
    for table, columns in stored_tables.items():
        for column in columns:
            tables_by_column.setdefault(column, set()).add(str(table))
    ignored, read = [], []
    for column in sorted(tables_by_column):
        if not is_model_named(column):
            continue
        item = {"column": column, "tables": sorted(tables_by_column[column])}
        if column not in variables and column not in stored_input_register:
            ignored.append(item)
        elif column in variables and column in stored_input_register:
            read.append(item)
    return {CHANGE_IGNORED: ignored, CHANGE_READ: read}


def reviewed_column_change_failures(
    changes: Mapping[str, Sequence[Mapping]],
    register: Mapping[str, ReviewedColumnChange] = RECERTIFICATION_COLUMN_CHANGES,
) -> list[str]:
    """One failure per change without a matching reviewed entry; [] if all are."""

    failures = []
    for direction in (CHANGE_IGNORED, CHANGE_READ):
        for item in changes.get(direction, ()):
            column = item["column"]
            entry = register.get(column)
            if entry is None:
                failures.append(
                    f"stored column {column!r} ({', '.join(item['tables'])}) is "
                    f"{'ignored' if direction == CHANGE_IGNORED else 'read'} by "
                    "the tested engine differently from the stored-input "
                    "register, and RECERTIFICATION_COLUMN_CHANGES has no "
                    "reviewed entry for it. Review what the engine change does "
                    "to the column and add the entry with its reason."
                )
            elif entry.change != direction:
                failures.append(
                    f"stored column {column!r}: the reviewed register says the "
                    f"engine change makes it {entry.change!r}, but the tested "
                    f"engine has it {direction!r}."
                )
    return failures


def recorded_column_changes(
    changes: Mapping[str, Sequence[Mapping]],
    register: Mapping[str, ReviewedColumnChange] = RECERTIFICATION_COLUMN_CHANGES,
) -> dict[str, list[dict]]:
    """The changes with each reviewed reason attached, as the report records them."""

    return {
        direction: [
            {**dict(item), "reason": register[item["column"]].reason}
            for item in changes.get(direction, ())
        ]
        for direction in (CHANGE_IGNORED, CHANGE_READ)
    }


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _json(path: Path, failures: list[str]) -> dict:
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        return payload
    except (OSError, ValueError) as exc:
        failures.append(f"{path.name}: {exc}")
        return {}


def write_json(path: Path, value: object) -> None:
    """Write a release JSON document the way every lane does: sorted, indented."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


#: dist-info files an installer writes, which no wheel carries: they say how a
#: package was installed, not what it is (``uv_cache.json`` is uv's).
_INSTALLER_FILES = frozenset(
    {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
)


def _identity_file(name: str) -> bool:
    """Whether an installed or wheel path belongs to the package's identity.

    Bytecode in ``__pycache__`` and installer bookkeeping are excluded; so are
    paths the installer generates outside the package (console scripts,
    ``../bin``).
    Everything else counts, parameter YAML and data files included, because
    every byte of it can change what the engine computes.
    """

    parts = name.split("/")
    if ".." in parts:
        return False
    # Only bytecode CPython writes into a __pycache__ directory is excluded. A
    # source file there, or a .pyc beside the sources (which an import loads
    # without any source), counts and so must be recorded.
    if len(parts) > 1 and parts[-2] == "__pycache__" and parts[-1].endswith(".pyc"):
        return False
    return not (parts[0].endswith(".dist-info") and parts[-1] in _INSTALLER_FILES)


def _canonical_package(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _package_tree_files(dist, identity_files: Sequence[str]) -> set[str]:
    """Every file on disk under the distribution's top-level packages.

    ``dist.files`` is the installer's record, not the tree: a module dropped
    into an installed package after installation is absent from it yet
    importable, and Core loads every variable module it finds. So the tree is
    listed from disk and must equal the record (bytecode caches aside), and
    a symlink anywhere in it is refused.
    """

    roots = {
        relative.split("/", 1)[0]
        for relative in identity_files
        if "/" in relative and not relative.split("/", 1)[0].endswith(".dist-info")
    }
    import os

    found = set()
    for root in roots:
        base = Path(dist.locate_file(root))
        for directory, subdirectories, filenames in os.walk(base, followlinks=False):
            for name in (*subdirectories, *filenames):
                path = Path(directory) / name
                if path.is_symlink():
                    # Core globs every subdirectory; a link would bring files
                    # from outside the verified tree into it.
                    raise ValueError(f"symlink in the installed package tree: {path}")
            for name in filenames:
                relative = (
                    f"{root}/{(Path(directory) / name).relative_to(base).as_posix()}"
                )
                if _identity_file(relative):
                    found.add(relative)
    return found


def runtime_identities(
    compatibility_wheels: Sequence[Path | str] = (), *, require_wheels: bool
) -> dict:
    """The three packages' installed contents, bound to their wheels.

    For each of :data:`COMPATIBILITY_PACKAGES`:

    - the digest covers every file the distribution installed
      (``_identity_file``), parameters and data as well as Python;
    - the files on disk under its packages must be exactly the recorded ones,
      so an unrecorded module cannot ride along (:func:`_package_tree_files`);
    - the package an import resolves to must be the installed one, so nothing
      earlier on the path (``PYTHONPATH``, a ``.pth`` entry) shadows it.

    With a wheel for the package, the installed inventory must equal the
    wheel's members and every installed file the member's bytes, so
    ``content_sha256`` then identifies the wheel's contents and
    ``wheel_sha256`` the wheel itself. ``require_wheels`` demands all three
    wheels and refuses an editable install. Nothing here depends on where a
    wheel was installed from, so honest replays in two environments agree.

    This proves the files; :func:`require_trusted_bytecode` and
    :func:`verify_loaded_modules` prove that what ran was compiled from them.

    Raises:
        ValueError: A wheel is unknown, duplicated, or differs from the
            installed package; the tree holds unrecorded files; an import
            resolves elsewhere; or ``require_wheels`` and a wheel is missing.
    """

    import importlib.util
    from email.parser import BytesParser
    from importlib import metadata
    from zipfile import ZipFile

    wheels: dict[str, tuple[Path, str, dict[str, bytes]]] = {}
    for raw in compatibility_wheels:
        path = Path(raw)
        with ZipFile(path) as archive:
            members = {
                name: archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/") and _identity_file(name)
            }
        metadata_files = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{path.name} must contain one wheel METADATA file")
        parsed = BytesParser().parsebytes(members[metadata_files[0]])
        name = _canonical_package(parsed["Name"])
        if name not in COMPATIBILITY_PACKAGES or name in wheels:
            raise ValueError(
                "compatibility wheels must uniquely name the country, Core and "
                f"calculator packages ({', '.join(COMPATIBILITY_PACKAGES)})"
            )
        wheels[name] = (path, parsed["Version"], members)
    if require_wheels and set(wheels) != set(COMPATIBILITY_PACKAGES):
        raise ValueError(
            "the runtime must be proven against all three wheels: "
            f"{', '.join(COMPATIBILITY_PACKAGES)}"
        )
    identities = {}
    for name in COMPATIBILITY_PACKAGES:
        dist = metadata.distribution(name)
        direct_url = dist.read_text("direct_url.json")
        if (
            require_wheels
            and direct_url
            and _mapping(_mapping(json.loads(direct_url)).get("dir_info")).get(
                "editable"
            )
        ):
            raise ValueError(f"{name} is an editable install, not a wheel identity")
        files = sorted(
            str(file) for file in dist.files or () if _identity_file(str(file))
        )
        if not files:
            raise ValueError(f"cannot list the installed files of {name}")
        unrecorded = _package_tree_files(dist, files) - set(files)
        if unrecorded:
            raise ValueError(
                f"{name} has files on disk its installation never recorded: "
                f"{sorted(unrecorded)[:5]}"
            )
        spec = importlib.util.find_spec(IMPORT_NAMES[name])
        installed_root = Path(dist.locate_file(IMPORT_NAMES[name])).resolve()
        if (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).resolve().is_relative_to(installed_root)
        ):
            raise ValueError(
                f"import {IMPORT_NAMES[name]} does not resolve to the installed "
                f"{name} at {installed_root}"
            )
        digest = hashlib.sha256()
        for relative in files:
            digest.update(relative.encode() + b"\0")
            digest.update(Path(dist.locate_file(relative)).read_bytes() + b"\0")
        identity = {
            "version": dist.version,
            "files": len(files),
            "content_sha256": digest.hexdigest(),
            "wheel_sha256": None,
        }
        if name in wheels:
            path, version, members = wheels[name]
            if version != dist.version:
                raise ValueError(
                    f"{name} wheel {version} is not the installed {dist.version}"
                )
            if set(members) != set(files):
                raise ValueError(
                    f"{name} installed files differ from its wheel's members"
                )
            for relative in files:
                if Path(dist.locate_file(relative)).read_bytes() != members[relative]:
                    raise ValueError(
                        f"{name} installed {relative} differs from its wheel"
                    )
            identity["wheel_sha256"] = sha256_file(path)
        identities[name] = identity
        _VERIFIED_TREES[name] = (
            Path(dist.locate_file(IMPORT_NAMES[name])).resolve(),
            Path(dist.locate_file("")).resolve(),
            frozenset(files),
        )
    return identities


#: package -> (import root, site directory, verified files), as the last
#: :func:`runtime_identities` call in this process proved them.
_VERIFIED_TREES: dict[str, tuple[Path, Path, frozenset[str]]] = {}


#: The environment variable that points CPython's bytecode cache somewhere
#: else at interpreter start (``sys.pycache_prefix``).
PYCACHE_PREFIX_ENVIRONMENT = "PYTHONPYCACHEPREFIX"


def bootstrap_bytecode_isolation() -> str:
    """Point this process's bytecode cache at a fresh empty directory.

    CPython runs a cached ``.pyc`` whose header matches its source's size and
    timestamp without reading the source, so a replaced cache payload runs
    code the authenticated source does not contain. A trusted entry point
    calls this before it imports any Microcosm module or engine package, so
    every module after it compiles from source. An interpreter started with
    :data:`PYCACHE_PREFIX_ENVIRONMENT` already has a prefix, which is kept.
    Returns the prefix.
    """

    import sys

    if sys.pycache_prefix is None:
        sys.pycache_prefix = tempfile.mkdtemp(prefix="recertification-pycache-")
    return sys.pycache_prefix


def _trusted_module(name: str, origin: object) -> bool:
    """Whether a loaded module's bytecode must come from the isolated cache."""

    roots = {"microcosm", "tools", *IMPORT_NAMES.values()}
    if name.split(".", 1)[0] in roots:
        return True
    if not isinstance(origin, str):
        return False
    resolved = Path(origin).resolve()
    return any(resolved.is_relative_to(root) for root, _, _ in _VERIFIED_TREES.values())


def bytecode_isolation_status() -> dict:
    """Whether every Microcosm and engine module ran from source compiled here.

    Each loaded module of the ``microcosm`` namespace, the ``tools`` package
    and the three engine packages (by name, or by file under a verified
    package tree, which covers the variable modules Core loads under
    synthetic names) must either carry no ``__cached__`` (a script, a
    namespace) or have it under ``sys.pycache_prefix``. A module read from
    install-tree bytecode, or any module when no prefix is set, is reported.
    """

    import sys

    prefix = sys.pycache_prefix
    offending = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        if not _trusted_module(name, origin):
            continue
        if isinstance(origin, str) and origin.endswith(".pyc"):
            offending.append(name)
            continue
        cached = getattr(module, "__cached__", None)
        if cached is None:
            continue
        if prefix is None or not Path(cached).resolve().is_relative_to(
            Path(prefix).resolve()
        ):
            offending.append(name)
    return {
        "isolated": prefix is not None and not offending,
        "offending": sorted(offending)[:10],
    }


def require_trusted_bytecode() -> None:
    """Refuse unless :func:`bytecode_isolation_status` is isolated.

    Raises:
        RuntimeError: Some Microcosm or engine module ran from bytecode the
            process did not compile from its source.
    """

    status = bytecode_isolation_status()
    if not status["isolated"]:
        raise RuntimeError(
            "Microcosm and engine modules must run from source compiled into a "
            "fresh bytecode cache: run through "
            "tools/build_us_compatibility_recertification.py, or start Python "
            f"with {PYCACHE_PREFIX_ENVIRONMENT} set to a new empty directory "
            f"(read from other bytecode: {status['offending']})"
        )


def verify_loaded_modules() -> int:
    """Every loaded module from the engine trees is a verified source file.

    Matched by file, not name, so the variable modules Core loads under
    synthetic names are covered. Each must be a ``.py`` file of the inventory
    :func:`runtime_identities` verified. Returns how many were checked.

    Raises:
        RuntimeError: No inventory was verified in this process, or a module
            came from a file outside it.
    """

    import sys

    if set(_VERIFIED_TREES) != set(COMPATIBILITY_PACKAGES):
        raise RuntimeError("runtime_identities() must verify the packages first")
    checked = 0
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(origin, str):
            continue
        path = Path(origin)
        for package, (root, site, files) in _VERIFIED_TREES.items():
            if not path.resolve().is_relative_to(root):
                continue
            relative = path.resolve().relative_to(site).as_posix()
            if path.suffix != ".py" or relative not in files:
                raise RuntimeError(
                    f"{name} was loaded from {origin}, not a verified {package} source"
                )
            checked += 1
    return checked


def is_recertification(release_dir: Path | str) -> bool:
    """Whether ``release_dir``'s manifest declares this release type.

    A reporting probe for the publisher; it validates nothing, and an
    unreadable manifest reads as ``False``.
    """

    try:
        manifest = json.loads((Path(release_dir) / "release_manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, Mapping)
        and manifest.get("release_type") == RECERTIFICATION_RELEASE_TYPE
    )


# ---------------------------------------------------------------------------
# The parent, from its published manifest.
# ---------------------------------------------------------------------------


def describe_parent(
    parent_release_dir: Path | str, parent_artifact_root: Path | str
) -> dict:
    """The parent block a child records, read from the parent's published files.

    ``parent_release_dir`` holds the parent's ``releases/<id>/`` files as the
    Hub serves them at the parent tag; ``parent_artifact_root`` holds its root
    artifacts (the H5 and the calibration NPZ). Every artifact the manifest
    declares must be present with the digest it pins, in exactly one of the
    two places: a bare filename in the release directory is a release file,
    anything else a root artifact. Offline, that placement rests on where the
    files sit locally; publication authenticates it against the parent tag on
    the Hub (:func:`require_parent_publication`), so a root artifact moved into
    the release directory cannot publish at the wrong path.

    A parent artifact key that the child uses for its own evidence is refused:
    the child's entry would replace it, and the artifact would silently leave
    the child's manifest.

    Raises:
        ValueError: The parent is not a supported published release, or a
            file is missing or differs from its pin.
    """

    parent_release_dir = Path(parent_release_dir)
    parent_artifact_root = Path(parent_artifact_root)
    manifest_path = parent_release_dir / "release_manifest.json"
    build_path = parent_release_dir / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("parent release_manifest.json must be an object")
    parent_id = parent_release_dir.name
    build = _mapping(manifest.get("build"))
    if build.get("build_id") != parent_id:
        raise ValueError(
            "parent release directory must be named for the manifest's build.build_id"
        )
    if not parent_id.startswith(US_RELEASE_PREFIX):
        raise ValueError("only a US release can be recertified by this lane")
    if manifest.get("release_type") not in PARENT_RELEASE_TYPES:
        raise ValueError(
            f"parent release_type {manifest.get('release_type')!r} is not a "
            "calibration release; recertify the calibration release itself"
        )
    role = manifest.get("dataset_role", NATIONAL_DEFAULT_ROLE)
    if role not in SUPPORTED_PARENT_ROLES:
        raise ValueError(
            f"parent dataset_role {role!r} is not supported yet; this lane "
            f"recertifies {SUPPORTED_PARENT_ROLES} releases"
        )
    release_files, root_artifacts = {}, {}
    for key, raw in _mapping(manifest.get("artifacts")).items():
        if key in CHILD_ARTIFACT_KEYS or key in CERTIFIED_ARTIFACT_KEYS:
            raise ValueError(
                f"parent artifact key {key!r} is one the child writes its own "
                "evidence under; the parent artifact would leave the child manifest"
            )
        entry = _mapping(raw)
        path, sha = entry.get("path"), entry.get("sha256")
        if not isinstance(path, str) or not isinstance(sha, str):
            raise ValueError(f"parent artifact {key!r} has no path and sha256")
        # The publisher writes a release file to releases/<id>/<name> and a
        # root artifact to its own path; it writes nothing to another
        # release's directory. A path it would not resolve to one of those two
        # places cannot be republished faithfully, so it is refused.
        if (
            path.startswith("/")
            or "\\" in path
            or posixpath.normpath(path) != path
            or ".." in path.split("/")
            or path.split("/", 1)[0] == "releases"
        ):
            raise ValueError(
                f"parent artifact {key!r} path {path!r} is not a release file or "
                "root artifact path the publisher would write"
            )
        if entry.get("revision") != parent_id or entry.get("repo_id") != REPO_ID:
            raise ValueError(
                f"parent artifact {key!r} must pin {REPO_ID} at the parent tag"
            )
        if Path(path).name == path and (parent_release_dir / path).is_file():
            if path in CHILD_OWNED_FILES:
                raise ValueError(
                    f"parent artifact {path!r} collides with a file the child writes"
                )
            local, place = parent_release_dir / path, release_files
        else:
            local, place = parent_artifact_root / path, root_artifacts
        if path in release_files or path in root_artifacts:
            raise ValueError(f"parent declares artifact path {path!r} twice")
        if not local.is_file() or sha256_file(local) != sha:
            raise ValueError(
                f"parent artifact {path!r} is missing or differs from its pin"
            )
        place[path] = sha
    national = _mapping(manifest.get("default_datasets")).get("national")
    dataset = _mapping(_mapping(manifest.get("artifacts")).get(national))
    if dataset.get("kind") != "microdata" or dataset.get("path") not in root_artifacts:
        raise ValueError(
            "parent default_datasets.national must name a root microdata artifact"
        )
    for name in required_release_files(parent_id):
        if name in {"release_manifest.json", "build_manifest.json"}:
            continue
        if name not in release_files:
            raise ValueError(f"parent manifest must declare its required file {name!r}")
    return {
        "build_id": parent_id,
        "repo_id": REPO_ID,
        "revision": parent_id,
        "release_type": manifest.get("release_type", "calibration"),
        "dataset_role": role,
        "release_manifest_sha256": sha256_file(manifest_path),
        "build_manifest_sha256": sha256_file(build_path),
        "built_with_model_package": dict(
            _mapping(build.get("built_with_model_package"))
        ),
        "built_with_core_package": dict(_mapping(build.get("built_with_core_package"))),
        "release_files": dict(sorted(release_files.items())),
        "root_artifacts": dict(sorted(root_artifacts.items())),
        "dataset": {
            "artifact": national,
            "filename": dataset["path"],
            "sha256": dataset["sha256"],
        },
    }


def _reconstructed_parent(release_dir: Path, parent: Mapping, staging: Path) -> Path:
    """The parent's release directory, rebuilt from the bytes the child carries."""

    directory = staging / str(parent["build_id"])
    directory.mkdir()
    shutil.copyfile(
        release_dir / PARENT_RELEASE_MANIFEST_FILE, directory / "release_manifest.json"
    )
    shutil.copyfile(
        release_dir / PARENT_BUILD_MANIFEST_FILE, directory / "build_manifest.json"
    )
    for name in _mapping(parent.get("release_files")):
        shutil.copyfile(release_dir / name, directory / name)
    return directory


#: The parent-contract outcome when every failure is the missing microcosm#1034
#: stored-input evidence, which the recertification re-measures itself under
#: both engines (the baseline scores' and the loader replay's verdicts).
PARENT_CONTRACT_PASSED = "passed"
_ABSENT = object()
PARENT_CONTRACT_STORED_INPUT_EVIDENCE_REPLACED = (
    "passed_with_stored_input_evidence_replaced"
)


def replay_parent_contract(
    release_dir: Path | str, parent: Mapping, *, artifact_root: Path | str | None
) -> dict:
    """Run today's release contract on the parent rebuilt from the carried bytes.

    A parent published before microcosm#1034 carries no ``stored_inputs.json``
    and fails only that check. The recertification measures the same rule
    itself, against the current register, under the built-with engine (the
    baseline scores) and the tested engine (the loader replay), so such a
    parent passes as :data:`PARENT_CONTRACT_STORED_INPUT_EVIDENCE_REPLACED`,
    with the replaced failure lines recorded.

    The substitution covers absent evidence and nothing else. It applies only
    when the parent carries no stored-input evidence anywhere (no evidence
    file, artifact, ``gate_evidence`` entry or manifest block) and its two
    manifests agree on every other ``gate_evidence`` field; the evidence check
    also compares those blocks whole, so without the second condition an
    unrelated gate disagreement would be excused with the missing evidence.
    Otherwise every failure is returned in ``other_failures`` and refuses the
    child.
    """

    from microcosm.data import contract

    release_dir = Path(release_dir)
    key = contract.US_STORED_INPUTS_EVIDENCE_KEY
    with tempfile.TemporaryDirectory(prefix=".recertification-parent-") as staging:
        parent_dir = _reconstructed_parent(release_dir, parent, Path(staging))
        try:
            contract.validate_release_dir(parent_dir, artifact_root=artifact_root)
            failures: list[str] = []
        except ReleaseContractError as exc:
            failures = list(exc.failures)
        evidence_failures: list[str] = []
        build = json.loads((parent_dir / "build_manifest.json").read_text())
        manifest = json.loads((parent_dir / "release_manifest.json").read_text())
        contract._check_us_stored_input_evidence(
            parent_dir,
            build_manifest=build,
            release_manifest=manifest,
            failures=evidence_failures,
            evidence_tier=False,
        )
        evidence_file_present = (
            parent_dir / contract.US_STORED_INPUTS_EVIDENCE_FILE
        ).exists()
    manifest_build = _mapping(manifest.get("build"))
    raw_blocks = [
        holder.get("gate_evidence", _ABSENT)
        for holder in (_mapping(build), manifest_build)
    ]
    gate_blocks = [_mapping(block) for block in raw_blocks]
    evidence_absent = (
        not evidence_file_present
        and key not in _mapping(manifest.get("artifacts"))
        and all(key not in holder for holder in (_mapping(build), manifest_build))
        and all(key not in block for block in gate_blocks)
    )
    # Compared as written, not as normalised: [] and {} disagree, as the
    # evidence check itself says.
    other_gates_agree = raw_blocks[0] == raw_blocks[1] and all(
        block is _ABSENT or isinstance(block, Mapping) for block in raw_blocks
    )
    replaceable = (
        set(evidence_failures) if evidence_absent and other_gates_agree else set()
    )
    replaced = [line for line in failures if line in replaceable]
    other = [line for line in failures if line not in replaceable]
    return {
        "status": PARENT_CONTRACT_PASSED
        if not failures
        else PARENT_CONTRACT_STORED_INPUT_EVIDENCE_REPLACED,
        "replaced_failures": replaced,
        "other_failures": other,
    }


# ---------------------------------------------------------------------------
# The loader replay under the tested wheels.
# ---------------------------------------------------------------------------


def _loader_values(frame, column):
    import numpy as np

    values = frame[column].to_numpy()
    if values.dtype == object:
        return np.asarray([str(value) for value in values], dtype=object)
    return values


def _h5_values(table, column):
    import numpy as np

    values = table[column]
    # PyTables stores a pandas bool column as a one-byte bitfield, which h5py
    # reads as uint8, and records the logical type in ``<column>_dtype``.
    if table.attrs.get(f"{column}_dtype") in (b"bool", "bool"):
        return np.asarray(values).astype(bool)
    if values.dtype.kind == "S":
        return np.asarray([value.decode() for value in values], dtype=object)
    return values


def _check_source_ownership(
    source_paths: Mapping[str, str | None], packages: Mapping[str, str]
) -> None:
    """Each named source file lies in the installed tree of its package."""

    from importlib import metadata

    for label, package in packages.items():
        root = Path(
            metadata.distribution(package).locate_file(IMPORT_NAMES[package])
        ).resolve()
        source = source_paths.get(label)
        if source is None or not Path(source).resolve().is_relative_to(root):
            raise ValueError(
                f"tested {label} source does not come from the installed {package}"
            )


def run_recertification_loader_replay(
    candidate_h5: Path | str,
    *,
    require_wheels: bool = False,
    compatibility_wheels: tuple[Path | str, ...] = (),
) -> dict:
    """Load the H5 under the installed engine and say what it reads differently.

    The receipt this returns is replayed at every later gate, so it holds only
    what is deterministic for the H5 and the installed runtime:

    1. The three packages' installed contents, parameters included, bound to
       the three wheels when ``require_wheels`` (:func:`runtime_identities`),
       run from source compiled into this process's isolated bytecode cache
       (:func:`require_trusted_bytecode`, which the caller's trusted entry
       point set up with :func:`bootstrap_bytecode_isolation` or
       ``PYTHONPYCACHEPREFIX``) and, after the run, every module loaded from
       the engine trees a verified source file (:func:`verify_loaded_modules`).
    2. The stored-input classification (:func:`stored_column_changes`) and
       the register digests. An unreviewed change refuses the replay.
    3. The country loader (``USSingleYearDataset``) returns every stored
       column the tested engine defines with the H5's exact values, and every
       id, membership and weight column too. The wrapper loader is recorded as
       not attested (:data:`WRAPPER_LOADER_STATUS`).
    4. When the H5 stores the native SPM role, its source comes from the
       calculator wheel and a complete household's supplied role reaches Core
       unchanged (``source_enrichment._check_native_input_precedence``).

    It runs no population simulation. Scoring the population is the drift
    step, outside the receipt.
    """

    require_trusted_bytecode()

    import inspect

    import h5py
    import numpy as np
    from policyengine_us.data import USSingleYearDataset
    from policyengine_us.system import system

    from microcosm.data import source_enrichment, stored_inputs

    candidate_h5 = Path(candidate_h5)
    packages = runtime_identities(compatibility_wheels, require_wheels=require_wheels)
    tables = stored_inputs.h5_stored_tables(candidate_h5)
    engine = stored_inputs.installed_us_engine()
    changes = stored_column_changes(
        tables,
        engine_variables=engine.variables,
        stored_input_register=stored_inputs.US_STORED_NON_VARIABLE_COLUMNS,
    )
    failures = reviewed_column_change_failures(changes)
    if failures:
        raise ValueError(" ".join(failures))
    role = source_enrichment.ROLE_VARIABLE
    stores_role = role in tables.get("person", ())
    source_paths = {"country_loader": inspect.getsourcefile(USSingleYearDataset)}
    loaded_source_packages = {"country_loader": "policyengine-us"}
    if stores_role:
        variable = system.variables.get(role)
        if (
            variable is None
            or variable.entity.key != "person"
            or (variable.value_type is not bool)
        ):
            raise ValueError(f"tested country must register {role} as a person bool")
        source_paths["native_role"] = inspect.getsourcefile(type(variable))
        loaded_source_packages["native_role"] = "spm-calculator"
    _check_source_ownership(source_paths, loaded_source_packages)
    country = USSingleYearDataset(file_path=str(candidate_h5))
    checked = 0
    with h5py.File(candidate_h5, "r") as h5:
        for entity in _ENTITIES:
            table = h5[f"{entity}/table"]
            names = [name for name in table.dtype.names if name != "index"]
            structural = {
                name
                for name in names
                if name in {f"{entity}_id", f"{entity}_weight"}
                or (
                    entity == "person"
                    and name.startswith("person_")
                    and name.endswith("_id")
                )
            }
            for column in names:
                if column not in engine.variables and column not in structural:
                    continue
                expected = _h5_values(table, column)
                for loader, frame in (("country", getattr(country, entity)),):
                    if column not in frame.columns:
                        raise ValueError(
                            f"{loader} native loader dropped {entity}.{column}"
                        )
                    observed = _loader_values(frame, column)
                    same = observed.shape == expected.shape and (
                        np.array_equal(observed, expected)
                        if expected.dtype == object
                        else observed.dtype == expected.dtype
                        and observed.tobytes() == expected.tobytes()
                    )
                    if not same:
                        raise ValueError(
                            f"{loader} native loader changed {entity}.{column}"
                        )
                checked += 1
    checks = [
        "country:every_stored_change_reviewed",
        "country:every_engine_variable_column_loaded_unchanged",
    ]
    if stores_role:
        source_enrichment._check_native_input_precedence(
            country, native_inputs={role: "person"}
        )
        checks.append("country:complete_household_native_role_overrides_default")
    verify_loaded_modules()
    require_trusted_bytecode()
    checks.append("runtime:every_loaded_module_compiled_from_the_verified_tree")
    columns = {column for names in tables.values() for column in names}
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "scope": "native_input_loading_only",
        "external_package_publication": "not_attested",
        "dataset_sha256": sha256_file(candidate_h5),
        "runner_sha256": {
            "compatibility_recertification": sha256_file(__file__),
            "source_enrichment": sha256_file(source_enrichment.__file__),
        },
        "packages": packages,
        "wrapper_loader": dict(WRAPPER_LOADER_STATUS),
        "loaded_source_packages": loaded_source_packages,
        "loaded_source_sha256": {
            key: sha256_file(path) for key, path in source_paths.items()
        },
        "stored_inputs": {
            "engine": engine.label,
            "register_sha256": stored_inputs.register_sha256(),
            "column_change_register_sha256": column_change_register_sha256(),
            "registered_non_variables": sorted(
                column
                for column in columns
                if stored_inputs.is_model_named(column)
                and column not in engine.variables
                and column in stored_inputs.US_STORED_NON_VARIABLE_COLUMNS
            ),
            "column_changes": changes,
        },
        "columns_checked": checked,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Drift: what is scored, and how two engines' scores combine.
# ---------------------------------------------------------------------------

#: The baseline period and the SPM selection every score document records.
#: The selection must equal the parent's post-export scoring selection.
STATISTIC_SUM = "sum"
#: SOI amount rows: the positive part of the signed value (the release tool's
#: ``_signed_component``), so a refund-producing negative tax is not netted.
STATISTIC_POSITIVE_SUM = "positive_sum"
#: SOI return rows: units whose positive part is positive.
STATISTIC_POSITIVE_COUNT = "positive_count"
STATISTICS = (STATISTIC_SUM, STATISTIC_POSITIVE_SUM, STATISTIC_POSITIVE_COUNT)
#: Household batches (every measure is additive over households) or batches
#: of whole states (for the Medicaid state sums below).
BATCHING_HOUSEHOLD = "household"
BATCHING_STATE = "state"
#: Formulas that aggregate over the simulation's whole population. Copied from
#: ``tools/build_us_fiscal_refresh_release.py``'s
#: ``US_POPULATION_AGGREGATE_VARIABLES`` (a build test holds them equal). A
#: household-batch engine refuses to have computed any of them.
POPULATION_AGGREGATE_VARIABLES = (
    "household_income_decile",
    "medicaid_slcsp_state_average_cost_index",
    "medicaid_slcsp_state_denominator",
    "spm_unit_income_decile",
)
#: The two of them that sum over one state (``sum_by_state`` in both
#: policyengine-us 2.2.1 and 2.60.4), so a batch of whole states computes them
#: exactly. A state-batch engine still refuses the deciles.
STATE_AGGREGATE_VARIABLES = (
    "medicaid_slcsp_state_average_cost_index",
    "medicaid_slcsp_state_denominator",
)


@dataclass(frozen=True)
class DriftMetric:
    """One scored quantity, defined over engine variables of one entity.

    ``variables`` are summed per unit of ``entity`` (a person variable over a
    tax-unit metric is first summed to the tax unit), except ``combine =
    "nonrefundable_ctc"``, which is ``max(min(ctc, ctc_limiting_tax_liability),
    0)`` as the release tool scores the SOI CTC rows, and ``combine =
    "part_b_irmaa"``, which is ``(gross_medicare_part_b_premium_if_enrolled -
    base_part_b_premium) * medicare_enrolled``. ``entity =
    "household_members"`` sums a person variable over each household, as the
    release tool's ``policyengine_variable`` rows do. ``itemizers_only`` masks
    by ``tax_unit_itemizes``. The weight is the H5's household weight of the
    unit's household, so both engines weigh the same rows identically.
    """

    metric_id: str
    label: str
    entity: str
    variables: tuple[str, ...]
    statistic: str
    unit: str
    combine: str = "add"
    itemizers_only: bool = False
    batching: str = BATCHING_HOUSEHOLD
    requirement_id: str | None = None
    target_name: str | None = None
    source_variable: str | None = None


#: The headline aggregates beside the critical rows. Each is a plain engine
#: total over the H5's household weights; none is a calibration target.
HEADLINE_METRICS = (
    DriftMetric(
        "income_tax_net",
        "federal income tax, net of refundable credits",
        "tax_unit",
        ("income_tax",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "ctc_total",
        "Child Tax Credit, total",
        "tax_unit",
        ("ctc",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "eitc_total",
        "Earned Income Tax Credit, total",
        "tax_unit",
        ("eitc",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "social_security_total",
        "Social Security benefits",
        "person",
        ("social_security",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "snap_total",
        "SNAP benefits",
        "spm_unit",
        ("snap",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "snap_units",
        "SPM units receiving SNAP",
        "spm_unit",
        ("snap",),
        STATISTIC_POSITIVE_COUNT,
        "count",
    ),
    DriftMetric(
        "ssi_total",
        "SSI payments",
        "person",
        ("ssi",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "medicaid_enrollment",
        "people enrolled in Medicaid",
        "person",
        ("medicaid_enrolled",),
        STATISTIC_POSITIVE_COUNT,
        "count",
        batching=BATCHING_STATE,
    ),
    DriftMetric(
        "medicaid_total",
        "Medicaid benefit value",
        "person",
        ("medicaid",),
        STATISTIC_SUM,
        "usd",
        batching=BATCHING_STATE,
    ),
    DriftMetric(
        "state_income_tax_total",
        "state income tax",
        "tax_unit",
        ("state_income_tax",),
        STATISTIC_SUM,
        "usd",
    ),
    # policyengine-us#10106 (in 2.60.4, not 2.2.1) lets IRMAA fall back to the
    # earliest input year's income, so a single-year H5 gets payers; 2.60.4
    # also adds takes_up_medicare_part_d_if_eligible (default true). Part B
    # IRMAA is the release's held-out definition (agent guide, microcosm#1206):
    # gross premium if enrolled minus the base premium, over enrollees.
    DriftMetric(
        "medicare_part_b_irmaa_total",
        "Medicare Part B income-related premium (IRMAA)",
        "person",
        (
            "gross_medicare_part_b_premium_if_enrolled",
            "base_part_b_premium",
            "medicare_enrolled",
        ),
        STATISTIC_SUM,
        "usd",
        combine="part_b_irmaa",
    ),
    DriftMetric(
        "medicare_part_d_irmaa_total",
        "Medicare Part D income-related surcharge",
        "person",
        ("income_adjusted_part_d_premium_surcharge",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "medicare_cost_total",
        "Medicare benefit cost",
        "person",
        ("medicare_cost",),
        STATISTIC_SUM,
        "usd",
    ),
    DriftMetric(
        "adjusted_gross_income_total",
        "adjusted gross income",
        "tax_unit",
        ("adjusted_gross_income",),
        STATISTIC_SUM,
        "usd",
    ),
)

#: The SOI ``source_variable`` names the release tool scores by identity, by
#: negative-part magnitude, or as ``max(value, 0)``; every other one by its
#: positive part (``tools/build_us_fiscal_refresh_release.py``'s
#: ``_signed_component``; a build test holds the two equal).
_SIGNED_IDENTITY = frozenset({"adjusted_gross_income", "rent_and_royalty_net_income"})
_LOSS_SOURCES = frozenset(
    {
        "business_net_losses",
        "capital_gains_losses",
        "estate_losses",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_losses",
    }
)


def signed_component(values, source_variable: str):
    """The part of a signed SOI value its row scores, as the release tool does."""

    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if source_variable in _SIGNED_IDENTITY:
        return values
    if source_variable == "capital_gains_losses":
        return np.maximum(values, 0.0)
    if source_variable.endswith("_losses") or source_variable in _LOSS_SOURCES:
        return np.minimum(values, 0.0) * -1.0
    return np.maximum(values, 0.0)


def _soi_row_metric(target: Mapping, requirement_id: str | None) -> DriftMetric | None:
    """A national, all-AGI, all-status SOI or SSA row as a metric, else ``None``.

    Only the rows whose release-tool materialization this module reproduces
    are scored: the SOI slice on the national all-returns population (no state,
    district, filing-status, AGI-band, child-count, positive-EITC or
    taxable-only filter, no ledger filters other than the trivial ones), and
    the plain-sum ``policyengine_variable`` rows.
    """

    metadata = _mapping(target.get("metadata"))
    name = str(target.get("name") or "")
    materializer = metadata.get("materializer")
    base = metadata.get("base_variables") or metadata.get("base_variable")
    if not isinstance(base, str) or not base:
        return None
    variables = tuple(part for part in base.split(",") if part)
    mode = metadata.get("measure_mode", "sum")
    if materializer == "policyengine_variable":
        if mode != "sum" or any(
            metadata.get(key)
            for key in (
                "indicator_map_to",
                "indicator_filter_variable",
                "less_than",
                "age_lower_bound",
                "age_upper_bound",
                "state_fips",
                "congressional_district_geoid",
            )
        ):
            return None
        return DriftMetric(
            f"target:{name}",
            str(metadata.get("diagnostic_target_label") or name),
            "household_members",
            variables,
            STATISTIC_SUM,
            str(metadata.get("ledger_measure_unit") or "usd"),
            requirement_id=requirement_id,
            target_name=name,
        )
    if materializer != "irs_soi_slice":
        return None
    if (
        metadata.get("agi_lower_bound") != "-inf"
        or metadata.get("agi_upper_bound") != "inf"
        or metadata.get("filing_status") != "All"
        or metadata.get("state_fips")
        or metadata.get("congressional_district_geoid")
        or metadata.get("taxable_only") == "true"
        or metadata.get("requires_total_eitc_uprating") == "true"
        or metadata.get("requires_total_soi_uprating") == "true"
        or any(
            key.startswith("ledger_filter_")
            and key not in {"ledger_filter_filing_status", "ledger_filter_income_range"}
            for key in metadata
        )
        or metadata.get("ledger_filter_filing_status", "all") != "all"
        or metadata.get("ledger_filter_income_range", "all") != "all"
    ):
        return None
    source_variable = str(metadata.get("source_variable") or metadata.get("variable"))
    if source_variable in {"count", "tax_filer_individual_count"}:
        return None
    combine = "add"
    if source_variable == "ctc":
        combine, variables = "nonrefundable_ctc", ("ctc", "ctc_limiting_tax_liability")
    return DriftMetric(
        f"target:{name}",
        str(metadata.get("diagnostic_target_label") or name),
        "tax_unit",
        variables,
        STATISTIC_POSITIVE_COUNT if mode == "indicator_sum" else STATISTIC_POSITIVE_SUM,
        "count" if mode == "indicator_sum" else "usd",
        combine=combine,
        itemizers_only=metadata.get("itemized_only") == "true",
        requirement_id=requirement_id,
        target_name=name,
        source_variable=source_variable,
    )


def drift_metric_definitions(calibration_diagnostics: Mapping) -> dict:
    """Every scored metric, derived from the parent's diagnostics and this module.

    Returns ``{"metrics": [...], "unscored_critical_rows": [...]}``. The
    metrics are :data:`HEADLINE_METRICS` plus one per national row of the 18
    exact critical requirements and of the SOI Table 1.4 dollar blanket that
    :func:`_soi_row_metric` reproduces. A critical row it cannot reproduce is
    listed, and :func:`combine_drift` refuses a drift that leaves one of the 18
    unscored.
    """

    from microcosm.data.contract import _target_registry_family
    from microcosm.data.us_critical_targets import (
        US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
        US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,
        is_congressional_district_target,
    )

    metrics = list(HEADLINE_METRICS)
    unscored = []
    seen = set()
    for target in calibration_diagnostics.get("targets") or ():
        if not isinstance(target, Mapping):
            continue
        name = str(target.get("name") or "")
        metadata = _mapping(target.get("metadata"))
        if not name or name in seen or is_congressional_district_target(name, metadata):
            continue
        # The contract selects critical rows by registry family, so this does.
        family = _target_registry_family(target)
        role = str(metadata.get("target_role") or "")
        requirement = next(
            (
                item
                for item in (
                    *US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
                    US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,
                )
                if item.matches(name=name, family=family, target_role=role)
            ),
            None,
        )
        if requirement is None:
            continue
        seen.add(name)
        metric = _soi_row_metric(target, requirement.requirement_id)
        if metric is None:
            unscored.append(
                {"target_name": name, "requirement_id": requirement.requirement_id}
            )
            continue
        metrics.append(metric)
    return {
        "metrics": [asdict(metric) for metric in metrics],
        "unscored_critical_rows": unscored,
    }


def _exact_requirement_ids() -> tuple[str, ...]:
    from microcosm.data.us_critical_targets import (
        US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
    )

    return tuple(
        item.requirement_id for item in US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS
    )


def _critical_tolerances() -> dict[str, float]:
    from microcosm.data.us_critical_targets import US_CRITICAL_TARGET_FIT_REQUIREMENTS

    return {
        item.requirement_id: float(item.max_abs_relative_error)
        for item in US_CRITICAL_TARGET_FIT_REQUIREMENTS
    }


#: Score variants. ``as_stored`` scores the H5 as published; the counterfactual
#: withholds the columns the tested engine newly reads (``column_changes.read``),
#: so ``as_stored - without_newly_read`` is their effect and
#: ``without_newly_read - baseline`` the rest of the engine change.
VARIANT_AS_STORED = "as_stored"
VARIANT_WITHOUT_NEWLY_READ = "without_newly_read_inputs"


#: Recorded by the tool's scorer; bump with any change to how it scores.
SCORING_METHOD = "household_batched_country_loader"


def _score_document_failures(
    scores: Mapping,
    *,
    dataset_sha256: str,
    definitions: Mapping,
    label: str,
    diagnostics_sha256: str,
    spm: object,
    expected_households: int,
) -> list[str]:
    """Shape and binding checks for one embedded score document.

    Beyond shape: the document scored the carried parent diagnostics' metric
    definitions under the parent's own post-export SPM selection, covered
    every household of the H5 in each of its passes, and ran on a runtime
    whose every package was proven against a wheel whose digest PyPI
    publishes for that version.
    """

    failures = []
    if scores.get("parent_diagnostics_sha256") != diagnostics_sha256:
        failures.append(f"{label} scores must score the carried parent diagnostics")
    scoring = _mapping(scores.get("scoring"))
    if (
        scoring.get("method") != SCORING_METHOD
        or scoring.get("period") != SCORING_PERIOD
        or scoring.get("spm") != spm
    ):
        failures.append(
            f"{label} scores must use the {SCORING_METHOD} method for "
            f"{SCORING_PERIOD} under the parent's post-export SPM selection {spm!r}"
        )
    batch = scoring.get("batch_households")
    if type(batch) is not int or batch <= 0:
        failures.append(f"{label} scores must record a positive batch size")
    passes = scoring.get("passes")
    if (
        scoring.get("n_households") != expected_households
        or not isinstance(passes, list)
        or {_mapping(item).get("batching") for item in passes}
        != {BATCHING_HOUSEHOLD, BATCHING_STATE}
        or any(
            _mapping(item).get("households") != expected_households for item in passes
        )
    ):
        failures.append(
            f"{label} scores must cover all {expected_households} households of "
            "the H5 in both their household and their state passes"
        )
    if _mapping(scores.get("bytecode")).get("isolated") is not True:
        failures.append(
            f"{label} scores must have run from source compiled into an isolated "
            "bytecode cache"
        )
    packages = _mapping(_mapping(scores.get("runtime")).get("packages"))
    for package in COMPATIBILITY_PACKAGES:
        identity = _mapping(packages.get(package))
        if not (
            isinstance(identity.get("version"), str)
            and isinstance(identity.get("content_sha256"), str)
            and isinstance(identity.get("wheel_sha256"), str)
            and identity.get("pypi_sha256_verified") is True
        ):
            failures.append(
                f"{label} scores must run {package} proven against a wheel whose "
                "digest PyPI publishes for its version"
            )
    if (
        scores.get("schema_version") != SCORES_SCHEMA_VERSION
        or scores.get("kind") != SCORES_KIND
    ):
        failures.append(
            f"{label} scores must be a schema-{SCORES_SCHEMA_VERSION} {SCORES_KIND}"
        )
    if _mapping(scores.get("dataset")).get("sha256") != dataset_sha256:
        failures.append(f"{label} scores must score the recertified H5")
    if scores.get("metric_definitions_sha256") != canonical_sha256(definitions):
        failures.append(
            f"{label} scores used other metric definitions than this release's"
        )
    values = _mapping(scores.get("metrics"))
    expected = {metric["metric_id"] for metric in definitions["metrics"]}
    if set(values) != expected:
        failures.append(f"{label} scores must score exactly the defined metrics")
    for metric_id, raw in values.items():
        value = _mapping(raw).get("value")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            failures.append(f"{label} metric {metric_id!r} has no finite value")
    code = _mapping(scores.get("code"))
    if set(_mapping(code.get("source_files_sha256"))) != set(PRODUCER_SOURCE_FILES):
        failures.append(f"{label} scores must record the producer source inventory")
    return failures


def _relative(value: float, reference: float | None) -> float | None:
    if reference is None or reference == 0:
        return None
    return (value - reference) / abs(reference)


def combine_drift(
    *,
    calibration_diagnostics: Mapping,
    dataset_sha256: str,
    baseline: Mapping,
    tested: Mapping,
    counterfactual: Mapping | None,
    baseline_engine: Mapping,
    tested_packages: Mapping,
    newly_read_columns: Sequence[str],
    diagnostics_sha256: str,
    expected_households: int,
) -> dict:
    """Combine score documents into the drift document, or raise ``ValueError``.

    ``baseline`` must have been scored under the parent's built-with
    policyengine-us and policyengine-core (``baseline_engine``); ``tested``
    under exactly ``tested_packages`` (every package's version, installed
    content digest and wheel digest, as the loader replay recorded them).
    ``counterfactual``, if given, must be the tested runtime with exactly
    ``newly_read_columns`` withheld, and is required whenever there are any.
    Every document must score the carried diagnostics (``diagnostics_sha256``)
    under the parent's SPM selection and cover all ``expected_households``
    (:func:`_score_document_failures`). Every row recomputes from the
    embedded documents.
    """

    definitions = drift_metric_definitions(calibration_diagnostics)
    spm = _mapping(
        _mapping(calibration_diagnostics.get("build")).get("post_export_scoring")
    ).get("spm")
    failures = []
    if not isinstance(spm, Mapping):
        failures.append(
            "the parent diagnostics record no post-export SPM selection to score under"
        )
    exact = set(_exact_requirement_ids())
    scored_exact = {
        metric["requirement_id"]
        for metric in definitions["metrics"]
        if metric["requirement_id"] in exact
    }
    if scored_exact != exact:
        failures.append(
            "drift must score every exact critical row; unscored: "
            f"{sorted(exact - scored_exact)}"
        )
    sides = [
        ("baseline", baseline, VARIANT_AS_STORED),
        ("tested", tested, VARIANT_AS_STORED),
    ]
    if newly_read_columns:
        if counterfactual is None:
            failures.append(
                "the tested engine newly reads stored columns, so the drift needs "
                f"the {VARIANT_WITHOUT_NEWLY_READ} counterfactual score"
            )
        else:
            sides.append(("counterfactual", counterfactual, VARIANT_WITHOUT_NEWLY_READ))
    elif counterfactual is not None:
        failures.append("a counterfactual score needs newly read columns to withhold")
    for label, scores, variant in sides:
        failures.extend(
            _score_document_failures(
                scores,
                dataset_sha256=dataset_sha256,
                definitions=definitions,
                label=label,
                diagnostics_sha256=diagnostics_sha256,
                spm=spm,
                expected_households=expected_households,
            )
        )
        recorded_variant = _mapping(scores.get("variant"))
        withheld = (
            sorted(newly_read_columns) if variant == VARIANT_WITHOUT_NEWLY_READ else []
        )
        if recorded_variant != {"id": variant, "withheld_columns": withheld}:
            failures.append(f"{label} scores must be the {variant!r} variant")
    baseline_packages = _mapping(_mapping(baseline.get("runtime")).get("packages"))
    for package, field in (("policyengine-us", "model"), ("policyengine-core", "core")):
        want = _mapping(baseline_engine.get(field)).get("version")
        if _mapping(baseline_packages.get(package)).get("version") != want:
            failures.append(
                f"baseline scores must run the parent's built-with {package} {want}"
            )
    for label, scores in (("tested", tested), ("counterfactual", counterfactual)):
        if scores is None:
            continue
        observed = _mapping(_mapping(scores.get("runtime")).get("packages"))
        for package in COMPATIBILITY_PACKAGES:
            want = _mapping(tested_packages.get(package))
            got = _mapping(observed.get(package))
            if any(
                got.get(field) != want.get(field)
                for field in ("version", "content_sha256", "wheel_sha256")
            ):
                failures.append(
                    f"{label} scores must run the certified {package} runtime"
                )
    if failures:
        raise ValueError(" ".join(failures))
    targets = {
        str(target.get("name")): target
        for target in calibration_diagnostics.get("targets") or ()
        if isinstance(target, Mapping)
    }
    tolerances = _critical_tolerances()
    rows = []
    for metric in definitions["metrics"]:
        metric_id = metric["metric_id"]
        base = float(baseline["metrics"][metric_id]["value"])
        new = float(tested["metrics"][metric_id]["value"])
        row = {
            "metric_id": metric_id,
            "label": metric["label"],
            "unit": metric["unit"],
            "requirement_id": metric["requirement_id"],
            "baseline": base,
            "tested": new,
            "difference": new - base,
            "relative_difference": _relative(new, base),
        }
        target = targets.get(metric["target_name"] or "")
        if target is not None:
            value = target.get("target")
            build = target.get("final_estimate")
            row["target"] = value
            row["build_estimate"] = build
            if isinstance(value, int | float) and value:
                row["target_relative_error"] = {
                    "baseline": _relative(base, value),
                    "tested": _relative(new, value),
                }
                tolerance = tolerances.get(metric["requirement_id"] or "")
                if tolerance is not None:
                    row["tolerance"] = tolerance
                    row["within_tolerance"] = {
                        "baseline": abs(_relative(base, value)) <= tolerance,
                        "tested": abs(_relative(new, value)) <= tolerance,
                    }
        if counterfactual is not None:
            withheld = float(counterfactual["metrics"][metric_id]["value"])
            row[VARIANT_WITHOUT_NEWLY_READ] = withheld
            row["newly_read_inputs_effect"] = new - withheld
            row["other_engine_changes_effect"] = withheld - base
        rows.append(row)
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "kind": DRIFT_KIND,
        "dataset_sha256": dataset_sha256,
        "metric_definitions_sha256": canonical_sha256(definitions),
        "unscored_critical_rows": definitions["unscored_critical_rows"],
        "newly_read_columns": sorted(newly_read_columns),
        "rows": rows,
        "filing_status": {
            "baseline": baseline.get("filing_status"),
            "tested": tested.get("filing_status"),
        },
        "tax_unit_roles": {
            "baseline": baseline.get("tax_unit_roles"),
            "tested": tested.get("tax_unit_roles"),
        },
        "scores": {
            "baseline": dict(baseline),
            "tested": dict(tested),
            **(
                {}
                if counterfactual is None
                else {"counterfactual": dict(counterfactual)}
            ),
        },
    }


#: The rows the drift summary names, in order: the headline programmes Max
#: asked about, then the critical rows that carry them.
SUMMARY_METRICS = (
    "income_tax_net",
    "eitc_total",
    "ctc_total",
    "social_security_total",
    "snap_total",
    "medicaid_total",
    "medicaid_enrollment",
    "medicare_part_b_irmaa_total",
    "medicare_part_d_irmaa_total",
)


def drift_summary(drift: Mapping) -> dict:
    """The headline the report carries beside the claim, recomputed from the rows."""

    rows = {row["metric_id"]: row for row in drift.get("rows", ())}
    critical = [row for row in drift.get("rows", ()) if row.get("requirement_id")]
    exact = set(_exact_requirement_ids())
    exact_rows = [row for row in critical if row["requirement_id"] in exact]
    largest = max(
        (row for row in exact_rows if row["relative_difference"] is not None),
        key=lambda row: abs(row["relative_difference"]),
        default=None,
    )

    def leaving_tolerance(rows):
        return sorted(
            row["metric_id"]
            for row in rows
            if _mapping(row.get("within_tolerance")).get("baseline") is True
            and _mapping(row.get("within_tolerance")).get("tested") is False
        )

    return {
        "headline": {
            metric_id: {
                key: rows[metric_id].get(key)
                for key in (
                    "baseline",
                    "tested",
                    "relative_difference",
                    "newly_read_inputs_effect",
                )
                if key in rows[metric_id]
            }
            for metric_id in SUMMARY_METRICS
            if metric_id in rows
        },
        "exact_critical_rows": len(exact_rows),
        "largest_exact_critical_move": None
        if largest is None
        else {
            "metric_id": largest["metric_id"],
            "requirement_id": largest["requirement_id"],
            "relative_difference": largest["relative_difference"],
        },
        "critical_rows_leaving_tolerance": leaving_tolerance(critical),
        "critical_rows_outside_tolerance_under_tested": sorted(
            row["metric_id"]
            for row in critical
            if _mapping(row.get("within_tolerance")).get("tested") is False
        ),
    }


def drift_from_scores(
    drift: Mapping,
    *,
    calibration_diagnostics: Mapping,
    dataset_sha256: str,
    diagnostics_sha256: str,
    expected_households: int,
    baseline_engine: Mapping,
    tested_packages: Mapping,
    newly_read_columns: Sequence[str],
) -> dict:
    """Recompute a drift document from the score documents it embeds.

    The dataset, diagnostics and household count come from the release being
    validated, never from the drift document, so a drift cannot vouch for
    itself.
    """

    scores = _mapping(drift.get("scores"))
    return combine_drift(
        calibration_diagnostics=calibration_diagnostics,
        dataset_sha256=dataset_sha256,
        baseline=_mapping(scores.get("baseline")),
        tested=_mapping(scores.get("tested")),
        counterfactual=scores.get("counterfactual"),
        baseline_engine=baseline_engine,
        tested_packages=tested_packages,
        newly_read_columns=newly_read_columns,
        diagnostics_sha256=diagnostics_sha256,
        expected_households=expected_households,
    )


def h5_household_count(path: Path | str) -> int:
    """Households in a US release H5, from HDF metadata alone."""

    import h5py

    with h5py.File(path, "r") as h5:
        return int(h5["household/table"].shape[0])


# ---------------------------------------------------------------------------
# The claim.
# ---------------------------------------------------------------------------


def _check_recertification_claim(
    specifier: object, *, tested: object, built_with: object
) -> None:
    """Refuse a model claim outside this lane's grammar.

    The source-enrichment guards probe a claim's bounds; they cannot prove
    them (``>=2.60.4,!=3.0.0,!=99999.0.0`` passes them and admits 4.0). A
    recertification therefore takes only two shapes, both bounded by
    construction:

    - ``==T``, the tested version exactly; or
    - ``>=L,<U`` with ``L <= T < U``, ``L`` above the built-with version (the
      parent's own entries carry that one, and nothing between the two was
      measured) and ``U`` at most the next minor release after ``T``.

    A prerelease tested version takes only the exact pin.

    Raises:
        ValueError: The claim has another shape or breaks a bound.
    """

    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        clauses = sorted(
            SpecifierSet(str(specifier)), key=lambda clause: clause.operator
        )
        tested_version = Version(str(tested))
        built_version = Version(str(built_with))
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise ValueError(f"publisher compatibility claim {specifier!r}: {exc}") from exc
    operators = [clause.operator for clause in clauses]
    try:
        bounds = [Version(clause.version) for clause in clauses]
    except InvalidVersion:
        # A wildcard ("==2.60.*") names a series, not a version.
        bounds = []
    if operators == ["=="] and bounds == [tested_version]:
        return
    next_minor = Version(f"{tested_version.major}.{tested_version.minor + 1}")
    if (
        operators == ["<", ">="]
        and not tested_version.is_prerelease
        and bounds[1] <= tested_version < bounds[0] <= next_minor
        and bounds[1] > built_version
    ):
        return
    raise ValueError(
        f"publisher compatibility claim {specifier!r} is not '=={tested_version}' "
        f"or '>=L,<U' with {built_version} < L <= {tested_version} < U <= "
        f"{next_minor}; a recertification claims what it measured and at most "
        "the rest of that minor series"
    )


def claim_entries(
    parent_manifest: Mapping,
    *,
    tested_packages: Mapping,
    specifier: str,
    declared_by: str,
) -> tuple[list[dict], list[dict], dict]:
    """The child's model and Core compatibility lists and its claims block.

    Every parent entry is kept, so consumers on the built-with version keep
    their certification. The model gains one ``publisher_claim`` entry, which
    must contain the tested version and pass the source-enrichment claim
    guards (bounded above and below; ``compatibility_claim_entry``). Core
    gains an exact pin of the tested Core as a ``publisher_claim``, unless a
    parent entry already covers it.

    Raises:
        ValueError: The claim is malformed, does not cover the tested version,
            or the parent already covers the tested model version.
    """

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from microcosm.data.source_enrichment import compatibility_claim_entry

    model_version = _mapping(tested_packages.get("policyengine-us")).get("version")
    core_version = _mapping(tested_packages.get("policyengine-core")).get("version")
    models = [
        dict(entry) for entry in parent_manifest.get("compatible_model_packages") or ()
    ]
    cores = [
        dict(entry) for entry in parent_manifest.get("compatible_core_packages") or ()
    ]

    def covered(entries, name, version):
        return any(
            entry.get("name") == name
            and Version(version) in SpecifierSet(entry["specifier"])
            for entry in entries
        )

    if covered(models, "policyengine-us", model_version):
        raise ValueError(
            f"the parent already certifies policyengine-us {model_version}; "
            "there is nothing to recertify"
        )
    built_with = _mapping(
        _mapping(parent_manifest.get("build")).get("built_with_model_package")
    ).get("version")
    _check_recertification_claim(specifier, tested=model_version, built_with=built_with)
    model = compatibility_claim_entry(
        specifier,
        package="policyengine-us",
        version=model_version,
        declared_by=declared_by,
    )
    claims = {"model": model}
    if not covered(cores, "policyengine-core", core_version):
        claims["core"] = {
            "name": "policyengine-core",
            "specifier": f"=={core_version}",
            "basis": PUBLISHER_CLAIM_BASIS,
            "declared_by": declared_by,
        }
        cores.append(claims["core"])
    models.append(model)
    return models, cores, claims


def claim_measurement(
    claims: Mapping,
    *,
    tested_version: str,
    baseline_version: str,
    drift: Mapping,
    drift_sha256: str,
) -> dict:
    """What the claim rests on: one measured version, its baseline, the drift."""

    return {
        "model": {
            "specifier": claims["model"]["specifier"],
            "measured_versions": [tested_version],
            "baseline_version": baseline_version,
            "drift": {
                "filename": DRIFT_FILE,
                "sha256": drift_sha256,
                "summary": drift_summary(drift),
            },
        }
    }


# ---------------------------------------------------------------------------
# The validator.
# ---------------------------------------------------------------------------


def _check_producer(code: Mapping, failures: list[str], *, label: str) -> None:
    from microcosm.data.source_enrichment import _check_producer_source_identity

    try:
        _check_producer_source_identity(code, inventory=PRODUCER_SOURCE_FILES)
    except (OSError, ValueError) as exc:
        failures.append(f"{label} producer source identity failed: {exc}")


def validate_recertification_candidate(
    release_dir: Path | str,
    *,
    artifact_root: Path | str | None,
    require_compatibility: bool = False,
    compatibility_wheels: tuple[Path | str, ...] = (),
) -> dict:
    """Check a recertification child; return its report or raise.

    Checks, in order: the carried parent manifests and the parent block; every
    parent artifact republished byte for byte with the child's revision; the
    parent's contract replayed on the carried bytes; the child's own manifest
    under the general release-manifest checks; and, once compatibility has
    passed, the loader replay, the claim and the drift, each recomputed. With
    ``require_compatibility`` (certification output, publish preflight and
    publication) a pending child is refused, the loader replay must bind the
    three wheels, and every producer is authenticated against this checkout.
    """

    from microcosm.data.contract import _check_release_manifest

    release_dir = Path(release_dir)
    artifact_root = Path(artifact_root) if artifact_root is not None else None
    failures: list[str] = []
    manifest = _json(release_dir / "release_manifest.json", failures)
    build = _json(release_dir / "build_manifest.json", failures)
    report = _json(release_dir / REPORT_FILE, failures)
    release_id = release_dir.name
    parent = _mapping(report.get("parent"))
    parent_id = parent.get("build_id")
    if manifest.get("release_type") != RECERTIFICATION_RELEASE_TYPE:
        failures.append(
            f"release_manifest.json must declare release_type={RECERTIFICATION_RELEASE_TYPE}"
        )
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("release_type") != RECERTIFICATION_RELEASE_TYPE
    ):
        failures.append(
            f"{REPORT_FILE} must be a schema-{REPORT_SCHEMA_VERSION} report"
        )
    if (
        not isinstance(parent_id, str)
        or release_id == parent_id
        or not release_id.startswith(US_RELEASE_PREFIX)
    ):
        failures.append("a recertification needs a NEW US release id, not its parent's")
    if (
        _mapping(manifest.get("build")).get("build_id") != release_id
        or build.get("build_id") != release_id
    ):
        failures.append(
            "both manifests' build_id must name the child release directory"
        )
    if build.get("release_type") != RECERTIFICATION_RELEASE_TYPE:
        failures.append(
            f"build_manifest.json must declare {RECERTIFICATION_RELEASE_TYPE}"
        )
    if report.get("transformation") != "none":
        failures.append(f"{REPORT_FILE} must declare transformation 'none'")
    if failures:
        raise ReleaseContractError(release_dir, failures)

    # 1. The carried parent manifests are the identity the report names.
    carried_manifest = release_dir / PARENT_RELEASE_MANIFEST_FILE
    carried_build = release_dir / PARENT_BUILD_MANIFEST_FILE
    for path, field in (
        (carried_manifest, "release_manifest_sha256"),
        (carried_build, "build_manifest_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != parent.get(field):
            failures.append(f"{path.name} must be the parent bytes the report pins")
    if failures:
        raise ReleaseContractError(release_dir, failures)
    with tempfile.TemporaryDirectory(prefix=".recertification-describe-") as staging:
        try:
            described = _describe_carried_parent(
                release_dir, parent, artifact_root, Path(staging)
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReleaseContractError(
                release_dir,
                [f"carried parent does not describe a supported parent: {exc}"],
            ) from exc
    if described != dict(parent):
        failures.append(
            f"{REPORT_FILE} parent block must equal what the carried parent manifest "
            "and files describe"
        )
    parent_manifest = json.loads(carried_manifest.read_text())

    # 2. Every parent artifact republished, bytes unchanged.
    dataset = _mapping(report.get("dataset"))
    if dataset != parent.get("dataset"):
        failures.append(
            f"{REPORT_FILE} dataset must be the parent's microdata, unchanged"
        )
    child_artifacts = _mapping(manifest.get("artifacts"))
    expected_artifacts = {}
    for key, raw in _mapping(parent_manifest.get("artifacts")).items():
        entry = dict(_mapping(raw))
        entry["revision"] = release_id
        expected_artifacts[key] = entry
    for key, name in CHILD_ARTIFACT_KEYS.items():
        expected_artifacts[key] = {
            "kind": "evidence",
            "path": name,
            "repo_id": REPO_ID,
            "revision": release_id,
            "sha256": sha256_file(release_dir / name)
            if (release_dir / name).is_file()
            else None,
        }
    compatibility = _mapping(report.get("compatibility"))
    passed = compatibility.get("status") == "passed"
    if passed:
        for key, name in CERTIFIED_ARTIFACT_KEYS.items():
            expected_artifacts[key] = {
                "kind": "diagnostics",
                "path": name,
                "repo_id": REPO_ID,
                "revision": release_id,
                "sha256": sha256_file(release_dir / name)
                if (release_dir / name).is_file()
                else None,
            }
    if dict(child_artifacts) != expected_artifacts:
        failures.append(
            "release manifest artifacts must be the parent's, re-pinned to the child "
            "tag, plus the child's own evidence, each hashing to its file"
        )
    for name, sha in _mapping(parent.get("release_files")).items():
        if not (release_dir / name).is_file() or sha256_file(release_dir / name) != sha:
            failures.append(f"carried parent file {name} must keep the parent bytes")
    if artifact_root is None:
        failures.append(
            "a recertification requires artifact_root for its root artifacts"
        )
    else:
        for path, sha in _mapping(parent.get("root_artifacts")).items():
            local = artifact_root / path
            if not local.is_file() or sha256_file(local) != sha:
                failures.append(f"root artifact {path} must keep the parent bytes")
            if (release_dir / path).exists() or (release_dir / path).is_symlink():
                failures.append(
                    f"root artifact {path} must live only in artifact_root, never in "
                    "the release directory"
                )
    expected_files = {
        *CHILD_ARTIFACT_KEYS.values(),
        "release_manifest.json",
        *_mapping(parent.get("release_files")),
        *(CERTIFIED_ARTIFACT_KEYS.values() if passed else ()),
    }
    present = {path.name for path in release_dir.iterdir()}
    if present != expected_files:
        failures.append(
            "release directory must hold exactly the carried parent files and the "
            f"child's own; unexpected {sorted(present - expected_files)}, missing "
            f"{sorted(expected_files - present)}"
        )
    elif not failures:
        # The publisher decides from the same directory what it uploads where.
        # Every carried artifact must land at its manifest-resolved path in
        # the child: release files under releases/<child>/, root artifacts at
        # their root paths.
        from microcosm.data import release as publisher

        planned_root = publisher._release_manifest_root_artifacts(release_dir)
        planned_release = {
            *publisher._release_manifest_release_artifacts(release_dir),
            *required_release_files(release_id),
        }
        if (
            planned_root != dict(_mapping(parent.get("root_artifacts")))
            or not set(_mapping(parent.get("release_files"))) <= planned_release
        ):
            failures.append(
                "the publisher would not upload every carried artifact to its "
                "manifest path: root "
                f"{sorted(planned_root)} against "
                f"{sorted(_mapping(parent.get('root_artifacts')))}"
            )

    # 3. The parent passes today's contract on the carried bytes.
    parent_contract = replay_parent_contract(
        release_dir, parent, artifact_root=artifact_root
    )
    if parent_contract["other_failures"]:
        failures.append(
            "the parent fails the release contract on its carried bytes: "
            + " ".join(parent_contract["other_failures"])
        )
    if report.get("parent_contract") != parent_contract:
        failures.append(
            f"{REPORT_FILE} parent_contract must equal the replayed contract"
        )

    # 4. The child's manifest is a release manifest; its fields are the parent's.
    parent_build = _mapping(parent_manifest.get("build"))
    for field in ("built_with_model_package", "built_with_core_package"):
        if _mapping(_mapping(manifest.get("build")).get(field)) != _mapping(
            parent_build.get(field)
        ):
            failures.append(
                f"build.{field} must stay the parent's: the H5 was built with it"
            )
    if _mapping(manifest.get("build")).get("built_at") != parent_build.get("built_at"):
        failures.append("build.built_at must stay the parent's: the H5 was built then")
    if _mapping(manifest.get("build")).get("recertified_from") != parent_id:
        failures.append("build.recertified_from must name the parent")
    for field in (
        "schema_version",
        "default_datasets",
        "calibration_diagnostics",
        "dataset_role",
    ):
        if manifest.get(field) != parent_manifest.get(field):
            failures.append(f"release manifest {field} must stay the parent's")
    _check_release_manifest(manifest, release_id, failures)
    calibration = _mapping(build.get("calibration"))
    diagnostics_sha = _mapping(parent.get("release_files")).get(
        "calibration_diagnostics.json"
    )
    if calibration != {
        "mode": "inherited",
        "parent_build_id": parent_id,
        "diagnostics_sha256": diagnostics_sha,
    }:
        failures.append(
            "build calibration must explicitly inherit the parent's diagnostics"
        )
    if _mapping(build.get("dataset")) != dict(dataset):
        failures.append("build manifest dataset must be the parent's microdata")

    # 5. Compatibility.
    if compatibility.get("status") == "pending":
        if require_compatibility:
            failures.append(
                "recertification compatibility is pending; certify it under the "
                "tested wheels"
            )
        for field in ("compatible_model_packages", "compatible_core_packages"):
            if manifest.get(field) != parent_manifest.get(field):
                failures.append(
                    f"a pending recertification must carry the parent's {field}"
                )
        if set(compatibility) != {"status"}:
            failures.append("a pending recertification records no claim or measurement")
    elif passed:
        _check_certified_compatibility(
            release_dir,
            manifest=manifest,
            parent_manifest=parent_manifest,
            report=report,
            parent_contract=parent_contract,
            compatibility=compatibility,
            candidate=None
            if artifact_root is None
            else artifact_root / str(dataset.get("filename")),
            require_wheels=require_compatibility,
            compatibility_wheels=compatibility_wheels,
            failures=failures,
        )
    else:
        failures.append(
            "recertification compatibility status must be pending or passed"
        )
    if require_compatibility:
        _check_producer(_mapping(build.get("code")), failures, label="release")
        if passed:
            drift = _json(release_dir / DRIFT_FILE, [])
            for label, scores in _mapping(drift.get("scores")).items():
                _check_producer(
                    _mapping(_mapping(scores).get("code")),
                    failures,
                    label=f"{label} scores",
                )
    if failures:
        raise ReleaseContractError(release_dir, failures)
    return report


def _describe_carried_parent(
    release_dir: Path, parent: Mapping, artifact_root: Path | None, staging: Path
) -> dict:
    """:func:`describe_parent` on the parent rebuilt from the carried bytes."""

    directory = _reconstructed_parent(release_dir, parent, staging)
    if artifact_root is None:
        raise ValueError("artifact_root is required")
    return describe_parent(directory, artifact_root)


def _check_certified_compatibility(
    release_dir: Path,
    *,
    manifest: Mapping,
    parent_manifest: Mapping,
    report: Mapping,
    parent_contract: Mapping,
    compatibility: Mapping,
    candidate: Path | None,
    require_wheels: bool,
    compatibility_wheels: tuple[Path | str, ...],
    failures: list[str],
) -> None:
    from microcosm.data.stored_inputs import StoredInputRefusalError, register_sha256

    receipt_path = release_dir / RECEIPT_FILE
    drift_path = release_dir / DRIFT_FILE
    if compatibility.get("filename") != RECEIPT_FILE or (
        not receipt_path.is_file()
        or compatibility.get("sha256") != sha256_file(receipt_path)
    ):
        failures.append("compatibility must hash-bind the loader replay receipt")
        return
    receipt = _json(receipt_path, failures)
    if candidate is None or not candidate.is_file():
        failures.append("the recertified H5 must be present to replay the loader")
        return
    try:
        actual = run_recertification_loader_replay(
            candidate,
            require_wheels=require_wheels,
            compatibility_wheels=compatibility_wheels,
        )
    except (
        StoredInputRefusalError,
        ValueError,
        OSError,
        ImportError,
        KeyError,
        TypeError,
    ) as exc:
        failures.append(f"loader replay failed: {exc}")
        return
    if receipt != actual:
        failures.append(
            "loader receipt differs from the replay under the installed runtime"
        )
        return
    tested_packages = _mapping(actual.get("packages"))
    changes = _mapping(_mapping(actual.get("stored_inputs")).get("column_changes"))
    if compatibility.get("stored_column_changes") != recorded_column_changes(changes):
        failures.append(
            "compatibility.stored_column_changes must be the replay's changes with "
            "their reviewed reasons"
        )
    parent_build = _mapping(parent_manifest.get("build"))
    baseline_engine = {
        "model": _mapping(parent_build.get("built_with_model_package")),
        "core": _mapping(parent_build.get("built_with_core_package")),
    }
    claims = _mapping(compatibility.get("publisher_claims"))
    model_claim = _mapping(claims.get("model"))
    try:
        models, cores, expected_claims = claim_entries(
            parent_manifest,
            tested_packages=tested_packages,
            specifier=model_claim.get("specifier"),
            declared_by=model_claim.get("declared_by"),
        )
    except (ValueError, TypeError) as exc:
        failures.append(f"publisher compatibility claim: {exc}")
        return
    if dict(claims) != expected_claims:
        failures.append(
            "compatibility.publisher_claims must be exactly the validated claims"
        )
    if manifest.get("compatible_model_packages") != models:
        failures.append(
            "compatible_model_packages must be the parent's entries plus the declared claim"
        )
    if manifest.get("compatible_core_packages") != cores:
        failures.append(
            "compatible_core_packages must be the parent's entries plus the tested Core"
        )
    if not drift_path.is_file():
        failures.append(f"a certified recertification must deliver {DRIFT_FILE}")
        return
    drift = _json(drift_path, failures)
    diagnostics = _json(release_dir / "calibration_diagnostics.json", failures)
    newly_read = [item["column"] for item in changes.get(CHANGE_READ, ())]
    try:
        expected_drift = drift_from_scores(
            drift,
            calibration_diagnostics=diagnostics,
            dataset_sha256=str(_mapping(report.get("dataset")).get("sha256")),
            diagnostics_sha256=sha256_file(
                release_dir / "calibration_diagnostics.json"
            ),
            expected_households=h5_household_count(candidate),
            baseline_engine=baseline_engine,
            tested_packages=tested_packages,
            newly_read_columns=newly_read,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        failures.append(f"drift does not recompute from its scores: {exc}")
        return
    if drift != expected_drift:
        failures.append(f"{DRIFT_FILE} differs from the drift its scores recompute")
    if drift.get("dataset_sha256") != _mapping(report.get("dataset")).get("sha256"):
        failures.append("drift must score the recertified H5")
    if parent_contract.get("status") == PARENT_CONTRACT_STORED_INPUT_EVIDENCE_REPLACED:
        verdict = _mapping(
            _mapping(
                _mapping(_mapping(drift.get("scores")).get("baseline")).get("engine")
            ).get("stored_input_verdict")
        )
        if (
            verdict.get("passed") is not True
            or verdict.get("register_sha256") != register_sha256()
        ):
            failures.append(
                "the parent carries no stored-input evidence, so the baseline scores "
                "must record a passing stored-input verdict under the built-with "
                "engine against the current register"
            )
    expected_measurement = claim_measurement(
        expected_claims,
        tested_version=str(
            _mapping(tested_packages.get("policyengine-us")).get("version")
        ),
        baseline_version=str(baseline_engine["model"].get("version")),
        drift=expected_drift,
        drift_sha256=sha256_file(drift_path),
    )
    if compatibility.get("claim_measurement") != expected_measurement:
        failures.append(
            "compatibility.claim_measurement must name the measured version, its "
            "baseline and the drift summary its scores recompute"
        )


# ---------------------------------------------------------------------------
# Certification.
# ---------------------------------------------------------------------------


def certify_recertification(
    release_dir: Path | str,
    output_dir: Path | str,
    *,
    artifact_root: Path | str,
    compatibility_wheels: tuple[Path | str, ...],
    baseline_scores: Path | str,
    tested_scores: Path | str,
    counterfactual_scores: Path | str | None,
    compatible_model_specifier: str,
    compatibility_claim_declared_by: str,
) -> Path:
    """Write a certified child next to the pending candidate, or raise.

    The loader replay runs under the installed runtime with the three wheels;
    the three score documents must be the ones the drift recomputes from. The
    claim is declared here and recorded as the declarer's, with the drift
    summary beside it. The candidate and its H5 are left unchanged.
    """

    from microcosm.data.source_enrichment import (
        check_compatibility_claim_declarer,
        parse_compatibility_claim_requirement,
    )

    release_dir, output_dir = Path(release_dir), Path(output_dir)
    if output_dir.exists() or output_dir.name != release_dir.name:
        raise ValueError(
            "output_dir must be new and keep the candidate release id as its basename"
        )
    specifier = parse_compatibility_claim_requirement(
        compatible_model_specifier, package="policyengine-us"
    )
    check_compatibility_claim_declarer(compatibility_claim_declared_by)
    report = validate_recertification_candidate(
        release_dir, artifact_root=artifact_root
    )
    if report["compatibility"] != {"status": "pending"}:
        raise ValueError("certify a pending candidate, not a certified bundle")
    candidate = Path(artifact_root) / report["dataset"]["filename"]
    receipt = run_recertification_loader_replay(
        candidate, require_wheels=True, compatibility_wheels=compatibility_wheels
    )
    tested_packages = receipt["packages"]
    parent_manifest = json.loads(
        (release_dir / PARENT_RELEASE_MANIFEST_FILE).read_text()
    )
    parent_build = parent_manifest["build"]
    models, cores, claims = claim_entries(
        parent_manifest,
        tested_packages=tested_packages,
        specifier=specifier,
        declared_by=compatibility_claim_declared_by,
    )
    changes = receipt["stored_inputs"]["column_changes"]
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    drift = combine_drift(
        calibration_diagnostics=diagnostics,
        dataset_sha256=report["dataset"]["sha256"],
        baseline=json.loads(Path(baseline_scores).read_text()),
        tested=json.loads(Path(tested_scores).read_text()),
        counterfactual=None
        if counterfactual_scores is None
        else json.loads(Path(counterfactual_scores).read_text()),
        baseline_engine={
            "model": parent_build["built_with_model_package"],
            "core": parent_build["built_with_core_package"],
        },
        tested_packages=tested_packages,
        newly_read_columns=[item["column"] for item in changes[CHANGE_READ]],
        diagnostics_sha256=sha256_file(release_dir / "calibration_diagnostics.json"),
        expected_households=h5_household_count(candidate),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".recertification-", dir=output_dir.parent
    ) as staging:
        staged = Path(staging) / release_dir.name
        shutil.copytree(release_dir, staged)
        for path in staged.iterdir():
            path.chmod(0o600)
        write_json(staged / RECEIPT_FILE, receipt)
        write_json(staged / DRIFT_FILE, drift)
        report["compatibility"] = {
            "status": "passed",
            "filename": RECEIPT_FILE,
            "sha256": sha256_file(staged / RECEIPT_FILE),
            "stored_column_changes": recorded_column_changes(changes),
            "publisher_claims": claims,
            "claim_measurement": claim_measurement(
                claims,
                tested_version=tested_packages["policyengine-us"]["version"],
                baseline_version=parent_build["built_with_model_package"]["version"],
                drift=drift,
                drift_sha256=sha256_file(staged / DRIFT_FILE),
            ),
        }
        write_json(staged / REPORT_FILE, report)
        manifest = json.loads((staged / "release_manifest.json").read_text())
        manifest["compatible_model_packages"] = models
        manifest["compatible_core_packages"] = cores
        for key, name in {**CHILD_ARTIFACT_KEYS, **CERTIFIED_ARTIFACT_KEYS}.items():
            manifest["artifacts"][key] = {
                "kind": "diagnostics" if key in CERTIFIED_ARTIFACT_KEYS else "evidence",
                "path": name,
                "repo_id": REPO_ID,
                "revision": release_dir.name,
                "sha256": sha256_file(staged / name),
            }
        write_json(staged / "release_manifest.json", manifest)
        validate_recertification_candidate(
            staged,
            artifact_root=artifact_root,
            require_compatibility=True,
            compatibility_wheels=compatibility_wheels,
        )
        for path in staged.iterdir():
            path.chmod(0o400)
        staged.rename(output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Publication: the parent must be published with the carried bytes.
# ---------------------------------------------------------------------------


def git_blob_sha1(content: bytes) -> str:
    """The id git (and so the Hub) gives a file of ``content``'s bytes."""

    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def require_parent_publication(
    api: object,
    release_dir: Path | str,
    *,
    repo_id: str,
    artifact_root: Path | str | None,
) -> None:
    """Refuse unless the parent tag on the Hub serves what the child carries.

    Runs after the publisher constructs its Hub client and before any branch,
    commit or upload. Two checks, both against the parent's immutable tag:

    - the carried ``release_manifest.json`` and ``build_manifest.json`` are
      byte for byte the parent's published ones. The release manifest pins
      every parent artifact's digest, so this anchors the whole lineage;
    - every parent artifact sits on the Hub where the child will publish it:
      each carried release file under ``releases/<parent>/`` and each root
      artifact at its root path, with the pinned digest (an LFS file's
      SHA-256, or the git blob id of the carried bytes). Offline, which
      artifacts are release files rests on where the files sit locally; this
      is what stops a root artifact moved into the release directory from
      publishing at the wrong path.

    Absence and difference both refuse. A transient Hub failure propagates.
    """

    from microcosm.data.release import _is_absent_entry_error, _is_absent_revision_error

    release_dir = Path(release_dir)
    report = json.loads((release_dir / REPORT_FILE).read_text())
    parent = _mapping(report.get("parent"))
    parent_id, revision = parent.get("build_id"), parent.get("revision")
    if parent.get("repo_id") != repo_id:
        raise ValueError(
            f"the recertification's parent lives in {parent.get('repo_id')!r}, "
            f"not the publication repo {repo_id!r}"
        )
    for carried, remote_name in (
        (PARENT_RELEASE_MANIFEST_FILE, "release_manifest.json"),
        (PARENT_BUILD_MANIFEST_FILE, "build_manifest.json"),
    ):
        path_in_repo = f"releases/{parent_id}/{remote_name}"
        try:
            remote = Path(
                api.hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    repo_type="dataset",
                    revision=revision,
                )
            ).read_bytes()
        except Exception as exc:
            if not (_is_absent_entry_error(exc) or _is_absent_revision_error(exc)):
                raise
            raise ValueError(
                f"parent {parent_id!r} is not published: {path_in_repo} is absent "
                f"at tag {revision!r} in {repo_id} ({exc}). A recertification "
                "republishes only a published parent."
            ) from exc
        if remote != (release_dir / carried).read_bytes():
            raise ValueError(
                f"{path_in_repo} at tag {revision!r} differs from the carried "
                f"{carried}; refusing to publish a child of a parent the Hub does "
                "not serve."
            )
    get_paths_info = getattr(api, "get_paths_info", None)
    if not callable(get_paths_info):
        raise TypeError(
            "publishing a recertification needs a Hub client with get_paths_info, "
            "to authenticate where the parent's artifacts are published"
        )
    expected = {}
    forbidden = set()
    for name, sha in _mapping(parent.get("release_files")).items():
        expected[f"releases/{parent_id}/{name}"] = (sha, release_dir / name)
        forbidden.add(name)
    root = Path(artifact_root) if artifact_root is not None else None
    for path, sha in _mapping(parent.get("root_artifacts")).items():
        expected[path] = (sha, None if root is None else root / path)
        if "/" not in path:
            forbidden.add(f"releases/{parent_id}/{path}")
    published = {
        getattr(info, "path", None): info
        for info in get_paths_info(
            repo_id=repo_id,
            paths=sorted({*expected, *forbidden}),
            revision=revision,
            repo_type="dataset",
        )
    }
    # One location per artifact: a parent that publishes an artifact at both
    # its release-file and its root location leaves the child's placement
    # ambiguous, and the publisher would upload only one of them.
    doubled = sorted(path for path in forbidden if path in published)
    if doubled:
        raise ValueError(
            f"parent {parent_id!r} publishes {doubled} at both a release-file and "
            "a root location; the child's placement of them is ambiguous"
        )
    for path_in_repo, (sha, local) in sorted(expected.items()):
        info = published.get(path_in_repo)
        if info is None:
            raise ValueError(
                f"parent {parent_id!r} publishes no {path_in_repo} at tag "
                f"{revision!r}; the child would publish an artifact the parent "
                "does not have there"
            )
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            matches = getattr(lfs, "sha256", None) == sha
        else:
            matches = (
                local is not None
                and local.is_file()
                and sha256_file(local) == sha
                and getattr(info, "blob_id", None) == git_blob_sha1(local.read_bytes())
            )
        if not matches:
            raise ValueError(
                f"{path_in_repo} at tag {revision!r} is not the artifact the child "
                "carries; refusing to publish it there"
            )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def recorded_claim_measurement(release_dir: Path | str) -> dict:
    """The claim measurement a certified child records, or ``{}``; a report path."""

    try:
        report = json.loads((Path(release_dir) / REPORT_FILE).read_text())
    except (OSError, ValueError):
        return {}
    record = _mapping(_mapping(report).get("compatibility")).get("claim_measurement")
    return dict(record) if isinstance(record, Mapping) else {}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m microcosm.data.compatibility_recertification",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-compatibility", action="store_true")
    parser.add_argument("--compatibility-wheel", action="append", type=Path, default=[])
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--tested-scores", type=Path)
    parser.add_argument("--counterfactual-scores", type=Path)
    parser.add_argument(
        "--compatible-model-specifier",
        help="the claim, as a PEP 508 requirement, e.g. 'policyengine-us==2.60.4'",
    )
    parser.add_argument(
        "--compatibility-claim-declared-by",
        help="who declares the claim, recorded in the bundle",
    )
    args = parser.parse_args(argv)
    if args.certify and (
        args.output_dir is None
        or args.baseline_scores is None
        or args.tested_scores is None
        or args.compatible_model_specifier is None
        or args.compatibility_claim_declared_by is None
    ):
        parser.error(
            "--certify needs --output-dir, --baseline-scores, --tested-scores, "
            "--compatible-model-specifier and --compatibility-claim-declared-by"
        )
    try:
        if args.certify:
            result = certify_recertification(
                args.release_dir,
                args.output_dir,
                artifact_root=args.artifact_root,
                compatibility_wheels=tuple(args.compatibility_wheel),
                baseline_scores=args.baseline_scores,
                tested_scores=args.tested_scores,
                counterfactual_scores=args.counterfactual_scores,
                compatible_model_specifier=args.compatible_model_specifier,
                compatibility_claim_declared_by=args.compatibility_claim_declared_by,
            )
            print(
                json.dumps(
                    {
                        "certified_bundle": str(result),
                        "published": False,
                        "claim_measurement": recorded_claim_measurement(result),
                    }
                )
            )
        else:
            report = validate_recertification_candidate(
                args.release_dir,
                artifact_root=args.artifact_root,
                require_compatibility=args.require_compatibility,
                compatibility_wheels=tuple(args.compatibility_wheel),
            )
            verdict = {
                "valid": True,
                "compatibility": report["compatibility"]["status"],
            }
            measurement = recorded_claim_measurement(args.release_dir)
            if measurement:
                verdict["claim_measurement"] = measurement
            print(json.dumps(verdict))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
