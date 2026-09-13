#!/usr/bin/env python3
"""Emit bounded candidate spec identities; never replace a coverage assertion."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import signal
import stat
import sys
import tempfile
import tomllib
from functools import cached_property
from pathlib import Path

LOCK_SHA256 = "751d5ef5d25406bbae1798667f0e29890d4aad933d323c12912c7d45d8809bb9"
UPLOAD_ACTION_SHA = "ea165f8d65b6e75b540449e92b4886f43607fa02"
CAPS = {
    "candidate-digests.json": 64 * 1024,
    "seed-protocol.json": 256 * 1024,
    "seed-map.json": 256 * 1024,
    "seed-bindings.json": 128 * 1024,
    "environment-and-source.json": 128 * 1024,
    "diagnostic-status.json": 64 * 1024,
}
MAX_TOTAL = 1024 * 1024
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "POPULACE_FIT_N_JOBS",
    "POPULACE_FIT_PREDICT_WORKERS",
)
DISTRIBUTIONS = (
    "jsonschema",
    "jsonschema-specifications",
    "microcosm-fit",
    "numpy",
    "policyengine-core",
    "policyengine-us",
    "pytest",
    "pyyaml",
    "quantile-forest",
    "referencing",
    "scikit-learn",
    "torch",
)
# Filled from the reviewed source-only roster; no import is used to construct it.
SOURCE_PATHS = (
    "packages/microcosm-build/src/microcosm/build/country_spec.py",
    "packages/microcosm-build/src/microcosm/build/frame_sampling.py",
    "packages/microcosm-build/src/microcosm/build/source_runtime.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/canonical.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/engine_abi.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/field_usage.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/loader.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/model.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/resolver.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/schemas.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py",
    "packages/microcosm-build/src/microcosm/build/spec_engine/typed_closure.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/adult_care.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/child_support.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/childcare.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_geography.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/disability_benefits.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/energy_subsidy.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/housing_inputs.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/immigration.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/medicaid_take_up.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/other_health_insurance.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/pregnancy.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puf_aggregate_records.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puf_capital_gains_tail.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puf_qrf_chain.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puf_source_agi.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/puma_ladder.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_contributions.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_financial_assets.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/snap_discretionary_exemption.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/snap_state_take_up.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/snap_take_up.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/source_runtime.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/spine_agreement.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_disability_criteria.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/take_up.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/take_up_contract.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/wic_claim.py",
    "packages/microcosm-build/src/microcosm/build/us_runtime/workers_compensation.py",
    "packages/microcosm-build/tests/test_spec_engine_loader.py",
    "packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py",
    "packages/microcosm-calibrate/src/microcosm/calibrate/gates.py",
    "packages/microcosm-calibrate/src/microcosm/calibrate/initialization.py",
    "packages/microcosm-calibrate/src/microcosm/calibrate/solve.py",
    "packages/microcosm-fit/src/microcosm/fit/qrf.py",
    "packages/microcosm-frame/src/microcosm/frame/adapters/_policyengine_us_source_index.py",
    "packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py",
)
SEED_MODULES = (
    "microcosm.build.frame_sampling",
    "microcosm.build.source_runtime",
    "microcosm.build.us_runtime.acs_transfer",
    "microcosm.build.us_runtime.adult_care",
    "microcosm.build.us_runtime.child_support",
    "microcosm.build.us_runtime.childcare",
    "microcosm.build.us_runtime.congressional_district_geography",
    "microcosm.build.us_runtime.disability_benefits",
    "microcosm.build.us_runtime.energy_subsidy",
    "microcosm.build.us_runtime.geography_ladder",
    "microcosm.build.us_runtime.housing_inputs",
    "microcosm.build.us_runtime.immigration",
    "microcosm.build.us_runtime.medicaid_take_up",
    "microcosm.build.us_runtime.other_health_insurance",
    "microcosm.build.us_runtime.pregnancy",
    "microcosm.build.us_runtime.prior_year_income",
    "microcosm.build.us_runtime.puf_aggregate_records",
    "microcosm.build.us_runtime.puf_capital_gains_tail",
    "microcosm.build.us_runtime.puf_qrf_chain",
    "microcosm.build.us_runtime.puf_source_agi",
    "microcosm.build.us_runtime.puf_support",
    "microcosm.build.us_runtime.puma_ladder",
    "microcosm.build.us_runtime.retirement_contributions",
    "microcosm.build.us_runtime.retirement_distributions",
    "microcosm.build.us_runtime.scf_auto_loans",
    "microcosm.build.us_runtime.scf_wealth",
    "microcosm.build.us_runtime.sipp_financial_assets",
    "microcosm.build.us_runtime.sipp_tips",
    "microcosm.build.us_runtime.sipp_vehicles",
    "microcosm.build.us_runtime.snap_discretionary_exemption",
    "microcosm.build.us_runtime.snap_state_take_up",
    "microcosm.build.us_runtime.snap_take_up",
    "microcosm.build.us_runtime.source_runtime",
    "microcosm.build.us_runtime.ssi_disability_criteria",
    "microcosm.build.us_runtime.ssi_take_up",
    "microcosm.build.us_runtime.take_up",
    "microcosm.build.us_runtime.weeks_unemployed",
    "microcosm.build.us_runtime.wic_claim",
    "microcosm.build.us_runtime.workers_compensation",
    "microcosm.calibrate.exact_k",
    "microcosm.calibrate.gates",
    "microcosm.calibrate.initialization",
    "microcosm.calibrate.solve",
    "microcosm.fit.qrf",
)


#: Public parameter data files the pinned policyengine-us distribution reads
#: at import time (pandas.read_csv at module import: HUD fair market rents,
#: income limits, payment standards and utility allowances; HHS rating areas
#: and second-lowest silver plan costs). The derive phase must import the
#: engine, so these exact files -- and nothing else with a data suffix -- may
#: be opened, and only when their bytes hash to the pin. The pins are the
#: distribution's RECORD digests, verified against policyengine-us 1.819.0 on
#: 2026-09-11. A distribution whose files differ fails closed before the
#: boundary is armed (ENGINE_PARAMETER_PIN).
ENGINE_PUBLIC_PARAMETER_FILES_POLICY = "pinned_engine_public_parameter_files_v1"
ENGINE_PUBLIC_PARAMETER_DISTRIBUTION = "policyengine-us"
ENGINE_PUBLIC_PARAMETER_FILES = (
    (
        "policyengine_us/parameters/gov/hhs/medicaid/geography/aca_rating_areas.csv",
        "a953284013a051956e1951f26e31252fcdf2275010953d09fd16546cb3df508e",
        68571,
    ),
    (
        "policyengine_us/parameters/gov/hhs/medicaid/geography/medicaid_rating_areas.csv",
        "98bd49798e56026262002fb82699b7940d6bfb26589934909682f355ee4e086e",
        59225,
    ),
    (
        "policyengine_us/parameters/gov/hhs/medicaid/geography/second_lowest_silver_plan_cost.csv",
        "b2e415151d5ecc662b62e8b05ea3e1878dcad9e63384c255a314ea3254e47c55",
        293454,
    ),
    (
        "policyengine_us/parameters/gov/hud/fmr/fair_market_rents.csv",
        "aaaa3fe7935e553c5333da51722640704351040092dbf8102c2b19b86a373183",
        1314027,
    ),
    (
        "policyengine_us/parameters/gov/hud/fmr/small_area_fair_market_rents.csv",
        "3ae1edbe6491bafd0b4c4d66401d914e9edf37304f8d93eb43a786a3e4d98ec1",
        246698,
    ),
    (
        "policyengine_us/parameters/gov/hud/income_limits/section8_income_limits.csv",
        "2a8de86c81a8806e75eb15b278a9708e855cd241ba4cc564871c94a565f46442",
        1570746,
    ),
    (
        "policyengine_us/parameters/gov/hud/payment_standards/zip_code_payment_standards.csv",
        "66179687d3e0d9d6c99a58a528aad672ee6a3af8820189209eac6be1ba02d097",
        84054,
    ),
    (
        "policyengine_us/parameters/gov/hud/utility_allowance/county_utility_allowances.csv",
        "0642a75980473ffa33be12d51a3c58ed735472598109b0998b3634d918563809",
        3913,
    ),
)


def pinned_engine_parameter_files(
    *,
    distribution: str = ENGINE_PUBLIC_PARAMETER_DISTRIBUTION,
    pins: tuple[tuple[str, str, int], ...] = ENGINE_PUBLIC_PARAMETER_FILES,
) -> tuple[dict, frozenset[Path]]:
    """Verify the pinned engine parameter files before the boundary is armed.

    Locates the installed distribution through its metadata (the engine is
    not imported), requires every pinned file to be a regular file inside the
    distribution root whose bytes hash to the pin and whose RECORD entry
    carries that digest, and returns the bounded report plus the resolved
    paths the audit hook may allow. Hashing happens once here; the hook only
    compares resolved paths.
    """
    from importlib import metadata

    try:
        dist = metadata.distribution(distribution)
    except metadata.PackageNotFoundError:
        raise RefusalError("ENGINE_PARAMETER_PIN") from None
    root = Path(dist.locate_file("")).resolve()
    record = {}
    for entry in dist.files or ():
        if entry.hash is not None and entry.hash.mode == "sha256":
            padded = entry.hash.value + "=" * (-len(entry.hash.value) % 4)
            record[str(entry)] = base64.urlsafe_b64decode(padded).hex()
    allowed: set[Path] = set()
    files = []
    for relative, expected, size in pins:
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "ENGINE_PARAMETER_PIN")
        require(path.is_file() and not path.is_symlink(), "ENGINE_PARAMETER_PIN")
        raw = path.read_bytes()
        actual = digest(raw)
        require(len(raw) == size and actual == expected, "ENGINE_PARAMETER_PIN")
        require(record.get(relative) == expected, "ENGINE_PARAMETER_PIN")
        allowed.add(path)
        files.append({"path": relative, "sha256": expected, "size": size})
    return (
        {
            "policy": ENGINE_PUBLIC_PARAMETER_FILES_POLICY,
            "distribution": distribution,
            "version": dist.version,
            "files": files,
            "count": len(files),
        },
        frozenset(allowed),
    )


class RefusalError(RuntimeError):
    """A fixed-code refusal with optional immutable, bounded code metadata."""

    def __init__(self, code: str, *, boundary_context: bytes | None = None):
        if boundary_context is not None and (
            type(boundary_context) is not bytes or len(boundary_context) > 4096
        ):
            raise TypeError("REFUSAL_CONTEXT")
        super().__init__(code)
        self.boundary_context = boundary_context


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RefusalError(code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def coordinates() -> dict[str, str]:
    names = {
        "checkout_sha": "DIAG_CHECKOUT_SHA",
        "event_sha": "GITHUB_SHA",
        "workflow_sha": "DIAG_WORKFLOW_SHA",
        "pr_head_sha": "DIAG_PR_HEAD_SHA",
        "pr_base_sha": "DIAG_PR_BASE_SHA",
        "event_merge_sha": "DIAG_MERGE_SHA",
    }
    result = {key: os.environ.get(name, "") for key, name in names.items()}
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    require(event in {"push", "pull_request"}, "EVENT")
    for key, value in result.items():
        # The event payload may omit merge_commit_sha. Preserve that absence;
        # checkout/event SHAs are separate coordinates, not a substitute value.
        optional = key == "event_merge_sha" or (
            event == "push" and key.startswith("pr_")
        )
        require(
            bool(re.fullmatch(r"[0-9a-f]{40}", value)) or (optional and not value),
            "COORDINATES",
        )
    require(result["checkout_sha"] == result["event_sha"], "CHECKOUT")
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        require(os.environ.get(key, "").isdigit(), "RUN_COORDINATES")
        result[key.lower()] = os.environ[key]
    result["event"] = event
    result["repository"] = os.environ.get("GITHUB_REPOSITORY", "")
    require(result["repository"] == "PolicyEngine/microcosm", "REPOSITORY")
    return result


def refusal_path_context(value: object, roots: tuple[tuple[str, Path], ...]) -> dict:
    """Classify an already supplied path without resolving, opening or statting it.

    Only bounded source-like paths under named code/system anchors are shown.
    Other names, data files and credential/log-like paths remain redacted.
    This function changes neither path resolution nor permission decisions.
    """
    if not isinstance(value, (str, bytes, Path)):
        return {"scope": "descriptor_or_nonpath", "path": None}
    path = Path(os.fsdecode(value))
    scope, relative = "unlisted", None
    if not path.is_absolute():
        scope = "relative"
    else:
        for label, anchor in roots:
            if path.is_relative_to(anchor):
                scope, relative = label, path.relative_to(anchor).as_posix()
                break
    source_suffixes = {
        ".py",
        ".pyc",
        ".so",
        ".dylib",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    }
    suffix = path.suffix.lower()
    text = relative if relative is not None else path.name
    private_name = re.search(
        r"(^|[/_.-])(secret|credentials?|password|tokens?|private|keys?|b19001|b25003|occupied)([/_.-]|$)",
        text.lower(),
    )
    shown = (
        text
        if suffix in source_suffixes
        and not private_name
        and re.fullmatch(r"[A-Za-z0-9_./+@-]{1,240}", text)
        and all(part not in ("", ".", "..") for part in Path(text).parts)
        else None
    )
    # Fixed labels identify these reviewed OS metadata candidates without
    # showing arbitrary data/archive filenames or changing read permissions.
    known_path_label = {
        "/etc/os-release": "etc_os_release",
        "/usr/lib/os-release": "usr_lib_os_release",
        "/lib/os-release": "lib_os_release",
        "/etc/localtime": "etc_localtime",
    }.get(path.as_posix())
    if path.name == f"python{sys.version_info.major}{sys.version_info.minor}.zip":
        known_path_label = "python_stdlib_zip_candidate"
    return {
        "scope": scope,
        "path": shown,
        "known_path_label": known_path_label,
        "path_form": "anchor_relative" if relative is not None else "basename_only",
        "source_like": suffix in source_suffixes,
        "name_redacted": shown is None,
    }


def read_refusal_context(code, event, path, requested, roots) -> bytes:
    """Bounded code coordinates only: no locals, source lines or exception text."""
    frames = []
    frame = sys._getframe(1)
    visited = 0
    while frame is not None and visited < 32 and len(frames) < 8:
        visited += 1
        filename = refusal_path_context(frame.f_code.co_filename, roots)
        name = frame.f_code.co_name
        if filename.get("path") is not None and (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
            or name in {"<module>", "<listcomp>", "<dictcomp>", "<genexpr>", "<lambda>"}
        ):
            frames.append({"file": filename, "function": name, "line": frame.f_lineno})
        frame = frame.f_back
    result = {
        "code": code,
        "event": event,
        "requested": refusal_path_context(requested, roots),
        "resolved": refusal_path_context(path, roots),
        "frames": frames,
        "stack_truncated": frame is not None,
        "paths_are_metadata_only": True,
    }
    del frame
    payload = encoded(result)
    if len(payload) > 4096:
        payload = encoded({"code": code, "event": event, "context": "size_bound"})
    return payload


_LOOPBACK_PROBE_HOSTS = frozenset({"::1", "127.0.0.1"})


def _is_urllib3_ipv6_probe(event: str, args: tuple[object, ...]) -> bool:
    """Recognise urllib3's import-time IPv6 capability probe and nothing else.

    ``urllib3.util.connection._has_ipv6`` creates one socket and binds it to
    the loopback address on an ephemeral port; it never connects, sends or
    resolves a name. Only ``socket.__new__`` and a loopback ``socket.bind``
    issued from that exact function are accepted; every other socket event
    keeps the NETWORK_OR_CHILD refusal.
    """
    if event == "socket.bind":
        address = args[1] if len(args) > 1 else None
        if not (
            isinstance(address, tuple)
            and address
            and address[0] in _LOOPBACK_PROBE_HOSTS
        ):
            return False
    elif event != "socket.__new__":
        return False
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code.co_name == "_has_ipv6" and code.co_filename.replace("\\", "/").endswith(
            "urllib3/util/connection.py"
        ):
            return True
        frame = frame.f_back
    return False


def install_boundary(
    root: Path,
    owned: Path,
    output: Path,
    *,
    first_refusal: list[bytes] | None = None,
    distinct_refusals: list[str] | None = None,
    code_contexts: dict[str, bytes] | None = None,
    allowed_data_files: frozenset[Path] = frozenset(),
) -> list[str]:
    refusals: list[str] = []
    blocked_files = (
        ".h5",
        ".hdf",
        ".hdf5",
        ".npz",
        ".npy",
        ".parquet",
        ".csv",
        ".gz",
        ".zip",
        ".sas7bdat",
        ".dta",
        ".pkl",
        ".pickle",
    )
    read_roots = (
        root / "packages",
        root / "tools",
        Path(sys.prefix),
        Path(sys.base_prefix),
        owned,
        output,
        Path("/usr/share/zoneinfo"),
    )
    read_files = {
        root / "uv.lock",
        root / ".github/workflows/test.yml",
        Path("/etc/os-release"),
        Path("/etc/localtime"),
        Path("/dev/null"),
        Path("/proc/cpuinfo"),
        Path("/proc/meminfo"),
        Path("/proc/self/maps"),
        # operation_path resolves /proc/self to this process's numeric PID.
        Path(f"/proc/{os.getpid()}/maps"),
        # psutil reads the CPU-times field layout from /proc/stat when it is
        # imported (joblib's loky backend imports it during the derive phase);
        # metadata only, like /proc/cpuinfo above.
        Path("/proc/stat"),
        # The pinned engine parameter files verified before the hook was armed.
        *allowed_data_files,
    }

    context_roots = (
        ("repository", root),
        ("owned", owned),
        ("output", output),
        ("environment", Path(sys.prefix)),
        ("base_environment", Path(sys.base_prefix)),
        ("system_usr", Path("/usr")),
        ("system_opt", Path("/opt")),
        ("system_lib", Path("/lib")),
        ("system_etc", Path("/etc")),
    )

    def refuse(code: str, *, event=None, path=None, requested=None) -> None:
        # Preserve the historical first-code refusal counter and decisions.
        if not refusals:
            refusals.append(code)
        if (
            distinct_refusals is not None
            and code not in distinct_refusals
            and len(distinct_refusals) < 8
        ):
            distinct_refusals.append(code)
        try:
            context = read_refusal_context(code, event, path, requested, context_roots)
        except Exception:
            context = encoded({"code": code, "context": "unavailable"})
        if first_refusal is not None and not first_refusal:
            first_refusal.append(context)
        # Retain the first immutable record for each fixed boundary code. An
        # optional dependency may catch a denial before a different one escapes.
        if (
            code_contexts is not None
            and code not in code_contexts
            and len(code_contexts) < 8
        ):
            code_contexts[code] = context
        # An optional import probe can swallow an earlier refusal. Attach the
        # actual terminal denial's own bytes instead of reporting that probe.
        raise RefusalError(code, boundary_context=context)

    def operation_path(
        value: object, directory_fd: object = None, *, follow: bool = True
    ) -> Path:
        if isinstance(value, int):
            # Linux is required; descriptor operations must identify an owned file.
            path = Path(os.readlink(f"/proc/self/fd/{value}"))
        else:
            if not isinstance(value, (str, bytes)):
                refuse("FILESYSTEM_PATH")
            path = Path(os.fsdecode(value))
            if not path.is_absolute():
                if directory_fd not in (None, -1):
                    if not isinstance(directory_fd, int):
                        refuse("FILESYSTEM_DESCRIPTOR")
                    path = Path(os.readlink(f"/proc/self/fd/{directory_fd}")) / path
                else:
                    path = Path.cwd() / path
        # Unlink/rename operate on a final symlink itself, not its target.
        return path.resolve() if follow else path.parent.resolve() / path.name

    def owned_mutation(
        value: object, directory_fd: object = None, *, follow: bool = True
    ) -> None:
        path = operation_path(value, directory_fd, follow=follow)
        if not any(path.is_relative_to(p) for p in (owned, output)):
            refuse("WRITE_SCOPE")

    def audit(event: str, args: tuple[object, ...]) -> None:
        # gethostname reads a local OS label; it opens no network connection.
        if (
            (
                event.startswith("socket.")
                and event != "socket.gethostname"
                and not _is_urllib3_ipv6_probe(event, args)
            )
            or event.startswith("subprocess.")
            or event
            in {
                "os.system",
                "os.exec",
                "os.fork",
                "os.forkpty",
                "os.posix_spawn",
                "pty.spawn",
            }
        ):
            refuse(
                "NETWORK_OR_CHILD",
                event=(
                    event
                    if event
                    in {
                        "subprocess.Popen",
                        "os.system",
                        "os.exec",
                        "os.fork",
                        "os.forkpty",
                        "os.posix_spawn",
                        "pty.spawn",
                    }
                    else "network_or_child_event"
                ),
            )
        if event in {"os.link", "os.symlink"}:
            refuse("FILESYSTEM_LINK")
        if event == "os.rename":
            owned_mutation(args[0], args[2], follow=False)
            owned_mutation(args[1], args[3], follow=False)
        elif event in {"os.remove", "os.rmdir"}:
            owned_mutation(args[0], args[1], follow=False)
        elif event in {"os.mkdir", "os.chmod"}:
            owned_mutation(args[0], args[2])
        elif event in {"os.chown", "os.utime"}:
            owned_mutation(args[0], args[3])
        elif event in {"os.truncate", "os.setxattr", "os.removexattr"}:
            owned_mutation(args[0])
        if event == "open" and args:
            path = operation_path(args[0])
            name = str(path)
            if name.lower().endswith(blocked_files) and path not in allowed_data_files:
                refuse("DATA_FILE", event="open", path=path, requested=args[0])
            flags = args[2] if len(args) > 2 else 0
            writing = isinstance(flags, int) and bool(
                flags
                & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writing and name != os.devnull:
                if not any(path.is_relative_to(p) for p in (owned, output)):
                    refuse("WRITE_SCOPE")
            if (
                not writing
                and path not in read_files
                and not any(path.is_relative_to(p) for p in read_roots)
            ):
                refuse("READ_SCOPE", event="open", path=path, requested=args[0])

    sys.addaudithook(audit)
    return refusals


def source_stamps(root: Path) -> dict[str, str]:
    paths = (
        *SOURCE_PATHS,
        "tools/spec_seed_identity_diagnostics.py",
        "uv.lock",
        ".github/workflows/test.yml",
    )
    result = {}
    for name in paths:
        path = root / name
        require(path.is_file() and not path.is_symlink(), "SOURCE_PATH")
        result[name] = digest(path.read_bytes())
    require(result["uv.lock"] == LOCK_SHA256, "LOCK")
    return result


def check_loaded_origins(root: Path) -> None:
    for name in SOURCE_PATHS:
        if "/src/" not in name:
            continue
        module_name = name.split("/src/", 1)[1].removesuffix(".py").replace("/", ".")
        module = sys.modules.get(module_name)
        if module is not None:
            require(
                Path(module.__file__).resolve() == (root / name).resolve(),
                "MODULE_ORIGIN",
            )
    helper = sys.modules.get("test_spec_engine_loader")
    if helper is not None:
        require(
            Path(helper.__file__).resolve()
            == (
                root / "packages/microcosm-build/tests/test_spec_engine_loader.py"
            ).resolve(),
            "HELPER_ORIGIN",
        )


def versions(lock: dict) -> dict[str, str]:
    actual = {}
    for name in DISTRIBUTIONS:
        expected = {p["version"] for p in lock["package"] if p["name"] == name}
        require(len(expected) == 1, "LOCK_VERSION")
        actual[name] = importlib.metadata.version(name)
        require(actual[name] in expected, "INSTALLED_VERSION")
    return actual


def omit_absent_stdlib_zip() -> bool:
    """Omit only CPython's exact absent ZIP search entry; admit no archive read."""
    candidate = str(
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    )
    if candidate not in sys.path:
        return False
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        # lstat, rather than exists, distinguishes an absent path from a broken
        # symlink. All other entries and their order remain unchanged.
        sys.path[:] = [entry for entry in sys.path if entry != candidate]
        return True
    return False


class _CpuOnlyCudaBindingsFinder:
    """Use Torch's real optional ImportError fallback during its import only."""

    def __init__(self):
        self.attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "cuda.bindings":
            return None
        self.attempts += 1
        require(self.attempts <= 8, "CUDA_BINDINGS_PROBE_LIMIT")
        raise ModuleNotFoundError(
            "cuda.bindings excluded by CPU-only diagnostic bootstrap",
            name="cuda.bindings",
        )


@contextlib.contextmanager
def cpu_only_torch_import(bootstrap: dict):
    """Temporarily exclude optional CUDA bindings, never replace a Torch module."""
    require(
        "torch" not in sys.modules
        and not any(
            name == "cuda.bindings" or name.startswith("cuda.bindings.")
            for name in sys.modules
        ),
        "TORCH_BOOTSTRAP_PRELOADED",
    )
    finder = _CpuOnlyCudaBindingsFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        # Remove this exact finder on both success and failure, retaining any
        # legitimate finder changes made by ordinary dependency imports.
        sys.meta_path[:] = [entry for entry in sys.meta_path if entry is not finder]
        bootstrap["cuda_bindings_import_attempts"] = finder.attempts
        bootstrap["temporary_finder_removed"] = not any(
            entry is finder for entry in sys.meta_path
        )


def verify_cpu_only_torch_import(bootstrap: dict) -> None:
    utilities = sys.modules.get("torch.cuda._utils")
    require(
        bootstrap["temporary_finder_removed"] is True
        and 1 <= bootstrap["cuda_bindings_import_attempts"] <= 8
        and utilities is not None
        and getattr(utilities, "_HAS_CUDA_BINDINGS", None) is False
        and getattr(utilities, "_cuda_bindings_runtime", False) is None
        and not any(
            name == "cuda.bindings" or name.startswith("cuda.bindings.")
            for name in sys.modules
        ),
        "TORCH_CPU_FALLBACK",
    )
    bootstrap["torch_optional_bindings_fallback_verified"] = True


def prime_processor_metadata(bootstrap: dict) -> tuple:
    """Use actual machine architecture for this process's processor metadata.

    This initializes only the documented writable cached_property on the real
    uname result. It replaces no platform function, descriptor, class or module.
    No equivalence to the external uname -p probe is asserted.
    """
    require(sys.platform == "linux", "PROCESSOR_PLATFORM")
    observed = os.uname()
    info = platform.uname()
    fields = ("system", "node", "release", "version", "machine")
    actual = (
        observed.sysname,
        observed.nodename,
        observed.release,
        observed.version,
        observed.machine,
    )
    require(
        type(info) is platform.uname_result
        and type(vars(type(info)).get("processor")) is cached_property
        and tuple(getattr(info, name) for name in fields) == actual
        and type(observed.machine) is str
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", observed.machine)),
        "PROCESSOR_UNAME_CONTRACT",
    )
    prior = vars(info).get("processor")
    require(
        "processor" not in vars(info)
        or (type(prior) is str and prior == observed.machine),
        "PROCESSOR_CACHE_CONFLICT",
    )
    evidence = {
        "policy": "os_uname_machine_as_processor_metadata_v1",
        "value": observed.machine,
        "source": "os.uname.machine",
        "uname_p_parity_claimed": False,
        "stdlib_callables_replaced": False,
        "lifetime": "diagnostic_process",
        "process_maps_permission": "exact_resolved_own_pid_maps_alias",
    }
    info.processor = observed.machine
    bootstrap["system_metadata"] = evidence
    return info, actual, encoded(evidence)


def verify_processor_metadata(state: tuple, bootstrap: dict) -> None:
    info, actual, evidence = state
    observed = os.uname()
    require(
        platform.uname() is info
        and tuple(
            getattr(info, name)
            for name in ("system", "node", "release", "version", "machine")
        )
        == actual
        and (
            observed.sysname,
            observed.nodename,
            observed.release,
            observed.version,
            observed.machine,
        )
        == actual
        and type(vars(info).get("processor")) is str
        and vars(info)["processor"] == actual[-1]
        and encoded(bootstrap.get("system_metadata")) == evidence,
        "PROCESSOR_METADATA_CHANGED",
    )


def thread_controls(torch) -> dict[str, object]:
    controls = {
        "environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "torch_intra_op": torch.get_num_threads(),
        "torch_inter_op": torch.get_num_interop_threads(),
    }
    require(
        all(value == "1" for value in controls["environment"].values())
        and controls["torch_intra_op"] == 1
        and controls["torch_inter_op"] == 1,
        "THREADS_CHANGED",
    )
    return controls


def validate_payloads(payloads: dict[str, bytes]) -> None:
    require(set(payloads) == set(CAPS), "OUTPUT_ROSTER")
    require(
        all(len(data) <= CAPS[name] for name, data in payloads.items()), "OUTPUT_CAP"
    )
    require(sum(map(len, payloads.values())) <= MAX_TOTAL, "TOTAL_CAP")


def publish(output: Path, payloads: dict[str, bytes], *, thread_state=None) -> None:
    validate_payloads(payloads)
    retained = tuple((name, payloads[name]) for name in CAPS)
    status_name = "diagnostic-status.json"
    status_value = json.loads(payloads[status_name])
    if status_value.get("completed") is True:
        require(
            status_value.get("payload_sha256")
            == {
                name: digest(data)
                for name, data in payloads.items()
                if name != status_name
            },
            "OUTPUT_SEAL",
        )
    # A failed prior write can leave only one of these known owned temporaries.
    # Remove it before using O_EXCL again; never recursively clean arbitrary paths.
    for name in CAPS:
        temporary = output / (name + ".part")
        if temporary.exists() or temporary.is_symlink():
            info = temporary.lstat()
            require(
                stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode), "OUTPUT_PART"
            )
            temporary.unlink()
    for name, data in payloads.items():
        if name == status_name:
            continue
        temporary = output / (name + ".part")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / name)
    # Prepare the terminal status, but keep the visible marker incomplete while
    # every payload and the prepared status are checked through their actual files.
    prepared = output / (status_name + ".part")
    with prepared.open("xb") as stream:
        stream.write(payloads[status_name])
        stream.flush()
        os.fsync(stream.fileno())
    existing_status = output / status_name
    expected_names = (set(CAPS) - {status_name}) | {prepared.name}
    if existing_status.exists():
        expected_names.add(status_name)
        info = existing_status.lstat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= CAPS[status_name],
            "OUTPUT_STATUS",
        )
        require(
            json.loads(existing_status.read_bytes()).get("completed") is not True,
            "OUTPUT_ALREADY_COMPLETE",
        )
    require({p.name for p in output.iterdir()} == expected_names, "OUTPUT_FILES")
    observed = {}
    for name, cap in CAPS.items():
        path = prepared if name == status_name else output / name
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, "OUTPUT_FILE")
        observed[name] = path.read_bytes()
    require(observed == payloads, "OUTPUT_CHANGED")
    validate_payloads(observed)
    if thread_state is not None:
        torch, controls, refusals = thread_state
        require(thread_controls(torch) == controls, "THREADS_CHANGED")
        require(not refusals, "BOUNDARY_REFUSAL")
    require(
        tuple((name, payloads[name]) for name in CAPS) == retained, "OUTPUT_CHANGED"
    )
    # Last filesystem operation: success becomes visible only after verification.
    # There is deliberately no post-commit read/check/cleanup in this function.
    prepared.replace(existing_status)


def derive(
    root: Path,
    owned: Path,
    run: dict[str, str],
    before: dict[str, str],
    installed: dict[str, str],
    torch,
    controls: dict[str, object],
    bootstrap: dict[str, object],
) -> dict[str, bytes]:
    # The exact helper is imported; pytest collection and its tests are never run.
    sys.path.insert(0, str(root / "packages/microcosm-build/tests"))
    from test_spec_engine_loader import SELECTION_KERNEL_IDS, _rich_minimal

    from microcosm.build.spec_engine import KernelRegistry, load_bundle
    from microcosm.build.spec_engine.canonical import canonical_json_bytes, sha256_json
    from microcosm.build.spec_engine.compiler_ir import compile_spec
    from microcosm.build.spec_engine.seeds import source_inventory_sha256

    check_loaded_origins(root)
    countries = {country: load_bundle(country) for country in ("am", "be", "uk")}
    minimal = load_bundle(
        _rich_minimal(owned / "xx", note="first", store="local:a"),
        kernel_registry=KernelRegistry.from_ids(SELECTION_KERNEL_IDS),
    )
    us = load_bundle("us")
    compiled = compile_spec(us)
    protocol = us.seed_protocol.to_wire()
    seed_map = compiled.seed_stream_map.to_wire()
    bindings = [binding.to_wire() for binding in us.seed_site_bindings]
    modules = tuple(
        sorted({m for k in us.seed_protocol.kernels for m in k.source_modules})
    )
    require(modules == SEED_MODULES, "SEED_MODULE_ROSTER")
    for kernel in us.seed_protocol.kernels:
        require(
            source_inventory_sha256(kernel.source_modules) == kernel.source_sha256,
            "SEED_SOURCE_CHANGED",
        )
    require(
        (
            len(protocol["sites"]),
            len(protocol["streams"]),
            sum(len(s.owners) for s in compiled.seed_stream_map.sites),
            len(seed_map["owners"]),
        )
        == (53, 14, 112, 54),
        "SEED_SHAPE",
    )
    require(
        compiled.seed_stream_map.implementation_sha256
        == us.seed_protocol.implementation_sha256,
        "SEED_HEADER",
    )
    candidates = {
        "status": "candidate_values_only",
        "coverage": "not_asserted_by_diagnostic",
        "digests": {
            **{country: value.spec_sha256 for country, value in countries.items()},
            "rich_minimal": minimal.spec_sha256,
            "seed_protocol": us.seed_protocol.implementation_sha256,
            "seed_map": sha256_json(seed_map),
        },
        "shape": {"sites": 53, "streams": 14, "bindings": 112, "owners": 54},
        "locations": {
            "am_be_uk": "packages/microcosm-build/tests/test_spec_engine_country_bundles.py",
            "rich_minimal": "packages/microcosm-build/tests/test_spec_engine_loader.py",
            "seed_protocol_seed_map": "packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:EXPECTED_HASHES",
        },
    }
    environment = {
        "coordinates": run,
        "python": sys.version,
        "platform": sys.platform,
        "runner_image": {k: os.environ.get(k, "") for k in ("ImageOS", "ImageVersion")},
        "uv_version": os.environ.get("DIAG_UV_VERSION", ""),
        "lock_sha256": LOCK_SHA256,
        "distributions": installed,
        "thread_controls": controls,
        "dependency_bootstrap": bootstrap,
        "source_sha256": before,
        "upload_action_commit": UPLOAD_ACTION_SHA,
        "seed_modules": list(modules),
        "compiler_source_inventory": list(compiled.compiler_ir_abi.source_inventory),
        "limits": {
            "cpu_seconds": 600,
            "wall_seconds": 900,
            "threads": 1,
            "total_output_bytes": MAX_TOTAL,
            "file_caps": CAPS,
        },
        "boundary": "live_engine_metadata; no_native_data_network_children_or_fits",
    }
    # Complete canonical retained values precede the final source/metadata reads.
    payloads = {
        "candidate-digests.json": encoded(candidates),
        "seed-protocol.json": canonical_json_bytes(protocol),
        "seed-map.json": canonical_json_bytes(seed_map),
        "seed-bindings.json": canonical_json_bytes(bindings),
        "environment-and-source.json": encoded(environment),
    }
    lock = tomllib.loads((root / "uv.lock").read_text())
    require(versions(lock) == installed, "VERSION_CHANGED")
    for kernel in us.seed_protocol.kernels:
        require(
            source_inventory_sha256(kernel.source_modules) == kernel.source_sha256,
            "SEED_SOURCE_CHANGED",
        )
    check_loaded_origins(root)
    require(source_stamps(root) == before, "SOURCE_CHANGED")
    require(thread_controls(torch) == controls, "THREADS_CHANGED")
    payloads["diagnostic-status.json"] = encoded(
        {
            "completed": True,
            "dependency_bootstrap": bootstrap,
            "status": "candidate_values_only",
            "coverage_pass": False,
            "payload_sha256": {name: digest(data) for name, data in payloads.items()},
        }
    )
    return payloads


@contextlib.contextmanager
def retained_owned_directory(runner_temp: Path):
    """Retain one private invented-fixture workspace for CI runner cleanup.

    CPython's descriptor-relative rmtree opens omit dir_fd in the audit event.
    Keeping the workspace avoids weakening the diagnostic's path restrictions.
    This directory is not one of the six uploaded diagnostic artifacts.
    """
    yield Path(tempfile.mkdtemp(prefix="spec-seed-owned-", dir=runner_temp))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.absolute()
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    require(output == runner_temp / "spec-seed-diagnostics", "OUTPUT_PATH")
    require(not output.exists() and not output.is_symlink(), "OUTPUT_EXISTS")
    output.mkdir(mode=0o700)
    phase = "environment"
    with retained_owned_directory(runner_temp) as owned:
        require(sys.version_info[:3] == (3, 14, 4), "PYTHON")
        require(sys.platform == "linux", "PLATFORM")
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV), "THREADS")
        sys.dont_write_bytecode = True
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
        signal.signal(
            signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RefusalError("WALL"))
        )
        signal.alarm(900)
        first_refusal: list[bytes] = []
        distinct_refusals: list[str] = []
        code_contexts: dict[str, bytes] = {}
        engine_parameter_files, allowed_data_files = pinned_engine_parameter_files()
        refusals = install_boundary(
            root,
            owned,
            output,
            first_refusal=first_refusal,
            distinct_refusals=distinct_refusals,
            code_contexts=code_contexts,
            allowed_data_files=allowed_data_files,
        )
        # Dependency-created temporaries must stay inside this invocation's scope.
        tempfile.tempdir = str(owned)
        os.environ["TMPDIR"] = str(owned)
        initial = {name: encoded({"status": "incomplete"}) for name in CAPS}
        initial["diagnostic-status.json"] = encoded(
            {
                "completed": False,
                "status": "in_progress",
                "coverage_pass": False,
            }
        )
        publish(output, initial)
        bootstrap = {
            "policy": "cpu_only_torch_optional_cuda_bindings_exclusion_v1",
            "finder_scope": "torch_import_only",
            "stdlib_zip_entry": "exact_base_prefix_lib_python_version_zip",
            "absent_stdlib_zip_omitted": False,
            "cuda_bindings_import_attempts": 0,
            "temporary_finder_removed": True,
            "torch_optional_bindings_fallback_verified": False,
            "owned_temporary_storage": {
                "policy": "retain_private_directory_until_ci_runner_cleanup",
                "parent": "RUNNER_TEMP",
                "directories_per_invocation": 1,
                "contents": "invented_spec_fixture_and_dependency_temporaries",
                "uploaded": False,
            },
            "engine_public_parameter_files": engine_parameter_files,
        }
        try:
            bootstrap["absent_stdlib_zip_omitted"] = omit_absent_stdlib_zip()
            run = coordinates()
            before = source_stamps(root)
            lock = tomllib.loads((root / "uv.lock").read_text())
            installed = versions(lock)
            phase = "derive"
            processor_state = prime_processor_metadata(bootstrap)
            with (
                open(os.devnull, "w") as quiet,
                contextlib.redirect_stdout(quiet),
                contextlib.redirect_stderr(quiet),
            ):
                # Keep real locked Torch/thread controls under the same audit
                # boundary, explicitly taking its optional CUDA bindings fallback.
                with cpu_only_torch_import(bootstrap):
                    import torch

                verify_cpu_only_torch_import(bootstrap)
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                controls = thread_controls(torch)
                payloads = derive(
                    root, owned, run, before, installed, torch, controls, bootstrap
                )
            verify_processor_metadata(processor_state, bootstrap)
            require(not refusals, "BOUNDARY_REFUSAL")
            exit_code = 0
        except Exception as error:
            code = str(error) if type(error) is RefusalError else "DERIVATION_REFUSED"
            payloads = {name: encoded({"status": "incomplete"}) for name in CAPS}
            payloads["diagnostic-status.json"] = encoded(
                {
                    "completed": False,
                    "status": "refused",
                    "phase": phase,
                    "code": code,
                    "dependency_bootstrap": bootstrap,
                    "first_boundary_refusal": (
                        json.loads(first_refusal[0]) if first_refusal else None
                    ),
                    "terminal_boundary_refusal": (
                        json.loads(error.boundary_context)
                        if type(error) is RefusalError
                        and error.boundary_context is not None
                        else None
                    ),
                    "boundary_refusal_codes": list(distinct_refusals),
                    "boundary_refusal_code_limit": 8,
                    "boundary_refusal_contexts": [
                        json.loads(context) for context in code_contexts.values()
                    ],
                    "boundary_refusal_context_limit": {
                        "codes": 8,
                        "bytes_per_code": 4096,
                    },
                    "coverage_pass": False,
                }
            )
            exit_code = 1
        retained = tuple((name, payloads[name]) for name in CAPS)
    # The private fixture workspace remains for CI runner cleanup. No borrowed
    # source/artifact I/O follows the terminal publication.
    require(
        tuple((name, payloads[name]) for name in CAPS) == retained, "OUTPUT_CHANGED"
    )
    try:
        if exit_code == 0:
            verify_processor_metadata(processor_state, bootstrap)
            require(thread_controls(torch) == controls, "THREADS_CHANGED")
            require(not refusals, "BOUNDARY_REFUSAL")
        publish(
            output,
            dict(retained),
            thread_state=(torch, controls, refusals) if exit_code == 0 else None,
        )
    except Exception as error:
        refused = {name: encoded({"status": "incomplete"}) for name in CAPS}
        refused["diagnostic-status.json"] = encoded(
            {
                "completed": False,
                "status": "refused",
                "phase": "publish",
                "code": "OUTPUT_REFUSED",
                "dependency_bootstrap": bootstrap,
                "first_boundary_refusal": (
                    json.loads(first_refusal[0]) if first_refusal else None
                ),
                "terminal_boundary_refusal": (
                    json.loads(error.boundary_context)
                    if type(error) is RefusalError
                    and error.boundary_context is not None
                    else None
                ),
                "boundary_refusal_codes": list(distinct_refusals),
                "boundary_refusal_code_limit": 8,
                "boundary_refusal_contexts": [
                    json.loads(context) for context in code_contexts.values()
                ],
                "boundary_refusal_context_limit": {"codes": 8, "bytes_per_code": 4096},
                "coverage_pass": False,
            }
        )
        publish(output, refused)
        signal.alarm(0)
        return 1
    signal.alarm(0)
    return exit_code


def outer_failure_context(error: Exception) -> bytes:
    """Record fixed exception labels and code coordinates without file I/O.

    This is stderr-only evidence. In particular, failed fixture cleanup must
    leave the initial incomplete artifact intact rather than publish success.
    """
    labels = {
        "AssertionError",
        "AttributeError",
        "FileNotFoundError",
        "ImportError",
        "IndexError",
        "IsADirectoryError",
        "KeyError",
        "MemoryError",
        "ModuleNotFoundError",
        "NameError",
        "NotADirectoryError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "RecursionError",
        "RuntimeError",
        "TypeError",
        "UnboundLocalError",
        "UnicodeDecodeError",
        "ValueError",
    }
    kind = type(error).__name__
    exception_type = (
        kind
        if type(error).__module__ == "builtins" and kind in labels
        else "other_exception"
    )
    code, boundary = "OUTER_FAILURE", None
    if type(error) is RefusalError:
        exception_type = "RefusalError"
        value = error.args[0] if len(error.args) == 1 else None
        code = (
            value
            if value
            in {
                "READ_SCOPE",
                "WRITE_SCOPE",
                "DATA_FILE",
                "FILESYSTEM_PATH",
                "FILESYSTEM_DESCRIPTOR",
                "FILESYSTEM_LINK",
                "NETWORK_OR_CHILD",
                "WALL",
                "OUTPUT_CHANGED",
                "THREADS_CHANGED",
                "BOUNDARY_REFUSAL",
                "OUTPUT_CAP",
                "OUTPUT_FILE",
                "OUTPUT_FILES",
                "OUTPUT_ROSTER",
                "OUTPUT_SEAL",
                "OUTPUT_PART",
                "OUTPUT_STATUS",
                "OUTPUT_ALREADY_COMPLETE",
            }
            else "OTHER_FIXED_REFUSAL"
        )
        if error.boundary_context is not None:
            boundary = json.loads(error.boundary_context)
    roots = (
        ("repository", Path(__file__).absolute().parents[1]),
        ("environment", Path(sys.prefix)),
        ("base_environment", Path(sys.base_prefix)),
        ("system_usr", Path("/usr")),
        ("system_opt", Path("/opt")),
        ("system_lib", Path("/lib")),
    )
    frames, visited = [], 0
    trace = error.__traceback__
    while trace is not None and visited < 32:
        visited += 1
        code_object = trace.tb_frame.f_code
        filename = refusal_path_context(code_object.co_filename, roots)
        name = code_object.co_name
        if filename.get("path") is not None and (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
            or name in {"<module>", "<listcomp>", "<dictcomp>", "<genexpr>", "<lambda>"}
        ):
            frames.append({"file": filename, "function": name, "line": trace.tb_lineno})
            frames = frames[-8:]
        trace = trace.tb_next
    truncated = trace is not None or visited > 8
    del trace
    result = {
        "diagnostic_outer_failure": True,
        "completed": False,
        "coverage_pass": False,
        "exception_type": exception_type,
        "code": code,
        "boundary_refusal": boundary,
        "traceback_code_frames": frames,
        "traceback_truncated": truncated,
        "artifact_status_not_modified": True,
        "message_locals_source_lines_emitted": False,
    }
    payload = encoded(result)
    if len(payload) > 8192:
        payload = encoded(
            {
                "diagnostic_outer_failure": True,
                "completed": False,
                "coverage_pass": False,
                "exception_type": exception_type,
                "code": code,
                "context": "size_bound",
                "artifact_status_not_modified": True,
            }
        )
    return payload


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # No second publication after failed cleanup. Emit only bounded metadata;
        # never exception messages, locals, source lines or an unfiltered trace.
        try:
            context = outer_failure_context(error)
        except Exception:
            context = b'{"diagnostic_outer_failure":true,"completed":false,"context":"unavailable"}\n'
        print(context.decode("ascii").rstrip("\n"), file=sys.stderr)
        raise SystemExit(1) from None
