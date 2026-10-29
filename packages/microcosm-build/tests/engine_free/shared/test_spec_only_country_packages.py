"""Country packages are declarative resource packages, not Python runtimes."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import pytest

from microcosm.build.country_spec import ALLOWED_COUNTRY_RESOURCE_KINDS
from microcosm.build.spec_engine import GENERATED_LOCK_FILENAMES, load_yaml12
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
COUNTRY_PACKAGE_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build"
SPEC_SUFFIXES = {".json", ".jsonld", ".yaml", ".yml"}
FORBIDDEN_EXECUTABLE_TOKENS = frozenset(
    {
        "callable",
        "callback",
        "entry",
        "entry_point",
        "entrypoint",
        "function",
        "handler",
        "import",
        "loader",
        "module",
        "python",
    }
)


def test__given_country_packages__then_no_python_files_are_present() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders = [
        path.relative_to(ROOT).as_posix()
        for root in country_roots
        for path in root.rglob("*.py")
    ]

    # Then
    assert offenders == []


def test__given_country_packages__then_only_spec_files_are_present() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders = [
        path.relative_to(ROOT).as_posix()
        for root in country_roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix not in SPEC_SUFFIXES
    ]

    # Then
    assert offenders == []


def test__given_country_package_manifests__then_resources_are_local_specs() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders: list[str] = []
    for root in country_roots:
        country = root.name
        manifest_path = root / "country_package.json"
        if not manifest_path.exists():
            offenders.append(f"{country}: missing country_package.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("country") != country:
            offenders.append(f"{country}: country mismatch")
        resource_rows = _resource_rows(manifest, country, offenders)
        resources = {path for path, _, _ in resource_rows}
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        generated_locks = actual_files & GENERATED_LOCK_FILENAMES
        expected_files = resources | {"country_package.json"} | generated_locks
        for generated_lock in sorted(resources & GENERATED_LOCK_FILENAMES):
            offenders.append(
                f"{country}: generated lock is an authored resource {generated_lock}"
            )
        for extra in sorted(actual_files - expected_files):
            offenders.append(f"{country}: unlisted resource {extra}")
        for resource, kind, schema_id in sorted(resource_rows):
            relative = PurePosixPath(resource)
            resource_path = root.joinpath(*relative.parts)
            if (
                relative.is_absolute()
                or resource != relative.as_posix()
                or ".." in relative.parts
            ):
                offenders.append(f"{country}: non-local resource {resource}")
            elif resource_path.suffix not in SPEC_SUFFIXES:
                offenders.append(f"{country}: non-spec resource {resource}")
            elif not resource_path.exists():
                offenders.append(f"{country}: missing resource {resource}")
            if kind not in ALLOWED_COUNTRY_RESOURCE_KINDS:
                offenders.append(f"{country}: unknown resource kind {kind}")
            if not schema_id:
                offenders.append(f"{country}: missing schema_id for {resource}")

    # Then
    assert offenders == []


def test__given_country_like_directories__then_each_has_a_manifest() -> None:
    # Given
    roots = [
        path
        for path in COUNTRY_PACKAGE_ROOT.iterdir()
        if path.is_dir() and re.fullmatch(r"[a-z]{2}", path.name)
    ]

    # When
    offenders = [
        root.relative_to(ROOT).as_posix()
        for root in roots
        if not (root / "country_package.json").exists()
    ]

    # Then
    assert offenders == []


def test__given_greenfield_country_packages__then_armenia_uses_the_typed_schema() -> (
    None
):
    # Given
    country_roots = {root.name: root for root in _country_package_roots()}

    # When
    armenia = country_roots["am"]
    manifest = json.loads(
        (armenia / "country_package.json").read_text(encoding="utf-8")
    )
    resource_kinds = {row["kind"] for row in manifest["resources"]}

    # Then
    assert {"am", "be"} <= set(country_roots)
    assert manifest["schema_version"] == 1
    assert resource_kinds == ALLOWED_COUNTRY_RESOURCE_KINDS & {
        "bundle",
        "catalogs",
        "geography",
        "legacy_json",
        "sources",
        "spine",
        "vintages",
    }


def test__given_greenfield_country_packages__then_new_zealand_uses_the_typed_schema() -> (
    None
):
    # Given
    country_roots = {root.name: root for root in _country_package_roots()}

    # When
    new_zealand = country_roots["nz"]
    manifest = json.loads(
        (new_zealand / "country_package.json").read_text(encoding="utf-8")
    )
    resource_kinds = {row["kind"] for row in manifest["resources"]}
    legacy_paths = {
        row["path"] for row in manifest["resources"] if row["kind"] == "legacy_json"
    }

    # Then
    assert manifest["schema_version"] == 1
    assert resource_kinds == ALLOWED_COUNTRY_RESOURCE_KINDS & {
        "bundle",
        "catalogs",
        "geography",
        "legacy_json",
        "sources",
        "spine",
        "vintages",
    }
    assert all(PurePosixPath(path).suffix == ".json" for path in legacy_paths)
    assert not (COUNTRY_PACKAGE_ROOT / "nz_runtime").exists()


def test__given_country_specs__then_no_python_entrypoints_are_declared() -> None:
    # Given
    specs = [
        path
        for country_root in _country_package_roots()
        for path in country_root.rglob("*")
        if path.is_file() and path.suffix in SPEC_SUFFIXES
    ]

    # When
    offenders: list[str] = []
    for path in specs:
        text = path.read_text(encoding="utf-8")
        payload = (
            json.loads(text)
            if path.suffix in {".json", ".jsonld"}
            else load_yaml12(text, source=path.relative_to(ROOT).as_posix())
        )
        _collect_executable_content(
            payload, path.relative_to(ROOT).as_posix(), offenders
        )

    # Then
    assert offenders == []


def test__given_schema_kernel_reference__then_it_is_not_an_entrypoint() -> None:
    assert not _looks_like_python_entrypoint("kernel:fit_qrf")


def _partition_fact_key_roster() -> dict:
    keys = [
        "ledger.aggregate_fact.v2:" + "a" * 24,
        "ledger.aggregate_fact.v2:" + "1" * 24,
    ]
    return {
        "role": "partition",
        "member_selectors": [{"aggregate_fact_key": key} for key in keys],
        "allocation_member_keys": keys.copy(),
    }


def _signed_partition_fact_key_roster() -> dict:
    partition = _partition_fact_key_roster()
    partition["discrepancy_policy"] = {
        "kind": "signed_proportional_reconciliation",
        "max_abs": 3,
    }
    partition["positive_allocation_member_keys"] = partition["allocation_member_keys"][
        :1
    ]
    return partition


@pytest.mark.parametrize(
    "shape",
    ["all_members", "recipient_subset", "signed", "signed_positive"],
)
def test__given_typed_partition_fact_keys__then_the_roster_is_data(shape) -> None:
    partition = _partition_fact_key_roster()
    if shape == "recipient_subset":
        partition["allocation_member_keys"] = partition["allocation_member_keys"][:1]
    elif shape.startswith("signed"):
        partition = _signed_partition_fact_key_roster()
        if shape == "signed":
            del partition["positive_allocation_member_keys"]
    offenders = []
    _collect_executable_content(partition, "reference.value_operands[2]", offenders)
    assert offenders == []


@pytest.mark.parametrize(
    "defect",
    [
        "mixed",
        "callable",
        "citation",
        "outside_partition",
        "wrong_role",
        "unpinned",
        "malformed_digest",
        "duplicate_roster",
        "positive_unsigned",
        "positive_two",
        "positive_outside",
    ],
)
def test__given_untyped_partition_keys__then_entrypoint_shapes_are_rejected(
    defect,
) -> None:
    partition = _partition_fact_key_roster()
    payload = partition
    expected = "reference.allocation_member_keys[0]"
    if defect == "mixed":
        partition["allocation_member_keys"].append("dotted.module:callable")
        expected = "reference.allocation_member_keys[2]"
    elif defect == "callable":
        partition["allocation_member_keys"] = ["dotted.module:callable"]
    elif defect == "citation":
        payload = {"partition": partition, "citation": "dotted.module:callable"}
        expected = "reference.citation"
    elif defect == "outside_partition":
        payload = {"allocation_member_keys": partition["allocation_member_keys"]}
    elif defect == "wrong_role":
        partition["role"] = "base"
    elif defect == "unpinned":
        partition["allocation_member_keys"] = ["ledger.aggregate_fact.v2:" + "b" * 24]
    elif defect == "malformed_digest":
        partition["allocation_member_keys"] = ["ledger.aggregate_fact.v2:callable"]
    elif defect == "duplicate_roster":
        partition["allocation_member_keys"] = [
            partition["allocation_member_keys"][0]
        ] * 2
    else:
        payload = partition = _signed_partition_fact_key_roster()
        expected = "reference.positive_allocation_member_keys[0]"
        if defect == "positive_unsigned":
            del partition["discrepancy_policy"]
        elif defect == "positive_two":
            partition["positive_allocation_member_keys"] = partition[
                "allocation_member_keys"
            ].copy()
        else:
            partition["positive_allocation_member_keys"] = [
                "ledger.aggregate_fact.v2:" + "b" * 24
            ]
    offenders = []
    _collect_executable_content(payload, "reference", offenders)
    assert expected in offenders


def _reconciliation_receipt() -> dict:
    keys = ["ledger.aggregate_fact.v2:" + letter * 24 for letter in "abc"]
    return {
        "leaf_selectors": [{"aggregate_fact_key": key} for key in keys[:2]],
        "total_selector": {"aggregate_fact_key": keys[2]},
        "raw_fact_keys": {"leaves": {"1": keys[0], "2": keys[1]}, "total": keys[2]},
    }


def test__given_reconciliation_raw_keys_copying_their_pins__then_they_are_data() -> (
    None
):
    offenders = []
    _collect_executable_content(_reconciliation_receipt(), "receipt", offenders)
    assert offenders == []


@pytest.mark.parametrize(
    "defect",
    ["callable_total", "unpinned_leaf", "outside_receipt", "extra_field"],
)
def test__given_untyped_raw_fact_keys__then_entrypoint_shapes_are_rejected(
    defect,
) -> None:
    receipt = _reconciliation_receipt()
    raw = receipt["raw_fact_keys"]
    payload = receipt
    expected = "reference.raw_fact_keys.leaves.1"
    if defect == "callable_total":
        raw["total"] = "dotted.module:callable"
        expected = "reference.raw_fact_keys.total"
    elif defect == "unpinned_leaf":
        raw["leaves"]["2"] = "ledger.aggregate_fact.v2:" + "d" * 24
        expected = "reference.raw_fact_keys.leaves.2"
    elif defect == "outside_receipt":
        payload = {"raw_fact_keys": raw}
    else:
        raw["citation"] = "dotted.module:callable"
        expected = "reference.raw_fact_keys.citation"
    offenders = []
    _collect_executable_content(payload, "reference", offenders)
    assert expected in offenders


def _collect_executable_content(
    value: object,
    location: str,
    offenders: list[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            declared_kernel_loader = (
                key == "loader"
                and isinstance(child, str)
                and child.startswith("kernel:")
                and not _looks_like_python_entrypoint(child)
            )
            compiler_identity_field = _is_compiler_identity_field(
                key=key,
                value=child,
                location=location,
            )
            if (
                _is_executable_key(key)
                and not declared_kernel_loader
                and not compiler_identity_field
            ):
                offenders.append(child_location)
            if (
                compiler_identity_field
                or _is_fact_key_field(key=key, value=child)
                or _is_partition_fact_key_roster(key=key, value=child, parent=value)
                or _is_reconciliation_raw_fact_keys(key=key, value=child, parent=value)
            ):
                continue
            _collect_executable_content(child, child_location, offenders)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_executable_content(child, f"{location}[{index}]", offenders)
    elif isinstance(value, str) and _looks_like_python_entrypoint(value):
        offenders.append(location)


def _country_package_roots() -> list[Path]:
    return sorted(
        path.parent for path in COUNTRY_PACKAGE_ROOT.glob("*/country_package.json")
    )


def _resource_rows(
    manifest: object,
    country: str,
    offenders: list[str],
) -> list[tuple[str, str, str]]:
    if not isinstance(manifest, dict):
        offenders.append(f"{country}: manifest is not an object")
        return []
    raw_resources = manifest.get("resources", ())
    if not isinstance(raw_resources, list):
        offenders.append(f"{country}: resources is not a list")
        return []

    rows: list[tuple[str, str, str]] = []
    for index, raw in enumerate(raw_resources):
        if isinstance(raw, str):
            # Generation-0 manifests are resolved to the same explicit row
            # shape by CountrySpec; package tests retain their compatibility.
            rows.append((raw, "legacy_json", "legacy_json"))
            continue
        if not isinstance(raw, dict):
            offenders.append(f"{country}: resource row {index} is not an object")
            continue
        if set(raw) != {"path", "kind", "schema_id"}:
            offenders.append(
                f"{country}: resource row {index} does not have exactly "
                "path/kind/schema_id"
            )
            continue
        path = raw["path"]
        kind = raw["kind"]
        schema_id = raw["schema_id"]
        if not all(isinstance(value, str) for value in (path, kind, schema_id)):
            offenders.append(f"{country}: resource row {index} has non-string fields")
            continue
        rows.append((path, kind, schema_id))
    return rows


def _is_executable_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized in FORBIDDEN_EXECUTABLE_TOKENS:
        return True
    tokens = normalized.split("_")
    return any(token in FORBIDDEN_EXECUTABLE_TOKENS for token in tokens)


def _looks_like_python_entrypoint(value: str) -> bool:
    if "://" in value:
        return False
    return bool(
        re.search(
            r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*\b",
            value,
        )
    )


_FACT_KEY_FIELDS = frozenset(
    {"aggregate_fact_key", "semantic_fact_key", "legacy_fact_key", "fact_key"}
)
_FACT_KEY_NAMESPACES = (
    "ledger.",
    "microcosm.derived_fact.",
    "microcosm.semantic_fact.",
)


def _is_fact_key_field(*, key: str, value: object) -> bool:
    """Admit consumer-fact identifiers in their published ``namespace:id`` shape.

    Chronicle keys its facts ``ledger.aggregate_fact.v2:<id>`` and Microcosm's
    derived facts (microcosm#1240) follow the same shape under the owned
    ``microcosm.derived_fact.`` and ``microcosm.semantic_fact.`` namespaces. They are opaque identifiers the
    compiler matches by prefix, never entrypoints; only the named key fields
    with a known namespace are admitted.
    """

    return (
        key in _FACT_KEY_FIELDS
        and isinstance(value, str)
        and value.startswith(_FACT_KEY_NAMESPACES)
    )


_AGGREGATE_FACT_KEY = r"ledger\.aggregate_fact\.v\d+:[0-9a-f]{24}"


def _is_partition_fact_key_roster(*, key: str, value: object, parent: dict) -> bool:
    """Admit only canonical recipient keys pinned by a typed atomic partition.

    The fields mirror the compiler's partition operand
    (``ledger_targets._validate_allocation_reference``): a signed partition adds
    ``discrepancy_policy`` and may name one positive recipient.
    """
    fields = {"role", "member_selectors", "allocation_member_keys"}
    if "discrepancy_policy" in parent:
        fields.add("discrepancy_policy")
        if "positive_allocation_member_keys" in parent:
            fields.add("positive_allocation_member_keys")
    if (
        key not in {"allocation_member_keys", "positive_allocation_member_keys"}
        or set(parent) != fields
        or parent.get("role") != "partition"
        or not isinstance(value, list)
        or not value
        or (key == "positive_allocation_member_keys" and len(value) != 1)
    ):
        return False
    if any(
        not isinstance(item, str) or re.fullmatch(_AGGREGATE_FACT_KEY, item) is None
        for item in value
    ):
        return False
    members = parent["member_selectors"]
    if not isinstance(members, list) or not members:
        return False
    keys = []
    for member in members:
        if not isinstance(member, dict):
            return False
        identity = member.get("aggregate_fact_key")
        if (
            not isinstance(identity, str)
            or re.fullmatch(_AGGREGATE_FACT_KEY, identity) is None
        ):
            return False
        keys.append(identity)
    return (
        len(set(keys)) == len(keys)
        and len(set(value)) == len(value)
        and set(value) <= set(keys)
    )


def _is_reconciliation_raw_fact_keys(*, key: str, value: object, parent: dict) -> bool:
    """Admit a reconciliation receipt's raw keys only as copies of its own pins.

    Each leaf key must equal the matching ``leaf_selectors`` pin (numbered from
    1) and the total must equal ``total_selector``'s, so the record names no
    string its selectors do not already pin.
    """
    if (
        key != "raw_fact_keys"
        or not isinstance(value, dict)
        or set(value) != {"leaves", "total"}
    ):
        return False
    selectors = parent.get("leaf_selectors")
    total = parent.get("total_selector")
    if (
        not isinstance(selectors, list)
        or not selectors
        or not all(isinstance(selector, dict) for selector in selectors)
        or not isinstance(total, dict)
    ):
        return False
    pinned = {
        str(code): selector.get("aggregate_fact_key")
        for code, selector in enumerate(selectors, 1)
    }
    keys = [*pinned.values(), total.get("aggregate_fact_key")]
    return (
        value["leaves"] == pinned
        and value["total"] == keys[-1]
        and all(
            isinstance(item, str) and re.fullmatch(_AGGREGATE_FACT_KEY, item)
            for item in keys
        )
    )


def _is_compiler_identity_field(*, key: str, value: object, location: str) -> bool:
    """Admit inert legacy-identity fields inside the typed compiler surface.

    F0 must reproduce the generation-0 resource-semantics receipt exactly, so
    the imputation IR carries its historical ``module``/``callable`` labels.
    They are data consumed by the constants adapter, never country entrypoints:
    only a node's schema-bound virtual-resource record may contain them.  The
    narrow location check preserves this test's refusal everywhere else.
    """

    virtual_resource_label = (
        key in {"module", "callable"}
        and isinstance(value, str)
        and bool(value)
        and re.search(r"\.virtual_resources(?:\.|\[)", location) is not None
    )
    generated_lock_count = (
        key == "entry_count"
        and type(value) is int
        and location.endswith(
            "engine_abi.lock.json.remaining_stage_input_manifest.receipt"
        )
    )
    # A Ledger fact identity (``ledger.aggregate_fact.v2:<hex>``) is content
    # addressing data, never an entrypoint; the colon separates the key kind
    # from its digest and the digest can happen to start with a letter. A
    # diagnostic reference pins the same identity as its ``ledger_fact_key``.
    ledger_fact_identity = (
        key in {"aggregate_fact_key", "ledger_fact_key"}
        and isinstance(value, str)
        and re.fullmatch(r"ledger\.[a-z_]+\.v\d+:[0-9a-f]{24}", value) is not None
    )
    return virtual_resource_label or generated_lock_count or ledger_fact_identity
