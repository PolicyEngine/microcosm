"""Metadata-only v2 contracts; dummy artifacts never qualify real source data.

Every digest below is computed from MOCK_METADATA_ONLY bytes. These tests
exercise map/schema behavior without H5, an engine or an original-worker run.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.data import source_parent_replay as replay

CHILD_KEY = "mock_metadata_only_child"
CHILD_FILE = f"{CHILD_KEY}.h5"
ROW_FIELDS = {
    "source_key",
    "source_path",
    "carried_key",
    "carried_path",
    "kind",
    "sha256",
}


def _manifest(extra_names=()):
    """Use real hashes of dummy bytes, never identities of a real release."""
    paths = {
        "mock_metadata_only_source": ("mock_metadata_only_source.h5", "microdata"),
        "parent_release_manifest": ("parent_release_manifest.json", "diagnostics"),
        "parent_build_manifest": ("parent_build_manifest.json", "diagnostics"),
        "build_manifest": ("build_manifest.json", "diagnostics"),
        "mock_source_report": ("mock_source_report.json", "diagnostics"),
        **{name: (f"{name}.json", "diagnostics") for name in extra_names},
    }
    return {
        "schema_version": 1,
        "default_datasets": {"national": "mock_metadata_only_source"},
        "artifacts": {
            key: {
                "kind": kind,
                "path": path,
                "repo_id": "MOCK_METADATA_ONLY/not-a-published-dataset",
                "sha256": hashlib.sha256(
                    f"MOCK_METADATA_ONLY:{key}".encode()
                ).hexdigest(),
            }
            for key, (path, kind) in paths.items()
        },
    }


def _map(manifest):
    return replay.build_source_parent_artifact_map(
        manifest, child_dataset_key=CHILD_KEY, child_dataset_filename=CHILD_FILE
    )


def _verify(manifest, mapping):
    replay.verify_source_parent_artifact_map(
        manifest,
        mapping,
        child_dataset_key=CHILD_KEY,
        child_dataset_filename=CHILD_FILE,
    )


def test_two_generations_have_a_complete_injective_inverse_map():
    manifest = _manifest()
    before = copy.deepcopy(manifest)
    mapping = _map(manifest)
    expected = {
        "mock_metadata_only_source": (CHILD_KEY, CHILD_FILE),
        "parent_release_manifest": (
            "ancestor_release_manifest",
            "ancestor_release_manifest.json",
        ),
        "parent_build_manifest": (
            "ancestor_build_manifest",
            "ancestor_build_manifest.json",
        ),
        "build_manifest": ("parent_build_manifest", "parent_build_manifest.json"),
        "mock_source_report": ("mock_source_report", "mock_source_report.json"),
    }
    assert len(mapping) == len(manifest["artifacts"])
    assert all(set(row) == ROW_FIELDS for row in mapping)
    assert {
        row["source_key"]: (row["carried_key"], row["carried_path"]) for row in mapping
    } == expected
    assert len({row["carried_key"] for row in mapping}) == len(mapping)
    assert len({row["carried_path"] for row in mapping}) == len(mapping)
    inverse = {
        row["source_key"]: {
            "path": row["source_path"],
            "kind": row["kind"],
            "sha256": row["sha256"],
        }
        for row in mapping
    }
    assert inverse == {
        key: {field: entry[field] for field in ("path", "kind", "sha256")}
        for key, entry in manifest["artifacts"].items()
    }
    _verify(manifest, mapping)
    assert manifest == before


@given(
    st.sets(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).map(
            lambda value: f"mock_extra_{value}"
        ),
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_additional_safe_artifacts_round_trip_without_loss(extra_names):
    manifest = _manifest(sorted(extra_names))
    mapping = _map(manifest)
    _verify(manifest, mapping)
    assert {row["source_key"] for row in mapping} == set(manifest["artifacts"])
    for row in mapping:
        entry = manifest["artifacts"][row["source_key"]]
        assert (row["source_path"], row["kind"], row["sha256"]) == (
            entry["path"],
            entry["kind"],
            entry["sha256"],
        )
    assert len({row["carried_key"] for row in mapping}) == len(mapping)
    assert len({row["carried_path"] for row in mapping}) == len(mapping)


@pytest.mark.parametrize(
    "path",
    ["../outside.json", "/outside.json", "a/b.json", "a\\b.json", ".", "..", ""],
)
def test_map_rejects_unsafe_original_paths(path):
    manifest = _manifest()
    manifest["artifacts"]["mock_source_report"]["path"] = path
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize("axis", ["key", "path"])
def test_map_rejects_destination_collisions(axis):
    manifest = _manifest()
    key = "ancestor_release_manifest" if axis == "key" else "mock_collision"
    manifest["artifacts"][key] = {
        **manifest["artifacts"]["mock_source_report"],
        "path": "mock_collision.json"
        if axis == "key"
        else "ancestor_release_manifest.json",
    }
    with pytest.raises(ValueError):
        _map(manifest)


def test_map_rejects_duplicate_original_filenames():
    manifest = _manifest()
    manifest["artifacts"]["mock_duplicate"] = dict(
        manifest["artifacts"]["mock_source_report"]
    )
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_key", "mock_unknown"),
        ("source_path", "mock_other.json"),
        ("carried_key", "mock_other"),
        ("carried_path", "mock_other.json"),
        ("kind", "mock_other_kind"),
        ("sha256", hashlib.sha256(b"MOCK_METADATA_ONLY:tampered").hexdigest()),
    ],
)
def test_verifier_binds_every_row_field(field, value):
    manifest = _manifest()
    mapping = [dict(row) for row in _map(manifest)]
    mapping[0][field] = value
    with pytest.raises(ValueError):
        _verify(manifest, mapping)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "unknown_field"])
def test_verifier_refuses_incomplete_or_ambiguous_maps(mutation):
    manifest = _manifest()
    mapping = [dict(row) for row in _map(manifest)]
    if mutation == "missing":
        mapping.pop()
    elif mutation == "extra":
        mapping.append({**mapping[0], "source_key": "mock_unknown"})
    elif mutation == "duplicate":
        mapping.append(dict(mapping[0]))
    else:
        mapping[0]["ignored"] = "not permitted"
    with pytest.raises(ValueError):
        _verify(manifest, mapping)


def _config(tmp_path):
    return {
        "original_python": str(tmp_path / "MOCK_METADATA_ONLY_python"),
        "original_checkout": str(tmp_path / "MOCK_METADATA_ONLY_checkout"),
        "original_wheels": [
            str(tmp_path / f"MOCK_METADATA_ONLY_{i}.whl") for i in range(4)
        ],
        "ancestor_h5": str(tmp_path / "MOCK_METADATA_ONLY_NOT_HDF5.h5"),
        "worker_timeout_seconds": 7.0,
    }


def test_context_round_trip_is_local_configuration_only(tmp_path):
    config = _config(tmp_path)
    context = replay.SourceParentReplayContext.from_dict(config)
    assert context.to_dict() == config
    path = tmp_path / "MOCK_METADATA_ONLY_context.json"
    path.write_text(json.dumps(config))
    assert replay.SourceParentReplayContext.from_json(path) == context


@pytest.mark.parametrize(
    "field", ["worker", "callback", "profile", "passed", "ignore_direct_url"]
)
def test_context_refuses_unknown_bypass_configuration(tmp_path, field):
    config = _config(tmp_path)
    config[field] = "MOCK_METADATA_ONLY"
    with pytest.raises(ValueError):
        replay.SourceParentReplayContext.from_dict(config)


@pytest.mark.parametrize(
    "field", ["original_python", "original_checkout", "original_wheels", "ancestor_h5"]
)
def test_context_refuses_missing_required_local_inputs(tmp_path, field):
    config = _config(tmp_path)
    del config[field]
    with pytest.raises(ValueError):
        replay.SourceParentReplayContext.from_dict(config)


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), "7"])
def test_context_requires_a_finite_positive_numeric_timeout(tmp_path, timeout):
    config = _config(tmp_path)
    config["worker_timeout_seconds"] = timeout
    with pytest.raises(ValueError):
        replay.SourceParentReplayContext.from_dict(config)


@pytest.mark.parametrize(
    "field", ["original_python", "original_checkout", "ancestor_h5"]
)
def test_context_refuses_relative_local_paths(tmp_path, field):
    config = _config(tmp_path)
    config[field] = "relative/MOCK_METADATA_ONLY"
    with pytest.raises(ValueError):
        replay.SourceParentReplayContext.from_dict(config)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "relative", "string"])
def test_context_requires_four_distinct_absolute_wheel_paths(tmp_path, mutation):
    config = _config(tmp_path)
    if mutation == "missing":
        config["original_wheels"].pop()
    elif mutation == "duplicate":
        config["original_wheels"][1] = config["original_wheels"][0]
    elif mutation == "relative":
        config["original_wheels"][0] = "MOCK_METADATA_ONLY.whl"
    else:
        config["original_wheels"] = "MOCK_METADATA_ONLY.whl"
    with pytest.raises(ValueError):
        replay.SourceParentReplayContext.from_dict(config)


def test_context_json_refuses_duplicate_fields(tmp_path):
    config = _config(tmp_path)
    encoded = json.dumps(config)
    duplicate = '{"original_python":"MOCK_METADATA_ONLY",' + encoded[1:]
    path = tmp_path / "MOCK_METADATA_ONLY_duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        replay.SourceParentReplayContext.from_json(path)


@pytest.mark.parametrize("field", ["child_dataset_key", "child_dataset_filename"])
@pytest.mark.parametrize(
    "value", ["../outside", "/outside", "a/b", "a\\b", "", ".", ".."]
)
def test_child_namespace_cannot_escape_or_alias(field, value):
    options = {"child_dataset_key": CHILD_KEY, "child_dataset_filename": CHILD_FILE}
    options[field] = value
    with pytest.raises(ValueError):
        replay.build_source_parent_artifact_map(_manifest(), **options)


@pytest.mark.parametrize(
    "key,path",
    [
        ("source_parent_evidence_v2", "mock_forbidden.json"),
        ("compatibility_recertification", "mock_forbidden.json"),
        ("release_manifest", "mock_forbidden.json"),
        ("mock_unknown_alias", "parent_release_manifest.json"),
        ("mock_unknown_alias", "parent_build_manifest.json"),
        ("mock_unknown_alias", "build_manifest.json"),
    ],
)
def test_parent_cannot_claim_child_or_ancestor_reserved_names(key, path):
    manifest = _manifest()
    manifest["artifacts"][key] = {
        "kind": "diagnostics",
        "path": path,
        "sha256": hashlib.sha256(b"MOCK_METADATA_ONLY:reserved").hexdigest(),
    }
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize("kind", [None, ""])
def test_map_requires_an_explicit_artifact_kind(kind):
    manifest = _manifest()
    manifest["artifacts"]["mock_source_report"]["kind"] = kind
    with pytest.raises(ValueError):
        _map(manifest)


def test_map_cannot_lose_the_selected_source_dataset():
    manifest = _manifest()
    del manifest["artifacts"]["mock_metadata_only_source"]
    with pytest.raises(ValueError):
        _map(manifest)


def _evidence(tmp_path, monkeypatch):
    from test_support.microcosm_data.source_parent_replay import (
        make_mock_source_parent,
        mock_current_record,
    )

    source = make_mock_source_parent(tmp_path)
    monkeypatch.setattr(replay, "_reviewed_profile", lambda: source.profile)
    manifest = json.loads((source.release_dir / "release_manifest.json").read_text())
    record = {
        "schema_version": 2,
        "kind": "source_parent_replay",
        "policy": "exact_legacy_receipt",
        "profile": dict(source.profile),
        "artifact_map": list(_map(manifest)),
        "legacy_actual": copy.deepcopy(source.legacy),
        "legacy_runtime": dict(source.runtime),
        "current": mock_current_record(source),
    }
    # Establish a valid schema baseline before any negative test mutates it.
    assert replay.SourceParentEvidenceV2.from_dict(record).to_dict() == record
    return record


def test_evidence_is_detached_deeply_frozen_json(tmp_path, monkeypatch):
    from dataclasses import FrozenInstanceError

    record = _evidence(tmp_path, monkeypatch)
    original = copy.deepcopy(record)
    evidence = replay.SourceParentEvidenceV2.from_dict(record)
    record["legacy_actual"]["checks"].append("MOCK_METADATA_ONLY_external_mutation")
    record["current"]["producer_code"]["git_dirty"] = True
    assert evidence.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        evidence.policy = "MOCK_METADATA_ONLY_bypass"
    with pytest.raises(TypeError):
        evidence.current["producer_code"]["git_dirty"] = True
    detached = evidence.to_dict()
    detached["legacy_actual"]["checks"].clear()
    assert evidence.to_dict() == original


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "kind",
        "policy",
        "profile",
        "artifact_map",
        "legacy_actual",
        "legacy_runtime",
        "current",
    ],
)
def test_evidence_requires_every_envelope_field(tmp_path, monkeypatch, field):
    record = _evidence(tmp_path, monkeypatch)
    del record[field]
    with pytest.raises(ValueError):
        replay.SourceParentEvidenceV2.from_dict(record)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top",
        "wrong_schema",
        "bool_schema",
        "legacy_policy",
        "wrong_kind",
        "nan",
    ],
)
def test_evidence_rejects_unknown_schemas_and_ignored_values(
    tmp_path, monkeypatch, mutation
):
    record = _evidence(tmp_path, monkeypatch)
    if mutation == "unknown_top":
        record["ignored"] = "MOCK_METADATA_ONLY"
    elif mutation == "wrong_schema":
        record["schema_version"] = 1
    elif mutation == "bool_schema":
        record["schema_version"] = True
    elif mutation == "legacy_policy":
        record["policy"] = "ignore_direct_url"
    elif mutation == "wrong_kind":
        record["kind"] = "compatibility_recertification"
    else:
        hashes = record["current"]["producer_code"]["source_files_sha256"]
        hashes["ignored"] = float("nan")
    with pytest.raises(ValueError):
        replay.SourceParentEvidenceV2.from_dict(record)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "kind",
        "profile",
        "checks",
        "native_receipt",
        "compatibility",
        "producer_code",
    ],
)
def test_evidence_requires_every_current_field(tmp_path, monkeypatch, field):
    record = _evidence(tmp_path, monkeypatch)
    del record["current"][field]
    with pytest.raises(ValueError):
        replay.SourceParentEvidenceV2.from_dict(record)


@pytest.mark.parametrize("check_index", range(7))
def test_no_current_qualification_check_can_be_omitted(
    tmp_path, monkeypatch, check_index
):
    record = _evidence(tmp_path, monkeypatch)
    record["current"]["checks"].pop(check_index)
    with pytest.raises(ValueError):
        replay.SourceParentEvidenceV2.from_dict(record)


@pytest.mark.parametrize("field", sorted(ROW_FIELDS))
def test_evidence_map_rows_do_not_drop_fields(tmp_path, monkeypatch, field):
    record = _evidence(tmp_path, monkeypatch)
    del record["artifact_map"][0][field]
    with pytest.raises(ValueError):
        replay.SourceParentEvidenceV2.from_dict(record)


# --- A3: case/Unicode-normalization aliases (review finding A3) -------------
#
# Expected semantics come from the written repair plan: destination and source
# identities collide when their NFC-normalized, casefolded forms are equal
# (normalized again after casefolding), because case-insensitive or
# normalization-insensitive filesystems such as APFS store them as one name.
# Accepted names keep their exact original strings; nothing is renamed.
# Every vector is MOCK_METADATA_ONLY metadata; no artifact bytes are read.

_NFC_CAFE = "café"
_NFD_CAFE = "café"


def _with_artifacts(*entries):
    manifest = _manifest()
    for key, path in entries:
        manifest["artifacts"][key] = {
            "kind": "diagnostics",
            "path": path,
            "sha256": hashlib.sha256(
                f"MOCK_METADATA_ONLY:{key}:{path}".encode()
            ).hexdigest(),
        }
    return manifest


@pytest.mark.parametrize(
    "first,second",
    [
        ("Foo.json", "foo.json"),
        ("MOCK_REPORT.json", "mock_report.json"),
        (f"{_NFC_CAFE}.json", f"{_NFD_CAFE}.json"),
        (f"{_NFC_CAFE.upper()}.json", f"{_NFD_CAFE}.json"),
        ("straße.json", "STRASSE.json"),
        ("Kelvin.json", "kelvin.json"),
        ("ǰ.json", "J̌.json"),
    ],
    ids=["case", "upper", "nfc_nfd", "upper_nfd", "sharp_s", "kelvin", "j_caron"],
)
def test_map_refuses_source_path_case_and_normalization_aliases(first, second):
    assert first != second
    manifest = _with_artifacts(("mock_alias_a", first), ("mock_alias_b", second))
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize(
    "first,second",
    [
        ("mock_Extra", "mock_extra"),
        (f"mock_{_NFC_CAFE}", f"mock_{_NFD_CAFE}"),
        (f"MOCK_{_NFC_CAFE.upper()}", f"mock_{_NFD_CAFE}"),
    ],
    ids=["case", "nfc_nfd", "upper_nfd"],
)
def test_map_refuses_source_and_carried_key_aliases(first, second):
    manifest = _with_artifacts(
        (first, "mock_alias_a.json"), (second, "mock_alias_b.json")
    )
    with pytest.raises(ValueError):
        _map(manifest)


_RESERVED_KEY_ALIASES = (
    "Release_Manifest",
    "RELEASE_MANIFEST",
    "Compatibility_Recertification",
    "Compatibility_Recertification_Loader",
    "Compatibility_Drift",
    "Source_Parent_Evidence_V2",
    "Ancestor_Release_Manifest",
    "ANCESTOR_BUILD_MANIFEST",
    # Carried key of the reviewed build_manifest row and the child dataset key.
    "Parent_Build_Manifest",
    CHILD_KEY.upper(),
)


@pytest.mark.parametrize("key", _RESERVED_KEY_ALIASES)
def test_map_refuses_reserved_or_mapped_key_aliases(key):
    manifest = _with_artifacts((key, "mock_reserved_key_alias.json"))
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize(
    "path",
    [
        "Release_Manifest.json",
        "RELEASE_MANIFEST.JSON",
        "Ancestor_Release_Manifest.json",
        "ancestor_build_manifest.JSON",
        "Source_Parent_Evidence_V2.json",
        "Compatibility_Recertification.json",
        "Parent_Release_Manifest.json",
        "PARENT_BUILD_MANIFEST.json",
        "Build_Manifest.json",
        CHILD_FILE.upper(),
        "Mock_Source_Report.json",
    ],
)
def test_map_refuses_reserved_or_mapped_path_aliases(path):
    manifest = _with_artifacts(("mock_reserved_path_alias", path))
    with pytest.raises(ValueError):
        _map(manifest)


@pytest.mark.parametrize(
    "key,path",
    [
        ("MICROCOSM_US_2024", "mock_alias.json"),
        ("mock_alias", "MICROCOSM_US_2024.h5"),
        ("mock_alias", "Microcosm_US_2024.H5"),
    ],
)
def test_production_child_dataset_identity_cannot_be_aliased(key, path):
    manifest = _with_artifacts((key, path))
    with pytest.raises(ValueError):
        replay.build_source_parent_artifact_map(
            manifest,
            child_dataset_key="microcosm_us_2024",
            child_dataset_filename="microcosm_us_2024.h5",
        )


@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
    st.data(),
)
@settings(max_examples=40, deadline=None)
def test_any_case_variant_of_an_accepted_name_is_refused(stem, data):
    base = f"mock_variant_{stem}"
    flips = data.draw(
        st.lists(st.booleans(), min_size=len(base), max_size=len(base)).filter(any)
    )
    variant = "".join(
        char.upper() if flip else char for char, flip in zip(base, flips, strict=True)
    )
    if variant == base:
        return
    accepted = _with_artifacts((base, f"{base}.json"))
    _map(accepted)
    for entries in (
        ((base, f"{base}.json"), (f"{base}_other", f"{variant}.json")),
        ((base, f"{base}.json"), (variant, f"{base}_other.json")),
    ):
        with pytest.raises(ValueError):
            _map(_with_artifacts(*entries))


@pytest.mark.parametrize(
    "key,path",
    [
        (f"mock_{_NFD_CAFE}", f"mock_{_NFD_CAFE}.json"),
        (f"mock_{_NFC_CAFE}", f"MOCK_{_NFC_CAFE.upper()}.json"),
        ("Mock_Mixed_Case", "Mock_Mixed_Case.json"),
    ],
)
def test_distinct_non_ascii_or_mixed_case_names_are_kept_exactly(key, path):
    manifest = _with_artifacts((key, path))
    before = copy.deepcopy(manifest)
    mapping = _map(manifest)
    (row,) = [row for row in mapping if row["source_key"] == key]
    # Evidence strings are carried byte-for-byte, never normalized or renamed.
    assert row["source_path"].encode() == path.encode()
    assert (row["carried_key"], row["carried_path"]) == (key, path)
    assert row["carried_key"].encode() == key.encode()
    _verify(manifest, mapping)
    assert manifest == before


# --- A1: installed package tree equals its authenticated wheel inventory -----
#
# A synthetic MOCK_METADATA_ONLY distribution on a test-local sys.path entry,
# never a real or importable model package (nothing below imports it). Expected
# semantics come from the written plan: the installed disk tree must equal the
# wheel/RECORD inventory, symlinks anywhere are refused, sourceless bytecode is
# refused even when recorded, and the import name must resolve to that tree.

_MOCK_IMPORT = "mock_metadata_only_installed_pkg"
_MOCK_DIST_INFO = f"{_MOCK_IMPORT}-0.0.0.dist-info"


def _mock_installed_tree(tmp_path, monkeypatch, *, extra_members=()):
    from importlib import metadata

    site = tmp_path / "MOCK_METADATA_ONLY_site"
    members = {
        f"{_MOCK_IMPORT}/__init__.py": b"MOCK = 'MOCK_METADATA_ONLY'\n",
        f"{_MOCK_IMPORT}/variables/__init__.py": b"",
        f"{_MOCK_IMPORT}/variables/mock_variable.py": b"VALUE = 1\n",
        f"{_MOCK_IMPORT}/mock_resource.yaml": b"mock: MOCK_METADATA_ONLY\n",
        f"{_MOCK_DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.1\nName: mock-metadata-only-installed-pkg\n"
            b"Version: 0.0.0\n"
        ),
        f"{_MOCK_DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n",
        **dict(extra_members),
    }
    for relative, content in members.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    record = [f"{relative},," for relative in members]
    record += [f"{_MOCK_DIST_INFO}/RECORD,,", f"{_MOCK_DIST_INFO}/INSTALLER,,"]
    (site / _MOCK_DIST_INFO / "RECORD").write_text("\n".join(record) + "\n")
    (site / _MOCK_DIST_INFO / "INSTALLER").write_text("MOCK_METADATA_ONLY\n")
    monkeypatch.syspath_prepend(str(site))
    dist = metadata.PathDistribution(site / _MOCK_DIST_INFO)
    return site, dist, members


def test_installed_tree_matching_its_wheel_authenticates(tmp_path, monkeypatch):
    site, dist, members = _mock_installed_tree(tmp_path, monkeypatch)
    root, actual_site, recorded = replay._authenticate_installed_tree(
        dist, members, import_name=_MOCK_IMPORT
    )
    assert (root, actual_site) == (site / _MOCK_IMPORT, site)
    assert set(recorded) == set(members)


_TREE_MUTATIONS = (
    "unrecorded_module",
    "unrecorded_resource",
    "symlinked_subdirectory",
    "symlinked_empty_directory",
    "symlinked_recorded_file",
    "symlinked_package_root",
    "changed_recorded_bytes",
    "missing_recorded_file",
    "wheel_member_not_installed",
    "sourceless_bytecode",
    "second_top_level_root",
    "shadowed_import_name",
)


@pytest.mark.parametrize("mutation", _TREE_MUTATIONS)
def test_installed_tree_differences_refuse(tmp_path, monkeypatch, mutation):
    import py_compile
    import shutil

    extra = {}
    if mutation == "sourceless_bytecode":
        source = tmp_path / "MOCK_METADATA_ONLY_planted.py"
        source.write_text("VALUE = 'MOCK_METADATA_ONLY planted'\n")
        compiled = tmp_path / "MOCK_METADATA_ONLY_planted.pyc"
        py_compile.compile(str(source), cfile=str(compiled), doraise=True)
        # Recorded and in the wheel: a sourceless module must still refuse.
        extra[f"{_MOCK_IMPORT}/variables/mock_planted.pyc"] = compiled.read_bytes()
    elif mutation == "second_top_level_root":
        extra["mock_metadata_only_second_root/__init__.py"] = b""
    site, dist, members = _mock_installed_tree(
        tmp_path, monkeypatch, extra_members=extra
    )
    package = site / _MOCK_IMPORT
    outside = tmp_path / "MOCK_METADATA_ONLY_outside"
    outside.mkdir()
    if mutation == "unrecorded_module":
        (package / "variables/mock_unrecorded.py").write_text("VALUE = 2\n")
    elif mutation == "unrecorded_resource":
        (package / "variables/mock_unrecorded.yaml").write_text("mock: 2\n")
    elif mutation == "symlinked_subdirectory":
        (outside / "mock_escape.py").write_text("VALUE = 3\n")
        (package / "mock_linked").symlink_to(outside)
    elif mutation == "symlinked_empty_directory":
        (package / "mock_linked_empty").symlink_to(outside)
    elif mutation == "symlinked_recorded_file":
        target = package / "variables/mock_variable.py"
        (outside / "mock_variable.py").write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside / "mock_variable.py")
    elif mutation == "symlinked_package_root":
        moved = outside / _MOCK_IMPORT
        shutil.move(str(package), moved)
        package.symlink_to(moved)
    elif mutation == "changed_recorded_bytes":
        (package / "variables/mock_variable.py").write_bytes(b"VALUE = 999\n")
    elif mutation == "missing_recorded_file":
        (package / "mock_resource.yaml").unlink()
    elif mutation == "wheel_member_not_installed":
        members = {**members, f"{_MOCK_IMPORT}/mock_wheel_only.py": b"VALUE = 4\n"}
    elif mutation == "shadowed_import_name":
        shadow = tmp_path / "MOCK_METADATA_ONLY_shadow"
        (shadow / _MOCK_IMPORT).mkdir(parents=True)
        (shadow / _MOCK_IMPORT / "__init__.py").write_text("SHADOW = True\n")
        monkeypatch.syspath_prepend(str(shadow))
    with pytest.raises(ValueError):
        replay._authenticate_installed_tree(dist, members, import_name=_MOCK_IMPORT)


def test_installed_tree_refuses_a_different_expected_import_name(tmp_path, monkeypatch):
    _, dist, members = _mock_installed_tree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        replay._authenticate_installed_tree(
            dist, members, import_name="mock_metadata_only_other_pkg"
        )
