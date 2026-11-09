"""Offline registration of the preserved ACS local-release staging builder.

The real builder's parser is extracted from its source without importing a
country engine. Tiny staged files exercise archive-cache links, the Census
fetcher's cache hit, the runner, and output receipts without Modal or network.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    COMMIT,
    ROOT,
    load_app,
    plan_lib,
)

TOOL = plan_lib.US_ACS_MULTISPINE_STAGING
PLAN_PATH = ROOT / "docs" / "us-modal-stage-acs-multispine-staging-plan.json"
LEGACY_SCRIPT = "tools/_legacy/build_us_acs_multispine_base.py"
MANIFEST_PATH = (
    ROOT
    / "packages/microcosm-build/src/microcosm/build/us_runtime"
    / "acs_2024_1yr_sources.json"
)
OPTIONS = {
    "acs_share": 0.5,
    "n_estimators": 32,
    "max_targets_per_fit": 8,
    "seed": 0,
    "geography_seed": 0,
    "chunksize": 100_000,
    "donor_channel": "auto",
    "hours_under15_policy": "us_hours_under15_zero_completion_v1",
    "max_households": 25,
    "period": 2024,
}
OWNED_FLAGS = {
    "--base-h5",
    "--out-h5",
    "--summary",
    "--source-manifest",
    "--inputs-dir",
    "--puma-ladder",
    "--donor-release-manifest",
    "--sipp-donor",
}


@pytest.fixture
def app(monkeypatch):
    return load_app(monkeypatch)


def _plan_data(**overrides) -> dict:
    pins = {link.input: link.sha256 for link in TOOL.input_links}
    pins.update({seed.input: seed.sha256 for seed in TOOL.home_seeds})
    names = {
        "base_h5": "populace_us_2024.h5",
        "donor_release_manifest": "release_manifest.json",
        "acs_household_zip": "csv_hus.zip",
        "acs_person_zip": "csv_pus.zip",
        "puma_ladder": "us_puma_ladder_2020.npz",
    }
    names.update({seed.input: Path(seed.path).name for seed in TOOL.home_seeds})
    inputs = {}
    for index, role in enumerate(TOOL.inputs):
        digest = pins.get(role, hashlib.sha256(role.encode()).hexdigest())
        inputs[role] = {
            "uri": f"volume://cas/sha256/{digest}/{names.get(role, str(index))}",
            "sha256": digest,
        }
    data = {
        "schema": plan_lib.PLAN_SCHEMA,
        "tool": TOOL.name,
        "stage": "all",
        "run_id": "acs-multispine-staging-test",
        "source": {"commit": COMMIT, "branch": "main"},
        "inputs": inputs,
        "options": dict(OPTIONS),
        "nonpreemptible": True,
        "max_wall_seconds": 57_600,
    }
    data.update(overrides)
    return data


def _parse_function(source: str):
    """Compile only argparse and its validators from the real implementation."""

    names = {
        "_parse_args",
        "_positive_int",
        "_nonnegative_int",
        "_open_unit_interval",
    }
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    constants = {
        "argparse": argparse,
        "Path": Path,
        "DEFAULT_INPUTS_DIR": Path("/unused/default-acs-cache"),
        "DEFAULT_PUMA_LADDER": Path("/unused/default-puma-ladder"),
        "PERIOD": 2024,
        "DEFAULT_CHUNKSIZE": 100_000,
        "DEFAULT_N_ESTIMATORS": 32,
        "DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT": 8,
        "ACS_DONOR_CHANNEL_AUTO": "auto",
        "ACS_UNDER15_ZERO_POLICY": "us_hours_under15_zero_completion_v1",
        "ACS_LOCAL_SSI_DISABILITY_ISSUE": "microcosm#1022",
    }
    # Every tuning value is supplied explicitly below; synthetic default
    # paths are never used. No executable engine or build code is extracted.
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, LEGACY_SCRIPT, "exec"), constants)
    return constants["_parse_args"]


def _value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1
    return argv[argv.index(flag) + 1]


def test_registration_and_example_plan_require_every_source() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    assert plan.tool is TOOL
    assert TOOL.script == "tools/build_us_acs_multispine_base.py"
    assert TOOL.parse_function == "_legacy._parse_args"
    assert set(TOOL.stages) == {"all"}
    assert set(plan.stage_spec.required_inputs) == set(TOOL.inputs)
    assert {
        "base_h5",
        "donor_release_manifest",
        "acs_household_zip",
        "acs_person_zip",
        "puma_ladder",
    } <= set(TOOL.inputs)
    assert set(TOOL.options) == set(OPTIONS)
    assert OWNED_FLAGS <= TOOL.owned_flags
    _, example = plan_lib.load_plan(PLAN_PATH)
    assert example.tool is TOOL and example.stage == "all"
    assert example.nonpreemptible


@pytest.mark.parametrize("role", TOOL.stages["all"].required_inputs)
def test_every_input_is_required(role: str) -> None:
    data = _plan_data()
    del data["inputs"][role]
    with pytest.raises(plan_lib.PlanError, match="requires inputs"):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize("stage", ["materialize", "calibrate", "smoke", "publish"])
def test_only_all_is_registered(stage: str) -> None:
    with pytest.raises(plan_lib.PlanError, match="no stage"):
        plan_lib.parse_plan(_plan_data(stage=stage))


@pytest.mark.parametrize(
    "option",
    [
        "base_h5",
        "out_h5",
        "summary",
        "source_manifest",
        "inputs_dir",
        "puma_ladder",
        "donor_release_manifest",
        "allow_dirty",
    ],
)
def test_plan_cannot_control_sources_or_outputs(option: str) -> None:
    with pytest.raises(plan_lib.PlanError, match="not allowlisted"):
        plan_lib.parse_plan(_plan_data(options={option: "/tmp/override"}))


@pytest.mark.parametrize("link", TOOL.input_links, ids=lambda link: link.input)
def test_archive_digest_must_match_the_builder_pin(link) -> None:
    data = _plan_data()
    data["inputs"][link.input] = {"uri": "volume://wrong.zip", "sha256": "0" * 64}
    with pytest.raises(plan_lib.PlanError, match="pins|pinned|requires"):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize("role", ["acs_household_zip", "acs_person_zip"])
def test_archives_are_staged_volume_inputs_and_never_downloaded(role: str) -> None:
    data = _plan_data()
    data["inputs"][role]["uri"] = (
        "hf://datasets/example/sources@pinned/" + Path(data["inputs"][role]["uri"]).name
    )
    with pytest.raises(plan_lib.PlanError, match="volume|download"):
        plan_lib.parse_plan(data)


@pytest.mark.parametrize(
    "env",
    [
        {"MICROCOSM_HF_TOKEN": "private"},
        {"POPULACE_SOURCE_API_KEY": "private"},
        {"PYTHONHASHSEED": "42"},
        {"HF_HUB_OFFLINE": "0"},
    ],
)
def test_offline_and_credential_guarantees_remain(env: dict) -> None:
    with pytest.raises(plan_lib.PlanError):
        plan_lib.parse_plan(_plan_data(env=env))
    result, removed = plan_lib.tool_environment(
        {"HF_TOKEN": "private", "PYTHONHASHSEED": "123", "PATH": "/usr/bin"}, {}
    )
    assert result["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in result and "PYTHONHASHSEED" not in result
    assert set(removed) == {"HF_TOKEN", "PYTHONHASHSEED"}
    assert plan_lib.hash_seed_record(TOOL, {}, {})["stages_run_with"] is None


@pytest.mark.parametrize("owned", sorted(OWNED_FLAGS))
def test_owned_flags_prefixes_and_equals_are_refused_twice(monkeypatch, owned) -> None:
    original = plan_lib.parse_plan(_plan_data())
    for flag in [
        owned,
        *(owned[:end] for end in range(3, len(owned))),
        f"{owned}=value",
        f"{owned[:6]}=value",
    ]:
        rogue = dataclasses.replace(
            TOOL, options={"escape": plan_lib.OptionFlag(flag, str)}
        )
        monkeypatch.setitem(plan_lib.TOOLS, TOOL.name, rogue)
        with pytest.raises(plan_lib.PlanError, match="runner-owned|contains '='"):
            plan_lib.parse_plan(_plan_data(options={"escape": "value"}))
        bypassed = dataclasses.replace(
            original, tool=rogue, options={"escape": "value"}
        )
        with pytest.raises(plan_lib.PlanError, match="runner-owned|contains '='"):
            plan_lib.planned_argv(bypassed)


def test_argv_owns_paths_and_groups_both_archives_under_one_cache() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    argv = plan_lib.planned_argv(plan, "/custom/work")
    assert argv[:3] == [f"{plan_lib.IMAGE_VENV}/bin/python", "-B", TOOL.script]
    assert "--stage" not in argv
    assert _value(argv, "--source-manifest") == str(
        Path(plan_lib.IMAGE_REPO_ROOT) / plan_lib.ACS_MULTISPINE_SOURCE_MANIFEST.path
    )
    assert _value(argv, "--inputs-dir") == "/custom/work/acs-pums"
    assert _value(argv, "--out-h5") == (
        "/custom/work/state/" + plan_lib.ACS_MULTISPINE_STAGING_ARTIFACT
    )
    assert _value(argv, "--summary") == (
        "/custom/work/state/" + plan_lib.ACS_MULTISPINE_STAGING_SUMMARY
    )
    for role, flag in [
        ("base_h5", "--base-h5"),
        ("donor_release_manifest", "--donor-release-manifest"),
        ("puma_ladder", "--puma-ladder"),
    ]:
        assert _value(argv, flag) == plan_lib.input_local_path(
            "/custom/work", plan.inputs[role]
        )
    for key, value in OPTIONS.items():
        assert _value(argv, TOOL.options[key].flag) == str(value)
    manifest = json.loads(MANIFEST_PATH.read_text())
    links = {link.input: link for link in TOOL.input_links}
    for artifact in manifest["artifacts"]:
        link = links[f"acs_{artifact['role']}_zip"]
        assert link.path == f"acs-pums/{artifact['sha256']}/{artifact['filename']}"
        assert (link.sha256, link.size_bytes) == (
            artifact["sha256"],
            artifact["size_bytes"],
        )


def test_local_builder_parser_accepts_every_allowlisted_option() -> None:
    source = (ROOT / LEGACY_SCRIPT).read_text()
    parsed = _parse_function(source)(
        plan_lib.planned_argv(plan_lib.parse_plan(_plan_data()))[3:]
    )
    for key, value in OPTIONS.items():
        assert getattr(parsed, key) == value
    assert parsed.inputs_dir == Path("/work/acs-pums")
    assert parsed.source_manifest == (
        Path(plan_lib.IMAGE_REPO_ROOT) / plan_lib.ACS_MULTISPINE_SOURCE_MANIFEST.path
    )
    # Every CLI argument belongs either to tuning or to runner ownership.
    tree = ast.parse(source)
    parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args"
    )
    flags = {
        argument.value
        for node in ast.walk(parser)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    assert flags == OWNED_FLAGS | {option.flag for option in TOOL.options.values()}


def test_source_manifest_pin_matches_the_packaged_manifest() -> None:
    tree_file = plan_lib.ACS_MULTISPINE_SOURCE_MANIFEST
    assert ROOT / tree_file.path == MANIFEST_PATH
    assert plan_lib.sha256_file(MANIFEST_PATH)[0] == tree_file.sha256
    (row,) = plan_lib.tree_file_rows(TOOL, ROOT)
    assert "problem" not in row
    assert row["sha256"] == tree_file.sha256
    assert TOOL.home_seeds == ()


def test_example_argv_is_accepted_by_its_pinned_builder() -> None:
    _, plan = plan_lib.load_plan(PLAN_PATH)
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{plan.commit}:{LEGACY_SCRIPT}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if shown.returncode:
        pytest.skip(
            "the example plan's pinned commit is absent from this clone (depth-1 CI)"
        )
    args = _parse_function(shown.stdout)(plan_lib.planned_argv(plan)[3:])
    assert args.base_h5 and args.donor_release_manifest and args.puma_ladder
    assert args.inputs_dir == Path("/work/acs-pums")
    tree_file = plan_lib.ACS_MULTISPINE_SOURCE_MANIFEST
    manifest = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{plan.commit}:{tree_file.path}"],
        capture_output=True,
        check=True,
    )
    assert hashlib.sha256(manifest.stdout).hexdigest() == tree_file.sha256


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("acs_share", 0.0),
        ("acs_share", 1.0),
        ("acs_share", math.nan),
        ("n_estimators", 0),
        ("max_targets_per_fit", 0),
        ("seed", -1),
        ("geography_seed", -1),
        ("chunksize", 0),
        ("max_households", 0),
        ("period", 0),
        ("hours_under15_policy", "unreviewed-policy"),
    ],
)
def test_builder_parser_refuses_invalid_tuning_values(key: str, value) -> None:
    # The plan validates types; check delegates numerical/choice constraints
    # to the parser from the pinned builder before any heavy run.
    options = {**OPTIONS, key: value}
    try:
        plan = plan_lib.parse_plan(_plan_data(options=options))
    except plan_lib.PlanError:
        return
    with pytest.raises(SystemExit):
        _parse_function((ROOT / LEGACY_SCRIPT).read_text())(
            plan_lib.planned_argv(plan)[3:]
        )


def _tiny_plan(tmp_path: Path):
    """A synthetic pin for each source, with archive layout kept unchanged."""

    original = plan_lib.parse_plan(_plan_data(max_wall_seconds=600))
    inputs = dict(original.inputs)
    paths = {}
    links = []
    payloads = {}
    for role, ref in inputs.items():
        staged = tmp_path / "inputs" / role / Path(ref.uri).name
        staged.parent.mkdir(parents=True, exist_ok=True)
        if role in {"acs_household_zip", "acs_person_zip"}:
            with zipfile.ZipFile(staged, "w") as archive:
                archive.writestr("synthetic.csv", "synthetic,aggregate\n1,2\n")
        else:
            staged.write_bytes(role.encode())
        payload = staged.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        inputs[role] = dataclasses.replace(ref, sha256=digest)
        paths[role] = str(staged)
        payloads[role] = payload
    for link in TOOL.input_links:
        digest = inputs[link.input].sha256
        links.append(
            dataclasses.replace(
                link,
                path=f"acs-pums/{digest}/{Path(link.path).name}",
                sha256=digest,
                size_bytes=len(payloads[link.input]),
            )
        )
    seeds = tuple(
        dataclasses.replace(seed, sha256=inputs[seed.input].sha256)
        for seed in TOOL.home_seeds
    )
    tool = dataclasses.replace(TOOL, input_links=tuple(links), home_seeds=seeds)
    return dataclasses.replace(original, tool=tool, inputs=inputs), paths, payloads


def test_links_assemble_an_existing_verified_cache_without_copying(
    tmp_path: Path,
) -> None:
    plan, paths, payloads = _tiny_plan(tmp_path)
    work = tmp_path / "work"
    rows = plan_lib.link_staged_inputs(plan, paths, work)
    assert {row["input"] for row in rows} == {
        link.input for link in plan.tool.input_links
    }
    for link in plan.tool.input_links:
        target = work / link.path
        assert target.is_symlink()
        assert target.resolve() == Path(paths[link.input]).resolve()
        assert target.read_bytes() == payloads[link.input]
        assert {row["input"]: row for row in rows}[link.input] == {
            "input": link.input,
            "path": str(target),
            "sha256": link.sha256,
            "bytes": link.size_bytes,
        }
    assert plan_lib.link_staged_inputs(plan, paths, work) == rows


@pytest.mark.parametrize("failure", ["missing", "wrong-size"])
def test_archive_assembly_refuses_unusable_inputs(tmp_path: Path, failure: str) -> None:
    plan, paths, _ = _tiny_plan(tmp_path)
    link = plan.tool.input_links[0]
    if failure == "missing":
        Path(paths[link.input]).unlink()
    elif failure == "wrong-size":
        Path(paths[link.input]).write_bytes(b"short")
    with pytest.raises(plan_lib.PlanError):
        plan_lib.link_staged_inputs(plan, paths, tmp_path / "work")


@pytest.mark.parametrize("failure", ["missing", "wrong-size", "wrong-digest"])
def test_check_refuses_unusable_archive_metadata(failure: str) -> None:
    plan = plan_lib.parse_plan(_plan_data())
    rows = [
        {"name": link.input, "sha256": link.sha256, "bytes": link.size_bytes}
        for link in TOOL.input_links
    ]
    assert plan_lib.input_link_problems(plan, rows) == []
    if failure == "missing":
        rows.pop()
    elif failure == "wrong-size":
        rows[0]["bytes"] -= 1
    else:
        rows[0]["sha256"] = "0" * 64
    assert plan_lib.input_link_problems(plan, rows)


@pytest.mark.parametrize(
    "path", ["../outside.zip", "/outside.zip", "acs-pums/../bad.zip"]
)
def test_archive_link_path_must_remain_inside_work(tmp_path: Path, path: str) -> None:
    plan, paths, _ = _tiny_plan(tmp_path)
    with pytest.raises((plan_lib.PlanError, ValueError), match="relative|unsafe|clean"):
        link = dataclasses.replace(plan.tool.input_links[0], path=path)
        rogue = dataclasses.replace(plan.tool, input_links=(link,))
        plan_lib.link_staged_inputs(
            dataclasses.replace(plan, tool=rogue), paths, tmp_path / "work"
        )


@pytest.mark.parametrize(
    "collision", ["foreign-file", "foreign-link", "symlink-parent"]
)
def test_archive_assembly_refuses_foreign_paths(tmp_path: Path, collision: str) -> None:
    plan, paths, _ = _tiny_plan(tmp_path)
    work = tmp_path / "work"
    link = plan.tool.input_links[0]
    target = work / link.path
    target.parent.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if collision == "foreign-file":
        target.write_bytes(b"foreign cache file")
    elif collision == "foreign-link":
        target.symlink_to(foreign / "not-an-input")
    else:
        target.parent.rmdir()
        target.parent.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(plan_lib.PlanError):
        plan_lib.link_staged_inputs(plan, paths, work)
    assert list(foreign.iterdir()) == []


def test_assembled_cache_prevents_both_census_downloads(
    tmp_path: Path, monkeypatch
) -> None:
    # Load the real fetch/cache code directly, avoiding us_runtime's country
    # imports even when an engine is installed in the test interpreter.
    path = MANIFEST_PATH.with_name("acs_sources.py")
    tree = ast.parse(path.read_text())
    original_count = len(tree.body)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "microcosm.build.us_runtime.acs_pums"
        )
    ]
    assert len(tree.body) == original_count - 1
    packaged = json.loads(MANIFEST_PATH.read_text())
    acs_sources = types.ModuleType("modal_acs_sources_under_test")
    acs_sources.__file__ = str(path)
    acs_sources.ACS_2024_1YR_SPINE = packaged["spine"]
    acs_sources.ACS_2024_1YR_VINTAGE = packaged["vintage"]
    acs_sources.AcsPumsSource = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, acs_sources.__name__, acs_sources)
    exec(compile(tree, str(path), "exec"), acs_sources.__dict__)

    plan, paths, _ = _tiny_plan(tmp_path)
    work = tmp_path / "work"
    plan_lib.link_staged_inputs(plan, paths, work)
    manifest = acs_sources.load_acs_source_manifest()
    artifacts = []
    links = {link.input: link for link in plan.tool.input_links}
    for artifact in manifest.artifacts:
        link = links[f"acs_{artifact.role}_zip"]
        artifacts.append(
            dataclasses.replace(
                artifact, sha256=link.sha256, size_bytes=link.size_bytes
            )
        )
    synthetic_manifest = dataclasses.replace(manifest, artifacts=tuple(artifacts))

    def no_network(*_args, **_kwargs):
        raise AssertionError("a staged ACS archive must never be downloaded")

    sources = acs_sources.fetch_acs_pums_sources(
        work / "acs-pums", manifest=synthetic_manifest, opener=no_network
    )
    assert sources.household_zip.resolve() == Path(paths["acs_household_zip"]).resolve()
    assert sources.person_zip.resolve() == Path(paths["acs_person_zip"]).resolve()


def test_resource_class_covers_observed_parallelism_and_modal_slowdown() -> None:
    plan = plan_lib.parse_plan(_plan_data())
    stage, resources = plan.stage_spec, plan.resources
    measured = plan_lib.MEASURED[(TOOL.name, "all")]
    assert (measured.peak_rss_bytes, measured.wall_seconds, measured.cpu_seconds) == (
        42_370_498_560,
        7218.1,
        103104.6,
    )
    assert "run-resources-and-staging-excerpt.json" in measured.source
    assert "full-staging-a3" in measured.source
    assert resources is plan_lib.ACS_STAGING
    assert resources.name == "acs-staging"
    assert resources.cpu == resources.cpu_limit == 16.0
    assert resources.cpu >= measured.cpu_seconds / measured.wall_seconds
    assert resources.memory_mib == 80 * 1024
    request_bytes = resources.memory_mib * 1024**2
    assert request_bytes >= measured.peak_rss_bytes * 1.5
    assert request_bytes >= measured.peak_rss_bytes + 24e9
    assert resources.timeout_s == 17 * 3600
    assert stage.runner_overhead_seconds == 1800
    assert plan.max_wall_seconds >= math.ceil(measured.wall_seconds * 7.6)
    assert plan.max_wall_seconds + stage.runner_overhead_seconds <= resources.timeout_s
    assert stage.min_free_disk_gib == 40
    assert stage.mirrored_state_gib == 12
    assert not stage.rest_is_resume_state
    assert plan_lib.work_disk_problem(stage, 40 * 1024**3) is None
    assert plan_lib.work_disk_problem(stage, 39 * 1024**3)
    budget = plan_lib.tool_budget(plan, plan.max_wall_seconds, 600)
    assert (
        budget["seconds"] + 600 + stage.runner_overhead_seconds <= resources.timeout_s
    )


def test_check_write_probe_covers_the_staging_output_reserve() -> None:
    stage = TOOL.stages["all"]
    okay = plan_lib.write_probe_verdict(stage, 1024**3, 10)
    slow = plan_lib.write_probe_verdict(stage, 1024**3, 180)
    assert "problem" not in okay
    assert "problem" in slow


def test_staging_has_preemptible_and_nonpreemptible_modal_runners(app) -> None:
    for placement in (False, True):
        options = app.RUNNERS[("acs-staging", placement)].modal_options
        assert options["cpu"] == (16.0, 16.0)
        assert options["memory"] == 80 * 1024
        assert options["timeout"] == 61_200
        assert options["retries"] == 0
        assert options.get("nonpreemptible", False) is placement


def _synthetic_builder_source(plan) -> str:
    """The actual parser with a small, engine-free stand-in for its build."""

    tree = ast.parse((ROOT / LEGACY_SCRIPT).read_text())
    names = {"_parse_args", "_positive_int", "_nonnegative_int", "_open_unit_interval"}
    parser_source = "\n\n".join(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    )
    archive_names = [
        (link.sha256, Path(link.path).name) for link in plan.tool.input_links
    ]
    return (
        "import argparse, json, os, sys, types\nfrom pathlib import Path\n"
        "DEFAULT_INPUTS_DIR = Path('/unused')\nDEFAULT_PUMA_LADDER = Path('/unused')\n"
        "PERIOD = 2024\nDEFAULT_CHUNKSIZE = 100000\nDEFAULT_N_ESTIMATORS = 32\n"
        "DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT = 8\nACS_DONOR_CHANNEL_AUTO = 'auto'\n"
        "ACS_UNDER15_ZERO_POLICY = 'us_hours_under15_zero_completion_v1'\n"
        "ACS_LOCAL_SSI_DISABILITY_ISSUE = 'microcosm#1022'\n"
        + parser_source
        + "\n_legacy = types.SimpleNamespace(_parse_args=_parse_args)\n"
        + "if __name__ == '__main__':\n"
        + "    args = _parse_args()\n"
        + "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        + "    assert 'HF_TOKEN' not in os.environ\n"
        + "    assert 'PYTHONHASHSEED' not in os.environ\n"
        + f"    archives = {archive_names!r}\n"
        + "    assert all((args.inputs_dir / digest / name).is_file() for digest, name in archives)\n"
        + "    args.out_h5.write_bytes(b'synthetic staging H5')\n"
        + "    args.summary.write_text(json.dumps({'aggregate': len(archives)}) + '\\n')\n"
        + "    print('synthetic staging aggregate complete')\n"
    )


def test_pinned_parser_check_follows_the_shim_dotted_attribute(
    app, tmp_path: Path, monkeypatch
) -> None:
    plan, paths, _ = _tiny_plan(tmp_path)
    script = tmp_path / "synthetic_builder.py"
    script.write_text(_synthetic_builder_source(plan))
    tool = dataclasses.replace(plan.tool, script=str(script))
    plan = dataclasses.replace(plan, tool=tool)
    argv = plan_lib.build_stage_argv(
        plan,
        python=sys.executable,
        input_paths=paths,
        state_dir=str(tmp_path / "state"),
    )
    monkeypatch.setattr(plan_lib, "IMAGE_REPO_ROOT", str(tmp_path))
    env, _ = plan_lib.tool_environment({}, {})
    assert app._parse_check(plan, argv, env)["returncode"] == 0
    bad = [*argv]
    bad[bad.index("--n-estimators") + 1] = "0"
    assert app._parse_check(plan, bad, env)["returncode"] != 0


@pytest.mark.parametrize(
    "failure", [None, "archive-size", "archive-missing", "manifest-pin", "parser"]
)
def test_check_validates_offline_argv_archive_metadata_and_manifest(
    app, tmp_path: Path, monkeypatch, failure: str | None
) -> None:
    monkeypatch.setattr(plan_lib, "WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setattr(plan_lib, "RUNS_MOUNT", str(tmp_path / "runs"))
    tree_file = plan_lib.ACS_MULTISPINE_SOURCE_MANIFEST
    clone = {
        "head_matches_plan": True,
        "tree_clean": True,
        "branch_verified": True,
        "tool_present": True,
        "tree_files_verified": failure != "manifest-pin",
        "tree_files": [
            {
                "name": "acs_source_manifest",
                "path": tree_file.path,
                **({"problem": "another version"} if failure == "manifest-pin" else {}),
            }
        ],
    }
    monkeypatch.setattr(app, "_git_state", lambda _plan: clone)
    monkeypatch.setattr(
        app.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(
            total=512 * 1024**3, used=0, free=100 * 1024**3
        ),
    )
    monkeypatch.setattr(app, "_write_probe", lambda _plan: {"test": "fast"})
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout='{"policyengine-us": "test"}', stderr=""
        ),
    )
    links = {link.input: link for link in TOOL.input_links}

    def check_input(ref, _work):
        row = {"name": ref.name, "sha256": ref.sha256, "bytes": 1}
        if ref.name in links:
            row["bytes"] = links[ref.name].size_bytes
        if ref.name == "acs_household_zip":
            if failure == "archive-size":
                row["bytes"] -= 1
            elif failure == "archive-missing":
                row["problem"] = "missing input"
        return row

    def parse_check(_plan, argv, env):
        assert env["HF_HUB_OFFLINE"] == "1"
        assert "HF_TOKEN" not in env
        assert _value(argv, "--inputs-dir") == str(tmp_path / "work" / "acs-pums")
        assert _value(argv, "--source-manifest").endswith(tree_file.path)
        return {"returncode": 2 if failure == "parser" else 0}

    monkeypatch.setattr(app, "_check_input", check_input)
    monkeypatch.setattr(app, "_parse_check", parse_check)
    monkeypatch.setenv("HF_TOKEN", "must-be-withheld")
    report = app.check_stage(_plan_data())
    assert report["ok"] is (failure is None)
    if failure is not None:
        assert report["problems"]
    assert "must-be-withheld" not in json.dumps(report)


def test_runner_assembles_archives_then_mirrors_h5_summary_and_receipt(
    app, tmp_path: Path, monkeypatch
) -> None:
    plan, _, payloads = _tiny_plan(tmp_path / "source")
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "synthetic_builder.py"
    script.write_text(_synthetic_builder_source(plan))
    tool = dataclasses.replace(plan.tool, script=script.name)
    monkeypatch.setitem(plan_lib.TOOLS, TOOL.name, tool)
    data = _plan_data(max_wall_seconds=600)
    for role, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        name = Path(plan.inputs[role].uri).name
        data["inputs"][role] = {
            "uri": f"volume://cas/sha256/{digest}/{name}",
            "sha256": digest,
        }
        cas = tmp_path / "inputs-volume" / "cas" / "sha256" / digest / name
        cas.parent.mkdir(parents=True, exist_ok=True)
        cas.write_bytes(payload)
    for key, path in {
        "INPUTS_MOUNT": tmp_path / "inputs-volume",
        "RUNS_MOUNT": tmp_path / "runs-volume",
        "WORK_ROOT": tmp_path / "work",
        "IMAGE_REPO_ROOT": repo,
        "IMAGE_VENV": Path(sys.executable).parent.parent,
    }.items():
        monkeypatch.setattr(plan_lib, key, str(path))
    monkeypatch.setattr(
        app,
        "_git_state",
        lambda _plan: {
            "head_matches_plan": True,
            "tree_clean": True,
            "branch_verified": True,
            "branch_tip": COMMIT,
            "tool_present": True,
            "tree_files_verified": True,
        },
    )
    monkeypatch.setattr(
        app.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(
            total=512 * 1024**3, used=0, free=100 * 1024**3
        ),
    )
    monkeypatch.setenv("HF_TOKEN", "must-be-withheld")
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    receipt = app._run_stage(data)
    assert receipt["status"] == "COMPLETED"
    assert receipt["runner"]["python_hash_seed"]["passed_to_tool"] is None
    assert receipt["runner"]["input_links"]
    assert {row["input"] for row in receipt["runner"]["input_links"]} == {
        "acs_household_zip",
        "acs_person_zip",
    }
    state = tmp_path / "runs-volume" / "runs" / data["run_id"] / "state"
    assert (
        state / plan_lib.ACS_MULTISPINE_STAGING_ARTIFACT
    ).read_bytes() == b"synthetic staging H5"
    assert json.loads(
        (state / plan_lib.ACS_MULTISPINE_STAGING_SUMMARY).read_text()
    ) == {"aggregate": 2}
    assert receipt["outputs_not_mirrored"] == []
    assert plan_lib.verify_receipt(receipt, state, strict=True) == []
    assert len(list(state.glob("logs/*.log"))) == 1
    assert "must-be-withheld" not in json.dumps(receipt)


def test_receipt_preserves_inputs_outputs_placement_and_failed_state(
    tmp_path: Path,
) -> None:
    data = _plan_data()
    plan = plan_lib.parse_plan(data)
    state = tmp_path / "state"
    state.mkdir()
    (state / plan_lib.ACS_MULTISPINE_STAGING_ARTIFACT).write_bytes(b"synthetic H5")
    (state / plan_lib.ACS_MULTISPINE_STAGING_SUMMARY).write_text('{"aggregate": 2}\n')
    inputs = [
        {"name": role, "sha256": ref.sha256, "uri": ref.uri}
        for role, ref in plan.inputs.items()
    ]
    receipt = plan_lib.build_receipt(
        plan,
        data,
        argv=plan_lib.planned_argv(plan),
        returncode=0,
        started_at="2026-10-13T00:00:00Z",
        finished_at="2026-10-13T02:00:00Z",
        wall_seconds=7218.1,
        peak_rss_bytes=42_370_498_560,
        inputs_verified=inputs,
        outputs=plan_lib.hash_tree(state),
        git={"head": plan.commit, "branch_verified": True, "branch_tip": plan.commit},
        runner={"input_links": [dataclasses.asdict(link) for link in TOOL.input_links]},
    )
    assert receipt["tool"] == TOOL.name and receipt["stage"] == "all"
    assert receipt["source"]["commit"] == plan.commit
    assert receipt["inputs"] == inputs
    assert receipt["resources"]["class"] == "acs-staging"
    assert receipt["resources"]["nonpreemptible"] is True
    assert receipt["resources"]["cpu_limit"] == 16.0
    assert receipt["outputs_not_mirrored"] == []
    assert plan_lib.verify_receipt(receipt, state, strict=True) == []
    failed = {**receipt, "status": "FAILED", "returncode": 1}
    assert plan_lib.verify_receipt(failed, state, require_completed=True)
    assert plan_lib.verify_receipt(failed, state, require_completed=False) == []
    (state / plan_lib.ACS_MULTISPINE_STAGING_SUMMARY).write_text('{"aggregate": 3}\n')
    assert any(
        "sha256 mismatch" in problem
        for problem in plan_lib.verify_receipt(receipt, state)
    )
