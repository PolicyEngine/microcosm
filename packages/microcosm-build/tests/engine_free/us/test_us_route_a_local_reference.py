"""Bind Route A's local comparison reference to actual files and chosen code."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_build.us_modal_stage_plan_tool import (
    ROOT,
    base_plan_data,
    plan_lib,
)


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    spec = importlib.util.spec_from_file_location(
        "route_a_make_local_reference", ROOT / "tools/route_a/make_local_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


#: The run-config keys of the inputs every pool's command names the same way.
_FIXED_ROLES = {
    "acs_2022_h5": "acs_h5",
    "puf_2024_h5": "puf_h5",
    "puf_2015_csv": "puf_source_year_csv",
    "base_ledger_facts": "ledger_facts",
    "block_ladder_npz": "block_ladder_artifact",
}
_POOLED_ROLE = re.compile(r"asec_(education_)?\d{4}_(h5|zip)")
DEFAULT_POOL = plan_lib.US_PUF_SUPPORT_BASE_2023_2025.name


@pytest.fixture(params=sorted(plan_lib.PUF_SUPPORT_BASE_POOLS))
def evidence(tmp_path, monkeypatch, helper, request):
    # Real temporary input/checkpoint bytes; only the registry pins of the
    # home-cache seeds and the crosswalk move to them, so tiny synthetic
    # source files can be used.
    tool = helper.stage_plan.TOOLS[request.param]
    pool = helper.stage_plan.PUF_SUPPORT_BASE_POOLS[request.param]
    monkeypatch.setattr(helper, "_committed_source_sha256", lambda *_args: "a" * 64)
    if request.param == DEFAULT_POOL:
        data = json.loads(
            (ROOT / "docs/us-modal-stage-route-a-rerun-base-plan.json").read_text()
        )
        data["source"]["commit"] = base_plan_data()["source"]["commit"]
    else:
        data = base_plan_data(tool=request.param)
    config = json.loads(
        (ROOT / "docs/us-modal-stage-route-a-base-local-reference.json").read_text()
    )["run_config"]
    config["builder_code_identity"]["source_sha256"] = "a" * 64
    # A seeded input that no flag names is read from the tool's home cache.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    seeds = {seed.input: seed for seed in tool.home_seeds}
    paths, digests = {}, {}
    for role, ref in data["inputs"].items():
        filename = ref["uri"].rsplit("/", 1)[1]
        if _POOLED_ROLE.fullmatch(role) or role in _FIXED_ROLES:
            source = tmp_path / "inputs" / role / filename
        else:
            source = home / seeds[role].path
        source.parent.mkdir(parents=True)
        source.write_bytes(f"synthetic {role}".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        ref.update(uri=f"volume://cas/sha256/{digest}/{filename}", sha256=digest)
        paths[role], digests[role] = str(source), digest
    config["asec_h5"] = [
        f"{year}={paths[f'asec_{year}_h5']}" for year in pool.asec_years
    ]
    config["asec_h5_sha256"] = {
        str(year): digests[f"asec_{year}_h5"] for year in pool.asec_years
    }
    config["asec_education_source"] = {
        str(year): paths[f"asec_education_{year}_zip"] for year in pool.education_years
    }
    for role, key in _FIXED_ROLES.items():
        config[key] = paths[role]
    config["puf_h5_sha256"] = digests["puf_2024_h5"]
    config["puf_source_year_csv_sha256"] = digests["puf_2015_csv"]
    crosswalk = tmp_path / "congressional_district_vintage_crosswalk.csv"
    crosswalk.write_text("synthetic crosswalk\n")
    config["congressional_district_vintage_crosswalk"] = str(crosswalk)
    monkeypatch.setitem(
        helper.stage_plan.TOOLS,
        request.param,
        dataclasses.replace(
            tool,
            home_seeds=tuple(
                dataclasses.replace(seed, sha256=digests[seed.input])
                for seed in tool.home_seeds
            ),
            tree_files={
                "cd_vintage_crosswalk": dataclasses.replace(
                    tool.tree_files["cd_vintage_crosswalk"],
                    sha256=hashlib.sha256(crosswalk.read_bytes()).hexdigest(),
                )
            },
        ),
    )
    config["out"] = str(tmp_path / "base-out")
    checkpoints = tmp_path / "base-checkpoints"
    checkpoints.mkdir()
    names = ["source_construction", "pre_clone_enrichment"]
    records = []
    for index, name in enumerate(names):
        filename = f"{index:03d}_{name}.frame.h5"
        path = checkpoints / filename
        path.write_bytes(f"synthetic {name} checkpoint".encode())
        records.append(
            {
                "stage": name,
                "checkpoint_filename": filename,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records[0]["metadata"] = {
        "base_source": {
            "sources": [
                {"year": year, "sha256": data["inputs"][f"asec_{year}_h5"]["sha256"]}
                for year in pool.asec_years
            ]
        }
    }
    records[1]["metadata"] = {"acs_sha256": data["inputs"]["acs_2022_h5"]["sha256"]}
    context = {
        "artifact_kind": "populace_outer_stage_run_context",
        "completed": names,
        "pipeline": {
            "stages": [
                {"name": name, "description": name + " → frame"} for name in names
            ]
        },
        "run_config": config,
        "stage_records": records,
    }
    context["pipeline_sha256"] = hashlib.sha256(
        json.dumps(
            context["pipeline"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    context_path = checkpoints / "stage_run_context.json"
    context_path.write_text(json.dumps(context))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(data))
    return plan_path, checkpoints, context, context_path


def test_reference_uses_actual_context_and_checkpoints(helper, evidence, tmp_path):
    plan, checkpoints, context, _context_path = evidence
    reference = helper.make_reference(plan, checkpoints, tmp_path)
    assert reference["commit"] == json.loads(plan.read_text())["source"]["commit"]
    assert reference["run_config"] == plan_lib.mask_config_paths(context["run_config"])
    assert reference["pipeline"] == context["pipeline"]
    assert reference["outputs"] == {
        f"base-checkpoints/{record['checkpoint_filename']}": record["checkpoint_sha256"]
        for record in context["stage_records"]
    }
    assert reference["argv"][0] == "<python>"
    assert reference["argv"][reference["argv"].index("--stage") + 1] == "all"
    assert reference["stages_compared"] == context["completed"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("checkpoint", "differs from its context record"),
        ("ladder", "block_ladder_npz sha256 differs"),
        ("puf_pin", "puf_h5_sha256 recorded pin differs"),
        ("source", "source identity differs"),
        ("hash_seed", "PYTHONHASHSEED must be 0"),
        ("extra_stage", "exactly completed stages 000 and 001"),
        ("seed", "run_config.seed differs"),
        ("pipeline", "pipeline digest differs"),
        ("environment", "plan non-thread environment"),
        ("acs_record", "ACS enrichment-stage digest differs"),
    ],
)
def test_refuses_evidence_mismatches(helper, evidence, tmp_path, change, message):
    plan, checkpoints, context, context_path = evidence
    if change == "checkpoint":
        (checkpoints / "001_pre_clone_enrichment.frame.h5").write_bytes(b"changed")
    elif change == "ladder":
        Path(context["run_config"]["block_ladder_artifact"]).write_bytes(b"old ladder")
    elif change == "puf_pin":
        context["run_config"]["puf_h5_sha256"] = "f" * 64
    elif change == "source":
        context["run_config"]["builder_code_identity"]["source_sha256"] = "f" * 64
    elif change == "hash_seed":
        context["run_config"]["thread_environment"]["PYTHONHASHSEED"] = "12345"
    elif change == "extra_stage":
        context["completed"].append("clone_feature_extraction")
    elif change == "pipeline":
        context["pipeline_sha256"] = "f" * 64
    elif change == "environment":
        plan_data = json.loads(plan.read_text())
        plan_data["env"]["MICROCOSM_UNRECORDED_SETTING"] = "1"
        plan.write_text(json.dumps(plan_data))
    elif change == "acs_record":
        context["stage_records"][1]["metadata"]["acs_sha256"] = "f" * 64
    else:
        context["run_config"]["seed"] = 1
    context_path.write_text(json.dumps(context))
    with pytest.raises(ValueError, match=message):
        helper.make_reference(plan, checkpoints, tmp_path)


def test_cli_refuses_overwrite_and_writes_no_acceptance(helper, evidence, tmp_path):
    plan, checkpoints, _context, _context_path = evidence
    output = tmp_path / "reference.json"
    args = [
        "--plan",
        str(plan),
        "--checkpoint-root",
        str(checkpoints),
        "--out",
        str(output),
    ]
    assert helper.main(args) == 0
    original = output.read_bytes()
    assert helper.main(args) == 2
    assert output.read_bytes() == original
    assert not (tmp_path / "ACCEPTED").exists()


def test_committed_source_digest_ignores_dirty_checkout(helper, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    identity = Path("packages/microcosm-build/src/microcosm/build/code_identity.py")
    (repo / identity).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / identity, repo / identity)
    (repo / "tools").mkdir()
    (repo / "tools/build_us_puf_support_base.py").write_text("# synthetic builder\n")
    (repo / "pyproject.toml").write_text("# synthetic project\n")
    (repo / "uv.lock").write_text("# synthetic lock\n")

    def git(*args):
        return subprocess.check_output(
            ["git", *NO_BACKGROUND_MAINTENANCE, "-C", str(repo), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = git("rev-parse", "HEAD")
    expected = helper._committed_source_sha256(repo, commit)
    (repo / "tools/build_us_puf_support_base.py").write_text("# changed working copy\n")
    assert helper._committed_source_sha256(repo, commit) == expected
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--quiet",
        "-m",
        "changed",
    )
    assert helper._committed_source_sha256(repo, git("rev-parse", "HEAD")) != expected


@pytest.mark.parametrize("evidence", [DEFAULT_POOL], indirect=True)
@pytest.mark.parametrize("change", ["missing", "bytes"])
def test_the_seeded_lkweeks_archive_is_hashed_in_the_home_cache(
    helper, evidence, tmp_path, change
):
    # The 2023-2025 pool names no LKWEEKS archive on the command line: the
    # local tool read it from its home cache, which the Modal runner seeds.
    plan, checkpoints, _context, _context_path = evidence
    (seed,) = helper.stage_plan.TOOLS[DEFAULT_POOL].home_seeds
    cached = Path.home() / seed.path
    reference = helper.make_reference(plan, checkpoints, tmp_path)
    assert reference["inputs"][seed.input] == (
        hashlib.sha256(cached.read_bytes()).hexdigest()
    )
    if change == "missing":
        cached.unlink()
        with pytest.raises(FileNotFoundError):
            helper.make_reference(plan, checkpoints, tmp_path)
    else:
        cached.write_bytes(b"another archive")
        with pytest.raises(ValueError, match=f"local {seed.input} sha256 differs"):
            helper.make_reference(plan, checkpoints, tmp_path)
