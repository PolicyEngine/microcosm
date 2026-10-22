"""Bind Route A's local comparison reference to actual files and chosen code."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
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


@pytest.fixture(params=["us-puf-support-base", "us-puf-support-base-rerun"])
def evidence(tmp_path, monkeypatch, helper, request):
    # Real temporary input/checkpoint bytes; only the archive home-cache
    # registry pin is disabled so tiny synthetic source files can be used.
    monkeypatch.setitem(
        helper.stage_plan.TOOLS,
        request.param,
        dataclasses.replace(helper.stage_plan.TOOLS[request.param], home_seeds=()),
    )
    monkeypatch.setattr(helper, "_committed_source_sha256", lambda *_args: "a" * 64)
    data = base_plan_data()
    data["tool"] = request.param
    config = json.loads(
        (ROOT / "docs/us-modal-stage-route-a-base-local-reference.json").read_text()
    )["run_config"]
    config["builder_code_identity"]["source_sha256"] = "a" * 64
    for role, ref in data["inputs"].items():
        filename = ref["uri"].rsplit("/", 1)[1]
        source = tmp_path / "inputs" / role / filename
        source.parent.mkdir(parents=True)
        source.write_bytes(f"synthetic {role}".encode())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        ref.update(uri=f"volume://cas/sha256/{digest}/{filename}", sha256=digest)
        if role.startswith("asec_education_"):
            config["asec_education_source"][role.split("_")[2]] = str(source)
        elif role.startswith("asec_"):
            year = role.split("_")[1]
            config["asec_h5_sha256"][year] = digest
            config["asec_h5"] = [
                f"{year}={source}" if item.startswith(year + "=") else item
                for item in config["asec_h5"]
            ]
        else:
            key = {
                "acs_2022_h5": "acs_h5",
                "puf_2024_h5": "puf_h5",
                "puf_2015_csv": "puf_source_year_csv",
                "base_ledger_facts": "ledger_facts",
                "block_ladder_npz": "block_ladder_artifact",
            }[role]
            config[key] = str(source)
            if role in {"puf_2024_h5", "puf_2015_csv"}:
                config[key + "_sha256"] = digest
    crosswalk = tmp_path / "congressional_district_vintage_crosswalk.csv"
    crosswalk.write_text("synthetic crosswalk\n")
    config["congressional_district_vintage_crosswalk"] = str(crosswalk)
    selected_tool = helper.stage_plan.TOOLS[request.param]
    monkeypatch.setitem(
        helper.stage_plan.TOOLS,
        request.param,
        dataclasses.replace(
            selected_tool,
            tree_files={
                "cd_vintage_crosswalk": dataclasses.replace(
                    selected_tool.tree_files["cd_vintage_crosswalk"],
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
                for year in helper.stage_plan.PUF_SUPPORT_BASE_ASEC_YEARS
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
