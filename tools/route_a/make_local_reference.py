"""Record existing local stage 000/001 evidence; never launch a build.

The checkpoint context supplies the actual config, pipeline and recorded frame
digests. Every configured input is hashed again against the filled Modal plan.
The recorded code identity must equal the chosen commit's committed sources.
This creates comparison evidence, never an exception or an ACCEPTED marker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import modal_us_stage_plan as stage_plan  # noqa: E402


def _committed_source_sha256(repo_root: Path, commit: str) -> str:
    """Use the shared identity function on sources extracted from git objects."""

    with tempfile.TemporaryDirectory(prefix="route-a-reference-") as temporary:
        root = Path(temporary)
        with subprocess.Popen(
            [
                "git",
                "-C",
                str(repo_root),
                "archive",
                commit,
                "tools/build_us_puf_support_base.py",
                "pyproject.toml",
                "uv.lock",
                "packages",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert process.stdout is not None
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    packaged = (
                        len(path.parts) >= 4
                        and path.parts[0] == "packages"
                        and path.parts[2] == "src"
                        and path.suffix in {".json", ".py", ".toml", ".yaml", ".yml"}
                    )
                    if not packaged and member.name not in {
                        "tools/build_us_puf_support_base.py",
                        "pyproject.toml",
                        "uv.lock",
                    }:
                        continue
                    if not member.isfile() or path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"unsupported committed source: {member.name}")
                    stream = archive.extractfile(member)
                    assert stream is not None
                    target = root / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as output:
                        shutil.copyfileobj(stream, output)
            _stdout, stderr = process.communicate()
            if process.returncode:
                raise ValueError(
                    f"cannot read chosen commit: {stderr.decode().strip()}"
                )
        identity_path = (
            root / "packages/microcosm-build/src/microcosm/build/code_identity.py"
        )
        spec = importlib.util.spec_from_file_location(
            "route_a_code_identity", identity_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("chosen commit lacks the shared code-identity helper")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.builder_code_identity(
            root,
            tool_path=root / "tools/build_us_puf_support_base.py",
            distributions=(),
        )["source_sha256"]


def _input_paths(config: dict) -> dict[str, Path]:
    paths = {
        f"asec_{year}_h5": Path(filename)
        for item in config["asec_h5"]
        for year, filename in [item.split("=", 1)]
    }
    paths.update(
        {
            f"asec_education_{year}_zip": Path(filename)
            for year, filename in config["asec_education_source"].items()
        }
    )
    for role, key in {
        "puf_2024_h5": "puf_h5",
        "puf_2015_csv": "puf_source_year_csv",
        "acs_2022_h5": "acs_h5",
        "base_ledger_facts": "ledger_facts",
        "block_ladder_npz": "block_ladder_artifact",
    }.items():
        paths[role] = Path(config[key])
    return paths


def make_reference(plan_path: Path, checkpoint_root: Path, repo_root: Path) -> dict:
    """Refuse mismatched evidence and return a reference for the actual files."""

    raw_plan, plan = stage_plan.load_plan(plan_path)
    if (
        plan.tool.name not in {"us-puf-support-base", "us-puf-support-base-rerun"}
        or plan.stage != "all"
    ):
        raise ValueError("a Route A PUF-support base all-stage plan is required")
    context_path = checkpoint_root / "stage_run_context.json"
    context = json.loads(context_path.read_text())
    names = ["source_construction", "pre_clone_enrichment"]
    if context.get("completed") != names:
        raise ValueError("local context must have exactly completed stages 000 and 001")
    if context.get("artifact_kind") != "populace_outer_stage_run_context":
        raise ValueError("not an outer-stage run context")
    config = context["run_config"]
    if config["builder_code_identity"]["source_sha256"] != _committed_source_sha256(
        repo_root, plan.commit
    ):
        raise ValueError("local source identity differs from the chosen commit")
    if config["thread_environment"].get("PYTHONHASHSEED") != "0":
        raise ValueError("local PYTHONHASHSEED must be 0")
    nonthread_env = {
        name: value
        for name, value in raw_plan.get("env", {}).items()
        if name not in config["thread_environment"]
    }
    if nonthread_env != {"PYTHONUNBUFFERED": "1"}:
        raise ValueError("plan non-thread environment must be only PYTHONUNBUFFERED=1")

    # These are the registry's fixed command settings and the builder defaults
    # for flags it withholds. A later config key is retained for compare-lineage.
    settings = {
        "target_year": stage_plan.PUF_SUPPORT_BASE_TARGET_YEAR,
        "seed": stage_plan.PUF_SUPPORT_BASE_SEED,
        "n_estimators": stage_plan.PUF_SUPPORT_BASE_N_ESTIMATORS,
        "congressional_district_seed": stage_plan.PUF_SUPPORT_BASE_CD_SEED,
        "assign_congressional_districts": True,
        "geography_ladder_seed": 0,
        "allow_geography_ladder_gate_failures": False,
        "without_block_ladder": False,
        "equivalence_deterministic_h5_metadata": False,
        "base_h5": None,
        "asec_max_households": None,
        "support_spine_spec": None,
        "asec_2023_weeks_unemployed_source": None,
    }
    for key, value in settings.items():
        if key not in config or config[key] != value:
            raise ValueError(
                f"local run_config.{key} differs from the registered command"
            )
    if Path(config["out"]).name != stage_plan.PUF_SUPPORT_BASE_OUT:
        raise ValueError("local --out must have the planned base-out basename")
    crosswalk = Path(config["congressional_district_vintage_crosswalk"])
    tree_pin = plan.tool.tree_files["cd_vintage_crosswalk"]
    if crosswalk.name != Path(tree_pin.path).name:
        raise ValueError("local crosswalk filename differs from the planned command")
    if stage_plan.sha256_file(crosswalk)[0] != tree_pin.sha256:
        raise ValueError("local crosswalk differs from the registered tree pin")

    paths = _input_paths(config)
    if set(paths) != set(plan.inputs):
        raise ValueError("local input roles differ from the plan")
    inputs = {}
    for role, ref in plan.inputs.items():
        filename = Path(stage_plan.input_local_path(stage_plan.WORK_ROOT, ref)).name
        if paths[role].name != filename:
            raise ValueError(f"local {role} filename differs from the planned command")
        digest, _size = stage_plan.sha256_file(paths[role])
        if digest != ref.sha256:
            raise ValueError(f"local {role} sha256 differs from the plan")
        inputs[role] = digest
    for year in stage_plan.PUF_SUPPORT_BASE_ASEC_YEARS:
        if config["asec_h5_sha256"].get(str(year)) != inputs[f"asec_{year}_h5"]:
            raise ValueError(f"local ASEC {year} recorded pin differs from its bytes")
    for key, role in {
        "puf_h5_sha256": "puf_2024_h5",
        "puf_source_year_csv_sha256": "puf_2015_csv",
    }.items():
        if config.get(key) != inputs[role]:
            raise ValueError(f"local {key} recorded pin differs from its bytes")
    if [item.split("=", 1)[0] for item in config["asec_h5"]] != [
        str(year) for year in stage_plan.PUF_SUPPORT_BASE_ASEC_YEARS
    ]:
        raise ValueError("local ASEC order differs from the registered command")

    outputs = {}
    records = context["stage_records"]
    if [record["stage"] for record in records] != names:
        raise ValueError("local stage records do not match completed stages 000/001")
    source_digests = {
        f"asec_{source['year']}_h5": source["sha256"]
        for source in records[0]["metadata"]["base_source"]["sources"]
    }
    for role, digest in source_digests.items():
        if inputs.get(role) != digest:
            raise ValueError(f"local {role} source-stage digest differs from its bytes")
    if set(source_digests) != {
        f"asec_{year}_h5" for year in stage_plan.PUF_SUPPORT_BASE_ASEC_YEARS
    }:
        raise ValueError("local source-stage ASEC roles differ from the plan")
    if records[1]["metadata"]["acs_sha256"] != inputs["acs_2022_h5"]:
        raise ValueError("local ACS enrichment-stage digest differs from its bytes")
    for index, (name, record) in enumerate(zip(names, records, strict=True)):
        filename = f"{index:03d}_{name}.frame.h5"
        digest, _size = stage_plan.sha256_file(checkpoint_root / filename)
        if (
            record["checkpoint_filename"] != filename
            or record["checkpoint_sha256"] != digest
        ):
            raise ValueError(f"local {filename} differs from its context record")
        outputs[f"base-checkpoints/{filename}"] = digest
    pipeline = context["pipeline"]
    if [item["name"] for item in pipeline["stages"]][:2] != names:
        raise ValueError("local pipeline does not start with stages 000/001")
    # Replay StagePipeline.sha256's public JSON wire representation without
    # importing its numpy/pandas runtime (outer_stage_runtime.py).
    pipeline_sha256 = hashlib.sha256(
        json.dumps(
            pipeline,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if context["pipeline_sha256"] != pipeline_sha256:
        raise ValueError("local pipeline digest differs from its context record")
    return {
        "schema": stage_plan.LOCAL_REFERENCE_SCHEMA,
        "description": (
            "Generated from existing local completed stages 000/001, their recorded "
            "config and actual frame/input bytes. argv is the registered all-stage "
            "command whose settings these named stages used. No build was launched; "
            "stages 002 onward have no local comparison here."
        ),
        "tool": plan.tool.name,
        "stage": plan.stage,
        "commit": plan.commit,
        "argv": stage_plan.tokenize_stage_argv(stage_plan.planned_argv(plan)),
        "env": raw_plan.get("env", {}),
        "inputs": inputs,
        "outputs": outputs,
        "stages_compared": names,
        "run_context_path": "base-checkpoints/stage_run_context.json",
        "pipeline": pipeline,
        "pipeline_sha256": pipeline_sha256,
        "run_config": stage_plan.mask_config_paths(config),
        "local_evidence": {
            "context_sha256": stage_plan.sha256_file(context_path)[0],
            "note": "Input bytes were rehashed at reference creation; retain the local context and checkpoints.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=TOOLS.parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(f"output already exists: {args.out}")
        reference = make_reference(args.plan, args.checkpoint_root, args.repo_root)
        with args.out.open("x") as output:
            output.write(json.dumps(reference, indent=2) + "\n")
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
