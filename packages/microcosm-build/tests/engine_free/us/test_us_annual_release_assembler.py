"""Assembling an annual build of a US national or local-area build.

National builds project into ``microcosm-us-2024-annual`` (microcosm#1570);
local-area builds into ``microcosm-us-2024-local-annual`` (microcosm#922).

``tools/assemble_us_annual_release_dir.py`` takes a local copy of the base
national build's folder, an annual candidate directory and its acceptance
report, and writes a build of ``microcosm-us-2024-annual`` that passes the
release contract. Every input here is a small fixture: no candidate is built
and nothing reaches the Hub.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import h5py
import pytest

from microcosm.data.build_folder_manifests import publication_commands
from microcosm.data.contract import (
    US_LOCAL_ANNUAL_RELEASE_ID,
    US_LOCAL_RELEASE_ID,
    ReleaseContractError,
    validate_release_dir,
)

# The national build-folder fixtures, which build on release_dir and
# artifact_root.
from test_support.microcosm_data.annual_projections import (  # noqa: F401
    US_ANNUAL_RELEASE,
    US_NATIONAL_RELEASE,
    _h5,
    artifact_root,
    national_build,
    release_dir,
)
from test_support.microcosm_data.local_annual import (
    LOCAL_BUILD,
    LOCAL_DATASET_KEY,
    LOCAL_REPO,
    acceptance_report,
    local_build_folder,
    local_releases_record,
    rewrite_json,
    write_local_candidate,
)
from test_support.microcosm_data.release import RELEASE_ID
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_REPOSITORY = "policyengine/populace-us"
#: The time the tests assemble at.
_ASSEMBLED_AT = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
_CHECKS = (
    "schema",
    "source_identity",
    "year",
    "demographics",
    "input_aggregates",
    "runtime",
)


def _load_tool():
    path = _TEST_PATHS.repository / "tools" / "assemble_us_annual_release_dir.py"
    spec = importlib.util.spec_from_file_location(
        "assemble_us_annual_release_dir", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text())


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def base_folder(national_build, artifact_root, tmp_path):  # noqa: F811
    """A local copy of the national build's folder, H5 and weights included,
    as it is downloaded from ``releases/microcosm-us-2024-national/<build>/``."""
    folder = tmp_path / "downloaded" / "releases" / US_NATIONAL_RELEASE / RELEASE_ID
    shutil.copytree(national_build, folder)
    manifest = _read(folder / "release_manifest.json")
    for entry in manifest["artifacts"].values():
        if not (folder / entry["path"]).is_file():
            shutil.copyfile(artifact_root / entry["path"], folder / entry["path"])
    return folder


def _write_candidate(base_folder, directory, *, base_dataset="microcosm_us_2024"):
    """The files the candidate writer leaves for one projected year, 2025."""
    directory.mkdir()
    manifest = _read(base_folder / "release_manifest.json")
    base_key = manifest["default_datasets"]["national"]
    base_h5 = base_folder / manifest["artifacts"][base_key]["path"]
    shutil.copyfile(base_h5, directory / f"{base_dataset}.h5")
    projected = directory / "microcosm_us_2025.h5"
    _h5(projected, 2025)
    receipt = directory / "projection_2025.json"
    _write(receipt, {"base_year": 2024, "year": 2025, "factors": {}})
    with h5py.File(base_h5, "r") as store:
        rows = {entity: len(store[f"{entity}/table"]) for entity in _ENTITIES}
    runtime = {
        "policyengine-us": manifest["built_with_model_package"]["version"],
        "policyengine-core": manifest["built_with_core_package"]["version"],
    }
    evidence = {
        "schema_version": 1,
        "kind": "us_annual_static_aging_candidate",
        "status": "complete",
        "base": {
            "dataset": base_dataset,
            "year": 2024,
            "parent_release": manifest["build"],
            "path": str(base_h5),
            "sha256": _sha(base_h5),
        },
        "model": {
            "version": runtime["policyengine-us"],
            "commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
        },
        "runtime": {"versions": runtime},
        "metadata": {
            "dataset_years": {
                base_dataset: {"2024": base_dataset, "2025": "microcosm_us_2025"}
            }
        },
        "artifacts": {
            base_dataset: {
                "path": f"{base_dataset}.h5",
                "sha256": _sha(base_h5),
                "year": 2024,
                "rows": rows,
            },
            "microcosm_us_2025": {
                "path": projected.name,
                "sha256": _sha(projected),
                "year": 2025,
                "rows": rows,
                "projection_receipt": {"path": receipt.name, "sha256": _sha(receipt)},
            },
        },
    }
    _write(directory / "build_status.json", {"status": "complete"})
    _write(directory / "annual_manifest.json", evidence)
    return directory


def _write_acceptance(candidate, path):
    evidence = _read(candidate / "annual_manifest.json")
    _write(
        path,
        {
            "schema_version": 1,
            "kind": "us_annual_projection_acceptance",
            "status": "passed",
            "projection_manifest_sha256": _sha(candidate / "annual_manifest.json"),
            "model": evidence["model"],
            "runtime": evidence["runtime"]["versions"],
            "years": {
                year: {
                    "dataset": key,
                    "sha256": evidence["artifacts"][key]["sha256"],
                    "checks": {name: "passed" for name in _CHECKS},
                }
                for year, key in next(
                    iter(evidence["metadata"]["dataset_years"].values())
                ).items()
            },
        },
    )
    return path


@pytest.fixture
def inputs(base_folder, tmp_path):
    candidate = _write_candidate(base_folder, tmp_path / "candidate")
    acceptance = _write_acceptance(candidate, tmp_path / "acceptance.json")
    return {
        "base": base_folder,
        "candidate": candidate,
        "acceptance": acceptance,
        "out": tmp_path / "assembled",
    }


def _assemble(tool, inputs, monkeypatch):
    monkeypatch.setattr(tool, "_utc_now", lambda: _ASSEMBLED_AT)
    return tool.main(
        [
            "--base-build-dir",
            str(inputs["base"]),
            "--candidate-dir",
            str(inputs["candidate"]),
            "--acceptance",
            str(inputs["acceptance"]),
            "--out",
            str(inputs["out"]),
        ]
    )


def test_the_assembler_writes_an_annual_build_that_passes_the_contract(
    inputs, monkeypatch, capsys
):
    tool = _load_tool()

    exit_code = _assemble(tool, inputs, monkeypatch)

    summary = json.loads(capsys.readouterr().out)
    out = inputs["out"]
    annual_dir = out / "releases" / US_ANNUAL_RELEASE
    artifacts = out / "artifacts"
    candidate_sha = _sha(inputs["candidate"] / "annual_manifest.json")
    build = f"{RELEASE_ID}-annual-20261110T120000Z-{candidate_sha[:8]}"
    assert exit_code == 0
    assert summary["release"] == US_ANNUAL_RELEASE
    assert summary["build"] == build
    assert summary["release_dir"] == str(annual_dir)
    assert summary["artifact_root"] == str(artifacts)
    commands = publication_commands(
        annual_dir, repo_id=_REPOSITORY, artifact_root=artifacts
    )
    assert summary["publish_command"] == commands["publish_command"]
    assert summary["promote_command"] == commands["promote_command"]
    assert validate_release_dir(annual_dir, artifact_root=artifacts) == []

    base = _read(inputs["base"] / "release_manifest.json")
    manifest = _read(annual_dir / "release_manifest.json")
    build_manifest = _read(annual_dir / "build_manifest.json")
    assert (manifest["release"], manifest["build"]) == (US_ANNUAL_RELEASE, build)
    assert (build_manifest["release"], build_manifest["build"]) == (
        US_ANNUAL_RELEASE,
        build,
    )
    base_dataset = base["artifacts"]["microcosm_us_2024"]
    assert manifest["metadata"]["annual_projection_base"] == {
        "repo_id": _REPOSITORY,
        "release": US_NATIONAL_RELEASE,
        "build": RELEASE_ID,
        "path": "microcosm_us_2024.h5",
        "sha256": base_dataset["sha256"],
    }
    assert manifest["metadata"]["dataset_years"] == {
        "microcosm_us_2024": {"2024": "microcosm_us_2024", "2025": "microcosm_us_2025"}
    }
    assert all("revision" not in entry for entry in manifest["artifacts"].values())
    # The base build's files are carried unchanged, its H5 under its own name.
    for key, entry in base["artifacts"].items():
        assert manifest["artifacts"][key] == entry
        carried = (
            artifacts / entry["path"]
            if entry["kind"] == "microdata"
            else annual_dir / entry["path"]
        )
        assert carried.read_bytes() == (inputs["base"] / entry["path"]).read_bytes()
    assert (annual_dir / "annual_manifest.json").read_bytes() == (
        inputs["candidate"] / "annual_manifest.json"
    ).read_bytes()
    assert (annual_dir / "annual_acceptance.json").read_bytes() == (
        inputs["acceptance"].read_bytes()
    )
    assert (annual_dir / "projection_2025.json").read_bytes() == (
        inputs["candidate"] / "projection_2025.json"
    ).read_bytes()
    assert (artifacts / "microcosm_us_2025.h5").read_bytes() == (
        inputs["candidate"] / "microcosm_us_2025.h5"
    ).read_bytes()
    assert manifest["artifacts"]["microcosm_us_2025"]["role"] == "dataset"
    assert not (annual_dir / "build_status.json").exists()
    assert sorted(path.name for path in out.iterdir()) == ["artifacts", "releases"]


def _break(problem, inputs):
    candidate = inputs["candidate"]
    evidence_path = candidate / "annual_manifest.json"
    if problem == "base_dataset":
        shutil.rmtree(candidate)
        _write_candidate(inputs["base"], candidate, base_dataset="populace_us_2024")
    elif problem in {"base_sha256", "parent_release"}:
        evidence = _read(evidence_path)
        if problem == "base_sha256":
            evidence["base"]["sha256"] = "0" * 64
        else:
            evidence["base"]["parent_release"] = "populace-us-2024-0000000-20260101"
        _write(evidence_path, evidence)
    elif problem == "not_national":
        path = inputs["base"] / "release_manifest.json"
        manifest = _read(path)
        manifest["release"] = "microcosm-us-2024-evidence"
        _write(path, manifest)
        return
    elif problem == "existing_output":
        (inputs["out"] / "releases" / US_ANNUAL_RELEASE).mkdir(parents=True)
    _write_acceptance(candidate, inputs["acceptance"])
    if problem == "acceptance":
        acceptance = _read(inputs["acceptance"])
        acceptance["projection_manifest_sha256"] = "0" * 64
        _write(inputs["acceptance"], acceptance)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("base_dataset", "--base-dataset"),
        ("base_sha256", "sha256"),
        ("parent_release", "parent_release"),
        ("acceptance", "acceptance"),
        ("not_national", US_NATIONAL_RELEASE),
        ("existing_output", "already exists"),
    ],
)
def test_the_assembler_refuses_inputs_that_do_not_belong_together(
    inputs, monkeypatch, problem, message
):
    tool = _load_tool()
    _break(problem, inputs)

    with pytest.raises(SystemExit, match=message):
        _assemble(tool, inputs, monkeypatch)

    assert not (inputs["out"] / "artifacts").exists()
    if problem != "existing_output":
        assert not (inputs["out"] / "releases").exists()


def test_a_build_that_fails_the_contract_is_not_moved_into_place(inputs, monkeypatch):
    tool = _load_tool()
    acceptance = _read(inputs["acceptance"])
    acceptance["years"]["2025"]["checks"]["runtime"] = "failed"
    _write(inputs["acceptance"], acceptance)

    with pytest.raises(ReleaseContractError, match="passing acceptance"):
        _assemble(tool, inputs, monkeypatch)

    assert not (inputs["out"] / "releases").exists()
    assert not (inputs["out"] / "artifacts").exists()
    assert list(inputs["out"].iterdir()) == []


def test_the_assembler_refreshes_the_checksum_ledger_and_drops_manifest_artifacts(
    inputs, monkeypatch, capsys
):
    """The base build's ledger keeps its lines for the carried files, gets the
    rewritten manifests' digests and a line for each added file. A manifest
    the base build declared as an artifact is left to the publisher, which
    adds both manifests itself."""
    tool = _load_tool()
    base = inputs["base"]
    manifest_path = base / "release_manifest.json"
    manifest = _read(manifest_path)
    manifest["artifacts"]["build_manifest"] = {
        "kind": "diagnostics",
        "path": "build_manifest.json",
        "repo_id": _REPOSITORY,
        "sha256": _sha(base / "build_manifest.json"),
        "role": "build_manifest",
    }
    _write(manifest_path, manifest)
    ledgered = sorted(
        {entry["path"] for entry in manifest["artifacts"].values()}
        | {"release_manifest.json"}
    )
    (base / "sha256sums.txt").write_text(
        "".join(f"{_sha(base / name)}  {name}\n" for name in ledgered)
    )

    assert _assemble(tool, inputs, monkeypatch) == 0

    annual_dir = inputs["out"] / "releases" / US_ANNUAL_RELEASE
    artifacts = inputs["out"] / "artifacts"
    assembled = _read(annual_dir / "release_manifest.json")
    assert "build_manifest.json" not in {
        entry["path"] for entry in assembled["artifacts"].values()
    }
    lines = (annual_dir / "sha256sums.txt").read_text().splitlines()
    digests = {line.split("  ")[1]: line.split("  ")[0] for line in lines}
    assert set(digests) == set(ledgered) | {
        "annual_manifest.json",
        "annual_acceptance.json",
        "projection_2025.json",
        "microcosm_us_2025.h5",
    }
    for name, digest in digests.items():
        local = annual_dir / name if (annual_dir / name).is_file() else artifacts / name
        assert digest == _sha(local), name
    assert validate_release_dir(annual_dir, artifact_root=artifacts) == []
    capsys.readouterr()


# --- A local-area base: the County Files (microcosm#922) --------------------


@pytest.fixture
def local_inputs(tmp_path):
    """A local-area build as the ACS package stage writes it (its H5 one level
    above its release directory), a candidate for 2026 and 2027 and a passing
    acceptance report."""
    package = tmp_path / "package"
    base = local_build_folder(package)
    candidate = write_local_candidate(base, package, tmp_path / "candidate")
    acceptance = tmp_path / "acceptance.json"
    _write(acceptance, acceptance_report(candidate / "annual_manifest.json"))
    return {
        "base": base,
        "root": package,
        "candidate": candidate,
        "acceptance": acceptance,
        "out": tmp_path / "assembled",
    }


def _assemble_local(tool, inputs, monkeypatch, *, root=True):
    monkeypatch.setattr(tool, "_utc_now", lambda: _ASSEMBLED_AT)
    return tool.main(
        [
            "--base-build-dir",
            str(inputs["base"]),
            *(["--base-artifact-root", str(inputs["root"])] if root else []),
            "--candidate-dir",
            str(inputs["candidate"]),
            "--acceptance",
            str(inputs["acceptance"]),
            "--out",
            str(inputs["out"]),
        ]
    )


def test_the_assembler_writes_a_local_annual_build_that_passes_the_contract(
    local_inputs, monkeypatch, capsys
):
    tool = _load_tool()

    assert _assemble_local(tool, local_inputs, monkeypatch) == 0

    summary = json.loads(capsys.readouterr().out)
    out = local_inputs["out"]
    annual_dir = out / "releases" / US_LOCAL_ANNUAL_RELEASE_ID
    artifacts = out / "artifacts"
    candidate_sha = _sha(local_inputs["candidate"] / "annual_manifest.json")
    build = f"{LOCAL_BUILD}-annual-20261110T120000Z-{candidate_sha[:8]}"
    assert summary["release"] == US_LOCAL_ANNUAL_RELEASE_ID
    assert summary["build"] == build
    assert summary["base"]["release"] == US_LOCAL_RELEASE_ID
    assert summary["years"] == ["2024", "2026", "2027"]
    assert validate_release_dir(annual_dir, artifact_root=artifacts) == []

    base = _read(local_inputs["base"] / "release_manifest.json")
    manifest = _read(annual_dir / "release_manifest.json")
    assert (manifest["release"], manifest["build"]) == (
        US_LOCAL_ANNUAL_RELEASE_ID,
        build,
    )
    assert manifest["default_datasets"] == {}
    assert manifest["dataset_role"] == base["dataset_role"]
    assert manifest["metadata"]["annual_projection_base"] == {
        "repo_id": _REPOSITORY,
        "release": US_LOCAL_RELEASE_ID,
        "build": LOCAL_BUILD,
        "path": f"{LOCAL_DATASET_KEY}.h5",
        "sha256": base["artifacts"][LOCAL_DATASET_KEY]["sha256"],
    }
    # The base build's files are carried unchanged; its H5 comes from the
    # artifact root, never from the candidate, which holds no copy.
    assert not (local_inputs["candidate"] / f"{LOCAL_DATASET_KEY}.h5").exists()
    for key, entry in base["artifacts"].items():
        assert manifest["artifacts"][key] == entry
        source = (
            local_inputs["root"] / entry["path"]
            if entry["kind"] == "microdata"
            else local_inputs["base"] / entry["path"]
        )
        carried = (
            artifacts / entry["path"]
            if entry["kind"] == "microdata"
            else annual_dir / entry["path"]
        )
        assert carried.read_bytes() == source.read_bytes()
    for year in (2026, 2027):
        name = f"microcosm_us_{year}_acs_local.h5"
        assert (artifacts / name).read_bytes() == (
            local_inputs["candidate"] / name
        ).read_bytes()
        assert manifest["artifacts"][f"microcosm_us_{year}_acs_local"]["role"] == (
            "dataset"
        )
        assert (annual_dir / f"projection_{year}.json").is_file()
    ledger = {
        line.split("  ")[1]
        for line in (annual_dir / "sha256sums.txt").read_text().splitlines()
    }
    assert {
        f"{LOCAL_DATASET_KEY}.h5",
        "microcosm_us_2026_acs_local.h5",
        "microcosm_us_2027_acs_local.h5",
        "annual_manifest.json",
        "annual_acceptance.json",
    } <= ledger


def test_a_downloaded_local_build_folder_holds_its_own_h5(
    local_inputs, monkeypatch, capsys
):
    """A local build downloaded from releases/microcosm-us-2024-local/<build>/
    holds its H5 in the folder, so no artifact root is needed."""
    tool = _load_tool()
    name = f"{LOCAL_DATASET_KEY}.h5"
    shutil.move(local_inputs["root"] / name, local_inputs["base"] / name)

    assert _assemble_local(tool, local_inputs, monkeypatch, root=False) == 0

    capsys.readouterr()
    annual_dir = local_inputs["out"] / "releases" / US_LOCAL_ANNUAL_RELEASE_ID
    artifacts = local_inputs["out"] / "artifacts"
    assert validate_release_dir(annual_dir, artifact_root=artifacts) == []


def _break_local(problem, inputs):
    evidence_path = inputs["candidate"] / "annual_manifest.json"
    if problem == "no_artifact_root":
        return
    if problem == "defaults":
        path = inputs["base"] / "release_manifest.json"
        manifest = _read(path)
        manifest["default_datasets"] = {"national": LOCAL_DATASET_KEY}
        _write(path, manifest)
        return
    evidence = _read(evidence_path)
    if problem == "parent_release":
        evidence["base"]["parent_release"] = "populace-us-2024-other"
    elif problem == "base_sha256":
        evidence["base"]["sha256"] = "0" * 64
    elif problem == "base_dataset":
        evidence["base"]["dataset"] = "microcosm_us_2024"
    _write(evidence_path, evidence)
    _write(inputs["acceptance"], acceptance_report(evidence_path))


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("no_artifact_root", "does not exist"),
        ("defaults", "claim no default dataset"),
        ("parent_release", "parent_release"),
        ("base_sha256", "sha256"),
        ("base_dataset", LOCAL_DATASET_KEY),
    ],
)
def test_the_assembler_refuses_a_local_candidate_that_does_not_belong(
    local_inputs, monkeypatch, problem, message
):
    tool = _load_tool()
    _break_local(problem, local_inputs)

    with pytest.raises(SystemExit, match=message):
        _assemble_local(
            tool, local_inputs, monkeypatch, root=problem != "no_artifact_root"
        )

    assert not (local_inputs["out"] / "releases").exists()
    assert not (local_inputs["out"] / "artifacts").exists()


def test_a_local_annual_build_that_fails_the_contract_is_not_moved_into_place(
    local_inputs, monkeypatch
):
    tool = _load_tool()
    acceptance = _read(local_inputs["acceptance"])
    acceptance["years"]["2027"]["checks"]["runtime"] = "failed"
    _write(local_inputs["acceptance"], acceptance)

    with pytest.raises(ReleaseContractError, match="passing acceptance"):
        _assemble_local(tool, local_inputs, monkeypatch)

    assert not (local_inputs["out"] / "releases").exists()
    assert not (local_inputs["out"] / "artifacts").exists()


class _PublishedLocalBase:
    """A Hub serving a local base build's two manifests as they were when it
    was published, whatever happens to its local folder afterwards.

    With ``cache`` set, the base folder is the Hub client's cached snapshot:
    a download into the default cache returns the folder's file, edited or
    not, as the client does for a cached file whose ETag is unchanged. A
    forced download into its own ``cache_dir`` reads the published bytes.
    It also serves the repository's ``releases.json``, listing the base build.
    """

    def __init__(self, directory, base, *, cache=False):
        self.directory = directory
        self.base = base
        self.cache = cache
        self.documents = {
            name: _read(base / name)
            for name in ("release_manifest.json", "build_manifest.json")
        }
        self.documents["releases.json"] = local_releases_record()

    def hf_hub_download(
        self,
        *,
        repo_id,
        filename,
        repo_type,
        revision,
        cache_dir=None,
        force_download=False,
    ):
        name = Path(filename).name
        cached = name != "releases.json"
        if self.cache and cached and cache_dir is None and not force_download:
            return str(self.base / name)
        local = Path(cache_dir or self.directory) / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        _write(local, self.documents[name])
        return str(local)


def _drop_limitations(manifest):
    assert manifest["reviewed_limitations"]
    manifest["reviewed_limitations"] = []


def _other_core(manifest):
    manifest["built_with_core_package"]["version"] = "3.99.9"


def _relabel_reliability(manifest):
    manifest["artifacts"]["county_reliability"].update(
        kind="calibration", role="calibration_weights"
    )


#: Weights added to the base folder after publication, unlike the H5's.
_UNPUBLISHED_WEIGHTS = b"weights [1, 2, 3] beside the H5's [10, 20, 30]"


def _add_unpublished_weights(manifest):
    manifest["artifacts"]["unpublished_weights"] = {
        "kind": "calibration",
        "path": "unpublished_weights.npz",
        "repo_id": LOCAL_REPO,
        "sha256": hashlib.sha256(_UNPUBLISHED_WEIGHTS).hexdigest(),
        "role": "calibration_weights",
    }


@pytest.mark.parametrize("cache", [False, True], ids=["folder", "cached-folder"])
@pytest.mark.parametrize(
    ("edit", "claim"),
    [
        (None, None),
        (_drop_limitations, "reviewed_limitations"),
        (_other_core, "built_with_core_package"),
        (_relabel_reliability, "county_reliability"),
        (_add_unpublished_weights, r"declares \['unpublished_weights'\]"),
    ],
)
def test_an_assembled_local_annual_build_hands_over_only_its_published_base_claims(
    tmp_path, monkeypatch, capsys, edit, claim, cache
):
    """The assembler takes the annual build's claims and artifact entries
    from the base build's local folder. A folder edited after its base was
    published still assembles and passes the contract, so the hand-over
    check must refuse it (review of #1627), including when the folder is the
    Hub client's cached snapshot; an unedited folder hands over."""
    from microcosm.data import annual_projections

    package = tmp_path / "package"
    base = local_build_folder(package)
    hub = _PublishedLocalBase(tmp_path / "hub", base, cache=cache)
    if edit is _add_unpublished_weights:
        (base / "unpublished_weights.npz").write_bytes(_UNPUBLISHED_WEIGHTS)
    if edit is not None:
        rewrite_json(base, package, "release_manifest.json", edit)
    candidate = write_local_candidate(base, package, tmp_path / "candidate")
    acceptance = tmp_path / "acceptance.json"
    _write(acceptance, acceptance_report(candidate / "annual_manifest.json"))
    inputs = {
        "base": base,
        "root": package,
        "candidate": candidate,
        "acceptance": acceptance,
        "out": tmp_path / "assembled",
    }

    assert _assemble_local(_load_tool(), inputs, monkeypatch) == 0

    capsys.readouterr()
    annual_dir = inputs["out"] / "releases" / US_LOCAL_ANNUAL_RELEASE_ID
    artifacts = inputs["out"] / "artifacts"
    assert validate_release_dir(annual_dir, artifact_root=artifacts) == []

    def hand_over():
        # As the publisher calls it, with the folder's artifact root.
        annual_projections.require_base_publication(
            hub, annual_dir, repo_id=LOCAL_REPO, artifact_root=artifacts
        )

    if claim is None:
        hand_over()
    else:
        with pytest.raises(ValueError, match=claim):
            hand_over()
