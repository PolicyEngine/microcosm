"""Fixtures for the compatibility recertification lane.

``parent`` is the publisher suite's synthetic US release (``release_dir``,
``artifact_root``) made a real published parent: its root artifacts carry the
digests its manifests pin, its critical diagnostics rows carry the release
tool's materializer metadata, and it passes today's release contract.
``candidate`` is the pending child the tool builds from it. Engine-free tests
replace the loader replay with :func:`fake_replay` and score documents with
:func:`score_document`; none of these values is a measurement.
"""

# ruff: noqa: F401, F403, F405

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from microcosm.data import compatibility_recertification as recert
from microcosm.data.contract import validate_release_dir
from test_support.microcosm_data.release import *
from test_support.microcosm_data.release import _bind_stored_input_evidence

CHILD_ID = "populace-us-2024-9f1260b-recert-pe-us-2.60.4"


def _identity(version, digit):
    return {
        "version": version,
        "files": 10,
        "content_sha256": digit * 64,
        "wheel_sha256": chr(ord(digit) + 1) * 64,
    }


#: Receipt-shaped identities (no PyPI flag; the replay does not check PyPI).
TESTED_PACKAGES = {
    "policyengine-us": _identity("2.60.4", "1"),
    "policyengine-core": _identity("3.33.3", "3"),
    "spm-calculator": _identity("1.0.0", "7"),
}
BASELINE_PACKAGES = {
    "policyengine-us": _identity("1.729.0", "a"),
    "policyengine-core": _identity("3.19.0", "c"),
    "spm-calculator": _identity("1.0.0", "7"),
}
HOUSEHOLDS = 3


def scored_runtime(packages):
    """Score-document identities: the receipt's plus the PyPI verification."""

    return {
        name: {**identity, "pypi_sha256_verified": True}
        for name, identity in packages.items()
    }


def write_fixture_h5(path: Path, households: int = HOUSEHOLDS) -> None:
    """A real HDF5 file whose metadata names ``households`` households.

    Engine-free tests never load it; validation reads only its household
    count from metadata, and digests its bytes.
    """

    import h5py
    import numpy as np

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "household/table",
            data=np.array(
                [(index, index + 1) for index in range(households)],
                dtype=[("index", "<i8"), ("household_id", "<i8")],
            ),
        )


DECLARER = "Max Ghenis (fixture declarer, not a real declaration)"


def _soi(source, base, mode, *, itemized=False):
    metadata = {
        "materializer": "irs_soi_slice",
        "source_variable": source,
        "variable": source,
        ("base_variables" if "," in base else "base_variable"): base,
        "measure_mode": mode,
        "agi_lower_bound": "-inf",
        "agi_upper_bound": "inf",
        "filing_status": "All",
    }
    if itemized:
        metadata["itemized_only"] = "true"
    return metadata


_HT2 = "irs_soi.ty2022.historic_table_2.us.all."
_EITC = "irs_soi.ty2024.filing_season_week47.eitc_all_returns.earned_income_credit."
_T21 = "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
CRITICAL_METADATA = {
    f"{_HT2}income_tax_liability_amount@2024": _soi("income_tax", "income_tax", "sum"),
    f"{_HT2}income_tax_liability_returns@2024": _soi(
        "income_tax", "income_tax", "indicator_sum"
    ),
    "ssa_supplement.cy2024.oasdi_ssi_payments.social_security_benefits."
    "payment_amount@2024": {
        "materializer": "policyengine_variable",
        "base_variable": "social_security",
        "measure_mode": "sum",
    },
    f"{_HT2}ctc_amount@2024": _soi("ctc", "ctc,ctc_limiting_tax_liability", "sum"),
    f"{_HT2}ctc_claims@2024": _soi(
        "ctc", "ctc,ctc_limiting_tax_liability", "indicator_sum"
    ),
    f"{_HT2}actc_amount@2024": _soi("refundable_ctc", "refundable_ctc", "sum"),
    f"{_HT2}actc_claims@2024": _soi(
        "refundable_ctc", "refundable_ctc", "indicator_sum"
    ),
    f"{_EITC}total_earned_income_credit_amount@2024": _soi("eitc", "eitc", "sum"),
    f"{_EITC}total_earned_income_credit_returns@2024": _soi(
        "eitc", "eitc", "indicator_sum"
    ),
    f"{_HT2}premium_tax_credit_amount@2024": _soi(
        "assigned_aca_ptc", "assigned_aca_ptc", "sum"
    ),
    f"{_HT2}premium_tax_credit_returns@2024": _soi(
        "assigned_aca_ptc", "assigned_aca_ptc", "indicator_sum"
    ),
    f"{_HT2}taxable_social_security_amount@2024": _soi(
        "taxable_social_security", "tax_unit_taxable_social_security", "sum"
    ),
    f"{_HT2}taxable_social_security_returns@2024": _soi(
        "taxable_social_security", "tax_unit_taxable_social_security", "indicator_sum"
    ),
    f"{_T21}home_mortgage_interest_returns@2024": _soi(
        "deductible_mortgage_interest",
        "deductible_mortgage_interest_tax_unit",
        "indicator_sum",
        itemized=True,
    ),
    f"{_T21}home_mortgage_interest_amount@2024": _soi(
        "deductible_mortgage_interest",
        "deductible_mortgage_interest_tax_unit",
        "sum",
        itemized=True,
    ),
    f"{_HT2}itemized_deductions_amount@2024": _soi(
        "itemized_taxable_income_deductions",
        "itemized_taxable_income_deductions",
        "sum",
        itemized=True,
    ),
    f"{_HT2}limited_state_local_taxes_amount@2024": _soi(
        "salt_deduction", "salt_deduction", "sum", itemized=True
    ),
    f"{_HT2}medical_dental_expense_amount@2024": _soi(
        "medical_expense_deduction", "medical_expense_deduction", "sum", itemized=True
    ),
    "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024": _soi(
        "employment_income", "employment_income", "sum"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite(path: Path, update) -> None:
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload))


def make_published_parent(release_dir: Path, artifact_root: Path) -> None:
    """Pin the fixture's real artifact bytes and materializer metadata."""

    h5 = artifact_root / "populace_us_2024.h5"
    npz = artifact_root / "populace_us_2024_calibration.npz"
    write_fixture_h5(h5)
    npz.write_bytes(b"synthetic parent calibration weights")
    diagnostics = release_dir / "calibration_diagnostics.json"

    def annotate(payload):
        for target in payload["targets"]:
            if target["name"] in CRITICAL_METADATA:
                target["metadata"] = {
                    **target["metadata"],
                    **CRITICAL_METADATA[target["name"]],
                }
        payload["build"] = {
            **payload.get("build", {}),
            "post_export_scoring": {"spm": {"geography_kind": "county"}},
        }

    _rewrite(diagnostics, annotate)

    def pin_release(payload):
        artifacts = payload["artifacts"]
        artifacts["populace_us_2024"]["sha256"] = _sha(h5)
        artifacts["populace_us_2024_calibration"]["sha256"] = _sha(npz)
        artifacts["calibration_diagnostics"]["sha256"] = _sha(diagnostics)
        payload["build"]["built_at"] = "2026-09-30T15:04:01+00:00"

    def pin_build(payload):
        payload["dataset"]["sha256"] = _sha(h5)
        payload["calibration"]["sha256"] = _sha(npz)

    _rewrite(release_dir / "release_manifest.json", pin_release)
    _rewrite(release_dir / "build_manifest.json", pin_build)
    _bind_stored_input_evidence(release_dir)
    validate_release_dir(release_dir, artifact_root=artifact_root)


@pytest.fixture
def parent(release_dir: Path, artifact_root: Path) -> tuple[Path, Path]:
    make_published_parent(release_dir, artifact_root)
    return release_dir, artifact_root


FIXTURE_PRODUCER = {
    "repository": "https://github.com/PolicyEngine/microcosm",
    "git_commit": "a" * 40,
    "git_dirty": False,
    "runtime": {"python": "3.13.0"},
    "source_files_sha256": dict.fromkeys(recert.PRODUCER_SOURCE_FILES, "b" * 64),
}


def load_tool():
    import tools.build_us_compatibility_recertification as tool

    return tool


@pytest.fixture
def candidate(parent, tmp_path, monkeypatch) -> tuple[Path, Path]:
    """The pending child the tool builds; returns (release_dir, artifact_root)."""

    tool = load_tool()
    monkeypatch.setattr(tool, "_producer_identity", lambda: dict(FIXTURE_PRODUCER))
    release_dir, artifact_root = parent
    output = tmp_path / "candidate"
    tool.build_candidate(
        parent_release_dir=release_dir,
        parent_artifact_root=artifact_root,
        output_dir=output,
        release_id=CHILD_ID,
    )
    return output / "releases" / CHILD_ID, output / "artifacts"


def fake_replay(changes=None, packages=None):
    """A loader replay stand-in returning a fixed receipt (not a measurement)."""

    changes = changes or {
        recert.CHANGE_IGNORED: [
            {"column": "first_home_mortgage_interest", "tables": ["tax_unit"]}
        ],
        recert.CHANGE_READ: [{"column": "tax_unit_role_input", "tables": ["person"]}],
    }
    receipt = {
        "schema_version": recert.RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "packages": packages or TESTED_PACKAGES,
        "stored_inputs": {"column_changes": changes},
    }

    def replay(candidate_h5, *, require_wheels=False, compatibility_wheels=()):
        return json.loads(json.dumps(receipt))

    return replay


def score_document(
    release_dir: Path,
    *,
    packages=TESTED_PACKAGES,
    variant=recert.VARIANT_AS_STORED,
    withheld=(),
    value=lambda metric_id: 1_000.0,
    stored_input_passed=True,
    households=HOUSEHOLDS,
    spm=None,
) -> dict:
    """A score document shaped as the tool writes it, with invented values."""

    from microcosm.data.stored_inputs import register_sha256

    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    definitions = recert.drift_metric_definitions(diagnostics)
    return {
        "schema_version": recert.SCORES_SCHEMA_VERSION,
        "kind": recert.SCORES_KIND,
        "dataset": report["dataset"],
        "variant": {"id": variant, "withheld_columns": sorted(withheld)},
        "metric_definitions_sha256": recert.canonical_sha256(definitions),
        "parent_diagnostics_sha256": recert.sha256_file(
            release_dir / "calibration_diagnostics.json"
        ),
        "runtime": {"packages": scored_runtime(packages)},
        "scoring": {
            "method": recert.SCORING_METHOD,
            "period": recert.SCORING_PERIOD,
            "spm": spm or diagnostics["build"]["post_export_scoring"]["spm"],
            "batch_households": 2_000,
            "n_households": HOUSEHOLDS,
            "passes": [
                {"batching": recert.BATCHING_HOUSEHOLD, "households": households},
                {"batching": recert.BATCHING_STATE, "households": households},
            ],
        },
        "engine": {
            "label": f"policyengine-us {packages['policyengine-us']['version']}",
            "stored_input_verdict": {
                "passed": stored_input_passed,
                "refused": [],
                "registered_non_variables": [],
                "register_sha256": register_sha256(),
            },
        },
        "metrics": {
            metric["metric_id"]: {"value": value(metric["metric_id"])}
            for metric in definitions["metrics"]
        },
        "filing_status": {},
        "tax_unit_roles": {},
        "bytecode": {"isolated": True},
        "code": dict(FIXTURE_PRODUCER),
    }


def write_scores(directory: Path, release_dir: Path, **overrides) -> dict[str, Path]:
    """Baseline, tested and counterfactual documents on disk, by role."""

    directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "baseline": score_document(
            release_dir,
            packages=BASELINE_PACKAGES,
            value=overrides.get("baseline", lambda metric_id: 1_000.0),
        ),
        "tested": score_document(
            release_dir, value=overrides.get("tested", lambda metric_id: 1_100.0)
        ),
        "counterfactual": score_document(
            release_dir,
            variant=recert.VARIANT_WITHOUT_NEWLY_READ,
            withheld=("tax_unit_role_input",),
            value=overrides.get("counterfactual", lambda metric_id: 1_040.0),
        ),
    }
    paths = {}
    for role, document in documents.items():
        paths[role] = directory / f"{role}.json"
        paths[role].write_text(json.dumps(document))
    return paths


class PathsInfoHub(FakeHub):
    """FakeHub with ``get_paths_info``: LFS for H5/NPZ, git blob ids otherwise."""

    def get_paths_info(self, *, repo_id, paths, revision=None, repo_type=None):
        from types import SimpleNamespace

        assert repo_type == "dataset" and repo_id == self.repo_id
        files = self._commits[self._refs.get(revision or "main", revision)]
        infos = []
        for path in paths:
            if path not in files:
                continue
            content = files[path]
            lfs = (
                SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())
                if path.endswith((".h5", ".npz"))
                else None
            )
            infos.append(
                SimpleNamespace(
                    path=path, lfs=lfs, blob_id=recert.git_blob_sha1(content)
                )
            )
        return infos


def seed_published_parent(hub, parent_dir: Path, artifact_root: Path) -> None:
    """Publish the parent's files at its tag on the fake Hub, as the publisher did."""

    tag = parent_dir.name
    manifest = json.loads((parent_dir / "release_manifest.json").read_text())
    files = {
        f"releases/{tag}/{name}": (parent_dir / name).read_bytes()
        for name in ("release_manifest.json", "build_manifest.json")
    }
    for entry in manifest["artifacts"].values():
        path = entry["path"]
        if (parent_dir / path).is_file():
            files[f"releases/{tag}/{path}"] = (parent_dir / path).read_bytes()
        else:
            files[path] = (artifact_root / path).read_bytes()
    hub._commits["commit-parent"] = files
    hub._refs[tag] = "commit-parent"


@pytest.fixture
def paths_hub(tmp_path) -> PathsInfoHub:
    fake = PathsInfoHub()
    fake._download_dir = tmp_path / "hub-cache"
    return fake


def certify(candidate, tmp_path, monkeypatch, *, specifier="policyengine-us==2.60.4"):
    """Certify the candidate with the fake replay and fixture scores."""

    release_dir, artifact_root = candidate
    monkeypatch.setattr(recert, "run_recertification_loader_replay", fake_replay())
    monkeypatch.setattr(recert, "_check_producer", lambda code, failures, label: None)
    scores = write_scores(tmp_path / "scores", release_dir)
    output = tmp_path / "certified" / release_dir.name
    recert.certify_recertification(
        release_dir,
        output,
        artifact_root=artifact_root,
        compatibility_wheels=(),
        baseline_scores=scores["baseline"],
        tested_scores=scores["tested"],
        counterfactual_scores=scores["counterfactual"],
        compatible_model_specifier=specifier,
        compatibility_claim_declared_by=DECLARER,
    )
    return output, artifact_root
