"""Publishing behavior: contract-gated uploads and a last-written pointer.

The fake Hub client records every upload in order, so the suite asserts the
real guarantees — an invalid release uploads nothing, and ``latest.json``
lands strictly after the files it points at — rather than implementation
details.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from populace.data import ReleaseContractError
from populace.data.contract import (
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    required_release_files,
)
from populace.data.release import (
    LATEST_POINTER_PATH,
    LATEST_POINTER_SCHEMA_VERSION,
    latest_pointer_payload,
    latest_release,
    publish_release,
)


@pytest.fixture(autouse=True)
def _no_slack_webhook(monkeypatch):
    """Keep publish_release's release alert hermetic: never post to a real
    webhook if the dev/CI environment happens to have one set."""
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_UK", raising=False)


RELEASE_ID = "populace-us-2024-9f1260b-20260611"
GIT_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
DATASET_SHA = "cfe0edd307e479920c6a177b316f944bc27839f89e081ede5218a32d6b6b16d8"
CALIBRATION_SHA = "ac31f2be76a0f8dc4da89b6935aa4b8b1b2e1bd4eb3d03b809333084f25b376e"
TARGET_SURFACE_SHA = "e" * 64
REGISTRY_VERSION = "registryabc123"
TARGET_COUNT = 18

DEDUCTION_CRITICAL_TARGETS = (
    (
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount",
        1_000_000_000_000.0,
        1_020_000_000_000.0,
        "itemized_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount",
        120_000_000_000.0,
        121_000_000_000.0,
        "salt_deduction_total",
    ),
    (
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount@2024",
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount",
        80_000_000_000.0,
        69_000_000_000.0,
        "medical_expense_deduction_total",
    ),
)


def _calibration_diagnostics() -> dict:
    return {
        "schema_version": 5,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": TARGET_COUNT,
            "n_records": 2,
            "constraint_matrix": {"rows": 1, "columns": 2, "nnz": 2},
            "sha256": TARGET_SURFACE_SHA,
            "names_sha256": "b" * 64,
            "values_sha256": "f" * 64,
        },
        "target_registry": {
            "country": "us",
            "version": REGISTRY_VERSION,
            "n_specs": TARGET_COUNT,
        },
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": [
            _target_row(
                "population@2024",
                target_name="population",
                target=1.0,
                initial_estimate=0.8,
                final_estimate=1.0,
                relative_error=0.0,
                family="cbo",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_amount@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all.income_tax_liability_amount"
                ),
                target=2_105_345_646_000.0,
                initial_estimate=2_000_000_000_000.0,
                final_estimate=2_067_762_165_736.424,
                relative_error=-0.0178514536722185,
                family="irs_soi",
                target_role="federal_income_tax_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all."
                "income_tax_liability_returns@2024",
                target_name=(
                    "irs_soi.ty2022.historic_table_2.us.all."
                    "income_tax_liability_returns"
                ),
                target=113_562_590.0,
                initial_estimate=105_421_734.40619682,
                final_estimate=105_437_267.69738781,
                relative_error=-0.07154928663226319,
                family="irs_soi",
            ),
            _target_row(
                "ssa_supplement.cy2024.oasdi_ssi_payments."
                "social_security_benefits.payment_amount@2024",
                target_name=(
                    "ssa_supplement.cy2024.oasdi_ssi_payments."
                    "social_security_benefits.payment_amount"
                ),
                target=1_471_195_000_000.0,
                initial_estimate=1_541_646_703_291.2527,
                final_estimate=1_541_540_768_722.367,
                relative_error=0.047815394099604024,
                family="ssa",
                target_role="social_security_total",
            ),
            _target_row(
                "irs_soi.ty2022.historic_table_2.us.all.ctc_amount@2024",
                target_name="irs_soi.ty2022.historic_table_2.us.all.ctc_amount",
                target=82_863_353_000.0,
                initial_estimate=132_000_000_000.0,
                final_estimate=90_000_000_000.0,
                relative_error=(90_000_000_000.0 - 82_863_353_000.0) / 82_863_353_000.0,
                family="irs_soi",
                target_role="ctc_total",
            ),
            *additional_critical_credit_rows(),
            *deduction_critical_target_rows(),
            # The SOI Table 1.4 national dollar blanket (populace#462) needs
            # at least one Table 1.4 dollar row on the surface, within its
            # 25% blocking tolerance (the live Build M wages row).
            _target_row(
                "irs_soi.ty2023.table_1_4.all.wages_salaries_amount@2024",
                target_name="irs_soi.ty2023.table_1_4.all.wages_salaries_amount",
                target=10_773_360_188_645.0,
                initial_estimate=10_500_000_000_000.0,
                final_estimate=10_774_383_029_502.0,
                relative_error=(10_774_383_029_502.0 - 10_773_360_188_645.0)
                / 10_773_360_188_645.0,
                family="irs_soi",
            ),
        ],
    }


def additional_critical_credit_rows() -> list[dict]:
    rows = [
        (
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.ctc_claims",
            38_068_980.0,
            36_607_400.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_amount",
            33_858_000_000.0,
            33_501_200_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims@2024",
            "irs_soi.ty2022.historic_table_2.us.all.actc_claims",
            17_691_400.0,
            17_434_500.0,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_amount",
            69_041_649_000.0,
            58_954_970_066.74941,
        ),
        (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns@2024",
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            "earned_income_credit.total_earned_income_credit_returns",
            23_837_149.0,
            23_349_300.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount",
            53_910_190_000.0,
            56_821_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns",
            7_841_370.0,
            8_385_450.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_amount",
            455_904_900_000.0,
            454_551_000_000.0,
        ),
        (
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns@2024",
            "irs_soi.ty2022.historic_table_2.us.all.taxable_social_security_returns",
            24_475_100.0,
            24_472_900.0,
        ),
    ]
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
        )
        for name, target_name, target, final in rows
    ]


def deduction_critical_target_rows() -> list[dict]:
    return [
        _target_row(
            name,
            target_name=target_name,
            target=target,
            initial_estimate=target * 1.5,
            final_estimate=final,
            relative_error=(final - target) / target,
            family="irs_soi",
            target_role=target_role,
        )
        for name, target_name, target, final, target_role in DEDUCTION_CRITICAL_TARGETS
    ]


def _target_row(
    name: str,
    *,
    target_name: str,
    target: float,
    initial_estimate: float,
    final_estimate: float,
    relative_error: float,
    family: str,
    target_role: str | None = None,
) -> dict:
    metadata = {"target_role": target_role} if target_role else {}
    return {
        "name": name,
        "target_name": target_name,
        "period": 2024,
        "entity": "household",
        "measure": {"kind": "column", "name": "household_count"},
        "filter": None,
        "source": "Fixture admin target",
        "metadata": metadata,
        "target": target,
        "compiled_target": target,
        "initial_estimate": initial_estimate,
        "final_estimate": final_estimate,
        "relative_error": relative_error,
        "within_tolerance": None,
        "registry": {"family": family},
    }


def _source_coverage_diagnostics() -> dict:
    return {
        "schema_version": 1,
        "classification": "release_gate",
        "source_contract": {
            "name": "us_source_coverage",
            "ledger_commit": "5fa48f07436a806ad75ff76fd22cfb8613bddbe0",
        },
        "gate": {
            "name": "us_source_coverage",
            "passed": True,
            "failures": [],
        },
        "coverage_summary": {
            "hard_target": {
                "families": 9,
                "package_aliases": 38,
                "covered_package_aliases": 38,
                "missing_package_aliases": 0,
                "reviewed_excluded_package_aliases": 0,
            },
            "validation_only": {"families": 6, "activated_families": 0},
            "source_gap": {"families": 6, "missing_source_packages": 11},
        },
        "hard_target_families": {"population_age_sex": {}},
        "validation_only_families": {"census_cps_spm": {}},
        "source_gap_families": {"usda_wic": {}},
        "active_target_aliases": ["census-pep-2024-national-age-sex"],
        "active_target_families": [],
        "missing_hard_targets": [],
        "reviewed_exclusions": {},
        "validation_only_activated": [],
        "fiscal_target_sources": {
            "cbo": {
                "label": "Congressional Budget Office revenue projections",
                "target_count": 1,
                "sources": ["Census PEP 2024"],
                "reference_urls": ["https://example.test/source"],
            },
            "irs_soi": {
                "label": "IRS Statistics of Income",
                "target_count": 16,
                "sources": ["IRS SOI Historic Table 2"],
                "reference_urls": ["https://example.test/soi"],
            },
            "ssa": {
                "label": "Social Security Administration",
                "target_count": 1,
                "sources": ["SSA Annual Statistical Supplement"],
                "reference_urls": ["https://example.test/ssa"],
            },
        },
    }


class FakeHub:
    """Model atomic Hub commits, refs, and downloads with an ordered event log."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.tags: list[dict[str, str | None]] = []
        self.events: list[tuple[str, dict]] = []
        self._commit_number = 0
        self._commits: dict[str, dict[str, bytes]] = {"commit-0": {}}
        self._refs: dict[str, str] = {"main": "commit-0"}
        self.fail_main_commit = False

    @staticmethod
    def _content(path_or_fileobj) -> bytes:
        if isinstance(path_or_fileobj, bytes):
            return path_or_fileobj
        return Path(path_or_fileobj).read_bytes()

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        content = self._content(path_or_fileobj)
        self.uploads.append((path_in_repo, content))
        self._commit_number += 1
        commit = f"commit-{self._commit_number}"
        files = dict(self._commits[self._refs["main"]])
        files[path_in_repo] = content
        self._commits[commit] = files
        self._refs["main"] = commit
        self.events.append(("upload_file", {"path": path_in_repo, "commit": commit}))
        return {"commit_hash": commit}

    def create_branch(
        self,
        *,
        repo_id,
        branch,
        repo_type,
        revision=None,
        exist_ok=False,
    ) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        if branch in self._refs and not exist_ok:
            raise ValueError(f"branch exists: {branch}")
        base = revision or "main"
        self._refs[branch] = self._refs.get(base, base)
        self.events.append(("create_branch", {"branch": branch, "revision": revision}))

    def repo_info(self, *, repo_id, repo_type, revision=None) -> dict[str, str]:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        ref = revision or "main"
        return {"sha": self._refs[ref]}

    def create_commit(
        self,
        *,
        repo_id,
        operations,
        commit_message,
        repo_type,
        revision=None,
        parent_commit=None,
    ):
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        ref = revision or "main"
        current_commit = self._refs[ref]
        if parent_commit is not None:
            assert parent_commit == current_commit
        if ref == "main" and self.fail_main_commit:
            self.events.append(("create_commit_failed", {"revision": ref}))
            raise RuntimeError("injected main commit failure")
        files = dict(self._commits[current_commit])
        paths: list[str] = []
        for operation in operations:
            path = operation.path_in_repo
            content = self._content(operation.path_or_fileobj)
            files[path] = content
            paths.append(path)
            self.uploads.append((path, content))
        self._commit_number += 1
        commit = f"commit-{self._commit_number}"
        self._commits[commit] = files
        self._refs[ref] = commit
        self.events.append(
            (
                "create_commit",
                {
                    "revision": ref,
                    "paths": paths,
                    "commit": commit,
                    "message": commit_message,
                    "parent_commit": parent_commit,
                },
            )
        )
        return {"commit_hash": commit}

    def create_tag(
        self, *, repo_id, tag, repo_type, revision=None, exist_ok=False
    ) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        if tag in self._refs and not exist_ok:
            raise ValueError(f"tag exists: {tag}")
        self._refs[tag] = revision or self._refs["main"]
        self.tags.append({"tag": tag, "revision": revision})
        self.events.append(("create_tag", {"tag": tag, "revision": revision}))

    def delete_branch(self, *, repo_id, branch, repo_type) -> None:
        assert repo_type == "dataset"
        assert repo_id == "policyengine/populace-us"
        del self._refs[branch]
        self.events.append(("delete_branch", {"branch": branch}))

    def seed_main_file(self, path: str, content: bytes) -> None:
        """Install a test fixture without recording a publication event."""
        self._commits[self._refs["main"]][path] = content

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision=None) -> str:
        assert repo_type == "dataset"
        ref = revision or "main"
        commit = self._refs.get(ref, ref)
        try:
            content = self._commits[commit][filename]
        except KeyError as exc:
            raise FileNotFoundError(f"{filename}@{ref}") from exc
        local = self._download_dir / ref / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        return str(local)


class NonAtomicHub:
    """Expose only the legacy upload/tag surface of a backing fake Hub."""

    def __init__(self, backing: FakeHub) -> None:
        self.backing = backing

    def upload_file(self, **kwargs):
        return self.backing.upload_file(**kwargs)

    def create_tag(self, **kwargs):
        return self.backing.create_tag(**kwargs)


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    fake = FakeHub()
    fake._download_dir = tmp_path / "hub-cache"
    return fake


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "releases" / RELEASE_ID
    directory.mkdir(parents=True)
    (directory / "build_manifest.json").write_text(
        json.dumps(
            {
                "build_id": RELEASE_ID,
                "build_sha": GIT_COMMIT[:7],
                "code": {
                    "repository": "PolicyEngine/populace",
                    "git_commit": GIT_COMMIT,
                    "git_dirty": False,
                },
                "runtime": {
                    "python": "3.14.0",
                    "policyengine-us": "1.729.0",
                    "policyengine-core": "3.19.0",
                },
                "dataset": {
                    "filename": "populace_us_2024.h5",
                    "sha256": DATASET_SHA,
                },
                "calibration": {
                    "filename": "populace_us_2024_calibration.npz",
                    "sha256": CALIBRATION_SHA,
                    "target_surface": {
                        "sha256": TARGET_SURFACE_SHA,
                        "n_targets": TARGET_COUNT,
                    },
                    "target_registry": {
                        "version": REGISTRY_VERSION,
                        "n_specs": TARGET_COUNT,
                    },
                },
                "gates": {"parity_gaps": 0},
            }
        )
    )
    (directory / "calibration_diagnostics.json").write_text(
        json.dumps(_calibration_diagnostics())
    )
    (directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE).write_text(
        json.dumps(_source_coverage_diagnostics())
    )
    diagnostics_sha = _sha256(directory / "calibration_diagnostics.json")
    source_coverage_sha = _sha256(directory / US_SOURCE_COVERAGE_DIAGNOSTICS_FILE)
    (directory / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_package": {"name": "populace-data", "version": "0.1.0"},
                "default_datasets": {"national": "populace_us_2024"},
                "build": {
                    "build_id": RELEASE_ID,
                    "built_with_core_package": {
                        "name": "policyengine-core",
                        "version": "3.19.0",
                    },
                    "built_with_model_package": {
                        "name": "policyengine-us",
                        "version": "1.729.0",
                    },
                },
                "compatible_core_packages": [
                    {"name": "policyengine-core", "specifier": "==3.19.0"}
                ],
                "compatible_model_packages": [
                    {"name": "policyengine-us", "specifier": "==1.729.0"}
                ],
                "artifacts": {
                    "populace_us_2024": {
                        "kind": "microdata",
                        "path": "populace_us_2024.h5",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": DATASET_SHA,
                    },
                    "populace_us_2024_calibration": {
                        "kind": "calibration",
                        "path": "populace_us_2024_calibration.npz",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": CALIBRATION_SHA,
                    },
                    "calibration_diagnostics": {
                        "kind": "diagnostics",
                        "path": "calibration_diagnostics.json",
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": diagnostics_sha,
                    },
                    "us_source_coverage": {
                        "kind": "diagnostics",
                        "path": US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
                        "repo_id": "policyengine/populace-us",
                        "revision": RELEASE_ID,
                        "sha256": source_coverage_sha,
                    },
                },
            }
        )
    )
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "populace_us_2024.h5").write_bytes(b"h5 payload")
    (directory / "populace_us_2024_calibration.npz").write_bytes(b"npz payload")
    return directory


def test_pointer_payload_names_every_contract_file() -> None:
    payload = latest_pointer_payload(RELEASE_ID, updated_at="2026-06-11T13:53:15+00:00")
    assert payload["schema_version"] == LATEST_POINTER_SCHEMA_VERSION
    assert payload["release_id"] == RELEASE_ID
    assert set(payload["paths"]) == {
        name.removesuffix(".json") for name in required_release_files(RELEASE_ID)
    }
    assert (
        payload["paths"]["build_manifest"]
        == f"releases/{RELEASE_ID}/build_manifest.json"
    )
    assert (
        payload["paths"]["us_source_coverage"]
        == f"releases/{RELEASE_ID}/{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE}"
    )


def test_publish_release_announces_after_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """The alert fires from publish_release itself — every publish path, not
    just the CLI — with the released id and timestamp."""
    calls: list = []
    monkeypatch.setattr(
        "populace.data.release.notify_release",
        lambda repo_id, release_id, updated_at, **kw: calls.append(
            (repo_id, release_id, updated_at, kw)
        ),
    )
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert calls == [
        (
            "policyengine/populace-us",
            RELEASE_ID,
            "2026-06-11T13:53:15+00:00",
            {"warn_if_unset": True},
        )
    ]


def test_publish_release_notify_false_skips_alert(
    hub: FakeHub, release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(
        "populace.data.release.notify_release", lambda *a, **k: calls.append(a)
    )
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        notify=False,
    )
    assert calls == []


def test_publish_uploads_pointer_last(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert final_commit["revision"] == "main"
    assert final_commit["paths"][-1] == LATEST_POINTER_PATH
    for filename in required_release_files(RELEASE_ID):
        assert f"releases/{RELEASE_ID}/{filename}" in final_commit["paths"][:-1]


def test_publish_no_latest_never_touches_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
        update_latest=False,
    )
    # Immutable branch + tag flow is unchanged; the final main commit exists
    # (release copies + root artifacts) but carries NO pointer operation.
    assert [event for event, _ in hub.events] == [
        "create_branch",
        "create_commit",
        "create_tag",
        "delete_branch",
        "create_commit",
    ]
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert final_commit["revision"] == "main"
    assert LATEST_POINTER_PATH not in final_commit["paths"]
    assert final_commit["message"] == f"Publish non-default release {RELEASE_ID}"


def test_publish_commits_immutable_release_before_root_and_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert [event for event, _ in hub.events] == [
        "create_branch",
        "create_commit",
        "create_tag",
        "delete_branch",
        "create_commit",
    ]

    staging_branch = f"release-staging/{RELEASE_ID}"
    immutable = hub.events[1][1]
    assert immutable["revision"] == staging_branch
    assert set(immutable["paths"]) == {
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
        *{
            f"releases/{RELEASE_ID}/{filename}"
            for filename in required_release_files(RELEASE_ID)
        },
    }
    assert LATEST_POINTER_PATH not in immutable["paths"]

    tag = hub.events[2][1]
    assert tag == {"tag": RELEASE_ID, "revision": immutable["commit"]}
    assert hub.events[3] == ("delete_branch", {"branch": staging_branch})

    convenience = hub.events[4][1]
    assert convenience["revision"] == "main"
    assert convenience["parent_commit"] == "commit-0"
    assert convenience["paths"][-3:] == [
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
        LATEST_POINTER_PATH,
    ]
    assert hub.tags == [{"tag": RELEASE_ID, "revision": immutable["commit"]}]


def test_failed_main_commit_leaves_root_and_pointer_unchanged(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    old_pointer = b'{"release_id": "old-release"}'
    old_artifact = b"old certified root artifact"
    hub.seed_main_file(LATEST_POINTER_PATH, old_pointer)
    hub.seed_main_file("populace_us_2024.h5", old_artifact)
    hub.fail_main_commit = True

    with pytest.raises(RuntimeError, match="injected main commit failure"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )

    main_commit = hub._commits[hub._refs["main"]]
    assert main_commit[LATEST_POINTER_PATH] == old_pointer
    assert main_commit["populace_us_2024.h5"] == old_artifact
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-1"}]
    assert hub.events[-1][0] == "create_commit_failed"


def test_non_atomic_backend_is_refused_before_remote_mutation(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(TypeError, match="immutable-first and atomic"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=NonAtomicHub(hub),
            artifact_root=artifact_root,
        )

    assert hub.events == []
    assert hub.uploads == []


def test_publish_uploads_manifest_release_diagnostics_from_release_dir(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (release_dir / "reform_validation.json").write_text('{"schema_version": 1}')
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["reform_validation"] = {
        "kind": "diagnostics",
        "path": "reform_validation.json",
        "repo_id": "policyengine/populace-us",
        "revision": RELEASE_ID,
        "sha256": _sha256(release_dir / "reform_validation.json"),
    }
    manifest_path.write_text(json.dumps(manifest))

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )

    uploaded_paths = [path for path, _ in hub.uploads]
    release_path = f"releases/{RELEASE_ID}/reform_validation.json"
    assert "reform_validation.json" not in uploaded_paths
    assert release_path in uploaded_paths
    assert uploaded_paths.index(release_path) < uploaded_paths.index(
        LATEST_POINTER_PATH
    )


def test_publish_uploads_ssi_take_up_diagnostics_without_extra_files(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    diagnostics_path = release_dir / "us_ssi_take_up.json"
    diagnostics_path.write_text('{"schema_version": 1}')
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["us_ssi_take_up"] = {
        "kind": "diagnostics",
        "path": diagnostics_path.name,
        "repo_id": "policyengine/populace-us",
        "revision": RELEASE_ID,
        "sha256": _sha256(diagnostics_path),
    }
    manifest_path.write_text(json.dumps(manifest))

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )

    uploaded_paths = [path for path, _ in hub.uploads]
    release_path = f"releases/{RELEASE_ID}/{diagnostics_path.name}"
    assert diagnostics_path.name not in uploaded_paths
    assert release_path in uploaded_paths
    assert uploaded_paths.index(release_path) < uploaded_paths.index(
        LATEST_POINTER_PATH
    )


def test_publish_requires_artifact_root_for_root_artifacts(
    hub: FakeHub, release_dir: Path
) -> None:
    with pytest.raises(ValueError, match="pass artifact_root"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
        )
    assert hub.uploads == []


def test_missing_root_artifact_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024_calibration.npz").unlink()
    with pytest.raises(FileNotFoundError, match="populace_us_2024_calibration"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []


def test_root_artifact_hash_mismatch_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024.h5").write_bytes(b"wrong payload")
    with pytest.raises(ValueError, match="release artifact 'populace_us_2024.h5'"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []


def test_release_tag_is_created_before_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        create_tag=True,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-1"}]
    assert [event for event, _ in hub.events][-3:] == [
        "create_tag",
        "delete_branch",
        "create_commit",
    ]
    assert hub.uploads[-1][0] == LATEST_POINTER_PATH


def test_release_id_artifact_revision_requires_release_tag(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="must create the matching Hugging Face tag"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            create_tag=False,
        )
    assert hub.uploads == []
    assert hub.tags == []


def test_release_id_artifact_revision_rejects_tag_name_override(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="tag_name must match the release id"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            tag_name="different-tag",
        )
    assert hub.uploads == []
    assert hub.tags == []


def test_invalid_release_uploads_nothing(hub: FakeHub, release_dir: Path) -> None:
    (release_dir / "build_manifest.json").unlink()
    with pytest.raises(ReleaseContractError):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_invalid_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_nonstandard_nan_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        '{"schema_version": 4, "targets": [], "loss_trajectory": [NaN], '
        '"skipped": [], "options": {}}'
    )
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []


def test_extra_files_ride_along_before_the_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        extra_files=("calibration_diagnostics.json",),
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    extra = f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    assert extra in uploaded_paths
    assert uploaded_paths.index(extra) < uploaded_paths.index(LATEST_POINTER_PATH)


def test_missing_extra_file_fails_loudly(hub: FakeHub, release_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="support_audit"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            extra_files=("support_audit.json",),
        )
    assert hub.uploads == []


def test_publish_then_resolve_round_trips(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    published = publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    pointer = latest_release("policyengine/populace-us", api=hub)
    assert pointer.release_id == RELEASE_ID
    assert pointer.updated_at == "2026-06-11T13:53:15+00:00"
    assert pointer.paths == published["paths"]


def test_future_pointer_schema_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION + 1}).encode(),
    )
    with pytest.raises(ValueError, match="Upgrade populace-data"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_without_release_id_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION}).encode(),
    )
    with pytest.raises(ValueError, match="release_id"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_without_contract_paths_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps(
            {
                "schema_version": LATEST_POINTER_SCHEMA_VERSION,
                "release_id": RELEASE_ID,
                "paths": {"build_manifest": "releases/x/build_manifest.json"},
            }
        ).encode(),
    )
    with pytest.raises(ValueError, match="paths"):
        latest_release("policyengine/populace-us", api=hub)


def test_pointer_with_swapped_contract_path_is_refused(hub: FakeHub) -> None:
    payload = latest_pointer_payload(RELEASE_ID)
    payload["paths"]["build_manifest"] = (
        f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    )
    hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match="malformed=\\['build_manifest'\\]"):
        latest_release("policyengine/populace-us", api=hub)
