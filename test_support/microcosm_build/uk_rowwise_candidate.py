"""Synthetic end-to-end contract for the first UK rowwise candidate."""

# ruff: noqa: F401

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.logbook import LOGBOOK_ROW_FIELDS, load_spool_rows
from microcosm.build.uk_runtime import (
    assemble_uk_oa_ladder,
    ladder_target_provenance,
    load_uk_oa_ladder,
    read_uk_single_year_weight_metadata,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from microcosm.frame import MassChangeRecord, WeightKind
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(autouse=True)
def _empty_support_exclusions_for_synthetic_rosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic candidates never carry the committed micro-LA exclusions.

    The committed ``local_area_support_exclusions.json`` names real local
    authorities measured on the licensed spine; a synthetic roster either
    lacks them (unknown) or meets the floor (stale), and the gate rightly
    fails either way. These tests exercise the machinery, so the support
    register is pinned empty here; the committed entries are covered by
    ``test_uk_battery_bindings.py``.
    """

    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    real_loader = battery_bindings.load_uk_local_area_support_exclusion_register

    def _loader(path, *, resource, **kwargs):
        if resource == "local_area_support_exclusions.json":
            return {"exclusions": {}, "bound_despite_support_floor": {}}
        return real_loader(path, resource=resource, **kwargs)

    monkeypatch.setattr(
        battery_bindings, "load_uk_local_area_support_exclusion_register", _loader
    )


@pytest.fixture(autouse=True)
def _spool_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPULACE_LEDGER_URL", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LEDGER_API_KEY", raising=False)
    monkeypatch.delenv("POPULACE_LOGBOOK_PREV_ROW_DIGEST", raising=False)
    monkeypatch.setenv(
        "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY",
        base64.b64encode(b"\x07" * 32).decode("ascii"),
    )


def _spool_rows(output_dir: Path):
    rows = load_spool_rows(output_dir / "logbook-spool")
    for row in rows:
        assert frozenset(row.to_mapping()) == LOGBOOK_ROW_FIELDS
    return rows


def _local_ref(path: Path) -> str:
    return f"local://{path.resolve().as_posix().lstrip('/')}"


def _fixture_hierarchy(
    name: str,
    *,
    provider_id: str,
    provider_label: str,
    category_id: str,
    category_label: str,
    geography_id: str,
    geography_label: str,
    geography_level: str,
    target_label: str,
) -> CalibrationHierarchy:
    return CalibrationHierarchy(
        provider=HierarchyNode(provider_id, provider_label),
        category=HierarchyCategory(category_id, category_label, provider_id),
        geography=HierarchyGeography(
            geography_id,
            geography_label,
            geography_level,
        ),
        dimensions=(),
        target=HierarchyNode(name, target_label),
    )


def _load_builder_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_uk_rowwise_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_rowwise_candidate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ladder_metadata() -> dict[str, object]:
    def layer(vintage: str) -> dict[str, object]:
        return {"vintage": vintage, "source": "synthetic test source"}

    return {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "synthetic",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("synthetic"),
            "msoa": layer("synthetic"),
            # Real April 2023 London codes, so the engine input resolves through
            # the names resource (the ladder refuses any other vintage).
            "local_authority": layer("2023_april_lad"),
            "ward": layer("synthetic"),
            "itl": layer("2021_itl"),
            "region": layer("synthetic"),
        },
    }


def _ladder_frame(
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
) -> pd.DataFrame:
    rows = [
        (
            "E00000001",
            "E12000007",
            "E14000001",
            "E05014284",
            "E09000001",
            "TLI31",
        ),
        (
            "W00000001",
            "W99999999",
            "W07000041",
            "W05001517",
            "W06000001",
            "TLL11",
        ),
        (
            "S00000001",
            "S99999999",
            "S14000001",
            "S13002835",
            "S12000033",
            "TLM50",
        ),
        (
            "N20000001",
            "N99999999",
            "N05000001",
            "N10000104",
            "N09000001",
            "TLN0A",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "oa_code": oa,
                "population": 100.0,
                "households": households,
                "constituency_code": constituency,
                "region_code": region,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": local_authority,
                "ward_code": ward,
                "itl3_code": itl3,
            }
            for (
                oa,
                region,
                constituency,
                ward,
                local_authority,
                itl3,
            ), households in zip(rows, household_counts, strict=True)
        ]
    )


def _write_ladder(
    path: Path,
    *,
    household_counts: tuple[float, float, float, float] = (
        3.0,
        10.0,
        10.0,
        10.0,
    ),
):
    payload = assemble_uk_oa_ladder(
        _ladder_frame(household_counts),
        _ladder_metadata(),
    )
    np.savez_compressed(path, **payload)
    return load_uk_oa_ladder(path)


def _write_staging_h5(
    path: Path,
    *,
    households_per_region: int = 3,
    region_masses: tuple[float, float, float, float] = (3.0, 10.0, 10.0, 10.0),
) -> None:
    if households_per_region < 3:
        raise ValueError("spine fixture needs one raw row and two derivatives")
    region_names = (
        "LONDON",
        "WALES",
        "SCOTLAND",
        "NORTHERN_IRELAND",
    )
    household_ids = list(range(1, 4 * households_per_region + 1))
    source_household_ids: list[int] = []
    support_clone_indices: list[int] = []
    spi_flags: list[bool] = []
    for region_index in range(4):
        first = region_index * households_per_region + 1
        raw_count = households_per_region - 2
        source_household_ids.extend(range(first, first + raw_count))
        source_household_ids.extend([first, first])
        support_clone_indices.extend([0] * raw_count + [0, 1])
        spi_flags.extend([False] * raw_count + [True, False])
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_weight": [
                mass / households_per_region
                for mass in region_masses
                for _ in range(households_per_region)
            ],
            "region": [
                region for region in region_names for _ in range(households_per_region)
            ],
            "source_household_id": source_household_ids,
            "source_household_key": [
                f"2023:{source_id}" for source_id in source_household_ids
            ],
            "household_source_id": source_household_ids,
            "household_support_clone_index": support_clone_indices,
            "household_is_spi_synthetic": spi_flags,
            "household_is_capital_gains_clone": [False] * len(household_ids),
            "household_is_cgt_band_donor": [False] * len(household_ids),
        }
    )
    person_ids = [10_000 + household_id for household_id in household_ids]
    benunit_ids = [20_000 + household_id for household_id in household_ids]
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": household_ids,
            "person_benunit_id": benunit_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": benunit_ids})
    dataset = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=33.0,
                new_total=33.0,
                declared_factor=1.0,
                reason="Synthetic staging mass record.",
            ),
        ),
    )
    write_uk_national_frame(dataset, path)


def _household_specs_for_ladder(ladder) -> list[TargetSpec]:
    specs = []
    for level, codes in (
        ("constituency", ladder.constituency_code),
        ("local_authority", ladder.local_authority_code),
    ):
        grouped = (
            pd.DataFrame({"code": codes, "value": ladder.households})
            .groupby("code", sort=True)["value"]
            .sum()
        )
        for area_code, value in grouped.items():
            census_year = 2022 if str(area_code).startswith("S") else 2021
            specs.append(
                TargetSpec(
                    name=f"ons.census.households@{area_code}",
                    entity="household",
                    measure="households",
                    value=float(value),
                    period=2025,
                    source="synthetic Chronicle fixture",
                    family="census_households",
                    metadata={
                        "contract_target_id": "ons.census.households",
                        "geography_level": level,
                        "geography_id": str(area_code),
                        "uprating_from_period": census_year,
                        "uprating_to_period": 2025,
                    },
                    hierarchy=_fixture_hierarchy(
                        f"ons.census.households@{area_code}",
                        provider_id="ons",
                        provider_label="Office for National Statistics",
                        category_id="ons.household_composition",
                        category_label="Household composition",
                        geography_id=str(area_code),
                        geography_label=f"Area {area_code}",
                        geography_level=level,
                        target_label="Occupied households",
                    ),
                )
            )
    return specs


def _mandatory_input_flags(input_h5: Path, ladder_path: Path) -> list[str]:
    return [
        "--input-sha256",
        hashlib.sha256(input_h5.read_bytes()).hexdigest(),
        "--ladder-sha256",
        hashlib.sha256(ladder_path.read_bytes()).hexdigest(),
        "--ledger-facts",
        str(ladder_path.parent / "synthetic-ledger"),
        "--ledger-facts-sha256",
        "1" * 64,
        "--ledger-manifest-sha256",
        "2" * 64,
        # Tests never reach the Hub: telemetry and the staged bundle stay local.
        "--staging-local-only",
    ]


def _configure_households_only_inputs(
    builder,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_h5: Path,
    ladder_path: Path,
) -> list[str]:
    ladder = load_uk_oa_ladder(ladder_path)
    artifact = SimpleNamespace(
        facts=None,
        provenance=lambda: {
            "facts_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "artifact_id": "synthetic-households-only-fixture",
        },
    )
    joint_inputs = {
        "artifact": artifact,
        "calibration_year": 2025,
        "national_registry": TargetRegistry([], country="uk"),
        "band_edge_registry": TargetRegistry([], country="uk"),
        "local_registry": TargetRegistry(
            _household_specs_for_ladder(ladder), country="uk"
        ),
        "measure_exclusions": {},
        "reviewed_unbound_higher_targets": {},
    }
    monkeypatch.setattr(
        builder, "_load_joint_target_inputs", lambda _args: joint_inputs
    )
    return [
        *_mandatory_input_flags(input_h5, ladder_path),
        "--households-only",
    ]


def _failing_gate_evaluator(builder, name: str, message: str):
    def evaluator(*_args, **_kwargs):
        return builder.GateResult(
            name=name, passed=False, failures=(message,), details={"minimum": 0}
        )

    return evaluator


def _joint_f100_args(input_h5: Path, ladder_path: Path, output_dir: Path) -> list[str]:
    return [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "dense",
        "--ladder",
        str(ladder_path),
        "--out",
        str(output_dir),
        "--n-clones",
        "2",
        "--seed",
        "7",
        "--epochs",
        "2",
        "--skip-holdout",
        *_mandatory_input_flags(input_h5, ladder_path),
        "--households-only",
    ]


def _load_tool(name: str):
    root = _TEST_PATHS.repository
    spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeHub:
    """One fake Hub serving the telemetry repo and the private dataset repo."""

    def __init__(
        self,
        *,
        fail_commit: bool = False,
        role: str = "write",
        scopes: list[dict] | None = None,
    ) -> None:
        self.role = role
        self.scopes = scopes
        self.commit_of: dict[tuple[str, str], str] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str]] = []
        self.commits: list[dict[str, object]] = []
        self.fail_commit = fail_commit
        self.sha = "a" * 40

    def paths(self, repo_id: str) -> list[str]:
        return sorted(path for repo, path in self.files if repo == repo_id)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
        assert repo_type == "dataset"
        self.files[(repo_id, path_in_repo)] = Path(path_or_fileobj).read_bytes()
        self.uploads.append((repo_id, path_in_repo))

    def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
        assert repo_type == "dataset"
        if (repo_id, filename) not in self.files:
            raise FileNotFoundError(filename)
        destination = (
            Path(kwargs.get("local_dir") or _FAKE_HUB_CACHE) / repo_id / filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[(repo_id, filename)])
        return str(destination)

    def file_exists(self, *, repo_id, filename, repo_type):
        assert repo_type == "dataset"
        return (repo_id, filename) in self.files

    def repo_info(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.sha)

    def whoami(self):
        token = {"role": self.role}
        if self.role == "fineGrained":
            token["fineGrained"] = {"global": [], "scoped": self.scopes or []}
        return {"name": "tester", "auth": {"accessToken": token}}

    def get_paths_info(self, *, repo_id, paths, expand, repo_type):
        assert expand and repo_type == "dataset"
        return [
            SimpleNamespace(
                path=path,
                last_commit=SimpleNamespace(oid=self.commit_of[(repo_id, path)]),
            )
            for path in paths
            if (repo_id, path) in self.commit_of
        ]

    def create_commit(
        self, *, repo_id, operations, commit_message, repo_type, parent_commit
    ):
        assert repo_type == "dataset"
        if self.fail_commit:
            raise RuntimeError("403 Forbidden token=do-not-record")
        assert parent_commit == self.sha
        for operation in operations:
            self.files[(repo_id, operation.path_in_repo)] = Path(
                operation.path_or_fileobj
            ).read_bytes()
        self.commits.append(
            {
                "repo_id": repo_id,
                "message": commit_message,
                "paths": sorted(op.path_in_repo for op in operations),
            }
        )
        self.sha = hashlib.sha256(commit_message.encode()).hexdigest()[:40]
        for operation in operations:
            self.commit_of[(repo_id, operation.path_in_repo)] = self.sha
        return SimpleNamespace(oid=self.sha)


_FAKE_HUB_CACHE = "/tmp/microcosm-fake-hub-cache"


def _staging_run_setup(builder, monkeypatch, tmp_path, *, remote: bool = False):
    """Fixture inputs plus the flag list; ``remote`` drops the local-only switch."""

    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    input_h5 = tmp_path / "staging.h5"
    ladder_path = tmp_path / "ladder.npz"
    _write_staging_h5(input_h5, households_per_region=52)
    ladder = _write_ladder(ladder_path)
    flags = _configure_households_only_inputs(
        builder, monkeypatch, input_h5=input_h5, ladder_path=ladder_path
    )
    import microcosm.build.uk_runtime.battery_bindings as battery_bindings

    monkeypatch.setattr(
        battery_bindings,
        "_local_area_roster",
        lambda _resource, levels: {
            "constituency": tuple(sorted(set(ladder.constituency_code))),
            "local_authority": tuple(sorted(set(ladder.local_authority_code))),
        },
    )
    if remote:
        flags = [flag for flag in flags if flag != "--staging-local-only"]
    return input_h5, ladder_path, flags


def _build_args(input_h5, ladder_path, flags, out, *extra):
    return [
        "--input-h5",
        str(input_h5),
        "--release-role",
        "dense",
        "--ladder",
        str(ladder_path),
        *flags,
        "--out",
        str(out),
        "--n-clones",
        "2",
        "--seed",
        "7",
        "--epochs",
        "2",
        "--skip-holdout",
        *extra,
    ]


def _single_run_id(out: Path) -> str:
    runs = sorted(path.name for path in (out / "staging" / "runs").iterdir())
    assert len(runs) == 1, runs
    return runs[0]


def _role_argv(tmp_path: Path, role: str, *extra: str) -> list[str]:
    return [
        "--input-h5",
        str(tmp_path / "spine.h5"),
        "--release-role",
        role,
        "--input-sha256",
        "2" * 64,
        "--ledger-facts",
        str(tmp_path / "ledger"),
        "--ledger-facts-sha256",
        "0" * 64,
        "--ledger-manifest-sha256",
        "1" * 64,
        "--out",
        str(tmp_path / "out"),
        *extra,
    ]


def _dense_argv(tmp_path: Path, *extra: str) -> list[str]:
    return _role_argv(
        tmp_path,
        "dense",
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--ladder-sha256",
        "3" * 64,
        *extra,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
