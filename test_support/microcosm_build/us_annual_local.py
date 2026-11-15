"""Engine-free fixtures for local-area annual candidates (microcosm#922).

A synthetic local base frame with every US entity, household geography and
person attributes, written in the local release's fixed nullable layout; a
local build folder whose manifests pin it; an SSA population file; and
stand-ins for the builder's engine touchpoints (the model and runtime
provenance, the tax-benefit system, the uprating map, the multi-year export
and the native loader). Nothing here imports ``policyengine_us``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build import us_annual_static_aging as annual
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights, apply_scale
from microcosm.frame.materialize import engine_tables

LOCAL_RELEASE = "microcosm-us-2024-local"
LOCAL_BUILD = "populace-us-2024-buildo-acs-local-abc1234-20261113T000000Z"
LOCAL_DATASET = "microcosm_us_2024_acs_local"
LOCAL_FILE = f"{LOCAL_DATASET}.h5"
REPO_ID = "policyengine/populace-us"
#: The block_v1 location attributes a located local build carries.
ROOT_ATTRIBUTES = {
    "populace_location_rule": "block_v1",
    "populace_location_seed": "7",
    "populace_location_clones": "1",
    "populace_block_ladder_sha256": "a" * 64,
    "populace_block_ladder_vintages": json.dumps({"block": 2020}, sort_keys=True),
}
#: Household geography columns the synthetic base carries.
HOUSEHOLD_GEOGRAPHY = (
    "state_fips",
    "county_fips",
    "county_fips_2023",
    "congressional_district_geoid",
    "puma",
    "block_geoid",
    "tract_geoid",
    "place_fips",
    "cbsa_code",
)
#: Person attributes the synthetic base carries.
PERSON_ATTRIBUTES = (
    "age",
    "is_female",
    "is_household_head",
    "worked_last_year",
    "industry_code",
    "occupation_code",
)
#: The monetary columns the stand-in uprating map follows.
MONETARY = ("employment_income_before_lsr", "self_employment_income_before_lsr")


@dataclass(frozen=True)
class Household:
    """One synthetic household: its geography, weight and members."""

    state: int
    county: int
    weight: float
    #: (age, is_female, worked, industry, occupation, wages, self-employment)
    members: tuple[tuple[int, bool, bool, int, int, float, float], ...]


def default_households() -> list[Household]:
    return [
        Household(6, 1, 120.0, ((40, False, True, 7860, 2310, 52_000.0, 0.0),)),
        Household(
            6,
            3,
            80.5,
            (
                (37, True, True, 8680, 4720, 31_000.0, -1_500.0),
                (9, False, False, 0, 0, 0.0, 0.0),
            ),
        ),
        Household(36, 61, 200.0, ((82, True, False, 0, 0, 0.0, 0.0),)),
        Household(
            36,
            61,
            55.25,
            (
                (91, False, False, 0, 0, 0.0, 2_000.0),
                (85, True, False, 0, 0, 0.0, 0.0),
            ),
        ),
        Household(9, 110, 64.0, ((26, False, True, 6170, 9130, 18_250.0, 0.0),)),
    ]


def local_frame(households: Sequence[Household]) -> Frame:
    """A six-entity US frame: one tax, SPM, family and marital unit per household."""
    person_rows = []
    for index, household in enumerate(households):
        for position, member in enumerate(household.members):
            age, female, worked, industry, occupation, wages, self_employment = member
            person_rows.append(
                {
                    "person_id": len(person_rows) + 1,
                    "person_household_id": index + 1,
                    "person_tax_unit_id": index + 1,
                    "person_spm_unit_id": index + 1,
                    "person_family_id": index + 1,
                    "person_marital_unit_id": index + 1,
                    "age": int(age),
                    "is_female": bool(female),
                    "is_household_head": position == 0,
                    "worked_last_year": bool(worked),
                    "industry_code": int(industry),
                    "occupation_code": int(occupation),
                    "employment_income_before_lsr": float(wages),
                    "self_employment_income_before_lsr": float(self_employment),
                }
            )
    person = pd.DataFrame(person_rows)
    ids = np.arange(1, len(households) + 1, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": ids,
            "state_fips": [h.state for h in households],
            "county_fips": [f"{h.state:02d}{h.county:03d}" for h in households],
            "county_fips_2023": [
                f"{h.state:02d}{(h.county + 100) if h.state == 9 else h.county:03d}"
                for h in households
            ],
            "congressional_district_geoid": [h.state * 100 + 1 for h in households],
            "puma": [f"{h.state:02d}{h.county:05d}" for h in households],
            "block_geoid": [
                f"{h.state:02d}{h.county:03d}{i:06d}0001"
                for i, h in enumerate(households)
            ],
            "tract_geoid": [
                f"{h.state:02d}{h.county:03d}{i:06d}" for i, h in enumerate(households)
            ],
            "place_fips": [f"{h.state:02d}{i:05d}" for i, h in enumerate(households)],
            "cbsa_code": [f"{10000 + h.county:05d}" for h in households],
        }
    )
    tables = {"person": person, "household": household}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    weights = np.array([h.weight for h in households], dtype=np.float64)
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def write_local_base(
    path: Path,
    frame: Frame,
    *,
    root_attributes: Mapping[str, str] | None = ROOT_ATTRIBUTES,
    period: int = 2024,
    artifact_kind: str = annual.LOCAL_ARTIFACT_KIND,
) -> Path:
    """Write ``frame`` as the local release tool writes its calibrated H5."""
    from microcosm.build.us_runtime.h5_io import write_nullable_us_h5

    write_nullable_us_h5(
        frame,
        path,
        period=period,
        artifact_kind=artifact_kind,
        root_attributes=root_attributes,
    )
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_local_build_folder(
    root: Path,
    h5: Path,
    *,
    build: str = LOCAL_BUILD,
    release: str = LOCAL_RELEASE,
    dataset: str = LOCAL_DATASET,
    recorded_sha256: str | None = None,
    ledger_sha256: str | None = None,
    ledger: bool = True,
) -> Path:
    """A local build folder, ``root/releases/<release>/``, pinning ``h5``.

    ``h5`` stays where it is, as the package stage leaves it beside
    ``releases/``. ``recorded_sha256`` and ``ledger_sha256`` override the
    digests the release manifest and the ledger record for it.
    """
    folder = root / "releases" / release
    folder.mkdir(parents=True)
    digest = sha256(h5)
    release_manifest = {
        "schema_version": 1,
        "release": release,
        "build": build,
        "default_datasets": {},
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "namespace": "buildo_acs_local",
        "artifacts": {
            dataset: {
                "kind": "microdata",
                "path": h5.name,
                "repo_id": REPO_ID,
                "sha256": recorded_sha256 or digest,
                "role": "dataset",
            },
            "gate_summary": {
                "kind": "diagnostics",
                "path": "gate_summary.json",
                "repo_id": REPO_ID,
                "sha256": "0" * 64,
                "role": "gate_summary",
            },
        },
    }
    (folder / "release_manifest.json").write_text(json.dumps(release_manifest))
    (folder / "build_manifest.json").write_text(
        json.dumps({"release": release, "build": build})
    )
    if ledger:
        (folder / "sha256sums.txt").write_text(
            f"{'0' * 64}  gate_summary.json\n{ledger_sha256 or digest}  {h5.name}\n"
        )
    return folder


def write_ssa_csv(path: Path, years: Sequence[int] = (2024, 2025, 2026, 2027)) -> Path:
    """An SSA population table: every age 0-100, each cell growing 1% a year."""
    rows = []
    for year in years:
        growth = 1.01 ** (year - years[0])
        for age in range(101):
            rows.append(
                {
                    "Year": year,
                    "Age": age,
                    "M Tot": 1_000.0 * growth * (1 + age % 7),
                    "F Tot": 1_050.0 * growth * (1 + age % 5),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def uprating_map(
    columns: Sequence[str], years: Sequence[int], *, system: object
) -> tuple[dict, dict, dict]:
    """The stand-in uprating map: wages follow a total, self-employment an index."""
    del system
    totals = {"wages": {year: 100.0 * 1.04 ** (year - years[0]) for year in years}}
    indices = {"cpi": {year: 1.03 ** (year - years[0]) for year in years}}
    mapping = {}
    if "employment_income_before_lsr" in columns:
        mapping["employment_income_before_lsr"] = "wages"
    if "self_employment_income_before_lsr" in columns:
        mapping["self_employment_income_before_lsr"] = "cpi"
    return totals, indices, mapping


def multi_year_export(
    bundle: Frame, base_year: int, years: Mapping[int, tuple[np.ndarray, Mapping]]
) -> SimpleNamespace:
    """The adapter's per-year export without the engine's column checks.

    Like :func:`microcosm.frame.adapters.policyengine_us.multi_year_dataset`,
    each year is the bundle's tables with that year's household weights and
    its factors applied in float64; every other column is the bundle's own.
    """
    base = engine_tables(bundle, weighted_entities=("household",))
    datasets = {base_year: SimpleNamespace(**base)}
    for year, (weights, factors) in years.items():
        tables = {name: table.copy() for name, table in base.items()}
        tables["household"]["household_weight"] = np.asarray(weights, dtype=float)
        for column, factor in factors.items():
            owner = next(name for name, t in tables.items() if column in t.columns)
            tables[owner][column] = apply_scale(
                tables[owner][column].to_numpy(dtype=float), factor
            )
        datasets[year] = SimpleNamespace(**tables)
    return SimpleNamespace(datasets=datasets)


def scaling_static_aging(scale: Mapping[int, float] | float = 1.1):
    """A stand-in for ``static_aging``: weights times ``scale``, wages times 1.05."""

    def project(frame, *, base_year, years, column_series=None, **_):
        (year,) = years
        factor = scale[year] if isinstance(scale, Mapping) else scale
        weights = frame.weights_for("household").values * factor
        factors = {column: 1.05 for column in (column_series or {})}
        projection = SimpleNamespace(
            weights=Weights(weights, WeightKind.CALIBRATED),
            factors=factors,
            demographic_fit=pd.DataFrame({"base": [1.0], "achieved": [1.0]}),
            fraction_within_10pct=1.0,
        )
        return SimpleNamespace(year=lambda requested: projection)

    return project


def install_engine_stand_ins(monkeypatch, *, static_aging=None) -> list:
    """Replace every engine touchpoint of the builder; return the native checks.

    ``static_aging`` stays the real operator unless a stand-in is given. The
    returned list records each native-loader check as ``(path, year, rows)``.
    """
    # Load the real writer before replacing the engine. Defer this to the
    # executing worker so every xdist worker need not load US runtime at collection.
    import_module("microcosm.build.us_runtime.h5_io")
    fake = types.ModuleType("policyengine_us")
    fake.CountryTaxBenefitSystem = lambda: object()
    monkeypatch.setitem(sys.modules, "policyengine_us", fake)
    monkeypatch.setattr(
        annual,
        "_model_provenance",
        lambda **kwargs: {
            "version": "2.2.1",
            "commit": "c" * 40,
            "source": str(kwargs["source"]),
            "source_tree_sha256": "d" * 64,
            "source_files": {},
        },
    )
    monkeypatch.setattr(
        annual,
        "_runtime_provenance",
        lambda: {
            "source_sha256": {"annual_static_aging.py": "e" * 64},
            "versions": {"policyengine-us": "2.2.1", "policyengine-core": "3.0.0"},
        },
    )
    monkeypatch.setattr(annual, "uprating_series", uprating_map)
    monkeypatch.setattr(annual, "multi_year_dataset", multi_year_export)
    if static_aging is not None:
        monkeypatch.setattr(annual, "static_aging", static_aging)
    native_checks: list = []

    def native(path, year, rows):
        native_checks.append((Path(path), year, dict(rows)))
        return True

    monkeypatch.setattr(annual, "_verify_native_local", native)
    return native_checks


def local_build_kwargs(
    tmp_path: Path,
    households: Sequence[Household] | None = None,
    *,
    output: str = "candidate",
    root_attributes: Mapping[str, str] | None = ROOT_ATTRIBUTES,
    **folder_options,
) -> dict:
    """A local base build on disk and the builder arguments that project it."""
    root = tmp_path / "package"
    root.mkdir()
    h5 = write_local_base(
        root / LOCAL_FILE,
        local_frame(default_households() if households is None else households),
        root_attributes=root_attributes,
    )
    folder = write_local_build_folder(root, h5, **folder_options)
    ssa = write_ssa_csv(tmp_path / "ssa.csv")
    return {
        "base_build_dir": folder,
        "base_artifact_root": root,
        "ssa_csv": ssa,
        "ssa_sha256": sha256(ssa),
        "model_source": tmp_path,
        "model_commit": "c" * 40,
        "model_version": "2.2.1",
        "output_dir": tmp_path / output,
        "end_year": 2027,
        "epochs": 2,
    }


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    from microcosm.frame.materialize import read_frame_table

    with pd.HDFStore(path, "r") as store:
        return {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
