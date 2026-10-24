"""County calibration pilot for the LWI County Files (microcosm#922).

Starts from the published ACS-local release (``policyengine/populace-us`` at
revision ``fb8ba90e``, the 2026-09-23 build ``buildo-acs-local-767312d60``: about
1.59M households, each already assigned a 2020 county inside its PUMA, with
weights calibrated to national, state and congressional-district targets). It
adds county target rows built from public county sources and recalibrates with
Microcosm's own calibrator (``microcosm.calibrate.calibrate``, the settings the
ACS-local release uses).

Two Modal steps, so the shared build host stays free:

``extract``
    Downloads the pinned release file, checks its sha256, and writes to the
    ``lwi-county-pilot`` volume: a lean frame (ids, geography, weights) and one
    row per household of the stored-input measures the county rows need.
    Connecticut households get their 2022 planning region from their 2020
    tract where they carry one, and otherwise from their PUMA and 2020 county:
    every CT (PUMA, county) cell lies in one region except Tolland County's part
    of PUMA 20301, which is drawn by 2020 population (``county_targets``).

``calibrate``
    Builds county rows (``county_targets.py``), calibrates, and writes weights,
    per-county fit and reliability tables, and a summary to the volume.

Run from the repo root::

    modal run experiments/us-county-pilot-20261024/pilot.py --step extract
    modal run experiments/us-county-pilot-20261024/pilot.py --step calibrate
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
#: The repo root locally; inside a container this file sits at /root/pilot.py.
REPO = HERE.parents[1] if modal.is_local() else HERE

HF_REPO = "policyengine/populace-us"
HF_REVISION = "fb8ba90e7f576ad0bb3f29b5aff2d1ec7007e912"
HF_FILENAME = "populace_us_2024_acs_local.h5"
HF_SHA256 = "769756c31f3ca646d12c272511744dec04c0e68870c6946dd945fbba65b6a7ec"
RELEASE_ID = "populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z"

VOLUME_NAME = "lwi-county-pilot"
VOLUME_ROOT = Path("/vol")
INPUTS = Path("/root/inputs")

#: Local copies of the public county sources, fetched 2026-10-24 (sha256 in
#: each folder's SHA256SUMS / the README).
ARTIFACTS = Path("/Users/maxghenis/PolicyEngine/_build_artifacts/lwi-county-20261024")
CHRONICLE = Path("/Users/maxghenis/PolicyEngine/chronicle")
#: Immutable Chronicle revision for the PEP and SOI county files.
CHRONICLE_REVISION = "200bf6d559cf85bc9c2c5ee4df8c50ba0f5d09d1"
CHRONICLE_MEMBERS = {
    "pep_county_2024": "db/data/census/pep_county_2024/co-est2024-alldata.csv",
    "soi_county_2022": "db/data/irs_soi/county_2022/22incyallnoagi.csv",
}
#: sha256 of every public input's bytes; a run refuses any other bytes.
INPUT_SHA256 = {
    "acs5_2024_B01003_county": "c83ef6e45601455855312f2e26ea755cd43a7c65ee58662e4cb4fad122a70bf2",
    "acs5_2024_B11001_county": "a0c2f64ad5f7172cded551364fc15d6bc37bbbc75ac135f7be00c8622d31d01a",
    "acs5_2024_B19001_county": "731d54b9443455987e088c4e65c848034c99ef19fb20aef60b87dc481450b3c2",
    "acs5_2024_B19013_county": "c574579d1f648472db366ea4726630c15f8d295ff9294e7dc9156b461d7e83a1",
    "acs5_2024_B19025_county": "8def9b6a5684a4b499d3ed0e23737358882ea4ad5627a06f7db5a63114bf7589",
    "acs5_2024_B19051_county": "919b6abbf7e46452761a04b1051a3e6bf085808b2d251a554d10f82940c4f5d3",
    "acs5_2024_B19061_county": "d3948e8118f7a5c5bbd82d2802591bb6edf184e012e227b478d71edd860f59f4",
    "pep_county_2024": "abcc8720d669e793bbfdcd440eeec37a78db3b452adbe4ccd1eadf7c72b522b9",
    "soi_county_2022": "2c04ff235bcbacb990b316bf79c7c628089a9a4c1dd9593ac51f74a109b13343",
    "crosswalk": "13ad002564e30e6dcd1df112fdaa985f2cd0cdb908f7f3017cd2723db294cff8",
    "block_mcd": "fe56bf38a30921e68360edc4e0ffabd2dae64ade6072761ed887c907bfadb653",
    "tract_puma": "5262f460b2c8dbe86b00549e916317b1791fdc595c9af72ab18f6f6a67040531",
    "tract_population": "2b31267f8391665b41d76ba083671efb5223347f70aee71482c0eb4faf946aa2",
}

PERSON_STRUCT = (
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
)
GROUP_IDS = {
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}


def _image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(
            "numpy==2.4.6",
            "pandas==3.0.3",
            "scipy==1.17.1",
            "tables==3.11.1",
            "pyarrow==25.0.0",
            "huggingface_hub==1.18.0",
            "pydantic>=2.10,<3",
        )
        .pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cpu")
        .env({"PYTHONPATH": "/root/src:/root/pilot", "PYTHONDONTWRITEBYTECODE": "1"})
    )
    for shard in ("frame", "calibrate", "graph", "diagnostics"):
        image = image.add_local_dir(
            REPO / "packages" / f"microcosm-{shard}" / "src" / "microcosm" / shard,
            f"/root/src/microcosm/{shard}",
            copy=True,
            ignore=["**/__pycache__/**"],
        )
    return image.add_local_file(
        HERE / "county_targets.py", "/root/pilot/county_targets.py"
    )


app = modal.App("lwi-county-pilot")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _verified(name: str, raw: bytes) -> bytes:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != INPUT_SHA256[name]:
        raise SystemExit(f"{name}: sha256 {digest} != pinned {INPUT_SHA256[name]}")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=_image(),
    volumes={VOLUME_ROOT: volume},
    cpu=4.0,
    memory=96 * 1024,
    timeout=3 * 3600,
)
def extract(ct_texts: dict[str, str]) -> dict:
    import county_targets
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from microcosm.frame import read_frame_table

    started = time.time()
    path = Path(
        hf_hub_download(
            HF_REPO,
            HF_FILENAME,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=VOLUME_ROOT / "release",
        )
    )
    volume.commit()
    sha = _sha256(path)
    if sha != HF_SHA256:
        raise SystemExit(f"{HF_FILENAME} sha256 {sha} != pinned {HF_SHA256}")
    out = VOLUME_ROOT / "extract"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"release_id": RELEASE_ID, "revision": HF_REVISION, "sha256": sha}
    with pd.HDFStore(path, mode="r") as store:
        report["keys"] = list(store.keys())
        household = read_frame_table(store, "household")
        person = read_frame_table(store, "person")
        groups = {
            name: read_frame_table(store, name)[[column]]
            for name, column in GROUP_IDS.items()
        }
    report["household_columns"] = list(household.columns)
    report["person_columns"] = list(person.columns)
    report["n_households"] = int(len(household))
    report["n_persons"] = int(len(person))

    weight_column = next(
        (c for c in ("household_weight", "weight") if c in household.columns), None
    )
    if weight_column is None:
        raise SystemExit(
            "no household weight column; household columns: "
            + ", ".join(household.columns)
        )
    weights = household[weight_column].to_numpy(np.float64)

    measures, measure_report = county_targets.household_measures(
        household, person, weights
    )
    report.update(measure_report)

    geo = pd.DataFrame({"household_id": household["household_id"].to_numpy()})
    for column in (
        "state_fips",
        "county_fips",
        "puma",
        "congressional_district_geoid",
        "tract_geoid",
        "block_geoid",
        "household_spine",
        "household_source_id",
    ):
        if column in household.columns:
            geo[column] = household[column].to_numpy()
    county = geo["county_fips"].astype(str).str.zfill(5)
    county_2023 = county.to_numpy().astype(object).copy()
    is_ct = county.str[:2].to_numpy() == "09"
    tract_region, cell_regions = county_targets.ct_planning_region_tables(
        ct_texts["crosswalk"],
        ct_texts["block_mcd"],
        ct_texts["tract_puma"],
        ct_texts["tract_population"],
    )
    ct_report: dict = {"households": int(is_ct.sum()), "by_tract": 0}
    ct_rows = np.flatnonzero(is_ct)
    by_tract = np.zeros(len(ct_rows), dtype=bool)
    if "tract_geoid" in geo.columns:
        tracts = geo["tract_geoid"].to_numpy()[ct_rows]
        for k, tract in enumerate(tracts):
            if tract is None or (isinstance(tract, float) and np.isnan(tract)):
                continue
            region = tract_region.get(str(tract).split(".")[0].zfill(11))
            if region is not None:
                county_2023[ct_rows[k]] = region
                by_tract[k] = True
        ct_report["by_tract"] = int(by_tract.sum())
    rest = ct_rows[~by_tract]
    if len(rest):
        puma = geo["puma"].astype(str).str.zfill(7).to_numpy()[rest]
        regions, cell_report = county_targets.assign_ct_regions(
            puma, county.to_numpy()[rest], cell_regions, seed=0
        )
        county_2023[rest] = regions
        ct_report["by_puma_county_cell"] = cell_report
    ct_report["households_by_region"] = (
        pd.Series(county_2023[ct_rows]).value_counts().sort_index().to_dict()
    )
    geo["county_fips_2023"] = county_2023.astype(str)
    report["ct"] = ct_report
    report["counties_2020"] = int(county.nunique())
    report["counties_2023"] = int(geo["county_fips_2023"].nunique())
    measures["weight"] = weights
    measures = measures.merge(geo, on="household_id", validate="one_to_one")
    measures.to_parquet(out / "households.parquet", index=False)

    lean_household = geo[["household_id"]].copy()
    lean_household["household_weight"] = weights
    with pd.HDFStore(out / "lean.h5", mode="w") as store:
        store.put("household", lean_household, format="fixed")
        store.put("person", person[list(PERSON_STRUCT)], format="fixed")
        for name, table in groups.items():
            store.put(name, table, format="fixed")
    report["weighted_households"] = float(weights.sum())
    report["weighted_persons"] = float((weights * measures["n_persons"]).sum())
    report["weighted_earnings"] = float((weights * measures["earnings"]).sum())
    report["seconds"] = round(time.time() - started, 1)
    (out / "extract_report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


#: Business-income inputs whose overlap with the earnings concept is audited
#: by spine before any of them joins a measure.
AUDIT_INCOME_COLUMNS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
    "farm_income",
    "farm_operations_income",
    "farm_rent_income",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
)


@app.function(
    image=_image(),
    volumes={VOLUME_ROOT: volume},
    cpu=4.0,
    memory=64 * 1024,
    timeout=3600,
)
def audit_income() -> dict:
    """Weighted totals, nonzero counts and co-occurrence by household spine."""
    import numpy as np
    import pandas as pd

    from microcosm.frame import read_frame_table

    path = VOLUME_ROOT / "release" / HF_FILENAME
    if _sha256(path) != HF_SHA256:
        raise SystemExit(f"{HF_FILENAME} does not match the pinned sha256")
    with pd.HDFStore(path, mode="r") as store:
        household = read_frame_table(store, "household")
        person = read_frame_table(store, "person")
    index = pd.Index(household["household_id"].to_numpy())
    position = index.get_indexer(person["person_household_id"].to_numpy())
    weight = household["household_weight"].to_numpy(np.float64)[position]
    spine = household["household_spine"].astype(str).to_numpy()[position]
    columns = [c for c in AUDIT_INCOME_COLUMNS if c in person.columns]
    values = {c: person[c].fillna(0.0).to_numpy(np.float64) for c in columns}
    report: dict = {"columns": columns, "by_spine": {}}
    for name in sorted(set(spine)):
        rows = spine == name
        w = weight[rows]
        entry = {"persons": int(rows.sum())}
        for column in columns:
            v = values[column][rows]
            entry[column] = {
                "nonzero": int((v != 0).sum()),
                "weighted_sum": float((w * v).sum()),
            }
        se = values["self_employment_income_before_lsr"][rows] != 0
        for column in ("sstb_self_employment_income_before_lsr", "farm_income"):
            if column in values:
                other = values[column][rows] != 0
                entry[f"{column}_with_self_employment"] = int((other & se).sum())
        report["by_spine"][name] = entry
    if "business_is_sstb" in person.columns:
        flag = person["business_is_sstb"].fillna(False).astype(bool).to_numpy()
        report["business_is_sstb_by_spine"] = {
            name: int(flag[spine == name].sum()) for name in sorted(set(spine))
        }
    return report


@app.function(
    image=_image(),
    volumes={VOLUME_ROOT: volume},
    cpu=8.0,
    memory=48 * 1024,
    timeout=8 * 3600,
)
def calibrate(sources: dict[str, bytes], settings: dict) -> dict:
    import county_targets

    return county_targets.run_pilot(
        VOLUME_ROOT / "extract",
        VOLUME_ROOT / settings["label"],
        sources,
        settings,
        commit=volume.commit,
    )


def _source_bytes() -> dict[str, bytes]:
    import subprocess

    acs = ARTIFACTS / "census_api_acs5_2024_county"
    sources = {
        path.stem: path.read_bytes() for path in sorted(acs.glob("acs5_2024_*.json"))
    }
    for name, member in CHRONICLE_MEMBERS.items():
        sources[name] = subprocess.run(
            ["git", "-C", str(CHRONICLE), "show", f"{CHRONICLE_REVISION}:{member}"],
            check=True,
            capture_output=True,
        ).stdout
    if sorted(sources) != sorted(k for k in INPUT_SHA256 if k not in _CT_FILES):
        raise SystemExit(f"unexpected source set {sorted(sources)}")
    return {name: _verified(name, raw) for name, raw in sources.items()}


#: The Connecticut geography files, relative to ARTIFACTS / "ct_sources", and
#: the text encoding each is decoded with.
_CT_FILES = {
    "crosswalk": ("ct_cou_to_cousub_crosswalk.txt", "utf-8-sig"),
    "block_mcd": ("BlockAssign_ST09_CT_MCD.txt", "utf-8"),
    "tract_puma": ("tract_puma/2020_Census_Tract_to_2020_PUMA.txt", "utf-8-sig"),
    "tract_population": ("tract_pop_2020/dec_pl_2020_P1_001N_tract_ct.json", "utf-8"),
}


def _ct_texts() -> dict[str, str]:
    base = ARTIFACTS / "ct_sources"
    return {
        name: _verified(name, (base / path).read_bytes()).decode(encoding)
        for name, (path, encoding) in _CT_FILES.items()
    }


@app.local_entrypoint()
def main(
    step: str = "extract",
    label: str = "run1",
    epochs: int = 800,
    max_weight_ratio: float = 5.0,
    target_loss_cap: float = 1.0,
) -> None:
    if step == "extract":
        report = extract.remote(_ct_texts())
    elif step == "audit":
        report = audit_income.remote()
    elif step == "calibrate":
        report = calibrate.remote(
            _source_bytes(),
            {
                "label": label,
                "epochs": epochs,
                "epoch_batch": 200,
                "learning_rate": 0.02,
                "max_weight_ratio": max_weight_ratio,
                "target_loss_cap": target_loss_cap,
                "seed": 0,
            },
        )
    else:
        raise SystemExit(f"unknown step {step!r}")
    local = HERE / "out"
    local.mkdir(exist_ok=True)
    (local / f"{step}_{label}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:6000])
