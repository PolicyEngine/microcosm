"""End-to-end byte-equivalence gate for the staged base-builder scaffold."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)

_INPUT_DIR_ENV = "POPULACE_US_PUF_EQUIVALENCE_INPUT_DIR"
_WEEKS_SOURCE_ENV = "POPULACE_US_PUF_EQUIVALENCE_WEEKS_SOURCE"
_MAX_HOUSEHOLDS_ENV = "POPULACE_US_PUF_EQUIVALENCE_MAX_HOUSEHOLDS"
_MAX_PUF_TAX_UNITS_ENV = "POPULACE_US_PUF_EQUIVALENCE_MAX_PUF_TAX_UNITS"
_DATASET_FILENAME = "base_populace_us_2024_puf_support.h5"


@pytest.mark.slow
def test_legacy_and_checkpointed_all_write_byte_identical_final_h5(
    tmp_path: Path,
) -> None:
    """Run the real tool in separate processes and compare final H5 bytes."""

    if importlib.util.find_spec("policyengine_us") is None:
        pytest.skip("requires the policyengine-us [us] extra")
    input_dir_value = os.environ.get(_INPUT_DIR_ENV)
    weeks_source_value = os.environ.get(_WEEKS_SOURCE_ENV)
    if input_dir_value is None or weeks_source_value is None:
        pytest.skip(
            f"set {_INPUT_DIR_ENV} and {_WEEKS_SOURCE_ENV} to run the slow gate"
        )

    input_dir = Path(input_dir_value)
    weeks_source = Path(weeks_source_value)
    inputs = {
        "asec_2024": input_dir / "census_cps_2024.h5",
        "asec_2023": input_dir / "census_cps_2023.h5",
        "asec_2022": input_dir / "census_cps_2022.h5",
        "puf": input_dir / "puf_2024.h5",
        "acs": input_dir / "acs_2022.h5",
        "weeks_source": weeks_source,
    }
    missing = [f"{name}={path}" for name, path in inputs.items() if not path.is_file()]
    assert not missing, "Missing equivalence input(s): " + ", ".join(missing)

    root = Path(__file__).resolve().parents[3]
    tool = root / "tools" / "build_us_puf_support_base.py"
    legacy_out = tmp_path / "legacy"
    checkpointed_out = tmp_path / "checkpointed"
    checkpoint_dir = tmp_path / "checkpoints"
    max_households = int(os.environ.get(_MAX_HOUSEHOLDS_ENV, "5000"))
    max_puf_tax_units = int(os.environ.get(_MAX_PUF_TAX_UNITS_ENV, "10000"))
    assert max_households > 0
    assert max_puf_tax_units > 0
    sampled_asec = _write_representative_asec_samples(
        {
            2024: inputs["asec_2024"],
            2023: inputs["asec_2023"],
            2022: inputs["asec_2022"],
        },
        destination=tmp_path / "asec_samples",
        max_households=max_households,
    )
    sampled_puf = _write_representative_puf_sample(
        inputs["puf"],
        destination=tmp_path / "puf_sample.h5",
        max_tax_units=max_puf_tax_units,
    )
    common = [
        sys.executable,
        str(tool),
        "--asec-h5",
        f"2024={sampled_asec[2024]}",
        "--asec-h5",
        f"2023={sampled_asec[2023]}",
        "--asec-h5",
        f"2022={sampled_asec[2022]}",
        "--asec-max-households",
        str(max_households),
        "--puf-h5",
        str(sampled_puf),
        "--acs-h5",
        str(inputs["acs"]),
        "--asec-2023-weeks-unemployed-source",
        str(inputs["weeks_source"]),
        "--target-year",
        "2024",
        "--seed",
        "0",
        "--n-estimators",
        "32",
        "--without-block-ladder",
        "--stage",
        "all",
    ]
    environment = {
        **os.environ,
        "POPULACE_FIT_PREDICT_WORKERS": "1",
        "PYTHONHASHSEED": "0",
    }

    _run_builder(
        [*common, "--out", str(legacy_out)],
        cwd=root,
        environment=environment,
    )
    _run_builder(
        [
            *common,
            "--out",
            str(checkpointed_out),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        cwd=root,
        environment=environment,
    )

    legacy_h5 = legacy_out / _DATASET_FILENAME
    checkpointed_h5 = checkpointed_out / _DATASET_FILENAME
    assert legacy_h5.read_bytes() == checkpointed_h5.read_bytes()
    assert (checkpoint_dir / "stage_all.frame.h5").is_file()
    profile = json.loads((checkpoint_dir / "stage_profile.json").read_text())
    record = profile["stages"]["all"]
    assert record["status"] == "succeeded"
    assert record["entry_rss_bytes"] > 0
    assert record["peak_rss_bytes"] >= record["entry_rss_bytes"]
    assert record["peak_rss_bytes"] >= record["exit_rss_bytes"]
    assert record["wall_seconds"] > 0


def _write_representative_asec_samples(
    sources: dict[int, Path],
    *,
    destination: Path,
    max_households: int,
) -> dict[int, Path]:
    """Write representative samples while preserving adjacent-year panels."""

    destination.mkdir(parents=True)
    sampled: dict[int, Path] = {}
    carry_person_ids: set[object] = set()
    for year in sorted(sources, reverse=True):
        source = sources[year]
        with pd.HDFStore(source, mode="r") as store:
            household = store["household"]
            person = store["person"]
        sample_size = min(max_households, len(household))
        rng = np.random.default_rng(year)
        required_households = set(
            person.loc[person["PERIDNUM"].isin(carry_person_ids), "PH_SEQ"].tolist()
        )
        child_support_signal = pd.to_numeric(person["CSP_VAL"], errors="coerce").gt(
            0
        ) | pd.to_numeric(person["CHSP_VAL"], errors="coerce").gt(0)
        signal_households = np.asarray(
            sorted(set(person.loc[child_support_signal, "PH_SEQ"].tolist()))
        )
        signal_target = min(
            len(signal_households),
            max(1, sample_size // 1000),
        )
        if len(signal_households) > signal_target:
            signal_households = rng.choice(
                signal_households, size=signal_target, replace=False
            )
        required_households.update(signal_households.tolist())
        required_positions = np.flatnonzero(
            household["H_SEQ"].isin(required_households).to_numpy()
        )
        if len(required_positions) > sample_size:
            required_positions = rng.choice(
                required_positions, size=sample_size, replace=False
            )
        remaining_positions = np.setdiff1d(
            np.arange(len(household)), required_positions, assume_unique=True
        )
        fill_size = sample_size - len(required_positions)
        fill_positions = rng.choice(remaining_positions, size=fill_size, replace=False)
        positions = np.sort(np.concatenate([required_positions, fill_positions]))
        household = household.iloc[positions].reset_index(drop=True)
        household_ids = set(household["H_SEQ"].tolist())
        person = person[person["PH_SEQ"].isin(household_ids)].reset_index(drop=True)
        carry_person_ids = set(person["PERIDNUM"].tolist())
        if "ED_VAL" not in person:
            # The currently pinned processed ASEC artifacts omit this raw
            # source column even though the production education stage
            # requires it.  Supply a bounded deterministic fixture surface so
            # this refactor gate can reach final export in both modes.
            education_assistance = np.zeros(len(person), dtype=np.float64)
            positive_rows = rng.choice(
                len(person), size=max(1, len(person) // 50), replace=False
            )
            education_assistance[positive_rows] = 1_000.0
            person["ED_VAL"] = education_assistance
        output = destination / f"census_cps_{year}.h5"
        household.to_hdf(output, key="household", mode="w", format="fixed")
        person.to_hdf(output, key="person", mode="a", format="fixed")
        sampled[year] = output
    return sampled


def _write_representative_puf_sample(
    source: Path,
    *,
    destination: Path,
    max_tax_units: int,
) -> Path:
    """Apply one deterministic entity-consistent sample to the PUF arrays."""

    rng = np.random.default_rng(2024)
    with h5py.File(source, mode="r") as input_h5:
        group_ids = np.asarray(input_h5["tax_unit_id"])
        sample_size = min(max_tax_units, len(group_ids))
        person_tax_unit_ids = np.asarray(input_h5["person_tax_unit_id"])
        required_group_ids: set[object] = set()
        signal_columns = (
            set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
            | set(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
        ) & set(input_h5)
        for column in sorted(signal_columns):
            values = np.asarray(input_h5[column])
            if len(values) == len(person_tax_unit_ids):
                entity_tax_unit_ids = person_tax_unit_ids
            elif len(values) == len(group_ids):
                entity_tax_unit_ids = group_ids
            else:
                continue
            for mask in (values > 0, values < 0):
                signal_ids = np.unique(entity_tax_unit_ids[mask])
                signal_target = min(len(signal_ids), 10)
                if len(signal_ids) > signal_target:
                    signal_ids = rng.choice(
                        signal_ids, size=signal_target, replace=False
                    )
                required_group_ids.update(signal_ids.tolist())
        required_positions = np.flatnonzero(
            np.fromiter(
                (value in required_group_ids for value in group_ids),
                dtype=bool,
                count=len(group_ids),
            )
        )
        remaining_positions = np.setdiff1d(
            np.arange(len(group_ids)), required_positions, assume_unique=True
        )
        fill_positions = rng.choice(
            remaining_positions,
            size=sample_size - len(required_positions),
            replace=False,
        )
        group_positions = np.sort(np.concatenate([required_positions, fill_positions]))
        selected_group_ids = set(group_ids[group_positions].tolist())
        person_mask = np.fromiter(
            (value in selected_group_ids for value in person_tax_unit_ids),
            dtype=bool,
            count=len(person_tax_unit_ids),
        )
        selected_marital_ids = set(
            np.asarray(input_h5["person_marital_unit_id"])[person_mask].tolist()
        )
        marital_ids = np.asarray(input_h5["marital_unit_id"])
        marital_mask = np.fromiter(
            (value in selected_marital_ids for value in marital_ids),
            dtype=bool,
            count=len(marital_ids),
        )
        lengths_to_selection = {
            len(group_ids): group_positions,
            len(person_tax_unit_ids): person_mask,
            len(marital_ids): marital_mask,
        }
        with h5py.File(destination, mode="w") as output_h5:
            for name, value in input_h5.attrs.items():
                output_h5.attrs[name] = value
            for name, dataset in input_h5.items():
                assert len(dataset.shape) == 1
                selection = lengths_to_selection.get(len(dataset))
                assert selection is not None, (
                    f"Unexpected PUF entity length for {name}: {len(dataset)}"
                )
                output_h5.create_dataset(
                    name,
                    data=np.asarray(dataset)[selection],
                    track_times=False,
                )
    return destination


def _run_builder(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Builder failed with exit {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
