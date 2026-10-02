"""Shared fixtures for the US export household subsampler tests.

``tools/sample_us_export_households.py`` writes a stratified whole-household
subsample of a US export H5. These helpers build small nested US populations
(households with several tax units, persons in every entity), write them in
``USSingleYearDataset``'s layout without the engine (one ``format="table",
data_columns=True`` frame per entity plus ``_time_period``, as
``USSingleYearDataset.save`` writes) and read them back as a
:class:`~microcosm.frame.Frame`, so the engine-free tests exercise the chunked
H5 path end to end.

This module must never import ``microcosm.build.us_runtime`` (or anything that
does). Importing that package runs ``spine_agreement``'s module-level registry,
which, wherever policyengine-us is installed, attests the take-up ABI lock by
building the whole policyengine-us tax-benefit system: 173 CPU-seconds and
1.66 GiB measured on 2026-10-02. On a contended host that import ran for more
than 49 minutes and was mistaken for a hang in the sampler. The sampler reads
only a probe's ``id`` and ``binding_inputs``, so :class:`SamplerProbe` stands in
for ``ReformCoverageProbe`` here; the probe tool's tests, which need the release
tool anyway, use the real class.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import US_SCHEMA
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

US_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")

#: Modules whose import the engine-free sampler path must never trigger.
ENGINE_MODULE_PREFIXES = ("policyengine_us", "microcosm.build.us_runtime")


def _load_tool(module_name: str, filename: str):
    """Import ``tools/<filename>`` as ``module_name``.

    The module is registered in ``sys.modules`` before it executes:
    ``dataclasses`` resolves the tools' string annotations (``from __future__
    import annotations``) through ``sys.modules[cls.__module__]``.
    """
    path = _TEST_PATHS.repository / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture(scope="module")
def sampler():
    return _load_tool("sample_us_export_households", "sample_us_export_households.py")


@pytest.fixture(scope="module")
def probe_tool():
    """``tools/probe_us_post_export.py``; it loads the release tool lazily."""
    return _load_tool("probe_us_post_export", "probe_us_post_export.py")


def loaded_engine_modules() -> list[str]:
    """The engine modules this process has imported (see the module docstring)."""
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ENGINE_MODULE_PREFIXES
        )
    )


# ---------------------------------------------------------------------------
# A nested synthetic export
# ---------------------------------------------------------------------------

#: Channels in the Route A export's order: every ASEC household first.
CHANNELS = ("asec", "puf_tax_detail")
SOURCE_YEARS = (2022, 2023, 2024)
STATES = (6, 24, 36, 48, 12)


def synthetic_export_frame(
    n_households: int = 60,
    *,
    seed: int = 0,
    rare_households: tuple[int, ...] = (),
    id_offset: int = 0,
) -> Frame:
    """A nested US frame with channels, source years and probe inputs.

    Households ``1..n`` (plus ``id_offset``) hold one to four persons; about a
    third split into two tax units. The first half are ``asec`` and the rest
    ``puf_tax_detail`` (the Route A export's layout); ``source_year`` is a
    person column constant within each household. ``keogh_distributions`` is
    nonzero only on ``rare_households`` (positions), ``employment_income``
    on most adults, and ``takes_up_snap_if_eligible`` varies. Weights are
    distinct positive numbers.
    """
    rng = np.random.default_rng(seed)
    person_rows = []
    household_rows = []
    person_id = 0
    marital = 0
    for position in range(n_households):
        household_id = id_offset + position + 1
        size = int(rng.integers(1, 5))
        split = size >= 2 and rng.random() < 0.35
        channel = CHANNELS[0] if position < n_households // 2 else CHANNELS[1]
        year = SOURCE_YEARS[int(rng.integers(0, len(SOURCE_YEARS)))]
        household_rows.append(
            {
                "household_id": household_id,
                "household_support_channel": channel,
                "state_fips": STATES[int(rng.integers(0, len(STATES)))],
            }
        )
        for index in range(size):
            person_id += 1
            marital += 1
            tax_unit = household_id * 10 + (1 if split and index >= size // 2 else 0)
            adult = index < 2
            person_rows.append(
                {
                    "person_id": id_offset * 10 + person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit,
                    "person_spm_unit_id": household_id * 100,
                    "person_family_id": household_id * 1000,
                    "person_marital_unit_id": (id_offset * 10 + marital) * 10,
                    "age": float(
                        rng.integers(25, 80) if adult else rng.integers(0, 18)
                    ),
                    "source_year": year,
                    "employment_income_before_lsr": float(
                        rng.integers(0, 90_000) if adult and rng.random() < 0.8 else 0
                    ),
                    "keogh_distributions": float(
                        5_000.0 if position in rare_households and index == 0 else 0.0
                    ),
                }
            )
    person = pd.DataFrame(person_rows)
    used_tax_units = np.unique(person["person_tax_unit_id"].to_numpy())
    household = pd.DataFrame(household_rows)
    household_ids = household["household_id"].to_numpy()
    spm = pd.DataFrame(
        {
            "spm_unit_id": household_ids * 100,
            "takes_up_snap_if_eligible": rng.random(len(household_ids)) < 0.8,
        }
    )
    tables = {
        "person": person,
        "household": household,
        "tax_unit": pd.DataFrame({"tax_unit_id": used_tax_units}),
        "spm_unit": spm,
        "family": pd.DataFrame({"family_id": household_ids * 1000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    weights = 100.0 + rng.permutation(n_households).astype(np.float64) * 7.25
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def write_tables_h5(tables, household_weights, path: Path, period: int = 2024) -> None:
    """Write entity tables (any row order) as ``USSingleYearDataset.save`` does."""
    path = Path(path)
    path.unlink(missing_ok=True)
    with pd.HDFStore(str(path)) as store:
        for entity in US_ENTITIES:
            table = tables[entity].copy()
            if entity == "household":
                table["household_weight"] = np.asarray(household_weights, np.float64)
            if len(table):
                store.put(entity, table, format="table", data_columns=True)
        store.put("_time_period", pd.Series([int(period)]), format="table")


def write_table_h5(frame: Frame, path: Path, period: int = 2024) -> None:
    """Write ``frame`` as ``USSingleYearDataset.save`` does, without the engine."""
    write_tables_h5(
        {entity: frame.table(entity) for entity in US_ENTITIES},
        frame.weights_for("household").values,
        path,
        period,
    )


def load_table_h5(path: Path, *, expected_sha256: str | None = None) -> Frame:
    """``_load_frame``'s result, read with pandas instead of the engine."""
    with pd.HDFStore(str(path), mode="r") as store:
        tables = {
            entity: store[entity].reset_index(drop=True) for entity in US_ENTITIES
        }
    weights = tables["household"].pop("household_weight").to_numpy(np.float64)
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


@dataclass(frozen=True)
class SamplerProbe:
    """The two fields of a ``ReformCoverageProbe`` the sampler reads."""

    id: str
    binding_inputs: tuple[str, ...]


def fixture_sampler_probes() -> tuple[SamplerProbe, ...]:
    """A rare input (certainty at small p), a common one and an absent one."""
    return (
        SamplerProbe("rare_keogh", ("keogh_distributions",)),
        SamplerProbe("common_wages", ("employment_income_before_lsr",)),
        SamplerProbe("absent_input", ("not_a_stored_column",)),
    )


def sample_synthetic(
    sampler,
    tmp_path: Path,
    frame: Frame,
    *,
    fraction: float,
    seed: int = 0,
    probes=None,
    chunk_rows: int | None = 7,
    name: str = "export",
) -> tuple[Path, dict]:
    """Write ``frame`` as an export, sample it; return (subsample path, receipt).

    ``chunk_rows`` defaults to 7 so every table is read in many chunks (the
    tool's byte-sized default floors a chunk at 1,024 rows).

    The deny-list boundary (``refuse_denied=True``, the CLI default) lives in
    ``microcosm.build.us_runtime.h5_io``; this engine-free path skips it, and
    a separate test pins that the default calls it.
    """
    source = tmp_path / name / "populace_us_2024.h5"
    source.parent.mkdir(parents=True, exist_ok=True)
    write_table_h5(frame, source)
    receipt = sampler.sample_export(
        source,
        tmp_path / f"{name}-sample",
        fraction=fraction,
        seed=seed,
        probes=fixture_sampler_probes() if probes is None else probes,
        write_dataset=lambda sample, path, period: write_table_h5(sample, path, period),
        chunk_rows=chunk_rows,
        refuse_denied=False,
    )
    return Path(receipt["output"]["path"]), receipt


__all__ = [name for name in globals() if not name.startswith("__")]
