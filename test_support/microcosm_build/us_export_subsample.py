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
import subprocess
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


def rich_export_frame(
    n_households: int = 50,
    *,
    seed: int = 0,
    rare_tax_units: tuple[int, ...] = (),
) -> Frame:
    """A synthetic export shaped like the real one where the sampler cares.

    Households of one to five persons hold one or two tax units, SPM units
    and families each, and couples share a marital unit (the real export has
    about 185k two-person marital units). Binding inputs live at every level:
    ``domestic_production_ald`` on the tax units at positions
    ``rare_tax_units`` (a rare group-level input), ``would_file_taxes_
    voluntarily`` as a uint8 tax-unit flag, ``spm_unit_energy_subsidy`` on
    some SPM units and ``household_vehicles_owned`` (int32) on households,
    plus person wages. Channels and source years as in
    :func:`synthetic_export_frame`.
    """
    rng = np.random.default_rng(seed)
    person_rows, household_rows = [], []
    tax_units, spm_units, families = [], [], []
    person_id = 0
    for position in range(n_households):
        household_id = position + 1
        size = int(rng.integers(1, 6))
        split = size >= 3 and rng.random() < 0.5
        year = SOURCE_YEARS[int(rng.integers(0, len(SOURCE_YEARS)))]
        household_rows.append(
            {
                "household_id": household_id,
                "household_support_channel": CHANNELS[
                    0 if position < n_households // 2 else 1
                ],
                "state_fips": STATES[int(rng.integers(0, len(STATES)))],
                "household_vehicles_owned": np.int32(rng.integers(0, 3)),
            }
        )
        couple = size >= 2 and rng.random() < 0.6
        for index in range(size):
            person_id += 1
            second = 1 if split and index >= size // 2 else 0
            marital = (
                household_id * 100
                if couple and index < 2
                else household_id * 100 + 10 + index
            )
            person_rows.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": household_id * 10 + second,
                    "person_spm_unit_id": household_id * 100 + second,
                    "person_family_id": household_id * 1000 + second,
                    "person_marital_unit_id": marital,
                    "age": float(
                        rng.integers(25, 80) if index < 2 else rng.integers(0, 18)
                    ),
                    "source_year": year,
                    "employment_income_before_lsr": float(
                        rng.integers(0, 90_000) if index < 2 else 0
                    ),
                }
            )
        for second in range(2 if split else 1):
            tax_units.append(
                {
                    "tax_unit_id": household_id * 10 + second,
                    "domestic_production_ald": 0.0,
                    "would_file_taxes_voluntarily": np.uint8(rng.random() < 0.3),
                }
            )
            spm_units.append(
                {
                    "spm_unit_id": household_id * 100 + second,
                    "spm_unit_energy_subsidy": float(
                        rng.integers(100, 900) if rng.random() < 0.2 else 0
                    ),
                    "takes_up_snap_if_eligible": bool(rng.random() < 0.8),
                }
            )
            families.append({"family_id": household_id * 1000 + second})
    tax_unit = pd.DataFrame(tax_units)
    for position in rare_tax_units:
        tax_unit.loc[position, "domestic_production_ald"] = 2_500.0
    tax_unit["would_file_taxes_voluntarily"] = tax_unit[
        "would_file_taxes_voluntarily"
    ].astype(np.uint8)
    person = pd.DataFrame(person_rows)
    household = pd.DataFrame(household_rows)
    household["household_vehicles_owned"] = household[
        "household_vehicles_owned"
    ].astype(np.int32)
    tables = {
        "person": person,
        "household": household,
        "tax_unit": tax_unit,
        "spm_unit": pd.DataFrame(spm_units),
        "family": pd.DataFrame(families),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.unique(person["person_marital_unit_id"].to_numpy())}
        ),
    }
    weights = 50.0 + rng.lognormal(mean=4.0, sigma=1.0, size=n_households)
    return Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )


def rich_sampler_probes() -> tuple[SamplerProbe, ...]:
    """Probes over the rich frame's inputs at every entity level."""
    return (
        SamplerProbe("rare_tax_unit", ("domestic_production_ald",)),
        SamplerProbe("flag_tax_unit", ("would_file_taxes_voluntarily",)),
        SamplerProbe("spm_subsidy", ("spm_unit_energy_subsidy",)),
        SamplerProbe("vehicles", ("household_vehicles_owned", "not_stored")),
        SamplerProbe("wages", ("employment_income_before_lsr",)),
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
    size_certainty_multiplier: float = 0.0,
    certainty_threshold: float = 5.0,
) -> tuple[Path, dict]:
    """Write ``frame`` as an export, sample it; return (subsample path, receipt).

    ``chunk_rows`` defaults to 7 so every table is read in many chunks (the
    tool's byte-sized default floors a chunk at 1,024 rows). The size
    certainty rule is off and the thin-probe threshold is 5 unless asked
    otherwise, so the certainty sets the tests pin stay exact whatever the
    tool's defaults become.

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
        size_certainty_multiplier=size_certainty_multiplier,
        certainty_threshold=certainty_threshold,
    )
    return Path(receipt["output"]["path"]), receipt


#: The commit git reports once :func:`move_head_after_load` has run.
LATER_HEAD = "f" * 40

#: The fields a tool records about the repository state it loaded from.
LOAD_STATE_FIELDS = ("commit", "dirty", "changes_sha256", "sha256")


def move_head_after_load(monkeypatch) -> None:
    """Make every later ``git`` call see a clean tree at :data:`LATER_HEAD`,
    as when the worktree moves on while a long run is still going."""
    real_run = subprocess.run

    def run(args, *rest, **options):
        args = list(args)
        if args[:1] == ["git"]:
            out = LATER_HEAD + "\n" if "rev-parse" in args else ""
            if not options.get("text"):
                out = out.encode()
            return subprocess.CompletedProcess(args, 0, out, out[:0])
        return real_run(args, *rest, **options)

    monkeypatch.setattr(subprocess, "run", run)


def pin_git_state_at_load(monkeypatch, tool) -> None:
    """Make ``tool._git_state`` report the state the tool loaded from."""
    loaded = tool._TOOL_SOURCE
    state = (loaded["commit"], loaded["dirty"], loaded["changes_sha256"])
    monkeypatch.setattr(tool, "_git_state", lambda *args, **kwargs: state)


__all__ = [name for name in globals() if not name.startswith("__")]
