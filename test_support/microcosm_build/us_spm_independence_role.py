"""The SPM independence role as a build-stage input leaf.

The stage's derivation body is a call into the certified
``derive_spm_role_source``; these tests pin the stage's manifest and plan
wiring, every refusal the brief names (a missing source column, a unit with
more or fewer than one ``SPM_HEAD``, a unit left without a classified adult, a
person with no ASEC origin, a Census count that does not reconcile), the frame
integration and idempotence, the signal gate, the release coverage contract,
and — beside the hand-built battery — a seeded agreement sweep showing the
stage's role equals the derivation's and the raw rule's on random populations.
"""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime import (
    US_DONORS,
    US_PUF_SUPPORT_STAGE_NAME,
    US_RELATIONSHIP_INPUTS_STAGE_NAME,
    US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_STAGE_NAME,
    US_STAGE_NAMES,
    derive_us_spm_independence_role_from_manifest,
    load_release_input_coverage_manifest,
    resolve_asec_spm_role_source_paths,
    us_spm_independence_role_signal_gate,
    us_spm_independence_role_stage_spec,
    us_spm_independence_role_summary,
    with_us_spm_independence_role,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    POST_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_independence_role import (
    US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND,
    US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS,
    US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
    US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY,
    US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY,
)
from microcosm.build.us_runtime.spm_role_source import (
    _OPTIONAL_RAW_CHECKS,
    _REQUIRED_RAW_CHECKS,
    ASEC_SPM_ROLE_SOURCES,
    NATIVE_SPM_ROLE,
    AsecSpmRoleSource,
    derive_spm_role_source,
    independent_minor_role,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

pytest.importorskip("tables")

_ROLE = NATIVE_SPM_ROLE
_INCOME_YEAR = 2024
_SOURCE_COLUMNS = (
    "PERIDNUM",
    "SPM_ID",
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
    "SPM_HAGE",
    "SPM_HEAD",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "SPM_NUMPER",
    "A_FAMTYP",
    "A_FAMREL",
    "A_SPOUSE",
    "PECOHAB",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic Census person files and the frames built from them
# ---------------------------------------------------------------------------


def _finish_units(rows: list[dict]) -> pd.DataFrame:
    """Fill the Census unit counts from the rows' own ages and roles."""

    source = pd.DataFrame(rows)
    role = independent_minor_role(source)
    adult = source.A_AGE.ge(18) | (source.A_AGE.ge(15) & role)
    units = source.assign(_adult=adult).groupby("SPM_ID")
    source["SPM_NUMADULTS"] = units["_adult"].transform("sum").astype(int)
    source["SPM_NUMPER"] = units["_adult"].transform("size").astype(int)
    source["SPM_NUMKIDS"] = source["SPM_NUMPER"] - source["SPM_NUMADULTS"]
    head_age = source.loc[source.SPM_HEAD.eq(1)].set_index("SPM_ID")["A_AGE"]
    source["SPM_HAGE"] = source["SPM_ID"].map(head_age).astype(int)
    source["PERIDNUM"] = [f"{index + 1:022}" for index in range(len(source))]
    return source[list(_SOURCE_COLUMNS)]


def _hand_built_source() -> pd.DataFrame:
    """Three Census units: a minor spouse is independent, a same-age child is not.

    Unit 100: a 17-year-old head, a 16-year-old spouse, a 10-year-old child.
    Unit 200: a 40-year-old head and a 16-year-old child.
    Unit 300: a 15-year-old living alone (an SPM head).
    """

    rows = [
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=1, A_LINENO=1, A_AGE=17, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=2, PECOHAB=0),
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=2, A_LINENO=2, A_AGE=16, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=2, A_SPOUSE=1, PECOHAB=0),
        dict(SPM_ID=100, PH_SEQ=1, P_SEQ=3, A_LINENO=3, A_AGE=10, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=3, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=200, PH_SEQ=2, P_SEQ=1, A_LINENO=1, A_AGE=40, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=200, PH_SEQ=2, P_SEQ=2, A_LINENO=2, A_AGE=16, SPM_HEAD=0, A_FAMTYP=1, A_FAMREL=3, A_SPOUSE=0, PECOHAB=0),
        dict(SPM_ID=300, PH_SEQ=3, P_SEQ=1, A_LINENO=1, A_AGE=15, SPM_HEAD=1, A_FAMTYP=4, A_FAMREL=1, A_SPOUSE=0, PECOHAB=0),
    ]  # fmt: skip
    return _finish_units(rows)


def _random_source(seed: int, n_units: int) -> pd.DataFrame:
    """A random Census person file with a realistic minority of teen heads."""

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for unit in range(1, n_units + 1):
        teen_headed = rng.random() < 0.02
        head_age = (
            int(rng.integers(15, 18)) if teen_headed else int(rng.integers(25, 71))
        )
        members = [
            dict(A_AGE=head_age, SPM_HEAD=1, A_FAMTYP=1, A_FAMREL=1, A_SPOUSE=0),
        ]
        if not teen_headed and rng.random() < 0.6:
            members.append(
                dict(
                    A_AGE=int(np.clip(head_age + rng.integers(-5, 6), 18, 90)),
                    SPM_HEAD=0,
                    A_FAMTYP=1,
                    A_FAMREL=2,
                    A_SPOUSE=1,
                )
            )
            members[0]["A_SPOUSE"] = 2
        if not teen_headed:
            for _child in range(int(rng.integers(0, 4))):
                members.append(
                    dict(
                        A_AGE=int(rng.integers(0, 18)),
                        SPM_HEAD=0,
                        A_FAMTYP=1,
                        A_FAMREL=3,
                        A_SPOUSE=0,
                    )
                )
        for line, member in enumerate(members, start=1):
            rows.append(
                dict(
                    SPM_ID=unit,
                    PH_SEQ=unit,
                    P_SEQ=line,
                    A_LINENO=line,
                    PECOHAB=0,
                    **member,
                )
            )
    return _finish_units(rows)


def _write_source(
    tmp_path: Path, source: pd.DataFrame
) -> tuple[Path, AsecSpmRoleSource]:
    path = tmp_path / "pppub25.csv"
    source.to_csv(path, index=False)
    pin = AsecSpmRoleSource(
        income_year=_INCOME_YEAR,
        survey_year=_INCOME_YEAR + 1,
        csv_sha256=_digest(path),
        csv_size_bytes=path.stat().st_size,
        persons=int(len(source)),
        units=int(source.SPM_ID.nunique()),
        official_archive_url="https://example.invalid/fixture.zip",
        archive_sha256="a" * 64,
        member="pppub25.csv",
    )
    return path, pin


def _person_table(
    source: pd.DataFrame, *, clone_units: tuple[int, ...] = ()
) -> pd.DataFrame:
    """The pooled person table a base carries for ``source``, plus optional clones.

    Mirrors what ``asec_pool`` and cloning leave on the frame: frozen-vintage
    identity (``source_year``, ``source_household_id``, ``source_person_id``,
    ``source_row_id``), the raw age/count fields, no ``SPM_HEAD`` (no frozen
    vintage carries it) and ``A_FAMTYP``/``A_FAMREL`` null for some rows.
    """

    native = source.copy()
    native["source_row_id"] = np.arange(len(native), dtype=np.int64)
    native["clone"] = 0
    pieces = [native]
    for index, unit in enumerate(clone_units, start=1):
        clone = native.loc[native.SPM_ID.eq(unit)].copy()
        clone["clone"] = index
        pieces.append(clone)
    person = pd.concat(pieces, ignore_index=True)
    person["source_year"] = _INCOME_YEAR
    person["source_person_id"] = person["PERIDNUM"]
    person["source_household_id"] = person["PH_SEQ"]
    person["person_id"] = np.arange(1001, 1001 + len(person), dtype=np.int64)
    unit_codes = pd.factorize(
        pd.MultiIndex.from_arrays([person["clone"], person["SPM_ID"]]), sort=True
    )[0]
    person["person_spm_unit_id"] = (unit_codes + 10).astype(np.int64)
    household_codes = pd.factorize(
        pd.MultiIndex.from_arrays([person["clone"], person["PH_SEQ"]]), sort=True
    )[0]
    person["person_household_id"] = (household_codes + 1).astype(np.int64)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(len(person), dtype=np.int64) + 4_000
    person["age"] = person["A_AGE"].astype(float)
    person["SPM_ID"] = pd.factorize(person["SPM_ID"], sort=True)[0] + 1
    person = person.drop(columns=["SPM_HEAD", "clone"])
    person["A_FAMREL"] = person["A_FAMREL"].astype(float)
    person.loc[person.index[:2], "A_FAMREL"] = np.nan
    person["A_FAMTYP"] = person["A_FAMTYP"].astype(float)
    person.loc[person.index[:2], "A_FAMTYP"] = np.nan
    return person


def _frame(person: pd.DataFrame, weights: np.ndarray | None = None) -> Frame:
    households = np.sort(person["person_household_id"].unique())
    tables = {
        "person": person.reset_index(drop=True),
        "household": pd.DataFrame({"household_id": households}),
        "tax_unit": pd.DataFrame({"tax_unit_id": households + 1_000}),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.sort(person["person_spm_unit_id"].unique())}
        ),
        "family": pd.DataFrame({"family_id": households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    if weights is None:
        weights = np.full(len(households), 100.0)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _run(frame: Frame, path: Path, pin: AsecSpmRoleSource) -> Frame:
    return with_us_spm_independence_role(
        frame,
        seed=0,
        time_period=_INCOME_YEAR,
        asec_spm_role_source_paths={_INCOME_YEAR: path},
        source_pins={_INCOME_YEAR: pin},
    )


def _operation():
    return next(
        operation
        for operation in us_spm_independence_role_stage_spec().operations
        if operation.kind == US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND
    )


def _context(frame: Frame, path: Path, pin: AsecSpmRoleSource) -> SourceRuntimeContext:
    return SourceRuntimeContext(
        config=SourceRuntimeConfig(
            seed=0,
            target_year=_INCOME_YEAR,
            extra={
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY: {_INCOME_YEAR: path},
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY: {_INCOME_YEAR: pin},
            },
        ),
        tables={"spm_unit": frame.table("spm_unit")},
    )


@pytest.fixture
def population(tmp_path: Path):
    source = _hand_built_source()
    path, pin = _write_source(tmp_path, source)
    frame = _frame(_person_table(source, clone_units=(100,)))
    return source, path, pin, frame


# ---------------------------------------------------------------------------
# Manifest, plan and shared constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The derivation through the stage: agreement and refusals
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frame integration, summary and gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Seeded agreement sweep beside the hand-built battery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Release coverage and export contract
# ---------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
