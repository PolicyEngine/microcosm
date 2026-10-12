"""The US release tool's stored-input gate and its register (microcosm#1026).

``microcosm.data.stored_inputs`` owns the rule and the reviewed register of
stored columns that are deliberately not policyengine-us variables. The data
shard depends on no other Microcosm shard, so it spells the register's names
itself; this file binds every entry to the producer definition it names. It
also pins the release tool's use of the rule: the batched pre-export gate over
the export frame's modeled stored tables, that model against the real writer
(``requires_us``), and the post-write check that the written H5 earns the
gate's verdict.
"""

from __future__ import annotations

# ruff: noqa: F401
import ast
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.outer_stage_runtime import _POOLED_SOURCE_PROVENANCE_COLUMNS
from microcosm.build.us_runtime import acs_pums, acs_transfer
from microcosm.build.us_runtime import puf_capital_gains_tail as puf_tail
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.eligibility_inputs import (
    US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS,
)
from microcosm.build.us_runtime.operator_boundary import _ACS_NATIVE_INPUT_CONTRACTS
from microcosm.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.data.stored_inputs import (
    _US_ENTITIES,
    US_STORED_NON_VARIABLE_COLUMNS,
    CertifiedEngine,
    h5_stored_tables,
)
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.units import (
    _TAX_UNIT_ROLE_COLUMN,
    TAX_UNIT_FILING_STATUS_COLUMN,
    US_SCHEMA,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_builder_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


def _engine(*variables: str) -> CertifiedEngine:
    return CertifiedEngine(
        label="policyengine-us 2.2.1", variables=frozenset(variables)
    )


_STRUCTURAL = (
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
    "household_id",
    "tax_unit_id",
    "spm_unit_id",
    "family_id",
    "marital_unit_id",
    "household_weight",
    "age",
    "takes_up_wic_if_eligible",
)


def _frame(**person_columns) -> Frame:
    """Two households, one person each, every US entity populated."""

    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids * 10,
            "person_spm_unit_id": ids * 100,
            "person_family_id": ids * 1_000,
            "person_marital_unit_id": ids * 10_000,
            "age": np.asarray([40, 70], dtype=np.int64),
            **person_columns,
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids * 10}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids * 100}),
            "family": pd.DataFrame({"family_id": ids * 1_000}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids * 10_000}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([1_500.0, 2_500.0]), WeightKind.CALIBRATED)},
    )


# ---------------------------------------------------------------------------
# The register is bound to the producers it names
# ---------------------------------------------------------------------------


#: The ACS-native amounts microcosm.build.us_runtime.acs_inputs maps under an
#: acs_ name: the acs_ keys of its contract. None is a variable of the
#: installed engine (test_the_inputs_the_acs_native_reasons_name_are_engine_inputs).
_ACS_NATIVE_AMOUNTS = frozenset(
    column for column in _ACS_NATIVE_INPUT_CONTRACTS if column.startswith("acs_")
)


# ---------------------------------------------------------------------------
# The release tool's gate
# ---------------------------------------------------------------------------


def _post_write_check(builder, monkeypatch, tmp_path, written, pre_export):
    """The post-write check on an H5 whose stored tables are ``written``."""

    monkeypatch.setattr(builder, "installed_us_engine", lambda: _engine(*_STRUCTURAL))
    monkeypatch.setattr(builder, "h5_stored_tables", lambda path: written)
    return builder._written_stored_input_verdict_mismatch(
        tmp_path / "populace_us_2024.h5", pre_export
    )


_STALE = {"person": ("person_id", "would_claim_wic")}
_CLEAN = {"person": ("person_id", "takes_up_wic_if_eligible")}


# ---------------------------------------------------------------------------
# Against the installed engine and the real writer
# ---------------------------------------------------------------------------


__all__ = [name for name in globals() if not name.startswith("__")]
