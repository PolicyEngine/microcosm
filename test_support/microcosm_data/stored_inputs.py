"""The stored-input contract (microcosm#1026, decision d271).

A US release is refused when a stored table carries a column that looks like a
policyengine-us variable, is not a variable of the certified engine, and is
not in the reviewed register. These tests pin that rule three ways:

- example tests, including the ``would_claim_wic`` regression against an
  engine that defines only ``takes_up_wic_if_eligible``;
- Hypothesis properties for the invariants the PR states (refusal if and only
  if, exact naming, monotonicity, register consistency, order independence,
  and an H5 metadata round trip). Hypothesis comes from the workspace sync;
  the wheels job installs no test extras, so those tests skip there;
- ``requires_us`` tests against the installed engine: the naming convention,
  the register, and the stored-column inventories of the two H5 files examined
  for #1026 (``fixtures/stored_input_inventories.json``).

The source-enrichment probe's use of the contract is pinned here too, with the
country and wrapper modules replaced by fakes so the order of its checks is
observable without either engine.
"""

from __future__ import annotations

# ruff: noqa: F401
import json
import random
import re
import string
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from microcosm.data import source_enrichment as enrichment
from microcosm.data import stored_inputs
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    CertifiedEngine,
    StoredInputRefusalError,
    StoredTableLayoutError,
    h5_stored_tables,
    h5_verdict,
    is_model_named,
    register_consistency_failures,
    register_sha256,
    require_h5_stored_inputs,
    stored_input_failures,
    undefined_stored_inputs,
)
from test_support.paths import paths_for

_INVENTORIES = (
    paths_for("microcosm-data").tests / "fixtures" / "stored_input_inventories.json"
)
_PUBLISHED_DEFAULT = "populace-us-2024-spm-20260915"
_REHEARSAL_EXPORT = "route-a-r4-rehearsal-export"
_RECEIPT_CHILD = "populace-us-2024-spm-receipts-20260923"
_STACKED_POOL = "buildq-stacked-pool-f010-s578"
_ACS_LOCAL_RELEASE = "populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z"
_QUOTED = re.compile(r"'([^']*)'")
#: The postal codes of the 50 states, DC, PR and VI: policyengine-us 2.2.1's
#: only variable names outside the lowercase convention.
_US_POSTAL_CODES = frozenset(
    "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO "
    "MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX UT VA VI VT WA "
    "WI WV WY".split()
)
_LOWER = string.ascii_lowercase
_MODEL_CHARS = frozenset(string.ascii_lowercase + string.digits + "_")


def _oracle_model_named(column: object) -> bool:
    """An independent spelling of the convention, for the property tests."""

    return (
        isinstance(column, str)
        and column[:1] in _LOWER
        and column[:1] != ""
        and all(character in _MODEL_CHARS for character in column)
    )


def _engine(*variables: str, label: str = "policyengine-us 9.9.9") -> CertifiedEngine:
    return CertifiedEngine(label=label, variables=frozenset(variables))


def _inventories() -> dict:
    return json.loads(_INVENTORIES.read_text())["files"]


def _write_pandas_layout(
    path: Path,
    tables: dict[str, list[str]],
    *,
    fixed: tuple[str, ...] = (),
    bytes_attrs: bool = False,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write the HDF layout pandas uses, with h5py and no pandas.

    A ``format="table"`` frame is a group whose ``table`` dataset has the
    ``index`` field followed by one field per data column; a ``format="fixed"``
    frame keeps its column labels in ``axis0``. ``metadata`` maps each
    top-level metadata key to its ``pandas_type``; by default it is the
    ``_time_period`` table series the country loader reads.
    """

    def attr(text: str):
        return np.bytes_(text) if bytes_attrs else text

    with h5py.File(path, "w") as h5:
        for key, columns in tables.items():
            group = h5.create_group(key)
            if key in fixed:
                group.attrs["pandas_type"] = attr("frame")
                group.create_dataset(
                    "axis0",
                    data=np.array([column.encode() for column in columns], dtype="S"),
                )
            else:
                group.attrs["pandas_type"] = attr("frame_table")
                dtype = np.dtype([("index", "<i8"), *((c, "<f8") for c in columns)])
                group.create_dataset("table", data=np.zeros(2, dtype=dtype))
        for key, pandas_type in (
            {"_time_period": "series_table"} if metadata is None else metadata
        ).items():
            h5.create_group(key).attrs["pandas_type"] = attr(pandas_type)


def _hypothesis():
    pytest.importorskip("hypothesis")
    import hypothesis
    from hypothesis import strategies as st

    return hypothesis, st


def _column_strategies(st):
    model_named = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)
    upper = st.from_regex(r"[A-Z][A-Z0-9_]{0,8}", fullmatch=True)
    mixed = st.from_regex(r"[a-z]{1,4}[A-Z][A-Za-z0-9_]{0,4}", fullmatch=True)
    odd = st.from_regex(r"[_0-9][a-z0-9_]{0,6}", fullmatch=True)
    anything = st.text(max_size=6)
    column = st.one_of(model_named, model_named, upper, mixed, odd, anything)
    return model_named, column


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


#: Register entries that postdate the files examined for #1026 and rest on a
#: live producer instead: the parent ids (microcosm#884), bound to
#: eligibility_inputs.US_ELIGIBILITY_INPUTS_PARENT_ID_COLUMNS by
#: test_us_stored_input_register.py (microcosm-data cannot import the build).
_PRODUCER_BOUND_ENTRIES = frozenset({"parent_1_id", "parent_2_id"})


#: The role columns policyengine-core's ``build_from_dataset`` reads to build
#: the group entities, for policyengine-us's five group entities. They are not
#: engine variables (pinned against the engine below).
_CORE_ROLE_COLUMNS = frozenset(
    {"role"}
    | {
        f"person_{group}_role"
        for group in ("household", "tax_unit", "spm_unit", "family", "marital_unit")
    }
)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


_LAYOUT_DEFECTS = (
    "values_block",
    "index_not_first",
    "no_index",
    "stray_dataset",
    "untyped_group",
    "unknown_series",
    "metadata_frame",
    "fixed_without_axis",
)


# ---------------------------------------------------------------------------
# The source-enrichment probe runs the contract before either loader
# ---------------------------------------------------------------------------


class _FakeCountryLoader:
    """Stands in for ``USSingleYearDataset``; reaching it ends the probe."""

    def __init__(self, *args, **kwargs):
        raise _LoaderReachedError


class _FakeWrapperLoader:
    pass


class _FakeRoleVariable:
    entity = types.SimpleNamespace(key="person")
    value_type = bool


class _LoaderReachedError(Exception):
    pass


@pytest.fixture
def fake_probe_runtime(monkeypatch):
    """The probe's imports resolve to fakes, so its ordering is observable."""

    system = types.SimpleNamespace(
        variables={enrichment.ROLE_VARIABLE: _FakeRoleVariable()}
    )
    modules = {
        "policyengine": types.ModuleType("policyengine"),
        "policyengine.tax_benefit_models": types.ModuleType(
            "policyengine.tax_benefit_models"
        ),
        "policyengine.tax_benefit_models.us": types.ModuleType(
            "policyengine.tax_benefit_models.us"
        ),
        "policyengine.tax_benefit_models.us.datasets": types.ModuleType(
            "policyengine.tax_benefit_models.us.datasets"
        ),
        "policyengine_us": types.ModuleType("policyengine_us"),
        "policyengine_us.data": types.ModuleType("policyengine_us.data"),
        "policyengine_us.system": types.ModuleType("policyengine_us.system"),
    }
    modules[
        "policyengine.tax_benefit_models.us.datasets"
    ].PolicyEngineUSDataset = _FakeWrapperLoader
    modules["policyengine_us.data"].USSingleYearDataset = _FakeCountryLoader
    modules["policyengine_us.system"].system = system
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(enrichment, "_runtime_package_identities", lambda *a, **k: {})
    engine = _engine(
        "takes_up_wic_if_eligible",
        "person_id",
        enrichment.ROLE_VARIABLE,
        label="policyengine-us 2.2.1",
    )
    monkeypatch.setattr(stored_inputs, "installed_us_engine", lambda: engine)


# ---------------------------------------------------------------------------
# Against the installed engine
# ---------------------------------------------------------------------------


#: Each examined file's expected verdict: the refused columns, and how many
#: register entries it stores. The published default, its receipt child and
#: the ACS local-area release store two retired engine inputs: the #1026 WIC
#: draw as would_claim_wic, and the Medicare Part B target as
#: medicare_part_b_premiums (an input in every policyengine-us version read
#: from 1.452.0 to 1.670.2, replaced by medicare_part_b_premiums_reported by
#: 1.690.7). So each is refused naming exactly those two. The Route A
#: rehearsal export and the Build Q stacked multispine pool pass.
_EXPECTED_VERDICTS = {
    _PUBLISHED_DEFAULT: (["medicare_part_b_premiums", "would_claim_wic"], 24),
    _RECEIPT_CHILD: (["medicare_part_b_premiums", "would_claim_wic"], 24),
    _ACS_LOCAL_RELEASE: (["medicare_part_b_premiums", "would_claim_wic"], 37),
    _REHEARSAL_EXPORT: ([], 30),
    _STACKED_POOL: ([], 43),
}


__all__ = [name for name in globals() if not name.startswith("__")]
