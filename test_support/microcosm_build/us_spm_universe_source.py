"""The SPM measurement-universe producer: the rule, and every refusal.

Every fixture here is invented. Nothing in this file opens a build artifact,
a pinned source, or a release.
"""

# ruff: noqa: F401

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.spm_universe_source import (
    ACS_ARM,
    ACS_GROUP_QUARTERS_KINDS,
    ACS_HOUSEHOLD_KIND_COLUMN,
    ACS_HOUSING_UNIT_KIND,
    ASEC_ARM,
    ASEC_NATIVE_UNIT_COLUMN,
    ASEC_RECORD_TYPE_FIELDS,
    HOUSEHOLD_SPINE_COLUMN,
    HOUSEHOLD_SUPPORT_CHANNEL_COLUMN,
    INCLUDED,
    OUTSIDE,
    SPM_UNIVERSE_CHANNEL_ARMS,
    SPM_UNIVERSE_REFUSALS,
    UNIVERSE_INPUT,
    UNRESOLVED,
    attach_spm_universe_status,
    classify_spm_universe,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_US_RUNTIME = _TEST_PATHS.package / "src" / "microcosm" / "build" / "us_runtime"
_ACS_CHANNEL = "acs_2024_1yr"
_ASEC_CHANNEL = "asec_puf"


def _tables(
    persons: list[dict[str, object]],
    *,
    channel_column: str = HOUSEHOLD_SPINE_COLUMN,
    household_extra: dict[int, dict[str, object]] | None = None,
    person_extra_columns: dict[str, list[object]] | None = None,
    extra_household_rows: list[dict[str, object]] | None = None,
    extra_unit_ids: list[int] | None = None,
    without_support_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Household, person and SPM unit ids from a list of invented persons.

    Each person dict carries ``hid`` (household), ``unit`` (SPM unit),
    ``chan`` (provenance tag), and optionally ``kind`` (``TYPEHUGQ``),
    ``spm_id`` (the ASEC native unit id) and ``clone`` (clone index).
    """
    prows = []
    households: dict[int, dict[str, object]] = {}
    for index, spec in enumerate(persons, start=1):
        hid = int(spec["hid"])
        row: dict[str, object] = {
            "person_id": index,
            "person_household_id": hid,
            "person_spm_unit_id": int(spec["unit"]),
            "person_tax_unit_id": hid,
            "person_family_id": hid,
            "person_marital_unit_id": index,
            # ``"clone": None`` models a pool that lost the index on one arm;
            # a float passes through unchanged so invalid indices can be tested.
            "person_support_clone_index": (
                np.nan
                if spec.get("clone", 0) is None
                else spec.get("clone", 0)
                if isinstance(spec.get("clone", 0), float)
                else int(spec.get("clone", 0))
            ),
        }
        if "spm_id" in spec:
            row[ASEC_NATIVE_UNIT_COLUMN] = spec["spm_id"]
        prows.append(row)
        household = {"household_id": hid, channel_column: spec["chan"]}
        if "kind" in spec:
            household[ACS_HOUSEHOLD_KIND_COLUMN] = spec["kind"]
        households.setdefault(hid, household)
    person = pd.DataFrame(prows)
    if without_support_metadata:
        person = person.drop(columns=["person_support_clone_index"])
    if person_extra_columns:
        for name, values in person_extra_columns.items():
            person[name] = values
    household_rows = [households[hid] for hid in sorted(households)]
    if household_extra:
        for row in household_rows:
            row.update(household_extra.get(int(row["household_id"]), {}))
    if extra_household_rows:
        household_rows.extend(extra_household_rows)
    household = pd.DataFrame(household_rows)
    if ACS_HOUSEHOLD_KIND_COLUMN in household.columns:
        household[ACS_HOUSEHOLD_KIND_COLUMN] = household[
            ACS_HOUSEHOLD_KIND_COLUMN
        ].astype(object)
    unit_ids = np.asarray(
        sorted({int(spec["unit"]) for spec in persons} | set(extra_unit_ids or []))
    )
    return household, person, unit_ids


def _classify(persons: list[dict[str, object]], **kwargs) -> dict[int, str]:
    table_kwargs = {
        key: kwargs.pop(key)
        for key in (
            "channel_column",
            "household_extra",
            "person_extra_columns",
            "extra_household_rows",
            "extra_unit_ids",
        )
        if key in kwargs
    }
    household, person, unit_ids = _tables(persons, **table_kwargs)
    result = classify_spm_universe(
        household=household, person=person, unit_ids=unit_ids, **kwargs
    )
    return dict(zip(result.unit_ids.tolist(), result.status.tolist(), strict=True))


def _frame(persons: list[dict[str, object]], **kwargs) -> Frame:
    household, person, unit_ids = _tables(persons, **kwargs)
    household_ids = household["household_id"].to_numpy()
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": pd.DataFrame({"tax_unit_id": household_ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": unit_ids}),
            "family": pd.DataFrame({"family_id": household_ids}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.full(len(household_ids), 100.0), WeightKind.CALIBRATED
            )
        },
        metadata={"stage": "invented"},
    )


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


























# --------------------------------------------------------------------------
# Stored encoding
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def _refuses(code: str):
    return pytest.raises(ValueError, match=rf"^{code}: ")
































































# --------------------------------------------------------------------------
# The frame wrapper
# --------------------------------------------------------------------------








# --------------------------------------------------------------------------
# Constants parity and refusal-code completeness
# --------------------------------------------------------------------------


def _module_constant(module: str, name: str) -> str:
    """Read one module-level string constant without importing the module.

    ``acs_pums`` and ``stacked_spine`` both pull the country engine and cost
    tens of seconds to import; parsing is exact and free.
    """
    tree = ast.parse((_US_RUNTIME / module).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Constant), (module, name)
            return node.value.value
    raise AssertionError(f"{module} declares no module-level {name}")

__all__ = [name for name in globals() if not name.startswith("__")]
