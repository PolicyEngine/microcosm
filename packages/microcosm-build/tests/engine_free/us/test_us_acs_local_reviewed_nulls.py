"""The engine-pass reviewed-null fill never defaults a declared source input.

``policyengine_us.spm.DATASET_SOURCE_INPUTS`` "does not permit synthesizing a
default value when data are absent". ``fill_reviewed_nulls`` therefore refuses
a register that names ``is_spm_independent_minor_role`` and a null role it was
never told about, naming the staging rebuild as the remedy, while ordinary
registered inputs still fill with the engine default.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE
from test_support.microcosm_build.us_acs_local_reviewed_nulls import (
    RENT,
    FakeEngine,
    FakeSystem,
    load_tool_module,
    reviewed_null_frame,
    write_summary,
)

_SPINES = ["asec_puf", "acs_2024_1yr", "acs_2024_1yr"]


def test_a_register_naming_the_role_is_refused_and_the_role_is_untouched(
    tmp_path,
) -> None:
    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, None, None], rent=[1.0, np.nan, 2.0], spines=_SPINES
    )
    summary = write_summary(tmp_path / "summary.json", {RENT: 1, NATIVE_SPM_ROLE: 2})

    with pytest.raises(module.DeclaredSourceInputNullError) as exc:
        module.fill_reviewed_nulls(
            frame, summary, engine=FakeEngine(), system=FakeSystem()
        )

    assert f"person.{NATIVE_SPM_ROLE}" in str(exc.value)
    assert "Rebuild the ACS multispine staging" in str(exc.value)
    assert frame.person[NATIVE_SPM_ROLE].tolist() == [True, None, None]
    assert np.isnan(frame.person[RENT].iloc[1])


def test_an_unregistered_null_role_is_refused_as_a_declared_source_input(
    tmp_path,
) -> None:
    """The 2026-09-28 failure, now with its remedy named."""

    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, None, None], rent=[1.0, np.nan, 2.0], spines=_SPINES
    )
    summary = write_summary(tmp_path / "summary.json", {RENT: 1})
    manifest = tmp_path / "fills.json"

    with pytest.raises(module.DeclaredSourceInputNullError) as exc:
        module.fill_reviewed_nulls(
            frame, summary, manifest, engine=FakeEngine(), system=FakeSystem()
        )

    assert isinstance(exc.value, module.UnregisteredNullError)
    assert "{'acs_2024_1yr': 2}" in str(exc.value)
    assert "may not be defaulted" in str(exc.value)
    assert frame.person[NATIVE_SPM_ROLE].tolist() == [True, None, None]
    recorded = json.loads(manifest.read_text())
    assert [
        (item["column"], item["declared_dataset_source_input"])
        for item in recorded["unregistered_violations"]
    ] == [(NATIVE_SPM_ROLE, True)]
    assert [item["column"] for item in recorded["fills"]] == [RENT]


def test_a_complete_role_passes_and_ordinary_inputs_still_fill(tmp_path) -> None:
    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, False, True], rent=[1.0, np.nan, 2.0], spines=_SPINES
    )
    summary = write_summary(tmp_path / "summary.json", {RENT: 1})

    fills, drift = module.fill_reviewed_nulls(
        frame, summary, engine=FakeEngine(), system=FakeSystem()
    )

    assert [(fill["column"], fill["filled_rows"]) for fill in fills] == [(RENT, 1)]
    assert drift == []
    assert frame.person[RENT].tolist() == [1.0, 0.0, 2.0]
    assert frame.person[NATIVE_SPM_ROLE].tolist() == [True, False, True]


def test_an_ordinary_unregistered_null_keeps_the_plain_error(tmp_path) -> None:
    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, False, True], rent=[1.0, np.nan, 2.0], spines=_SPINES
    )
    summary = write_summary(tmp_path / "summary.json", {"age": 0})

    with pytest.raises(module.UnregisteredNullError) as exc:
        module.fill_reviewed_nulls(
            frame, summary, engine=FakeEngine(), system=FakeSystem()
        )

    assert not isinstance(exc.value, module.DeclaredSourceInputNullError)
    assert f"person.{RENT}" in str(exc.value)


def test_the_engine_declaration_extends_the_refused_set(tmp_path) -> None:
    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, False, True], rent=[1.0, np.nan, 2.0], spines=_SPINES
    )
    summary = write_summary(tmp_path / "summary.json", {RENT: 1})
    engine = FakeEngine(declares=frozenset({NATIVE_SPM_ROLE, RENT}))

    with pytest.raises(module.DeclaredSourceInputNullError, match=RENT):
        module.fill_reviewed_nulls(frame, summary, engine=engine, system=FakeSystem())
