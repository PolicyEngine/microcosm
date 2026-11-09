"""The guards' declared-source-input set is the installed engine's declaration.

The staging null audit and the engine-pass fill refuse to default every name in
``us_declared_dataset_source_inputs``. These tests pin that set against the
installed policyengine-us, and run the fill once with the real adapter and
tax-benefit system, whose raw default for the role is ``False``.
"""

from __future__ import annotations

import numpy as np
import policyengine_us.spm as spm
import pytest

from microcosm.build.us_runtime.spm_independence_role import (
    US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS,
    us_declared_dataset_source_inputs,
)
from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from test_support.microcosm_build.us_acs_local_reviewed_nulls import (
    RENT,
    load_tool_module,
    reviewed_null_frame,
    write_summary,
)


def test_the_guard_set_is_the_engine_declaration() -> None:
    engine = PolicyEngineUSEngine()

    declared = us_declared_dataset_source_inputs(engine)

    assert declared == spm.DATASET_SOURCE_INPUTS | set(
        US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS
    )
    assert declared == frozenset({NATIVE_SPM_ROLE})
    assert engine.default_values(sorted(declared)) == {}


def test_the_real_engine_pass_refuses_a_null_role(tmp_path) -> None:
    module = load_tool_module()
    frame = reviewed_null_frame(
        role=[True, None],
        rent=[1.0, np.nan],
        spines=["asec_puf", "acs_2024_1yr"],
    )
    summary = write_summary(tmp_path / "summary.json", {RENT: 1})

    with pytest.raises(module.DeclaredSourceInputNullError, match=NATIVE_SPM_ROLE):
        module.fill_reviewed_nulls(frame, summary)

    assert frame.person[NATIVE_SPM_ROLE].tolist() == [True, None]
