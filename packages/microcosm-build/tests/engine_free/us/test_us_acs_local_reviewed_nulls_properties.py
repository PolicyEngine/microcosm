"""For every frame and register, the reviewed-null fill never defaults the role.

Each test draws a null pattern over ``is_spm_independent_minor_role`` and an
ordinary input, and a register over both, and checks the complete outcome
against an oracle:

- an empty register is refused outright;
- a register naming the role, or any null role, raises
  ``DeclaredSourceInputNullError`` and leaves every role value as it was;
- otherwise an unregistered null ordinary input raises the plain
  ``UnregisteredNullError``;
- otherwise the fill succeeds, fills only the ordinary input, and never
  touches the role.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE  # noqa: E402
from test_support.microcosm_build.us_acs_local_reviewed_nulls import (  # noqa: E402
    RENT,
    FakeEngine,
    FakeSystem,
    load_tool_module,
    reviewed_null_frame,
    write_summary,
)

_MODULE = load_tool_module()


@st.composite
def _cases(draw):
    n = draw(st.integers(min_value=1, max_value=6))
    role = draw(st.lists(st.sampled_from([True, False, None]), min_size=n, max_size=n))
    rent = draw(st.lists(st.sampled_from([0.0, 150.0, np.nan]), min_size=n, max_size=n))
    spines = draw(
        st.lists(st.sampled_from(["asec_puf", "acs_2024_1yr"]), min_size=n, max_size=n)
    )
    register = draw(
        st.dictionaries(
            st.sampled_from([RENT, NATIVE_SPM_ROLE, "age"]), st.integers(0, 6)
        )
    )
    declares = draw(st.sampled_from([None, frozenset({NATIVE_SPM_ROLE})]))
    return role, rent, spines, register, declares


@settings(max_examples=200, deadline=None)
@given(case=_cases())
def test_the_role_is_never_default_filled(case, tmp_path_factory) -> None:
    role, rent, spines, register, declares = case
    frame = reviewed_null_frame(role=role, rent=rent, spines=spines)
    summary = write_summary(tmp_path_factory.mktemp("summary") / "s.json", register)
    role_is_null = any(value is None for value in role)
    rent_is_null = any(np.isnan(value) for value in rent)

    def fill():
        return _MODULE.fill_reviewed_nulls(
            frame, summary, engine=FakeEngine(declares=declares), system=FakeSystem()
        )

    if not register:
        with pytest.raises(ValueError, match="carries no reviewed_engine_input_nulls"):
            fill()
    elif NATIVE_SPM_ROLE in register or role_is_null:
        with pytest.raises(_MODULE.DeclaredSourceInputNullError):
            fill()
    elif rent_is_null and RENT not in register:
        with pytest.raises(_MODULE.UnregisteredNullError) as exc:
            fill()
        assert not isinstance(exc.value, _MODULE.DeclaredSourceInputNullError)
    else:
        fills, _ = fill()
        assert {item["column"] for item in fills} <= {RENT}
        assert not frame.person[RENT].isna().any()
    assert frame.person[NATIVE_SPM_ROLE].tolist() == role
