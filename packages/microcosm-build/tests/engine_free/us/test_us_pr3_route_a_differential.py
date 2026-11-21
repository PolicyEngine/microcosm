"""Main's gate code and the live code agree on Route A frames (d2283 PR 3).

d2283 PR 3 edits four gates for the one-population frame: weeks unemployed,
Medicare take-up, retirement contributions and retirement distributions.
Route A's frames, the base builder's pre-clone frame and the stacked pool's
frames must keep main's verdicts and statistics. ``pr3_gate_differential``
(``test_support/microcosm_build/us_pr3_gate_differential.py``) grades a frame
with main's code, vendored per gate, and with the live code, in one process.
This file runs it on:

* the X5 reader population's Route A frame and its two clone-defect
  variants, with all four gates;
* the contributions fixture's pre-clone frame (the five outputs and no
  support metadata), with the contributions gate, as the base builder's
  pre-clone path grades it;
* each gate's stacked fixture, with that gate.

Each pair must give equal digests, neither side may raise, and main's
``details`` must not be empty. A pairing on which main's code raises is left
out, not compared as two equal raises.

The sanity cases check that the oracle is main's code. On X5's ASEC-only
assembled frame main's gates keep their scopes from before PR 3: no row in the
weeks UC constraint, and the native rows as the source rows of the other
three (and as the contributions reconciliation rows). PR 3 has widened the
live Medicare gate's assembled source mask to every CPS row, so its live
half counts every row there, and its payload differs from main's in
``source_rows`` alone. The other three live gates still run main's code at
this commit, so their digests match main's. The structural checks confirm
that each oracle holds its own copies, reads no live copy and imports only
from its live module and that module's helpers.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from microcosm.build.us_runtime import (
    medicare_take_up,
    retirement_contributions,
    retirement_distributions,
    weeks_unemployed,
)
from microcosm.build.us_runtime.retirement_contributions import (
    with_us_retirement_contribution_inputs,
)
from microcosm.build.us_runtime.support_provenance import support_clone_index_column
from microcosm.frame import Frame
from test_support.microcosm_build import (
    us_medicare_take_up,
    us_pr3_main_oracle_contributions,
    us_pr3_main_oracle_distributions,
    us_pr3_main_oracle_medicare,
    us_pr3_main_oracle_weeks,
    us_retirement_contributions,
    us_retirement_distributions,
    us_weeks_unemployed,
)
from test_support.microcosm_build.us_one_population_contract import (
    READER_CLONE_DEFECTS,
    ReaderPopulation,
    reader_population,
)
from test_support.microcosm_build.us_pr3_gate_differential import (
    PR3_GATES,
    payload_sha256,
    pr3_gate_differential,
)


@cache
def _readers(clone_defect: str | None) -> ReaderPopulation:
    return reader_population(clone_defect=clone_defect)


def _route_a(clone_defect: str | None = None) -> Frame:
    return _readers(clone_defect).route_a


def _pre_clone_contributions() -> Frame:
    # The first step of the stacked contributions fixture, before the clone.
    return with_us_retirement_contribution_inputs(
        us_retirement_contributions._frame(), seed=0, time_period=2024
    )


#: Each frame the differential grades, by name.
_FRAMES: dict[str, Callable[[], Frame]] = {
    "route_a": _route_a,
    **{
        f"route_a_{defect}": partial(_route_a, defect)
        for defect in READER_CLONE_DEFECTS
    },
    "pre_clone_contributions": _pre_clone_contributions,
    "stacked_weeks": us_weeks_unemployed._stacked_gate_frame,
    "stacked_medicare": us_medicare_take_up._stacked_frame,
    "stacked_distributions": us_retirement_distributions._stacked_frame,
    "stacked_contributions": us_retirement_contributions._stacked_frame,
}

#: Each frame with the gates whose inputs it carries.
_PAIRS: tuple[tuple[str, str], ...] = (
    *(
        (frame, gate)
        for frame in _FRAMES
        if frame.startswith("route_a")
        for gate in PR3_GATES
    ),
    ("pre_clone_contributions", "retirement_contributions"),
    ("stacked_weeks", "weeks_unemployed"),
    ("stacked_medicare", "medicare_take_up"),
    ("stacked_distributions", "retirement_distributions"),
    ("stacked_contributions", "retirement_contributions"),
)


@pytest.mark.parametrize(
    ("frame_name", "gate"),
    _PAIRS,
    ids=[f"{frame}-{gate}" for frame, gate in _PAIRS],
)
def test_main_and_live_gates_agree_on_route_a_frames(
    frame_name: str, gate: str
) -> None:
    result = pr3_gate_differential(_FRAMES[frame_name](), gates=(gate,))[gate]

    for side in ("oracle", "live"):
        assert "raised" not in result[side], (side, result[side])
    assert result["oracle"]["details"], "main's gate returned no statistics"
    assert result["live_sha256"] == result["oracle_sha256"], (
        result["oracle"],
        result["live"],
    )


@pytest.mark.parametrize("gate", PR3_GATES)
def test_the_oracle_is_mains_code_on_the_assembled_frame(gate: str) -> None:
    assembled = _readers(None).assembled
    clone_index = assembled.table("person")[support_clone_index_column("person")]
    natives = int(np.count_nonzero(clone_index.to_numpy() == 0))
    # The frame has clone rows, so a count equal to the natives is not
    # every row.
    assert 0 < natives < len(clone_index)

    result = pr3_gate_differential(assembled)[gate]

    for side in ("oracle", "live"):
        assert "raised" not in result[side], (side, result[side])
    details = result["oracle"]["details"]
    if gate == "weeks_unemployed":
        assert details["uc_constraint_rows"] == 0
    elif gate == "retirement_contributions":
        assert details["source_rows"] == natives
        assert details["source_reconciliation_rows"] == natives
    else:
        assert details["source_rows"] == natives
    if gate == "medicare_take_up":
        live = result["live"]
        assert live["details"]["source_rows"] == len(clone_index)
        moved = {key for key in details if details[key] != live["details"].get(key)}
        assert moved == {"source_rows"}
        assert set(live["details"]) == set(details)
        assert (live["passed"], live["failures"]) == (
            result["oracle"]["passed"],
            result["oracle"]["failures"],
        )
        assert result["live_sha256"] != result["oracle_sha256"]
    else:
        assert result["live_sha256"] == result["oracle_sha256"]


#: Each gate's oracle module, live module and copied names.
_COPIES: dict[str, tuple[ModuleType, ModuleType, tuple[str, ...]]] = {
    "weeks_unemployed": (
        us_pr3_main_oracle_weeks,
        weeks_unemployed,
        (
            "_weeks_unemployed_gate_scopes",
            "us_weeks_unemployed_summary",
            "us_weeks_unemployed_signal_gate",
        ),
    ),
    "medicare_take_up": (
        us_pr3_main_oracle_medicare,
        medicare_take_up,
        (
            "_asec_source_mask",
            "us_medicare_take_up_summary",
            "us_medicare_take_up_signal_gate",
        ),
    ),
    "retirement_contributions": (
        us_pr3_main_oracle_contributions,
        retirement_contributions,
        (
            "_asec_source_mask",
            "_source_reconciliation_mask",
            "us_retirement_contributions_summary",
            "us_retirement_contributions_signal_gate",
        ),
    ),
    "retirement_distributions": (
        us_pr3_main_oracle_distributions,
        retirement_distributions,
        (
            "_asec_source_mask",
            "_source_reconciliation_mask",
            "us_retirement_distributions_summary",
            "us_retirement_distributions_signal_gate",
        ),
    ),
}

#: What an oracle may import from besides its live module: the helpers the
#: live gate modules import, the frame and the array libraries.
_ORACLE_IMPORT_SOURCES = frozenset(
    {
        "__future__",
        "microcosm.build.gates",
        "microcosm.build.source_runtime",
        "microcosm.build.us_runtime.support_provenance",
        "microcosm.frame",
        "numpy",
        "pandas",
    }
)


@pytest.mark.parametrize("gate", PR3_GATES)
def test_each_oracle_holds_its_own_copies(gate: str) -> None:
    oracle, live, names = _COPIES[gate]
    live_copies = [getattr(live, name) for name in names]

    for name in names:
        assert getattr(oracle, name) is not getattr(live, name), name
        assert getattr(oracle, name).__module__ == oracle.__name__, name
    # No copy can reach the live code by another name.
    for name, value in vars(oracle).items():
        assert value is not live, name
        assert not any(value is copy for copy in live_copies), name
    allowed = _ORACLE_IMPORT_SOURCES | {live.__name__}
    tree = ast.parse(Path(oracle.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= allowed, node.lineno
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed and not node.level, node.lineno
            imported = {alias.name for alias in node.names}
            assert "*" not in imported and not imported & set(names), node.lineno


def test_a_changed_details_value_changes_the_digest() -> None:
    payload = {
        "name": "gate",
        "passed": True,
        "failures": [],
        "details": {"rows": 3, "share": 0.25},
    }
    reordered = {
        "details": {"share": 0.25, "rows": 3},
        "failures": [],
        "passed": True,
        "name": "gate",
    }

    assert payload_sha256(reordered) == payload_sha256(payload)
    assert payload_sha256(
        {**payload, "details": {"rows": 3, "share": 0.5}}
    ) != payload_sha256(payload)
    assert payload_sha256(
        {**payload, "details": {"rows": 3.0, "share": 0.25}}
    ) != payload_sha256(payload)
