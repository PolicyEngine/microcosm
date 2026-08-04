"""Contract tests for the documented-absent-inputs register (populace #351/#249)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


#: The unsourced SNAP work-requirement input families: exemption inputs
#: with no survey source (#351) and the work-program participation
#: family (#249).
_EXPECTED = {
    "is_homeless",
    "was_in_foster_care",
    "is_snap_work_program_participant",
    "weekly_snap_work_program_hours",
    "is_snap_workfare_participant",
}


def test_register_covers_exactly_the_known_unsourced_inputs() -> None:
    builder = _load_builder_module()
    assert set(builder.US_DOCUMENTED_ABSENT_INPUTS) == _EXPECTED


def test_every_entry_names_a_tracking_issue() -> None:
    builder = _load_builder_module()
    for column, reason in builder.US_DOCUMENTED_ABSENT_INPUTS.items():
        assert re.search(r"PolicyEngine/populace#\d+", reason), (
            f"{column}: reason must cite a tracking issue"
        )


def test_seeded_inputs_are_not_documented_as_absent() -> None:
    # These are produced by stages; listing them here would contradict
    # the build. is_incapable_of_self_care joined this set with the
    # adult_care_inputs stage (populace#451 item 1), which seeds it from
    # the measured ASEC PEDISDRS self-care difficulty item.
    builder = _load_builder_module()
    for column in (
        "is_pregnant",
        "is_disabled",
        "is_snap_abawd_discretionary_exempt",
        "is_incapable_of_self_care",
    ):
        assert column not in builder.US_DOCUMENTED_ABSENT_INPUTS


def test_register_is_disjoint_from_the_degenerate_exclusions() -> None:
    # The reviewed-exclusions register covers PERSISTED columns stuck at
    # defaults; this register covers columns that are not persisted at
    # all. An input in both would be contradictory.
    builder = _load_builder_module()
    assert not set(builder.US_DOCUMENTED_ABSENT_INPUTS) & set(
        builder.US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS
    )
