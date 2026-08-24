"""The E7 identity receipt must never pass vacuously (#747 review).

A receipt's whole value is that it certifies something. The E7 branch
recomputed the support-channel layer only when the synthetic flag was
present and otherwise returned an empty recomputation — over which the
mismatch loops never ran, so the receipt reported
``identical_under_permutation: true`` and ``matches_stored_columns: true``
with exit 0 on an artifact where nothing had been checked. These tests pin
the refusals that replaced that silence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

_TOOL_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "verify_uk_identity_stability.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_uk_identity_stability", _TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    synthetic: bool = True,
    source_keys: bool = True,
    stored_channel: bool = True,
):
    """A two-household frame carrying the E7 support-channel layer."""

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_benunit_id": [10, 10, 20],
            "person_household_id": [100, 100, 200],
        }
    )
    benunit = pd.DataFrame({"benunit_id": [10, 20]})
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_weight": [10.0, 20.0],
        }
    )
    if synthetic:
        household["household_is_spi_synthetic"] = [False, True]
    if source_keys:
        household["source_year"] = [2024, 2024]
        household["source_household_id"] = [100, 100]
    if stored_channel:
        household["household_support_channel"] = ["frs", "spi"]
        household["household_support_clone_index"] = [0, 1]
        household["source_household_key"] = ["2024:100", "2024:100"]
        person["person_support_channel"] = ["frs", "frs", "spi"]
        benunit["benunit_support_channel"] = ["frs", "spi"]
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestE7Receipt:
    def test_a_complete_artifact_receipts_green(self) -> None:
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(_frame(), permutation_seed=7)
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is True
        # The receipt names what it compared, so a green result is auditable.
        assert receipt["columns_compared"]["household"] == [
            "household_support_channel",
            "household_support_clone_index",
            "source_household_key",
        ]

    def test_an_artifact_without_the_e7_layer_is_refused(self) -> None:
        # Previously this returned a green receipt over an empty comparison.
        tool = _load_tool()
        with pytest.raises(ValueError, match="no\\s+household_is_spi_synthetic"):
            tool.e7_identity_receipt(_frame(synthetic=False), permutation_seed=7)

    def test_missing_source_keys_are_refused_not_skipped(self) -> None:
        # Skipping the source key would silently shrink the receipt's
        # coverage while still reporting a pass.
        tool = _load_tool()
        with pytest.raises(ValueError, match="source key cannot be recomputed"):
            tool.e7_identity_receipt(_frame(source_keys=False), permutation_seed=7)

    def test_a_column_absent_from_the_store_is_a_mismatch(self) -> None:
        # The store not carrying a column this receipt certifies is a failed
        # comparison, not a narrower one.
        tool = _load_tool()
        receipt = tool.e7_identity_receipt(
            _frame(stored_channel=False), permutation_seed=7
        )
        assert receipt["identical_under_permutation"] is True
        assert receipt["matches_stored_columns"] is False
        assert (
            "household_support_channel"
            in receipt["stored_column_mismatches"]["household"]
        )

    def test_a_corrupted_stored_channel_is_caught(self) -> None:
        tool = _load_tool()
        frame = _frame()
        household = frame.table("household")
        household.loc[household.index[-1], "household_support_channel"] = "frs"
        receipt = tool.e7_identity_receipt(frame, permutation_seed=7)
        assert receipt["matches_stored_columns"] is False
