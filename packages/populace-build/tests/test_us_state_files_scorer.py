import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_state_scorer_module():
    root = Path(__file__).resolve().parents[3]
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    path = root / "tools" / "score_us_state_files.py"
    spec = importlib.util.spec_from_file_location("score_us_state_files", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test__given_partial_state_directory__then_inventory_reports_missing_files(
    tmp_path,
) -> None:
    scorer = _load_state_scorer_module()
    (tmp_path / "CA.h5").write_bytes(b"placeholder")

    inventory = scorer.required_state_file_inventory(tmp_path)

    assert inventory["expected_state_count"] == 51
    assert str(tmp_path / "CA.h5") in inventory["present_files"]
    assert len(inventory["missing_files"]) == 50
    with pytest.raises(SystemExit, match="never downloads inputs"):
        scorer.assert_required_state_files_present(inventory)


def test__given_parent_with_states_subdir__then_inventory_uses_nested_state_files(
    tmp_path,
) -> None:
    scorer = _load_state_scorer_module()
    nested = tmp_path / "states"
    nested.mkdir()
    (nested / "CA.h5").write_bytes(b"placeholder")

    inventory = scorer.required_state_file_inventory(tmp_path)

    assert inventory["state_h5_dir"] == str(nested)
    assert str(nested / "CA.h5") in inventory["present_files"]


def test__given_overlapping_state_ids__then_state_strata_make_concat_unambiguous():
    scorer = _load_state_scorer_module()
    ca = scorer._frame_with_state_stratum(_minimal_state_frame(6, 10.0), "CA")
    ny = scorer._frame_with_state_stratum(_minimal_state_frame(36, 20.0), "NY")

    combined = ca.concat(ny)

    assert set(combined.strata) == {"legacy_state:CA", "legacy_state:NY"}
    assert combined.n("household") == 2
    assert combined.n("person") == 2
    assert combined.table("household")["household_id"].tolist() == [1, 2]
    assert combined.weights_for("household").values.tolist() == [10.0, 20.0]


def test__given_state_file_metadata__then_collection_hash_is_order_stable():
    scorer = _load_state_scorer_module()
    state_files = [
        {"state": "NY", "sha256": "b" * 64},
        {"state": "CA", "sha256": "a" * 64},
    ]

    forward = scorer._collection_sha256(state_files)
    reverse = scorer._collection_sha256(tuple(reversed(state_files)))

    assert forward == reverse
    assert len(forward) == 64


def _minimal_state_frame(state_fips: int, weight: float) -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1], dtype="int64"),
                "person_household_id": np.asarray([1], dtype="int64"),
                "person_tax_unit_id": np.asarray([1], dtype="int64"),
                "person_spm_unit_id": np.asarray([1], dtype="int64"),
                "person_family_id": np.asarray([1], dtype="int64"),
                "person_marital_unit_id": np.asarray([1], dtype="int64"),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1], dtype="int64"),
                "state_fips": np.asarray([state_fips], dtype="int64"),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": np.asarray([1], dtype="int64")}),
        "spm_unit": pd.DataFrame({"spm_unit_id": np.asarray([1], dtype="int64")}),
        "family": pd.DataFrame({"family_id": np.asarray([1], dtype="int64")}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([1], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([weight], dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )
