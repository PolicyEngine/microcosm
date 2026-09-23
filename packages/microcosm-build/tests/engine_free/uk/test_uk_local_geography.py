from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import align_area_targets


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["S001", "E001"],
            "name": ["Somewhere", "Elsewhere"],
            "population": [0.5, 1.5],
            "uc_households": [1.0, 2.0],
        }
    )


def test_align_area_targets_orders_rows_and_selects_metric_columns() -> None:
    aligned = align_area_targets(_targets(), ["E001", "S001"])

    assert aligned.index.tolist() == ["E001", "S001"]
    # Metadata columns (code, name) are excluded from the inferred metrics.
    assert aligned.columns.tolist() == ["population", "uc_households"]
    np.testing.assert_allclose(aligned["population"], [1.5, 0.5])


def test_align_area_targets_honours_explicit_metric_names() -> None:
    aligned = align_area_targets(
        _targets(),
        ["S001"],
        metric_names=["uc_households"],
    )
    assert aligned.columns.tolist() == ["uc_households"]
    np.testing.assert_allclose(aligned["uc_households"], [1.0])

    with pytest.raises(ValueError, match="missing metric column"):
        align_area_targets(_targets(), ["S001"], metric_names=["nonexistent"])


def test_align_area_targets_rejects_missing_and_duplicate_codes() -> None:
    with pytest.raises(ValueError, match="missing area code"):
        align_area_targets(_targets(), ["E001", "X999"])

    with pytest.raises(ValueError, match="must be unique"):
        align_area_targets(_targets(), ["E001", "E001"])

    duplicated = pd.concat([_targets(), _targets().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique; duplicate value"):
        align_area_targets(duplicated, ["E001"])


def test_align_area_targets_requires_finite_values() -> None:
    bad = _targets()
    bad.loc[0, "population"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        align_area_targets(bad, ["E001", "S001"])
