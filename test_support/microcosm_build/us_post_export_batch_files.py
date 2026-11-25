"""Shared checks for the post-export scorer's household batch files.

The scorer keeps each household batch of the written H5 in its own file and
reads it back each time it scores that batch, so the scoring process holds no
frame of the written H5 between batches. These helpers compare a batch read
back from its file with the batch the scorer selected from the loaded H5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.frame import Frame


def assert_same_batch(restored: Frame, expected: Frame) -> None:
    """``restored`` holds exactly ``expected``'s data.

    Same entities, columns, dtypes and index; numeric, boolean and date
    columns equal byte for byte (so the sign of a zero and a NaN's bit pattern
    count); object columns equal value by value with the same Python types;
    the same weights, weight kinds and strata.
    """

    assert restored.entities == expected.entities
    for entity in expected.entities:
        restored_table = restored.table(entity)
        expected_table = expected.table(entity)
        pd.testing.assert_frame_equal(restored_table, expected_table, check_exact=True)
        for column in expected_table.columns:
            restored_values = restored_table[column].to_numpy()
            expected_values = expected_table[column].to_numpy()
            if expected_values.dtype.kind == "O":
                assert [type(value) for value in restored_values] == [
                    type(value) for value in expected_values
                ], (entity, column)
            else:
                assert restored_values.tobytes() == expected_values.tobytes(), (
                    entity,
                    column,
                )
    assert restored.weighted_entities == expected.weighted_entities
    for entity in expected.weighted_entities:
        restored_weights = restored.weights_for(entity)
        expected_weights = expected.weights_for(entity)
        assert restored_weights.kind == expected_weights.kind, entity
        assert (
            np.asarray(restored_weights.values).tobytes()
            == np.asarray(expected_weights.values).tobytes()
        ), entity
    pd.testing.assert_series_equal(restored.strata, expected.strata, check_exact=True)
