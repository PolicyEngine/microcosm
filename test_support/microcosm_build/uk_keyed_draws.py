"""A stage's QRF draws move only for the records whose own inputs move.

Every UK stage that imputes with ``RegimeGatedQRF`` draws from uniforms keyed
on each record's id (``stable_identity_uniforms``), never from a positional
stream. A positional stream let one engine change re-randomise the whole spine:
a few households' net income moved, the high-value copies changed the frame's
length, and every later household got different random numbers. This helper
checks the property the keying gives, on a stage's own imputation function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

#: An id far from any fixture's, for the recipient the check adds.
_ADDED_ID_OFFSET = 1_000_000


def assert_draws_are_keyed_on_ids(
    draw: Callable[[pd.DataFrame, np.ndarray], pd.DataFrame],
    recipient: pd.DataFrame,
    ids: np.ndarray,
    *,
    perturbed_column: str,
    perturbed_value: object,
) -> None:
    """Reorder, change one recipient and add another: no other draw moves.

    ``draw(recipient, ids)`` returns the draws in recipient row order. The
    check reverses the recipients, gives the last one (now first) a new
    ``perturbed_column`` value, appends a copy of the first recipient under a
    new id, and asserts that every recipient it did not touch draws exactly
    what it drew before.
    """

    ids = np.asarray(ids)
    base = draw(recipient.reset_index(drop=True), ids).set_axis(ids)
    changed = recipient.iloc[::-1].reset_index(drop=True)
    changed_ids = ids[::-1].copy()
    changed.loc[0, perturbed_column] = perturbed_value
    changed = pd.concat([changed, recipient.iloc[[0]]], ignore_index=True)
    changed_ids = np.append(changed_ids, ids.max() + _ADDED_ID_OFFSET)
    after = draw(changed, changed_ids).set_axis(changed_ids)
    untouched = ids[:-1]
    pd.testing.assert_frame_equal(
        after.loc[untouched], base.loc[untouched], check_dtype=False
    )
