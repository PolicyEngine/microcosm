"""Tests split from packages/microcosm-build/tests/test_us_spm_universe_source.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_spm_universe_source import *


def test_the_real_support_clone_operator_output_is_accepted():
    """The copy-key rule holds on ``clone_us_frame_for_puf_support`` itself.

    Invented rows. Imported lazily: the operator's module pulls the country
    engine, which the rest of this file never needs.
    """
    from microcosm.build.us_runtime.puf_support import (
        clone_us_frame_for_puf_support,
    )

    frame = _frame(
        [
            {"hid": 1, "unit": 1, "chan": _ASEC_CHANNEL, "spm_id": 11},
            {"hid": 1, "unit": 2, "chan": _ASEC_CHANNEL, "spm_id": 12},
            {"hid": 2, "unit": 3, "chan": _ASEC_CHANNEL, "spm_id": 21},
        ],
        without_support_metadata=True,
    )
    cloned = clone_us_frame_for_puf_support(frame)
    person = cloned.table("person")
    # The operator's shape this rule depends on: SPM_ID copied verbatim onto
    # a second copy whose SPM units are new ids.
    assert sorted(person["person_support_clone_index"].unique().tolist()) == [0, 1]
    by_copy = person.groupby("person_support_clone_index")
    assert by_copy[ASEC_NATIVE_UNIT_COLUMN].apply(sorted).tolist() == [
        [11, 12, 21],
        [11, 12, 21],
    ]
    assert person["person_spm_unit_id"].nunique() == 6

    result = attach_spm_universe_status(cloned)
    assert result.status.tolist() == [INCLUDED] * 6
