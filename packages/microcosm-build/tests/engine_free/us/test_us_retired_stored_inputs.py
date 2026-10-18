"""Moving the donor's retired stored inputs to the live ones (microcosm#1263).

Every frame here is invented. The engine is a variable-name set, so no
policyengine-us is imported. The installed engine's side of the premise (the
live input is a Person Boolean defaulting to True, and neither retired name is
defined) is pinned by
``packages/microcosm-data/tests/engine_contract/us/test_stored_inputs.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.retired_stored_inputs import (  # noqa: E402
    LIVE_WIC_INPUT,
    RETIRED_PART_B_INPUT,
    RETIRED_WIC_INPUT,
    with_live_stored_inputs,
)
from microcosm.data.stored_inputs import undefined_stored_inputs  # noqa: E402
from microcosm.frame import Frame  # noqa: E402
from test_support.microcosm_build.us_acs_spm_role import acs_frame  # noqa: E402

#: The live engine as far as these tests need it: the live WIC input, and
#: neither retired name.
ENGINE = frozenset({LIVE_WIC_INPUT, "age", "person_id", "wic"})
ACS_CHANNEL = "acs_2024_1yr"


def _remedy(frame, engine=ENGINE, allowed=None):
    """The rule with the release tool's mask: the ACS-spine people may lack one."""

    if allowed is None:
        spine = frame.table("person")["person_spine"]
        allowed = spine.eq(ACS_CHANNEL).fillna(False).to_numpy(dtype=bool)
    return with_live_stored_inputs(
        frame, engine_variables=engine, may_lack_decision=allowed
    )


def _frame(channels, decisions=None, part_b=None) -> Frame:
    """One person per household, with the given channels and stored columns."""

    frame = acs_frame([[(20, 40.0)] for _ in channels])
    person = frame.table("person").copy()
    person["person_support_channel"] = channels
    person["person_spine"] = [
        ACS_CHANNEL if channel == ACS_CHANNEL else "asec_puf" for channel in channels
    ]
    if decisions is not None:
        person[RETIRED_WIC_INPUT] = pd.Series(decisions, dtype=object)
    if part_b is not None:
        person[RETIRED_PART_B_INPUT] = part_b
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )


def test_donor_decisions_are_kept_and_acs_gaps_take_wic_up():
    frame = _frame(
        ["asec", "puf_tax_detail", ACS_CHANNEL, ACS_CHANNEL],
        [True, False, None, np.nan],
    )

    result = _remedy(frame)

    person = result.frame.table("person")
    assert RETIRED_WIC_INPUT not in person.columns
    assert person[LIVE_WIC_INPUT].dtype == bool
    assert person[LIVE_WIC_INPUT].tolist() == [True, False, True, True]
    wic = result.provenance["wic"]
    assert wic["renamed"] is True
    assert wic["stored_decisions_kept"] == 2
    assert wic["stored_decisions_true"] == 1
    assert wic["people_assumed_true"] == 2
    assert wic["people_allowed_without_decision"] == 2
    assert wic["status"] == "interim"
    assert "microcosm#1156" in wic["flip_condition"]


def test_part_b_premiums_stop_being_stored():
    frame = _frame(["asec", ACS_CHANNEL, "puf_tax_detail"], part_b=[120.0, np.nan, 0.0])

    result = _remedy(frame)

    assert RETIRED_PART_B_INPUT not in result.frame.table("person").columns
    assert result.provenance["medicare_part_b_premiums"] == {
        "dropped": True,
        "stored_values": 2,
    }


def test_the_remedied_columns_pass_the_stored_inputs_rule():
    frame = _frame(["asec", ACS_CHANNEL], [False, None], part_b=[10.0, np.nan])
    before = undefined_stored_inputs(
        frame.table("person").columns, engine_variables=ENGINE
    )
    assert set(before) >= {RETIRED_WIC_INPUT, RETIRED_PART_B_INPUT}

    person = _remedy(frame).frame.table("person")
    after = undefined_stored_inputs(person.columns, engine_variables=ENGINE)

    assert RETIRED_WIC_INPUT not in after
    assert RETIRED_PART_B_INPUT not in after
    assert LIVE_WIC_INPUT not in after


def test_a_frame_without_retired_inputs_is_returned_unchanged():
    frame = _frame(["asec", ACS_CHANNEL])

    result = _remedy(frame)

    assert result.frame is frame
    assert result.provenance["wic"] == {"renamed": False}
    assert result.provenance["medicare_part_b_premiums"] == {"dropped": False}


def test_ids_weights_and_other_columns_are_unchanged():
    frame = _frame(
        ["asec", ACS_CHANNEL, ACS_CHANNEL], [True, None, None], part_b=[1.0, None, None]
    )

    result = _remedy(frame).frame

    before, after = frame.table("person"), result.table("person")
    kept = [
        c for c in before.columns if c not in (RETIRED_WIC_INPUT, RETIRED_PART_B_INPUT)
    ]
    pd.testing.assert_frame_equal(before[kept], after[kept])
    for entity in frame.entities:
        if entity != "person":
            pd.testing.assert_frame_equal(frame.table(entity), result.table(entity))
    np.testing.assert_array_equal(
        frame.weights_for("household").values, result.weights_for("household").values
    )


def test_the_rule_reads_no_provenance_column():
    """Only the caller's mask says who may lack a decision."""

    frame = _frame(["asec", ACS_CHANNEL], [None, False])

    person = _remedy(frame, allowed=np.array([True, False])).frame.table("person")

    assert person[LIVE_WIC_INPUT].tolist() == [True, False]


@pytest.mark.parametrize(
    ("channels", "decisions", "match"),
    [
        (["asec", ACS_CHANNEL], [None, None], "did not allow"),
        (["puf_tax_detail"], [np.nan], "did not allow"),
        (["asec"], ["yes"], "not a Boolean"),
        (["asec"], [2], "not a Boolean"),
        (["asec"], [1 + 0j], "not a Boolean"),
    ],
)
def test_it_refuses_rather_than_guessing(channels, decisions, match):
    with pytest.raises(ValueError, match=match):
        _remedy(_frame(channels, decisions))


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (pd.array([True, False, None], dtype="boolean"), [True, False, True]),
        (np.array([True, False, True]), [True, False, True]),
        (np.array([1.0, 0.0, np.nan]), [True, False, True]),
        (pd.Series([True, False, None], dtype=object), [True, False, True]),
    ],
    ids=["pandas-boolean-NA", "numpy-bool", "float-0-1-nan", "object"],
)
def test_every_stored_boolean_dtype_reads_alike(stored, expected):
    frame = _frame(["asec", "puf_tax_detail", ACS_CHANNEL])
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].assign(**{RETIRED_WIC_INPUT: stored})
    typed = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )

    person = _remedy(typed).frame.table("person")

    assert person[LIVE_WIC_INPUT].dtype == bool
    assert person[LIVE_WIC_INPUT].tolist() == expected


def test_a_fraction_is_not_read_as_a_decision():
    """Only bools and int/float 0/1, as policyengine.py's reader accepts."""

    from fractions import Fraction

    with pytest.raises(ValueError, match="not a Boolean"):
        _remedy(_frame(["asec"], [Fraction(1)]))


def test_zero_one_numbers_are_read_as_decisions():
    frame = _frame(["asec", "asec", "asec"], [1, 0.0, np.float64(1.0)])

    person = _remedy(frame).frame.table("person")

    assert person[LIVE_WIC_INPUT].tolist() == [True, False, True]


def test_it_refuses_both_wic_names():
    frame = _frame(["asec"], [True])
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].assign(**{LIVE_WIC_INPUT: [False]})
    both = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )

    with pytest.raises(ValueError, match="both"):
        _remedy(both)


@pytest.mark.parametrize(
    "allowed",
    [np.array([True]), np.array([1, 0]), np.array([[True, False]])],
)
def test_the_mask_must_be_one_boolean_per_person(allowed):
    with pytest.raises(ValueError, match="one entry per person"):
        _remedy(_frame(["asec", ACS_CHANNEL], [True, None]), allowed=allowed)


@pytest.mark.parametrize(
    "engine",
    [
        ENGINE - {LIVE_WIC_INPUT},
        ENGINE | {RETIRED_WIC_INPUT},
        ENGINE | {RETIRED_PART_B_INPUT},
    ],
)
def test_it_refuses_an_engine_the_remedy_does_not_fit(engine):
    with pytest.raises(ValueError, match="Review microcosm.data.stored_inputs"):
        _remedy(_frame(["asec"], [True]), engine=engine)


_CHANNELS = st.sampled_from(["asec", "puf_tax_detail", ACS_CHANNEL])


@st.composite
def _people(draw):
    """Channels and decisions where only ACS people may lack one."""

    channels = draw(st.lists(_CHANNELS, min_size=1, max_size=40))
    decisions = []
    for channel in channels:
        if channel == ACS_CHANNEL and draw(st.booleans()):
            decisions.append(None)
        else:
            decisions.append(draw(st.booleans()))
    return channels, decisions


@settings(max_examples=150, deadline=None)
@given(people=_people())
def test_stored_decisions_survive_and_only_acs_gaps_are_filled(people):
    channels, decisions = people

    result = _remedy(_frame(channels, decisions))

    live = result.frame.table("person")[LIVE_WIC_INPUT].tolist()
    assert len(live) == len(decisions)
    for stored, value, channel in zip(decisions, live, channels, strict=True):
        if stored is None:
            assert channel == ACS_CHANNEL
            assert value is True
        else:
            assert value is stored
    wic = result.provenance["wic"]
    assert wic["stored_decisions_kept"] + wic["people_assumed_true"] == len(live)
    assert wic["stored_decisions_true"] == sum(d is True for d in decisions)
    # A second pass finds nothing retired and changes nothing.
    again = _remedy(result.frame)
    assert again.frame is result.frame
