"""Invented identity controls using the maintained clone and assignment operators."""

import numpy as np
import pandas as pd
import pytest
from test_us_atomic_survey_clone_graph import invented_population

from microcosm.build import atomic_geography as geography
from microcosm.build.us_runtime import atomic_block_support as blocks
from microcosm.build.us_runtime import puf_support
from microcosm.build.us_runtime.graph_atomic_survey_clone import (
    ASSIGNMENT_IDENTITY,
    atomic_survey_clone_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations


def _case(*, last_id=42):
    original = invented_population(last_id=last_id)
    cloned = puf_support.clone_us_frame_for_puf_support(
        original, clone_attachment_fraction=1.0, clone_attachment_seed=0
    )
    ids = {name: "invented-" + name for name in ("district", "population", "puma")}
    areas = [
        int(base) + i
        for base in ("010010201001000", "020130001001000")
        for i in range(64)
    ]
    payload = blocks.assemble_atomic_block_support(
        block_population={area: 1 for area in areas},
        cd_by_block={area: 101 if area < 2 * 10**13 else 200 for area in areas},
        puma_by_tract={areas[0] // 10000: 100001, areas[64] // 10000: 200002},
        source_ids=ids,
    )
    definition = blocks.assignment_definition(
        identity=ASSIGNMENT_IDENTITY,
        state_column="observed_state",
        puma_column="observed_puma",
        source_ids=ids,
        seed=17,
    )
    return (
        original,
        cloned,
        definition,
        {blocks.SYSTEM: geography.decode_atomic_support(payload)},
    )


def _draws(households, definition, supports):
    result = geography.assign_atomic(households, definition, supports)
    return (
        pd.concat([households.loc[:, list(ASSIGNMENT_IDENTITY)], result], axis=1)
        .set_index(list(ASSIGNMENT_IDENTITY), verify_integrity=True)
        .sort_index()
    )


@pytest.mark.parametrize("change", ("order", "subset", "cohort_stride"))
def test_same_source_clone_draw_survives_cohort_coordinate_changes(change):
    _original, cloned, definition, supports = _case()
    households = cloned.table("household")
    expected = _draws(households, definition, supports)
    if change == "order":
        selected = households.iloc[::-1]
    elif change == "subset":
        selected = households.iloc[[6, 1, 4]]
    else:
        _other, alternate, same_definition, same_support = _case(last_id=9999)
        assert same_definition == definition
        assert same_support[blocks.SYSTEM].sha256 == supports[blocks.SYSTEM].sha256
        selected = alternate.table("household")
        previous_ids = households.set_index(list(ASSIGNMENT_IDENTITY)).household_id
        next_ids = selected.set_index(list(ASSIGNMENT_IDENTITY)).household_id
        common = previous_ids.index.intersection(next_ids.index)
        assert len(common) == 4
        assert np.any(
            previous_ids.loc[common].to_numpy() != next_ids.loc[common].to_numpy()
        )
        selected = selected.loc[
            pd.MultiIndex.from_frame(selected.loc[:, list(ASSIGNMENT_IDENTITY)]).isin(
                common
            )
        ]
    actual = _draws(selected, definition, supports)
    pd.testing.assert_frame_equal(actual, expected.loc[actual.index], check_exact=True)


def test_completed_clone_keys_permit_distinct_block_draws():
    _original, cloned, definition, supports = _case()
    households = cloned.table("household")
    assert not households.duplicated(list(ASSIGNMENT_IDENTITY)).any()
    actual = _draws(households, definition, supports)
    pairs = actual.census_block_geoid.unstack(ASSIGNMENT_IDENTITY[1])
    assert set(pairs.columns) == {0, 1}
    assert pairs.notna().all().all()
    # Fixed invented support and seed: require evidence of independent clone
    # assignment, while permitting individual pairs to choose the same block.
    assert pairs[0].ne(pairs[1]).any()


def test_required_unknown_state_survives_initial_clone_and_refuses_assignment():
    original, _cloned, definition, supports = _case()
    original.table("household").loc[0, "observed_state"] = pd.NA
    cloned = puf_support.clone_us_frame_for_puf_support(
        original, clone_attachment_fraction=1.0, clone_attachment_seed=0
    )
    households = cloned.table("household")
    missing = households.observed_state.isna()
    assert missing.sum() == 2
    assert set(households.loc[missing, ASSIGNMENT_IDENTITY[1]]) == {0, 1}
    with pytest.raises(ValueError, match="missing required observed geography"):
        geography.assign_atomic(households, definition, supports)


@pytest.mark.parametrize("defect", ("missing_discriminator", "duplicate_key"))
def test_postclone_identity_contract_refuses_ambiguous_rows(defect):
    original, cloned, definition, supports = _case()
    if defect == "missing_discriminator":
        definition = {**definition, "identity": [ASSIGNMENT_IDENTITY[0]]}
        with pytest.raises(ValueError, match="stable source/clone identity"):
            atomic_survey_clone_nodes(
                definition, frame_column_declarations(original), base="source"
            )
    else:
        households = cloned.table("household").copy(deep=True)
        households[ASSIGNMENT_IDENTITY[1]] = np.zeros(len(households), dtype=np.int64)
        with pytest.raises(ValueError, match="identity"):
            geography.assign_atomic(households, definition, supports)
