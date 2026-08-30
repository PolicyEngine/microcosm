"""Whole-pool neutral immigration assignment ignores representation weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime.immigration import (
    US_IMMIGRATION_ALL_OUTPUT_COLUMNS,
    with_us_immigration_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _spine(*, cps: bool, offset: int = 0) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
            "age": [30, 50],
            "source_year": 2024,
            "source_household_id": ids + offset,
            "source_person_id": ids + offset,
            **(
                {
                    "PERIDNUM": [f"p{offset + 1}", f"p{offset + 2}"],
                    "PRCITSHP": [1, 5],
                    "PENATVTY": [57, 164],
                    "PEINUSYR": [0, 28],
                }
                if cps
                else {
                    "CIT": [1, 5],
                    "POBP": [57, 200],
                    "YOEP": [np.nan, 2021.0],
                }
            ),
        }
    )
    return Frame(
        {
            "person": person,
            **{
                entity: pd.DataFrame({f"{entity}_id": ids})
                for entity in US_SCHEMA.group_entities
            },
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([2.0, 2.0]), WeightKind.DESIGN)},
    )


def test_deprecated_weight_scale_is_inert() -> None:
    frame = _spine(cps=True)
    standalone = with_us_immigration_inputs(frame, seed=0, time_period=2024)
    scaled = with_us_immigration_inputs(
        frame, seed=0, time_period=2024, person_weight_scale=200.0
    )
    pd.testing.assert_frame_equal(standalone.person, scaled.person)


def test_postclone_chain_assigns_whole_pool_without_weight_scaling() -> None:
    assembled = assemble_spines(
        {"asec": _spine(cps=True), "acs": _spine(cps=False, offset=100)},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned = clone_us_frame_for_puf_support(assembled)
    observed_row_counts: list[int] = []

    def immigration(available: Frame) -> Frame:
        observed_row_counts.append(available.n("person"))
        return with_us_immigration_inputs(available, seed=0, time_period=2024)

    completed = multispine_pool_module._run_source_operator_chain(
        cloned,
        phase="post_clone",
        operator_names=("with_us_immigration_inputs",),
        operators={"with_us_immigration_inputs": immigration},
    )

    assert observed_row_counts == [cloned.n("person")]
    assert (
        completed.frame.person[list(US_IMMIGRATION_ALL_OUTPUT_COLUMNS)]
        .notna()
        .all()
        .all()
    )
    np.testing.assert_array_equal(
        completed.frame.weights_for("household").values,
        cloned.weights_for("household").values,
    )
    assert "person_design_weight_scaling" not in completed.receipt["suboperators"][
        0
    ].get("kernel_receipt", {})
