"""The copies' childcare take-up re-draw and the nested universal flag."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.childcare_copy_take_up import (
    CHILDCARE_COPY_TAKE_UP_STAGE_NAME,
    UKChildcareCopyTakeUpStageTransform,
    copy_benunits,
    redraw_childcare_copy_take_up,
)
from microcosm.build.uk_runtime.frs_take_up import (
    UK_CHILDCARE_HOURS_OUTPUT,
    UK_CHILDCARE_TAKE_UP_OUTPUTS,
    UK_UNDER_TWO_USAGE_OUTPUT,
    assert_universal_childcare_nesting_declaration,
    derive_childcare_take_up,
    derive_under_two_childcare_usage,
    uk_take_up_signal_gate,
    universal_childcare_marginal_rate,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

_NATIVES = 6_000
_OFFSET = 100_000
# One branch per copy stage: (household lineage overrides, copy multiple).
_COPY_BRANCHES = (
    ({"household_support_channel": "spi", "household_support_clone_index": 1}, 1),
    ({"household_is_cgt_support_copy": True, "cgt_support_copy_index": 1}, 2),
    (
        {
            "household_is_high_value_support_copy": True,
            "high_value_support_copy_index": 1,
        },
        3,
    ),
    ({"household_is_capital_gains_clone": True}, 4),
    (
        {
            "household_is_cgt_residential_clone": True,
            "cgt_residential_clone_index": 1,
        },
        5,
    ),
)


def _stage():
    return load_country_spec("uk").sources.stage_map()[
        CHILDCARE_COPY_TAKE_UP_STAGE_NAME
    ]


def _frame(contract, *, natives: int = _NATIVES):
    """Native households plus one copy of each per copy stage.

    Every copy carries its host's childcare draws, as the copy stages leave
    them: ``frs_take_up`` drew them keyed on the host's benefit unit.
    """

    native_ids = np.arange(1, natives + 1, dtype=np.int64)
    lineage = {
        "household_support_channel": "frs",
        "household_support_clone_index": 0,
        "household_is_cgt_support_copy": False,
        "cgt_support_copy_index": 0,
        "household_is_high_value_support_copy": False,
        "high_value_support_copy_index": 0,
        "household_is_capital_gains_clone": False,
        "household_is_cgt_residential_clone": False,
        "cgt_residential_clone_index": 0,
    }
    households = [
        pd.DataFrame(
            {"household_id": native_ids, "source_household_id": native_ids, **lineage}
        )
    ]
    for overrides, multiple in _COPY_BRANCHES:
        households.append(
            pd.DataFrame(
                {
                    "household_id": native_ids + multiple * _OFFSET,
                    "source_household_id": native_ids,
                    **{**lineage, **overrides},
                }
            )
        )
    household = pd.concat(households, ignore_index=True)
    household["household_weight"] = 1.0
    household["region"] = "LONDON"
    ids = household["household_id"].to_numpy(dtype=np.int64)
    sources = household["source_household_id"].to_numpy(dtype=np.int64)
    # Every other host's child is under 2, and its copies inherit its age.
    ages = np.where(sources % 2 == 0, 1, 3)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": ages,
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": ids,
            "benunit_source_id": sources,
            "benunit_support_clone_index": household[
                "household_support_clone_index"
            ].to_numpy(dtype=np.int64),
        }
    )
    inherited = derive_childcare_take_up(sources, contract=contract)
    for output in UK_CHILDCARE_TAKE_UP_OUTPUTS:
        benunit[output] = inherited[output]
    benunit["would_claim_uc_childcare"] = True
    benunit["is_married"] = False
    # frs_take_up drew each host child's usage keyed on the host's person and
    # unit; the copies inherit it.
    person[UK_UNDER_TWO_USAGE_OUTPUT] = derive_under_two_childcare_usage(
        pd.DataFrame({"person_id": sources, "age": ages}),
        claiming=inherited["would_claim_extended_childcare"],
        contract=contract,
    )
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period="2024"
    )


def test_natives_keep_their_draws_and_copies_are_redrawn_deterministically() -> None:
    contract = load_uk_take_up_contract()
    frame = _frame(contract)
    before = frame.table("benunit")

    first = UKChildcareCopyTakeUpStageTransform(contract=contract, stage=_stage())(
        frame
    )
    second = UKChildcareCopyTakeUpStageTransform(contract=contract, stage=_stage())(
        frame
    )
    after = first.table("benunit")
    native = (before["benunit_id"] <= _NATIVES).to_numpy()

    pd.testing.assert_frame_equal(after, second.table("benunit"))
    for output in UK_CHILDCARE_TAKE_UP_OUTPUTS:
        np.testing.assert_array_equal(
            after.loc[native, output].to_numpy(), before.loc[native, output].to_numpy()
        )
        assert after[output].dtype == before[output].dtype
    # Copies draw on their own benefit units through frs_take_up's helper, so
    # they no longer repeat their host's draws.
    copies = ~native
    redrawn = derive_childcare_take_up(
        after.loc[copies, "benunit_id"].to_numpy(), contract=contract
    )
    for output in UK_CHILDCARE_TAKE_UP_OUTPUTS:
        np.testing.assert_array_equal(
            after.loc[copies, output].to_numpy(), redrawn[output]
        )
    inherited = before.loc[copies, "would_claim_extended_childcare"].to_numpy()
    assert (redrawn["would_claim_extended_childcare"] != inherited).mean() > 0.3
    extended = after.loc[copies, "would_claim_extended_childcare"].to_numpy()
    expected = stable_identity_uniforms(
        after.loc[copies, "benunit_id"].to_numpy(),
        seed=0,
        salt="would_claim_extended_childcare",
    ) < contract.rate("extended_childcare")
    np.testing.assert_array_equal(extended, expected)
    # The anchored and UC childcare flags are not touched.
    np.testing.assert_array_equal(
        after["would_claim_uc_childcare"], before["would_claim_uc_childcare"]
    )


def test_copy_draws_hold_the_nesting_and_the_contract_rates() -> None:
    contract = load_uk_take_up_contract()
    after = redraw_childcare_copy_take_up(_frame(contract), contract=contract).table(
        "benunit"
    )
    copies = (after["benunit_id"] > _NATIVES).to_numpy()
    extended = after.loc[copies, "would_claim_extended_childcare"].to_numpy(bool)
    universal = after.loc[copies, "would_claim_universal_childcare"].to_numpy(bool)

    assert not (extended & ~universal).any()
    assert extended.mean() == pytest.approx(
        contract.rate("extended_childcare"), abs=0.02
    )
    assert universal.mean() == pytest.approx(
        universal_childcare_marginal_rate(contract), abs=0.02
    )
    # Among copies not claiming the extended hours, the conditional rate.
    assert universal[~extended].mean() == pytest.approx(
        contract.rate("universal_childcare"), abs=0.03
    )
    for output, key in (
        ("would_claim_tfc", "tax_free_childcare"),
        ("would_claim_targeted_childcare", "targeted_childcare"),
    ):
        realised = after.loc[copies, output].to_numpy(bool).mean()
        assert realised == pytest.approx(contract.rate(key), abs=0.02)
    distribution = contract.continuous_entry(UK_CHILDCARE_HOURS_OUTPUT)
    hours = after.loc[copies, UK_CHILDCARE_HOURS_OUTPUT].to_numpy(float)
    assert hours.min() >= float(distribution["lower"])
    assert hours.max() <= float(distribution["upper"])


def test_copies_are_read_from_every_clone_path() -> None:
    contract = load_uk_take_up_contract()
    frame = _frame(contract, natives=3)
    person, benunit, household = (
        frame.table("person"),
        frame.table("benunit").copy(),
        frame.table("household"),
    )
    copies = copy_benunits(person, benunit, household)

    assert copies.tolist() == [False] * 3 + [True] * 3 * len(_COPY_BRANCHES)
    # A unit's own support clone index marks it a copy as well.
    benunit.loc[0, "benunit_support_clone_index"] = 2
    assert copy_benunits(person, benunit, household)[0]


def test_stage_refuses_a_manifest_without_the_nesting_or_the_rewrites() -> None:
    stage = _stage()
    assert_universal_childcare_nesting_declaration(stage)
    contract = load_uk_take_up_contract()
    stripped = tuple(
        SourceOperationSpec(
            op.kind,
            {k: v for k, v in op.parameters.items() if k != "nested_under"},
        )
        for op in stage.operations
    )
    with pytest.raises(ValueError, match="nested_under"):
        UKChildcareCopyTakeUpStageTransform(
            contract=contract, stage=replace(stage, operations=stripped)
        )(_frame(contract, natives=3))
    with pytest.raises(ValueError, match="must declare the rewrites"):
        UKChildcareCopyTakeUpStageTransform(
            contract=contract,
            stage=replace(stage, rewrites=stage.rewrites[:-1]),
        )(_frame(contract, natives=3))


def test_frs_take_up_nests_universal_and_keeps_the_extended_draw() -> None:
    contract = load_uk_take_up_contract()
    ids = np.arange(1, 20_001, dtype=np.int64)
    drawn = derive_childcare_take_up(ids, contract=contract)
    extended = drawn["would_claim_extended_childcare"]
    universal = drawn["would_claim_universal_childcare"]

    # The extended flag is the draw it always was: same stream, same rate.
    np.testing.assert_array_equal(
        extended,
        stable_identity_uniforms(ids, seed=0, salt="would_claim_extended_childcare")
        < contract.rate("extended_childcare"),
    )
    assert not (extended & ~universal).any()
    np.testing.assert_array_equal(
        universal[~extended],
        stable_identity_uniforms(ids, seed=0, salt="would_claim_universal_childcare")[
            ~extended
        ]
        < contract.rate("universal_childcare"),
    )


def test_signal_gate_targets_the_nested_marginal_and_refuses_violations() -> None:
    contract = load_uk_take_up_contract()
    frame = redraw_childcare_copy_take_up(_frame(contract), contract=contract)
    benunit = frame.table("benunit")
    result = uk_take_up_signal_gate(frame, contract=contract)
    universal = result.details["benunit.would_claim_universal_childcare"]

    assert universal["target"] == pytest.approx(
        universal_childcare_marginal_rate(contract)
    )
    assert universal["extended_without_universal_units"] == 0
    assert not any("universal_childcare" in failure for failure in result.failures)

    broken = benunit.copy()
    broken["would_claim_universal_childcare"] = False
    broken_frame = uk_national_frame(
        person=frame.table("person"),
        benunit=broken,
        household=frame.table("household"),
        time_period="2024",
        household_weights=frame.weights_for("household").values,
    )
    failures = uk_take_up_signal_gate(broken_frame, contract=contract).failures
    assert any("without the universal hours" in failure for failure in failures)


def test_copies_children_redraw_their_under_two_usage_on_their_own_draws() -> None:
    contract = load_uk_take_up_contract()
    frame = _frame(contract)
    before = frame.table("person")
    after_frame = UKChildcareCopyTakeUpStageTransform(
        contract=contract, stage=_stage()
    )(frame)
    after = after_frame.table("person")
    benunit = after_frame.table("benunit")
    native = (after["person_id"] <= _NATIVES).to_numpy()

    # Natives keep their frs_take_up draws.
    np.testing.assert_array_equal(
        after.loc[native, UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(),
        before.loc[native, UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(),
    )
    # A copy's child draws against its own unit's re-drawn flag, keyed on its
    # own person_id.
    copies = ~native
    claiming = benunit.set_index("benunit_id").loc[
        after.loc[copies, "person_benunit_id"], "would_claim_extended_childcare"
    ]
    expected = derive_under_two_childcare_usage(
        after.loc[copies].reset_index(drop=True),
        claiming=claiming.to_numpy(dtype=bool),
        contract=contract,
    )
    np.testing.assert_array_equal(
        after.loc[copies, UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(), expected
    )
    under_two = (after["age"] < 2).to_numpy()
    drawn = (
        copies
        & under_two
        & np.isin(
            after["person_benunit_id"],
            benunit.loc[benunit["would_claim_extended_childcare"], "benunit_id"],
        )
    )
    usage = after.loc[drawn, UK_UNDER_TWO_USAGE_OUTPUT].to_numpy() > 0
    assert usage.mean() == pytest.approx(
        contract.rate("extended_childcare_under_2_usage"), abs=0.03
    )
    assert (after.loc[~under_two, UK_UNDER_TWO_USAGE_OUTPUT] == 30.0).all()
