"""Re-draw the childcare take-up of support and CGT copies on their own units.

``frs_take_up`` draws the childcare flags and the extended-hours usage keyed on
each FRS benefit unit before any copy exists, so every copy the later stages
make (the SPI support channel and its income-band donors, the high-value home
support, the CGT support split, the CGT incidence clone and the residential
arms) inherits its host's draws. A small cell of hosts then fixes the take-up
of all its copies: the high-income young-family cell had a few hundred
independent draws and a realised extended take-up well above the contract
rate. This stage, which runs after the last copy stage, re-draws
``would_claim_tfc``, ``would_claim_extended_childcare``,
``would_claim_universal_childcare`` (nested under the extended flag),
``would_claim_targeted_childcare`` and ``maximum_extended_childcare_hours_usage``
for the copies only, through ``frs_take_up``'s own
``derive_childcare_take_up`` at the same contract rates, seed and salts, and
the copies' children's ``max_free_entitlement_hours_used`` (the under-2 usage
draw) against their own unit's re-drawn working-parent flag, through
``derive_under_two_childcare_usage`` keyed on the copy's own ``person_id``.

The draws are keyed on the copy's own ``benunit_id``. Every copy stage gives a
copy the structural id ``source id + k × offset`` with an offset above every
id in the frame, so a copy's id is unique and differs from every native id:
the copy's streams are independent of its host's, and a rebuild of the same
frame draws the same values. Native benefit units keep their ``frs_take_up``
draws untouched. Anchored flags (Child Benefit, Pension Credit, Universal
Credit) and ``would_claim_uc_childcare`` are left as they are.

A copy is a benefit unit whose household carries a non-empty clone path
(``uk_household_clone_paths``: the SPI support channel, the CGT support split,
the high-value support, the CGT incidence clone or a residential arm), or
whose own support clone index is positive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_brma import _benunit_household_map
from microcosm.build.uk_runtime.frs_take_up import (
    UK_CHILDCARE_HOURS_OUTPUT,
    UK_CHILDCARE_TAKE_UP_FLAGS,
    UK_CHILDCARE_TAKE_UP_OUTPUTS,
    UK_EXTENDED_CHILDCARE_OUTPUT,
    UK_UNDER_TWO_USAGE_OUTPUT,
    assert_under_two_usage_declaration,
    assert_universal_childcare_nesting_declaration,
    derive_childcare_take_up,
    derive_under_two_childcare_usage,
    person_unit_flag,
    take_up_signal_outputs,
    uk_take_up_signal_receipt,
)
from microcosm.build.uk_runtime.household_lineage import (
    uk_household_clone_paths,
    without_geographic_branch,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.take_up_contract import UKTakeUpContract
from microcosm.frame import Frame

__all__ = [
    "CHILDCARE_COPY_TAKE_UP_REWRITES",
    "CHILDCARE_COPY_TAKE_UP_STAGE_NAME",
    "UKChildcareCopyTakeUpStageTransform",
    "copy_benunits",
    "redraw_childcare_copy_take_up",
]

CHILDCARE_COPY_TAKE_UP_STAGE_NAME = "childcare_copy_take_up"
BENUNIT_SUPPORT_CLONE_INDEX = "benunit_support_clone_index"
#: The columns the stage rewrites: the benefit-unit childcare draws, then the
#: children's under-2 usage.
CHILDCARE_COPY_TAKE_UP_REWRITES = (
    *UK_CHILDCARE_TAKE_UP_OUTPUTS,
    UK_UNDER_TWO_USAGE_OUTPUT,
)


@dataclass(frozen=True)
class UKChildcareCopyTakeUpStageTransform:
    """Whole-stage callable re-drawing the copies' childcare take-up."""

    contract: UKTakeUpContract
    stage: SourceStageSpec
    last_receipt: Mapping[str, object] | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        assert_universal_childcare_nesting_declaration(self.stage)
        assert_under_two_usage_declaration(self.stage)
        declared = tuple(self.stage.rewrites)
        if declared != CHILDCARE_COPY_TAKE_UP_REWRITES:
            raise ValueError(
                f"stage {self.stage.stage!r} must declare the rewrites "
                f"{list(CHILDCARE_COPY_TAKE_UP_REWRITES)}; declared {list(declared)}"
            )
        result = redraw_childcare_copy_take_up(frame, contract=self.contract)
        # The flags this stage re-draws, at its weights after the re-draw, for
        # the stage-time take-up signal gate (María's ruling of 2026-10-29,
        # microcosm#1095).
        object.__setattr__(
            self,
            "last_receipt",
            uk_take_up_signal_receipt(
                result,
                outputs=take_up_signal_outputs(CHILDCARE_COPY_TAKE_UP_REWRITES),
                contract=self.contract,
            ),
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_receipt is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {
            "evidence": {
                "stage": CHILDCARE_COPY_TAKE_UP_STAGE_NAME,
                "take_up_signal": dict(self.last_receipt),
            }
        }


def copy_benunits(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> np.ndarray:
    """True for every benefit unit that is a copy of a spine benefit unit."""

    _, paths = uk_household_clone_paths(household)
    copy_households = household.loc[
        np.array([bool(path) for path in without_geographic_branch(paths)], dtype=bool),
        "household_id",
    ]
    household_of = _benunit_household_map(person)
    unit_household = benunit["benunit_id"].map(household_of)
    if unit_household.isna().any():
        raise ValueError(
            "every benefit unit needs a person placing it in a household; the "
            "childcare copy re-draw cannot tell a copy from a native unit otherwise"
        )
    copies = unit_household.isin(set(copy_households)).to_numpy(dtype=bool, copy=True)
    if BENUNIT_SUPPORT_CLONE_INDEX in benunit.columns:
        clone_index = pd.to_numeric(
            benunit[BENUNIT_SUPPORT_CLONE_INDEX], errors="raise"
        ).to_numpy()
        copies |= clone_index > 0
    return copies


def redraw_childcare_copy_take_up(frame: Frame, *, contract: UKTakeUpContract) -> Frame:
    """Re-draw the copies' childcare take-up; native units keep their draws."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    missing = [
        column for column in UK_CHILDCARE_TAKE_UP_OUTPUTS if column not in benunit
    ]
    if missing:
        raise KeyError(
            f"benunit columns {missing} are missing; frs_take_up must draw them "
            "before the copies' childcare take-up is re-drawn"
        )
    if UK_UNDER_TWO_USAGE_OUTPUT not in person:
        raise KeyError(
            f"person.{UK_UNDER_TWO_USAGE_OUTPUT} is missing; frs_take_up must draw "
            "it before the copies' childcare take-up is re-drawn"
        )
    copies = copy_benunits(person, benunit, household)
    ids = benunit["benunit_id"].to_numpy()
    if pd.Series(ids).duplicated().any():
        raise ValueError(
            "benunit_id must be unique: the copies' childcare draws are keyed on it"
        )
    if copies.any():
        redrawn = derive_childcare_take_up(ids[copies], contract=contract)
        for output, _ in UK_CHILDCARE_TAKE_UP_FLAGS:
            values = benunit[output].to_numpy(dtype=bool, copy=True)
            values[copies] = redrawn[output]
            benunit[output] = values
        hours = benunit[UK_CHILDCARE_HOURS_OUTPUT].to_numpy(dtype=np.float64, copy=True)
        hours[copies] = redrawn[UK_CHILDCARE_HOURS_OUTPUT]
        benunit[UK_CHILDCARE_HOURS_OUTPUT] = hours
        # The copies' children draw their usage against their own unit's
        # re-drawn flag, keyed on their own person_id; natives keep theirs.
        copy_people = person_unit_flag(person, benunit, copies)
        if pd.Series(person["person_id"]).duplicated().any():
            raise ValueError(
                "person_id must be unique: the copies' under-2 usage draws are "
                "keyed on it"
            )
        usage = person[UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(dtype=np.float64, copy=True)
        usage[copy_people] = derive_under_two_childcare_usage(
            person.loc[copy_people],
            claiming=person_unit_flag(
                person.loc[copy_people],
                benunit,
                benunit[UK_EXTENDED_CHILDCARE_OUTPUT].to_numpy(),
            ),
            contract=contract,
        )
        person[UK_UNDER_TWO_USAGE_OUTPUT] = usage
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result
