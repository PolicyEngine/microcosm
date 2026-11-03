"""Store the donor's retired engine inputs under the engine's live inputs.

The ACS local-area chain inherits its donor's person table. The donor named by
the current staging, the reported-receipt child
``populace-us-2024-spm-receipts-20260923``, stores two columns under engine
input names that policyengine-us 2.2.1 no longer defines:

- ``would_claim_wic``, the WIC take-up draw. policyengine-us 1.777.0 renamed
  the input ``takes_up_wic_if_eligible`` (Person, MONTH, Boolean, default
  ``True``).
- ``medicare_part_b_premiums``, the ASEC ``PEMCPREM`` transfer target. Its
  successor name ``medicare_part_b_premiums_reported`` is read by no engine
  version from 1.690.7 to 2.15.1, and microcosm#590 dropped the transfer.

The stored-inputs contract (:mod:`microcosm.data.stored_inputs`, decision d271)
refuses an artifact that stores either column, because the engine ignores it
by name: every WIC-eligible person would take WIC up, whatever the draw says.
Its register (``US_STORED_NON_VARIABLE_COLUMNS``) leaves both out on purpose
and names the remedy, which :func:`with_live_stored_inputs` applies to the
consumer artifact (microcosm#1263):

- ``would_claim_wic`` becomes ``takes_up_wic_if_eligible``. Every stored
  decision is kept as it is, and a person without one takes WIC up when
  eligible (``True``), the engine's own default. The staging builder generates
  no decision for ACS people (microcosm#1154; the generator is
  microcosm#1156), and the default is the assumption policyengine.py applies
  to exactly those cells today (``legacy_inputs._temporary_acs_wic_values``).
  The caller says which people may lack a decision (the release tool passes
  the ACS spine); a missing decision on anyone else is refused, never filled.
  The rule itself reads no provenance column.
- ``medicare_part_b_premiums`` is no longer stored.

Every count recorded is an aggregate; no row value is.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import Frame

#: The WIC take-up draw: the retired engine input and the live one.
RETIRED_WIC_INPUT = "would_claim_wic"
LIVE_WIC_INPUT = "takes_up_wic_if_eligible"
#: The retired Part B premium input, which the artifact stops storing.
RETIRED_PART_B_INPUT = "medicare_part_b_premiums"

#: The interim WIC rule, quoted in the provenance and the release limitation.
ACS_WIC_ASSUMPTION = (
    "An ACS person without a stored WIC take-up decision takes WIC up when "
    "eligible: takes_up_wic_if_eligible = True, the engine's own default and "
    "the assumption policyengine.py applies to the same cells today."
)
#: When the interim WIC rule ends.
ACS_WIC_FLIP_CONDITION = (
    "Drop the assumption in the first refresh whose staging generates ACS WIC "
    "participation (microcosm#1154, generator microcosm#1156)."
)


@dataclass(frozen=True)
class LiveStoredInputsResult:
    """The frame with live inputs, and what was changed (counts only)."""

    frame: Frame
    provenance: dict[str, Any]


def with_live_stored_inputs(
    frame: Frame,
    *,
    engine_variables: Collection[str],
    may_lack_decision: np.ndarray,
) -> LiveStoredInputsResult:
    """Return ``frame`` with the retired stored inputs moved to the live ones.

    ``engine_variables`` are the installed policyengine-us's variable names.
    The rename applies only while the engine defines the live input and not
    the retired one; anything else is refused, because then the remedy is no
    longer the one the stored-inputs register prescribes.
    ``may_lack_decision`` is a Boolean vector aligned with the person table's
    rows: the people whose missing WIC decision takes the default.

    Raises:
        ValueError: if the engine's variables do not fit the remedy, if both
            WIC names are stored, if a stored decision is not a Boolean (or
            0/1), if ``may_lack_decision`` is not aligned with the person
            table, or if a decision is missing for anyone it does not cover.
    """

    names = set(engine_variables)
    if RETIRED_WIC_INPUT in names or LIVE_WIC_INPUT not in names:
        raise ValueError(
            f"The installed policyengine-us does not fit the {RETIRED_WIC_INPUT} "
            f"-> {LIVE_WIC_INPUT} remedy: it must define {LIVE_WIC_INPUT!r} and "
            f"not {RETIRED_WIC_INPUT!r}. Review microcosm.data.stored_inputs "
            "before packaging against it."
        )
    if RETIRED_PART_B_INPUT in names:
        raise ValueError(
            f"The installed policyengine-us defines {RETIRED_PART_B_INPUT!r}, "
            "so it is no longer a retired input to stop storing. Review "
            "microcosm.data.stored_inputs before packaging against it."
        )

    person = frame.table("person")
    allowed = np.asarray(may_lack_decision)
    if allowed.dtype != bool or allowed.shape != (len(person),):
        raise ValueError(
            "may_lack_decision must be a Boolean vector with one entry per "
            f"person ({len(person)}); got dtype {allowed.dtype} and shape "
            f"{allowed.shape}."
        )
    provenance: dict[str, Any] = {
        "decision": "microcosm#1263",
        "authority": "microcosm.data.stored_inputs (d271)",
    }
    changed = person
    if RETIRED_WIC_INPUT in person.columns:
        changed, provenance["wic"] = _live_wic(changed, allowed)
    else:
        provenance["wic"] = {"renamed": False}
    if RETIRED_PART_B_INPUT in changed.columns:
        provenance["medicare_part_b_premiums"] = {
            "dropped": True,
            "stored_values": int(changed[RETIRED_PART_B_INPUT].notna().sum()),
        }
        changed = changed.drop(columns=RETIRED_PART_B_INPUT)
    else:
        provenance["medicare_part_b_premiums"] = {"dropped": False}
    if changed is person:
        return LiveStoredInputsResult(frame, provenance)
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = changed
    remedied = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return LiveStoredInputsResult(remedied, provenance)


def _live_wic(
    person: pd.DataFrame, allowed: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if LIVE_WIC_INPUT in person.columns:
        raise ValueError(
            f"The person table stores both {RETIRED_WIC_INPUT!r} and "
            f"{LIVE_WIC_INPUT!r}; refusing to choose between them."
        )
    stored = person[RETIRED_WIC_INPUT]
    missing = stored.isna().to_numpy()
    outside = int((missing & ~allowed).sum())
    if outside:
        raise ValueError(
            f"{RETIRED_WIC_INPUT!r} is missing for {outside} people the caller "
            "did not allow to lack a decision; refusing to fill them."
        )
    decisions = np.ones(len(person), dtype=bool)
    present = ~missing
    decisions[present] = _booleans(stored[present])
    renamed = person.drop(columns=RETIRED_WIC_INPUT).assign(
        **{LIVE_WIC_INPUT: decisions}
    )
    assumed = missing
    return renamed, {
        "renamed": True,
        "from": RETIRED_WIC_INPUT,
        "to": LIVE_WIC_INPUT,
        "stored_decisions_kept": int(present.sum()),
        "stored_decisions_true": int(decisions[present].sum()),
        "people_assumed_true": int(assumed.sum()),
        "rule": ACS_WIC_ASSUMPTION,
        "status": "interim",
        "flip_condition": ACS_WIC_FLIP_CONDITION,
    }


def _booleans(values: pd.Series) -> np.ndarray:
    """Stored decisions as Booleans: bools, or real numbers equal to 0 or 1."""

    out = np.empty(len(values), dtype=bool)
    for position, value in enumerate(values.to_numpy(dtype=object)):
        if isinstance(value, (bool, np.bool_)):
            out[position] = bool(value)
        elif (
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and value in (0, 1)
        ):
            out[position] = bool(value)
        else:
            raise ValueError(
                f"{RETIRED_WIC_INPUT!r} stores a value that is not a Boolean "
                f"or 0/1 ({type(value).__name__}); refusing to read it as a "
                "decision."
            )
    return out
