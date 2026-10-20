"""Who claims Universal Credit when DWP closes their legacy benefits
(uk-data#492, ported from uk-data#534; microcosm#1095).

policyengine-uk 2.128.0 (pe-uk#2119) moves a family off a legacy benefit DWP
has closed (tax credits from 2025-26) and takes its other legacy awards away;
the family then claims Universal Credit if it would claim anyway
(``would_claim_uc``) or if ``would_claim_uc_at_legacy_closure`` is true. The
engine's default, true, put every such family on Universal Credit.

This stage draws the flag at DWP's Move to Universal Credit claim rates, as
uk-data#534 does (a draft, at ``c67077a66``):

* a benefit unit with a member reporting Universal Credit, a unit already
  drawn to claim it, and a unit reporting none of the six legacy benefits are
  true, the engine's default;
* each other unit is drawn at ``(r - p) / (1 - p)``, clipped to [0, 1]. ``r``
  is DWP's claimed / (claimed + did not claim) for the unit's combination of
  legacy benefits, or the all-households rate where DWP reports none; ``p`` is
  the combination's share of Great Britain's such units already drawn to
  claim, at the spine's design weights (uk-data#534's is unweighted), and the
  probability applies to the combination's units in Northern Ireland too.
  So, in expectation, the weighted share of Great Britain's drawn units
  claiming at the closure in each combination is DWP's rate wherever ``p`` is
  at most ``r``; where ``p`` exceeds it the probability is 0, and the receipt
  flags the combination (``rate_unreachable``), never ``p`` itself.

Two limits carry over from uk-data#534: the drawn cohort includes units on
the Pension Credit route, whose legacy benefits policyengine-uk does not
close, which dilutes the rate among the units it does move; and DWP's rates
count every household sent a notice, including households not entitled to
Universal Credit.

The rates are DWP's Stat-Xplore households sent a migration notice in Great
Britain, July 2022 to June 2026, by legacy benefits and move status, vendored
from the pinned Chronicle feed (``dwp_move_to_uc_facts.json``); households
still in progress are left out. Each unit's draw reads an identity-keyed
uniform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _benunit_households,
    _household_to_benunit_weights,
)
from microcosm.frame import Frame

__all__ = [
    "LEGACY_BENEFITS",
    "LEGACY_CLOSURE_OPERATION_KIND",
    "LEGACY_CLOSURE_OUTPUT",
    "LEGACY_CLOSURE_RECORD_SETS",
    "LEGACY_CLOSURE_RESOURCE",
    "LEGACY_CLOSURE_STAGE_NAME",
    "UKLegacyClosureTakeUpResult",
    "UKLegacyClosureTakeUpStageTransform",
    "assign_uc_claim_at_legacy_closure",
    "legacy_benefit_combinations",
    "legacy_closure_claim_probabilities",
    "legacy_closure_operation_parameters",
    "load_move_to_uc_claim_rates",
    "move_to_uc_claim_rates_from_rows",
]

LEGACY_CLOSURE_STAGE_NAME = "uc_legacy_closure_take_up"
LEGACY_CLOSURE_OPERATION_KIND = "draw_uc_claim_at_legacy_closure"
LEGACY_CLOSURE_OUTPUT = "would_claim_uc_at_legacy_closure"
LEGACY_CLOSURE_RESOURCE = "dwp_move_to_uc_facts.json"
#: The build period DWP's cumulative Move to UC rates are mapped to (the verify
#: operation's ``mapped_build_period``); the stage refuses a frame at any other.
LEGACY_CLOSURE_BUILD_PERIOD = 2024
_DETAIL_RECORD_SET = "dwp.statx.mtuc_households.legacy_benefit_move_status.2026-06"
#: The cells by combination and move status, and their totals (which carry the
#: all-households row).
LEGACY_CLOSURE_RECORD_SETS: tuple[str, ...] = (
    _DETAIL_RECORD_SET,
    f"{_DETAIL_RECORD_SET}.total",
)
_CONCEPT = "dwp.mtuc_households_sent_migration_notice"
_CLAIMED = "Claimed UC"
_DID_NOT_CLAIM = "Did not claim UC - Legacy benefit closed"
_ALL = "all"
#: The legacy means-tested benefits, in the order a combination key joins them
#: (uk-data#534's order), with DWP's Stat-Xplore names.
LEGACY_BENEFITS: tuple[tuple[str, str], ...] = (
    ("child_tax_credit", "Child Tax Credit"),
    ("working_tax_credit", "Working Tax Credit"),
    ("housing_benefit", "Housing Benefit"),
    ("esa_income", "Employment Support Allowance"),
    ("income_support", "Income Support"),
    ("jsa_income", "Jobseekers Allowance"),
)
_STREAM_SEED = 0


def _combination_key(label: str) -> str:
    """DWP's label ("Child Tax Credit and Housing Benefit") as a key."""

    if label == _ALL:
        return _ALL
    text = label.removesuffix(" only")
    names = {dwp: name for name, dwp in LEGACY_BENEFITS}
    parts = text.split(" and ")
    unknown = [part for part in parts if part not in names]
    if unknown:
        raise ValueError(f"Unknown legacy benefit in Move to UC label {label!r}.")
    members = {names[part] for part in parts}
    return "+".join(name for name, _ in LEGACY_BENEFITS if name in members)


def move_to_uc_claim_rates_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Claim rate by combination: claimed / (claimed + did not claim)."""

    cells: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("concept") != _CONCEPT:
            continue
        dimensions = row.get("dimensions") or {}
        key = (
            _combination_key(str(dimensions.get("legacy_benefit"))),
            str(dimensions.get("move_status")),
        )
        if key in cells:
            raise ValueError(f"The Move to UC facts repeat {key}.")
        cells[key] = float(row["value"])
    rates: dict[str, float] = {}
    for combination in sorted({key for key, _ in cells}):
        claimed = cells.get((combination, _CLAIMED))
        did_not = cells.get((combination, _DID_NOT_CLAIM))
        if claimed is None or did_not is None or claimed + did_not <= 0:
            raise ValueError(
                f"The Move to UC facts lack the claimed and closed counts for "
                f"{combination!r}."
            )
        rates[combination] = claimed / (claimed + did_not)
    if _ALL not in rates:
        raise ValueError("The Move to UC facts carry no all-households row.")
    if not all(0.0 < rate <= 1.0 for rate in rates.values()):
        raise ValueError("Move to UC claim rates must lie in (0, 1].")
    return rates


def load_move_to_uc_claim_rates() -> dict[str, float]:
    """The rates from the committed vendored resource, checked against the pin."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return move_to_uc_claim_rates_from_rows(vendored_rows(LEGACY_CLOSURE_RESOURCE))


def _reported_by_benunit(
    person: pd.DataFrame, benunit: pd.DataFrame, column: str
) -> np.ndarray:
    if column not in person.columns:
        raise ValueError(f"The legacy closure draw needs person {column!r}.")
    values = pd.to_numeric(person[column], errors="raise").to_numpy(dtype=float)
    reporters = person.loc[values > 0, "person_benunit_id"].unique()
    return benunit["benunit_id"].isin(reporters).to_numpy()


def legacy_benefit_combinations(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    """Each benefit unit's reported legacy benefits, joined by "+"."""

    reported = [
        _reported_by_benunit(person, benunit, f"{name}_reported")
        for name, _ in LEGACY_BENEFITS
    ]
    return np.asarray(
        [
            "+".join(
                name for (name, _), has in zip(LEGACY_BENEFITS, row, strict=True) if has
            )
            for row in zip(*reported, strict=True)
        ],
        dtype=object,
    )


def legacy_closure_claim_probabilities(
    *,
    combination: np.ndarray,
    drawn: np.ndarray,
    claims_anyway: np.ndarray,
    weights: np.ndarray,
    rates: Mapping[str, float],
    great_britain: np.ndarray,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    """Each drawn unit's claim probability, and a receipt by combination.

    Within a combination, ``(r - p) / (1 - p)`` clipped to [0, 1], where ``r``
    is DWP's rate and ``p`` the weighted share of Great Britain's drawn units
    that would claim anyway (DWP's rates cover Great Britain); the probability
    applies to every unit of the combination. The expected weighted share of
    Great Britain's drawn units claiming at the closure is ``r`` wherever
    ``p <= r``. The receipt bands each cohort on its distinct source
    households and flags an unreachable rate only where ten or more carry p
    and neither of its sides rests on one to nine.
    """

    gb = np.asarray(great_britain, dtype=bool)
    probability = np.zeros(len(combination))
    receipt: dict[str, dict[str, object]] = {}

    def count(mask: np.ndarray) -> int:
        return distinct_count(source_ids, mask, weights)

    for key in sorted(set(combination[drawn])):
        cohort = drawn & (combination == key)
        basis = cohort & gb
        rate = float(rates.get(key, rates[_ALL]))
        basis_weight = float(weights[basis].sum())
        p = (
            float(weights[basis & claims_anyway].sum()) / basis_weight
            if basis_weight > 0
            else 0.0
        )
        probability[cohort] = (
            0.0 if p >= 1 else float(np.clip((rate - p) / (1 - p), 0.0, 1.0))
        )

        # The flag rests on p, so on Great Britain's sources, and says on which
        # side of DWP's rate p lies: it is given only where ten or more sources
        # carry p and neither the units claiming anyway nor the others rest on
        # one to nine. A combination with no Great Britain unit has no p.
        sides = (count(basis & claims_anyway), count(basis & ~claims_anyway))
        if not basis.any():
            flag: bool | str | None = None
        elif count(basis) < 10 or 0 < min(sides) < 10:
            flag = "<10"
        else:
            flag = bool(p > rate)
        receipt[str(key)] = {
            "rate": rate,
            "rate_source": "combination" if key in rates else "all_households",
            "rate_unreachable": flag,
            "support": support_band(count(cohort)),
        }
    return probability, receipt


@dataclass(frozen=True)
class UKLegacyClosureTakeUpResult:
    """The frame with the drawn flag, and the stage's aggregate receipt."""

    frame: Frame
    receipt: dict[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": LEGACY_CLOSURE_STAGE_NAME, **self.receipt}


def assign_uc_claim_at_legacy_closure(
    frame: Frame, *, rates: Mapping[str, float]
) -> UKLegacyClosureTakeUpResult:
    """Draw ``would_claim_uc_at_legacy_closure`` for every benefit unit."""

    validate_uk_national_frame(frame)
    # Compared as the label the frame carries, so that a "2024-25" or a later
    # year is refused by this message, not by a failed int().
    period = uk_time_period(frame)
    if period != str(LEGACY_CLOSURE_BUILD_PERIOD):
        raise ValueError(
            "DWP's Move to UC claim rates are mapped to build period "
            f"{LEGACY_CLOSURE_BUILD_PERIOD}, but the frame's period is {period!r}."
        )
    person = frame.table("person")
    benunit = frame.table("benunit").copy()
    household = frame.table("household")
    if LEGACY_CLOSURE_OUTPUT in benunit.columns:
        raise ValueError(
            f"{LEGACY_CLOSURE_OUTPUT!r} is already stored: this stage is its only "
            "writer."
        )
    if "would_claim_uc" not in benunit.columns:
        raise ValueError("The legacy closure draw needs benunit 'would_claim_uc'.")
    # The graph node declares the lineage, since the executor hands a stage
    # only its declared cells.
    household_sources = pd.Series(
        source_household_ids(household, stage="The legacy closure stage"),
        index=household["household_id"].to_numpy(),
    )
    household_weights = frame.weights_for("household")
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=np.asarray(household_weights.values, dtype=float),
    )
    combination = legacy_benefit_combinations(person, benunit)
    on_uc = _reported_by_benunit(person, benunit, "universal_credit_reported")
    claims_anyway = benunit["would_claim_uc"].to_numpy().astype(bool)
    drawn = ~on_uc & (combination != "")
    unit_households = pd.Series(_benunit_households(benunit, person=person))
    by_household = household.set_index("household_id")
    great_britain = (
        unit_households.map(by_household["region"].astype(str)).to_numpy()
        != "NORTHERN_IRELAND"
    )
    source_ids = unit_households.map(household_sources).to_numpy()
    probability, combinations_drawn = legacy_closure_claim_probabilities(
        combination=combination,
        drawn=drawn,
        claims_anyway=claims_anyway,
        weights=weights,
        rates=rates,
        great_britain=great_britain,
        source_ids=source_ids,
    )
    uniforms = stable_identity_uniforms(
        benunit["benunit_id"].to_numpy(), seed=_STREAM_SEED, salt=LEGACY_CLOSURE_OUTPUT
    )
    flag = ~drawn | claims_anyway | (uniforms < probability)
    benunit[LEGACY_CLOSURE_OUTPUT] = flag
    result = uk_national_frame(
        person=person.copy(),
        benunit=benunit,
        household=household.copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)

    # Great Britain's weighted aggregates only, DWP's universe; a figure one to
    # nine source households carry is "<10" (UKDS disclosure control).
    def sources(mask: np.ndarray) -> int:
        return distinct_count(source_ids, mask, weights)

    gb_drawn = drawn & great_britain
    gb_sources = sources(gb_drawn)
    drawn_weight = float(weights[gb_drawn].sum())
    claiming = gb_drawn & flag
    # The share claiming times the drawn weight gives back either side, so the
    # share is given only where neither side rests on one to nine sources.
    share_small = 0 < min(sources(claiming), sources(gb_drawn & ~flag)) < 10
    if not drawn_weight:
        share: float | str | None = None
    elif share_small:
        share = "<10"
    else:
        share = small_cell_share(
            float(weights[claiming].sum()) / drawn_weight, gb_sources
        )
    receipt = {
        "combinations": combinations_drawn,
        "great_britain_drawn_benunit_weight": small_cell_share(
            drawn_weight, gb_sources
        ),
        "great_britain_weighted_share_claiming_at_closure": share,
        "p_basis": "spine_design_weights_great_britain",
    }
    return UKLegacyClosureTakeUpResult(frame=result, receipt=receipt)


def legacy_closure_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "move_to_uc_facts",
                "resource": LEGACY_CLOSURE_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(LEGACY_CLOSURE_RECORD_SETS),
                "source_vintage": "July 2022 to June 2026",
                "mapped_build_period": LEGACY_CLOSURE_BUILD_PERIOD,
                "period_mapping": "cumulative_claim_rates_applied_at_every_closure",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            LEGACY_CLOSURE_OPERATION_KIND,
            {
                "output": LEGACY_CLOSURE_OUTPUT,
                "port_of": "uk-data#534 at c67077a66",
                "legacy_benefits": [name for name, _ in LEGACY_BENEFITS],
                "true_without_draw": (
                    "a member reports universal_credit_reported, would_claim_uc is "
                    "true, or no member reports a legacy benefit"
                ),
                "rate": (
                    "claimed / (claimed + did not claim) for the combination, the "
                    "all-households rate where DWP reports none"
                ),
                "probability": "clip((rate - p) / (1 - p), 0, 1)",
                "p": (
                    "the combination's weighted share of Great Britain's drawn "
                    "units with would_claim_uc true, at the spine's design "
                    "weights; the probability applies in Northern Ireland too"
                ),
                "stream": {"salt": LEGACY_CLOSURE_OUTPUT, "seed": _STREAM_SEED},
            },
        ),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    _assert_closed_world_operations(stage, legacy_closure_operation_parameters())
    if stage.stage != LEGACY_CLOSURE_STAGE_NAME:
        raise ValueError(
            f"Expected stage {LEGACY_CLOSURE_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if tuple(stage.outputs) != (LEGACY_CLOSURE_OUTPUT,) or tuple(stage.rewrites):
        raise ValueError(
            f"The legacy closure stage writes {LEGACY_CLOSURE_OUTPUT!r} only."
        )
    if stage.grain != "benunit":
        raise ValueError(
            f"The legacy closure stage is benunit grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKLegacyClosureTakeUpStageTransform:
    """Whole-stage callable for the Universal Credit claims at the closure."""

    stage: SourceStageSpec
    #: The claim rates; read from the vendored DWP facts when not given.
    rates: Mapping[str, float] | None = field(default=None, repr=False)
    last_result: UKLegacyClosureTakeUpResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        rates = self.rates if self.rates is not None else load_move_to_uc_claim_rates()
        result = assign_uc_claim_at_legacy_closure(frame, rates=rates)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (LEGACY_CLOSURE_OUTPUT,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
