"""ASEC-measured adult/disabled-dependent care inputs for the section 21 CDCC.

PolicyEngine-US 1.819.0 computes the CDCC adult-care leg from two person
inputs neither the retired eCPS nor any prior Microcosm base carried:
``is_incapable_of_self_care`` (the section 21(b)(1)(B)/(C) qualifying test
and the 21(d)(2) spouse deemed-earnings gate) and
``pre_subsidy_care_expenses`` (the dollar leg summed into
``cdcc_relevant_expenses``).  Both were structural zeros: any CDCC reform
binding through adult care scored exactly $0 (PolicyEngine/microcosm#451
item 1, the #368 absent-input class).

Source decision, with the survey dictionaries read as receipts
(Census API variable dictionaries, ASEC 2024 and SIPP 2023):

* ASEC carries the qualifying concept directly: person item ``PEDISDRS``
  ("difficulty dressing or bathing", the self-care difficulty item) is the
  measured operationalization of "physically or mentally incapable of
  self-care", and is already ingested by the eligibility stage.  SIPP's
  ``ESELFCARE`` is the identical instrument item, corroborating the concept
  without adding a donor artifact.
* Neither instrument's current public-use release carries in-household
  adult-care expenditures.  ASEC's only care-expense items are childcare
  (``HCHCARE_VAL``, ``SPM_CHILDCAREXPNS``).  The 2018 SIPP panel's Child &
  Dependent Care topical module did collect paid dependent-care fields
  (``ECREPAYANYON``, ``TDEPNDNTEXP``), but they are absent from the 2023
  SIPP PUF dictionary — the pinned donor vintage — whose only adult-care
  payment amount (``TDPCAREAMT``) covers care of a *former* household
  member, failing section 21's same-principal-abode requirement.

The dollar leg is therefore an explicitly documented same-instrument proxy:
the measured ASEC work-related childcare expense distribution — the same
section 21 employment-related expense class, subject to the same per-person
statutory cap — supplies both the paid-care usage rate and the positive
expense level distribution.  The donor universe is measured at SPM-unit
grain (where the childcare leaf lives): units with any member under the
qualifying age and any positive member earnings, on the measured ASEC
channel — an approximation of the binding childcare population at donor
grain, not a reproduction of the tax-unit minimum-earnings test.  Support
is restricted to tax units where the statute can bind: a measured disabled
qualifying individual (dependent, or married head/spouse) plus the section
21(d) earnings structure computed exactly as the engine's
``min_head_spouse_earned`` binds — both spouses earning, or a 21(d)(2)
floor-eligible spouse (incapable of self-care, or the measured full-time
college student) deemed while the OTHER spouse actually earns.  Assignment
is a seeded, weight-targeted draw over sorted unit ids (invariant to
person-row order) that preserves the SCREENED level distribution — donor
values above the expense plausibility ceiling are refused as interpolation
knots with a logged receipt, while the measured frame values stay
untouched — with the level grid taken from the selected units' own
cumulative weights; no level or usage number is invented outside the
screened measured donor distribution.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PERSON_SUPPORT_CHANNEL_COLUMN,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT",
    "US_ADULT_CARE_EARNED_INCOME_SOURCES",
    "US_ADULT_CARE_OUTPUT_COLUMNS",
    "US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS",
    "US_ADULT_CARE_STAGE_NAME",
    "derive_us_adult_care_from_manifest",
    "us_adult_care_signal_gate",
    "us_adult_care_stage_spec",
    "us_adult_care_summary",
    "with_us_adult_care_inputs",
]

US_ADULT_CARE_STAGE_NAME = "adult_care_inputs"
US_ADULT_CARE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_incapable_of_self_care",
    "pre_subsidy_care_expenses",
)
# The measured ASEC self-care difficulty item, the finished childcare donor
# surface (SPM-unit grain, attached per person), the measured full-time
# college indicator for 21(d)(2) student deeming, and the clone-channel
# provenance column the donor statistics are certified against.
US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "PEDISDRS",
    "spm_unit_pre_subsidy_childcare_expenses",
    "is_full_time_college_student",
    PERSON_SUPPORT_CHANNEL_COLUMN,
)
# 26 USC 21(b)(1)(A): a dependent under this age qualifies by age (the
# engine's gov.irs.credits.cdcc.eligibility.child_age parameter carries the
# same value 13 under a 21(d)(1)(A) citation). The measured childcare-usage
# donor universe conditions on it.
US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT = 13
US_ADULT_CARE_EARNED_INCOME_SOURCES: tuple[str, ...] = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
)

_FLAG_OUTPUT, _EXPENSE_OUTPUT = US_ADULT_CARE_OUTPUT_COLUMNS
_SELF_CARE_SOURCE = "PEDISDRS"
_CHILDCARE_SOURCE = "spm_unit_pre_subsidy_childcare_expenses"
_STUDENT_SOURCE = "is_full_time_college_student"
_PERSON_WEIGHT_COLUMN = "person_weight"
_TAX_UNIT_WEIGHT_COLUMN = "adult_care_tax_unit_weight"
_SPM_UNIT_WEIGHT_COLUMN = "adult_care_spm_unit_weight"
_ROLE_COLUMN = "tax_unit_role_input"
_AGE_COLUMN = "age"
_FLAG_SHARE_BAND = (0.002, 0.12)
# Sanity ceiling for a single unit's annual employment-related care expense.
# Its job is refusing corrupted magnitudes wherever they appear: measured
# ASEC childcare itself carries two implausible values ($730,000 and
# $360,000) that are screened out of the donor pool below, and healed
# surfaces can carry overflow artifacts.
_EXPENSE_PLAUSIBILITY_CEILING = 250_000.0


def us_adult_care_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged adult-care stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_ADULT_CARE_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_ADULT_CARE_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_ADULT_CARE_STAGE_NAME]
    if tuple(spec.outputs) != US_ADULT_CARE_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_ADULT_CARE_STAGE_NAME!r} outputs must be "
            f"{list(US_ADULT_CARE_OUTPUT_COLUMNS)}; got {list(spec.outputs)}."
        )
    return spec


def _finite(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US adult-care derivation requires source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US adult-care source {column!r} contains {nonfinite} nonnumeric "
            "or nonfinite value(s)."
        )
    return values


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Interpolated weighted quantiles of a positive donor distribution."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    total = float(sorted_weights.sum())
    if total <= 0.0:
        raise SourceRuntimeError("US adult-care donor weights sum to zero.")
    positions = cumulative / total
    return np.interp(probabilities, positions, sorted_values)


def _role(person: pd.DataFrame) -> pd.Series:
    if _ROLE_COLUMN not in person.columns:
        raise SourceRuntimeError(
            f"US adult-care derivation requires {_ROLE_COLUMN!r} to identify "
            "heads, spouses, and dependents."
        )
    return person[_ROLE_COLUMN].map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )


def _screen_implausible_donors(
    donor_childcare: np.ndarray,
    donor_weight: np.ndarray,
    level_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refuse donor knots above the plausibility ceiling, with a receipt.

    The ceiling's charter is refusing corrupted magnitudes, and a corrupt
    magnitude is corrupt at the SOURCE: two measured ASEC units ($730,000
    and $360,000 childcare) sat latent in every build's donor pool, and
    whether a recipient's quantile position reached those top knots was
    grid luck — Build P3's tail clones shifted the grid and 12 draws
    entered the >$250k top-tail interpolation region (three between the
    ceiling and $360k, nine at or above $360k; microcosm#567). The measured
    values stay in the frame untouched; they are only refused as
    imputation donors. The receipt is a deterministic recomputation-stable
    build-log record, not a persisted artifact.
    """

    implausible = level_mask & (donor_childcare > _EXPENSE_PLAUSIBILITY_CEILING)
    clean_mask = level_mask & ~implausible
    level_weight = float(donor_weight[level_mask].sum())
    retained_values = donor_childcare[clean_mask]
    receipt = {
        "count": int(np.count_nonzero(implausible)),
        "values": sorted(float(value) for value in donor_childcare[implausible]),
        "weight": float(donor_weight[implausible].sum()),
        "excluded_weight_share": (
            float(donor_weight[implausible].sum()) / level_weight
            if level_weight > 0.0
            else 0.0
        ),
        "retained_maximum": (
            float(retained_values.max()) if retained_values.size else 0.0
        ),
        "ceiling": _EXPENSE_PLAUSIBILITY_CEILING,
    }
    return clean_mask, receipt


def derive_us_adult_care_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Derive the qualifying flag and the donor-matched expense leg."""

    if operation.kind != "derive_adult_care_inputs":
        raise SourceRuntimeError(
            f"US adult-care derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US adult-care derivation requires the person table to be read first."
        )
    expected_parameters = {
        "self_care_difficulty_source": _SELF_CARE_SOURCE,
        "childcare_expense_source": _CHILDCARE_SOURCE,
        "full_time_student_source": _STUDENT_SOURCE,
        "support_channel_source": PERSON_SUPPORT_CHANNEL_COLUMN,
        "child_qualifying_age_limit": US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
        "earned_income_sources": list(US_ADULT_CARE_EARNED_INCOME_SOURCES),
        "seed_from_build_config": True,
        "output_flag": _FLAG_OUTPUT,
        "output_expenses": _EXPENSE_OUTPUT,
    }
    parameters = dict(operation.parameters)
    if parameters != expected_parameters:
        raise SourceRuntimeError(
            "US adult-care derivation drifted from the pinned method: "
            f"expected {expected_parameters}, got {parameters}."
        )

    person = frame
    # 21(b)(1): the measured ASEC self-care difficulty item is the qualifying
    # flag. ASEC codes 1 = yes; every other in/out-of-universe code is False.
    self_care = _finite(person, _SELF_CARE_SOURCE)
    flag = self_care == 1.0
    # 21(d)(2) also deems earnings for a full-time-student spouse; the
    # measured college indicator (eligibility stage) is what reaches the
    # engine's is_full_time_student aggregation.
    if _STUDENT_SOURCE not in person.columns:
        raise SourceRuntimeError(
            f"US adult-care derivation requires source column {_STUDENT_SOURCE!r}."
        )
    student = _finite(person, _STUDENT_SOURCE) == 1.0

    age = _finite(person, _AGE_COLUMN)
    earned = np.zeros(len(person), dtype=np.float64)
    for column in US_ADULT_CARE_EARNED_INCOME_SOURCES:
        earned += _finite(person, column)
    childcare = _finite(person, _CHILDCARE_SOURCE)
    negative_childcare = int(np.count_nonzero(childcare < 0.0))
    if negative_childcare:
        raise SourceRuntimeError(
            f"US adult-care donor {_CHILDCARE_SOURCE!r} contains "
            f"{negative_childcare} negative value(s)."
        )
    if not has_support_role_metadata(person, entity="person"):
        raise SourceRuntimeError(
            "US adult-care derivation requires support-role provenance; "
            "without it the "
            "measured-ASEC donor statistics cannot be certified."
        )
    support_role = support_role_series(person, entity="person")

    role = _role(person)
    is_head = (role == "HEAD").to_numpy()
    is_spouse = (role == "SPOUSE").to_numpy()
    is_dependent = (role == "DEPENDENT").to_numpy()
    unclassified = ~(is_head | is_spouse | is_dependent)
    if bool(unclassified.any()):
        raise SourceRuntimeError(
            "US adult-care derivation found "
            f"{int(np.count_nonzero(unclassified))} person(s) with an "
            "unrecognized tax-unit role."
        )

    if "person_tax_unit_id" not in person.columns:
        raise SourceRuntimeError(
            "US adult-care derivation requires person_tax_unit_id."
        )
    if _TAX_UNIT_WEIGHT_COLUMN not in person.columns:
        raise SourceRuntimeError(
            "US adult-care derivation requires the attached tax-unit weight."
        )
    if _SPM_UNIT_WEIGHT_COLUMN not in person.columns:
        raise SourceRuntimeError(
            "US adult-care derivation requires the attached SPM-unit weight."
        )

    unit_frame = pd.DataFrame(
        {
            "unit": person["person_tax_unit_id"].to_numpy(),
            "flag_head_or_spouse": flag & (is_head | is_spouse),
            "flag_dependent": flag & is_dependent,
            "spouse": is_spouse,
            "child": age < float(US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT),
            "earned": earned,
            "head_earned": np.where(is_head, earned, 0.0),
            "spouse_earned": np.where(is_spouse, earned, 0.0),
            "head_floor_eligible": is_head & (flag | student),
            "spouse_floor_eligible": is_spouse & (flag | student),
            "weight": pd.to_numeric(
                person[_TAX_UNIT_WEIGHT_COLUMN], errors="coerce"
            ).to_numpy(dtype=np.float64),
            "childcare": childcare,
            "spm_weight": pd.to_numeric(
                person[_SPM_UNIT_WEIGHT_COLUMN], errors="coerce"
            ).to_numpy(dtype=np.float64),
            "spm_unit": (
                person["person_spm_unit_id"].to_numpy()
                if "person_spm_unit_id" in person.columns
                else person["person_tax_unit_id"].to_numpy()
            ),
            "asec": (support_role == BASE_ASEC_SUPPORT_CHANNEL).to_numpy(),
        }
    )
    units = unit_frame.groupby("unit", sort=False).agg(
        married=("spouse", "any"),
        flag_head_or_spouse=("flag_head_or_spouse", "any"),
        dependent_prong=("flag_dependent", "any"),
        head_earned=("head_earned", "sum"),
        spouse_earned=("spouse_earned", "sum"),
        head_floor_eligible=("head_floor_eligible", "any"),
        spouse_floor_eligible=("spouse_floor_eligible", "any"),
        weight=("weight", "first"),
    )
    # 21(b)(1)(C) limits the spouse prong to a married head or spouse;
    # 21(b)(1)(B) covers any disabled dependent.
    units["spouse_prong"] = units["flag_head_or_spouse"] & units["married"]

    # 21(d) exactly as the engine computes min_head_spouse_earned: a married
    # unit binds when both spouses earn, or when a 21(d)(2) floor-eligible
    # spouse (incapable of self-care, or the measured full-time student) is
    # deemed while the OTHER spouse actually earns. Deeming never rescues a
    # unit whose only earner is the floor-eligible spouse. A single filer
    # needs the head's earnings.
    head_earned = units["head_earned"].to_numpy()
    spouse_earned = units["spouse_earned"].to_numpy()
    head_floor = units["head_floor_eligible"].to_numpy()
    spouse_floor = units["spouse_floor_eligible"].to_numpy()
    married = units["married"].to_numpy()
    both_earn = (head_earned > 0.0) & (spouse_earned > 0.0)
    deemed = (head_floor & (spouse_earned > 0.0)) | (spouse_floor & (head_earned > 0.0))
    work_test = np.where(married, both_earn | deemed, head_earned > 0.0)
    eligible = (
        units["spouse_prong"].to_numpy() | units["dependent_prong"].to_numpy()
    ) & work_test

    # Measured donor statistics on the measured ASEC channel. The donor
    # universe is SPM-unit grain (the grain the childcare leaf lives at):
    # units with any member under the qualifying age and any positive member
    # earnings. This approximates, at donor grain, the population whose
    # childcare leg of the same section 21 expense class can bind; it does
    # not reproduce the tax-unit head/spouse minimum-earnings test.
    donor_units = (
        unit_frame[unit_frame["asec"]]
        .groupby("spm_unit", sort=False)
        .agg(
            any_child=("child", "any"),
            earned=("earned", "sum"),
            childcare=("childcare", "first"),
            weight=("spm_weight", "first"),
        )
    )
    donor_universe = donor_units[
        donor_units["any_child"] & (donor_units["earned"] > 0.0)
    ]
    if donor_universe.empty:
        raise SourceRuntimeError(
            "US adult-care derivation found no measured childcare-class donor "
            "units; the usage rate and level distribution are undefined."
        )
    donor_weight = donor_universe["weight"].to_numpy(dtype=np.float64)
    donor_childcare = donor_universe["childcare"].to_numpy(dtype=np.float64)
    total_donor_weight = float(donor_weight.sum())
    if total_donor_weight <= 0.0:
        raise SourceRuntimeError("US adult-care donor universe has zero weight.")
    donor_positive = donor_childcare > 0.0
    usage_rate = float(donor_weight[donor_positive].sum()) / total_donor_weight
    if not 0.0 < usage_rate < 1.0:
        raise SourceRuntimeError(
            "US adult-care measured paid-care usage rate "
            f"{usage_rate:.4f} is degenerate."
        )
    # Zero-weight donors carry no measured mass and must not become
    # interpolation knots.
    level_mask = donor_positive & (donor_weight > 0.0)
    # Implausible donor knots are refused with a logged receipt (see
    # _screen_implausible_donors).
    level_mask, implausible_donor_receipt = _screen_implausible_donors(
        donor_childcare,
        donor_weight,
        level_mask,
    )
    if implausible_donor_receipt["count"]:
        print(
            "US adult-care donor screen refused "
            f"{implausible_donor_receipt['count']} implausible donor knot(s) "
            f"above ${_EXPENSE_PLAUSIBILITY_CEILING:,.0f}: "
            f"{implausible_donor_receipt['values']} "
            f"(weight {implausible_donor_receipt['weight']:,.2f}, "
            f"{implausible_donor_receipt['excluded_weight_share']:.4%} of "
            "level weight; retained maximum "
            f"${implausible_donor_receipt['retained_maximum']:,.0f}); the "
            "measured frame values are untouched."
        )
    level_values = donor_childcare[level_mask]
    level_weights = donor_weight[level_mask]
    if not level_values.size:
        raise SourceRuntimeError(
            "US adult-care derivation found no positively weighted paid-care donors."
        )

    # Seeded, weight-targeted selection over the eligible units in sorted
    # unit-id order (invariant to person-row order), taking the permuted
    # prefix whose weight reaches the measured usage rate; the level draw is
    # a weighted-quantile map whose grid is the selected units' own
    # cumulative-weight midpoints, so the recipient-weighted distribution
    # reproduces the donor-weighted distribution.
    expenses = np.zeros(len(person), dtype=np.float64)
    eligible_ids = units.index.to_numpy()[eligible]
    if eligible_ids.size:
        eligible_weights = units["weight"].to_numpy(dtype=np.float64)[eligible]
        id_order = np.argsort(eligible_ids, kind="stable")
        eligible_ids = eligible_ids[id_order]
        eligible_weights = eligible_weights[id_order]
        seed = context.config.seed if context is not None else 0
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(eligible_ids.size)
        cumulative = np.cumsum(eligible_weights[order])
        target = usage_rate * float(eligible_weights.sum())
        take = int(np.searchsorted(cumulative, target, side="left") + 1)
        take = min(take, eligible_ids.size)
        selected_ids = eligible_ids[order[:take]]
        selected_weights = eligible_weights[order[:take]]

        pair_order = np.argsort(selected_ids, kind="stable")
        paired_ids = selected_ids[pair_order]
        paired_weights = selected_weights[pair_order]
        cumulative_selected = (
            np.cumsum(paired_weights) - 0.5 * paired_weights
        ) / float(paired_weights.sum())
        draws = _weighted_quantile(level_values, level_weights, cumulative_selected)
        by_unit = dict(zip(paired_ids.tolist(), draws.tolist(), strict=True))

        person_units = person["person_tax_unit_id"].to_numpy()
        person_ids = pd.to_numeric(person["person_id"], errors="coerce").to_numpy()
        person_married = (
            pd.Series(person_units).map(units["married"]).fillna(False).to_numpy()
        )
        qualifying_person = flag & (
            is_dependent | ((is_head | is_spouse) & person_married)
        )
        # The carrier is the qualifying person with the smallest person_id in
        # the unit — invariant to person-row order.
        carrier_ids = (
            pd.Series(np.where(qualifying_person, person_ids, np.inf))
            .groupby(person_units)
            .transform("min")
            .to_numpy()
        )
        is_carrier = qualifying_person & (person_ids == carrier_ids)
        mapped = pd.Series(person_units).map(by_unit)
        place = is_carrier & mapped.notna().to_numpy()
        expenses[place] = mapped[place].to_numpy(dtype=np.float64)

    result = person.copy(deep=True)
    result[_FLAG_OUTPUT] = flag
    result[_EXPENSE_OUTPUT] = expenses
    return result


def with_us_adult_care_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    allow_existing_without_source: bool = False,
) -> Frame:
    """Materialize the CDCC adult-care flag and expense inputs.

    ``allow_existing_without_source`` is only for a downstream release build
    consuming a base artifact that already passed this stage's signal gate.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US adult-care inputs require the US schema.")
    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    source_available = all(
        column in person or column in spm_unit
        for column in US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if allow_existing_without_source and _adult_care_surface_carries_signal(frame):
            return frame
        missing = [
            column
            for column in US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS
            if column not in person and column not in spm_unit
        ]
        raise ValueError(
            "US adult-care stage cannot heal a default surface without "
            f"measured source column(s): {missing}."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values

    if _CHILDCARE_SOURCE not in stage_person.columns:
        childcare_by_unit = pd.Series(
            pd.to_numeric(spm_unit[_CHILDCARE_SOURCE], errors="coerce").to_numpy(
                dtype=np.float64
            ),
            index=spm_unit["spm_unit_id"].to_numpy(),
        )
        stage_person[_CHILDCARE_SOURCE] = (
            person["person_spm_unit_id"].map(childcare_by_unit).to_numpy()
        )

    tax_unit = frame.table("tax_unit")
    tax_unit_weights = pd.Series(
        np.asarray(frame.resolve_weights("tax_unit").values, dtype=np.float64),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    stage_person[_TAX_UNIT_WEIGHT_COLUMN] = (
        person["person_tax_unit_id"].map(tax_unit_weights).to_numpy()
    )
    spm_weights = pd.Series(
        np.asarray(frame.resolve_weights("spm_unit").values, dtype=np.float64),
        index=spm_unit["spm_unit_id"].to_numpy(),
    )
    stage_person[_SPM_UNIT_WEIGHT_COLUMN] = (
        person["person_spm_unit_id"].map(spm_weights).to_numpy()
    )

    output = run_source_stage(
        us_adult_care_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_adult_care_inputs": derive_us_adult_care_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_ADULT_CARE_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US adult-care stage output {column!r} does not cover every person."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_FLAG_OUTPUT] = aligned[_FLAG_OUTPUT].astype(bool).to_numpy()
    tables["person"][_EXPENSE_OUTPUT] = aligned[_EXPENSE_OUTPUT].to_numpy(
        dtype=np.float64
    )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_adult_care_summary(frame: Frame) -> dict[str, object]:
    """Weighted signal, validity, and statute-structure diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    # A malformed flag column (nulls, non-{0,1} values, or a string/object
    # dtype) must surface as invalid rather than be coerced: .astype(bool)
    # reads NaN, 2, and "0" as True. Only bool and plain numeric dtypes can
    # carry a trustworthy indicator.
    flag_numeric = pd.to_numeric(person[_FLAG_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    invalid_flag_values = int(
        np.count_nonzero(
            ~np.isfinite(flag_numeric) | ~np.isin(flag_numeric, (0.0, 1.0))
        )
    )
    if getattr(person[_FLAG_OUTPUT].dtype, "kind", "") not in "biuf":
        invalid_flag_values = max(invalid_flag_values, len(person))
    flag = flag_numeric == 1.0
    expenses = pd.to_numeric(person[_EXPENSE_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite = np.isfinite(expenses)
    positive = finite & (expenses > 0.0)

    result: dict[str, object] = {
        "flag_rows": int(np.count_nonzero(flag)),
        "flag_share": (
            float(weights[flag].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "flag_share_band": list(_FLAG_SHARE_BAND),
        "invalid_flag_values": invalid_flag_values,
        "expense_rows": int(np.count_nonzero(positive)),
        "expense_weighted_total": float((np.nan_to_num(expenses) * weights).sum()),
        "expense_on_unflagged": int(np.count_nonzero(positive & ~flag)),
        "expense_above_ceiling": int(
            np.count_nonzero(finite & (expenses > _EXPENSE_PLAUSIBILITY_CEILING))
        ),
        "expense_ceiling": _EXPENSE_PLAUSIBILITY_CEILING,
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (expenses < 0.0))),
    }

    # Flag identity against the measured source, when it is still present
    # (base-builder frames and release stores that carry raw ASEC columns).
    if _SELF_CARE_SOURCE in person.columns:
        self_care = pd.to_numeric(person[_SELF_CARE_SOURCE], errors="coerce").to_numpy(
            dtype=np.float64
        )
        result["flag_identity_violations"] = int(
            np.count_nonzero(flag != (self_care == 1.0))
        )

    # Statute-structure certificate: every expense carrier must be a
    # qualifying person of its unit, and units carry at most one carrier.
    # Without the role/unit columns the structure cannot be certified.
    if _ROLE_COLUMN not in person.columns or "person_tax_unit_id" not in person.columns:
        result["structure"] = {
            "missing": [
                column
                for column in (_ROLE_COLUMN, "person_tax_unit_id")
                if column not in person.columns
            ]
        }
        return result
    role = _role(person)
    is_head = (role == "HEAD").to_numpy()
    is_spouse = (role == "SPOUSE").to_numpy()
    is_dependent = (role == "DEPENDENT").to_numpy()
    person_units = person["person_tax_unit_id"].to_numpy()
    unit_married = (
        pd.Series(is_spouse).groupby(person_units).transform("any").to_numpy()
    )
    qualifying = flag & (is_dependent | ((is_head | is_spouse) & unit_married))
    carriers_per_unit = (
        pd.Series(positive.astype(np.int64)).groupby(person_units).transform("sum")
    ).to_numpy()
    result["structure"] = {
        "ineligible_carriers": int(np.count_nonzero(positive & ~qualifying)),
        "multi_carrier_units": int(
            len(set(person_units[positive & (carriers_per_unit > 1)]))
        ),
    }
    return result


def us_adult_care_signal_gate(frame: Frame) -> GateResult:
    """Require a plausible, statute-consistent adult-care input surface."""

    person = frame.table("person")
    missing = [
        column for column in US_ADULT_CARE_OUTPUT_COLUMNS if column not in person
    ]
    if missing:
        return GateResult(
            name="adult_care_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_adult_care_summary(frame)
    failures: list[str] = []
    if summary["invalid_flag_values"]:
        failures.append(
            f"{_FLAG_OUTPUT}: {int(summary['invalid_flag_values'])} null or "
            "non-boolean value(s)."
        )
    if summary["nonfinite"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['nonfinite'])} nonfinite values."
        )
    if summary["negative"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['negative'])} negative values."
        )
    if summary["expense_above_ceiling"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['expense_above_ceiling'])} "
            "value(s) above the "
            f"${summary['expense_ceiling']:,.0f} plausibility ceiling."
        )
    flag_identity = summary.get("flag_identity_violations")
    if flag_identity:
        failures.append(
            f"{_FLAG_OUTPUT}: {int(flag_identity)} value(s) diverge from the "
            f"measured {_SELF_CARE_SOURCE} == 1 identity."
        )
    structure = summary.get("structure")
    if not isinstance(structure, dict):
        failures.append("structure diagnostics missing from summary.")
    elif "missing" in structure:
        failures.append(
            f"statute-structure certificate sources missing: {structure['missing']}."
        )
    else:
        if structure["ineligible_carriers"]:
            failures.append(
                f"{_EXPENSE_OUTPUT}: {int(structure['ineligible_carriers'])} "
                "carrier(s) are not section 21 qualifying persons of their "
                "unit."
            )
        if structure["multi_carrier_units"]:
            failures.append(
                f"{_EXPENSE_OUTPUT}: {int(structure['multi_carrier_units'])} "
                "unit(s) carry more than one expense carrier."
            )
    if not summary["flag_rows"]:
        failures.append(
            f"{_FLAG_OUTPUT}: no carriers; the CDCC adult-care leg would stay "
            "a structural zero."
        )
    share = float(summary["flag_share"])
    low, high = summary["flag_share_band"]
    if summary["flag_rows"] and not (low <= share <= high):
        failures.append(
            f"{_FLAG_OUTPUT}: weighted share {share:.4f} outside plausibility "
            f"band [{low}, {high}]."
        )
    if not summary["expense_rows"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: no positive expenses; the CDCC adult-care "
            "dollar leg would stay a structural zero."
        )
    if summary["expense_on_unflagged"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['expense_on_unflagged'])} "
            "positive value(s) on people without the qualifying flag."
        )
    return GateResult(
        name="adult_care_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _adult_care_surface_carries_signal(frame: Frame) -> bool:
    if any(
        column not in frame.table("person") for column in US_ADULT_CARE_OUTPUT_COLUMNS
    ):
        return False
    return us_adult_care_signal_gate(frame).passed
