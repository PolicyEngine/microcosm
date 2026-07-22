"""ASEC-measured adult/disabled-dependent care inputs for the section 21 CDCC.

PolicyEngine-US 1.764.6 computes the CDCC adult-care leg from two person
inputs neither the retired eCPS nor any prior Populace base carried:
``is_incapable_of_self_care`` (the section 21(b)(1)(B)/(C) qualifying test
and the 21(d)(2) spouse deemed-earnings gate) and
``pre_subsidy_care_expenses`` (the dollar leg summed into
``cdcc_relevant_expenses``).  Both were structural zeros: any CDCC reform
binding through adult care scored exactly $0 (PolicyEngine/populace#451
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
expense level distribution.  Support is restricted to tax units where the
statute can bind: a measured disabled qualifying individual (dependent, or
married head/spouse) plus the section 21(d) earnings structure, honoring the
21(d)(2) deeming rule for an incapacitated spouse.  Assignment is a seeded,
weight-targeted, distribution-preserving draw; no level or usage number is
invented outside the measured donor distribution.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

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
# The measured ASEC self-care difficulty item and the finished childcare
# donor surface; the childcare column is SPM-unit grain attached per person.
US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "PEDISDRS",
    "spm_unit_pre_subsidy_childcare_expenses",
)
# 26 USC 21(d)(1)(A) via gov.irs.credits.cdcc.eligibility.child_age
# (PolicyEngine-US 1.764.6): a dependent under this age qualifies by age, so
# the measured childcare-usage donor universe conditions on it.
US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT = 13
US_ADULT_CARE_EARNED_INCOME_SOURCES: tuple[str, ...] = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
)

_FLAG_OUTPUT, _EXPENSE_OUTPUT = US_ADULT_CARE_OUTPUT_COLUMNS
_SELF_CARE_SOURCE = "PEDISDRS"
_CHILDCARE_SOURCE = "spm_unit_pre_subsidy_childcare_expenses"
_PERSON_WEIGHT_COLUMN = "person_weight"
_TAX_UNIT_WEIGHT_COLUMN = "adult_care_tax_unit_weight"
_SPM_UNIT_WEIGHT_COLUMN = "adult_care_spm_unit_weight"
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_ROLE_COLUMN = "tax_unit_role_input"
_AGE_COLUMN = "age"
_FLAG_SHARE_BAND = (0.002, 0.12)


def us_adult_care_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged adult-care stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
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
            "asec": (
                (
                    person[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str)
                    == _BASE_ASEC_SUPPORT_CHANNEL
                ).to_numpy()
                if _PERSON_SUPPORT_CHANNEL_COLUMN in person.columns
                else np.ones(len(person), dtype=bool)
            ),
        }
    )
    units = unit_frame.groupby("unit", sort=False).agg(
        married=("spouse", "any"),
        flag_head_or_spouse=("flag_head_or_spouse", "any"),
        dependent_prong=("flag_dependent", "any"),
        head_earned=("head_earned", "sum"),
        spouse_earned=("spouse_earned", "sum"),
        weight=("weight", "first"),
    )
    # 21(b)(1)(C) limits the spouse prong to a married head or spouse;
    # 21(b)(1)(B) covers any disabled dependent.
    units["spouse_prong"] = units["flag_head_or_spouse"] & units["married"]

    # 21(d): both spouses must work unless the 21(d)(2) deeming rule covers
    # an incapacitated spouse; a single filer needs earnings of the head.
    joint_min = np.minimum(
        units["head_earned"].to_numpy(), units["spouse_earned"].to_numpy()
    )
    joint_max = np.maximum(
        units["head_earned"].to_numpy(), units["spouse_earned"].to_numpy()
    )
    married = units["married"].to_numpy()
    work_test = np.where(
        units["spouse_prong"].to_numpy(),
        joint_max > 0.0,
        np.where(married, joint_min > 0.0, units["head_earned"].to_numpy() > 0.0),
    )
    eligible = (
        units["spouse_prong"].to_numpy() | units["dependent_prong"].to_numpy()
    ) & work_test

    # Measured donor statistics: the paid-childcare usage rate and positive
    # expense level distribution among units where the childcare leg of the
    # same section 21 expense class binds, on the measured ASEC channel.
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
    donor_positive = donor_universe["childcare"].to_numpy(dtype=np.float64) > 0.0
    total_donor_weight = float(donor_weight.sum())
    if total_donor_weight <= 0.0:
        raise SourceRuntimeError("US adult-care donor universe has zero weight.")
    usage_rate = float(donor_weight[donor_positive].sum()) / total_donor_weight
    if not 0.0 < usage_rate < 1.0:
        raise SourceRuntimeError(
            "US adult-care measured paid-care usage rate "
            f"{usage_rate:.4f} is degenerate."
        )
    level_values = donor_universe.loc[donor_positive, "childcare"].to_numpy(
        dtype=np.float64
    )
    level_weights = donor_weight[donor_positive]

    # Seeded, weight-targeted selection: a deterministic permutation of the
    # eligible units, taking the prefix whose weight reaches the measured
    # usage rate, then a distribution-preserving weighted-quantile draw.
    expenses = np.zeros(len(person), dtype=np.float64)
    eligible_ids = units.index.to_numpy()[eligible]
    if eligible_ids.size:
        eligible_weights = units["weight"].to_numpy(dtype=np.float64)[eligible]
        seed = context.config.seed if context is not None else 0
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(eligible_ids.size)
        cumulative = np.cumsum(eligible_weights[order])
        target = usage_rate * float(eligible_weights.sum())
        take = int(np.searchsorted(cumulative, target, side="left") + 1)
        take = min(take, eligible_ids.size)
        selected_positions = order[:take]
        selected_ids = eligible_ids[selected_positions]

        grid = (np.arange(take, dtype=np.float64) + 0.5) / float(take)
        draws = _weighted_quantile(level_values, level_weights, grid)
        # Deterministic pairing: the permutation order carries the grid.
        by_unit = dict(zip(selected_ids.tolist(), draws.tolist(), strict=True))

        person_units = person["person_tax_unit_id"].to_numpy()
        person_married = (
            pd.Series(person_units).map(units["married"]).fillna(False).to_numpy()
        )
        qualifying_person = flag & (
            is_dependent | ((is_head | is_spouse) & person_married)
        )
        first_qualifying = (
            pd.Series(qualifying_person).groupby(person_units).cumsum().eq(1)
            & qualifying_person
        )
        mapped = pd.Series(person_units).map(by_unit)
        place = first_qualifying.to_numpy() & mapped.notna().to_numpy()
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
    )


def us_adult_care_summary(frame: Frame) -> dict[str, object]:
    """Weighted signal diagnostics for the adult-care surface."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    flag = person[_FLAG_OUTPUT].astype(bool).to_numpy()
    expenses = pd.to_numeric(person[_EXPENSE_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite = np.isfinite(expenses)
    positive = finite & (expenses > 0.0)
    return {
        "flag_rows": int(np.count_nonzero(flag)),
        "flag_share": (
            float(weights[flag].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "flag_share_band": list(_FLAG_SHARE_BAND),
        "expense_rows": int(np.count_nonzero(positive)),
        "expense_weighted_total": float((np.nan_to_num(expenses) * weights).sum()),
        "expense_on_unflagged": int(np.count_nonzero(positive & ~flag)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (expenses < 0.0))),
    }


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
    if summary["nonfinite"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['nonfinite'])} nonfinite values."
        )
    if summary["negative"]:
        failures.append(
            f"{_EXPENSE_OUTPUT}: {int(summary['negative'])} negative values."
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
