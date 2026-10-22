"""UK FRS benefit-unit stochastic take-up assignments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    assign_binary_with_anchored_residual,
    clipped_normal_from_uniforms,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.frame import Frame

FRS_TAKE_UP_OUTPUT_COLUMNS = (
    "would_claim_child_benefit",
    "child_benefit_opts_out",
    "would_claim_pc",
    "would_claim_uc",
    "would_claim_tfc",
    "would_claim_extended_childcare",
    "would_claim_universal_childcare",
    "would_claim_targeted_childcare",
    "would_claim_uc_childcare",
    "maximum_extended_childcare_hours_usage",
)
# A child under 2 in a benefit unit that claims the working-parent hours uses
# them at the contract's under-2 usage rate: DfE registers far fewer of the
# youngest eligible children than of those aged 2 to 4, mostly for parental
# leave, so one family rate fitted at 2 to 4 would fund too many of them. A
# child that does not use them gets no funded hours from the engine
# (max_free_entitlement_hours_used 0); every other person keeps the engine's
# default of 30. The draw is keyed on the person (seed 0, salt the output);
# ``childcare_copy_take_up`` re-draws the copies' children against their own
# unit's flag. The stage manifest declares the operation;
# ``assert_under_two_usage_declaration`` refuses a manifest that drops it.
UK_UNDER_TWO_USAGE_OUTPUT = "max_free_entitlement_hours_used"
UK_UNDER_TWO_USAGE_KEY = "extended_childcare_under_2_usage"
UK_UNDER_TWO_AGE = 2
UK_ENGINE_FREE_HOURS_DEFAULT = 30.0
FRS_TAKE_UP_PERSON_OUTPUT_COLUMNS = (UK_UNDER_TWO_USAGE_OUTPUT,)
FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS = (
    "maximum_extended_childcare_hours_usage",
    UK_UNDER_TWO_USAGE_OUTPUT,
)
UK_TAKE_UP_ANCHOR_AGGREGATES = {
    "child_benefit_reported_anchor": "child_benefit_reported",
    "pension_credit_reported_anchor": "pension_credit_reported",
    "universal_credit_reported_anchor": "universal_credit_reported",
}
UK_TAKE_UP_DECLARED_SEEDS = {
    output: 0
    for output in (*FRS_TAKE_UP_OUTPUT_COLUMNS, *FRS_TAKE_UP_PERSON_OUTPUT_COLUMNS)
}
# Universal Credit is claimable only by a benefit unit whose claimant or
# partner meets its age conditions: policyengine-uk 2.147.2's ``is_uc_eligible``
# (capital aside) needs a single or joint claimant, taken from
# ``is_uc_claimant``, who meets UC's minimum age and has not reached Pension
# Credit qualifying age. The draw's population is therefore the units with a
# UC claimant or partner aged 18 or over and under that age; any other unit is
# never drawn (#882, microcosm#867, microcosm#1095). The bounds come from the
# engine at the build instant through
# ``uk_take_up_population_policy``: the qualifying age is the State Pension age
# the ``gov.dwp.state_pension.age`` timetable sets for the cohort reaching it
# in the build year, which equals the engine's Pension Credit qualifying age at
# 2024 and 2025, and the minimum age is UC's 18. Three engine cases are not
# mirrored, and the lockstep test names them: 16- and 17-year-old claimants
# under the UC Regs 2013 reg 8 exceptions, whom the engine includes; mixed-age
# couples on the Pension Credit route, whom it excludes; and, from 2.128.0, a
# pension-age family moved onto Universal Credit when its legacy benefit
# closes, whom it includes (SI 2014/1230 reg 60A; whether such a family
# claims is ``would_claim_uc_at_legacy_closure``, which
# ``uc_legacy_closure_take_up`` draws, microcosm#1095). The stage
# manifest declares the same population on the ``would_claim_uc`` operation;
# ``assert_take_up_stage_population_declaration`` refuses a manifest that stops
# saying so, so the declaration and the code cannot drift apart.
UK_ENGINE_ADULT_AGE = 18
UK_UC_AGE_ELIGIBLE_AGGREGATE = "uc_age_eligible"
UK_UC_AGE_ELIGIBLE_METHOD = "any_uc_claimant_aged_18_under_pension_credit_age"
UK_UC_AGE_ELIGIBLE_SOURCE = "age"
UK_UC_AGE_ELIGIBLE_ROLE = "is_uc_claimant"
UK_UC_TAKE_UP_OUTPUT = "would_claim_uc"
# The universal 15 hours are nested under the working-parent entitlement: a
# benefit unit that claims the extended (working-parent) hours claims the
# universal hours with them, and a unit that does not claims them at the
# contract's universal_childcare rate, which is the rate conditional on not
# claiming the extended hours. policyengine-uk allocates a working-parent
# child's first 15 hours to the universal entitlement when the unit's
# standalone universal flag is false, so two independent draws let a unit over
# the £100k limit gain universal hours when the limit is removed. The stage
# manifest declares the nesting on the universal operation
# (``nested_under``); ``assert_universal_childcare_nesting_declaration``
# refuses a manifest that stops saying so. ``childcare_copy_take_up``
# re-draws the copies through the same helper.
UK_EXTENDED_CHILDCARE_OUTPUT = "would_claim_extended_childcare"
UK_UNIVERSAL_CHILDCARE_OUTPUT = "would_claim_universal_childcare"
UK_UNIVERSAL_CHILDCARE_NESTED_UNDER = UK_EXTENDED_CHILDCARE_OUTPUT
# The childcare draws, in the order they are drawn: the flag outputs with
# their contract keys, then the extended-hours usage distribution.
UK_CHILDCARE_TAKE_UP_FLAGS = (
    ("would_claim_tfc", "tax_free_childcare"),
    (UK_EXTENDED_CHILDCARE_OUTPUT, "extended_childcare"),
    (UK_UNIVERSAL_CHILDCARE_OUTPUT, "universal_childcare"),
    ("would_claim_targeted_childcare", "targeted_childcare"),
)
UK_CHILDCARE_HOURS_OUTPUT = "maximum_extended_childcare_hours_usage"
UK_CHILDCARE_TAKE_UP_OUTPUTS = (
    *(output for output, _ in UK_CHILDCARE_TAKE_UP_FLAGS),
    UK_CHILDCARE_HOURS_OUTPUT,
)
# The Universal Credit childcare element is claimed at one rate per family
# type (DWP publishes the single / couple split of the households receiving
# it); the couple rate applies where the benefit unit is a couple (#882).
UK_UC_CHILDCARE_RATE_KEYS = {
    "single": "uc_childcare_single",
    "couple": "uc_childcare_couple",
}
# would_claim_pc is redrawn after the SPI chain by ``pension_credit_take_up``,
# whose rates are shares of the entitled units the engine finds; its stage
# gate measures them, so the terminal signal gate does not (microcosm#1069 R9).
# would_claim_child_benefit and child_benefit_opts_out are redrawn by
# ``child_benefit_take_up`` against HMRC's claim rates of eligible children and
# opted-out share of registered families; its stage gate measures them
# (microcosm#1063).
UK_TAKE_UP_SIGNAL_OUTPUTS = (
    ("benunit", "would_claim_uc", "universal_credit"),
    ("benunit", "would_claim_tfc", "tax_free_childcare"),
    ("benunit", "would_claim_extended_childcare", "extended_childcare"),
    ("benunit", "would_claim_universal_childcare", "universal_childcare"),
    ("benunit", "would_claim_targeted_childcare", "targeted_childcare"),
    ("benunit", "would_claim_uc_childcare", "uc_childcare"),
    ("person", UK_UNDER_TWO_USAGE_OUTPUT, UK_UNDER_TWO_USAGE_KEY),
    ("person", "would_claim_marriage_allowance", "marriage_allowance"),
    ("person", "would_claim_scp", "scp"),
    ("household", "household_owns_tv", "tv_ownership_rate"),
    ("household", "would_evade_tv_licence_fee", "tv_licence_evasion_rate"),
    (
        "household",
        "main_residential_property_purchased_is_first_home",
        "first_time_buyer_rate",
    ),
    ("household", "property_purchased", "property_purchase_rate"),
)


@dataclass(frozen=True)
class UKTakeUpPopulationPolicy:
    """Engine bounds of the Universal Credit take-up population at one instant.

    ``adult_age`` is UC's minimum age and ``state_pension_age`` the age the
    engine's ``gov.dwp.state_pension.age`` timetable sets for the cohort
    attaining it in the build year, which equals its Pension Credit qualifying
    age at 2024 and 2025. ``working_age`` is the age half of the population;
    ``uc_age_eligible_benunits`` adds the claimant role.
    """

    adult_age: int
    state_pension_age: int
    instant: str
    source: str

    def working_age(self, age: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(age, dtype=float)
        return (values >= self.adult_age) & (values < self.state_pension_age)


def uk_take_up_population_policy(build_period: int | str) -> UKTakeUpPopulationPolicy:
    """Read the working-age bounds from the engine at ``{year}-01-01``."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "UK take-up population bounds require `uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    year = int(build_period)
    instant = f"{year}-01-01"
    timetable = parameters.gov.dwp.state_pension.age.age_by_birth_date(instant)
    # From policyengine-uk 2.112.0 State Pension age follows the statutory
    # timetable by date of birth (in months; zero where the statute sets a
    # day instead). The population needs one whole-year age: the age at which
    # everyone born in a single calendar year attains it, for the cohort that
    # attains it in the build year.
    ages = [
        age
        for age in range(UK_ENGINE_ADULT_AGE, 100)
        if (timetable.calc(_birth_dates_in_year(year - age)) == age * 12).all()
    ]
    if len(ages) != 1:
        raise ValueError(
            "the take-up population needs one State Pension age for every adult "
            f"at {instant}; the engine's timetable gives {ages or 'none'}, so the "
            "stage must consume dates of birth before it can form the population"
        )
    return UKTakeUpPopulationPolicy(
        adult_age=UK_ENGINE_ADULT_AGE,
        state_pension_age=ages[0],
        instant=instant,
        source="policyengine-uk parameters " + metadata.version("policyengine-uk"),
    )


def _birth_dates_in_year(year: int) -> np.ndarray:
    """Every date of ``year`` as the YYYYMMDD numbers the timetable is keyed by."""

    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return (days.year * 10_000 + days.month * 100 + days.day).to_numpy()


def assert_take_up_stage_population_declaration(stage: SourceStageSpec) -> None:
    """Refuse a manifest whose declared UC population differs from the code's."""

    declared_aggregate = any(
        op.kind == "aggregate_person_to_benunit"
        and op.parameters.get("method") == UK_UC_AGE_ELIGIBLE_METHOD
        and dict(op.parameters.get("aggregates") or {})
        == {UK_UC_AGE_ELIGIBLE_AGGREGATE: UK_UC_AGE_ELIGIBLE_SOURCE}
        and op.parameters.get("role_column") == UK_UC_AGE_ELIGIBLE_ROLE
        for op in stage.operations
    )
    declared_population = any(
        op.kind == "assign_binary_with_anchored_residual"
        and op.parameters.get("output") == UK_UC_TAKE_UP_OUTPUT
        and op.parameters.get("population") == UK_UC_AGE_ELIGIBLE_AGGREGATE
        for op in stage.operations
    )
    if not (declared_aggregate and declared_population):
        seen = [
            (op.kind, dict(op.parameters))
            for op in stage.operations
            if op.kind == "aggregate_person_to_benunit"
            or op.parameters.get("output") == UK_UC_TAKE_UP_OUTPUT
        ]
        raise ValueError(
            f"stage {stage.stage!r} must declare the {UK_UC_AGE_ELIGIBLE_AGGREGATE!r} "
            f"aggregate ({UK_UC_AGE_ELIGIBLE_METHOD} over "
            f"{UK_UC_AGE_ELIGIBLE_SOURCE}, role {UK_UC_AGE_ELIGIBLE_ROLE}) and "
            f"population={UK_UC_AGE_ELIGIBLE_AGGREGATE!r} "
            f"on the {UK_UC_TAKE_UP_OUTPUT!r} operation; the code draws Universal "
            "Credit take-up over that population and refuses a manifest that says "
            f"otherwise (declared: {seen!r})"
        )


def assert_universal_childcare_nesting_declaration(stage: SourceStageSpec) -> None:
    """Refuse a manifest that does not nest universal under extended take-up."""

    declared = [
        dict(op.parameters)
        for op in stage.operations
        if op.parameters.get("output") == UK_UNIVERSAL_CHILDCARE_OUTPUT
    ]
    if not declared or any(
        params.get("nested_under") != UK_UNIVERSAL_CHILDCARE_NESTED_UNDER
        for params in declared
    ):
        raise ValueError(
            f"stage {stage.stage!r} must declare "
            f"nested_under={UK_UNIVERSAL_CHILDCARE_NESTED_UNDER!r} on the "
            f"{UK_UNIVERSAL_CHILDCARE_OUTPUT!r} operation; the code draws "
            "universal take-up as extended OR (u < the conditional universal "
            f"rate) and refuses a manifest that says otherwise (declared: {declared!r})"
        )


def assert_under_two_usage_declaration(stage: SourceStageSpec) -> None:
    """Refuse a manifest that does not declare the under-2 usage draw."""

    declared = [
        dict(op.parameters)
        for op in stage.operations
        if op.parameters.get("output") == UK_UNDER_TWO_USAGE_OUTPUT
    ]
    expected = {"rate_key": UK_UNDER_TWO_USAGE_KEY, "assignment_unit": "person"}
    if not declared or any(
        any(params.get(key) != value for key, value in expected.items())
        for params in declared
    ):
        raise ValueError(
            f"stage {stage.stage!r} must declare the {UK_UNDER_TWO_USAGE_OUTPUT!r} "
            f"operation with {expected!r}; the code draws whether each child under "
            f"{UK_UNDER_TWO_AGE} in a unit claiming the working-parent hours uses "
            f"them and refuses a manifest that says otherwise (declared: {declared!r})"
        )


@dataclass(frozen=True)
class UKFRSTakeUpStageTransform:
    """Whole-stage callable for UK benefit-unit take-up assignments."""

    contract: UKTakeUpContract
    stage: SourceStageSpec
    population_policy: UKTakeUpPopulationPolicy | None = None

    def __call__(self, frame: Frame) -> Frame:
        assert_take_up_stage_population_declaration(self.stage)
        assert_universal_childcare_nesting_declaration(self.stage)
        assert_under_two_usage_declaration(self.stage)
        policy = self.population_policy or uk_take_up_population_policy(
            uk_time_period(frame)
        )
        return add_frs_take_up(frame, contract=self.contract, population_policy=policy)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (*FRS_TAKE_UP_OUTPUT_COLUMNS, *FRS_TAKE_UP_PERSON_OUTPUT_COLUMNS)


def add_frs_take_up(
    frame: Frame,
    *,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
) -> Frame:
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    uc_age_eligible = uc_age_eligible_benunits(person, benunit, population_policy)
    derived = derive_frs_take_up(
        benunit, anchors=anchors, contract=contract, uc_age_eligible=uc_age_eligible
    )
    for column in FRS_TAKE_UP_OUTPUT_COLUMNS:
        benunit[column] = derived[column].to_numpy()
    person[UK_UNDER_TWO_USAGE_OUTPUT] = derive_under_two_childcare_usage(
        person,
        claiming=person_unit_flag(
            person, benunit, benunit[UK_EXTENDED_CHILDCARE_OUTPUT].to_numpy()
        ),
        contract=contract,
    )
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def aggregate_person_reported_to_benunit(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> pd.DataFrame:
    """Return anchor booleans by benefit unit for positive reported amounts."""

    grouped = pd.DataFrame(index=benunit["benunit_id"])
    for output, source in UK_TAKE_UP_ANCHOR_AGGREGATES.items():
        if source not in person.columns:
            raise KeyError(
                f"take-up anchor source column {source!r} is missing from the "
                "person table; anchored assignment refuses to run without it"
            )
        reporter_ids = person.loc[
            pd.to_numeric(person[source], errors="coerce").fillna(0) > 0,
            "person_benunit_id",
        ]
        grouped[output] = grouped.index.isin(reporter_ids)
    return grouped.reset_index(drop=True)


def uc_age_eligible_benunits(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    policy: UKTakeUpPopulationPolicy,
) -> np.ndarray:
    """True where a UC claimant or partner of the unit is within ``policy``'s ages."""

    missing = [
        column
        for column in (UK_UC_AGE_ELIGIBLE_SOURCE, UK_UC_AGE_ELIGIBLE_ROLE)
        if column not in person.columns
    ]
    if missing:
        raise KeyError(
            f"person columns {missing} are missing; the Universal Credit take-up "
            "population cannot be formed without them"
        )
    age = pd.to_numeric(person[UK_UC_AGE_ELIGIBLE_SOURCE], errors="coerce").fillna(0)
    claimant = person[UK_UC_AGE_ELIGIBLE_ROLE].to_numpy(dtype=bool)
    eligible_adult = policy.working_age(age) & claimant
    eligible_ids = set(person.loc[eligible_adult, "person_benunit_id"])
    return benunit["benunit_id"].isin(eligible_ids).to_numpy(dtype=bool)


def derive_frs_take_up(
    benunit: pd.DataFrame,
    *,
    anchors: pd.DataFrame,
    contract: UKTakeUpContract,
    uc_age_eligible: np.ndarray,
) -> pd.DataFrame:
    ids = benunit["benunit_id"].to_numpy()
    values = pd.DataFrame(index=benunit.index)
    population = np.asarray(uc_age_eligible, dtype=bool)
    if population.shape != ids.shape:
        raise ValueError("uc_age_eligible must align with the benefit-unit table")
    values["would_claim_child_benefit"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_child_benefit"),
        contract.rate("child_benefit"),
        anchors["child_benefit_reported_anchor"].to_numpy(dtype=bool),
    )
    values["child_benefit_opts_out"] = assign_binary_from_rate(
        _draws(ids, "child_benefit_opts_out"),
        contract.rate("child_benefit_opts_out_rate"),
    )
    values["would_claim_pc"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_pc"),
        contract.rate("pension_credit"),
        anchors["pension_credit_reported_anchor"].to_numpy(dtype=bool),
    )
    values["would_claim_uc"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_uc"),
        contract.rate("universal_credit"),
        anchors["universal_credit_reported_anchor"].to_numpy(dtype=bool),
        population=population,
    )
    childcare = derive_childcare_take_up(ids, contract=contract)
    for output, _ in UK_CHILDCARE_TAKE_UP_FLAGS:
        values[output] = childcare[output]
    values["would_claim_uc_childcare"] = _draws(
        ids, "would_claim_uc_childcare"
    ) < uc_childcare_rates(benunit, contract)
    values[UK_CHILDCARE_HOURS_OUTPUT] = childcare[UK_CHILDCARE_HOURS_OUTPUT]
    return values


def nest_universal_childcare(
    extended: np.ndarray, draws: np.ndarray, conditional_rate: float
) -> np.ndarray:
    """Universal take-up nested under the working-parent flag.

    A unit that claims the extended hours claims the universal hours; any
    other unit claims them where its universal draw falls under
    ``conditional_rate``, the universal rate among units not claiming the
    extended hours. P(universal | extended) is 1 by construction.
    """

    return np.asarray(extended, dtype=bool) | assign_binary_from_rate(
        draws, conditional_rate
    )


def derive_childcare_take_up(
    ids: np.ndarray, *, contract: UKTakeUpContract
) -> dict[str, np.ndarray]:
    """The childcare take-up flags and extended-hours usage keyed on ``ids``.

    Each output draws its own identity stream (seed 0, salt the output name):
    Tax-Free Childcare, extended and targeted at their contract rates, the
    universal flag nested under the extended one (``nest_universal_childcare``)
    and the extended-hours usage from the contract's clipped normal. Both
    ``frs_take_up`` (keyed on FRS benefit units) and ``childcare_copy_take_up``
    (keyed on the copies' own benefit units) draw through it.
    """

    values: dict[str, np.ndarray] = {}
    for output, key in UK_CHILDCARE_TAKE_UP_FLAGS:
        if output == UK_UNIVERSAL_CHILDCARE_OUTPUT:
            values[output] = nest_universal_childcare(
                values[UK_UNIVERSAL_CHILDCARE_NESTED_UNDER],
                _draws(ids, output),
                contract.rate(key),
            )
        else:
            values[output] = assign_binary_from_rate(
                _draws(ids, output), contract.rate(key)
            )
    distribution = contract.continuous_entry(UK_CHILDCARE_HOURS_OUTPUT)
    values[UK_CHILDCARE_HOURS_OUTPUT] = clipped_normal_from_uniforms(
        _draws(ids, UK_CHILDCARE_HOURS_OUTPUT),
        mean=float(distribution["mean"]),
        sd=float(distribution["sd"]),
        lower=float(distribution["lower"]),
        upper=float(distribution["upper"]),
    )
    return values


def person_unit_flag(
    person: pd.DataFrame, benunit: pd.DataFrame, flag: np.ndarray
) -> np.ndarray:
    """Each person's benefit-unit ``flag``, through ``person_benunit_id``."""

    lookup = pd.Series(
        np.asarray(flag, dtype=bool), index=benunit["benunit_id"].to_numpy()
    )
    mapped = lookup.reindex(person["person_benunit_id"].to_numpy())
    if mapped.isna().any():
        raise ValueError("every person must belong to a benefit unit in the frame")
    return mapped.to_numpy(dtype=bool)


def under_two_children(person: pd.DataFrame) -> np.ndarray:
    """Persons of whole-year age under ``UK_UNDER_TWO_AGE``."""

    if "age" not in person.columns:
        raise KeyError("person.age is missing; the under-2 usage draw needs it")
    age = pd.to_numeric(person["age"], errors="coerce")
    if age.isna().any():
        raise ValueError("person.age must be numeric for the under-2 usage draw")
    return age.to_numpy(dtype=float) < UK_UNDER_TWO_AGE


def derive_under_two_childcare_usage(
    person: pd.DataFrame, *, claiming: np.ndarray, contract: UKTakeUpContract
) -> np.ndarray:
    """``max_free_entitlement_hours_used`` for every person in ``person``.

    A child under 2 whose unit claims the working-parent hours uses them where
    its draw (keyed on ``person_id``) falls under the contract's under-2 usage
    rate and gets 0 hours otherwise; every other person keeps the engine's
    default.
    """

    ids = person["person_id"].to_numpy()
    uses = assign_binary_from_rate(
        _draws(ids, UK_UNDER_TWO_USAGE_OUTPUT), contract.rate(UK_UNDER_TWO_USAGE_KEY)
    )
    drawn = under_two_children(person) & np.asarray(claiming, dtype=bool)
    return np.where(drawn & ~uses, 0.0, UK_ENGINE_FREE_HOURS_DEFAULT)


def universal_childcare_marginal_rate(contract: UKTakeUpContract) -> float:
    """The universal flag's expected share: r_e + (1 - r_e) * r_u_cond."""

    extended = contract.rate("extended_childcare")
    return extended + (1.0 - extended) * contract.rate("universal_childcare")


def uc_childcare_rates(benunit: pd.DataFrame, contract: UKTakeUpContract) -> np.ndarray:
    """Per-unit childcare-element take-up rate: the couple or single contract rate."""

    if "is_married" not in benunit.columns:
        raise KeyError(
            "benunit.is_married is missing; the Universal Credit childcare "
            "take-up rate is drawn by family type"
        )
    couple = benunit["is_married"].fillna(False).to_numpy(dtype=bool)
    return np.where(
        couple,
        contract.rate(UK_UC_CHILDCARE_RATE_KEYS["couple"]),
        contract.rate(UK_UC_CHILDCARE_RATE_KEYS["single"]),
    )


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_TAKE_UP_DECLARED_SEEDS[output], salt=output
    )


def uk_take_up_signal_gate(
    frame: Frame,
    *,
    contract: UKTakeUpContract | None = None,
    maximum_share_deviation: float = 0.05,
    population_policy: UKTakeUpPopulationPolicy | None = None,
) -> GateResult:
    """Require non-constant UK stochastic flags near contract target shares."""

    resolved = contract if contract is not None else load_uk_take_up_contract()
    failures: list[str] = []
    details: dict[str, object] = {}
    policy: UKTakeUpPopulationPolicy | None = population_policy
    for entity, output, key in UK_TAKE_UP_SIGNAL_OUTPUTS:
        table = frame.table(entity)
        if output not in table.columns:
            failures.append(f"{entity}.{output}: missing stochastic column.")
            continue
        values = np.asarray(table[output], dtype=bool)
        unique_count = int(pd.Series(values).nunique(dropna=False))
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        population = np.ones(values.shape, dtype=bool)
        if key == "universal_credit":
            # The contract rate is a share of the units that can claim: those
            # with a UC claimant or partner within the engine's age bounds at
            # the build instant. The gate measures the realized share on the
            # same population.
            if policy is None:
                policy = uk_take_up_population_policy(uk_time_period(frame))
            population = uc_age_eligible_benunits(frame.table("person"), table, policy)
            details["universal_credit_population_policy"] = {
                "adult_age": policy.adult_age,
                "state_pension_age": policy.state_pension_age,
                "instant": policy.instant,
                "source": policy.source,
            }
        if key == UK_UNDER_TWO_USAGE_KEY:
            # The rate is a share of the children under 2 in units that claim
            # the working-parent hours; every other person keeps 30 hours.
            benunit_table = frame.table("benunit")
            population = under_two_children(table) & person_unit_flag(
                table,
                benunit_table,
                benunit_table[UK_EXTENDED_CHILDCARE_OUTPUT].to_numpy(),
            )
        if not population.any() or float(weights[population].sum()) <= 0.0:
            failures.append(
                f"{entity}.{output}: no unit in the draw's population carries "
                "weight; the take-up share cannot be measured."
            )
            continue
        share = float(
            np.average(values[population].astype(float), weights=weights[population])
        )
        target = _target_share(table, key, resolved, weights)
        details[f"{entity}.{output}"] = {
            "weighted_share": share,
            "target": target,
            "absolute_deviation": abs(share - target),
            "unique_count": unique_count,
            "population_units": int(population.sum()),
        }
        if (
            output == UK_UNIVERSAL_CHILDCARE_OUTPUT
            and UK_UNIVERSAL_CHILDCARE_NESTED_UNDER in table.columns
        ):
            # Nested draw: P(universal | extended) = 1, so no benefit unit
            # claims the extended hours without the universal ones.
            extended = np.asarray(
                table[UK_UNIVERSAL_CHILDCARE_NESTED_UNDER], dtype=bool
            )
            violations = int((extended & ~values).sum())
            details[f"{entity}.{output}"]["extended_without_universal_units"] = (
                violations
            )
            if violations:
                failures.append(
                    f"{entity}.{output}: {violations} benefit unit(s) claim the "
                    "extended hours without the universal hours; the universal "
                    "flag is nested under would_claim_extended_childcare."
                )
        if unique_count < 2:
            failures.append(
                f"{entity}.{output}: constant column; stochastic take-up "
                "assignment must not collapse to a universal default."
            )
        if abs(share - target) > maximum_share_deviation:
            failures.append(
                f"{entity}.{output}: weighted share {share:.3f} differs from "
                f"contract target {target:.3f} by more than "
                f"{maximum_share_deviation:.3f}."
            )
    return GateResult(
        name="take_up_signal",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _target_share(
    table: pd.DataFrame,
    key: str,
    contract: UKTakeUpContract,
    weights: np.ndarray,
) -> float:
    if key == "uc_childcare":
        return float(np.average(uc_childcare_rates(table, contract), weights=weights))
    if key == "universal_childcare":
        # The contract rate is conditional on not claiming the extended
        # hours; the flag's share is the nested draw's marginal.
        return universal_childcare_marginal_rate(contract)
    if key != "scp":
        return contract.rate(key)
    age = pd.to_numeric(table["age"], errors="coerce").fillna(0).to_numpy()
    targets = np.where(
        age < 6,
        contract.rate("scp_under_6"),
        contract.rate("scp_6_plus"),
    )
    return float(np.average(targets, weights=weights))
