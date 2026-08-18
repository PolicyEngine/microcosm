"""UK FRS benefit-unit stochastic take-up assignments."""

from __future__ import annotations

from dataclasses import dataclass

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
    "maximum_extended_childcare_hours_usage",
)
FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS = ("maximum_extended_childcare_hours_usage",)
UK_TAKE_UP_ANCHOR_AGGREGATES = {
    "child_benefit_reported_anchor": "child_benefit_reported",
    "pension_credit_reported_anchor": "pension_credit_reported",
    "universal_credit_reported_anchor": "universal_credit_reported",
}
UK_TAKE_UP_DECLARED_SEEDS = {output: 0 for output in FRS_TAKE_UP_OUTPUT_COLUMNS}
UK_TAKE_UP_SIGNAL_OUTPUTS = (
    ("benunit", "would_claim_child_benefit", "child_benefit"),
    ("benunit", "child_benefit_opts_out", "child_benefit_opts_out_rate"),
    ("benunit", "would_claim_pc", "pension_credit"),
    ("benunit", "would_claim_uc", "universal_credit"),
    ("benunit", "would_claim_tfc", "tax_free_childcare"),
    ("benunit", "would_claim_extended_childcare", "extended_childcare"),
    ("benunit", "would_claim_universal_childcare", "universal_childcare"),
    ("benunit", "would_claim_targeted_childcare", "targeted_childcare"),
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
class UKFRSTakeUpStageTransform:
    """Whole-stage callable for UK benefit-unit take-up assignments."""

    contract: UKTakeUpContract
    stage: SourceStageSpec

    def __call__(self, frame: Frame) -> Frame:
        return add_frs_take_up(frame, contract=self.contract)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_TAKE_UP_OUTPUT_COLUMNS


def add_frs_take_up(frame: Frame, *, contract: UKTakeUpContract) -> Frame:
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    derived = derive_frs_take_up(benunit, anchors=anchors, contract=contract)
    for column in FRS_TAKE_UP_OUTPUT_COLUMNS:
        benunit[column] = derived[column].to_numpy()
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


def derive_frs_take_up(
    benunit: pd.DataFrame,
    *,
    anchors: pd.DataFrame,
    contract: UKTakeUpContract,
) -> pd.DataFrame:
    ids = benunit["benunit_id"].to_numpy()
    values = pd.DataFrame(index=benunit.index)
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
    )
    for output, key in (
        ("would_claim_tfc", "tax_free_childcare"),
        ("would_claim_extended_childcare", "extended_childcare"),
        ("would_claim_universal_childcare", "universal_childcare"),
        ("would_claim_targeted_childcare", "targeted_childcare"),
    ):
        values[output] = assign_binary_from_rate(
            _draws(ids, output), contract.rate(key)
        )
    distribution = contract.continuous_entry("maximum_extended_childcare_hours_usage")
    values["maximum_extended_childcare_hours_usage"] = clipped_normal_from_uniforms(
        _draws(ids, "maximum_extended_childcare_hours_usage"),
        mean=float(distribution["mean"]),
        sd=float(distribution["sd"]),
        lower=float(distribution["lower"]),
        upper=float(distribution["upper"]),
    )
    return values


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_TAKE_UP_DECLARED_SEEDS[output], salt=output
    )


def uk_take_up_signal_gate(
    frame: Frame,
    *,
    contract: UKTakeUpContract | None = None,
    maximum_share_deviation: float = 0.05,
) -> GateResult:
    """Require non-constant UK stochastic flags near contract target shares."""

    resolved = contract if contract is not None else load_uk_take_up_contract()
    failures: list[str] = []
    details: dict[str, object] = {}
    for entity, output, key in UK_TAKE_UP_SIGNAL_OUTPUTS:
        table = frame.table(entity)
        if output not in table.columns:
            failures.append(f"{entity}.{output}: missing stochastic column.")
            continue
        values = np.asarray(table[output], dtype=bool)
        unique_count = int(pd.Series(values).nunique(dropna=False))
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        share = float(np.average(values.astype(float), weights=weights))
        target = _target_share(table, key, resolved, weights)
        details[f"{entity}.{output}"] = {
            "weighted_share": share,
            "target": target,
            "absolute_deviation": abs(share - target),
            "unique_count": unique_count,
        }
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
    if key != "scp":
        return contract.rate(key)
    age = pd.to_numeric(table["age"], errors="coerce").fillna(0).to_numpy()
    targets = np.where(
        age < 6,
        contract.rate("scp_under_6"),
        contract.rate("scp_6_plus"),
    )
    return float(np.average(targets, weights=weights))
