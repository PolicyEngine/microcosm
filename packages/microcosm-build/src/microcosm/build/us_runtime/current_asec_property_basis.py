"""Pure, descriptive ASEC property donor basis; never a source authority.

The caller retains the preparation and qualified-source owners around I/O.
These tables establish neither a source seal nor a tax treatment. Original
household design weights describe coverage; they do not alter observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

import numpy as np
import pandas as pd

from .property_income_constants import PROPERTY_COMPONENTS, PROPERTY_REPORTED_TOTAL

# Modeling bridge exclusions, not alternative source codebooks.
OTHER_PROPERTY_CATEGORIES = frozenset({5, 6, 7, 8})
OTHER_UNSPECIFIED_CATEGORY = 19


@dataclass(frozen=True)
class AsecPropertyBasis:
    """Detached descriptive outputs. Constructing or copying grants no authority."""

    person: pd.DataFrame
    provenance: pd.DataFrame
    exclusions: pd.DataFrame
    summary: pd.DataFrame


def _require(condition, reason):
    if not condition:
        raise ValueError("ASEC_PROPERTY_BASIS_" + reason)


def _axis(frame):
    _require(type(frame) is pd.DataFrame, "TABLE")
    _require(
        frame.index.name == "person_id"
        and frame.index.dtype == np.dtype("int64")
        and frame.index.is_unique
        and frame.columns.is_unique,
        "PERSON_AXIS",
    )
    _require({"native_person_id", "source_age"} <= set(frame), "IDENTITY_COLUMNS")
    native = frame.native_person_id
    _require(
        native.dtype == np.dtype("int64") and native.is_unique,
        "NATIVE_AXIS",
    )
    age = _numeric(frame.source_age, "AGE")
    _require(
        np.isfinite(age).all()
        and ((age >= 0) & (age <= 99) & (age == np.floor(age))).all(),
        "AGE",
    )
    return age


def _numeric(series, label):
    _require(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype),
        "NUMERIC:" + label,
    )
    values = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    _require(not np.isinf(values).any(), "NONFINITE:" + label)
    return values


def _text(frame, name):
    _require(name in frame, "COLUMN:" + name)
    series = frame[name]
    _require(
        series.notna().all() and all(type(v) is str for v in series),
        "STATUS:" + name,
    )
    return series.to_numpy(copy=True)


def _code(frame, name):
    _require(name + "_code" in frame, "COLUMN:" + name)
    code = _numeric(frame[name + "_code"], name)
    status = _text(frame, name + "_literal_status")
    _require(((np.isnan(code)) | (code == np.floor(code))).all(), "CODE:" + name)
    return code, status == "in_printed_range"


def _amount(frame, name, *, signed=False, ordinary=False):
    for suffix in ("_amount", "_amount_known", "_reporting_status"):
        _require(name + suffix in frame, "COLUMN:" + name + suffix)
    amount = _numeric(frame[name + "_amount"], name)
    known = frame[name + "_amount_known"]
    _require(
        pd.api.types.is_bool_dtype(known.dtype) and known.notna().all(),
        "KNOWNNESS:" + name,
    )
    known = known.to_numpy(dtype=bool, copy=True)
    status = _text(frame, name + "_reporting_status")
    allowed = np.isin(status, ["known_receipt", "known_nonreceipt"])
    if ordinary:
        allowed |= status == "observed_zero_component"
    _require(
        np.array_equal(known, np.isfinite(amount))
        and np.array_equal(known, allowed)
        and (signed or (amount[known] >= 0).all())
        and (amount[status == "known_nonreceipt"] == 0).all(),
        "AMOUNT_CONTRACT:" + name,
    )
    return amount


def _retirement(interest):
    receipt, readable = _code(interest, "RINT_YN")
    slots = []
    unused = []
    nonreceipt = []
    for n in (1, 2):
        name = f"RINT_VAL{n}"
        amount = _amount(interest, name)
        code, valid = _code(interest, f"RINT_SC{n}")
        published = _numeric(interest[name + "_published_amount"], name)
        literal = _text(interest, name + "_literal_status")
        status = _text(interest, name + "_reporting_status")
        active = (
            valid
            & (code >= 1)
            & (code <= 7)
            & (amount > 0)
            & (status == "known_receipt")
        )
        structural = (
            valid
            & (code == 0)
            & (published == 0)
            & (literal == "in_printed_range")
            & (status == "unreported_account_slot")
            & np.isnan(amount)
        )
        no = (
            valid
            & (code == 0)
            & (published == 0)
            & (amount == 0)
            & (literal == "in_printed_range")
            & (status == "known_nonreceipt")
        )
        slots.append((amount, active))
        unused.append(structural)
        nonreceipt.append(no)
    yes = readable & (receipt == 1)
    resolved_yes = (
        yes
        & (slots[0][1] | slots[1][1])
        & (slots[0][1] | unused[0])
        & (slots[1][1] | unused[1])
    )
    resolved_no = readable & (receipt == 2) & nonreceipt[0] & nonreceipt[1]
    total = np.full(len(interest), np.nan)
    total[resolved_no] = 0
    with np.errstate(over="raise", invalid="raise"):
        total[resolved_yes] = (
            np.where(unused[0], 0, slots[0][0])[resolved_yes]
            + np.where(unused[1], 0, slots[1][0])[resolved_yes]
        )
    derived = [u & resolved_yes for u in unused]
    count = derived[0].astype(np.int8) + derived[1].astype(np.int8)
    label = np.full(len(interest), "unresolved_retirement_interest", dtype=object)
    label[resolved_no] = "known_nonreceipt_sum"
    label[resolved_yes] = "declared_account_sum"
    label[count > 0] = "declared_account_sum_with_unused_slot_zero"
    return total, count, label, derived


def _other_route(routing):
    receipt, receipt_ok = _code(routing, "other_income_receipt")
    category, category_ok = _code(routing, "other_income_category")
    status = _text(routing, "other_income_reporting_status")
    route = _text(routing, "other_income_routing_status")
    reported = (
        receipt_ok & category_ok & (receipt == 1) & (route == "reported_category")
    )
    possible = reported & np.isin(category, tuple(OTHER_PROPERTY_CATEGORIES))
    clear = (
        receipt_ok
        & category_ok
        & (receipt == 2)
        & (category == 0)
        & (status == "known_nonreceipt")
        & (route == "niu_category")
    ) | (
        reported
        & (category >= 1)
        & (category <= 20)
        & ~np.isin(category, (*OTHER_PROPERTY_CATEGORIES, OTHER_UNSPECIFIED_CATEGORY))
    )
    return clear, possible


def _survivor_route(dividend, age):
    # The qualifier already owns this route's source interpretation. Check the
    # lightweight declared code/status agreement, never reread its source here.
    receipt, receipt_ok = _code(dividend, "SUR_YN")
    first, first_ok = _code(dividend, "SUR_SC1")
    second, second_ok = _code(dividend, "SUR_SC2")
    readable = first_ok & second_ok
    yes = receipt_ok & (receipt == 1)
    possible = yes & ((first == 8) | (second == 8))
    clear = (receipt_ok & (receipt == 2) & readable & (first == 0) & (second == 0)) | (
        yes
        & readable
        & ((first > 0) | (second > 0))
        & (first >= 0)
        & (first <= 9)
        & (first != 8)
        & (second >= 0)
        & (second <= 9)
        & (second != 8)
    )
    clear &= age >= 15
    possible &= age >= 15
    declared = dividend.survivor_property_route_clear
    _require(
        pd.api.types.is_bool_dtype(declared.dtype)
        and np.array_equal(clear, declared.fillna(False).to_numpy(dtype=bool))
        and np.array_equal(
            possible, declared.eq(False).fillna(False).to_numpy(dtype=bool)
        ),
        "SURVIVOR_ROUTE_AGREEMENT",
    )
    # SRVS_VAL includes unedited third/fourth sources absent from SUR_SC1/2
    # (2025 dictionary, PDF49 / printed6C-28). Visible-slot clearance cannot
    # establish full property scope for a person reporting survivor income.
    # Retain that descriptive clearance, but only known nonreceipt can clear
    # the first donor bridge until the additional-source scope is qualified.
    full_clear = clear & receipt_ok & (receipt == 2)
    extra_sources_unresolved = yes & (age >= 15)
    return clear, possible, full_clear, extra_sources_unresolved


def _weights(index, membership, weights):
    _require(
        type(membership) is pd.Series
        and membership.index.equals(index)
        and membership.index.name == index.name
        and membership.dtype == np.dtype("int64"),
        "HOUSEHOLD_MEMBERSHIP",
    )
    _require(
        type(weights) is pd.Series
        and weights.index.dtype == np.dtype("int64")
        and weights.index.name == "household_id"
        and weights.index.is_unique
        and set(membership) == set(weights.index),
        "DESIGN_WEIGHT_MEMBERSHIP",
    )
    values = _numeric(weights, "DESIGN_WEIGHTS")
    _require(np.isfinite(values).all() and (values >= 0).all(), "DESIGN_WEIGHTS")
    return weights.loc[membership].to_numpy(dtype=np.float64, copy=True)


def build_asec_property_basis(
    *,
    interest: pd.DataFrame,
    income_routing: pd.DataFrame,
    dividend: pd.DataFrame,
    original_household_membership: pd.Series,
    original_household_design_weights: pd.Series,
) -> AsecPropertyBasis:
    """Compose exactly aligned *original*, qualified ASEC person descriptions.

    No sorting, inner joins, dropping, imputation or source qualification occurs.
    A jointly permuted input is valid; a permutation of one source alone refuses.
    The two masks distinguish reported-total eligibility from complete, balanced
    joint-component fit eligibility. Neither requests a second aggregate model.
    All mass summaries use explicitly supplied original household design weights.
    """
    age = _axis(interest)
    for frame in (income_routing, dividend):
        other_age = _axis(frame)
        _require(frame.index.equals(interest.index), "PERSON_ALIGNMENT")
        _require(
            frame.native_person_id.equals(interest.native_person_id), "NATIVE_ALIGNMENT"
        )
        _require(np.array_equal(age, other_age), "AGE_ALIGNMENT")
    design = _weights(
        interest.index, original_household_membership, original_household_design_weights
    )
    ordinary = _amount(interest, "TRDINT_VAL", ordinary=True)
    reported_interest = _amount(interest, "INT_VAL")
    dividends = _amount(dividend, "DIV_VAL")
    div_receipt, div_readable = _code(dividend, "DIV_YN")
    _require(
        (
            ~np.isfinite(dividends)
            | (
                div_readable
                & (
                    ((div_receipt == 1) & (dividends > 0))
                    | ((div_receipt == 2) & (dividends == 0))
                )
            )
        ).all(),
        "DIVIDEND_RECEIPT_AGREEMENT",
    )
    property_amount = _numeric(income_routing.net_property_known_amount, "RNT_VAL")
    property_status = _text(income_routing, "net_property_reporting_status")
    _require(
        np.array_equal(
            np.isfinite(property_amount),
            np.isin(property_status, ["known_receipt", "known_nonreceipt"]),
        )
        and (property_amount[property_status == "known_nonreceipt"] == 0).all()
        and (property_amount[property_status == "known_receipt"] != 0).all(),
        "PROPERTY_KNOWNNESS",
    )
    retirement, count, derivation, derived_slots = _retirement(interest)
    other_clear, other_possible = _other_route(income_routing)
    (
        survivor_visible_clear,
        survivor_possible,
        survivor_clear,
        survivor_extra_unresolved,
    ) = _survivor_route(dividend, age)
    person = interest[["native_person_id", "source_age"]].copy(deep=True)
    person["original_household_id"] = original_household_membership.to_numpy(copy=True)
    person["original_household_design_weight"] = design
    person["survivor_visible_routes_clear"] = survivor_visible_clear
    person["survivor_full_scope_clear"] = survivor_clear
    for name, amount in zip(
        PROPERTY_COMPONENTS,
        (ordinary, retirement, dividends, property_amount),
        strict=True,
    ):
        person[name] = amount
    with np.errstate(over="raise", invalid="raise"):
        person[PROPERTY_REPORTED_TOTAL] = (
            reported_interest + dividends + property_amount
        )
        person["property_component_sum"] = (
            ordinary + retirement + dividends + property_amount
        )
        person["interest_component_discrepancy"] = reported_interest - (
            ordinary + retirement
        )
        person["reported_minus_component_total"] = (
            person[PROPERTY_REPORTED_TOTAL].to_numpy()
            - person.property_component_sum.to_numpy()
        )
    person["retirement_interest_derivation"] = pd.array(derivation, dtype="string")
    person["retirement_structural_zero_slot_count"] = count
    for n, values in enumerate(derived_slots, 1):
        person[f"retirement_slot{n}_derived_unused_zero"] = values
    exclusions = pd.DataFrame(
        {
            "under15": age < 15,
            "reported_total_unknown": person[PROPERTY_REPORTED_TOTAL].isna(),
            "ordinary_interest_unknown": np.isnan(ordinary),
            "retirement_interest_unknown": np.isnan(retirement),
            "dividends_unknown": np.isnan(dividends),
            "property_receipts_unknown": np.isnan(property_amount),
            "interest_discrepancy_unknown": person.interest_component_discrepancy.isna(),
            "interest_discrepancy_nonzero": person.interest_component_discrepancy.notna()
            & person.interest_component_discrepancy.ne(0),
            "other_income_possible_property": other_possible,
            "other_income_route_unresolved": ~other_clear & ~other_possible,
            "survivor_possible_property": survivor_possible,
            "survivor_route_unresolved": ~survivor_visible_clear & ~survivor_possible,
            "survivor_additional_sources_unresolved": survivor_extra_unresolved,
        },
        index=interest.index,
    )
    person["reported_total_eligible"] = (
        (age >= 15)
        & np.isfinite(person[PROPERTY_REPORTED_TOTAL])
        & other_clear
        & survivor_clear
    )
    person["joint_component_fit_eligible"] = ~exclusions.any(axis=1)
    # Detached qualified columns preserve source statuses, amounts, allocation
    # and disclosure flags. Prefixes make disagreements visible without choosing
    # one qualifier's value or attributing a discrepancy to disclosure treatment.
    provenance = pd.concat(
        [
            frame.copy(deep=True).add_prefix(prefix + ".")
            for prefix, frame in (
                ("interest", interest),
                ("income_routing", income_routing),
                ("dividend", dividend),
            )
        ],
        axis=1,
    )
    masks = {"all": np.ones(len(person), dtype=bool)}
    masks.update(
        {
            name: person[name].to_numpy()
            for name in ("reported_total_eligible", "joint_component_fit_eligible")
        }
    )
    masks["excluded_joint_component_fit"] = ~masks["joint_component_fit_eligible"]
    masks.update(
        {"excluded:" + name: exclusions[name].to_numpy() for name in exclusions}
    )
    rows = []
    for name, mask in masks.items():
        households = original_household_membership.iloc[np.flatnonzero(mask)].unique()
        with np.errstate(over="raise", invalid="raise"):
            mass = fsum(design[mask])
            household_mass = fsum(
                original_household_design_weights.loc[households].to_numpy(
                    dtype=np.float64
                )
            )
        _require(np.isfinite(mass) and np.isfinite(household_mass), "MASS_OVERFLOW")
        rows.append((name, int(mask.sum()), len(households), mass, household_mass))
    summary = pd.DataFrame(
        rows,
        columns=[
            "selection",
            "person_count",
            "household_count",
            "design_weighted_person_mass",
            "union_household_design_mass",
        ],
    ).set_index("selection")
    return AsecPropertyBasis(person, provenance, exclusions, summary)
