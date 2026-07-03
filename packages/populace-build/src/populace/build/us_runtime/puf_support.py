"""US support expansion for PUF tax-detail imputations.

The PUF tax-detail donor needs a distinct support channel: the ASEC/CPS
records remain as the baseline channel, and a cloned channel receives
PUF-sourced tax detail without overwriting the baseline rows. The expansion is
mass-conserving: with two support channels, each channel receives half of the
incoming weights so the frame's aggregate population does not double.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from populace.frame import US_SCHEMA, Frame, WeightKind, Weights
from populace.frame.schema import EntitySchema

QRF: Any | None = None

__all__ = [
    "BASE_ASEC_SUPPORT_CHANNEL",
    "PUF_TAX_DETAIL_SUPPORT_CHANNEL",
    "US_PUF_SUPPORT_STAGE_NAME",
    "clone_us_frame_for_puf_support",
    "impute_us_puf_tax_detail_support",
    "puf_tax_unit_donor_from_arrays",
    "support_channel_column",
    "support_clone_index_column",
    "support_source_id_column",
]

BASE_ASEC_SUPPORT_CHANNEL = "asec"
PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
US_PUF_SUPPORT_STAGE_NAME = "puf_support_channel"

_DEFAULT_SUPPORT_CHANNELS = (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
)

PUF_TAX_DETAIL_DEFAULT_PREDICTORS = (
    "puf_predictor_filing_status_code",
    "puf_predictor_tax_unit_person_count",
    "puf_predictor_employment_income",
    "puf_predictor_self_employment_income",
    "puf_predictor_taxable_interest_income",
    "puf_predictor_dividend_income",
    "puf_predictor_short_term_capital_gains",
    "puf_predictor_long_term_capital_gains",
)

PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "tax_exempt_interest_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "real_estate_taxes",
    "home_mortgage_interest",
    "student_loan_interest",
    # The engine owns the realized contribution amounts through the
    # IRA-limit scale and self-employment caps; the persistable leaves are
    # the desired contributions, equal to the PUF's observed deductions at
    # baseline (issue #278).
    "traditional_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
    "rental_income",
    "estate_income",
    "farm_income",
    "miscellaneous_income",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
)

PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
)

PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS: tuple[str, ...] = (
    "first_home_mortgage_balance",
    "second_home_mortgage_balance",
    "first_home_mortgage_interest",
    "second_home_mortgage_interest",
    "first_home_mortgage_origination_year",
    "second_home_mortgage_origination_year",
    "health_savings_account_ald",
)

_PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS = frozenset(
    {
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
)
_PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS = frozenset(
    {
        "taxable_interest_income",
    }
)

PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS = frozenset(
    {
        "interest_deduction",
        "state_withheld_income_tax",
        # The self-employed ALDs are engine aggregates of formula-owned
        # per-person caps; the persistable leaves are the contribution and
        # premium inputs those formulas cap.
        "self_employed_pension_contribution_ald",
        "self_employed_pension_contribution_ald_person",
        "self_employed_health_insurance_ald",
        "self_employed_health_insurance_ald_person",
    }
)

_PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS = frozenset(
    {
        "employment_income",
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "tax_exempt_interest_income",
        "long_term_capital_gains_before_response",
        "taxable_pension_income",
        "taxable_private_pension_income",
        "taxable_ira_distributions",
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
        "charitable_cash_donations",
        "charitable_non_cash_donations",
        "real_estate_taxes",
        "home_mortgage_interest",
        "student_loan_interest",
        "traditional_ira_contributions_desired",
        "self_employed_pension_contributions_desired",
        "health_savings_account_ald",
        "first_home_mortgage_balance",
        "second_home_mortgage_balance",
        "first_home_mortgage_interest",
        "second_home_mortgage_interest",
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
)
# Person-grain PE input leaves the processed PUF artifact only observes as
# tax-unit-grain deduction arrays. The donor carries the tax-unit total under
# the leaf's name; imputed totals are distributed over the cloned people by
# the leaf's distribution basis below.
_PERSON_OUTPUT_TAX_UNIT_GRAIN_SOURCES: Mapping[str, str] = {
    "self_employed_pension_contributions_desired": (
        "self_employed_pension_contribution_ald"
    ),
}
# Distribution basis for person outputs the ASEC channel carries no values
# for: the imputed tax-unit total is split by these person columns' shares
# (falling back to the unit's first person when the basis is empty).
# Contributions require compensation, so an earnings basis keeps per-person
# contribution limits binding the way they do on the donor records.
_PERSON_OUTPUT_DISTRIBUTION_BASIS: Mapping[str, tuple[str, ...]] = {
    "traditional_ira_contributions_desired": (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ),
    "self_employed_pension_contributions_desired": (
        "self_employment_income_before_lsr",
    ),
}
_PREDICTOR_LEAF_ALIASES: Mapping[str, tuple[str, ...]] = {
    "employment_income": ("employment_income_before_lsr",),
    "self_employment_income": ("self_employment_income_before_lsr",),
    "long_term_capital_gains": ("long_term_capital_gains_before_response",),
    "taxable_pension_income": (
        "taxable_private_pension_income",
        "taxable_public_pension_income",
    ),
    "social_security": (
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
    ),
}

_FILING_STATUS_CODES = {
    "SINGLE": 1.0,
    "JOINT": 2.0,
    "SEPARATE": 3.0,
    "HEAD_OF_HOUSEHOLD": 4.0,
    "SURVIVING_SPOUSE": 5.0,
}
_PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS = {
    "health_insurance_premiums_without_medicare_part_b": 0.453,
    "other_medical_expenses": 0.325,
    "medicare_part_b_premiums": 0.137,
    "over_the_counter_health_expenses": 0.085,
}
_PUF_PREDICTOR_PREFIX = "puf_predictor_"


def clone_us_frame_for_puf_support(
    frame: Frame,
    *,
    channels: Sequence[str] = _DEFAULT_SUPPORT_CHANNELS,
) -> Frame:
    """Clone a US frame into support channels for PUF detail imputation.

    Args:
        frame: A US-schema frame after unit assignment and CPS-carried
            derivations.
        channels: Ordered support-channel names. The first channel keeps the
            original IDs; later channels receive remapped IDs. Weights are
            split evenly across channels.

    Returns:
        A new frame with every entity table cloned once per support channel,
        channel/source metadata added with entity-prefixed column names, all
        structural IDs remapped consistently, and typed weights mass-conserved.

    Raises:
        ValueError: If the frame is not US-schema, channel names are invalid,
            metadata columns already exist, or an ID remapping would collide.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF support expansion currently requires the US schema.")
    support_channels = _validate_channels(channels)
    _reject_metadata_collisions(frame, support_channels)
    id_multiplier = _id_multiplier_for_frame(frame)

    tables: dict[str, pd.DataFrame] = {}
    for entity in frame.entities:
        tables[entity] = _clone_entity_table(
            frame.table(entity),
            entity=entity,
            schema=frame.schema,
            channels=support_channels,
            id_multiplier=id_multiplier,
        )
    for link_name in frame.links:
        tables[link_name] = _clone_link_table(
            frame.link(link_name),
            link_name=link_name,
            schema=frame.schema,
            channels=support_channels,
            id_multiplier=id_multiplier,
        )

    weights = {
        entity: Weights(
            values=np.tile(
                frame.weights_for(entity).values / len(support_channels),
                len(support_channels),
            ),
            kind=frame.weights_for(entity).kind,
        )
        for entity in frame.weighted_entities
    }
    strata = pd.concat(
        [frame.strata.copy() for _channel in support_channels],
        ignore_index=True,
    )
    return Frame(
        tables,
        frame.schema,
        weights,
        strata,
        mass_log=frame.mass_log,
    )


def puf_tax_unit_donor_from_arrays(
    arrays: Mapping[str, Sequence[Any]],
    *,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
) -> pd.DataFrame:
    """Build a tax-unit donor table from processed PUF array columns.

    The processed PUF array artifact stores person-grain and tax-unit-grain
    arrays in one HDF file. This helper reduces person-grain PUF variables to
    tax-unit aggregates so the shared QRF imputer can fit one tax-unit model
    and later distribute person outputs over the cloned CPS people.

    Args:
        arrays: Mapping from column name to array-like values.
        person_outputs: Person-grain PE input variables to aggregate by tax
            unit.
        tax_unit_outputs: Tax-unit-grain PE input variables to carry or derive.

    Returns:
        A tax-unit donor DataFrame with numeric predictors, requested outputs,
        and a ``weight`` column.
    """

    _require_array_columns(
        arrays,
        ["tax_unit_id", "household_weight", "filing_status", "person_tax_unit_id"],
        label="PUF arrays",
    )
    tax_unit_id = _numeric_array(arrays["tax_unit_id"]).astype("int64")
    person_tax_unit_id = _numeric_array(arrays["person_tax_unit_id"]).astype("int64")

    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": tax_unit_id,
            "weight": _numeric_array(arrays["household_weight"]),
            "filing_status_code": _filing_status_codes(arrays["filing_status"]),
        }
    )
    person = pd.DataFrame({"tax_unit_id": person_tax_unit_id})

    _reject_formula_owned_outputs(person_outputs, tax_unit_outputs)
    person_source_columns = set(person_outputs)
    if "interest_deduction" in tax_unit_outputs:
        person_source_columns.add("home_mortgage_interest")
    for output in sorted(person_source_columns):
        source = _person_source_values(arrays, output)
        if source is not None:
            person[output] = source

    grouped = person.groupby("tax_unit_id", sort=False).sum(numeric_only=True)
    tax_unit = tax_unit.join(grouped, on="tax_unit_id")
    tax_unit["tax_unit_person_count"] = (
        person.groupby("tax_unit_id", sort=False).size().reindex(tax_unit_id).to_numpy()
    )
    for output in person_outputs:
        if output in tax_unit.columns:
            continue
        source = _PERSON_OUTPUT_TAX_UNIT_GRAIN_SOURCES.get(output)
        if source is None or source not in arrays:
            continue
        values = _numeric_array(arrays[source])
        if len(values) == len(tax_unit_id):
            tax_unit[output] = values

    for output in tax_unit_outputs:
        values = _tax_unit_source_values(arrays, tax_unit_id, output, grouped)
        if values is not None:
            tax_unit[output] = values

    required_outputs = [*person_outputs, *tax_unit_outputs]
    missing = [column for column in required_outputs if column not in tax_unit.columns]
    if missing:
        raise ValueError(f"PUF donor cannot derive requested output(s): {missing}.")

    for column in tax_unit.columns:
        if column == "tax_unit_id":
            continue
        tax_unit[column] = pd.to_numeric(tax_unit[column], errors="coerce").fillna(0.0)
    _add_predictor_aliases(tax_unit, PUF_TAX_DETAIL_DEFAULT_PREDICTORS)
    return tax_unit


def impute_us_puf_tax_detail_support(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    seed: int = 0,
    n_estimators: int = 100,
) -> Frame:
    """Impute PUF-observed inputs onto the PUF support channel.

    Baseline ASEC support rows are left untouched. PUF support rows receive
    tax-unit predictions from the PUF donor; person-grain predicted tax-unit
    totals are distributed over the cloned people using their copied ASEC
    within-tax-unit shares, falling back to the first person in the unit when
    the copied support has no mass for a variable.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("PUF tax-detail support imputation requires the US schema.")
    tax_unit_channel = support_channel_column("tax_unit")
    person_channel = support_channel_column("person")
    if tax_unit_channel not in frame.table("tax_unit").columns:
        raise ValueError("PUF support metadata is missing from the tax_unit table.")
    if person_channel not in frame.table("person").columns:
        raise ValueError("PUF support metadata is missing from the person table.")

    _reject_formula_owned_outputs(person_outputs, tax_unit_outputs)
    predictors = tuple(predictors)
    person_outputs = tuple(person_outputs)
    tax_unit_outputs = tuple(tax_unit_outputs)
    outputs = (*person_outputs, *tax_unit_outputs)
    donor_tax_units = donor_tax_units.copy()
    _add_predictor_aliases(donor_tax_units, predictors)
    missing_donor = [
        column
        for column in (*predictors, *outputs, "weight")
        if column not in donor_tax_units
    ]
    if missing_donor:
        raise ValueError(
            f"PUF donor tax-unit table missing column(s): {missing_donor}."
        )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF

    donor = donor_tax_units.loc[:, [*predictors, *outputs, "weight"]].copy()
    for column in donor.columns:
        donor[column] = pd.to_numeric(donor[column], errors="coerce").fillna(0.0)
    donor_frame = _tax_unit_model_frame(donor)
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        donor_frame,
        list(predictors),
        list(outputs),
        weights="design",
    )

    features = _tax_unit_feature_frame(frame, predictors)
    puf_mask = (
        frame.table("tax_unit")[tax_unit_channel].to_numpy()
        == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    if not puf_mask.any():
        raise ValueError("PUF support channel has no tax-unit rows.")
    predictions = fitted.predict(features.loc[puf_mask, list(predictors)])
    for column in outputs:
        if column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS:
            predictions[column] = predictions[column].clip(lower=0.0)
        if column in _PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )
        if column in _PUF_TAX_DETAIL_DISCRETE_TAX_UNIT_OUTPUTS:
            predictions[column] = _snap_to_observed_values(
                predictions[column],
                donor[column],
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tax_unit_ids = tables["tax_unit"].loc[puf_mask, "tax_unit_id"].to_numpy()
    _reconcile_puf_social_security_components(
        predictions,
        tables["person"],
        person_channel=person_channel,
        tax_unit_ids=tax_unit_ids,
        requested_components=person_outputs,
    )
    for column in tax_unit_outputs:
        _ensure_float_output_column(tables["tax_unit"], column)
        tables["tax_unit"].loc[puf_mask, column] = predictions[column].to_numpy()

    person_puf_mask = tables["person"][person_channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    for column in person_outputs:
        _ensure_float_output_column(tables["person"], column)
        _write_person_tax_unit_totals(
            tables["person"],
            mask=person_puf_mask,
            column=column,
            totals=pd.Series(predictions[column].to_numpy(), index=tax_unit_ids),
            nonnegative=column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS,
            fallback_basis_columns=_PERSON_OUTPUT_DISTRIBUTION_BASIS.get(column, ()),
        )
    for column in person_outputs:
        if column in _PUF_TAX_DETAIL_SPARSE_PERSON_OUTPUTS:
            _sparsify_tax_unit_person_output_to_donor_positive_rate(
                tables,
                column=column,
                donor_positive_rate=_weighted_positive_rate(
                    donor[column],
                    donor["weight"],
                ),
                household_weights=frame.weights_for("household").values,
                person_channel=person_channel,
                tax_unit_channel=tax_unit_channel,
            )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def support_channel_column(entity: str) -> str:
    """Return the entity-prefixed support-channel metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_channel"


def support_clone_index_column(entity: str) -> str:
    """Return the entity-prefixed clone-index metadata column."""

    _require_entity_name(entity)
    return f"{entity}_support_clone_index"


def support_source_id_column(entity: str) -> str:
    """Return the entity-prefixed original-ID provenance column."""

    _require_entity_name(entity)
    return f"{entity}_source_id"


def _clone_entity_table(
    table: pd.DataFrame,
    *,
    entity: str,
    schema: EntitySchema,
    channels: tuple[str, ...],
    id_multiplier: int,
) -> pd.DataFrame:
    id_columns = _entity_id_columns(schema, entity)
    source_id = support_source_id_column(entity)
    channel_column = support_channel_column(entity)
    clone_index_column = support_clone_index_column(entity)
    primary_id = schema.entity_id_column(entity)

    clones: list[pd.DataFrame] = []
    for clone_index, channel in enumerate(channels):
        clone = table.copy(deep=True)
        clone[source_id] = table[primary_id].to_numpy()
        clone[channel_column] = channel
        clone[clone_index_column] = clone_index
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    result = pd.concat(clones, ignore_index=True)
    if result[primary_id].duplicated().any():
        duplicates = result.loc[result[primary_id].duplicated(), primary_id].unique()
        raise ValueError(
            f"remapped {primary_id!r} values are not unique; id multiplier "
            "is too small. Duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )
    return result


def _clone_link_table(
    table: pd.DataFrame,
    *,
    link_name: str,
    schema: EntitySchema,
    channels: tuple[str, ...],
    id_multiplier: int,
) -> pd.DataFrame:
    links = {link.name: link for link in schema.links}
    link = links[link_name]
    id_columns = (
        schema.entity_id_column(link.left_entity),
        schema.entity_id_column(link.right_entity),
    )
    missing = [column for column in id_columns if column not in table]
    if missing:
        raise ValueError(
            f"link table {link_name!r} is missing ID column(s): {missing}."
        )

    clones: list[pd.DataFrame] = []
    for clone_index, _channel in enumerate(channels):
        clone = table.copy(deep=True)
        for column in id_columns:
            clone[column] = _remap_ids(
                clone[column].to_numpy(),
                clone_index=clone_index,
                id_multiplier=id_multiplier,
            )
        clones.append(clone)
    return pd.concat(clones, ignore_index=True)


def _entity_id_columns(schema: EntitySchema, entity: str) -> tuple[str, ...]:
    if entity == schema.person_entity:
        return (
            schema.person_id_column,
            *(schema.membership_column(group) for group in schema.group_entities),
        )
    return (schema.entity_id_column(entity),)


def _validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, str):
        raise ValueError("support channels must be a sequence of names, not a string.")
    values = tuple(channels)
    if len(values) < 2:
        raise ValueError("PUF support expansion requires at least two channels.")
    bad = [value for value in values if not isinstance(value, str) or not value]
    if bad:
        raise ValueError("support channels must be non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"support channels must be unique, got {values!r}.")
    return values


def _reject_metadata_collisions(
    frame: Frame,
    channels: tuple[str, ...],
) -> None:
    expected = {
        column
        for entity in frame.entities
        for column in (
            support_source_id_column(entity),
            support_channel_column(entity),
            support_clone_index_column(entity),
        )
    }
    existing = {
        column for entity in frame.entities for column in frame.table(entity).columns
    }
    collisions = sorted(expected & existing)
    if collisions:
        raise ValueError(
            "PUF support expansion metadata column(s) already exist: "
            f"{collisions}. The stage should run exactly once."
        )
    if channels[0] != BASE_ASEC_SUPPORT_CHANNEL:
        raise ValueError(
            f"support channels must start with {BASE_ASEC_SUPPORT_CHANNEL!r} "
            "so the baseline ASEC channel keeps the original IDs."
        )
    if PUF_TAX_DETAIL_SUPPORT_CHANNEL not in channels:
        raise ValueError(
            f"support channels must include {PUF_TAX_DETAIL_SUPPORT_CHANNEL!r}."
        )


def _id_multiplier_for_frame(frame: Frame) -> int:
    values = [
        frame.table(entity)[frame.schema.entity_id_column(entity)]
        for entity in frame.entities
    ]
    return _id_multiplier_for_values(*values)


def _id_multiplier_for_values(*values: Sequence[Any]) -> int:
    if not values:
        raise ValueError("at least one ID value sequence is required.")
    max_id = 0
    for sequence in values:
        numeric = pd.to_numeric(pd.Series(sequence), errors="raise").astype("int64")
        if len(numeric):
            max_id = max(max_id, int(numeric.abs().max()))
    return 10 ** max(1, len(str(max_id)))


def _remap_ids(
    ids: Sequence[Any],
    *,
    clone_index: int,
    id_multiplier: int,
) -> np.ndarray:
    values = pd.to_numeric(pd.Series(ids), errors="raise").astype("int64").to_numpy()
    if clone_index == 0:
        return values.copy()
    return values + clone_index * id_multiplier


def _require_entity_name(entity: str) -> None:
    if not isinstance(entity, str) or not entity:
        raise ValueError("entity must be a non-empty string.")


def _tax_unit_model_frame(donor: pd.DataFrame) -> Frame:
    n = len(donor)
    tax_unit = donor.drop(columns=["weight"]).copy()
    tax_unit.insert(0, "tax_unit_id", np.arange(1, n + 1, dtype="int64"))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_tax_unit_id": tax_unit["tax_unit_id"].to_numpy(),
        }
    )
    schema = EntitySchema(group_entities=("tax_unit",))
    return Frame(
        {"person": person, "tax_unit": tax_unit},
        schema,
        {
            "tax_unit": Weights(
                donor["weight"].to_numpy(dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _reject_formula_owned_outputs(
    person_outputs: Sequence[str],
    tax_unit_outputs: Sequence[str],
) -> None:
    requested = set(person_outputs) | set(tax_unit_outputs)
    formula_owned = sorted(requested & PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS)
    if formula_owned:
        raise ValueError(
            "PUF tax-detail support outputs must be PolicyEngine leaf inputs, "
            f"not formula-owned aggregate outputs: {formula_owned}."
        )


def _tax_unit_feature_frame(frame: Frame, columns: Sequence[str]) -> pd.DataFrame:
    tax_unit = frame.table("tax_unit")
    person = frame.table("person")
    result = pd.DataFrame(index=tax_unit.index)
    for column in columns:
        source_column = _predictor_source_column(column)
        if source_column == "filing_status_code":
            source = tax_unit.get("filing_status_input")
            if source is None:
                source = tax_unit.get("filing_status")
            if source is None:
                raise ValueError("tax_unit table lacks filing-status input.")
            result[column] = _filing_status_codes(source)
        elif source_column == "tax_unit_person_count":
            result[column] = (
                person.groupby("person_tax_unit_id", sort=False)
                .size()
                .reindex(tax_unit["tax_unit_id"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
        elif source_column in tax_unit.columns:
            result[column] = pd.to_numeric(
                tax_unit[source_column], errors="coerce"
            ).fillna(0.0)
        else:
            result[column] = _person_tax_unit_sum(frame, source_column)
    return result


def _person_tax_unit_sum(frame: Frame, column: str) -> np.ndarray:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    if column == "dividend_income" and column not in person.columns:
        values = _optional_person(
            person, "non_qualified_dividend_income"
        ) + _optional_person(person, "qualified_dividend_income")
    elif column not in person.columns and column in _PREDICTOR_LEAF_ALIASES:
        values = np.zeros(len(person), dtype=np.float64)
        for leaf in _PREDICTOR_LEAF_ALIASES[column]:
            values += _optional_person(person, leaf)
    else:
        if column not in person.columns:
            raise ValueError(
                f"Cannot build tax-unit predictor {column!r}; no matching "
                "tax_unit column or person column exists."
            )
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0)
    grouped = (
        pd.DataFrame(
            {
                "person_tax_unit_id": person["person_tax_unit_id"],
                column: np.asarray(values, dtype=np.float64),
            }
        )
        .groupby("person_tax_unit_id", sort=False)[column]
        .sum()
    )
    return grouped.reindex(tax_unit["tax_unit_id"]).fillna(0.0).to_numpy()


def _optional_person(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        return np.zeros(len(person), dtype=np.float64)
    return pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()


def _ensure_float_output_column(table: pd.DataFrame, column: str) -> None:
    if column not in table.columns:
        table[column] = 0.0
        return
    table[column] = (
        pd.to_numeric(table[column], errors="coerce").fillna(0.0).astype("float64")
    )


def _snap_to_observed_values(
    values: Sequence[Any],
    observed: Sequence[Any],
) -> np.ndarray:
    """Map continuous model predictions to the nearest observed donor value."""

    value_array = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    observed_values = pd.to_numeric(pd.Series(observed), errors="coerce").fillna(0.0)
    observed_array = np.unique(
        np.rint(observed_values.to_numpy(dtype=np.float64)).clip(min=0.0)
    )
    if len(observed_array) == 0:
        return value_array.to_numpy(dtype=np.float64)
    positions = np.abs(
        value_array.to_numpy(dtype=np.float64)[:, None] - observed_array[None, :]
    ).argmin(axis=1)
    return observed_array[positions]


def _write_person_tax_unit_totals(
    person: pd.DataFrame,
    *,
    mask: pd.Series,
    column: str,
    totals: pd.Series,
    nonnegative: bool,
    fallback_basis_columns: tuple[str, ...] = (),
) -> None:
    row_ids = person.loc[mask, "person_tax_unit_id"]
    current = pd.to_numeric(person.loc[mask, column], errors="coerce").fillna(0.0)
    basis = current.clip(lower=0.0) if nonnegative else current
    basis_sum = basis.groupby(row_ids, sort=False).transform("sum")
    target = row_ids.map(totals).fillna(0.0).to_numpy(dtype=np.float64)
    first = row_ids.groupby(row_ids, sort=False).cumcount() == 0

    allocation = np.zeros(len(row_ids), dtype=np.float64)
    has_basis = basis_sum.to_numpy(dtype=np.float64) != 0.0
    allocation[has_basis] = (
        target[has_basis]
        * basis.to_numpy(dtype=np.float64)[has_basis]
        / basis_sum.to_numpy(dtype=np.float64)[has_basis]
    )
    unallocated = ~has_basis
    if fallback_basis_columns:
        fallback = np.zeros(len(row_ids), dtype=np.float64)
        for basis_column in fallback_basis_columns:
            if basis_column not in person.columns:
                continue
            fallback += (
                pd.to_numeric(person.loc[mask, basis_column], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=np.float64)
            )
        fallback_sum = (
            pd.Series(fallback, index=row_ids.index)
            .groupby(row_ids, sort=False)
            .transform("sum")
            .to_numpy(dtype=np.float64)
        )
        use_fallback = unallocated & (fallback_sum > 0.0)
        allocation[use_fallback] = (
            target[use_fallback] * fallback[use_fallback] / fallback_sum[use_fallback]
        )
        unallocated &= ~use_fallback
    allocation[unallocated & first.to_numpy()] = target[unallocated & first.to_numpy()]
    person.loc[mask, column] = allocation


def _weighted_positive_rate(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = (
        pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    total_weight = float(numeric_weights.sum())
    if total_weight <= 0.0:
        return 0.0
    return float((numeric_weights * (numeric_values > 0.0)).sum() / total_weight)


def _sparsify_tax_unit_person_output_to_donor_positive_rate(
    tables: Mapping[str, pd.DataFrame],
    *,
    column: str,
    donor_positive_rate: float,
    household_weights: np.ndarray,
    person_channel: str,
    tax_unit_channel: str,
) -> None:
    positive_rate = float(np.clip(donor_positive_rate, 0.0, 1.0))
    person = tables["person"]
    household = tables["household"]
    tax_unit = tables["tax_unit"]
    if len(household_weights) != len(household):
        raise ValueError(
            "household_weights must align with household rows, got "
            f"{len(household_weights)} weights for {len(household)} households."
        )
    household_weight = pd.Series(
        np.asarray(household_weights, dtype=np.float64),
        index=household["household_id"],
    )
    tax_unit_household_id = (
        person.groupby("person_tax_unit_id", sort=False)["person_household_id"]
        .first()
        .astype("int64")
    )
    tax_unit_weight = tax_unit_household_id.map(household_weight).fillna(0.0)
    tax_unit_amount = (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .groupby(person["person_tax_unit_id"], sort=False)
        .sum()
    )

    for channel in tax_unit[tax_unit_channel].dropna().unique():
        channel_tax_unit_ids = tax_unit.loc[
            tax_unit[tax_unit_channel] == channel,
            "tax_unit_id",
        ]
        amounts = tax_unit_amount.reindex(channel_tax_unit_ids).fillna(0.0)
        weights = tax_unit_weight.reindex(channel_tax_unit_ids).fillna(0.0)
        positive = amounts > 0.0
        positive_weight = float(weights[positive].sum())
        desired_positive_weight = positive_rate * float(weights.sum())
        if positive_weight <= desired_positive_weight or positive_weight <= 0.0:
            continue

        ranked = (
            pd.DataFrame({"amount": amounts[positive], "weight": weights[positive]})
            .sort_values("amount", ascending=False)
            .copy()
        )
        cumulative = ranked["weight"].cumsum()
        keep = cumulative <= desired_positive_weight
        if not keep.any() and len(keep) > 0:
            keep.iloc[0] = True
        kept_ids = set(ranked.index[keep])

        sparse_totals = amounts.copy()
        sparse_totals.loc[positive & ~amounts.index.isin(kept_ids)] = 0.0
        original_total = float((amounts * weights).sum())
        sparse_total = float((sparse_totals * weights).sum())
        if original_total != 0.0 and sparse_total != 0.0:
            sparse_totals *= original_total / sparse_total

        mask = person[person_channel] == channel
        _write_person_tax_unit_totals(
            person,
            mask=mask,
            column=column,
            totals=sparse_totals,
            nonnegative=column in _PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS,
        )


def _reconcile_puf_social_security_components(
    predictions: pd.DataFrame,
    person: pd.DataFrame,
    *,
    person_channel: str,
    tax_unit_ids: np.ndarray,
    requested_components: Sequence[str],
) -> None:
    components = tuple(
        component
        for component in PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS
        if component in requested_components
    )
    if not components:
        return
    if set(components) != set(PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS):
        raise ValueError(
            "PUF Social Security support must request all Social Security "
            f"component leaves, got {components!r}."
        )
    missing = [component for component in components if component not in predictions]
    if missing:
        raise ValueError(
            "PUF Social Security predictions are missing component column(s): "
            f"{missing}."
        )

    puf_person = person.loc[
        person[person_channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ["person_tax_unit_id"],
    ].copy()
    for component in components:
        if component in person.columns:
            puf_person[component] = (
                pd.to_numeric(
                    person.loc[
                        person[person_channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL,
                        component,
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=np.float64)
            )
        else:
            puf_person[component] = np.zeros(
                len(puf_person),
                dtype=np.float64,
            )
    basis = (
        puf_person.groupby("person_tax_unit_id", sort=False)[list(components)]
        .sum()
        .reindex(tax_unit_ids)
        .fillna(0.0)
    )
    basis_values = basis.to_numpy(dtype=np.float64)
    basis_sums = basis_values.sum(axis=1)
    shares = np.zeros_like(basis_values)
    has_basis = basis_sums > 0
    shares[has_basis] = basis_values[has_basis] / basis_sums[has_basis, np.newaxis]
    shares[~has_basis] = 1.0 / len(components)

    total = (
        predictions.loc[:, list(components)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .sum(axis=1)
        .to_numpy(dtype=np.float64)
    )
    for index, component in enumerate(components):
        predictions[component] = shares[:, index] * total


def _person_source_values(
    arrays: Mapping[str, Sequence[Any]],
    output: str,
) -> np.ndarray | None:
    if output in arrays:
        return _numeric_array(arrays[output])
    source_aliases = {
        "employment_income_before_lsr": ("employment_income",),
        "self_employment_income_before_lsr": ("self_employment_income",),
        "long_term_capital_gains_before_response": ("long_term_capital_gains",),
        "taxable_private_pension_income": ("taxable_pension_income",),
        # The PUF observes realized IRA deductions; at baseline the engine's
        # desired-contribution leaf equals the realized amount.
        "traditional_ira_contributions_desired": ("traditional_ira_contributions",),
    }
    for source in source_aliases.get(output, ()):
        if source in arrays:
            return _numeric_array(arrays[source])
    if output == "partnership_income" and "partnership_s_corp_income" in arrays:
        return _numeric_array(arrays["partnership_s_corp_income"])
    if output == "s_corp_income" and "partnership_s_corp_income" in arrays:
        return np.zeros_like(_numeric_array(arrays["partnership_s_corp_income"]))
    if output == "partnership_self_employment_net_earnings" and (
        "partnership_se_income" in arrays
    ):
        return _numeric_array(arrays["partnership_se_income"])
    if output == "partnership_income" and {"E25980", "E25960"}.issubset(arrays):
        return _numeric_array(arrays["E25980"]) - _numeric_array(arrays["E25960"])
    if output == "s_corp_income" and {"E26190", "E26180"}.issubset(arrays):
        return _numeric_array(arrays["E26190"]) - _numeric_array(arrays["E26180"])
    if output == "partnership_self_employment_net_earnings" and {
        "E25960",
        "E26180",
    }.issubset(arrays):
        return _numeric_array(arrays["E25960"]) + _numeric_array(arrays["E26180"])
    if output in _PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS and "E17500" in arrays:
        return (
            _numeric_array(arrays["E17500"])
            * _PUF_MEDICAL_EXPENSE_CATEGORY_BREAKDOWNS[output]
        )
    if output == "unemployment_compensation" and (
        "taxable_unemployment_compensation" in arrays
    ):
        return _numeric_array(arrays["taxable_unemployment_compensation"])
    if output in PUF_TAX_DETAIL_SOCIAL_SECURITY_COMPONENT_OUTPUTS:
        if output != "social_security_retirement":
            for source in ("social_security", "total_social_security", "E02400"):
                if source in arrays:
                    return np.zeros_like(_numeric_array(arrays[source]))
            return None
        for source in ("social_security", "total_social_security", "E02400"):
            if source in arrays:
                return _numeric_array(arrays[source])
    return None


def _add_predictor_aliases(
    table: pd.DataFrame,
    predictors: Sequence[str],
) -> None:
    for predictor in predictors:
        if predictor in table.columns:
            continue
        source = _predictor_source_column(predictor)
        if source in table.columns:
            table[predictor] = table[source]
        elif source == "dividend_income" and {
            "qualified_dividend_income",
            "non_qualified_dividend_income",
        }.issubset(table.columns):
            table[predictor] = pd.to_numeric(
                table["qualified_dividend_income"],
                errors="coerce",
            ).fillna(0.0) + pd.to_numeric(
                table["non_qualified_dividend_income"],
                errors="coerce",
            ).fillna(0.0)
        elif source in _PREDICTOR_LEAF_ALIASES:
            pieces = [
                pd.to_numeric(table[leaf], errors="coerce").fillna(0.0)
                for leaf in _PREDICTOR_LEAF_ALIASES[source]
                if leaf in table.columns
            ]
            if pieces:
                table[predictor] = sum(pieces)


def _predictor_source_column(column: str) -> str:
    if column.startswith(_PUF_PREDICTOR_PREFIX):
        return column.removeprefix(_PUF_PREDICTOR_PREFIX)
    return column


def _tax_unit_source_values(
    arrays: Mapping[str, Sequence[Any]],
    tax_unit_id: np.ndarray,
    output: str,
    grouped_person: pd.DataFrame,
) -> np.ndarray | None:
    if output in arrays:
        values = _numeric_array(arrays[output])
        if len(values) == len(tax_unit_id):
            return values
    if (
        output == "state_withheld_income_tax"
        and "state_and_local_sales_or_income_tax" in arrays
    ):
        return _numeric_array(arrays["state_and_local_sales_or_income_tax"])
    if output == "interest_deduction":
        if "home_mortgage_interest" in grouped_person:
            return (
                grouped_person["home_mortgage_interest"]
                .reindex(tax_unit_id)
                .fillna(0.0)
                .to_numpy()
            )
        pieces = [
            _numeric_array(arrays[name])
            for name in (
                "first_home_mortgage_interest",
                "second_home_mortgage_interest",
            )
            if name in arrays
        ]
        if pieces:
            return np.sum(np.column_stack(pieces), axis=1)
    return None


def _filing_status_codes(values: Sequence[Any]) -> np.ndarray:
    decoded = pd.Series(values).map(_decode_status).str.upper()
    return decoded.map(_FILING_STATUS_CODES).fillna(0.0).to_numpy(dtype=np.float64)


def _decode_status(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _numeric_array(values: Any) -> np.ndarray:
    if values is None:
        raise ValueError("Cannot convert missing array to numeric values.")
    if np.isscalar(values):
        return np.asarray(values, dtype=np.float64)
    return (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _require_array_columns(
    arrays: Mapping[str, Sequence[Any]],
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column not in arrays]
    if missing:
        raise ValueError(f"{label} missing required array(s): {missing}.")
