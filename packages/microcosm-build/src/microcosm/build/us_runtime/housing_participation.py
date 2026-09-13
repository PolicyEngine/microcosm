"""Pure source-known housing participation and explicit assisted-family routing.

Observation classification belongs to the existing ASEC source owner. These
helpers neither establish source authority nor infer annual participation. A
caller must bind the stated modeling assumptions and resolve unknown receipt
before exporting country Boolean inputs. No benefit dollars are read or written.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, Frame

from .asec_housing_status import DERIVED_COLUMNS, _codebook
from .asec_housing_status_source import ATTACHED_COLUMNS

HOUSING_PARTICIPATION_ASSUMPTIONS = (
    ("A1", "The household head's SPM unit is the modeled assisted family."),
    (
        "A2",
        "Interview-time participation is carried through the preceding income year.",
    ),
    (
        "A3",
        "Public-housing and reduced-rent reports represent the modeled HUD assistance family.",
    ),
    (
        "A4",
        "ACS group quarters are structurally excluded from this modeled household-assistance family; this is not observed nonreceipt or a general eligibility claim.",
    ),
)


class HousingParticipationError(ValueError):
    """A value-free refusal of incomplete or contradictory participation inputs."""


@dataclass(frozen=True)
class ObservedParticipation:
    """Nullable receipt plus detached evidence retaining each unresolved state."""

    receipt: pd.Series
    evidence: pd.DataFrame


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HousingParticipationError(reason)


def _integer_values(values: pd.Series, reason: str) -> np.ndarray:
    _require(
        pd.api.types.is_integer_dtype(values.dtype)
        and not pd.api.types.is_bool_dtype(values.dtype)
        and not values.isna().any(),
        reason,
    )
    return values.to_numpy()


def _native_ids(values: pd.Series, reason: str, *, unique: bool) -> pd.Index:
    _integer_values(values, reason)
    _require(not unique or not values.duplicated().any(), reason)
    return pd.Index(values.to_numpy(copy=True), name=values.name)


def observed_participation(status_table: pd.DataFrame) -> ObservedParticipation:
    """Project only source-known receipt/nonreceipt, preserving all other states.

    Accept the classifier's column names or their normal attached aliases,
    including graph-promoted integer widths. Unknown, not-in-universe and
    conflicting observations all remain missing receipt, with distinct source
    evidence in the returned table. Group quarters and allocation quality are
    retained as independent axes; neither changes a valid source response here.
    This checks the supplied interpretation, not source authentication.
    """
    _require(
        isinstance(status_table, pd.DataFrame)
        and status_table.columns.is_unique
        and "household_id" in status_table,
        "STATUS_TABLE",
    )
    ids = _native_ids(status_table["household_id"], "HOUSEHOLD_ID", unique=True)
    aliases = dict(zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True))
    columns = {}
    for name, alias in aliases.items():
        _require(
            not (name in status_table and alias in status_table), "AMBIGUOUS_STATUS"
        )
        if name in status_table:
            columns[name] = status_table[name]
        elif alias in status_table:
            columns[name] = status_table[alias]
    _require({"status", "receipt_valid"} <= columns.keys(), "STATUS_COLUMNS")
    values = {
        name: _integer_values(column, "STATUS_INTEGER")
        for name, column in columns.items()
    }
    codebook = _codebook()
    status_codes = codebook["status"]
    status = values["status"]
    valid = values["receipt_valid"]
    _require(np.isin(status, tuple(status_codes.values())).all(), "STATUS_DOMAIN")
    _require(np.isin(valid, (0, 1)).all(), "VALIDITY_DOMAIN")
    known = np.isin(status, (status_codes["receipt"], status_codes["nonreceipt"]))
    _require(np.array_equal(valid, known), "CONTRADICTORY_KNOWNNESS")
    for name in ("route", "group_quarters"):
        if name in values:
            _require(
                np.isin(values[name], tuple(codebook[name].values())).all(),
                "EVIDENCE_DOMAIN",
            )
    for name in ("public_quality", "lower_quality"):
        if name in values:
            _require(
                np.isin(values[name], tuple(codebook["quality"].values())).all(),
                "QUALITY_DOMAIN",
            )
    for name, value in values.items():
        if name.endswith("_valid") or name.endswith("_zero_origin"):
            _require(np.isin(value, (0, 1)).all(), "INDICATOR_DOMAIN")
    for name, codebook_name in (
        ("conflicts", "conflict_bits"),
        ("unknown_reasons", "unknown_reason_bits"),
    ):
        if name in values:
            value = values[name]
            maximum = sum(codebook[codebook_name].values())
            _require(((value >= 0) & (value <= maximum)).all(), "REASON_DOMAIN")
    if "conflicts" in values:
        _require(
            np.array_equal(values["conflicts"] > 0, status == status_codes["conflict"]),
            "CONTRADICTORY_CONFLICT",
        )
    if "unknown_reasons" in values:
        unresolved = status == status_codes["unknown"]
        _require(
            (values["unknown_reasons"][unresolved] > 0).all()
            and (
                values["unknown_reasons"][
                    known | (status == status_codes["not_in_universe"])
                ]
                == 0
            ).all(),
            "CONTRADICTORY_UNKNOWN",
        )
    if "route" in values:
        routes = codebook["route"]
        route = values["route"]
        _require(
            (
                (
                    (status == status_codes["receipt"])
                    & np.isin(route, (routes["public_housing"], routes["lower_rent"]))
                )
                | (
                    (status == status_codes["nonreceipt"])
                    & (route == routes["both_negative"])
                )
                | (
                    (status == status_codes["not_in_universe"])
                    & (route == routes["not_in_universe"])
                )
                | (
                    np.isin(status, (status_codes["unknown"], status_codes["conflict"]))
                    & (route == routes["unresolved"])
                )
            ).all(),
            "CONTRADICTORY_ROUTE",
        )
    receipt = pd.Series(
        pd.NA, index=ids, dtype="boolean", name="observed_housing_participation"
    )
    receipt.iloc[np.flatnonzero(known)] = status[known] == status_codes["receipt"]
    evidence = status_table.drop(columns="household_id").copy(deep=True)
    evidence.index = ids.copy(deep=True)
    evidence.columns = evidence.columns.copy(deep=True)
    return ObservedParticipation(receipt=receipt, evidence=evidence)


def route_participation(frame: Frame, household_receipt: pd.Series) -> pd.DataFrame:
    """Route resolved modeled participation once, to each actual head's SPM unit.

    Household receipt must be explicitly keyed by the exact current household
    IDs. DataFrame index labels and person order have no routing meaning. This
    function makes no eligibility or payment calculation and does not mutate
    population inputs. The caller retains source knownness and the A1/A2/A3
    interpretation separately from these country-model Boolean leaves. When a
    qualifier explicitly supplies ``housing_participation_universe``, a
    ``group_quarters`` row may lack a head only with already-resolved false
    receipt. A4 is a modeled exclusion, never a source-observation override.
    """
    _require(isinstance(frame, Frame) and frame.schema == US_SCHEMA, "US_FRAME")
    try:
        # Revalidation normalizes its instance's strata attribute. Validate a
        # shallow shell so even that attribute on the caller remains untouched;
        # the normal kernel checks inspect the original tables and weights.
        copy(frame).revalidate()
    except (TypeError, ValueError, KeyError, AssertionError):
        raise HousingParticipationError("FRAME_IDENTITY") from None
    for entity in frame.entities:
        table = frame.table(entity)
        column = frame.schema.entity_id_column(entity)
        _require(table.columns.is_unique and column in table, "ENTITY_COLUMNS")
        _native_ids(table[column], "ENTITY_ID", unique=True)
    person = frame.table("person")
    household_ids = _native_ids(
        frame.table("household")["household_id"], "HOUSEHOLD_ID", unique=True
    )
    unit_ids = _native_ids(
        frame.table("spm_unit")["spm_unit_id"], "SPM_ID", unique=True
    )
    for name in ("person_household_id", "person_spm_unit_id"):
        _native_ids(person[name], "MEMBERSHIP_ID", unique=False)
    _require(isinstance(household_receipt, pd.Series), "RECEIPT_SERIES")
    _native_ids(pd.Series(household_receipt.index), "RECEIPT_ID", unique=True)
    _require(
        len(household_receipt) == len(household_ids)
        and household_receipt.index.isin(household_ids).all(),
        "RECEIPT_HOUSEHOLD_COVERAGE",
    )
    _require(
        pd.api.types.is_bool_dtype(household_receipt.dtype)
        and not household_receipt.isna().any(),
        "UNRESOLVED_OR_NONBOOLEAN_RECEIPT",
    )
    _require("is_household_head" in person, "HOUSEHOLD_HEAD_REQUIRED")
    head = person["is_household_head"]
    _require(
        pd.api.types.is_bool_dtype(head.dtype) and not head.isna().any(),
        "HOUSEHOLD_HEAD_BOOLEAN",
    )
    heads = person.iloc[np.flatnonzero(head.to_numpy(dtype=bool))]
    head_ids = pd.Index(heads["person_household_id"].to_numpy())
    household = frame.table("household")
    group_quarters = np.zeros(len(household), dtype=bool)
    if "housing_participation_universe" in household:
        universe = household["housing_participation_universe"]
        _require(
            universe.isin(("occupied_housing_unit", "group_quarters")).all(),
            "MODELED_HOUSING_UNIVERSE",
        )
        group_quarters = universe.eq("group_quarters").to_numpy(dtype=bool)
        _require(
            not household_receipt.reindex(household_ids[group_quarters]).any(),
            "POSITIVE_GROUP_QUARTERS_RECEIPT",
        )
    required_heads = household_ids[~group_quarters]
    _require(
        head_ids.is_unique
        and head_ids.isin(household_ids).all()
        and required_heads.isin(head_ids).all(),
        "EXACTLY_ONE_HOUSEHOLD_HEAD",
    )
    _require(
        person.groupby("person_spm_unit_id")["person_household_id"]
        .nunique()
        .eq(1)
        .all(),
        "SPM_HOUSEHOLD_NESTING",
    )
    head_receipt = household_receipt.reindex(head_ids).to_numpy(dtype=bool)
    awarded = unit_ids.isin(
        heads.iloc[np.flatnonzero(head_receipt)]["person_spm_unit_id"]
    )
    return pd.DataFrame(
        {
            "receives_housing_assistance": awarded,
            "takes_up_housing_assistance_if_eligible": awarded.copy(),
        },
        index=unit_ids,
    )
