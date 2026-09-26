"""Pure complete-unit transport at the native SPM engine boundary.

The retained source owner must validate before and after using this projection.
Detached Frames and the result of this function are not source capabilities.
No source roles, group membership, weights or annual scope are inferred here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.spm_input_contract import (
    OUTSIDE,
    ROLE_INPUT,
    UNIVERSE_INPUT,
    UNIVERSE_STATUSES,
)
from microcosm.frame import Frame

from . import current_survey_health_coverage as attachment
from . import support_provenance as provenance
from .current_survey_hours_source import _table_seal


def _require(condition, reason):
    if not condition:
        raise ValueError("SPM_PROJECTION_" + reason)


def _ids(table, column):
    _require(
        type(table) is pd.DataFrame
        and table.columns.is_unique
        and column in table
        and table[column].dtype == np.dtype("int64"),
        "IDENTIFIER_TYPE:" + column,
    )
    return table[column]


def _entity_axis(frame, entity):
    # Frame only exposes convenience attributes for selected entities. Kernel
    # contexts use lightweight table views, while retained owners use Frame.
    table = frame.table(entity) if isinstance(frame, Frame) else getattr(frame, entity)
    ids = _ids(table, entity + "_id")
    _require(ids.is_unique and len(ids) > 0, "ENTITY_AXIS:" + entity)
    return table, pd.Index(ids.to_numpy(), name=entity + "_id")


def _group_transport(source_frame, receiving, entity):
    """Prove a bijection of complete original groups and their two clones."""
    source_table, source_ids = _entity_axis(source_frame, entity)
    target_table, target_ids = _entity_axis(receiving, entity)
    original = _ids(target_table, provenance.support_source_id_column(entity))
    clones = _ids(target_table, provenance.support_clone_index_column(entity))
    native = _ids(target_table, provenance.spine_source_id_column(entity))
    original_native = _ids(source_table, provenance.spine_source_id_column(entity))
    channel_name = provenance.support_channel_column(entity)
    _require(channel_name in source_table and channel_name in target_table, "CHANNEL")
    _require(
        len(target_ids) == 2 * len(source_ids)
        and set(original) == set(source_ids)
        and np.isin(clones, (0, 1)).all()
        and pd.MultiIndex.from_arrays((original, clones)).is_unique,
        "GROUP_CLONE_ROSTER:" + entity,
    )
    native_lookup = pd.Series(original_native.to_numpy(), index=source_ids)
    channel_lookup = pd.Series(source_table[channel_name].to_numpy(), index=source_ids)
    _require(
        channel_lookup.isin(("acs", "asec")).all()
        and np.array_equal(
            native.to_numpy(), native_lookup.reindex(original).to_numpy()
        )
        and np.array_equal(
            target_table[channel_name].to_numpy(),
            channel_lookup.reindex(original).to_numpy(),
        ),
        "GROUP_SOURCE_IDENTITY:" + entity,
    )
    source_people = source_frame.person
    target_people = receiving.person
    link = "person_" + entity + "_id"
    source_link = _ids(source_people, link)
    target_link = _ids(target_people, link)
    _require(
        set(source_link) == set(source_ids) and set(target_link) == set(target_ids),
        "GROUP_MEMBERSHIP_ROSTER:" + entity,
    )
    source_groups = {}
    for pid, group in zip(source_people.person_id, source_link, strict=True):
        source_groups.setdefault(int(group), set()).add(int(pid))
    _require(
        np.array_equal(
            channel_lookup.reindex(source_link).to_numpy(),
            source_people[provenance.support_channel_column("person")].to_numpy(),
        ),
        "GROUP_PERSON_SOURCE:" + entity,
    )
    target_groups = {}
    for group, pid, clone in zip(
        target_link,
        target_people[provenance.support_source_id_column("person")],
        target_people[provenance.support_clone_index_column("person")],
        strict=True,
    ):
        target_groups.setdefault(int(group), set()).add((int(pid), int(clone)))
    mapping = []
    for final_id, original_id, clone in zip(target_ids, original, clones, strict=True):
        expected = {(pid, int(clone)) for pid in source_groups[int(original_id)]}
        _require(target_groups[int(final_id)] == expected, "GROUP_MEMBERSHIP:" + entity)
        mapping.append((int(final_id), int(original_id), int(clone)))
    return tuple(mapping)


@dataclass(frozen=True, eq=False)
class SpmInputProjection:
    """New engine columns plus the original nullable observation and mapping.

    OUTSIDE placeholders are storage representations, not observed False/True.
    Arrays are defensive copies; this result is descriptive, not an issuer.
    """

    columns: Mapping[tuple[str, str], pd.Series]
    nullable_source_roles: pd.Series
    placeholder_person_ids: tuple[int, ...]
    unit_mapping: tuple[tuple[int, int, int], ...]
    year: int


def spm_projection_seal(result):
    """Pure detached snapshot at each entity grain; grants no source authority."""
    _require(type(result) is SpmInputProjection, "RESULT_TYPE")
    keys = (("person", ROLE_INPUT), ("spm_unit", UNIVERSE_INPUT))
    _require(
        type(result.columns) is MappingProxyType and set(result.columns) == set(keys),
        "RESULT_COLUMNS",
    )
    columns = []
    for entity, name in keys:
        values = result.columns[entity, name]
        _require(
            type(values) is pd.Series
            and values.index.dtype == np.dtype("int64")
            and values.index.name == entity + "_id"
            and values.index.is_unique
            and values.name == name,
            "RESULT_AXIS",
        )
        _require(
            values.dtype == np.dtype("bool")
            if entity == "person"
            else isinstance(values.dtype, pd.StringDtype)
            and values.isin(UNIVERSE_STATUSES).all(),
            "RESULT_DTYPE",
        )
        columns.append((entity, name, _table_seal(values.to_frame())))
    nullable = result.nullable_source_roles
    _require(
        type(nullable) is pd.Series
        and nullable.dtype == pd.BooleanDtype()
        and nullable.index.dtype == np.dtype("int64")
        and nullable.index.name == "person_id"
        and nullable.index.is_unique
        and type(nullable.name) in (type(None), str, int),
        "RESULT_SOURCE_ROLES",
    )
    placeholders = result.placeholder_person_ids
    _require(
        type(placeholders) is tuple
        and all(type(pid) is int for pid in placeholders)
        and len(set(placeholders)) == len(placeholders)
        and set(placeholders) <= set(result.columns["person", ROLE_INPUT].index),
        "RESULT_PLACEHOLDERS",
    )
    mapping = result.unit_mapping
    _require(
        type(mapping) is tuple
        and all(
            type(row) is tuple
            and len(row) == 3
            and all(type(v) is int for v in row)
            and row[2] in (0, 1)
            for row in mapping
        )
        and tuple(row[0] for row in mapping)
        == tuple(result.columns["spm_unit", UNIVERSE_INPUT].index),
        "RESULT_MAPPING",
    )
    _require(type(result.year) is int and result.year == 2024, "RESULT_YEAR")
    return (
        tuple(columns),
        (type(nullable.name), nullable.name),
        _table_seal(nullable.to_frame()),
        placeholders,
        mapping,
        result.year,
    )


def validate_spm_projection(result, revalidate):
    """Protect a pure projection across a caller's final validation callback.

    The caller remains responsible for choosing a real source-owner validator.
    This function proves result preservation only, and issues no capability.
    """
    expected = spm_projection_seal(result)
    revalidate()
    # No callback or I/O may follow this detached-output comparison.
    try:
        unchanged = spm_projection_seal(result) == expected
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError("SPM_PROJECTION_RESULT_CHANGED") from error
    _require(unchanged, "RESULT_CHANGED")
    return result


def project_spm_inputs(
    source_frame,
    origins,
    roles,
    unit_status,
    receiving,
    *,
    source_year,
    year,
    outside_role_placeholder,
):
    """Transport checked source inputs without mutating either population.

    This bridge supports the current 2024-income source contract only. Missing
    roles refuse unless the complete source unit is explicitly OUTSIDE. Known
    roles in UNRESOLVED units stay UNRESOLVED for the country's refusal gate.
    """
    _require(
        type(source_year) is int and type(year) is int and source_year == year == 2024,
        "ANNUAL_SCOPE",
    )
    _require(type(outside_role_placeholder) is bool, "OUTSIDE_REPRESENTATION")
    source_people, source_person_ids = _entity_axis(source_frame, "person")
    target_people, _ = _entity_axis(receiving, "person")
    _, source_unit_ids = _entity_axis(source_frame, "spm_unit")
    _require(
        type(origins) is pd.DataFrame
        and origins.index.equals(source_person_ids)
        and origins.index.dtype == np.dtype("int64")
        and origins.index.name == "person_id"
        and origins.columns.is_unique
        and {"source", "native_person_id", "source_year", "survey_year"}
        <= set(origins),
        "ORIGIN_AXIS",
    )
    for column in ("native_person_id", "source_year", "survey_year"):
        _ids(origins, column)
    _require(
        origins.source.isin(("acs", "asec")).all()
        and origins.source_year.eq(source_year).all()
        and origins.survey_year.eq(
            origins.source.map({"acs": 2024, "asec": 2025})
        ).all()
        and np.array_equal(
            origins.native_person_id.to_numpy(),
            _ids(source_people, provenance.spine_source_id_column("person")).to_numpy(),
        )
        and np.array_equal(
            origins.source.to_numpy(),
            source_people[provenance.support_channel_column("person")].to_numpy(),
        ),
        "ORIGIN_IDENTITY",
    )
    _require(
        type(roles) is pd.Series
        and roles.index.equals(source_person_ids)
        and roles.index.dtype == np.dtype("int64")
        and roles.index.name == "person_id"
        and roles.dtype == pd.BooleanDtype(),
        "ROLE_AXIS",
    )
    _require(
        type(unit_status) is pd.Series
        and unit_status.index.equals(source_unit_ids)
        and unit_status.index.dtype == np.dtype("int64")
        and unit_status.index.name == "spm_unit_id"
        and unit_status.isin(UNIVERSE_STATUSES).all(),
        "STATUS_AXIS",
    )
    target_units, _ = _entity_axis(receiving, "spm_unit")
    _require(UNIVERSE_INPUT not in target_units, "ATTACH_OWNERSHIP_COLLISION")
    # Existing person transport proves all source identities and both clones.
    role_columns = attachment.attach_columns(
        origins, receiving, roles.to_frame(ROLE_INPUT)
    )
    unit_mapping = _group_transport(source_frame, receiving, "spm_unit")
    _group_transport(source_frame, receiving, "household")
    source_spm = source_people.person_spm_unit_id
    source_household = source_people.person_household_id
    roster = pd.DataFrame(
        {
            "unit": source_spm.to_numpy(),
            "household": source_household.to_numpy(),
            "source": origins.source.to_numpy(),
        }
    )
    _require(
        roster.groupby("unit", sort=False).household.nunique().eq(1).all()
        and roster.groupby("unit", sort=False).source.nunique().eq(1).all()
        and roster.groupby("household", sort=False).source.nunique().eq(1).all(),
        "SOURCE_GROUP_MEMBERSHIP",
    )
    target_status = pd.Series(
        [unit_status.at[original] for _, original, _ in unit_mapping],
        index=pd.Index([final for final, _, _ in unit_mapping], name="spm_unit_id"),
        name=UNIVERSE_INPUT,
        dtype=pd.StringDtype(storage="python", na_value=pd.NA),
    )
    person_status = target_status.reindex(target_people.person_spm_unit_id).to_numpy()
    nullable = role_columns["person", ROLE_INPUT]
    missing = nullable.isna().to_numpy()
    _require(
        np.all(~missing | (person_status == OUTSIDE)), "NULL_ROLE_REQUIRES_OUTSIDE"
    )
    columns = {
        ("person", ROLE_INPUT): nullable.fillna(outside_role_placeholder).astype(bool),
        ("spm_unit", UNIVERSE_INPUT): target_status,
    }
    return SpmInputProjection(
        MappingProxyType(columns),
        roles.copy(deep=True),
        tuple(int(pid) for pid in nullable.index[missing]),
        unit_mapping,
        year,
    )
