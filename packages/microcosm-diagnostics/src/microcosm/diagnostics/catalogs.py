"""Hierarchy catalogs: schema 8 diagnostics to schema 9 and back.

Producers assemble schema 8 (:class:`CalibrationDiagnosticsV8`), where every
target row repeats its provider, category, geography and dimension labels.
Microcosm checks that the facts inside one target agree, but nothing checks
across targets, so one geography or dimension value can arrive with two
labels from two publishers. Schema 9 (:class:`CalibrationDiagnosticsV9`)
declares each label once in ``hierarchy_catalogs`` and has the rows refer to
it by ID.

:func:`catalog_calibration_diagnostics` builds the catalogs from the rows and
refuses (:class:`HierarchyLabelConflictError`) when the rows give one
identifier two labels, or one geography two levels, unless a
:class:`HierarchyLabelChoices` names the label to use. A choice applies only
where the rows disagree; an identifier the rows label one way keeps that
label. :func:`expand_calibration_diagnostics` rebuilds the schema 8 document.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from microcosm.diagnostics.schema import (
    CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION,
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    CalibrationDiagnosticsV8,
    CalibrationDiagnosticsV9,
)

__all__ = [
    "HierarchyLabelChoices",
    "HierarchyLabelConflict",
    "HierarchyLabelConflictError",
    "catalog_calibration_diagnostics",
    "expand_calibration_diagnostics",
]

_SHOWN_CONFLICTS = 10
_LOGGER = logging.getLogger(__name__)


def _frozen_mapping(value: Mapping) -> Mapping:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class HierarchyLabelChoices:
    """Labels chosen for identifiers that a country's sources spell differently.

    ``geographies`` maps a geography ID to its label. ``dimension_values`` maps
    a ``(dimension ID, value ID)`` pair to its label; when
    ``dimension_value_spellings`` lists the spellings a value's choice was
    reviewed for, the choice applies only while every spelling the rows use is
    among them, so a new spelling is still refused (as Chronicle's
    ``REVIEWED_VALUE_LABEL_VARIANTS`` does, chronicle#377). A dimension listed in
    ``geography_dimensions`` has geography IDs as its values (Chronicle writes
    them in lower case, for example ``e06000001``); when the rows label such a
    value two ways, it takes the label of the geography with that ID, compared
    without case: the payload's own geography label, else ``geographies``.
    """

    geographies: Mapping[str, str] = field(default_factory=dict)
    dimension_values: Mapping[tuple[str, str], str] = field(default_factory=dict)
    geography_dimensions: frozenset[str] = frozenset()
    dimension_value_spellings: Mapping[tuple[str, str], frozenset[str]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "geographies", _frozen_mapping(self.geographies))
        object.__setattr__(
            self, "dimension_values", _frozen_mapping(self.dimension_values)
        )
        object.__setattr__(
            self, "geography_dimensions", frozenset(self.geography_dimensions)
        )
        object.__setattr__(
            self,
            "dimension_value_spellings",
            _frozen_mapping(
                {
                    key: frozenset(spellings)
                    for key, spellings in self.dimension_value_spellings.items()
                }
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchyLabelChoices):
            return NotImplemented
        return (
            dict(self.geographies) == dict(other.geographies)
            and dict(self.dimension_values) == dict(other.dimension_values)
            and self.geography_dimensions == other.geography_dimensions
            and dict(self.dimension_value_spellings)
            == dict(other.dimension_value_spellings)
        )

    __hash__ = None  # type: ignore[assignment]

    def merged(self, other: HierarchyLabelChoices) -> HierarchyLabelChoices:
        """These choices with ``other``'s added; ``other`` wins on a shared key."""

        return HierarchyLabelChoices(
            geographies={**self.geographies, **other.geographies},
            dimension_values={**self.dimension_values, **other.dimension_values},
            geography_dimensions=self.geography_dimensions | other.geography_dimensions,
            dimension_value_spellings={
                **self.dimension_value_spellings,
                **other.dimension_value_spellings,
            },
        )


@dataclass(frozen=True)
class HierarchyLabelConflict:
    """One identifier the rows label (or place at a level) more than one way."""

    #: What disagrees, for example ``geography 0400000US23`` or
    #: ``dimension age value 85_plus``.
    subject: str
    #: Each distinct value, with the first target that carries it, in row order.
    spellings: tuple[tuple[str, str], ...]

    def describe(self) -> str:
        spellings = ", ".join(
            f"{value!r} (first target {target})" for value, target in self.spellings
        )
        return f"{self.subject}: {spellings}"


class HierarchyLabelConflictError(ValueError):
    """The rows give identifiers more than one label, and no label is chosen."""

    def __init__(self, conflicts: Iterable[HierarchyLabelConflict]) -> None:
        self.conflicts = tuple(conflicts)
        shown = "; ".join(
            conflict.describe() for conflict in self.conflicts[:_SHOWN_CONFLICTS]
        )
        more = len(self.conflicts) - _SHOWN_CONFLICTS
        super().__init__(
            "Calibration diagnostics schema "
            f"{CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION} declares one label per "
            f"identifier, but the target rows give {len(self.conflicts)} "
            "identifier(s) more than one label or level, and no label is chosen "
            f"for them: {shown}"
            + (f"; and {more} more" if more > 0 else "")
            + ". Make the sources agree, or choose the label in the country's "
            "hierarchy label choices."
        )


class _Observed:
    """Distinct values seen per key, each with the first target carrying it."""

    def __init__(self) -> None:
        self._values: dict[object, dict[str, str]] = {}

    def record(self, key: object, value: str, target: str) -> None:
        self._values.setdefault(key, {}).setdefault(value, target)

    def keys(self) -> list[object]:
        return list(self._values)

    def values(self, key: object) -> dict[str, str]:
        return self._values[key]


def _single_or_conflict(
    observed: _Observed,
    key: object,
    subject: str,
    conflicts: list[HierarchyLabelConflict],
    chosen: str | None = None,
    resolved: list[tuple[HierarchyLabelConflict, str]] | None = None,
) -> str:
    """The one value the rows give ``key``; else the chosen one; else a conflict.

    A disagreement a choice settles is appended to ``resolved``, so it is
    logged rather than hidden.
    """

    values = observed.values(key)
    if len(values) == 1:
        return next(iter(values))
    if chosen is not None:
        if resolved is not None:
            resolved.append(
                (HierarchyLabelConflict(subject, tuple(values.items())), chosen)
            )
        return chosen
    conflicts.append(HierarchyLabelConflict(subject, tuple(values.items())))
    # Any value keeps the catalog well formed; the conflict is raised below.
    return next(iter(values))


def catalog_calibration_diagnostics(
    diagnostics: CalibrationDiagnosticsV8,
    *,
    label_choices: HierarchyLabelChoices | None = None,
) -> CalibrationDiagnosticsV9:
    """Publish a schema 8 document as schema 9, one label per identifier.

    Catalog entries are listed in the order the rows first use them, and a
    dimension's values in the order its rows first use them. Raises
    :class:`HierarchyLabelConflictError` listing every identifier the rows
    label (or, for a geography, place at a level) more than one way without
    a chosen label. Provider, category and dimension labels, a category's
    provider and a geography's level are never chosen.
    """

    choices = label_choices or HierarchyLabelChoices()
    providers = _Observed()
    categories = _Observed()
    category_providers = _Observed()
    geographies = _Observed()
    geography_levels = _Observed()
    dimensions = _Observed()
    dimension_values = _Observed()
    for row in diagnostics.targets:
        hierarchy = row.hierarchy
        providers.record(hierarchy.provider.id, hierarchy.provider.label, row.name)
        categories.record(hierarchy.category.id, hierarchy.category.label, row.name)
        category_providers.record(
            hierarchy.category.id, hierarchy.category.provider_id, row.name
        )
        geographies.record(hierarchy.geography.id, hierarchy.geography.label, row.name)
        geography_levels.record(
            hierarchy.geography.id, hierarchy.geography.level, row.name
        )
        for dimension in hierarchy.dimensions:
            dimensions.record(dimension.id, dimension.label, row.name)
            dimension_values.record(
                (dimension.id, dimension.value_id), dimension.value_label, row.name
            )

    conflicts: list[HierarchyLabelConflict] = []
    resolved: list[tuple[HierarchyLabelConflict, str]] = []
    provider_entries = [
        {
            "id": provider_id,
            "label": _single_or_conflict(
                providers, provider_id, f"provider {provider_id}", conflicts
            ),
        }
        for provider_id in providers.keys()
    ]
    category_entries = [
        {
            "id": category_id,
            "label": _single_or_conflict(
                categories, category_id, f"category {category_id}", conflicts
            ),
            "provider_id": _single_or_conflict(
                category_providers,
                category_id,
                f"category {category_id} provider",
                conflicts,
            ),
        }
        for category_id in categories.keys()
    ]
    geography_entries = [
        {
            "id": geography_id,
            "label": _single_or_conflict(
                geographies,
                geography_id,
                f"geography {geography_id}",
                conflicts,
                chosen=choices.geographies.get(geography_id),
                resolved=resolved,
            ),
            "level": _single_or_conflict(
                geography_levels,
                geography_id,
                f"geography {geography_id} level",
                conflicts,
            ),
        }
        for geography_id in geographies.keys()
    ]
    area_labels = {
        **{
            geography_id.casefold(): label
            for geography_id, label in choices.geographies.items()
        },
        **{entry["id"].casefold(): entry["label"] for entry in geography_entries},
    }

    def value_label(dimension_id: str, value_id: str) -> str:
        key = (dimension_id, value_id)
        subject = f"dimension {dimension_id} value {value_id}"
        chosen = choices.dimension_values.get(key)
        reviewed = choices.dimension_value_spellings.get(key)
        if (
            chosen is not None
            and reviewed is not None
            and not set(dimension_values.values(key)) <= reviewed
        ):
            # A spelling nobody reviewed: refuse rather than relabel it.
            chosen = None
            subject += " (a spelling its reviewed list does not include)"
        elif chosen is None and dimension_id in choices.geography_dimensions:
            chosen = area_labels.get(value_id.casefold())
        return _single_or_conflict(
            dimension_values, key, subject, conflicts, chosen=chosen, resolved=resolved
        )

    values_by_dimension: dict[str, list[dict[str, str]]] = {}
    for dimension_id, value_id in dimension_values.keys():
        values_by_dimension.setdefault(dimension_id, []).append(
            {"id": value_id, "label": value_label(dimension_id, value_id)}
        )
    dimension_entries = [
        {
            "id": dimension_id,
            "label": _single_or_conflict(
                dimensions, dimension_id, f"dimension {dimension_id}", conflicts
            ),
            "values": values_by_dimension[dimension_id],
        }
        for dimension_id in dimensions.keys()
    ]
    if conflicts:
        raise HierarchyLabelConflictError(conflicts)
    if resolved:
        _LOGGER.warning(
            "Calibration diagnostics: the target rows label %d identifier(s) more "
            "than one way; the chosen label is written for each: %s",
            len(resolved),
            "; ".join(
                f"{conflict.describe()} -> {label!r}" for conflict, label in resolved
            ),
        )

    document = diagnostics.model_dump(mode="python")
    for row in document["targets"]:
        hierarchy = row["hierarchy"]
        row["hierarchy"] = {
            "category_id": hierarchy["category"]["id"],
            "geography_id": hierarchy["geography"]["id"],
            "dimensions": [
                {"id": dimension["id"], "value_id": dimension["value_id"]}
                for dimension in hierarchy["dimensions"]
            ],
            "target": hierarchy["target"],
        }
    document["schema_version"] = CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION
    document["hierarchy_catalogs"] = {
        "providers": provider_entries,
        "categories": category_entries,
        "geographies": geography_entries,
        "dimensions": dimension_entries,
    }
    return CalibrationDiagnosticsV9.model_validate(document)


def expand_calibration_diagnostics(
    diagnostics: CalibrationDiagnosticsV9,
) -> CalibrationDiagnosticsV8:
    """The schema 8 document a schema 9 document was published from."""

    catalogs = diagnostics.hierarchy_catalogs
    providers = {provider.id: provider for provider in catalogs.providers}
    categories = {category.id: category for category in catalogs.categories}
    geographies = {geography.id: geography for geography in catalogs.geographies}
    dimensions = {dimension.id: dimension for dimension in catalogs.dimensions}
    values = {
        (dimension.id, value.id): value
        for dimension in catalogs.dimensions
        for value in dimension.values
    }
    document = diagnostics.model_dump(mode="python")
    del document["hierarchy_catalogs"]
    document["schema_version"] = CALIBRATION_DIAGNOSTICS_ASSEMBLY_SCHEMA_VERSION
    for row in document["targets"]:
        reference = row["hierarchy"]
        category = categories[reference["category_id"]]
        provider = providers[category.provider_id]
        geography = geographies[reference["geography_id"]]
        row["hierarchy"] = {
            "provider": {"id": provider.id, "label": provider.label},
            "category": {
                "id": category.id,
                "label": category.label,
                "provider_id": category.provider_id,
            },
            "geography": {
                "id": geography.id,
                "label": geography.label,
                "level": geography.level,
            },
            "dimensions": [
                {
                    "id": dimension["id"],
                    "label": dimensions[dimension["id"]].label,
                    "value_id": dimension["value_id"],
                    "value_label": values[
                        (dimension["id"], dimension["value_id"])
                    ].label,
                }
                for dimension in reference["dimensions"]
            ],
            "target": reference["target"],
        }
    return CalibrationDiagnosticsV8.model_validate(document)
