"""Typed, fail-closed population inputs for behavioral adapters.

Microcosm owns the measured or latent population columns declared here.
PolicyEngine may consume those columns and owns any behavioral mechanics that
use them.  This module deliberately has no formula, probability, eligibility,
take-up-assignment, or Axiom-concept surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from microcosm.frame import Frame

__all__ = [
    "PopulationInputContract",
    "PopulationInputNotReadyError",
    "PopulationInputProfile",
    "SchemePopulationMapping",
    "validate_population_input_frame",
]

PopulationSemanticKind = Literal["receipt", "application", "legal_status", "choice"]
PopulationDataKind = Literal["measured", "latent"]
InputReadiness = Literal["ready", "required_missing"]
MappingReadiness = Literal["ready", "required_missing"]
PeriodReadiness = Literal["ready", "exact_alignment_missing"]
CompletenessReadiness = Literal["ready", "complete_imputation_required_missing"]
PublisherSourceReadiness = Literal[
    "ready",
    "native_publisher_artifact_required_missing",
]

_INPUT_KEYS = frozenset(
    {
        "input_id",
        "column",
        "entity",
        "dtype",
        "nullable",
        "semantic_kind",
        "data_kind",
        "owner",
        "consumer",
        "mechanics_owner",
        "axiom_role",
        "description",
    }
)
_MAPPING_KEYS = frozenset(
    {
        "mapping_id",
        "target_reference",
        "input_id",
        "chronicle_source_record_id",
        "chronicle_entity",
        "chronicle_entity_role",
        "chronicle_geography_level",
        "chronicle_geography_id",
        "chronicle_geography_vintage",
        "chronicle_period_type",
        "chronicle_period",
        "microcosm_entity",
        "microcosm_geography_level",
        "microcosm_geography_id",
        "microcosm_geography_vintage",
        "microcosm_period_type",
        "microcosm_period",
        "publisher_source_readiness",
        "input_readiness",
        "mapping_readiness",
        "period_readiness",
        "completeness_readiness",
        "notes",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "country",
        "profile_id",
        "activation",
        "description",
        "inputs",
        "mappings",
    }
)
_SEMANTIC_KINDS = frozenset({"receipt", "application", "legal_status", "choice"})
_DATA_KINDS = frozenset({"measured", "latent"})
_INPUT_READINESS = frozenset({"ready", "required_missing"})
_MAPPING_READINESS = frozenset({"ready", "required_missing"})
_PERIOD_READINESS = frozenset({"ready", "exact_alignment_missing"})
_COMPLETENESS_READINESS = frozenset({"ready", "complete_imputation_required_missing"})
_PUBLISHER_SOURCE_READINESS = frozenset(
    {"ready", "native_publisher_artifact_required_missing"}
)
_BEHAVIORAL_NAME_TOKENS = (
    "take_up",
    "takeup",
    "takes_up",
    "if_eligible",
    "propensity",
    "elasticity",
)


class PopulationInputNotReadyError(RuntimeError):
    """A required population input cannot safely be used yet."""


def _closed_mapping(
    raw: object,
    keys: frozenset[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be an object.")
    actual = set(raw)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ValueError(
            f"{context} keys differ; missing={missing}, unknown={unknown}."
        )
    return raw


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _identifier(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if re.fullmatch(r"[a-z][a-z0-9_]*", result) is None:
        raise ValueError(f"{field} must match [a-z][a-z0-9_]*, got {result!r}.")
    return result


def _period(value: object, *, field: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or non-empty string.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field} must be an integer or non-empty string.")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PopulationInputContract:
    """One null-preserving Microcosm-owned boolean population input column."""

    input_id: str
    column: str
    entity: str
    dtype: str
    nullable: bool
    semantic_kind: PopulationSemanticKind
    data_kind: PopulationDataKind
    owner: str
    consumer: str
    mechanics_owner: str
    axiom_role: str
    description: str

    def __post_init__(self) -> None:
        _identifier(self.input_id, field="population input input_id")
        _identifier(self.column, field=f"population input {self.input_id!r} column")
        _identifier(self.entity, field=f"population input {self.input_id!r} entity")
        if self.dtype != "bool":
            raise ValueError(
                f"population input {self.input_id!r} dtype must be 'bool'."
            )
        if self.nullable is not True:
            raise ValueError(
                f"population input {self.input_id!r} nullable must be true so "
                "unknown status remains null until an explicit completeness gate."
            )
        if self.semantic_kind not in _SEMANTIC_KINDS:
            raise ValueError(
                f"population input {self.input_id!r} has unsupported semantic_kind "
                f"{self.semantic_kind!r}."
            )
        if self.data_kind not in _DATA_KINDS:
            raise ValueError(
                f"population input {self.input_id!r} has unsupported data_kind "
                f"{self.data_kind!r}."
            )
        if self.owner != "Microcosm":
            raise ValueError(
                f"population input {self.input_id!r} owner must be 'Microcosm'."
            )
        if self.consumer != "PolicyEngine" or self.mechanics_owner != "PolicyEngine":
            raise ValueError(
                f"population input {self.input_id!r} consumer and mechanics_owner "
                "must both be 'PolicyEngine'."
            )
        if self.axiom_role != "none":
            raise ValueError(
                f"population input {self.input_id!r} axiom_role must be 'none'; "
                "this contract cannot invent an Axiom input or formula."
            )
        _text(self.description, field=f"population input {self.input_id!r} description")
        for field, value in (("input_id", self.input_id), ("column", self.column)):
            normalized = value.casefold()
            forbidden = [
                token for token in _BEHAVIORAL_NAME_TOKENS if token in normalized
            ]
            if forbidden:
                raise ValueError(
                    f"population input {self.input_id!r} {field} {value!r} looks "
                    f"behavioral ({forbidden}); declare only receipt, application, "
                    "legal-status, or choice data."
                )

    @classmethod
    def from_mapping(cls, raw: object) -> PopulationInputContract:
        """Parse one closed-world population-input declaration."""

        values = _closed_mapping(raw, _INPUT_KEYS, context="population input")
        return cls(**values)


@dataclass(frozen=True)
class SchemePopulationMapping:
    """Declared Chronicle scheme population to one Microcosm input column."""

    mapping_id: str
    target_reference: str
    input_id: str
    chronicle_source_record_id: str
    chronicle_entity: str
    chronicle_entity_role: str
    chronicle_geography_level: str
    chronicle_geography_id: str
    chronicle_geography_vintage: str
    chronicle_period_type: str
    chronicle_period: int | str
    microcosm_entity: str
    microcosm_geography_level: str
    microcosm_geography_id: str
    microcosm_geography_vintage: str
    microcosm_period_type: str
    microcosm_period: int | str
    publisher_source_readiness: PublisherSourceReadiness
    input_readiness: InputReadiness
    mapping_readiness: MappingReadiness
    period_readiness: PeriodReadiness
    completeness_readiness: CompletenessReadiness
    notes: str

    def __post_init__(self) -> None:
        _identifier(self.mapping_id, field="scheme-population mapping_id")
        _text(
            self.target_reference,
            field=f"scheme-population mapping {self.mapping_id!r} target_reference",
        )
        _identifier(
            self.input_id,
            field=f"scheme-population mapping {self.mapping_id!r} input_id",
        )
        _text(
            self.chronicle_source_record_id,
            field=(
                f"scheme-population mapping {self.mapping_id!r} "
                "chronicle_source_record_id"
            ),
        )
        for field_name in (
            "chronicle_entity",
            "chronicle_entity_role",
            "chronicle_geography_level",
            "chronicle_period_type",
            "microcosm_entity",
            "microcosm_geography_level",
            "microcosm_period_type",
        ):
            _identifier(
                getattr(self, field_name),
                field=f"scheme-population mapping {self.mapping_id!r} {field_name}",
            )
        for field_name in (
            "chronicle_geography_id",
            "chronicle_geography_vintage",
            "microcosm_geography_id",
            "microcosm_geography_vintage",
        ):
            _text(
                getattr(self, field_name),
                field=f"scheme-population mapping {self.mapping_id!r} {field_name}",
            )
        _period(
            self.chronicle_period,
            field=f"scheme-population mapping {self.mapping_id!r} chronicle_period",
        )
        _period(
            self.microcosm_period,
            field=f"scheme-population mapping {self.mapping_id!r} microcosm_period",
        )
        if self.publisher_source_readiness not in _PUBLISHER_SOURCE_READINESS:
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} has unknown "
                "publisher_source_readiness "
                f"{self.publisher_source_readiness!r}."
            )
        if self.input_readiness not in _INPUT_READINESS:
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} has unknown "
                f"input_readiness {self.input_readiness!r}."
            )
        if self.mapping_readiness not in _MAPPING_READINESS:
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} has unknown "
                f"mapping_readiness {self.mapping_readiness!r}."
            )
        if self.period_readiness not in _PERIOD_READINESS:
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} has unknown "
                f"period_readiness {self.period_readiness!r}."
            )
        if self.completeness_readiness not in _COMPLETENESS_READINESS:
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} has unknown "
                "completeness_readiness "
                f"{self.completeness_readiness!r}."
            )
        _text(self.notes, field=f"scheme-population mapping {self.mapping_id!r} notes")

        if (
            self.chronicle_geography_level == "statistical_scope"
            and self.microcosm_geography_level != "statistical_scope"
        ):
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} cannot reinterpret "
                "a Chronicle statistical_scope as a resident NUTS or other "
                "geography."
            )
        if self.mapping_readiness == "ready":
            source_identity = (
                self.chronicle_entity,
                self.chronicle_geography_level,
                self.chronicle_geography_id,
                self.chronicle_geography_vintage,
            )
            population_identity = (
                self.microcosm_entity,
                self.microcosm_geography_level,
                self.microcosm_geography_id,
                self.microcosm_geography_vintage,
            )
            if source_identity != population_identity:
                raise ValueError(
                    f"scheme-population mapping {self.mapping_id!r} marked ready "
                    "without an exact entity/geography identity."
                )
        if self.period_readiness == "ready" and (
            self.chronicle_period_type,
            self.chronicle_period,
        ) != (self.microcosm_period_type, self.microcosm_period):
            raise ValueError(
                f"scheme-population mapping {self.mapping_id!r} marked period "
                "ready without an exact period identity."
            )

    @classmethod
    def from_mapping(cls, raw: object) -> SchemePopulationMapping:
        """Parse one closed-world scheme-population mapping."""

        values = _closed_mapping(
            raw,
            _MAPPING_KEYS,
            context="scheme-population mapping",
        )
        return cls(**values)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Readiness fields that still block execution."""

        statuses = {
            "publisher_source_readiness": self.publisher_source_readiness,
            "input_readiness": self.input_readiness,
            "mapping_readiness": self.mapping_readiness,
            "period_readiness": self.period_readiness,
            "completeness_readiness": self.completeness_readiness,
        }
        return tuple(
            f"{key}={value!r}" for key, value in statuses.items() if value != "ready"
        )

    def require_ready(self) -> None:
        """Refuse execution until source, input, mapping, and period are ready."""

        if self.blockers:
            raise PopulationInputNotReadyError(
                f"scheme-population mapping {self.mapping_id!r} is not ready: "
                + ", ".join(self.blockers)
                + "."
            )


@dataclass(frozen=True)
class PopulationInputProfile:
    """A value-free inventory of inputs and exact scheme mappings."""

    country: str
    profile_id: str
    description: str
    inputs: tuple[PopulationInputContract, ...]
    mappings: tuple[SchemePopulationMapping, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        *,
        country: str,
    ) -> PopulationInputProfile:
        """Parse and cross-check a closed, explicitly activated profile."""

        values = _closed_mapping(raw, _PROFILE_KEYS, context="population input profile")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("population input profile schema_version must be 1.")
        if values["country"] != country:
            raise ValueError(
                f"population input profile country must match {country!r}."
            )
        if values["activation"] != "explicit_only":
            raise ValueError(
                "population input profile activation must be 'explicit_only'."
            )
        _identifier(country, field="population input profile country")
        profile_id = _identifier(
            values["profile_id"], field="population input profile profile_id"
        )
        description = _text(
            values["description"], field="population input profile description"
        )
        raw_inputs = values["inputs"]
        raw_mappings = values["mappings"]
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise ValueError(
                "population input profile inputs must be a non-empty list."
            )
        if not isinstance(raw_mappings, list) or not raw_mappings:
            raise ValueError(
                "population input profile mappings must be a non-empty list."
            )
        inputs = tuple(PopulationInputContract.from_mapping(row) for row in raw_inputs)
        mappings = tuple(
            SchemePopulationMapping.from_mapping(row) for row in raw_mappings
        )
        cls._validate_links(inputs, mappings)
        return cls(country, profile_id, description, inputs, mappings)

    @staticmethod
    def _validate_links(
        inputs: tuple[PopulationInputContract, ...],
        mappings: tuple[SchemePopulationMapping, ...],
    ) -> None:
        input_ids = [row.input_id for row in inputs]
        columns = [(row.entity, row.column) for row in inputs]
        mapping_ids = [row.mapping_id for row in mappings]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("population input profile has duplicate input ids.")
        if len(columns) != len(set(columns)):
            raise ValueError(
                "population input profile has duplicate entity/column inputs."
            )
        if len(mapping_ids) != len(set(mapping_ids)):
            raise ValueError("population input profile has duplicate mapping ids.")

        input_by_id = {row.input_id: row for row in inputs}
        used_inputs: set[str] = set()
        for mapping in mappings:
            input_contract = input_by_id.get(mapping.input_id)
            if input_contract is None:
                raise ValueError(
                    f"scheme-population mapping {mapping.mapping_id!r} references "
                    f"unknown input {mapping.input_id!r}."
                )
            if input_contract.entity != mapping.microcosm_entity:
                raise ValueError(
                    f"scheme-population mapping {mapping.mapping_id!r} entity "
                    f"{mapping.microcosm_entity!r} does not match input entity "
                    f"{input_contract.entity!r}."
                )
            used_inputs.add(mapping.input_id)
        orphaned = sorted(set(input_ids) - used_inputs)
        if orphaned:
            raise ValueError(
                "population input profile has inputs with no scheme-population "
                f"mapping: {orphaned}."
            )

    def input(self, input_id: str) -> PopulationInputContract:
        """Return a declared input by id, refusing omissions."""

        for contract in self.inputs:
            if contract.input_id == input_id:
                return contract
        raise KeyError(f"Unknown population input {input_id!r}.")

    def mapping(self, mapping_id: str) -> SchemePopulationMapping:
        """Return a declared mapping by id, refusing bypass by omission."""

        for mapping in self.mappings:
            if mapping.mapping_id == mapping_id:
                return mapping
        raise KeyError(f"Unknown scheme-population mapping {mapping_id!r}.")


def _canonical_row_id(value: object, *, column: str) -> int | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(
            f"population input row identity column {column!r} must contain only "
            "integer or string ids."
        )
    return value


def validate_population_input_frame(
    frame: Frame,
    profile: PopulationInputProfile,
    *,
    mapping_id: str,
) -> Mapping[str, object]:
    """Validate one ready Frame column and return a row/value identity receipt.

    Readiness is checked before the Frame is inspected.  The receipt contains
    only counts and hashes, never raw microdata ids or values.
    """

    if not isinstance(profile, PopulationInputProfile):
        raise TypeError("profile must be a PopulationInputProfile.")
    mapping = profile.mapping(mapping_id)
    mapping.require_ready()
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a microcosm Frame.")

    contract = profile.input(mapping.input_id)
    table = frame.table(contract.entity)
    if contract.column not in table.columns:
        raise ValueError(
            f"ready population input {contract.input_id!r} is missing Frame column "
            f"{contract.column!r} on entity {contract.entity!r}."
        )
    values = table[contract.column]
    if values.empty:
        raise ValueError(
            f"ready population input {contract.input_id!r} has no entity rows."
        )
    if bool(values.isna().any()):
        raise ValueError(
            f"ready population input {contract.input_id!r} contains missing values."
        )
    raw_values = values.to_numpy(dtype=object, copy=True)
    if not all(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError(
            f"ready population input {contract.input_id!r} must contain only "
            "boolean values; integer 0/1 and other proxies are refused."
        )
    boolean_values = [bool(value) for value in raw_values]

    id_column = frame.schema.entity_id_column(contract.entity)
    row_ids = [
        _canonical_row_id(value, column=id_column)
        for value in table[id_column].to_numpy(dtype=object, copy=True)
    ]
    contract_payload = {
        "input": asdict(contract),
        "mapping": asdict(mapping),
    }
    unsigned_receipt: dict[str, object] = {
        "schema_version": 1,
        "country": profile.country,
        "profile_id": profile.profile_id,
        "mapping_id": mapping.mapping_id,
        "target_reference": mapping.target_reference,
        "input_id": contract.input_id,
        "entity": contract.entity,
        "column": contract.column,
        "semantic_kind": contract.semantic_kind,
        "data_kind": contract.data_kind,
        "chronicle_entity_role": mapping.chronicle_entity_role,
        "chronicle_source_record_id": mapping.chronicle_source_record_id,
        "chronicle_geography_level": mapping.chronicle_geography_level,
        "chronicle_geography_id": mapping.chronicle_geography_id,
        "chronicle_geography_vintage": mapping.chronicle_geography_vintage,
        "chronicle_period_type": mapping.chronicle_period_type,
        "chronicle_period": mapping.chronicle_period,
        "publisher_source_readiness": mapping.publisher_source_readiness,
        "n_rows": len(boolean_values),
        "n_true": sum(boolean_values),
        "n_false": len(boolean_values) - sum(boolean_values),
        "n_unknown": 0,
        "contract_sha256": _digest(contract_payload),
        "row_ids_sha256": _digest({"entity": contract.entity, "row_ids": row_ids}),
        "values_sha256": _digest({"column": contract.column, "values": boolean_values}),
        "row_values_sha256": _digest(
            {
                "entity": contract.entity,
                "column": contract.column,
                "row_values": list(zip(row_ids, boolean_values, strict=True)),
            }
        ),
    }
    return {
        **unsigned_receipt,
        "receipt_sha256": _digest(unsigned_receipt),
    }
