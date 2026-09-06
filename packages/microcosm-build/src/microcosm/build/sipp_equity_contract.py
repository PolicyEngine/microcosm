"""Packaged consumer contract for the SIPP nominal December-2022 equity fit.

Closed pins for one already accepted source artifact, the target-specific
admission rule, the single predictor, and the tiny deterministic model
configuration. The pins are compared against a supplied artifact; a caller can
never confer authority by supplying matching metadata.

Scope: a known-target subset of December 2022 residences in nominal dollars.
Not a US population estimate, not a PolicyEngine input, not a 2024 stock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from types import MappingProxyType

from microcosm.graph.canonical import canonical_json, sha256_domain

__all__ = [
    "CONTRACT_RESOURCE",
    "SippEquityContract",
    "TARGET_NAME",
    "PREDICTOR_NAME",
    "ROSTER_PREDICTOR_NAME",
    "TARGET_PERIOD",
    "ContractError",
    "load_contract",
]

CONTRACT_RESOURCE = "sipp_equity_contract_v1.json"
TARGET_NAME = "sipp_all_vehicle_equity_dec2022"
PREDICTOR_NAME = "household_size"
ROSTER_PREDICTOR_NAME = "roster_household_size"
TARGET_PERIOD = 2022
TARGET_CONCEPT = "sipp_theq_veh"


class ContractError(ValueError):
    """Sanitized refusal at the packaged consumer-contract boundary."""


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class SippEquityContract:
    """The packaged pins plus their own canonical identity."""

    document: MappingProxyType
    identity: str

    @property
    def source(self):
        return self.document["source"]

    @property
    def model(self):
        return self.document["model"]

    @property
    def admission(self):
        return self.document["admission"]

    @property
    def target(self):
        return self.document["target"]

    @property
    def predictor(self):
        return self.document["predictor"]

    def require_request(self, *, concept: str, period: int) -> None:
        """Refuse any concept or period other than the named December-2022 target.

        This runs before any source access or kernel execution. A 2024 request
        carries the outstanding year-bridge requirement, which this consumer
        does not implement.
        """
        target = self.document["target"]
        if concept != target["name"]:
            raise ContractError(f"UNSUPPORTED_CONCEPT:{concept}")
        if type(period) is not int or period != target["period"]:
            raise ContractError(f"UNSUPPORTED_PERIOD:{period}:YEAR_BRIDGE_REQUIRED")


@cache
def load_contract() -> SippEquityContract:
    """Read, validate and identify the packaged contract resource.

    This is the only production route to a contract. Tests build invented
    contracts through :func:`_contract_from_document`, which is deliberately
    private: a caller-supplied pin never becomes production authority.
    """
    return _contract_from_document(
        json.loads((files(__package__) / CONTRACT_RESOURCE).read_bytes())
    )


def _contract_from_document(document) -> SippEquityContract:
    """Validate and identify a contract document. Test-only construction route."""
    if not isinstance(document, dict):
        raise ContractError("CONTRACT_SHAPE")
    if document.get("contract_id") != "sipp-nominal-equity-consumer/1":
        raise ContractError("CONTRACT_ID")
    for section in ("target", "predictor", "admission", "model", "source"):
        if not isinstance(document.get(section), dict):
            raise ContractError(f"CONTRACT_SECTION:{section}")
    target = document["target"]
    if (
        target["name"] != TARGET_NAME
        or target["concept"] != TARGET_CONCEPT
        or target["period"] != TARGET_PERIOD
        or target["month"] != 12
        or target["signed"] is not True
    ):
        raise ContractError("CONTRACT_TARGET")
    predictor = document["predictor"]
    if (
        predictor["name"] != PREDICTOR_NAME
        or predictor["roster_column"] != ROSTER_PREDICTOR_NAME
    ):
        raise ContractError("CONTRACT_PREDICTOR")
    model = document["model"]
    if (
        type(model["n_estimators"]) is not int
        or type(model["seed"]) is not int
        or type(model["zero_atol"]) is not float
        or model["max_samples_leaf"] is not None
    ):
        raise ContractError("CONTRACT_MODEL")
    for key in ("producer_binding", "projection_key", "receipt_key"):
        value = document["source"][key]
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"CONTRACT_SOURCE_KEY:{key}")
    identity = sha256_domain(
        "sipp-nominal-equity-consumer-contract/1", canonical_json(document)
    )
    return SippEquityContract(_frozen(document), identity)
