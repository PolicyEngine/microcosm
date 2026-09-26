"""Fixed-cohort numeric source authentication and tax-result composition.

This module does not calculate housing benefits. Public source authority requires
an existing verified ASEC source and privately staged exact cohort bytes.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.frame import Frame, Weights

from . import asec_current_money_source as money_source
from . import asec_household_observations as household_reader
from . import asec_housing_status as status_module
from .asec_current_money import MoneyRefusalError
from .asec_current_money_units import CurrentMoneyTaxUnitResult
from .asec_housing_status import (
    CONTEXT_COLUMNS,
    DERIVED_COLUMNS,
    OBSERVED_COLUMNS,
    RAW_COLUMNS,
    AuthenticatedHousingStatus,
    HousingStatusRefusalError,
    _json,
    _parse,
    _require,
    _sha,
)
from .operator_boundary import assert_operator_free_source_frame

_READ_COLUMNS = ("H_SEQ", "H_YEAR") + CONTEXT_COLUMNS + OBSERVED_COLUMNS
ATTACHED_COLUMNS = tuple("asec_housing_observed_" + name for name in DERIVED_COLUMNS)
_COMPOSE_TOKEN = object()


def _implementation() -> dict:
    """Actual classifier, reader, source, Frame, codec and composition closure."""
    package = resources.files(__package__)
    return {
        "source_verification_sha256": money_source._verification_identity(),
        "definition_sha256": status_module._definition_identity(),
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in (
                "asec_housing_status_source.py",
                "asec_housing_status.py",
                "asec_current_money.py",
                "_asec_current_money_codec.py",
                "asec_current_money_units.py",
            )
        },
    }


def _read_cohort(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        return household_reader._read_numeric_household(
            handle, columns=_READ_COLUMNS, require_same_block=True
        )


def load_authenticated_housing_status(
    source: money_source.AuthenticatedCurrentMoneySource,
    *,
    cohort_paths: Mapping[int, str | Path],
) -> AuthenticatedHousingStatus:
    """Join declared full cohort snapshots to the verified source's native keys.

    No caller pin override. This does not certify release or annual participation.
    The source's sealed scope supplies ordered IDs/year/native keys; the freshly
    authenticated cohort context must also equal the carried attachment context.
    """
    try:
        return _load(source, cohort_paths)
    except HousingStatusRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingStatusRefusalError(error.reason) from None
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        raise HousingStatusRefusalError("SOURCE_CONTRACT_REFUSAL") from None


def _load(source, cohort_paths: Mapping[int, str | Path]):
    _require(
        type(source) is money_source.AuthenticatedCurrentMoneySource,
        "AUTHENTICATED_SOURCE",
    )
    source.validate()
    before = _implementation()
    evidence = _parse(source.source.identity)
    pins = tuple((year, pin) for year, pin in evidence["cohorts"])
    _require(
        isinstance(cohort_paths, Mapping)
        and set(cohort_paths) == {year for year, _ in pins}
        and all(type(year) is int for year in cohort_paths),
        "COHORT_PATHS",
    )
    # Scope tuples were sealed from the actual Frame by the source loader.
    # They cannot borrow mutable pandas index/column backing arrays.
    scope = source.scope
    output = {
        "household_id": np.asarray(scope.household_ids, dtype=np.int64),
        "income_year": np.asarray(scope.household_years, dtype=np.int64),
        "survey_year": np.empty(len(scope.household_ids), dtype=np.int64),
        "H_SEQ": np.asarray(
            [int(key) for key in scope.household_native_keys], dtype=np.int64
        ),
        **{
            name: np.empty(len(scope.household_ids), dtype=np.int64)
            for name in CONTEXT_COLUMNS + OBSERVED_COLUMNS
        },
    }
    carried = (
        source.frame.table("household")
        .loc[:, ["household_id"] + ["asec_" + n for n in CONTEXT_COLUMNS]]
        .copy(deep=True)
    )
    carried.index = carried.index.copy(deep=True)
    carried.columns = carried.columns.copy(deep=True)
    _require(
        np.array_equal(carried.household_id.to_numpy(), output["household_id"]),
        "HOUSEHOLD_ORDER",
    )
    _require(
        all(carried[name].dtype == np.dtype("int64") for name in carried),
        "CARRIED_INT64",
    )
    joins = []
    with tempfile.TemporaryDirectory(prefix="microcosm-housing-status-") as directory:
        for year, pin in pins:
            staged = Path(directory) / f"cohort-{year}.h5"
            money_source._stage_verified(cohort_paths[year], pin, staged)
            raw = _read_cohort(staged)
            staged.unlink()
            _require(
                not raw.H_SEQ.duplicated().any() and bool((raw.H_SEQ > 0).all()),
                "NATIVE_KEY",
            )
            _require(bool((raw.H_YEAR == year + 1).all()), "SURVEY_YEAR")
            positions = np.flatnonzero(output["income_year"] == year)
            source_positions = pd.Index(raw.H_SEQ).get_indexer(
                output["H_SEQ"][positions]
            )
            _require(bool((source_positions >= 0).all()), "SOURCE_KEY_COVERAGE")
            joined = raw.iloc[source_positions]
            for name in CONTEXT_COLUMNS:
                _require(
                    np.array_equal(
                        joined[name].to_numpy(),
                        carried["asec_" + name].iloc[positions].to_numpy(),
                    ),
                    "CARRIED_CONTEXT",
                )
            output["survey_year"][positions] = joined.H_YEAR.to_numpy()
            for name in CONTEXT_COLUMNS + OBSERVED_COLUMNS:
                output[name][positions] = joined[name].to_numpy()
            joins.append(
                {
                    "income_year": year,
                    "sha256": pin,
                    "source_rows": len(raw),
                    "joined_rows": len(positions),
                    "unreferenced_source_rows": len(raw) - len(positions),
                }
            )
    source.validate()
    _require(_implementation() == before, "IMPLEMENTATION_CHANGED")
    return status_module._issue(
        pd.DataFrame(output, columns=RAW_COLUMNS),
        source={"identity": source.source.identity.decode(), "joins": joins},
        implementation=before,
    )


def _owned_frame(frame: Frame) -> Frame:
    # Frame copies native buffers; explicitly detach axes and weight vectors too.
    # This is one complete result copy, not a partial/fabricated US population.
    strata = frame.strata.copy(deep=True)
    strata.index = money_source._owned_index(strata.index)
    result = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            **{name: frame.link(name) for name in frame.links},
        },
        frame.schema,
        {
            entity: Weights(
                frame.weights_for(entity).values, frame.weights_for(entity).kind
            )
            for entity in frame.weighted_entities
        },
        strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    money_source._detach_frame_axes(result)
    return result


def _check_parent_status(parent, status):
    _require(type(parent) is CurrentMoneyTaxUnitResult, "TAX_RESULT_REQUIRED")
    _require(type(status) is AuthenticatedHousingStatus, "AUTHENTICATED_STATUS")
    parent.validate()
    status.validate()
    _require(
        status.header_data["source"]["identity"].encode()
        == parent.money.bindings.spec.source.identity,
        "STATUS_SOURCE_BINDING",
    )
    household = parent.frame.table("household")
    _require(
        np.array_equal(household.household_id.to_numpy(), status.array("household_id")),
        "STATUS_HOUSEHOLD_BINDING",
    )
    for name in CONTEXT_COLUMNS:
        _require(
            np.array_equal(household["asec_" + name].to_numpy(), status.array(name)),
            "STATUS_CONTEXT_BINDING",
        )


def validate_attached_frame(
    frame: Frame,
    *,
    parent: CurrentMoneyTaxUnitResult,
    status: AuthenticatedHousingStatus,
) -> None:
    """Verify a live or checkpoint-reloaded composition against both trusted parents."""
    try:
        _check_parent_status(parent, status)
        _require(type(frame) is Frame, "COMPOSED_FRAME")
        frame.revalidate()
        assert_operator_free_source_frame(frame, label="ASEC housing status attachment")
        household = frame.table("household")
        _require(set(ATTACHED_COLUMNS) <= set(household), "ATTACHMENT_COLUMNS")
        for name, attached in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True):
            _require(
                household[attached].dtype == np.dtype("uint8")
                and household[attached].to_numpy().tobytes()
                == status.array(name).tobytes(),
                "ATTACHMENT_OUTPUT",
            )
        # Drop ONLY our columns, then compare the existing full Frame signature.
        recovered = _owned_frame(frame)
        recovered.table("household").drop(columns=list(ATTACHED_COLUMNS), inplace=True)
        _require(
            money_source._frame_signature(recovered)
            == parent.receipt["output_frame_sha256"],
            "PARENT_FRAME_RECOVERY",
        )
        parent.validate()
    except HousingStatusRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingStatusRefusalError(error.reason) from None
    except (ValueError, TypeError, KeyError, AssertionError):
        raise HousingStatusRefusalError("ATTACHMENT_CONTRACT_REFUSAL") from None


@dataclass(frozen=True)
class HousingStatusAttachedAsec:
    """Separate composed result retaining the identical sealed tax/money parents."""

    frame: Frame
    tax_result: CurrentMoneyTaxUnitResult
    status: AuthenticatedHousingStatus
    _receipt: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _COMPOSE_TOKEN, "ATTACHMENT_CONSTRUCTOR_UNAVAILABLE")

    @property
    def money(self):
        return self.tax_result.money

    @property
    def receipt(self):
        return _parse(self._receipt)

    def validate(self) -> None:
        validate_attached_frame(self.frame, parent=self.tax_result, status=self.status)
        _require(
            self.receipt["housing_content_sha256"] == self.status.content_sha256
            and self.receipt["tax_receipt_sha256"]
            == _sha(_json(self.tax_result.receipt))
            and self.receipt["output_frame_sha256"]
            == money_source._frame_signature(self.frame),
            "COMPOSED_RESULT_CHANGED",
        )


def attach_housing_status(
    tax_result: CurrentMoneyTaxUnitResult, status: AuthenticatedHousingStatus
) -> HousingStatusAttachedAsec:
    """Compose source-status evidence after actual tax reconstruction, without mutation."""
    try:
        _check_parent_status(tax_result, status)
        _require(
            not set(ATTACHED_COLUMNS) & set(tax_result.frame.table("household")),
            "EXISTING_ATTACHMENT_COLUMNS",
        )
        result = _owned_frame(tax_result.frame)
        _require(
            money_source._frame_signature(result)
            == tax_result.receipt["output_frame_sha256"],
            "PARENT_CAPTURE_CHANGED",
        )
        for name, output in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True):
            result.table("household")[output] = status.array(name).copy()
        validate_attached_frame(result, parent=tax_result, status=status)
        receipt = {
            "schema_version": 1,
            "artifact_kind": "microcosm.asec_housing_status_attachment.v1",
            "release_eligible": False,
            "tax_receipt_sha256": _sha(_json(tax_result.receipt)),
            "housing_content_sha256": status.content_sha256,
            "parent_frame_sha256": tax_result.receipt["output_frame_sha256"],
            "output_frame_sha256": money_source._frame_signature(result),
            "outputs": ATTACHED_COLUMNS,
        }
        return HousingStatusAttachedAsec(
            result, tax_result, status, _json(receipt), _token=_COMPOSE_TOKEN
        )
    except HousingStatusRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingStatusRefusalError(error.reason) from None
    except (ValueError, TypeError, KeyError, AssertionError):
        raise HousingStatusRefusalError("ATTACHMENT_CONTRACT_REFUSAL") from None
