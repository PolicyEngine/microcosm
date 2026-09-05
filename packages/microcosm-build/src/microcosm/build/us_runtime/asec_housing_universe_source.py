"""Authenticate carried ASEC HU evidence and preserve the chosen parent chain."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from importlib import resources

import numpy as np
import pandas as pd

from microcosm.frame import Frame

from . import asec_current_money_source as money_source
from . import asec_housing_status_source as housing_source
from . import asec_housing_universe as universe_module
from .asec_current_money import MoneyRefusalError
from .asec_current_money_units import CurrentMoneyTaxUnitResult
from .asec_housing_status import HousingStatusRefusalError
from .asec_housing_status_source import HousingStatusAttachedAsec, _owned_frame
from .asec_housing_universe import (
    CONTEXT_COLUMNS,
    DERIVED_COLUMNS,
    RAW_COLUMNS,
    AuthenticatedHousingUniverse,
    HousingUniverseRefusalError,
    _json,
    _parse,
    _require,
    _sha,
)
from .operator_boundary import assert_operator_free_source_frame

ATTACHED_COLUMNS = tuple("asec_housing_universe_" + name for name in DERIVED_COLUMNS)
_COMPOSE_TOKEN = object()


def _implementation() -> dict:
    """Bind new interpretation/composition without editing any old verifier roster."""
    package = resources.files(__package__)
    return {
        "source_verification_sha256": money_source._verification_identity(),
        "definition_sha256": universe_module._definition_identity(),
        "housing_implementation": housing_source._implementation(),
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in ("asec_housing_universe_source.py", "asec_housing_universe.py")
        },
    }


def load_authenticated_housing_universe(
    source: money_source.AuthenticatedCurrentMoneySource,
) -> AuthenticatedHousingUniverse:
    """Read only the already-authenticated household attachment, with no new joins."""
    try:
        _require(
            type(source) is money_source.AuthenticatedCurrentMoneySource,
            "AUTHENTICATED_SOURCE",
        )
        source.validate()
        implementation = _implementation()
        scope = source.scope
        table = source.frame.table("household")
        columns = ("household_id",) + tuple(
            "asec_" + name for name in ("H_SEQ",) + CONTEXT_COLUMNS
        )
        _require(
            table.columns.is_unique and set(columns) <= set(table), "CARRIED_COLUMNS"
        )
        _require(
            all(table[name].dtype == np.dtype("int64") for name in columns),
            "CARRIED_INT64",
        )
        raw = pd.DataFrame(
            {
                "household_id": np.asarray(scope.household_ids, dtype="int64"),
                "income_year": np.asarray(scope.household_years, dtype="int64"),
                "H_SEQ": np.asarray(
                    [int(key) for key in scope.household_native_keys], dtype="int64"
                ),
                **{
                    name: table["asec_" + name].to_numpy(copy=True)
                    for name in CONTEXT_COLUMNS
                },
            },
            columns=RAW_COLUMNS,
        )
        _require(
            np.array_equal(table.household_id.to_numpy(), raw.household_id.to_numpy()),
            "HOUSEHOLD_ORDER",
        )
        _require(
            np.array_equal(table.asec_H_SEQ.to_numpy(), raw.H_SEQ.to_numpy()),
            "NATIVE_KEY_BINDING",
        )
        result = universe_module._issue(
            raw,
            source={
                "identity": source.source.identity.decode(),
                "carried_columns": columns,
                "income_year_origin": "authenticated_money_scope",
            },
            implementation=implementation,
        )
        source.validate()
        _require(_implementation() == implementation, "IMPLEMENTATION_CHANGED")
        return result
    except HousingUniverseRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingUniverseRefusalError(error.reason) from None
    except (ValueError, TypeError, KeyError, OverflowError, AssertionError, OSError):
        raise HousingUniverseRefusalError("SOURCE_CONTRACT_REFUSAL") from None


def _tax_parent(parent) -> CurrentMoneyTaxUnitResult:
    _require(
        type(parent) in (CurrentMoneyTaxUnitResult, HousingStatusAttachedAsec),
        "TAX_OR_HOUSING_PARENT_REQUIRED",
    )
    return parent if type(parent) is CurrentMoneyTaxUnitResult else parent.tax_result


def _check_parent(parent, universe):
    tax = _tax_parent(parent)
    _require(type(universe) is AuthenticatedHousingUniverse, "AUTHENTICATED_UNIVERSE")
    parent.validate()
    universe.validate()
    _require(
        universe.header_data["source"]["identity"].encode()
        == tax.money.bindings.spec.source.identity,
        "UNIVERSE_SOURCE_BINDING",
    )
    household = parent.frame.table("household")
    _require(
        household.household_id.dtype == np.dtype("int64")
        and np.array_equal(
            household.household_id.to_numpy(), universe.array("household_id")
        ),
        "UNIVERSE_HOUSEHOLD_BINDING",
    )
    _require(
        _parse(tax.money.header)["household_year_sha256"]
        == _sha(universe.array("income_year").tobytes()),
        "UNIVERSE_YEAR_BINDING",
    )
    for name in ("H_SEQ",) + CONTEXT_COLUMNS:
        column = household["asec_" + name]
        _require(
            column.dtype == np.dtype("int64")
            and column.to_numpy().tobytes() == universe.array(name).tobytes(),
            "UNIVERSE_CONTEXT_BINDING",
        )


def validate_attached_frame(
    frame: Frame,
    *,
    parent: CurrentMoneyTaxUnitResult | HousingStatusAttachedAsec,
    universe: AuthenticatedHousingUniverse,
) -> None:
    """Check a whole owned parent extension; a selected child cannot inherit this authority."""
    try:
        _check_parent(parent, universe)
        _require(type(frame) is Frame, "COMPOSED_FRAME")
        frame.revalidate()
        assert_operator_free_source_frame(
            frame, label="ASEC housing universe attachment"
        )
        household = frame.table("household")
        _require(set(ATTACHED_COLUMNS) <= set(household), "ATTACHMENT_COLUMNS")
        for name, attached in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True):
            column = household[attached]
            _require(
                column.dtype == np.dtype("uint8")
                and column.to_numpy().tobytes() == universe.array(name).tobytes(),
                "ATTACHMENT_OUTPUT",
            )
        recovered = _owned_frame(frame)
        recovered.table("household").drop(columns=list(ATTACHED_COLUMNS), inplace=True)
        _require(
            money_source._frame_signature(recovered)
            == parent.receipt["output_frame_sha256"],
            "PARENT_FRAME_RECOVERY",
        )
        parent.validate()
        universe.validate()
    except HousingUniverseRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingUniverseRefusalError(error.reason) from None
    except HousingStatusRefusalError:
        raise HousingUniverseRefusalError("HOUSING_PARENT_REFUSAL") from None
    except (ValueError, TypeError, KeyError, AssertionError):
        raise HousingUniverseRefusalError("ATTACHMENT_CONTRACT_REFUSAL") from None


@dataclass(frozen=True)
class HousingUniverseAttachedAsec:
    """Owned HU extension retaining its exact tax or housing-status parent."""

    frame: Frame
    parent: CurrentMoneyTaxUnitResult | HousingStatusAttachedAsec
    universe: AuthenticatedHousingUniverse
    _receipt: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _COMPOSE_TOKEN, "ATTACHMENT_CONSTRUCTOR_UNAVAILABLE")

    @property
    def tax_result(self):
        return _tax_parent(self.parent)

    @property
    def money(self):
        return self.tax_result.money

    @property
    def receipt(self):
        return _parse(self._receipt)

    def validate(self) -> None:
        validate_attached_frame(self.frame, parent=self.parent, universe=self.universe)
        _require(
            self.receipt == _receipt(self.parent, self.universe, self.frame),
            "COMPOSED_RESULT_CHANGED",
        )


def _receipt(parent, universe, frame) -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_housing_universe_attachment.v1",
        "release_eligible": False,
        "parent_kind": type(parent).__name__,
        "parent_receipt_sha256": _sha(_json(parent.receipt)),
        "tax_receipt_sha256": _sha(_json(_tax_parent(parent).receipt)),
        "money_header_sha256": _sha(parent.money.header),
        "universe_content_sha256": universe.content_sha256,
        "parent_frame_sha256": parent.receipt["output_frame_sha256"],
        "output_frame_sha256": money_source._frame_signature(frame),
        "outputs": list(ATTACHED_COLUMNS),
    }


def attach_housing_universe(
    parent: CurrentMoneyTaxUnitResult | HousingStatusAttachedAsec,
    universe: AuthenticatedHousingUniverse,
) -> HousingUniverseAttachedAsec:
    """Preserve all chosen parent columns, including student and subsidy evidence."""
    try:
        _check_parent(parent, universe)
        _require(
            not set(ATTACHED_COLUMNS) & set(parent.frame.table("household")),
            "EXISTING_ATTACHMENT_COLUMNS",
        )
        frame = _owned_frame(parent.frame)
        _require(
            money_source._frame_signature(frame)
            == parent.receipt["output_frame_sha256"],
            "PARENT_CAPTURE_CHANGED",
        )
        for name, alias in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True):
            frame.table("household")[alias] = universe.array(name).copy()
        validate_attached_frame(frame, parent=parent, universe=universe)
        receipt = _receipt(parent, universe, frame)
        result = HousingUniverseAttachedAsec(
            frame, parent, universe, _json(receipt), _token=_COMPOSE_TOKEN
        )
        result.validate()
        return result
    except HousingUniverseRefusalError:
        raise
    except MoneyRefusalError as error:
        raise HousingUniverseRefusalError(error.reason) from None
    except HousingStatusRefusalError:
        raise HousingUniverseRefusalError("HOUSING_PARENT_REFUSAL") from None
    except (ValueError, TypeError, KeyError, AssertionError):
        raise HousingUniverseRefusalError("ATTACHMENT_CONTRACT_REFUSAL") from None


def verify_housing_universe_rows(
    household_table: pd.DataFrame,
    universe: AuthenticatedHousingUniverse,
    *,
    lineage_ids: np.ndarray,
) -> None:
    """Verify source-derived cells of subsets/clones, without issuing Frame authority.

    ``lineage_ids`` are the caller's original source household IDs, repeated for
    clones. Their custody/assembly provenance is a separate obligation. This
    checks carried H_SEQ/context and derived values, not a selected Frame receipt.
    """
    try:
        _require(
            type(universe) is AuthenticatedHousingUniverse, "AUTHENTICATED_UNIVERSE"
        )
        universe.validate()
        _require(
            type(household_table) is pd.DataFrame and household_table.columns.is_unique,
            "HOUSEHOLD_TABLE",
        )
        _require(
            type(lineage_ids) is np.ndarray
            and lineage_ids.dtype == np.dtype("int64")
            and lineage_ids.shape == (len(household_table),),
            "LINEAGE_INT64",
        )
        positions = pd.Index(universe.array("household_id")).get_indexer(lineage_ids)
        _require(bool((positions >= 0).all()), "LINEAGE_COVERAGE")
        if "household_source_id" in household_table:
            carried_ids = household_table.household_source_id
            _require(
                carried_ids.dtype == np.dtype("int64")
                and np.array_equal(carried_ids.to_numpy(), lineage_ids),
                "LINEAGE_BINDING",
            )
        for name, column, dtype in (
            *((n, "asec_" + n, "int64") for n in ("H_SEQ",) + CONTEXT_COLUMNS),
            *(
                (n, c, "uint8")
                for n, c in zip(DERIVED_COLUMNS, ATTACHED_COLUMNS, strict=True)
            ),
        ):
            values = household_table[column]
            _require(
                values.dtype == np.dtype(dtype)
                and np.array_equal(values.to_numpy(), universe.array(name)[positions]),
                "ROW_EVIDENCE",
            )
        universe.validate()
    except HousingUniverseRefusalError:
        raise
    except (ValueError, TypeError, KeyError, AssertionError):
        raise HousingUniverseRefusalError("ROW_CONTRACT_REFUSAL") from None
