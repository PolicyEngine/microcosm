"""Actual tax-only reconstruction from verified target-current ASEC money.

The transient view has a closed read set. All other observed source columns and
US entities remain untouched. This is not a policy impact or release evaluator.
"""

import inspect
import sys
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_table_string_dtypes
from microcosm.frame import Frame, Weights
from microcosm.frame import units as frame_units

from ._asec_current_money_codec import current_money_content_sha256
from .asec_current_money import (
    MICROUNIT_VERSION,
    MoneyRefusalError,
    ReadyCurrentMoney,
    _json,
    _parse,
    _require,
    _sha,
    _validate_money,
)
from .asec_current_money_source import (
    AuthenticatedCurrentMoneySource,
    _detach_frame_axes,
    _frame_signature,
    _same_table,
    _source_verification_identity,
)
from .asec_student_controls import (
    ALIASES as STUDENT_ALIASES,
)
from .asec_student_controls import (
    CONTROLS as STUDENT_CONTROLS,
)
from .asec_student_controls import (
    StudentControlsAttachedAsec,
)

MICROUNIT_CURRENT_MONEY_COLUMNS = tuple(
    "WSAL_VAL SEMP_VAL FRSE_VAL INT_VAL DIV_VAL RNT_VAL CAP_VAL UC_VAL OI_VAL ANN_VAL PNSN_VAL SS_VAL PTOTVAL".split()
)
_STRUCTURAL_COLUMNS = (
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
)
_CONTROL_COLUMNS = (
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
)
MICROUNIT_CURRENT_INPUT_COLUMNS = (
    _STRUCTURAL_COLUMNS + _CONTROL_COLUMNS + MICROUNIT_CURRENT_MONEY_COLUMNS
)
TAX_PERSON_OUTPUTS = (
    "person_tax_unit_id",
    "tax_unit_role_input",
    "is_related_to_head_or_spouse",
)
_RESULT_TOKEN = object()


class MicrounitControlRefusalError(MoneyRefusalError):
    """Closed control-field diagnostics without changing the money-source schema."""

    def __init__(self, reason, column):
        super().__init__(reason)
        self.field = (
            column if column in _STRUCTURAL_COLUMNS + _CONTROL_COLUMNS else "contract"
        )
        self.args = (f"{self.field}: {reason}",)


def _require_control(condition, reason, column):
    if not condition:
        raise MicrounitControlRefusalError(reason, column)


# Exact 0.1.0 import/runtime closure: __init__ imports core/diagnostics/registry;
# tax construction calls rule_helpers, whose cached threshold reads this YAML.
# No PolicyEngine package is needed for those rules.
_MICROUNIT_EXECUTION_PINS = {
    "__init__.py": "6164ebb289659a027d2bdd419781bd1025483bb2baf5576487f8e3f4ac594e01",
    "core.py": "95ec837e23e9a61a20bb48bcf20f87ffbfa536aaf0536706d020c6f79dc47701",
    "diagnostics.py": "c1a683b98ceac7a44438653eb992240e5d9ed130119f75f69f6bed3a02e6c7c6",
    "registry.py": "3ae5442a7bc637f8f786913abae97f63997baba5cedac948a80ce3b00f8cf673",
    "rule_helpers.py": "7d32f1b3ab855a21ec0ca2e6d30431c4dcb030ffb250a28ef1978c50ed334243",
    "tax_unit_construction.py": "79007065113f9600601433dc2633f4392c675e28b87a7383ab8a1f7c6279d47e",
    "data/dependent_gross_income_limit.yaml": "3e20dd8f44d5ff919497fe38c2f68aa97a9f333539fbea7d0ca4f2c29e588da0",
}


def _runtime_file_bytes(path):
    _require(path.stat().st_size <= 1024 * 1024, "MICROUNIT_EXECUTION_SIZE")
    value = path.read_bytes()
    _require(len(value) <= 1024 * 1024, "MICROUNIT_EXECUTION_SIZE")
    return value


def _execution():
    try:
        distribution = metadata.distribution("microunit")
        _require(distribution.version == MICROUNIT_VERSION, "MICROUNIT_VERSION")
        root = Path(distribution.locate_file("microunit")).resolve()
        inventory = {str(p) for p in distribution.files or ()}
        for name, expected in _MICROUNIT_EXECUTION_PINS.items():
            _require(
                "microunit/" + name in inventory
                and _sha(_runtime_file_bytes(root / name)) == expected,
                "MICROUNIT_EXECUTION_PIN",
            )
        import microunit

        _require(
            Path(microunit.__file__).resolve() == root / "__init__.py",
            "MICROUNIT_IMPORT_ORIGIN",
        )
        _require(
            inspect.getsourcefile(microunit.construct_tax_units)
            == str(root / "tax_unit_construction.py"),
            "MICROUNIT_IMPORT_ORIGIN",
        )
        for name in _MICROUNIT_EXECUTION_PINS:
            if name.endswith(".py"):
                module_name = (
                    "microunit" if name == "__init__.py" else "microunit." + name[:-3]
                )
                module = sys.modules.get(module_name)
                _require(
                    module is not None
                    and Path(module.__file__).resolve() == root / name,
                    "MICROUNIT_IMPORT_ORIGIN",
                )
        package = resources.files(__package__)
        files = {
            name: _sha(package.joinpath(name).read_bytes())
            for name in (
                "asec_current_money_units.py",
                "asec_current_money_source.py",
                "asec_current_money.py",
                "_asec_current_money_codec.py",
                "asec_current_money_resources.py",
                "asec_student_controls.py",
            )
        }
        files["frame.units"] = _sha(Path(frame_units.__file__).read_bytes())
        identity = {
            "version": distribution.version,
            "microunit": _MICROUNIT_EXECUTION_PINS,
            "adapter_modules": files,
            "dependencies": {
                n: metadata.version(n) for n in ("numpy", "pandas", "pyyaml")
            },
            "year": 2024,
            "mode": "policyengine",
        }
        return microunit.construct_tax_units, _json(identity)
    except (metadata.PackageNotFoundError, OSError, ImportError):
        raise MoneyRefusalError("MICROUNIT_EXECUTION_UNAVAILABLE") from None


def current_money_microunit_view(source, ready):
    """Project only reviewed controls and corrected amounts, in actual person order."""
    _require(
        type(source) in (AuthenticatedCurrentMoneySource, StudentControlsAttachedAsec)
        and type(ready) is ReadyCurrentMoney,
        "AUTHENTICATED_MONEY_REQUIRED",
    )
    expected = source.ready()
    _require(
        ready.bindings == expected.bindings and ready.fields == expected.fields,
        "MONEY_CONTENT_MISMATCH",
    )
    _require(ready.bindings.spec == source.spec, "SPEC_MISMATCH")
    _validate_money(ready, source.spec, nominal=False)
    _require(
        all(field.validity.all() for field in ready.fields), "MISSING_REQUIRED_AMOUNT"
    )
    person = source.frame.person
    _require(
        set(_STRUCTURAL_COLUMNS + _CONTROL_COLUMNS) <= set(person),
        "MICROUNIT_INPUT_ROSTER",
    )
    view = (
        person.loc[:, _STRUCTURAL_COLUMNS + _CONTROL_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )
    if type(source) is StudentControlsAttachedAsec:
        # Only authenticated, code-owned aliases replace the transient logical
        # controls. Original source observations, including NaNs, stay untouched.
        source.validate()
        for name, alias in zip(STUDENT_CONTROLS, STUDENT_ALIASES, strict=True):
            view[name] = person[alias].to_numpy(copy=True)
    for column in view:
        values = view[column]
        _require_control(
            pd.api.types.is_numeric_dtype(values.dtype)
            and not pd.api.types.is_bool_dtype(values.dtype),
            "MICROUNIT_CONTROL_DTYPE",
            column,
        )
        _require_control(not values.isna().any(), "MICROUNIT_CONTROL_MISSING", column)
        array = values.to_numpy(dtype=np.float64)
        _require_control(
            np.isfinite(array).all() and (array == np.floor(array)).all(),
            "MICROUNIT_CONTROL_VALUES",
            column,
        )
    _require(
        np.array_equal(view.PH_SEQ.to_numpy(), person.person_household_id.to_numpy()),
        "MICROUNIT_HOUSEHOLD_BINDING",
    )
    _require(
        not view[["PH_SEQ", "A_LINENO"]].duplicated().any()
        and (view.A_LINENO > 0).all(),
        "MICROUNIT_PERSON_LINES",
    )
    for field in MICROUNIT_CURRENT_MONEY_COLUMNS:
        view[field] = ready.field(field).amounts.copy()
    _require(
        tuple(view.columns) == MICROUNIT_CURRENT_INPUT_COLUMNS, "MICROUNIT_INPUT_ROSTER"
    )
    return view


def _partitions(frame):
    grouped = {}
    for person, group in zip(
        frame.person.person_id, frame.person.person_tax_unit_id, strict=True
    ):
        grouped.setdefault(int(group), []).append(int(person))
    return tuple(sorted(tuple(sorted(members)) for members in grouped.values()))


@dataclass(frozen=True)
class CurrentMoneyTaxUnitResult:
    frame: Frame
    money: ReadyCurrentMoney
    _receipt: bytes
    _token: InitVar[object] = None
    _student_source: StudentControlsAttachedAsec | None = None

    def __post_init__(self, _token):
        _require(_token is _RESULT_TOKEN, "TAX_RESULT_CONSTRUCTOR_UNAVAILABLE")

    @property
    def receipt(self):
        return _parse(self._receipt)

    def validate(self):
        """Check the permitted tax-only reattachment before consuming a live result."""
        receipt = self.receipt
        if self._student_source is not None:
            _require(
                type(self._student_source) is StudentControlsAttachedAsec,
                "STUDENT_TAX_PARENT",
            )
            self._student_source.validate()
            _require(
                self.money.bindings == self._student_source.money.bindings
                and self.money.fields == self._student_source.money.fields
                and receipt["student_controls"] == self._student_source.receipt,
                "STUDENT_TAX_PARENT",
            )
        else:
            _require(receipt["student_controls"] is None, "STUDENT_TAX_PARENT")
        _validate_money(self.money, self.money.bindings.spec, nominal=False)
        source_evidence = _parse(self.money.bindings.spec.source.identity)
        _require(
            _source_verification_identity(source_evidence)
            == source_evidence["verification_sha256"],
            "SOURCE_IMPLEMENTATION_CHANGED",
        )
        try:
            frame_signature = _frame_signature(self.frame)
        except (ValueError, TypeError, KeyError, AssertionError):
            raise MoneyRefusalError("TAX_RESULT_CHANGED") from None
        _require(
            _sha(self.money.header) == receipt["money_header_sha256"]
            and current_money_content_sha256(self.money)
            == receipt["money_content_sha256"]
            and frame_signature == receipt["output_frame_sha256"],
            "TAX_RESULT_CHANGED",
        )
        _require(
            _execution()[1] == _json(receipt["implementation"]),
            "MICROUNIT_EXECUTION_CHANGED",
        )


def reconstruct_current_money_tax_units(source, ready):
    """Run the pinned real engine, graft tax outputs, and verify preservation."""
    try:
        return _reconstruct(source, ready)
    except MoneyRefusalError:
        raise
    except (ValueError, TypeError, KeyError, AssertionError):
        raise MoneyRefusalError("TAX_RECONSTRUCTION_REFUSAL") from None


def _reconstruct(source, ready):
    view = current_money_microunit_view(source, ready)
    construct, implementation = _execution()
    original = source.frame
    assignments, tax_unit = frame_units._construct_tax_units(
        construct, view, year=2024, mode="policyengine"
    )
    tax_unit = canonicalize_table_string_dtypes(
        tax_unit,
        boundary="ASEC current-money tax reconstruction",
        table_name="tax_unit",
    )
    _require(
        assignments.index.equals(view.index) and len(assignments) == len(view),
        "MICROUNIT_OUTPUT_ROWS",
    )
    person = original.person.copy(deep=True)
    for target, output in zip(
        TAX_PERSON_OUTPUTS,
        ("TAX_ID", "tax_unit_role_input", "is_related_to_head_or_spouse"),
        strict=True,
    ):
        dtype = person[target].dtype if target in person else assignments[output].dtype
        person[target] = pd.array(assignments[output].to_numpy(copy=True), dtype=dtype)
    tables = {entity: original.table(entity) for entity in original.entities}
    tables["person"] = person
    tables["tax_unit"] = tax_unit
    result = Frame(
        tables,
        original.schema,
        {
            entity: Weights(
                original.weights_for(entity).values, original.weights_for(entity).kind
            )
            for entity in original.weighted_entities
        },
        original.strata,
        mass_log=original.mass_log,
        metadata=original.metadata,
    )
    _detach_frame_axes(result)
    source.validate()
    _require(_execution()[1] == implementation, "MICROUNIT_EXECUTION_CHANGED")
    for entity in original.entities:
        if entity not in ("person", "tax_unit"):
            _same_table(result.table(entity), original.table(entity))
    untouched = [name for name in original.person if name not in TAX_PERSON_OUTPUTS]
    _same_table(result.person[untouched], original.person[untouched])
    pd.testing.assert_series_equal(result.strata, original.strata, check_exact=True)
    _require(
        result.metadata == original.metadata and result.mass_log == original.mass_log,
        "TAX_CONTEXT_PRESERVATION",
    )
    for entity in original.weighted_entities:
        _require(
            result.weights_for(entity).kind == original.weights_for(entity).kind
            and result.weights_for(entity).values.tobytes()
            == original.weights_for(entity).values.tobytes(),
            "TAX_WEIGHT_PRESERVATION",
        )
    receipt = {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_current_money_tax_units",
        "release_eligible": False,
        "all_current_money_consumers_wired": False,
        "year": 2024,
        "mode": "policyengine",
        "money_header_sha256": _sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "source_frame_sha256": _parse(source.source.identity)["frame_sha256"],
        "student_controls": source.receipt
        if type(source) is StudentControlsAttachedAsec
        else None,
        "output_frame_sha256": _frame_signature(result),
        "implementation": _parse(implementation),
        "person_outputs": TAX_PERSON_OUTPUTS,
        "tax_table_outputs": tuple(tax_unit.columns),
        "old_tax_units": original.n("tax_unit"),
        "new_tax_units": result.n("tax_unit"),
        "old_partition_sha256": _sha(_json(_partitions(original))),
        "new_partition_sha256": _sha(_json(_partitions(result))),
        "scope_sha256": _parse(ready.header)["scope_sha256"],
    }
    return CurrentMoneyTaxUnitResult(
        result,
        ready,
        _json(receipt),
        _token=_RESULT_TOKEN,
        _student_source=source if type(source) is StudentControlsAttachedAsec else None,
    )
