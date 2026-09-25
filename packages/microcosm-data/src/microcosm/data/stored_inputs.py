"""Refuse stored model inputs the certified engine does not define.

PolicyEngine-US 2.x renamed the WIC take-up input from ``would_claim_wic`` to
``takes_up_wic_if_eligible`` (microcosm#1026). The published national default
``populace-us-2024-spm-20260915`` and its reported-receipt child
``populace-us-2024-spm-receipts-20260923`` both record policyengine-us 2.2.1
as ``build.built_with_model_package``. 2.2.1 defines
``takes_up_wic_if_eligible`` (a Person, MONTH Boolean whose default is
``True``; ``wic`` is ``defined_for`` it) and not ``would_claim_wic``, yet both
releases store the seeded draw under the old name. policyengine-core's
``Simulation.build_from_dataset`` sets a stored column as an input only when
the tax-benefit system has a variable of that name; it collects every other
column, logs one warning and ignores it. So the draw was ignored, every
WIC-eligible person took WIC up, and no release gate refused either release.

This module is the recurrence guard (decision d271). A US release is refused
when one of its stored tables carries a column that

- looks like a model input (:func:`is_model_named`);
- is not a variable of the policyengine-us version the release is certified
  against; and
- is not in :data:`US_STORED_NON_VARIABLE_COLUMNS`, the reviewed register of
  columns a build stores on purpose although the engine defines no variable
  for them. Each entry carries the reason the engine may ignore it.

The rule itself is pure. :func:`undefined_stored_inputs` and
:func:`stored_input_failures` take the stored column names, the engine's
variable names and the register, and never read a row.
:func:`h5_stored_tables` lists an H5's stored tables from HDF metadata alone.
:func:`installed_us_engine` reads the variable names of the installed
policyengine-us. Two certification seams run the rule, each at the point where
the installed engine is provably the one the release is certified against:

- the US fiscal-refresh release tool's batched pre-export gates
  (``tools/build_us_fiscal_refresh_release.py``), on the export frame's stored
  tables, before the H5 is written. The tool writes the H5 with the installed
  engine and records that engine's version as
  ``build.built_with_model_package``. After the write it checks that the
  verdict on the written bytes is the verdict the gate reached;
- the source-enrichment native-loader probe
  (:func:`microcosm.data.source_enrichment.run_native_loader_compatibility`),
  on the candidate H5, through :func:`require_h5_stored_inputs`. The probe
  runs in the tested runtime when a candidate is certified and again whenever
  the contract replays it (validation, the publisher's preflight and
  publication), and the contract refuses a bundle whose
  ``build.built_with_model_package`` differs from the runtime the probe
  tested. Both releases above went through this lane.

**Naming convention, measured rather than assumed.** policyengine-us 2.2.1
defines 6,167 variables. 6,114 match ``[a-z][a-z0-9_]*``. The other 53 are the
postal codes of the 50 states, DC, PR and VI:
``variables/household/demographic/geographic/state/in_state.py`` generates one
household Boolean formula per state, named by its code. No variable starts
with an underscore or a digit. So any column outside the convention passes by
rule: a column with an uppercase letter (the raw Census fields ``A_AGE``,
``H_TENURE``, ``SPM_WICVAL`` and so on), or one that starts with an underscore
or a digit. The files examined for microcosm#1026 store 163 uppercase columns;
none is an engine variable and none is one of the 53 codes. The 53 are all
formulas, and the release writer
(:class:`microcosm.frame.adapters.policyengine_us.PolicyEngineUSEngine`)
refuses to store a formula-owned column, so a release cannot store one of them
either. Tests pin the convention and the formula ownership against the locked
engine, so an engine that adds a variable outside the convention fails CI
rather than passing silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "MODEL_NAMED_COLUMN_PATTERN",
    "US_STORED_NON_VARIABLE_COLUMNS",
    "CertifiedEngine",
    "StoredInputRefusalError",
    "StoredTableLayoutError",
    "h5_stored_tables",
    "h5_verdict",
    "installed_us_engine",
    "is_model_named",
    "register_consistency_failures",
    "register_sha256",
    "require_h5_stored_inputs",
    "stored_input_failures",
    "undefined_stored_inputs",
]

#: The policyengine-us variable naming convention: lowercase ASCII letters,
#: digits and underscores, starting with a letter. See the module docstring for
#: the measurement behind it.
MODEL_NAMED_COLUMN_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

#: The one top-level H5 object that is not an entity table. The country loader
#: (``USSingleYearDataset``) reads it as the dataset's period.
_TIME_PERIOD_KEY = "_time_period"
#: ``pandas_type`` attribute of a pandas ``format="table"`` frame.
_PANDAS_TABLE_FRAME = "frame_table"
#: ``pandas_type`` attribute of a pandas ``format="fixed"`` frame. The Frame
#: HDF boundary (``microcosm.frame.materialize.put_frame_table``) falls back to
#: it for a table holding a nullable Boolean with missing values.
_PANDAS_FIXED_FRAME = "frame"
#: The field a pandas table store writes for the frame index. The loader
#: restores it as the index, never as a column.
_PANDAS_INDEX_FIELD = "index"
#: Field names a pandas table store uses when ``data_columns`` was not set. The
#: stored column names are then hidden in pickled attributes, so the check
#: refuses the layout rather than guess.
_VALUES_BLOCK_FIELD = re.compile(r"values_block_\d+")

_REGISTER_PATH = "microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS"

_SUPPORT_SOURCE_ID = (
    "Support provenance: the {entity} record's assembly-unique pre-clone source "
    "id (microcosm.build.us_runtime.support_provenance."
    "support_source_id_column). It identifies the record across clones and "
    "enters the seeded-draw key grammar (microcosm.build.spec_engine.seeds); "
    "it is not a model input."
)
_SUPPORT_CHANNEL = (
    "Support provenance: the support channel the {entity} record came from, "
    "such as asec or puf_tax_detail (microcosm.build.us_runtime."
    "support_provenance.support_channel_column). Build lineage, not a model "
    "input."
)
_SUPPORT_CLONE_INDEX = (
    "Support provenance: the clone index of the {entity} record within its "
    "source record (microcosm.build.us_runtime.support_provenance."
    "support_clone_index_column). Build lineage, not a model input."
)
_POOLED_SOURCE = (
    "Pooled source provenance: {detail}. Assigned when a survey is pooled "
    "(microcosm.build.us_runtime.asec_pool for the ASEC, "
    "microcosm.build.us_runtime.acs_pums for the ACS) and carried through the "
    "outer stages as one of microcosm.build.outer_stage_runtime."
    "_POOLED_SOURCE_PROVENANCE_COLUMNS. {use} It is not a model input."
)
_POOLED_SOURCE_SEED_KEY = (
    "It identifies the source record and enters the seeded-draw key grammar "
    "(microcosm.build.spec_engine.seeds)."
)
_PUF_TAIL = (
    "PUF capital-gains own-tail transfer provenance ({detail}), declared in "
    "microcosm.build.us_runtime.puf_capital_gains_tail and read by "
    "assert_puf_capital_gains_tail_survives_selection. Build lineage, not a "
    "model input."
)
_TAX_UNIT_CONSTRUCTION = (
    "Tax-unit construction output: {detail} ({constant}). Later build stages "
    "read it (microcosm.build.us_runtime.us_late_producer_registry and the US "
    "imputation spec, microcosm/build/us/spec/imputation.yaml). "
    "policyengine-us 1.764.6 and 2.2.1 neither define nor read a variable of "
    "this name, and there is no live input to rename it to: the engine "
    "derives {engine} with formulas, which the release writer refuses to "
    "store."
)
#: The six US entity tables, in ``microcosm.frame.units.US_SCHEMA`` order.
#: microcosm-data depends on no other Microcosm shard, so the names are spelled
#: here; ``test_us_stored_input_register.py`` (microcosm-build) binds every
#: register entry to the producer definition it names.
_US_ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")

#: Reviewed register of lowercase stored columns that are deliberately not
#: policyengine-us variables: column -> why the engine may ignore it.
#:
#: Built from evidence, not guesses. It holds exactly the model-named
#: non-variable columns stored by the H5 files examined for microcosm#1026 that
#: are provenance or build construction outputs: the published default
#: ``populace-us-2024-spm-20260915`` (26 such columns, 24 registered) and the
#: Route A rehearsal export built from main's tools (30 such columns, all
#: registered). ``tests/fixtures/stored_input_inventories.json`` records both
#: files' stored columns, and a test requires every entry to appear in one of
#: them. Two model-named non-variable columns of the published default are
#: deliberately absent, because each is a stale model input rather than
#: metadata:
#:
#: - ``would_claim_wic``: the WIC take-up draw under its retired name. The live
#:   input is ``takes_up_wic_if_eligible`` (#746 moved the builder).
#: - ``medicare_part_b_premiums``: an ASEC ``PEMCPREM`` transfer target under a
#:   name no engine version defines. #590 dropped it from the build, and the
#:   rehearsal export no longer stores it. Neither policyengine-us 1.764.6 nor
#:   2.2.1 defines or reads it. The reported leaf both define,
#:   ``medicare_part_b_premiums_reported``, has no consumer in either, and the
#:   engine computes ``medicare_part_b_premium`` itself.
#:
#: An entry must never be a variable of the certified engine. The engine reads
#: such a column as an input, so the entry would be dead, and its reason (that
#: the engine ignores the column) would be false. Tests hold the register to
#: that rule against the installed engine.
US_STORED_NON_VARIABLE_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        **{
            f"{entity}_source_id": _SUPPORT_SOURCE_ID.format(entity=entity)
            for entity in _US_ENTITIES
        },
        **{
            f"{entity}_support_channel": _SUPPORT_CHANNEL.format(entity=entity)
            for entity in _US_ENTITIES
        },
        **{
            f"{entity}_support_clone_index": _SUPPORT_CLONE_INDEX.format(entity=entity)
            for entity in _US_ENTITIES
        },
        "source_year": _POOLED_SOURCE.format(
            detail="the source file's year (the ASEC year or the ACS vintage)",
            use=_POOLED_SOURCE_SEED_KEY,
        ),
        "source_household_id": _POOLED_SOURCE.format(
            detail=(
                "the source household id (the ASEC PH_SEQ, or the ACS household id)"
            ),
            use=_POOLED_SOURCE_SEED_KEY,
        ),
        "source_person_id": _POOLED_SOURCE.format(
            detail="the source person id (the ASEC PERIDNUM, or the ACS SPORDER)",
            use=_POOLED_SOURCE_SEED_KEY,
        ),
        "source_row_id": _POOLED_SOURCE.format(
            detail="the row position within the source file",
            use=(
                "It links a record back to its source row (for example "
                "microcosm.build.us_runtime.acs_release_predictors)."
            ),
        ),
        "tax_unit_role_input": _TAX_UNIT_CONSTRUCTION.format(
            detail="the HEAD / SPOUSE / DEPENDENT role microunit assigns each person",
            constant="microcosm.frame.units._TAX_UNIT_ROLE_COLUMN",
            engine="is_tax_unit_head, is_tax_unit_spouse and is_tax_unit_dependent",
        ),
        "filing_status_input": _TAX_UNIT_CONSTRUCTION.format(
            detail="the filing status microunit assigns each tax unit",
            constant="microcosm.frame.units.TAX_UNIT_FILING_STATUS_COLUMN",
            engine="filing_status",
        ),
        "puf_capital_gains_tail_transfer_applied": _PUF_TAIL.format(
            detail="whether this tax unit received a tail donor"
        ),
        "puf_capital_gains_tail_donor_source_id": _PUF_TAIL.format(
            detail="the donor PUF record id"
        ),
        "puf_capital_gains_tail_donor_is_synthetic": _PUF_TAIL.format(
            detail="whether the donor is a synthetic PUF record"
        ),
        "puf_capital_gains_tail_donor_filing_status_code": _PUF_TAIL.format(
            detail="the donor's PUF filing-status code"
        ),
        "puf_capital_gains_tail_donor_agi_band_index": _PUF_TAIL.format(
            detail="the donor's AGI band"
        ),
        "puf_capital_gains_tail_transfer_weight": _PUF_TAIL.format(
            detail="the donor weight carried by the transfer"
        ),
    }
)


class StoredTableLayoutError(ValueError):
    """An H5 whose stored tables the check cannot list from metadata."""


class StoredInputRefusalError(ValueError):
    """A release stores model-named columns its certified engine lacks.

    ``failures`` holds one :func:`stored_input_failures` line per refused
    column; the message joins them.
    """

    def __init__(self, failures: Iterable[str]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "stored-input contract refused the release: " + " ".join(self.failures)
        )


@dataclass(frozen=True)
class CertifiedEngine:
    """The engine a release is certified against, as the check sees it."""

    #: ``"policyengine-us <version>"``, used in every failure message.
    label: str
    #: Every variable name the engine defines, inputs and formulas alike.
    variables: frozenset[str]


def is_model_named(column: object) -> bool:
    """Whether ``column`` follows the policyengine-us variable naming convention."""

    return (
        isinstance(column, str)
        and MODEL_NAMED_COLUMN_PATTERN.fullmatch(column) is not None
    )


def undefined_stored_inputs(
    columns: Iterable[object],
    *,
    engine_variables: Collection[str],
    register: Mapping[str, str] = US_STORED_NON_VARIABLE_COLUMNS,
) -> tuple[str, ...]:
    """The stored columns the check refuses, sorted and without repeats.

    A column is refused if and only if it is model-named, is not one of
    ``engine_variables`` and is not a key of ``register``.
    """

    return tuple(
        sorted(
            {
                column
                for column in columns
                if is_model_named(column)
                and column not in engine_variables
                and column not in register
            }
        )
    )


def stored_input_failures(
    stored_tables: Mapping[str, Iterable[object]],
    *,
    engine: CertifiedEngine,
    register: Mapping[str, str] = US_STORED_NON_VARIABLE_COLUMNS,
) -> list[str]:
    """One failure line per refused column; an empty list is a pass.

    ``stored_tables`` maps each stored table (entity) to its column names. Each
    line quotes exactly one refused column and nothing else, names every table
    that stores it and says how to resolve it. Lines are in column order.
    """

    tables_by_column: dict[object, set[str]] = {}
    for table, columns in stored_tables.items():
        for column in columns:
            tables_by_column.setdefault(column, set()).add(str(table))
    refused = undefined_stored_inputs(
        tables_by_column, engine_variables=engine.variables, register=register
    )
    failures = []
    for column in refused:
        tables = sorted(tables_by_column[column])
        where = f"{', '.join(tables)} table{'s' if len(tables) > 1 else ''}"
        failures.append(
            f"stored column '{column}' ({where}) looks like a model input but "
            f"is not a variable in {engine.label}, so the engine ignores it. "
            "Rename it to the live input it stands for (or stop storing it if "
            "the build no longer produces that input), or, if it is "
            "deliberately not a model input, add a reviewed entry with its "
            f"reason to {_REGISTER_PATH}."
        )
    return failures


def register_consistency_failures(
    register: Mapping[str, str],
    *,
    engine_variables: Collection[str],
) -> list[str]:
    """Why ``register`` is not a sound register for this engine; empty if sound.

    Every entry must be model-named (any other column already passes by rule,
    so the entry would be dead), must not be an engine variable (the engine
    reads that column as an input, so the entry would be dead and its reason
    false) and must carry a non-blank reason.
    """

    failures = []
    for column, reason in sorted(register.items(), key=lambda item: repr(item[0])):
        if not is_model_named(column):
            failures.append(
                f"register entry {column!r} is not model-named, so the check "
                "never consults it."
            )
        elif column in engine_variables:
            failures.append(
                f"register entry {column!r} is a variable of the engine, which "
                "reads the stored column as an input: remove the entry."
            )
        if not isinstance(reason, str) or not reason.strip():
            failures.append(f"register entry {column!r} has no reason.")
    return failures


def register_sha256(
    register: Mapping[str, str] = US_STORED_NON_VARIABLE_COLUMNS,
) -> str:
    """SHA-256 of ``register`` as canonical JSON (sorted keys, no whitespace).

    A receipt that records it binds the exact register the check consulted, so
    a later register change is visible when the receipt is replayed.
    """

    payload = json.dumps(
        dict(register), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _attribute_text(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return None


def h5_stored_tables(path: Path | str) -> dict[str, tuple[str, ...]]:
    """Each stored table of a US H5 and its column names, in stored order.

    Reads HDF metadata only: the top-level keys, each group's ``pandas_type``
    attribute and each table's column labels (a ``format="table"`` table's
    compound field names, or a ``format="fixed"`` frame's ``axis0`` labels).
    No row is loaded. The layout is the one ``USSingleYearDataset`` reads: one
    pandas frame per entity plus the ``_time_period`` series. The pandas index
    field is not a stored column, and ``_time_period`` is not a table. Every
    other pandas frame is listed, whatever its key, so a column cannot hide in
    an unexpected table.

    Raises:
        StoredTableLayoutError: A top-level object is not a pandas frame, or a
            table hides its column names in ``values_block_*`` fields, so the
            check cannot see what it stores.
    """

    import h5py

    tables: dict[str, tuple[str, ...]] = {}
    with h5py.File(path, "r") as h5:
        for key in h5:
            if key == _TIME_PERIOD_KEY:
                continue
            group = h5[key]
            pandas_type = (
                _attribute_text(group.attrs.get("pandas_type"))
                if isinstance(group, h5py.Group)
                else None
            )
            if pandas_type == _PANDAS_TABLE_FRAME:
                table = group.get("table")
                names = table.dtype.names if isinstance(table, h5py.Dataset) else None
                if not names or names[0] != _PANDAS_INDEX_FIELD:
                    raise StoredTableLayoutError(
                        f"{path}: table {key!r} has no pandas index field, so "
                        "the stored-input check cannot read its columns."
                    )
                hidden = [name for name in names if _VALUES_BLOCK_FIELD.fullmatch(name)]
                if hidden:
                    raise StoredTableLayoutError(
                        f"{path}: table {key!r} stores {hidden} blocks, which "
                        "hide its column names; write it with data_columns=True."
                    )
                tables[key] = tuple(names[1:])
            elif pandas_type == _PANDAS_FIXED_FRAME:
                labels = group.get("axis0")
                if not isinstance(labels, h5py.Dataset) or labels.ndim != 1:
                    raise StoredTableLayoutError(
                        f"{path}: fixed frame {key!r} has no column axis."
                    )
                names = tuple(_attribute_text(label) for label in labels[()])
                if any(name is None for name in names):
                    raise StoredTableLayoutError(
                        f"{path}: fixed frame {key!r} has non-text column labels."
                    )
                tables[key] = names
            else:
                raise StoredTableLayoutError(
                    f"{path}: top-level object {key!r} is not a pandas entity "
                    "frame the stored-input check can read."
                )
    return tables


def installed_us_engine() -> CertifiedEngine:
    """The installed policyengine-us, as :class:`CertifiedEngine`.

    Reads the module-level tax-benefit system that importing
    ``policyengine_us.system`` builds, the same one the native-loader
    compatibility probe reads, so the check constructs no second system.

    Raises:
        ImportError: policyengine-us is not installed. A release cannot be
            certified without its engine, so callers fail closed.
    """

    from importlib import metadata

    from policyengine_us.system import system

    return CertifiedEngine(
        label=f"policyengine-us {metadata.version('policyengine-us')}",
        variables=frozenset(system.variables),
    )


def require_h5_stored_inputs(
    path: Path | str,
    *,
    engine: CertifiedEngine,
    register: Mapping[str, str] = US_STORED_NON_VARIABLE_COLUMNS,
) -> dict[str, object]:
    """Refuse an H5 that stores a model input ``engine`` lacks; else summarize.

    The summary is what a certification receipt records: the register's digest
    and the registered non-variable columns the H5 stores. It names no engine
    version, because the receipt already records the tested packages.

    Raises:
        StoredInputRefusalError: The H5 stores at least one refused column.
        StoredTableLayoutError: The H5's stored tables cannot be listed.
    """

    tables = h5_stored_tables(path)
    failures = stored_input_failures(tables, engine=engine, register=register)
    if failures:
        raise StoredInputRefusalError(failures)
    columns = {column for names in tables.values() for column in names}
    return {
        "register_sha256": register_sha256(register),
        "registered_non_variables": sorted(
            column
            for column in columns
            if is_model_named(column)
            and column not in engine.variables
            and column in register
        ),
    }


def h5_verdict(
    path: Path | str,
    *,
    engine: CertifiedEngine,
    register: Mapping[str, str] = US_STORED_NON_VARIABLE_COLUMNS,
) -> dict[str, object]:
    """The check's full reading of one H5, for an operator or a report."""

    tables = h5_stored_tables(path)
    columns = {column for names in tables.values() for column in names}
    failures = stored_input_failures(tables, engine=engine, register=register)
    outside_convention = {column for column in columns if not is_model_named(column)}
    return {
        "path": str(path),
        "engine": engine.label,
        "passed": not failures,
        "refused": list(
            undefined_stored_inputs(
                columns, engine_variables=engine.variables, register=register
            )
        ),
        "failures": failures,
        "columns_per_table": {table: len(names) for table, names in tables.items()},
        "stored_columns": len(columns),
        "registered_non_variables": sorted(
            column
            for column in columns
            if is_model_named(column)
            and column not in engine.variables
            and column in register
        ),
        "outside_naming_convention": len(outside_convention),
        "outside_naming_convention_that_are_engine_variables": sorted(
            outside_convention & engine.variables
        ),
        "register_sha256": register_sha256(register),
        "register_consistency_failures": register_consistency_failures(
            register, engine_variables=engine.variables
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Check local US H5 files against the installed policyengine-us.

    Prints one JSON verdict per file and exits 1 if any file is refused or
    cannot be read. Only HDF metadata is read, so a multi-gigabyte release H5
    costs seconds and no row memory.
    """

    parser = argparse.ArgumentParser(
        prog="python -m microcosm.data.stored_inputs", description=main.__doc__
    )
    parser.add_argument("h5", nargs="+", type=Path, help="US release H5 file(s).")
    args = parser.parse_args(argv)
    engine = installed_us_engine()
    verdicts: list[dict[str, object]] = []
    for path in args.h5:
        try:
            verdicts.append(h5_verdict(path, engine=engine))
        except (OSError, StoredTableLayoutError) as error:
            verdicts.append(
                {
                    "path": str(path),
                    "engine": engine.label,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    json.dump(verdicts, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if all(verdict["passed"] for verdict in verdicts) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
