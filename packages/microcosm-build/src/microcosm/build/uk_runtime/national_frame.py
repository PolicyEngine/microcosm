"""The UK national build's Frame carrier: schema, construction, provenance.

DESIGN.md makes :class:`microcosm.frame.Frame` the atom of the stack —
structure established once at assembly, every operator working on the
validated bundle. This module is the UK national build's construction seam
for that carrier (#612): the schema constant, the constructor helper (which
is where Frame's linkage validation runs), the metadata accessors, the UK
residue validation, and the provenance record that travels *beside* the
Frame. Provenance cannot live in the Frame: ``Frame.metadata`` freezes to
plain values (no ``Path``), ``__slots__`` blocks attributes, and Frame
operations hard-construct ``Frame(...)`` so a subclass sheds its type.

Mass accounting rule for stages: a table-reshaping stage hard-constructs a
new Frame via :func:`uk_national_frame` with an explicitly extended
``mass_log``; a weight-only update on unchanged tables goes through
:meth:`Frame.with_weights`, which enforces the forward-only kind transition
and appends the record itself.

The ``household_weight`` column itself is a materialized export contract,
not carrier state (``engine_tables`` regenerates it from the typed weights
at every export boundary). The H5 loader validates stored-vs-typed
agreement before consuming the column into the Frame's typed vector.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.frame import (
    EntitySchema,
    Frame,
    MassChangeRecord,
    WeightKind,
    Weights,
    engine_tables,
    put_frame_table,
    read_frame_table,
)

__all__ = [
    "UK_BUILD_IDENTIFIER_COLUMNS",
    "UK_ENGINE_IDENTIFIER_MAX",
    "UK_ENERGY_PRICE_BASIS",
    "UK_ENERGY_PRICE_BASIS_ATTR",
    "UK_ENERGY_PRICE_BASIS_COLUMNS",
    "UK_EXPORTED_WEIGHT_COLUMNS",
    "UK_HOUSEHOLD_WEIGHT_KIND_ATTR",
    "UK_MASS_LOG_ATTR",
    "UK_NATIONAL_H5_TABLES",
    "UK_NATIONAL_SCHEMA",
    "UK_TIME_PERIOD_METADATA_KEY",
    "UKNationalStage",
    "UKStagingProvenance",
    "load_uk_national_frame",
    "uk_build_identifier_frame",
    "uk_build_identifier_tables",
    "uk_build_identifiers",
    "require_uk_producer_isa_income_contract",
    "uk_household_weight_kind",
    "uk_identifier_range_failures",
    "uk_isa_income_contract",
    "uk_national_frame",
    "uk_release_identifier_tables",
    "uk_household_mass_conservation_receipt",
    "uk_time_period",
    "validate_uk_national_frame",
    "uk_release_export_frame",
    "write_uk_national_frame",
]

UK_NATIONAL_SCHEMA = EntitySchema(group_entities=("benunit", "household"))

UK_TIME_PERIOD_METADATA_KEY = "time_period"

UK_NATIONAL_H5_TABLES = ("person", "benunit", "household", "time_period")
UK_HOUSEHOLD_WEIGHT_KIND_ATTR = "populace_household_weight_kind"
UK_MASS_LOG_ATTR = "populace_mass_log_json"
#: The root attribute in which policyengine-uk reads the price period of a
#: dataset's stored energy spending (policyengine-uk 2.148.1,
#: PolicyEngine/policyengine-uk#2571): a JSON object of ``period_type`` and
#: ``period_value``. The engine projects declared energy spending from that
#: period by its consumer price index; spending without a declaration stays
#: at its stored prices in every later year (PolicyEngine/policyengine-uk#2526).
UK_ENERGY_PRICE_BASIS_ATTR = "energy_price_basis"
#: The household spending the declaration describes: the three columns
#: policyengine-uk projects only from a declared period.
UK_ENERGY_PRICE_BASIS_COLUMNS = (
    "domestic_energy_consumption",
    "electricity_consumption",
    "gas_consumption",
)
#: Their price period. ``lcfs_consumption``, the one stage that writes them,
#: sets every household spend column at calendar-2024 prices, and the
#: stage-time gate ``uk_stage_lcfs_consumption_basis`` holds each step that
#: sets a level to that year (microcosm#1113).
UK_ENERGY_PRICE_BASIS: Mapping[str, object] = MappingProxyType(
    {"period_type": "calendar_year", "period_value": 2024}
)

# The persisted weight columns of the materialized export contract above:
# real columns on the tables, but plumbing rather than input mass. Consumers
# that total column mass (the input-mass parity surface, its reference
# emitter) must exclude them — the frame schema cannot, because the column
# is deliberately load-bearing here.
UK_EXPORTED_WEIGHT_COLUMNS = frozenset({"household_weight"})


def _assert_uk_benunit_nesting(person: pd.DataFrame) -> None:
    required = {"person_benunit_id", "person_household_id"}
    if not required.issubset(person.columns):
        # Belt-and-braces, not an escape hatch: at the validator seam the
        # Frame guarantees both membership columns exist, and at the
        # construction seam a missing column is refused a line later by the
        # kernel's own linkage validation, which names it.
        return
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    split = placements.loc[
        placements["person_benunit_id"].duplicated(keep=False),
        "person_benunit_id",
    ].drop_duplicates()
    if not split.empty:
        raise ValueError(
            "each benunit must belong to exactly one household; split "
            f"benunit id(s): {split.head(5).tolist()}."
        )


@dataclass(frozen=True)
class _UKSourceFileFingerprint:
    """Cheap stable-file identity used to bind a prior hash to an H5 load.

    Scope-reduced by #612 increment 3: stat identity (device/inode/mtime)
    cannot survive a file copy or a machine move, so it no longer carries
    any *run* identity — that role belongs to content addressing
    (``uk_frame_content_identity`` in the descent fences, the checkpointed
    build's content-addressed run config). What stays here is exactly what
    stat identity is good at: the mid-read race guard (the file must not
    change while it is being loaded) and re-binding the certified-candidate
    hash to the same on-disk file within one process.
    """

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _uk_source_file_fingerprint(path: Path) -> _UKSourceFileFingerprint:
    stat = path.stat()
    return _UKSourceFileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _weight_kind_from_stored(value: object) -> WeightKind:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return WeightKind(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown stored UK household weight kind {value!r}.") from exc


def _read_weight_metadata(path: Path) -> tuple[object, object]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to read UK national metadata.") from exc
    with h5py.File(path, mode="r") as file:
        return (
            file.attrs.get(
                UK_HOUSEHOLD_WEIGHT_KIND_ATTR,
                WeightKind.DESIGN.value,
            ),
            file.attrs.get(UK_MASS_LOG_ATTR, "[]"),
        )


def _write_weight_metadata(
    path: Path,
    *,
    weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to write UK national metadata.") from exc
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_HOUSEHOLD_WEIGHT_KIND_ATTR] = weight_kind.value
        file.attrs[UK_MASS_LOG_ATTR] = json.dumps(
            [_mass_change_record_payload(record) for record in mass_log],
            sort_keys=True,
        )


def _write_energy_price_basis(path: Path, household: pd.DataFrame) -> None:
    """Declare the energy spending's price period in an H5 that stores it.

    The JSON is written as policyengine-uk writes its own declaration, keys
    sorted, so the engine reads the basis the build priced the columns at.
    """

    if not any(column in household.columns for column in UK_ENERGY_PRICE_BASIS_COLUMNS):
        return
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to write UK national metadata.") from exc
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_ENERGY_PRICE_BASIS_ATTR] = json.dumps(
            dict(UK_ENERGY_PRICE_BASIS), sort_keys=True
        )


def _write_isa_income_contract(path: Path, contract: Mapping[str, object]) -> None:
    """Declare how the H5's savings interest holds ISA interest.

    policyengine-uk 2.162.2 reads the contract from this root attribute: the
    frame's own declaration, so a file never claims a layout its columns do not
    hold. Written as the engine writes its own declaration, keys sorted
    (microcosm#1095, microcosm#1360).
    """

    from microcosm.frame.adapters.policyengine_uk import (
        UK_ISA_INCOME_CONTRACT_ATTR,
        uk_isa_income_contract_from,
    )

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to write UK national metadata.") from exc
    with h5py.File(path, mode="r+") as file:
        file.attrs[UK_ISA_INCOME_CONTRACT_ATTR] = json.dumps(
            uk_isa_income_contract_from(contract), sort_keys=True
        )


def _read_isa_income_contract(path: Path) -> dict[str, object]:
    """The H5's declared contract; a file declaring none reads as legacy gross."""

    from microcosm.frame.adapters.policyengine_uk import (
        UK_ISA_INCOME_CONTRACT_ATTR,
        UK_LEGACY_ISA_INCOME_CONTRACT,
        uk_isa_income_contract_from,
    )

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - UK H5 runtime dependency
        raise RuntimeError("h5py is required to read UK national metadata.") from exc
    with h5py.File(path, mode="r") as file:
        stored = file.attrs.get(UK_ISA_INCOME_CONTRACT_ATTR)
    if stored is None:
        return dict(UK_LEGACY_ISA_INCOME_CONTRACT)
    return uk_isa_income_contract_from(stored)


def _mass_log_from_stored(value: object) -> tuple[MassChangeRecord, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("Stored UK microcosm mass log is not valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("Stored UK microcosm mass log must be a JSON list.")
    records: list[MassChangeRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Stored UK microcosm mass-log entries must be objects.")
        try:
            records.append(
                MassChangeRecord(
                    entity=str(entry["entity"]),
                    old_total=float(entry["old_total"]),
                    new_total=float(entry["new_total"]),
                    declared_factor=(
                        None
                        if entry.get("declared_factor") is None
                        else float(entry["declared_factor"])
                    ),
                    reason=str(entry["reason"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Stored UK microcosm mass-log entry is malformed."
            ) from exc
    return tuple(records)


def _mass_change_record_payload(record: MassChangeRecord) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }


@dataclass(frozen=True)
class UKStagingProvenance:
    """Where a UK national frame's bytes came from, carried beside it.

    The same idiom as ``UKCertifiedCandidateIdentity``: provenance is its own
    frozen record next to the data, never a field smuggled into the carrier.
    """

    source_h5: Path
    fingerprint: _UKSourceFileFingerprint


@dataclass(frozen=True)
class UKNationalStage:
    """One named ``Frame -> Frame`` national transform.

    The June national driver is retired, but a few stage factories still expose
    this small callable container as their compatibility surface.
    """

    name: str
    transform: Callable[[Frame], Frame]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UKNationalStage.name must be non-empty.")
        if not callable(self.transform):
            raise TypeError("UKNationalStage.transform must be callable.")

    def run(self, frame: Frame) -> Frame:
        result = self.transform(frame)
        if not isinstance(result, Frame):
            raise TypeError(
                f"UK national stage {self.name!r} must return a microcosm Frame, "
                f"got {type(result).__name__}."
            )
        return result


def _read_uk_national_tables(
    path: str | Path,
) -> tuple[dict[str, Any], _UKSourceFileFingerprint, Path]:
    """Read a compact UK single-year H5's payload with the race guard."""

    requested_path = Path(path).expanduser()
    if requested_path.suffix != ".h5":
        raise ValueError("UK national dataset path must end with '.h5'.")
    input_path = requested_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"UK national dataset not found: {input_path}.")

    fingerprint_before = _uk_source_file_fingerprint(input_path)
    stored_kind, stored_mass_log = _read_weight_metadata(input_path)
    isa_income_contract = _read_isa_income_contract(input_path)
    with pd.HDFStore(input_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(UK_NATIONAL_H5_TABLES) - keys)
        if missing:
            raise ValueError(f"UK national dataset is missing table(s): {missing}.")
        raw_period = store["time_period"]
        if len(raw_period) != 1:
            raise ValueError(
                "UK national dataset time_period must contain exactly one value."
            )
        payload = {
            "person": read_frame_table(store, "person"),
            "benunit": read_frame_table(store, "benunit"),
            "household": read_frame_table(store, "household"),
            "time_period": str(raw_period.iloc[0]),
            "household_weight_kind": _weight_kind_from_stored(stored_kind),
            "mass_log": _mass_log_from_stored(stored_mass_log),
            "isa_income_contract": isa_income_contract,
        }
    fingerprint_after = _uk_source_file_fingerprint(input_path)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "UK national source H5 changed while it was being loaded; refusing "
            "to bind mixed or stale bytes to build stages."
        )
    return payload, fingerprint_after, input_path


def load_uk_national_frame(
    path: str | Path,
) -> tuple[Frame, UKStagingProvenance]:
    """Load a compact UK single-year H5 as a validated Frame plus provenance."""

    payload, fingerprint, input_path = _read_uk_national_tables(path)
    frame = uk_national_frame(
        person=payload["person"],
        benunit=payload["benunit"],
        household=payload["household"],
        time_period=payload["time_period"],
        weight_kind=payload["household_weight_kind"],
        mass_log=payload["mass_log"],
        isa_income_contract=payload["isa_income_contract"],
    )
    validate_uk_national_frame(frame)
    return frame, UKStagingProvenance(source_h5=input_path, fingerprint=fingerprint)


def uk_national_frame(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: int | str,
    weight_kind: WeightKind = WeightKind.DESIGN,
    household_weights: Sequence[float] | np.ndarray | pd.Series | None = None,
    mass_log: tuple[MassChangeRecord, ...] = (),
    isa_income_contract: Mapping[str, object] | None = None,
) -> Frame:
    """Assemble the UK national tables into a validated Frame.

    ``isa_income_contract`` is how the savings columns hold ISA interest; the
    producer's own contract unless the tables came from a file declaring
    another (``load_uk_national_frame``).

    The typed household weights are built from ``household_weights`` when
    supplied, otherwise from a legacy input ``household_weight`` column. The
    stored carrier table never keeps that export column; materialization
    regenerates it from the typed vector. The Frame constructor enforces the
    structural invariants the shadow carrier never checked: group ids unique
    and sorted ascending, membership equality in both directions, global
    column uniqueness, weight health, and the UK invariant that a benunit's
    members share a household.
    """

    period = "" if time_period is None else str(time_period)
    if not period.strip():
        raise ValueError("UK national frame time_period must be a non-empty string.")
    if household_weights is None and "household_weight" not in household.columns:
        raise ValueError(
            "uk_national_frame requires household_weights or a household_weight "
            "input column to seed the typed household vector."
        )
    weight_values = (
        household_weights
        if household_weights is not None
        else household["household_weight"].to_numpy(dtype="float64")
    )
    weights = Weights(
        values=np.asarray(weight_values, dtype="float64"),
        kind=weight_kind,
    )
    _assert_uk_benunit_nesting(person)
    carrier_household = household.drop(columns=["household_weight"], errors="ignore")
    from microcosm.frame.adapters.policyengine_uk import (
        UK_ISA_INCOME_CONTRACT,
        UK_ISA_INCOME_CONTRACT_ATTR,
        uk_isa_income_contract_from,
    )

    contract = uk_isa_income_contract_from(
        UK_ISA_INCOME_CONTRACT if isa_income_contract is None else isa_income_contract
    )
    return Frame(
        tables={"person": person, "benunit": benunit, "household": carrier_household},
        schema=UK_NATIONAL_SCHEMA,
        weights={"household": weights},
        mass_log=mass_log,
        # The contract travels as the H5 attribute's JSON, which the graph's
        # frame-metadata store serialises as it is.
        metadata={
            UK_TIME_PERIOD_METADATA_KEY: period,
            UK_ISA_INCOME_CONTRACT_ATTR: json.dumps(contract, sort_keys=True),
        },
    )


def uk_isa_income_contract(frame: Frame) -> dict[str, object]:
    """How the frame's savings columns hold ISA interest (microcosm#1360)."""

    from microcosm.frame.adapters.policyengine_uk import frame_isa_income_contract

    return frame_isa_income_contract(frame)


def require_uk_producer_isa_income_contract(frame: Frame, *, label: str) -> None:
    """Refuse a build input whose savings columns are not the producer's.

    A stage writes the producer's columns under the producer's declaration, so
    an input holding another layout (an older gross-interest file) would come
    out mislabelled; it is rebuilt from source instead.
    """

    from microcosm.frame.adapters.policyengine_uk import UK_ISA_INCOME_CONTRACT

    declared = uk_isa_income_contract(frame)
    if declared != dict(UK_ISA_INCOME_CONTRACT):
        raise ValueError(
            f"{label} declares the ISA income contract {declared}, not the "
            f"producer's {dict(UK_ISA_INCOME_CONTRACT)}; rebuild it from source."
        )


def uk_household_mass_conservation_receipt(
    frame: Frame, reason: str
) -> MassChangeRecord:
    """The receipt a column-writing stage records: the household weights pass
    through untouched, so old and new totals agree and the declared factor is
    1.0. The terminal family gate requires exactly this record under the
    stage's own reason, so a build whose stage moved household mass, or never
    ran, fails by name (the E8 pattern, now on every source stage)."""

    total = float(frame.weights_for("household").total)
    return MassChangeRecord(
        entity="household",
        old_total=total,
        new_total=total,
        declared_factor=1.0,
        reason=reason,
    )


def uk_time_period(frame: Frame) -> str:
    """The frame's UK time period label (e.g. ``"2023"``), from metadata."""

    period = frame.metadata.get(UK_TIME_PERIOD_METADATA_KEY)
    if not isinstance(period, str) or not period.strip():
        raise ValueError(
            "UK national frame time_period metadata must be a non-empty string."
        )
    return period


def uk_household_weight_kind(frame: Frame) -> WeightKind:
    """The kind of the frame's explicit household weights."""

    return frame.weights_for("household").kind


def _write_uk_single_year_tables(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    time_period: str,
    weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
    isa_income_contract: Mapping[str, object],
    path: Path,
) -> Path:
    """The one physical writer for every UK single-year H5."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5")
    try:
        with pd.HDFStore(temporary_path) as store:
            put_frame_table(
                store,
                "person",
                person,
                preferred_format="table",
                data_columns=True,
            )
            put_frame_table(
                store,
                "benunit",
                benunit,
                preferred_format="table",
                data_columns=True,
            )
            put_frame_table(
                store,
                "household",
                household,
                preferred_format="table",
                data_columns=True,
            )
            store.put(
                "time_period",
                pd.Series([time_period]),
                format="table",
                data_columns=True,
            )
        _write_weight_metadata(
            temporary_path, weight_kind=weight_kind, mass_log=mass_log
        )
        _write_energy_price_basis(temporary_path, household)
        _write_isa_income_contract(temporary_path, isa_income_contract)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


#: Columns the release boundary drops from the written candidate (microcosm#1063
#: c9): the enhanced FRS stores ``incapacity_benefit_reported`` as an all-zero
#: legacy layer and the export-surface gate's reviewed exclusion requires the
#: candidate to drop it. The spine keeps the column (the SPI income stage
#: reads it); only the release H5 and its readback expectation lose it.
#: ``terminal_gates.UK_REVIEWED_EXPORT_EXCLUSIONS`` names the first; the
#: five internal disability carriers (``frs_disability``'s
#: ``UK_INTERNAL_DISABILITY_REPORTED_COLUMNS``) are stage-internal inputs
#: that the export registers keep off the release (review of #1089).
#:
#: ``household.property_wealth`` (microcosm#1106, the uk-data#543 defect) is
#: formula-owned in policyengine-uk: it adds ``residential_property_value``
#: (itself ``main_residence_value`` + ``other_residential_property_value``)
#: and ``non_residential_property_value``. A persisted copy overrides that
#: sum, and the variable has no uprating index, so a saved WAS total stays at
#: its base-year level in every later year while the components grow (the
#: published aa31bdf6 year files hold it at growth 1.0000 from 2026 to 2030
#: against 1.1303 for the sum), and it also counts ``owned_land`` and the
#: ``other_property_value`` remainder the engine sum leaves out. The spine
#: keeps it (the WAS stage's identity gate, donor-range cap and later
#: segments read it); the engine derives it from the persisted components.
UK_RELEASE_EXPORT_DROPPED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "person": (
        "incapacity_benefit_reported",
        "attendance_allowance_reported",
        "dla_sc_reported",
        "dla_m_reported",
        "pip_m_reported",
        "pip_dl_reported",
        # The salary-sacrifice conversion's pre-conversion pay carrier
        # (salary_sacrifice.SALSAC_PRE_CONVERSION_PAY_COLUMN): it exists so
        # the identity receipts can reverse the stage's in-place rewrite of a
        # converted record's pay, and is no engine input.
        "salary_sacrifice_pre_conversion_pay",
        # The landlord carrier (frs_spine.frs_reports_rent_from_other_property)
        # that keeps loss-making landlords in the CGT residential signal; it is
        # no engine input.
        "reports_rent_from_other_property",
        # The FRS bonus carrier (frs_spine.FRS_BONUS_CARRIER_COLUMN) the bonus
        # sacrifice conversion reads (microcosm#1340); it is no engine input.
        "frs_bonus_amount",
        # The in-household care hours carrier
        # (frs_spine.FRS_IN_HOUSEHOLD_CARE_HOURS_COLUMN) the Carer's Allowance
        # draw reads (microcosm#1418); it is no engine input.
        "frs_care_hours_in_household",
        # policyengine-uk's sum of the ISA interest parts (microcosm#1360). The
        # producer stores individual_savings_account_interest_income instead,
        # so no stage writes it; a persisted copy would override the sum and
        # never be uprated, so the boundary drops one should it appear.
        "tax_free_savings_income",
    ),
    # The value the spine's regional home values replaced (microcosm#1112):
    # the dense node ranks on it, and it is no engine input.
    "household": ("property_wealth", "main_residence_value_rank_basis"),
}


#: The largest identifier policyengine-uk stores. Its five identifier
#: variables (``person_id``, ``benunit_id``, ``household_id``,
#: ``person_benunit_id``, ``person_household_id``) are 32-bit integers, and a
#: larger stored value wraps (microcosm#1194, pe-uk#2273).
UK_ENGINE_IDENTIFIER_MAX = 2_147_483_647

#: The column that keeps each entity's build identifier beside its release
#: identifier. The clone stages number copies by adding a power of ten per
#: copy, so a built frame's identifiers run far past the engine's range; every
#: draw in the build is keyed on them, so the build keeps them and the release
#: boundary renumbers. The lineage columns (``*_source_id``, the clone
#: indices) describe the build identifier.
UK_BUILD_IDENTIFIER_COLUMNS: Mapping[str, str] = {
    "person": "person_build_id",
    "benunit": "benunit_build_id",
    "household": "household_build_id",
}

_UK_ENTITY_IDENTIFIER_COLUMNS: Mapping[str, str] = {
    "person": "person_id",
    "benunit": "benunit_id",
    "household": "household_id",
}
_UK_PERSON_MEMBERSHIP_COLUMNS: Mapping[str, str] = {
    "benunit": "person_benunit_id",
    "household": "person_household_id",
}


def _uk_build_identifier_state(tables: Mapping[str, pd.DataFrame]) -> bool:
    """Whether the tables carry the build-identifier columns: all or none."""

    present = [
        entity
        for entity, column in UK_BUILD_IDENTIFIER_COLUMNS.items()
        if column in tables[entity].columns
    ]
    if present and len(present) != len(UK_BUILD_IDENTIFIER_COLUMNS):
        missing = sorted(set(UK_BUILD_IDENTIFIER_COLUMNS) - set(present))
        raise ValueError(
            "UK tables carry a build identifier on some entities only; "
            f"{missing} lack theirs."
        )
    return bool(present)


def _uk_mapped_identifiers(
    values: np.ndarray, source: np.ndarray, target: np.ndarray, *, label: str
) -> np.ndarray:
    """``values`` read through the one-to-one map ``source[i] -> target[i]``."""

    order = np.argsort(source, kind="stable")
    ordered = source[order]
    if len(ordered) > 1 and (ordered[1:] == ordered[:-1]).any():
        raise ValueError(f"UK {label} identifiers are not unique.")
    position = np.searchsorted(ordered, values)
    inside = position < len(ordered)
    if not inside.all() or not (ordered[position[inside]] == values[inside]).all():
        raise ValueError(f"UK {label} memberships name an identifier with no row.")
    return target[order][position]


def uk_release_identifier_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    universe: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """The three tables with identifiers inside the engine's range (microcosm#1194).

    Per entity the release identifier is one plus the rank of the build
    identifier among the entity's rows, so row order is untouched and the
    numbering depends on nothing but the identifiers themselves. The person
    table's two memberships map through the same arrays. The build identifier
    stays beside each release identifier (``UK_BUILD_IDENTIFIER_COLUMNS``).

    ``universe`` is the tables the release holds when ``tables`` are a subset
    of their rows (a per-clone engine block): the rank is then taken in the
    universe, so each row gets the identifier it has in the release.

    Idempotent: tables that already carry the build identifiers are returned
    as they are.
    """

    if _uk_build_identifier_state(tables):
        return dict(tables)
    whole = tables if universe is None else universe
    result: dict[str, pd.DataFrame] = {}
    whole_build: dict[str, np.ndarray] = {}
    whole_release: dict[str, np.ndarray] = {}
    for entity, column in _UK_ENTITY_IDENTIFIER_COLUMNS.items():
        whole_build[entity] = whole[entity][column].to_numpy(dtype=np.int64)
        order = np.argsort(whole_build[entity], kind="stable")
        ordered = whole_build[entity][order]
        if len(ordered) > 1 and (ordered[1:] == ordered[:-1]).any():
            raise ValueError(f"UK {entity} identifiers are not unique.")
        whole_release[entity] = np.empty(len(order), dtype=np.int64)
        whole_release[entity][order] = np.arange(1, len(order) + 1, dtype=np.int64)
        table = tables[entity]
        build = table[column].to_numpy(dtype=np.int64)
        release = (
            whole_release[entity]
            if universe is None
            else _uk_mapped_identifiers(
                build, whole_build[entity], whole_release[entity], label=entity
            )
        )
        renumbered = table.copy()
        renumbered[column] = release.astype(table[column].dtype)
        renumbered.insert(
            renumbered.columns.get_loc(column) + 1,
            UK_BUILD_IDENTIFIER_COLUMNS[entity],
            build,
        )
        result[entity] = renumbered
    person = result["person"]
    for entity, column in _UK_PERSON_MEMBERSHIP_COLUMNS.items():
        person[column] = _uk_mapped_identifiers(
            person[column].to_numpy(dtype=np.int64),
            whole_build[entity],
            whole_release[entity],
            label=entity,
        ).astype(tables["person"][column].dtype)
    return result


def uk_build_identifier_tables(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """The inverse of :func:`uk_release_identifier_tables`.

    A reader that replays the build's identifier-keyed draws, or joins a
    release to its spine, needs the build identifiers back. Tables that
    carry none are returned as they are.
    """

    if not _uk_build_identifier_state(tables):
        return dict(tables)
    result: dict[str, pd.DataFrame] = {}
    build: dict[str, np.ndarray] = {}
    release: dict[str, np.ndarray] = {}
    for entity, column in _UK_ENTITY_IDENTIFIER_COLUMNS.items():
        table = tables[entity]
        carrier = UK_BUILD_IDENTIFIER_COLUMNS[entity]
        build[entity] = table[carrier].to_numpy(dtype=np.int64)
        release[entity] = table[column].to_numpy(dtype=np.int64)
        restored = table.drop(columns=[carrier])
        restored[column] = build[entity].astype(table[column].dtype)
        result[entity] = restored
    person = result["person"]
    for entity, column in _UK_PERSON_MEMBERSHIP_COLUMNS.items():
        person[column] = _uk_mapped_identifiers(
            person[column].to_numpy(dtype=np.int64),
            release[entity],
            build[entity],
            label=entity,
        ).astype(tables["person"][column].dtype)
    return result


def uk_identifier_range_failures(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Why written identifiers would not survive policyengine-uk, if they would not.

    The engine stores them as 32-bit integers: a value above
    ``UK_ENGINE_IDENTIFIER_MAX`` wraps, and two that wrap to the same value
    merge. The messages name the column only: never an identifier, and never a
    count of rows.
    """

    failures: list[str] = []
    columns = [
        *(
            (entity, column, True)
            for entity, column in _UK_ENTITY_IDENTIFIER_COLUMNS.items()
        ),
        *(
            ("person", column, False)
            for column in _UK_PERSON_MEMBERSHIP_COLUMNS.values()
        ),
    ]
    for entity, column, unique in columns:
        if column not in tables[entity].columns:
            failures.append(f"Written {entity} table lacks {column!r}.")
            continue
        values = pd.to_numeric(tables[entity][column], errors="coerce")
        if values.isna().any() or (values != np.floor(values)).any():
            failures.append(f"Written {column!r} is not a whole number on every row.")
            continue
        if (values < 1).any():
            failures.append(f"Written {column!r} has a value below 1.")
        if (values > UK_ENGINE_IDENTIFIER_MAX).any():
            failures.append(
                f"Written {column!r} has a value above the 32-bit range "
                "policyengine-uk stores."
            )
        if unique and values.duplicated().any():
            failures.append(f"Written {column!r} repeats a value.")
    return failures


def uk_build_identifiers(table: pd.DataFrame, entity: str) -> np.ndarray:
    """One entity table's build identifiers, whether or not it was renumbered."""

    carrier = UK_BUILD_IDENTIFIER_COLUMNS[entity]
    column = (
        carrier if carrier in table.columns else _UK_ENTITY_IDENTIFIER_COLUMNS[entity]
    )
    return table[column].to_numpy(dtype=np.int64)


def uk_build_identifier_frame(frame: Frame) -> Frame:
    """A loaded release frame back on its build identifiers; others unchanged."""

    tables = {
        entity: frame.table(entity) for entity in ("person", "benunit", "household")
    }
    if not _uk_build_identifier_state(tables):
        return frame
    restored = uk_build_identifier_tables(tables)
    return uk_national_frame(
        person=restored["person"],
        benunit=restored["benunit"],
        household=restored["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def uk_release_export_frame(frame: Frame) -> Frame:
    """The frame the release boundary writes: reviewed export exclusions dropped.

    The three area codes leave under the consumers' names (microcosm#1114,
    ``geography_ladder.UK_EXPORT_AREA_CODE_COLUMNS``); a national frame
    carries none. The identifiers leave inside the 32-bit range the engine
    stores, with the build identifiers beside them (microcosm#1194,
    :func:`uk_release_identifier_tables`). A frame that already left passes
    unchanged.
    """

    from .geography_ladder import export_area_code_columns

    tables: dict[str, pd.DataFrame] = {}
    changed = False
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity)
        dropped = [
            column
            for column in UK_RELEASE_EXPORT_DROPPED_COLUMNS.get(entity, ())
            if column in table.columns
        ]
        if dropped:
            table = table.drop(columns=dropped)
            changed = True
        if entity == "household":
            renamed = export_area_code_columns(table)
            if renamed is not table:
                table = renamed
                changed = True
        tables[entity] = table
    if not _uk_build_identifier_state(tables):
        tables = uk_release_identifier_tables(tables)
        changed = True
    if not changed:
        return frame
    return uk_national_frame(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
        isa_income_contract=uk_isa_income_contract(frame),
    )


def write_uk_national_frame(frame: Frame, path: str | Path) -> Path:
    """Atomically write a validated UK national Frame as a staging H5."""

    validate_uk_national_frame(frame)
    output_path = Path(path)
    if output_path.suffix != ".h5":
        raise ValueError("UK national staging path must end with '.h5'.")
    tables = engine_tables(frame, weighted_entities=("household",))
    return _write_uk_single_year_tables(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
        isa_income_contract=uk_isa_income_contract(frame),
        path=output_path,
    )


@dataclass(frozen=True)
class _UKGateSurface:
    """The duck-attr evidence surface the legacy UK gate modules read.

    Mirrors the national build's gate-evidence adapter: the three entity
    tables plus the metadata gates consult (``time_period``,
    ``household_weight_kind``, ``mass_log``). Lives here — the Frame-carrier
    seam — as the single copy shared by the battery bindings and the legacy
    report evaluator, neither of which may import the other.
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    household_weight_kind: WeightKind
    mass_log: tuple[MassChangeRecord, ...]


def _uk_gate_surface(frame: Frame) -> _UKGateSurface:
    if not isinstance(frame, Frame):
        raise TypeError("UK gate evidence must be a Frame instance.")
    tables = engine_tables(frame)
    return _UKGateSurface(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        household_weight_kind=uk_household_weight_kind(frame),
        mass_log=frame.mass_log,
    )


def validate_uk_national_frame(frame: Frame) -> None:
    """Structural revalidation plus the UK residue, at every seam.

    ``Frame.table`` returns the stored tables, not copies, so a stage that
    mutates them in place can break the invariants construction proved —
    the retired shadow-carrier validator rechecked structure at every seam,
    and this validator must be no weaker: it re-runs every constructor
    invariant via :meth:`Frame.revalidate`, then checks what only the UK
    contract knows — the exact export schema (person/benunit/household,
    household-only typed weights, no links), the time-period metadata,
    the UK invariant that a benunit's members share a household, and
    agreement between the weight total and the latest household
    :class:`MassChangeRecord`.
    """

    if not isinstance(frame, Frame):
        raise TypeError("UK national stages must operate on Frame instances.")
    frame.revalidate()
    if tuple(frame.entities) != UK_NATIONAL_SCHEMA.entities:
        raise ValueError(
            f"UK national frame entities must be {UK_NATIONAL_SCHEMA.entities}; "
            f"got {tuple(frame.entities)}."
        )
    if tuple(frame.weighted_entities) != ("household",):
        raise ValueError(
            "UK national frames carry household typed weights only; got "
            f"explicit weights for {tuple(frame.weighted_entities)}. The "
            "staging artifact exports household_weight alone, and the loader "
            "rejects any other reserved weight column."
        )
    if frame.schema.links:
        raise ValueError(
            "UK national frames declare no links; the staging writer "
            "persists entity tables only and would silently drop them."
        )
    _assert_uk_benunit_nesting(frame.table("person"))
    uk_time_period(frame)
    weights = frame.weights_for("household")
    household = frame.table("household")
    reserved = sorted(UK_EXPORTED_WEIGHT_COLUMNS & set(household.columns))
    if reserved:
        raise ValueError(
            "UK national Frame carrier must not persist exported weight "
            f"column(s): {reserved}; use typed weights instead."
        )
    household_records = [
        record for record in frame.mass_log if record.entity == "household"
    ]
    if household_records and not np.isclose(
        household_records[-1].new_total,
        float(weights.values.sum()),
        rtol=1e-9,
        atol=0.0,
    ):
        raise ValueError(
            "UK national frame household weight total disagrees with the "
            "latest household MassChangeRecord."
        )
