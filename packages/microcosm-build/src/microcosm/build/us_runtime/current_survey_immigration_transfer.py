"""Retain one paired immigration transfer on authenticated original people.

The complete original ASEC assignment supplies DESIGN-weighted donors. The
same live graph parent's selected-original ACS+ASEC allocation supplies the
IMPORTANCE-weighted receiving population. This development owner issues no
national-stock, graph-fit, clone-attachment, engine or release qualification.
"""

from __future__ import annotations

import hashlib
import sys
import weakref
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from types import FunctionType
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.fit import qrf
from microcosm.frame import Frame, WeightKind

from . import acs_transfer as transfer
from . import acs_transfer_bank as bank
from . import current_acs_immigration_source_projection as acs_owner
from . import current_asec_immigration_assignment as assignment_owner
from . import graph_atomic_survey_financial as host
from . import immigration as rules
from . import survey_atomic_geography as allocation
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.original-survey-immigration-transfer.v1"
OUTPUTS = rules.US_IMMIGRATION_OUTPUT_COLUMNS
YEAR = assignment_owner.OBSERVATION_YEAR_COLUMN
_ISSUED = {}


def _require(condition, reason):
    if not condition:
        raise ValueError("ORIGINAL_SURVEY_IMMIGRATION_TRANSFER_" + reason)


def _require_fit_implementation():
    # QRF is the package-level alias; qrf owns the sealed concrete class.
    _require(transfer._qrf() is qrf.RegimeGatedQRF, "FIT_IMPLEMENTATION_CHANGED")


def _modules():
    return tuple(
        dict.fromkeys(
            (
                sys.modules[__name__],
                transfer,
                bank,
                qrf,
                host,
                allocation,
                allocation.graph,
                allocation.replay,
                source,
                *assignment_owner._modules(),
                acs_owner,
                acs_owner.literals,
            )
        )
    )


def _live():
    result = {}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for key, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, key] = source._function_seal(
                            function
                        )
            elif name.isupper() and name not in (
                "_ISSUED",
                "_RETAINED",
                "_ISSUED_RUNS",
                "_LIVE",
                "QRF",
                "_MEMO",
                "_EPOCHS",
                "_EPOCH_RECORD",
            ):
                # Exclude only the maintained issuer/verification registries and
                # lazy class cache; executable helpers remain in the seal.
                result[module.__name__, name] = source._runtime_marker(value)
    return result


def _implementation():
    return source._encode(
        {
            module.__name__: hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in _modules()
        }
    )


def _file_stats():
    modules = (
        *_modules(),
        *assignment_owner.donor_owner._modules(),
        *source._modules(),
    )
    paths = list(dict.fromkeys(Path(module.__file__) for module in modules))
    paths.append(Path(rules.__file__).parent.parent / "us" / "source_stages.json")
    return tuple(source._stat_identity(path.lstat()) for path in paths)


def _frame_with_person(frame, person):
    return assignment_owner._with_person(frame, person)


def _temporary(frame, person):
    """Keep exact entity coordinates/weights; expose only qualified predictors."""
    tables = {
        entity: frame.table(entity)
        .loc[:, [frame.schema.id_column(entity)]]
        .copy(deep=True)
        for entity in frame.schema.group_entities
    }
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.copy(deep=True),
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


def _lineage_unique(person):
    names = assignment_owner._LINEAGE
    _require(person.loc[:, names].notna().all().all(), "LINEAGE_MISSING")
    keys = person.loc[:, names].astype(str).agg(":".join, axis=1)
    _require(keys.is_unique, "DRAW_KEY_COLLISION")


def _lineage_text(values):
    """Lossless temporary draw-key text; native triples retain original types."""
    values = list(values)
    _require(
        all(
            (type(value) in (str, np.str_, int) or isinstance(value, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and str(value) != ""
            for value in values
        ),
        "LINEAGE_SCALAR",
    )
    return pd.array(
        [str(value) for value in values],
        dtype=assignment_owner.donor_owner.literals.STRING,
    )


def _project_inputs(original, assigned, acs, origins):
    """Pure coordinate projection; callers retain and validate all source owners."""
    schema = original.schema
    axis = [
        schema.person_id_column,
        *(schema.membership_column(e) for e in schema.group_entities),
    ]
    people = original.person
    receiving_households = dict(
        zip(people.person_id, people.person_household_id, strict=True)
    )
    _require(len(receiving_households) == len(people), "ORIGINAL_PERSON_BIJECTION")
    _require(
        np.array_equal(people.person_id.to_numpy(), origins.index.to_numpy())
        and origins.source.isin(("asec", "acs")).all()
        and origins.source_year.eq(2024).all()
        and origins.survey_year.eq(
            origins.source.map({"asec": 2025, "acs": 2024})
        ).all(),
        "ORIGINAL_ORIGIN_AXES",
    )
    asec_mask = origins.source.eq("asec").to_numpy()
    acs_mask = ~asec_mask
    _require(asec_mask.any() and acs_mask.any(), "BOTH_ORIGINAL_SOURCES_REQUIRED")
    _require(set(acs.person.index) == set(origins.index[acs_mask]), "ACS_ROSTER")
    donor_people = assigned.frame.person.set_index("person_id", drop=False)
    _require(
        donor_people.index.is_unique and assigned.pairs.index.is_unique, "DONOR_AXES"
    )
    recipient = people.loc[:, axis].copy(deep=True)
    # Assemble homogeneous columns, not heterogeneous records that coerce keys
    # through floats. Native IDs remain Python/string values throughout joins.
    columns = {
        name: []
        for name in (
            "age",
            "is_female",
            "state_fips",
            "A_AGE",
            "A_LINENO",
            "PRCITSHP",
            "PENATVTY",
            "PEINUSYR",
            "CIT",
            "POBP",
            "YOEP",
            *assignment_owner._LINEAGE,
            YEAR,
            *OUTPUTS,
        )
    }
    donor_states = assigned.frame.broadcast("state_fips").to_numpy()
    donor_state_by_id = dict(
        zip(assigned.frame.person.person_id, donor_states, strict=True)
    )
    for pid, origin in origins.iterrows():
        if origin.source == "asec":
            native = origin.native_person_id
            _require(
                native in donor_people.index and native in assigned.pairs.index,
                "ASEC_NATIVE_JOIN",
            )
            observed, pair = donor_people.loc[native], assigned.pairs.loc[native]
            expected_key = (
                int(origin.raw_native_household_id),
                origin.raw_native_person_id,
                int(origin.native_line_numeric_original),
            )
            _require(
                assignment_owner.donor_owner.literals.original._key(observed, "asec")
                == expected_key
                and assignment_owner.donor_owner.literals.original._key(pair, "asec")
                == expected_key
                and pair.income_year == 2024,
                "ASEC_LITERAL_KEY",
            )
            values = dict(
                age=observed.age,
                is_female=observed.is_female,
                state_fips=donor_state_by_id[native],
                A_AGE=observed.A_AGE,
                A_LINENO=int(observed.A_LINENO),
                PRCITSHP=observed.PRCITSHP,
                PENATVTY=observed.PENATVTY,
                PEINUSYR=observed.PEINUSYR,
                CIT=np.nan,
                POBP=np.nan,
                YOEP=np.nan,
                source_year=2024,
                source_household_id=expected_key[0],
                source_person_id=expected_key[1],
                **{YEAR: 2025},
                **{name: pair[name] for name in OUTPUTS},
            )
        else:
            observed = acs.person.loc[pid]
            key = assignment_owner.donor_owner.literals.original._key(
                acs.raw.loc[pid], "acs"
            )
            _require(
                observed.native_person_id == origin.native_person_id
                and observed.person_household_id == receiving_households[pid]
                and key
                == (
                    origin.raw_native_household_id,
                    origin.raw_native_person_id,
                    int(origin.native_line_numeric_original),
                )
                and observed.source_year == observed.observation_year == 2024,
                "ACS_LITERAL_KEY",
            )
            values = dict(
                age=observed.AGEP,
                is_female=observed.is_female,
                state_fips=observed.state_fips,
                A_AGE=np.nan,
                A_LINENO=np.nan,
                PRCITSHP=np.nan,
                PENATVTY=np.nan,
                PEINUSYR=np.nan,
                CIT=observed.CIT,
                POBP=observed.POBP,
                YOEP=np.nan if pd.isna(observed.YOEP) else observed.YOEP,
                source_year=2024,
                source_household_id=key[0],
                source_person_id=key[1],
                **{YEAR: 2024},
                **{name: pd.NA for name in OUTPUTS},
            )
        for name in columns:
            columns[name].append(values[name])
    for name, values in columns.items():
        if name in OUTPUTS:
            recipient[name] = pd.array(
                values, dtype=assignment_owner.donor_owner.literals.STRING
            )
        elif name in ("source_household_id", "source_person_id"):
            # ASEC integer and ACS literal coordinates share one temporary
            # string dtype. The maintained rule stringifies these same exact
            # values for draw keys; no native coordinate passes through float.
            recipient[name] = _lineage_text(values)
        else:
            recipient[name] = values
    _lineage_unique(recipient)
    for name in OUTPUTS:
        _require(
            np.array_equal(recipient[name].isna().to_numpy(), acs_mask), "TARGET_MASK"
        )
    donor = assigned.frame.person.copy(deep=True)
    donor["source_year"] = 2024
    donor["source_household_id"] = _lineage_text(donor.PH_SEQ.map(int))
    donor["source_person_id"] = _lineage_text(donor.PERIDNUM)
    donor[YEAR] = 2025
    _lineage_unique(donor)
    return (
        _frame_with_person(assigned.frame, donor),
        _temporary(original, recipient),
        acs_mask,
    )


def _ancestry(run, run_entry, assigned, assigned_entry, acs, acs_entry):
    """Require object ancestry, not merely equal hashes from foreign issuers."""
    host._pure_run(run, run_entry)
    _require(
        run_entry[2].completion_boundary is not None
        and run_entry[2].completion_boundary.household_roles is True
        and run_entry[2].person_status_boundary is not None,
        "COMPLETED_PARENT_REQUIRED",
    )
    preparation = run.prefix.preparation
    preparation_entry = run_entry[2].preparation_entry
    astate, cstate = assigned_entry[2], acs_entry[2]
    dstate = astate.donor_entry[2]
    _require(
        preparation is dstate.preparation is cstate.preparation
        and preparation_entry is dstate.preparation_entry is cstate.preparation_entry
        and preparation_entry[2].native[0] is cstate.native
        and preparation_entry[2].native[1] is dstate.native,
        "COMMON_PREPARATION_REQUIRED",
    )
    assignment_owner._final(assigned, assigned_entry)
    acs_owner._final(acs, acs_entry)
    _require(
        preparation_entry[2].plan.fraction == Fraction(1),
        "COMPLETE_SOURCE_SELECTION_REQUIRED",
    )
    _require(
        assigned.frame.weights_for("household").kind is WeightKind.DESIGN
        and run.prefix.allocated_population.frame.weights_for("household").kind
        is WeightKind.IMPORTANCE,
        "WEIGHT_AUTHORITIES",
    )
    return preparation_entry


class _State(NamedTuple):
    run: object
    run_entry: tuple
    assigned: object
    assigned_entry: tuple
    acs: object
    acs_entry: tuple
    original: object
    original_stamp: str
    frame: Frame
    frame_seal: str
    pairs: pd.DataFrame
    pairs_seal: str
    stage: bytes
    controls: object
    controls_bytes: bytes
    implementation: bytes
    file_stats: tuple


def _pure(value, entry):
    state = entry[2]
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require_fit_implementation()
    _ancestry(
        state.run,
        state.run_entry,
        state.assigned,
        state.assigned_entry,
        state.acs,
        state.acs_entry,
    )
    _require(
        _ISSUED.get(id(value)) is entry
        and entry[0]() is value
        and value.receipt == entry[1]
        and value.frame is state.frame
        and value.pairs is state.pairs
        and state.run.prefix.allocated_population is state.original
        and allocation._population_stamp(state.original) == state.original_stamp
        and source._frame_identity(state.frame) == state.frame_seal
        and assignment_owner.donor_owner.literals._table_seal(state.pairs)
        == state.pairs_seal
        and source._encode(asdict(state.controls)) == state.controls_bytes,
        "RETAINED_CHANGED",
    )


@dataclass(frozen=True, eq=False)
class CurrentSurveyImmigrationTransfer:
    """Borrowed original-person result; a copy or receipt grants no authority."""

    frame: Frame
    pairs: pd.DataFrame
    receipt: bytes

    def validate(self) -> None:
        entry = _ISSUED.get(id(self))
        _require(
            type(self) is CurrentSurveyImmigrationTransfer
            and entry is not None
            and entry[0]() is self,
            "ISSUED_OWNER_REQUIRED",
        )
        _pure(self, entry)
        state = entry[2]
        state.assigned.validate()
        state.acs.validate()
        _require(
            _implementation() == state.implementation, "IMPLEMENTATION_BYTES_CHANGED"
        )
        _require(assignment_owner._stage() == state.stage, "CONTROL_STAGE_CHANGED")
        # Store/source authority is rechecked after downstream control/code I/O.
        # Finish with stat fences and pure seals of all borrowed projections.
        state.run.checked_view()
        _require(_file_stats() == state.file_stats, "IMPLEMENTATION_FILES_CHANGED")
        prepared = state.run_entry[2].preparation_entry[2]
        _require(
            source._file_stats(prepared.root) == prepared.file_stats,
            "SOURCE_FILES_CHANGED",
        )
        _pure(self, entry)


def transfer_current_survey_immigration(
    run: host.AtomicSurveyFinancialRunValues,
    assigned: assignment_owner.CurrentAsecImmigrationAssignment,
    acs: acs_owner.CurrentAcsImmigrationProjection,
    *,
    seed: int,
    n_estimators: int = 100,
    bank_root: str | Path | None = None,
) -> CurrentSurveyImmigrationTransfer:
    """Draw ACS pairs once on original allocated people with immutable ASEC."""
    _require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    _require(type(n_estimators) is int and n_estimators > 0, "TREES")
    run_entry = host._run_entry(run)
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require_fit_implementation()
    run.checked_view()
    _require(
        type(assigned) is assignment_owner.CurrentAsecImmigrationAssignment,
        "ASSIGNMENT_TYPE",
    )
    _require(type(acs) is acs_owner.CurrentAcsImmigrationProjection, "ACS_TYPE")
    assigned.validate()
    acs.validate()
    assigned_entry = assignment_owner._ISSUED[id(assigned)]
    acs_entry = acs_owner._RETAINED[id(acs)]
    preparation_entry = _ancestry(
        run, run_entry, assigned, assigned_entry, acs, acs_entry
    )
    view = run.prefix.preparation.checked_view()
    original = run.prefix.allocated_population
    allocation._raw_allocation(view, original)
    original_stamp = allocation._population_stamp(original)
    _require(not set(OUTPUTS) & set(original.frame.person), "PREEXISTING_OUTPUT")
    implementation, file_stats = _implementation(), _file_stats()
    stage_spec = rules.us_immigration_stage_spec()
    stage = source._encode(asdict(stage_spec))
    _require(stage == assigned_entry[2].stage, "ASSIGNMENT_CONTROL_STAGE")
    controls = rules.us_immigration_controls(stage=stage_spec)
    controls_bytes = source._encode(asdict(controls))
    origins = acs_entry[2].qualified.origins
    donor, recipient, mutable = _project_inputs(original.frame, assigned, acs, origins)
    donor_seal, recipient_seal = (
        source._frame_identity(donor),
        source._frame_identity(recipient),
    )
    contract = transfer.acs_transfer_execution_contract_identity(
        targets=OUTPUTS, derive_schedule_d=False, observation_year_column=YEAR
    )
    identity = {
        "protocol": PROTOCOL,
        "parent_sha256": source._sha(run_entry[1]),
        "preparation_sha256": source._sha(preparation_entry[1]),
        "assignment_sha256": source._sha(assigned.receipt),
        "acs_sha256": source._sha(acs.receipt),
        "original_frame_sha256": source._frame_identity(original.frame),
        "donor_sha256": donor_seal,
        "recipient_sha256": recipient_seal,
        "control_stage_sha256": source._sha(stage),
        "controls_sha256": source._sha(controls_bytes),
        "implementation_sha256": source._sha(implementation),
        "seed": seed,
        "n_estimators": n_estimators,
        "target_families": {"person": {"source_operator_immigration": list(OUTPUTS)}},
        "observation_year_column": YEAR,
        "execution_contract": contract,
    }
    target_bank = (
        None
        if bank_root is None
        else bank.AcsTransferTargetBankStore(bank_root, identity=identity)
    )
    result = transfer.transfer_acs_inputs(
        recipient,
        donor,
        target_families={"person": {"source_operator_immigration": OUTPUTS}},
        donor_channel=None,
        seed=seed,
        n_estimators=n_estimators,
        target_bank=target_bank,
        derive_schedule_d=False,
        execution_contract=contract,
        observation_year_column=YEAR,
        immigration_controls=controls,
    )
    _require(
        source._frame_identity(donor) == donor_seal
        and source._frame_identity(recipient) == recipient_seal
        and source._encode(asdict(stage_spec)) == stage
        and source._encode(asdict(controls)) == controls_bytes,
        "CONSUMED_INPUT_CHANGED",
    )
    _require(
        result.resolved_donor_channel is None
        and not result.deferred_inputs
        and len(result.imputed_inputs) == 2
        and {(r.entity, r.column) for r in result.imputed_inputs}
        == {("person", n) for n in OUTPUTS}
        and all(
            r.reconciliation is not None
            and r.imputed_recipient_rows == int(mutable.sum())
            and r.unmodeled_recipient_rows == 0
            for r in result.imputed_inputs
        ),
        "PAIRED_RECONCILIATION_REQUIRED",
    )
    people = original.frame.person.copy(deep=True)
    _require(
        np.array_equal(result.frame.person.person_id, recipient.person.person_id),
        "OUTPUT_AXES",
    )
    for name, domain in zip(
        OUTPUTS,
        (rules.SSN_CARD_TYPE_VALUES, rules.IMMIGRATION_STATUS_VALUES),
        strict=True,
    ):
        values = result.frame.person[name]
        _require(values.isin(domain).all(), "OUTPUT_DOMAIN")
        _require(
            np.array_equal(
                values.to_numpy()[~mutable], recipient.person[name].to_numpy()[~mutable]
            ),
            "IMMUTABLE_ASEC_PAIR",
        )
        people[name] = values.to_numpy(copy=True)
    frame = _frame_with_person(original.frame, people)
    pairs = frame.person.set_index("person_id").loc[:, OUTPUTS].copy(deep=True)
    frame_seal = source._frame_identity(frame)
    pairs_seal = assignment_owner.donor_owner.literals._table_seal(pairs)
    _require(
        source._frame_identity(
            _frame_with_person(frame, frame.person.drop(columns=list(OUTPUTS)))
        )
        == source._frame_identity(original.frame),
        "NONOWNED_FRAME_CHANGED",
    )
    receipt = source._encode(
        {
            **identity,
            "purpose": "development_imputation",
            "person_rows": len(pairs),
            "asec_immutable_rows": int((~mutable).sum()),
            "acs_imputed_rows": int(mutable.sum()),
            "donor_weight_kind": "design",
            "recipient_weight_kind": "importance",
            "donor_weight_source": "original_HSUP_WGT/100",
            "allocation_weight_source": "original_anchor*source_share/inclusion_probability",
            "frame_sha256": frame_seal,
            "pairs_sha256": pairs_seal,
            "fit_records": [asdict(r) for r in result.fit_records],
            "imputed_inputs": [asdict(r) for r in result.imputed_inputs],
            "bank_identity_sha256": None
            if target_bank is None
            else target_bank.identity_sha256,
            "consumed_controls_sha256": source._sha(controls_bytes),
            "national_stock_alignment_qualified": False,
            "observed_legal_status": False,
            "graph_fit_artifact_qualified": False,
            "clone_attachment_performed": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    value = CurrentSurveyImmigrationTransfer(frame, pairs, receipt)
    state = _State(
        run,
        run_entry,
        assigned,
        assigned_entry,
        acs,
        acs_entry,
        original,
        original_stamp,
        frame,
        frame_seal,
        pairs,
        pairs_seal,
        stage,
        controls,
        controls_bytes,
        implementation,
        file_stats,
    )
    key = id(value)
    _ISSUED[key] = (
        weakref.ref(value, lambda _: _ISSUED.pop(key, None)),
        receipt,
        state,
    )
    value.validate()
    return value


_LIVE = _live()
