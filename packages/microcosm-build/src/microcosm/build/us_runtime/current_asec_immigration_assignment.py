"""Retained development imputation on the complete original ASEC donor.

One source draw uses original HSUP_WGT/100 design weights, before selection,
source shares or support cloning. The result is an imputed status pair, not an
observed legal status or a qualification of the mixed-date national controls.
"""

from __future__ import annotations

import hashlib
import json
import sys
import weakref
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import NamedTuple

import numpy as np
import pandas as pd

from microcosm.build import source_manifest, source_runtime
from microcosm.frame import Frame, WeightKind

from . import current_asec_immigration_donor as donor_owner
from . import immigration as rules
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.full-asec-immigration-assignment.v1"
OBSERVATION_YEAR_COLUMN = "immigration_observation_year"
_LINEAGE = ("source_year", "source_household_id", "source_person_id")
# These describe the unchanged adopted manifest, not newly researched controls.
# Values and citations are retained verbatim in the stage inside the receipt.
CONTROL_REFERENCE_BASES = (
    ("undocumented_workers", "2023 estimate, published 2025"),
    ("undocumented_population_anchor", "2023 estimate, published 2025"),
    (
        "undocumented_students",
        "March 2021 update; underlying observation year not qualified",
    ),
    ("humanitarian_status_stocks.paroled_one_year.afghanistan", "as of 2022-03-31"),
    ("humanitarian_status_stocks.paroled_one_year.ukraine", "as of 2023-09-30"),
    ("humanitarian_status_stocks.paroled_one_year.nicaragua", "through 2024-12"),
    ("humanitarian_status_stocks.paroled_one_year.venezuela", "through 2024-12"),
    ("humanitarian_status_stocks.refugee", "FY2023-FY2024 admissions proxy"),
    ("humanitarian_status_stocks.asylee", "FY2022-FY2024 grants proxy"),
    ("humanitarian_status_stocks.tps", "as of 2025-03-31"),
    (
        "humanitarian_status_stocks.deportation_withheld",
        "explicit zero modeling assumption; no observed zero stock",
    ),
)
_ISSUED = {}


def _require(condition, reason):
    if not condition:
        raise ValueError("FULL_ASEC_IMMIGRATION_ASSIGNMENT_" + reason)


def _modules():
    return (sys.modules[__name__], rules, source_manifest, source_runtime)


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
            elif name.isupper() and name not in ("_ISSUED", "_LIVE"):
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


def _stage():
    return source._encode(asdict(rules.us_immigration_stage_spec()))


def _file_stats():
    # Reuse the retained owners' declared implementation closures: assignment
    # reads must not invalidate donor/preparation code checked earlier either.
    modules = (*_modules(), *donor_owner._modules(), *source._modules())
    paths = list(dict.fromkeys(Path(module.__file__) for module in modules))
    paths.append(Path(rules.__file__).parent.parent / "us" / "source_stages.json")
    return tuple(source._stat_identity(path.lstat()) for path in paths)


def _with_person(frame, person):
    return Frame(
        {
            entity: person
            if entity == "person"
            else frame.table(entity).copy(deep=True)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.copy(deep=True),
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


class _State(NamedTuple):
    donor: donor_owner.CurrentAsecImmigrationDonor
    donor_entry: tuple
    frame: Frame
    frame_seal: str
    pairs: pd.DataFrame
    pairs_seal: str
    implementation: bytes
    stage: bytes
    file_stats: tuple


def _final(value, entry):
    state = entry[2]
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    donor_owner._final(state.donor, state.donor_entry)
    _require(
        _ISSUED.get(id(value)) is entry
        and entry[0]() is value
        and value.receipt == entry[1]
        and value.frame is state.frame
        and value.pairs is state.pairs
        and source.asec_native._frame_identity(state.frame) == state.frame_seal
        and donor_owner.literals._table_seal(state.pairs) == state.pairs_seal,
        "ASSIGNMENT_PROJECTION_CHANGED",
    )


@dataclass(frozen=True, eq=False)
class CurrentAsecImmigrationAssignment:
    """Checked full-source pair; consumers must validate after final relevant I/O."""

    frame: Frame
    pairs: pd.DataFrame
    receipt: bytes

    def validate(self):
        entry = _ISSUED.get(id(self))
        _require(
            type(self) is CurrentAsecImmigrationAssignment
            and entry is not None
            and entry[0]() is self,
            "ISSUED_OWNER_REQUIRED",
        )
        _final(self, entry)
        state = entry[2]
        state.donor.validate()
        _require(
            _implementation() == state.implementation, "IMPLEMENTATION_BYTES_CHANGED"
        )
        _require(_stage() == state.stage, "CONTROL_STAGE_CHANGED")
        # Assignment I/O follows donor validation. Close that interval using
        # the existing retained source stat fence, then perform no further I/O.
        preparation = state.donor_entry[2].preparation_entry[2]
        _require(_file_stats() == state.file_stats, "IMPLEMENTATION_FILES_CHANGED")
        _require(
            source._file_stats(preparation.root) == preparation.file_stats,
            "SOURCE_FILES_CHANGED",
        )
        _final(self, entry)


def assign_full_asec_immigration(donor, *, seed):
    """Assign once on a live complete original donor, never selected/allocated rows.

    The retained result can be reused for subsequent selected-origin projection;
    neither an assigned Frame nor a detached pair is accepted as source authority.
    No caller-supplied controls, time period or weight multiplier are admitted.
    """
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require(type(donor) is donor_owner.CurrentAsecImmigrationDonor, "DONOR_TYPE")
    _require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    # Neither a partial nor a complete unowned pair can supply donor authority.
    _require(
        not set(rules.US_IMMIGRATION_OUTPUT_COLUMNS) & set(donor.frame.person),
        "PREEXISTING_OUTPUT",
    )
    donor.validate()
    donor_entry = donor_owner._ISSUED[id(donor)]
    file_stats = _file_stats()
    implementation = _implementation()
    stage_spec = rules.us_immigration_stage_spec()
    stage = source._encode(asdict(stage_spec))
    document = json.loads(donor.receipt)
    _require(
        document["income_year"] == 2024
        and document["observation_year"] == 2025
        and document["weight_source"] == "original_HSUP_WGT/100"
        and donor.frame.weights_for("household").kind is WeightKind.DESIGN
        and tuple(rules.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS)
        == donor_owner.literals.ASEC_VALUE_COLUMNS,
        "DONOR_CONTRACT",
    )
    person = donor.frame.person.copy(deep=True)
    _require(
        not {*_LINEAGE, OBSERVATION_YEAR_COLUMN, "person_weight"} & set(person),
        "RULE_INPUT_COLLISION",
    )
    person["source_year"] = document["income_year"]
    person["source_household_id"] = person.PH_SEQ.map(int)
    person["source_person_id"] = person.PERIDNUM
    person[OBSERVATION_YEAR_COLUMN] = document["observation_year"]
    _require(not person.duplicated(list(_LINEAGE)).any(), "SOURCE_KEY_BIJECTION")
    # Use the exact captured stage with the maintained runtime and rule handler.
    # The convenience Frame wrapper reloads the manifest independently, which
    # cannot attest which control document was actually consumed by this draw.
    person["person_weight"] = np.asarray(
        donor.frame.resolve_weights("person").values, dtype=np.float64
    )
    output = source_runtime.run_source_stage(
        stage_spec,
        tables={"person": person},
        operation_handlers={
            "derive_immigration_status": partial(
                rules.derive_us_immigration_status_from_manifest,
                observation_year_column=OBSERVATION_YEAR_COLUMN,
            )
        },
        config=source_runtime.SourceRuntimeConfig(
            seed=seed, target_year=document["observation_year"]
        ),
    )
    _require(source._encode(asdict(stage_spec)) == stage, "CONSUMED_STAGE_CHANGED")
    _require(
        output.person_id.is_unique
        and set(output.person_id) == set(donor.frame.person.person_id),
        "OUTPUT_PERSON_BIJECTION",
    )
    aligned = output.set_index("person_id").reindex(donor.frame.person.person_id)
    assigned_person = donor.frame.person.copy(deep=True)
    for name in rules.US_IMMIGRATION_OUTPUT_COLUMNS:
        assigned_person[name] = aligned[name].to_numpy(copy=True)
    frame = _with_person(donor.frame, assigned_person)
    _require(
        source.asec_native._frame_identity(
            _with_person(
                frame,
                frame.person.drop(columns=list(rules.US_IMMIGRATION_OUTPUT_COLUMNS)),
            )
        )
        == donor_entry[2].frame_seal,
        "NONOWNED_FRAME_CHANGED",
    )
    for name, values in (
        ("ssn_card_type", rules.SSN_CARD_TYPE_VALUES),
        ("immigration_status_str", rules.IMMIGRATION_STATUS_VALUES),
    ):
        _require(frame.person[name].isin(values).all(), "OUTPUT_DOMAIN")
    pairs = donor.raw.loc[:, donor_owner.literals.original.ASEC_KEYS].copy(deep=True)
    pairs.insert(0, "income_year", document["income_year"])
    for name in rules.US_IMMIGRATION_OUTPUT_COLUMNS:
        pairs[name] = frame.person[name].to_numpy(copy=True)
    _require(
        np.array_equal(pairs.index, frame.person.person_id), "PAIR_PERSON_BIJECTION"
    )
    frame_seal = source.asec_native._frame_identity(frame)
    pairs_seal = donor_owner.literals._table_seal(pairs)
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "purpose": "development_imputation",
            "donor_receipt_sha256": source._sha(donor.receipt),
            "donor_frame_sha256": donor_entry[2].frame_seal,
            "source_literals_sha256": donor_entry[2].raw_seal,
            "household_projection_sha256": donor_entry[2].household_seal,
            "implementation": json.loads(implementation),
            "control_stage_sha256": source._sha(stage),
            "control_stage": json.loads(stage),
            "control_reference_bases": dict(CONTROL_REFERENCE_BASES),
            "control_convention": "unchanged mixed-date development imputation assumptions",
            "income_year": document["income_year"],
            "observation_year": document["observation_year"],
            "seed": seed,
            "draw_key": ["income_year", "integer(PH_SEQ)", "literal(PERIDNUM)"],
            "pair_source_keys": list(donor_owner.literals.original.ASEC_KEYS),
            "weight_kind": "design",
            "weight_source": document["weight_source"],
            "person_weight_scale": 1.0,
            "person_weight_authority": "none",
            "readset": rules.US_IMMIGRATION_REQUIRED_SOURCE_COLUMNS,
            "persons": frame.n("person"),
            "households": frame.n("household"),
            "frame_sha256": frame_seal,
            "pairs_sha256": pairs_seal,
            "status_assignment_performed": True,
            "observed_legal_status": False,
            "national_stock_alignment_qualified": False,
            "composition_gate_performed": False,
            "prior_income_columns_consumed": False,
            "source_admission_issued": False,
        }
    )
    assigned = CurrentAsecImmigrationAssignment(frame, pairs, receipt)
    state = _State(
        donor,
        donor_entry,
        frame,
        frame_seal,
        pairs,
        pairs_seal,
        implementation,
        stage,
        file_stats,
    )
    owner_id = id(assigned)
    _ISSUED[owner_id] = (
        weakref.ref(assigned, lambda _: _ISSUED.pop(owner_id, None)),
        receipt,
        state,
    )
    assigned.validate()
    return assigned


_LIVE = _live()
