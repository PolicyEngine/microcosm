"""Admit a checked financial graph result under its original sampling budget.

This narrow IMPORTANCE-to-IMPORTANCE binding changes only the seven current
financial leaves authenticated by the actual twenty-node runner. It neither
changes the original budget's initial population nor admits arbitrary Frames.

Age composition still needs an explicit dependency on financial ATTACH and a
checked binding to the actual receiving graph's Population before calibration
admission. This module does not supply those graph edges or a new age runner.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass

import numpy as np

from microcosm.graph import ArtifactType

PROTOCOL = "microcosm.us.sampling-origin-financial-successor.v1"
FINANCIAL_SUCCESSOR_TYPE = ArtifactType(
    "microcosm.us.sampling_origin_financial_successor", 1
)
MAX_PAYLOAD_BYTES = 256 * 1024
_ISSUED = {}


class SurveyFinancialSuccessorError(ValueError):
    """A financial result lacks its exact original budget/run ancestry."""


def _require(condition, reason):
    if not condition:
        raise SurveyFinancialSuccessorError("SURVEY_FINANCIAL_SUCCESSOR_" + reason)


def _entry(value):
    entry = _ISSUED.get(id(value))
    _require(
        type(value) is SamplingOriginFinancialSuccessor
        and entry is not None
        and entry[0]() is value
        and type(value.payload) is bytes
        and value.payload == entry[1],
        "UNISSUED_OR_CHANGED",
    )
    return entry


@dataclass(frozen=True)
class _State:
    budget: object
    budget_entry: tuple
    financial_run: object
    run_entry: tuple
    previous: object
    current: object
    previous_identity: tuple
    current_identity: tuple


def _document(state, budget_view, run_view):
    # Local imports avoid a budget -> financial owner -> runner -> host cycle.
    from . import graph_atomic_survey_financial as runner
    from . import survey_origin_budget as budget

    run, prefix = state.financial_run, state.financial_run.prefix
    original = state.budget_entry[2]
    _require(
        budget._entry(state.budget, budget.SamplingOriginBudget) is state.budget_entry
        and runner._run_entry(run) is state.run_entry
        and original.preparation is prefix.preparation
        and original.allocated is prefix.allocated_population
        and original.expanded is prefix.clone_population
        and original.geography_config is prefix.geography_config
        and budget_view.preparation is prefix.preparation
        and budget_view.allocated_population is prefix.allocated_population
        and budget_view.initial_population is state.previous is prefix.clone_population
        and budget_view.payload == state.budget_entry[1]
        and budget_view.digest == budget._sha(state.budget_entry[1])
        and run_view.population is state.current is run.financial_population
        and run_view.payload == state.run_entry[1]
        and run_view.digest == budget._sha(state.run_entry[1]),
        "ORIGINAL_BUDGET_RUN_ANCESTRY",
    )
    previous, current = state.previous, state.current
    _require(
        previous.version == current.version == runner.atomic.clone.COMBINED_CLONE_NODE
        and current.mass_ledger == previous.mass_ledger
        and current.weight_kind == previous.weight_kind
        and np.array_equal(
            current.frame.weights_for("household").values,
            previous.frame.weights_for("household").values,
        ),
        "UNCHANGED_SAMPLING_MASS",
    )
    budget.graph._check_design_anchors(current, previous.design_weights["household"])
    owners = {
        **previous.owners,
        **{
            ("person", column): runner.financial.ATTACH_NODE
            for column in runner.values.OUTPUTS
        },
    }
    budget.graph._check_population_state(
        current,
        version=previous.version,
        owners=owners,
        kind=budget.WeightKind.IMPORTANCE,
        ledger=previous.mass_ledger,
    )
    # The checked run freshly compares the complete result to source-derived
    # current values and its pinned raw draws; no caller column table is used.
    return budget._json(
        {
            "protocol": PROTOCOL,
            "budget_sha256": budget._sha(state.budget_entry[1]),
            "financial_run_sha256": budget._sha(state.run_entry[1]),
            "financial_run": json.loads(state.run_entry[1]),
            "previous_version": previous.version,
            "current_version": current.version,
            "previous_frame_sha256": budget.source._frame_identity(previous.frame),
            "current_frame_sha256": budget.source._frame_identity(current.frame),
            "owned_columns": list(runner.values.OUTPUTS),
            "sampling_bounds_changed": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )


def _pure_state(state):
    """Final retained-owner checks; never opens an artifact or source file."""
    from . import graph_atomic_survey_financial as runner
    from . import survey_origin_budget as budget

    _require(
        budget._entry(state.budget, budget.SamplingOriginBudget) is state.budget_entry,
        "FINAL_BUDGET_ISSUANCE",
    )
    runner._pure_run(state.financial_run, state.run_entry)
    _require(
        budget._population_identity(state.previous) == state.previous_identity
        and budget._population_identity(state.current) == state.current_identity,
        "FINAL_POPULATION_SEAL",
    )
    _require(budget._live() == budget._LIVE, "FINAL_PRODUCER_SEAL")


def _final_state(state):
    from . import survey_origin_budget as budget

    # Complete the last budget/support I/O before the financial owner's pure
    # seal, so that borrow cannot invalidate the already-checked financial run.
    budget._final_budget_state(state.budget_entry[2])
    _pure_state(state)


@dataclass(frozen=True)
class SamplingOriginFinancialSuccessor:
    payload: bytes

    def __post_init__(self):
        raise SurveyFinancialSuccessorError("NO_PUBLIC_FINANCIAL_SUCCESSOR_CONSTRUCTOR")

    def checked_view(self):
        from . import graph_atomic_survey_financial as runner

        entry = _entry(self)
        state = entry[2]
        _pure_state(state)
        budget_view = state.budget.checked_view()
        run_view = runner.check_atomic_survey_financial_run(state.financial_run)
        _require(_document(state, budget_view, run_view) == entry[1], "RECONSTRUCTION")
        _final_state(state)
        _require(_entry(self) is entry, "FINAL_ISSUANCE")
        # Do not expose a detached view to the final support-borrow callback.
        # All owner/file I/O and retained-state seals are complete above.
        return CheckedSamplingOriginFinancialSuccessor(
            entry[1],
            runner.codec.sha(entry[1]),
            state.budget,
            state.financial_run,
            state.previous,
            state.current,
        )

    def to_bytes(self):
        return self.checked_view().payload


@dataclass(frozen=True)
class CheckedSamplingOriginFinancialSuccessor:
    """Values borrowed from a checked handle, never decoded authority."""

    payload: bytes
    digest: str
    budget: object
    financial_run: object
    previous: object
    current: object


def admit_survey_financial_population(budget, *, financial_run, candidate=None):
    """Admit only a live completed graph run under its own original budget."""
    from . import graph_atomic_survey_financial as runner
    from . import survey_origin_budget as budgets

    _require(
        candidate is None
        or (type(candidate) is bytes and len(candidate) <= MAX_PAYLOAD_BYTES),
        "CANDIDATE_TYPE_OR_BOUND",
    )
    # An unissued descriptive dataclass must fail before any owner/file borrow.
    run_entry = runner._run_entry(financial_run)
    budget_entry = budgets._entry(budget, budgets.SamplingOriginBudget)
    budget_view = budget.checked_view()
    run_view = runner.check_atomic_survey_financial_run(financial_run)
    state = _State(
        budget,
        budget_entry,
        financial_run,
        run_entry,
        budget_entry[2].expanded,
        financial_run.financial_population,
        budgets._population_identity(budget_entry[2].expanded),
        budgets._population_identity(financial_run.financial_population),
    )
    payload = _document(state, budget_view, run_view)
    _require(len(payload) <= MAX_PAYLOAD_BYTES, "PAYLOAD_BOUND")
    _require(candidate is None or candidate == payload, "CANDIDATE_RECONSTRUCTION")
    result = object.__new__(SamplingOriginFinancialSuccessor)
    object.__setattr__(result, "payload", payload)
    identifier = id(result)

    def forget(reference):
        entry = _ISSUED.get(identifier)
        if entry is not None and entry[0] is reference:
            _ISSUED.pop(identifier, None)

    reference = weakref.ref(result, forget)
    _ISSUED[identifier] = (reference, payload, state)
    entry = _entry(result)
    _final_state(state)
    _require(_entry(result) is entry, "FINAL_ISSUANCE")
    return result


def verify_survey_financial_successor(binding):
    _entry(binding)
    binding.checked_view()
    return binding
