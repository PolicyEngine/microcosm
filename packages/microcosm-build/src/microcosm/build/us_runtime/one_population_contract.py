"""The d2283 one-population contract: provenance, refusals, role record, absences.

d2283's national child population joins every CPS row of the parent's Route A
export (native ASEC rows at clone index 0 and their PUF tax-detail clones at
clone indices 1 and 2) with the ACS households PR 1's selection
(:mod:`~microcosm.build.us_runtime.one_population_selection`) takes from a
staging run built on that export as donor. The prepare step applies this
library between the selection and the H5 write:

- :func:`to_assembled_provenance` rewrites both origins' support provenance
  into the assembled form (``<entity>_spine_source_id`` on every row) and
  builds the spine assembly record from the clone-0 rows;
- :func:`require_one_population` refuses a joined frame that breaks any
  clause of the contract, with a message that names the clause's rule;
- :func:`combined_spm_role_record` writes the SPM independence role record
  for the joined frame, with per-origin counts;
- :func:`absences` lists the columns one origin carries as a reviewed
  default fill, a zero written in place of a measure, or not at all, each
  with its kind;
- :func:`cps_unit_ids` derives each CPS household's unit key, so the
  selection manifest's digest and the contract's key check share one rule.

The engine's variable classes at 2024 (known, computed, input) come from an
injected :class:`EngineVariableIndex`. The default,
:class:`PolicyEngineUSVariableIndex`, reads the installed policyengine-us
source metadata the first time it is asked, never at import, so importing
this module loads no engine.

The module certifies no numbers: no build calls it yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from microcosm.build.us_runtime.one_population_selection import SelectionManifest
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ABSENCES_FILENAME",
    "ACS_SUPPORT_CHANNEL",
    "ASSEMBLED_CHANNELS",
    "CPS_SPINE_TAG",
    "ENGINE_PERIOD",
    "STRUCTURAL_COLUMNS",
    "AssembledTables",
    "EngineVariableIndex",
    "OnePopulationContractError",
    "PolicyEngineUSVariableIndex",
    "absences",
    "combined_spm_role_record",
    "cps_unit_ids",
    "require_one_population",
    "to_assembled_provenance",
]

#: The national period the engine's variable classes are read at: the
#: release tool's ``PERIOD``, which its export guard passes to the same
#: classifier (F-14, F-15).
ENGINE_PERIOD = 2024

#: The support channel (and ``<entity>_spine`` tag) staging gives ACS rows,
#: ``acs_pums.ACS_2024_1YR_SPINE``. A test pins the two together.
ACS_SUPPORT_CHANNEL = "acs_2024_1yr"

#: The ``<entity>_spine`` tag staging gives donor rows,
#: ``base_pool.ASEC_PUF_SPINE``. The contract stamps it on every CPS row
#: (d2283 Q3), so no row of the joined frame has a null tag. A test pins the
#: two together.
CPS_SPINE_TAG = "asec_puf"

#: The channels an assembled one-population frame declares, in record order.
#: ``puf_tax_detail`` is declared because CPS clones carry it, with a clone-0
#: count of 0 on every entity.
ASSEMBLED_CHANNELS = (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ACS_SUPPORT_CHANNEL,
)

#: The file the prepare step writes :func:`absences` to.
ABSENCES_FILENAME = "us_population_absences.json"

#: The person id and each group's id and membership column. None of them is
#: an engine input or an engine-computed column for this contract, as in
#: the release tool's export guard.
STRUCTURAL_COLUMNS = frozenset(
    {
        US_SCHEMA.person_id_column,
        *(US_SCHEMA.id_column(group) for group in US_SCHEMA.group_entities),
        *(US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities),
    }
)

# Raw source columns the clauses read. CPS rows carry the ASEC person id;
# ACS rows carry the PUMS household serial number (household table) and the
# person's line in it (person table), plus the household type code.
_PERIDNUM = "PERIDNUM"
_SERIALNO = "SERIALNO"
_SPORDER = "SPORDER"
_TYPEHUGQ = "TYPEHUGQ"
_SOURCE_YEAR = "source_year"
_SOURCE_HOUSEHOLD_ID = "source_household_id"
_SOURCE_PERSON_ID = "source_person_id"
#: A CPS unit id is the first 20 characters of PERIDNUM (H_IDNUM), as the
#: support-mix bake-off reads it.
_CPS_UNIT_ID_LENGTH = 20


class OnePopulationContractError(ValueError):
    """A one-population frame, or one origin's input, breaks a contract clause.

    Each message names the clause's rule, so a caller (and a test) can tell
    the refusals apart.
    """


# ------------------------------------------------------------ provenance


def _spine_column(entity: str) -> str:
    """The ``<entity>_spine`` tag column staging writes on every pooled row."""

    return f"{entity}_spine"


def _provenance_columns(entity: str) -> tuple[str, str, str, str, str]:
    """``entity``'s source id, channel, clone index, spine source id and spine tag."""

    return (
        support_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
        spine_source_id_column(entity),
        _spine_column(entity),
    )


@dataclass(frozen=True, eq=False)
class AssembledTables:
    """The joined tables :func:`to_assembled_provenance` returns, with their record.

    ``tables`` holds the six entity tables, CPS rows and ACS rows together,
    each group table sorted by its id as :class:`~microcosm.frame.Frame`
    requires. ``metadata`` is ``{SPINE_ASSEMBLY_MANIFEST_KEY: record}``.

    Weights are the caller's. The prepare step aligns PR 1's starting
    weights to ``tables["household"]["household_id"]`` and builds the frame as
    ``Frame(tables, US_SCHEMA, {"household": weights},
    metadata=dict(assembled.metadata))``.
    """

    tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, object]


def to_assembled_provenance(
    cps_tables: Mapping[str, pd.DataFrame],
    acs_tables: Mapping[str, pd.DataFrame],
) -> AssembledTables:
    """Join CPS and ACS rows in the assembled provenance form (d2283 K-1).

    CPS rows (the parent export's) keep Route A's source id, channel and
    clone index (``asec`` on clone 0, ``puf_tax_detail`` on clones 1 and 2),
    gain ``<entity>_spine_source_id`` equal to the source id, and are tagged
    ``<entity>_spine = asec_puf``. ACS rows (staging's) follow
    :func:`~microcosm.build.us_runtime.spine_assembly.assemble_spines`'s rule:
    the spine source id takes staging's native source id, the source id takes
    the pooled ``<entity>_id``, the channel stays ``acs_2024_1yr`` and the
    clone index 0. No row is labelled ``acs``, so the ACS predictor join
    passes the frame through.

    The record is
    :func:`~microcosm.build.us_runtime.support_provenance.spine_assembly_manifest`
    over every entity's clone-0 rows, declaring :data:`ASSEMBLED_CHANNELS`.

    Raises:
        OnePopulationContractError: If either origin lacks complete support
            metadata, a (source id, rank) pair repeats, frame ids or
            memberships collide across origins, or an ACS row's
            ``SERIALNO``/``SPORDER`` lineage does not survive.
    """

    raise NotImplementedError(
        "to_assembled_provenance is built in a later phase of d2283 PR 2."
    )


# ------------------------------------------------------- engine variables


@runtime_checkable
class EngineVariableIndex(Protocol):
    """What the contract asks of the engine's variable metadata, at 2024.

    The answers must cohere: every input is known, is not computed at 2024
    and is not a structural column. The contract refuses an index whose
    answers do not.
    """

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        """The names among ``columns`` the engine defines, formulas included."""
        ...

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """The tables' columns the engine computes at 2024."""
        ...

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """The tables' known, non-structural columns not computed at 2024."""
        ...


def _present_columns(tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
    return frozenset(
        column
        for table in tables.values()
        for column in table.columns
        if isinstance(column, str)
    )


class PolicyEngineUSVariableIndex:
    """:class:`EngineVariableIndex` over the installed policyengine-us sources.

    It wraps
    :class:`~microcosm.frame.adapters.policyengine_us.PolicyEngineUSVariableMetadataIndex`,
    built the first time an answer is needed (it parses the installed
    wheel's variable sources without importing the engine), never at
    construction or import.

    - "Computed at 2024" is that index's
      ``_engine_computed_columns(tables, period=2024)``, the call the release
      tool's export guard makes, so the contract and the release cannot
      disagree. It refuses the four compatibility aggregates at any period
      and never counts a structural id or membership column.
    - "Known" is ``variables()`` plus ``formula_owned_outputs``: every name
      the sources define, inputs and formulas alike.
    - "Input at 2024" is known, not structural and not computed at 2024. A
      variable whose only formula starts after 2024 (``weeks_worked`` at
      policyengine-us 2.2.1) is therefore an input here, although the
      year-blind ``variables()`` leaves it out.

    Args:
        metadata_index: The index to wrap. ``None`` builds the installed one
            on first use.
    """

    def __init__(self, metadata_index: Any | None = None) -> None:
        self._metadata_index = metadata_index

    def _index(self) -> Any:
        if self._metadata_index is None:
            from microcosm.frame.adapters.policyengine_us import (
                PolicyEngineUSVariableMetadataIndex,
            )

            self._metadata_index = PolicyEngineUSVariableMetadataIndex()
        return self._metadata_index

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        names = {column for column in columns if isinstance(column, str)}
        index = self._index()
        inputs = names.intersection(index.variables())
        formulas = names.intersection(index.formula_owned_outputs(names))
        return frozenset(inputs | formulas)

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        return frozenset(
            self._index()._engine_computed_columns(tables, period=ENGINE_PERIOD)
        )

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        known = self.known_variables(_present_columns(tables))
        return frozenset(known - STRUCTURAL_COLUMNS - self.computed_at_2024(tables))


@dataclass(frozen=True)
class _EngineColumns:
    """One frame's columns by engine class, from one coherent index answer."""

    known: frozenset[str]
    computed: frozenset[str]
    inputs: frozenset[str]


def _engine_columns(
    tables: Mapping[str, pd.DataFrame],
    variable_index: EngineVariableIndex | None,
) -> _EngineColumns:
    """Ask the index (the installed engine's by default) and check the answers."""

    index = (
        variable_index if variable_index is not None else PolicyEngineUSVariableIndex()
    )
    present = _present_columns(tables)
    known = frozenset(index.known_variables(sorted(present)))
    computed = frozenset(index.computed_at_2024(tables))
    inputs = frozenset(index.inputs_at_2024(tables))
    problems = [
        f"{label} names columns no table holds: {sorted(names - present)[:5]}"
        for label, names in (
            ("known", known),
            ("computed at 2024", computed),
            ("input at 2024", inputs),
        )
        if not names <= present
    ]
    if not inputs <= known:
        problems.append(f"inputs not known: {sorted(inputs - known)[:5]}")
    if inputs & computed:
        problems.append(f"inputs also computed: {sorted(inputs & computed)[:5]}")
    if inputs & STRUCTURAL_COLUMNS:
        problems.append(
            f"inputs include structural columns: {sorted(inputs & STRUCTURAL_COLUMNS)}"
        )
    if problems:
        raise OnePopulationContractError(
            "The engine variable index answers do not cohere (an input must be "
            "known, not computed at 2024 and not structural): "
            + "; ".join(problems)
            + "."
        )
    return _EngineColumns(known=known, computed=computed, inputs=inputs)


# ----------------------------------------------------------- the contract


def require_one_population(
    frame: Frame,
    manifest: SelectionManifest,
    *,
    parent_tail_donor_ids: frozenset[int],
    declared_absences: Mapping[str, str] | None = None,
    variable_index: EngineVariableIndex | None = None,
    register: Mapping[str, str] | None = None,
) -> None:
    """Refuse a joined frame that breaks any clause of the contract.

    The clauses, in the order they run: assembled provenance (R1); one native
    row per CPS key and one row per ACS household, against ``manifest`` (R2);
    no column the engine computes at 2024 (R3); no null ESI premium and the
    premium present (R4); a classified adult in every SPM unit (R5); no null
    in an engine input at 2024 outside the always-on producers' listed
    outputs (R6); no null PUF tail column on a tax unit (R7); the tail donor
    set equal to ``parent_tail_donor_ids`` (R8); every stored lower-case
    column an engine variable or registered (R9); a model-named column filled
    on CPS rows and null on ACS rows only when ``declared_absences`` declares
    it (R10); no location-clone index column (R11); a congressional district
    on every household (R12); ``TYPEHUGQ`` on every ACS household (T1).

    Args:
        frame: The joined frame, built from :func:`to_assembled_provenance`.
        manifest: PR 1's selection manifest for the same selection.
        parent_tail_donor_ids: The parent export's applied capital-gains tail
            donor ids.
        declared_absences: Column to the owner's decision, for each
            model-named column ACS rows lack by decision.
        variable_index: The engine's variable classes at 2024. ``None`` uses
            :class:`PolicyEngineUSVariableIndex`.
        register: The stored non-variable register. ``None`` uses
            ``microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS``.

    Raises:
        OnePopulationContractError: Naming the first clause the frame breaks.
    """

    raise NotImplementedError(
        "require_one_population is built in a later phase of d2283 PR 2."
    )


def combined_spm_role_record(
    cps_record: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    frame: Frame,
) -> dict[str, object]:
    """The SPM independence role record for the joined frame (F-17, K-6).

    ``cps_record`` is a gate-keyed record over exactly the frame's CPS rows,
    which the prepare step re-derives from raw ASEC inputs; it is checked
    against a recount of those rows (counts and digest) and must say
    ``role_source == "source_column"``. ``staging_receipt`` is staging's ACS
    role receipt, whose keys are checked. The result recounts the whole
    frame in the keys
    :func:`~microcosm.build.us_runtime.spm_independence_role.us_spm_independence_role_signal_gate`
    reads, keeps per-origin counts, and cites both origins' evidence.

    Raises:
        OnePopulationContractError: If either origin's evidence disagrees
            with the frame.
    """

    raise NotImplementedError(
        "combined_spm_role_record is built in a later phase of d2283 PR 2."
    )


def absences(
    frame: Frame,
    *,
    fill_manifest: Mapping[str, Any],
    zero_written: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, object]:
    """What one origin carries as a default fill, a written zero, or not at all.

    Returns the content of :data:`ABSENCES_FILENAME`: one entry per column
    with its origin, rows and weighted share and a kind. The kinds are
    "owner-declared" (needs the owner's decision, from ``declared``),
    "release-written" (an always-on producer's listed output), "raw source
    column" (upper-case survey columns), "reviewed default fill" (from the
    prepare step's ``fill_manifest``) and "zero written" (``zero_written``
    maps each such column to the receipt that records it). Both directions
    are listed: columns CPS rows fill and ACS rows lack, and the reverse.
    Origins are keyed by support channel.
    """

    raise NotImplementedError("absences is built in a later phase of d2283 PR 2.")


def cps_unit_ids(person: pd.DataFrame) -> pd.DataFrame:
    """Each CPS household's income year, unit id and selection key.

    The unit id is ``PERIDNUM[:20]`` (H_IDNUM), on which every person of the
    household must agree, as the support-mix bake-off reads it. It comes from
    ``PERIDNUM`` when the column is present, else from ``source_person_id``
    only where that is a string of at least 20 digits (the ASEC pool writes
    ``PERIDNUM`` there; its integer fallback is refused). The income year is
    ``source_year``. The key is
    :func:`~microcosm.build.us_runtime.one_population_selection.cps_unit_key`,
    so the prepare step's selection keys and the contract's key check cannot
    drift.

    Returns:
        One row per ``person_household_id``, indexed by it, with columns
        ``income_year``, ``unit_id`` and ``key``.

    Raises:
        OnePopulationContractError: If an id or year is missing or malformed,
            or a household's persons disagree.
    """

    raise NotImplementedError("cps_unit_ids is built in a later phase of d2283 PR 2.")
