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
  with its kind, and :func:`write_absences` writes that list as
  :data:`ABSENCES_FILENAME`;
- :func:`cps_unit_ids` derives each CPS household's unit key, so the
  selection manifest's digest and the contract's key check share one rule.

After the write, :func:`one_population_declares_acs_rows` says whether a
frame declares ACS rows, so a release gate can tell a d2283 child from a
Route A frame. Nothing calls it yet.

The engine's variable classes at 2024 (known, computed, input) come from an
injected :class:`EngineVariableIndex`. The default,
:class:`PolicyEngineUSVariableIndex`, reads the installed policyengine-us
source metadata the first time it is asked, never at import, so importing
this module loads no engine.

Who reads the spine assembly record. At this module's commit
``validate_assembly_provenance`` is called by ``spine_assembly.py`` (:136),
``spine_agreement.py`` (:330), ``acs_release_predictors.py`` (:1407, only
when a row carries channel ``acs``, which this module never writes),
``puf_support.py`` (:797, :859, :878), ``multispine_pool.py`` (:3597,
:3605, :3629, :3660, :3681, all inside ``run_multispine_pool_path``, the
stacked-pool build path, not a loader), ``stacked_spine.py`` (:1442),
``support_provenance.py``'s receipt and count helpers (:265, :284) and
``tools/build_us_multispine_pool.py`` (:3390, :4217). The release tool calls
none of them, and the pool loader it uses
(``h5_io.load_authenticated_us_multispine_pool_for_release``) never
validates. So on main nothing on the release tool's arm reads the record:
it serves :func:`require_one_population`, PR 5's loader and later callers.

What the later d2283 PRs owe this library:

- PR 3 flips six strict xfails of the whole-frame differential. Four
  compare statistics on the reader frame: the weeks-unemployed UC-rule
  scope, Medicare take-up, retirement contributions and retirement
  distributions. Two compare verdicts on a frame with one clone-only
  defect (d2283 lead call L7): a clone row's ``MCARE`` disagreeing with its
  take-up flag fails Route A's Medicare reconciliation and passes the
  assembled one, and a clone row with weeks unemployed and no
  unemployment compensation fails Route A's UC rule and passes the
  assembled one. On the assembled form the weeks UC rows key on the
  stored channel and clone index (``weeks_unemployed.py`` :1366-1373), and
  the other three readers' ``_asec_source_mask`` returns the
  stored-``asec`` rows.
- PR 4 (prepare) fills on ACS rows every engine input outside
  :data:`RELEASE_WRITTEN_INPUTS`, ``takes_up_snap_if_eligible`` included
  (the SNAP take-up gate misgrades an ACS null). Lead call L1 moved ten
  producer outputs off that list: the release's first engine pass refuses
  a null in the nine float and integer ones at load, and later passes
  compute from ``is_self_employed`` before its producer writes it. The
  lead-call review moved an eleventh, ``takes_up_medicaid_if_eligible``,
  which the ACA pass computes from before Medicaid writes it. Their fill
  sources: the
  three integer recodes (``detailed_occupation_recode``,
  ``detailed_industry_recode``, ``major_industry_recode``) come from
  staging, which writes them natively from ``OCCP`` and ``INDP``
  (``acs_inputs.ACS_WORK_EXPERIENCE_NATIVE_MAPPINGS``); any finite value
  serves for ``cps_race``, ``hourly_wage``, ``fsla_overtime_premium``,
  ``selected_marketplace_plan_benchmark_ratio`` and
  ``other_health_insurance_premiums``, and any boolean for
  ``takes_up_medicaid_if_eligible``, since their producers rewrite them and
  no earlier pass's result depends on them (the ACA pass's two reads of the
  Medicaid flag are masked by Medicaid eligibility in policyengine-us
  2.2.1). ``health_insurance_premiums`` and ``is_self_employed`` take the
  producer's own pinned rule, so the ACA, Medicaid and SNAP-state passes,
  which compute from them on ACS rows before their producer runs (staging
  writes neither), see there the values the producer later ships (the
  11-19 call of the US data owner, who owns d2283 Q2). Prepare runs
  ``other_health_insurance.attribute_us_se_health_premiums_from_manifest``
  with the one operation of kind ``attribute_self_employed_health_premiums``
  in the packaged stage (``us_other_health_insurance_stage_spec()``; by
  kind, never by position, and never the function's defaults), on the
  joined person table after the reviewed default fill and before this
  contract. That fill holds both columns out, as it holds out the SSI and
  Medicaid take-up columns, or it would write 0/False and file fills the
  attribution then overwrites. Prepare writes the two outputs on ACS rows
  only, as float64 and bool; refuses unless the rule reproduces every CPS
  row's carried values (premiums within the export gate's 1e-9, the flag
  exactly), a parity check that the binding and inputs reproduce the
  parent's producer; and records the operation, its parameters and the
  ``us/source_stages.json`` sha256 in its manifest. Nothing after it may
  write the rule's inputs or outputs. A null PR 4 leaves in an exempt
  column must be None or a float NaN, never ``pd.NA``, ``pd.NaT``, numpy's
  ``NaT`` or any other null. The first engine pass cannot load a
  ``pd.NA`` in the twelve flags it reads before their producers write
  them; the other six (the four SCF amounts,
  ``meets_ssi_disability_criteria`` and ``takes_up_head_start_if_eligible``)
  have producers that run before that pass and rewrite a non-finite
  value. R6 refuses every other null in all 18 as a fail-closed rule.
- PR 4 writes on ACS rows every raw predictor of
  ``acs_release_predictors._MODEL_PREDICTORS``, all fourteen
  (:data:`RAW_COLUMN_READERS` names each one's readers): ``PEDISDRS``,
  ``PEDISEAR``, ``PEDISEYE``, ``PEDISOUT``, ``PEDISPHY``, ``PEDISREM``,
  ``PRDTRACE``, ``PRDTHSP``, ``SPM_TENMORTSTATUS``, ``PEIOOCC``,
  ``POCCU2``, ``WEIND``, ``WEMIND`` and ``WKSWORK``, by the join's own
  crosswalk from the ACS PUMS sources or from staging where staging writes
  them. The join itself passes the joined frame through, since no row
  carries channel ``acs``. A null stops the build in the SSI disability
  stage (a non-finite ``PEDIS*``) or the ORG stage (incomplete ``WEIND``,
  ``WEMIND`` or ``WKSWORK``; ``org_wages.py`` :590-655), and passes
  silently where a reader fills it (``sipp_vehicles`` reads a null
  ``SPM_TENMORTSTATUS`` as 3, ``scf_wealth`` a null ``PRDTRACE`` or
  ``PRDTHSP`` as 0). PR 4 also writes the six
  :data:`PUF_TAIL_TAX_UNIT_COLUMNS` defaults on ACS tax units (applied
  False, donor id -1), which R7 and R8 require.
- PR 4 passes :func:`to_assembled_provenance` each origin's household
  weights keyed: CPS by ``household_id``, ACS by ``acs_unit_key(SERIALNO)``
  (``pd.Series(starting.acs, index=plan.acs_keys)``). PR 1's ACS starting
  weights come in selection order and staging's household table in id
  order, so a weight given by position is refused (lead call L3's check).
- PR 4 runs the reviewed default fill on the joined frame so its
  manifest's row counts are the frame's; reads the parent's applied tail
  donor ids from the export's tax-unit columns (checked against the base
  manifest's ``records[*].donor_source_id`` when the base run folder is at
  hand); re-derives the CPS role record with
  ``with_us_spm_independence_role`` on the selected CPS rows, in the joined
  frame's row order and dtypes (the digest is dtype-sensitive); builds the
  selection keys with :func:`cps_unit_ids`; builds ``staging_lineage`` from
  staging's output, not from the ACS tables it passes, keyed by each
  person's native (household id, person id) (staging's
  ``household_source_id`` of its household and its ``person_source_id``)
  with its (``SERIALNO``, ``SPORDER``) as the value; keeps staging's
  ``RELSHIPP`` on every ACS person row, from which
  :func:`combined_spm_role_record` re-derives each ACS role (inferred, to
  confirm at spike step 0: staging's pooled tables keep it, as they keep
  ``TYPEHUGQ``; if they drop it, prepare joins it from the ACS PUMS
  person source, which requires it, by (``SERIALNO``, ``SPORDER``), the
  pair the lineage check ties to each person). Nothing ties the retained
  ``RELSHIPP`` itself to staging, so a swap of both ``RELSHIPP`` and the
  role between two persons passes; a keyed staging role projection would
  catch it. PR 4 also supplies
  :func:`absences` with the reviewed default fill's manifest for the
  joined frame and the ``zero_written`` receipts; and declares (or fills)
  each CPS-only register entry the first live run of :func:`absences`
  lists, the US data owner deciding.
- PR 4 reads the id dtypes of staging's real pooled tables, then tightens
  :func:`_integral_ids`, which today passes object-held digit strings and
  booleans through ``pd.to_numeric`` and refuses a boolean dtype and any
  data but integers, real numbers and strings (lead call L5).
- PR 4 writes both frame records into the population manifest: the spine
  assembly record :func:`to_assembled_provenance` puts under
  ``SPINE_ASSEMBLY_MANIFEST_KEY`` and the record
  :func:`combined_spm_role_record` returns, with the CPS role record and
  staging's receipt it keeps under ``evidence``. The shared H5 writer
  stores tables, household weights, the period and its own small metadata
  row, never ``Frame.metadata`` (``h5_io``'s module docstring;
  ``_artifact_metadata``), so a record left only in the frame is lost at
  the write.
- PR 5 (the loader) restores both records from the population manifest
  into ``Frame.metadata`` (``us_spine_assembly_manifest`` and
  ``spm_independence_role``) and checks each on the loaded frame: the
  assembly record through
  :func:`~microcosm.build.us_runtime.support_provenance.validate_assembly_provenance`
  (a loader that restores only the role record leaves every reader of the
  assembly record without one), and the role record recomputed after the
  load rather than trusted by its pre-write digest. It also joins the
  absences file to the release's ``us_register_consistency_gate``. The
  absences record names no frame or H5 digest, so PR 5 binds it to its
  file by the run folder.
- PR 5's continuity anchor compares the CPS row set and the native and
  clone structure both ways against the parent export. Whether a clone
  row's values cohere with its native row's is out of this library's
  scope (lead call L6).

Two upstream conditions: R4 (the ESI employer premium on every person)
refuses every real child until d2283 PR 0 lands, and PR 4's CPS role
record re-derivation (d2283 Q6 (a)) needs spike step 0 to find the raw
ASEC role inputs on the export; if they are absent, Q6 reopens.

The module certifies no numbers: no build calls it yet.
"""

from __future__ import annotations

import decimal
import json
import math
import numbers
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.esi_premiums import (
    US_ESI_EMPLOYER_PREMIUM_COLUMN,
    refuse_unassigned_us_esi_premiums,
)
from microcosm.build.us_runtime.one_population_selection import (
    SelectionManifest,
    acs_unit_key,
    cps_unit_key,
    keys_sha256,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
)
from microcosm.build.us_runtime.spm_composition import (
    check_spm_composition,
    spm_independence_role,
)
from microcosm.build.us_runtime.spm_independence_role import _role_binding_sha256
from microcosm.build.us_runtime.spm_role_source import (
    NATIVE_SPM_ROLE,
    SPM_ADULT_RULE,
    SPM_ROLE_RULE,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_CAPITAL_GAINS_TAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    require_assembled_support_provenance,
    spine_assembly_manifest,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_copy_rank_series,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = [
    "ABSENCES_FILENAME",
    "ABSENCE_KINDS",
    "ACS_NATIVE_AMOUNTS",
    "ACS_SUPPORT_CHANNEL",
    "ASSEMBLED_CHANNELS",
    "CPS_SPINE_TAG",
    "ENGINE_PERIOD",
    "PUF_TAIL_TAX_UNIT_COLUMNS",
    "RAW_COLUMN_READERS",
    "RELEASE_WRITTEN_INPUTS",
    "STRUCTURAL_COLUMNS",
    "EngineVariableIndex",
    "OnePopulationContractError",
    "PolicyEngineUSVariableIndex",
    "absences",
    "combined_spm_role_record",
    "cps_unit_ids",
    "one_population_declares_acs_rows",
    "require_one_population",
    "to_assembled_provenance",
    "write_absences",
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
#: The ``source_year`` staging's ACS rows carry: ``acs_pums`` writes the
#: source vintage, 2024 for the 2024 1-year file the channel names.
_ACS_SOURCE_YEAR = 2024

#: Route A's labels on the parent export's rows: the native ASEC row at
#: clone index 0, the PUF tax-detail copy at 1 and the capital-gains tail
#: copy at 2, both copies labelled ``puf_tax_detail``.
_CPS_CHANNELS = (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
_CPS_CLONE_INDICES = (
    0,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_CAPITAL_GAINS_TAIL_CLONE_INDEX,
)
_CPS = "CPS"
_ACS = "ACS"
#: What each origin's household weights are keyed by for
#: :func:`to_assembled_provenance`: the CPS household's frame id (PR 1's CPS
#: starting weights follow the export's household rows) and the ACS
#: household's selection key (PR 1's ``StartingWeights.acs`` follows
#: ``plan.acs_keys``).
_WEIGHT_KEYS = {
    _CPS: US_SCHEMA.id_column("household"),
    _ACS: f"acs_unit_key({_SERIALNO})",
}
#: The boundary name the assembly validator reports.
_BOUNDARY = "one-population provenance"
#: The consumer name the ESI premium refusal reports.
_CONSUMER = "d2283 one-population contract"

#: The six tax-unit provenance columns the capital-gains tail transfer
#: writes on every tax unit, with non-null defaults where it applies no donor
#: (``puf_capital_gains_tail.py`` :1697-1711). Not
#: ``PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS``, which is the section 1250
#: gain leg.
PUF_TAIL_TAX_UNIT_COLUMNS = (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
)

#: The column suffix location cloning adds to every entity table
#: (``block_location.LOCATION_CLONE_INDEX_COLUMN``; a test pins the two).
_LOCATION_CLONE_INDEX_SUFFIX = "_location_clone_index"

#: The household type codes the ACS loader keeps (1 housing unit, 2
#: institutional and 3 non-institutional group quarters).
_ACS_HOUSEHOLD_TYPES = (1, 2, 3)

#: The release's first engine pass, on every arm: the SSI December pass
#: (``_ssi_person_uncapped_amount``). Like every pass built from the frame,
#: it loads each frame column that is an engine variable, whether or not it
#: computes anything from it.
#:
#: Release-tool citations here and below name the function or statement
#: and give its line in ``tools/build_us_fiscal_refresh_release.py`` at
#: this module's commit (blob ``f672b8465``, unchanged by the merge of main
#: ``cfd5a2a5e``); the name holds when the tool's lines move. The stage
#: order the verdicts read is pinned by name in the tests.
_SSI_DECEMBER_PASS = (
    "the _ssi_person_uncapped_amount call, "
    "tools/build_us_fiscal_refresh_release.py:17060"
)
_NO_EARLIER_READER = (
    "Earlier readers: none; no engine pass runs before this stage (the first, "
    f"the SSI December pass, is {_SSI_DECEMBER_PASS}) and no pool-arm stage "
    "before it names the column"
)
#: How policyengine-core 3.32.5 (the lock's pin) loads a null boolean input:
#: it casts with ``astype(bool)``, skipping the null refusal it applies to
#: float and integer inputs (``holder.py`` :257, :276-291).
_BOOLEAN_NULL_LOAD = (
    "policyengine-core 3.32.5 loads a boolean NaN as True and a None as False "
    "and refuses a pd.NA"
)
_LOADED_NOT_COMPUTED = (
    f"Earlier readers: the SSI December pass ({_SSI_DECEMBER_PASS}) loads it but "
    "computes nothing from it (it is outside uncapped_ssi's dependency "
    f"closure); {_BOOLEAN_NULL_LOAD} (static reads of policyengine-us 2.2.1 and "
    "policyengine-core 3.32.5, d2283 lead call L1's trace)"
)
_SCF_ASSETS_EVIDENCE = (
    "scf_wealth stage: with_us_scf_wealth_inputs re-imputes all three asset "
    "columns on every person when any value is not finite "
    "(U/scf_wealth.py:1031; pass-through only when assets and net worth both "
    f"carry signal, :1129; rewrite, :1149). {_NO_EARLIER_READER}"
)
_ORG_EVIDENCE = (
    "org_wages stage: with_us_org_wages_inputs re-imputes the family on every "
    "entry and passes through only when each stored output equals the "
    "recomputed one, which a null never does (U/org_wages.py:990, :998, "
    ":1046); otherwise it writes every output on every person (:1050). "
    f"{_LOADED_NOT_COMPUTED}"
)

#: The engine inputs the release tool's always-on producers rewrite on every
#: row, or refuse on a null, before any release stage reads them on ACS rows,
#: so prepare may leave them null there (the null-input clause, R6, exempts
#: them; d2283 Q1 (a), narrowed by lead call L1 of 11-18). Each entry is the
#: column and its evidence, re-read at this module's commit: the stage
#: (whose ``us/source_stages.json`` ``outputs`` name the column; a test pins
#: that), the lines that rewrite every row or raise on a null, and the
#: release's earlier readers (an engine pass built from the frame before the
#: producer), from L1's static trace of the release tool's pool arm. A test
#: pins the stage order and the engine pins those verdicts read.
#:
#: Left out, so prepare must fill them on ACS rows: producers that pass a
#: frame through when the column carries signal after dropping nulls
#: (``is_snap_abawd_discretionary_exempt``, the three SCF auto-loan columns,
#: the two SIPP vehicle columns, the two SIPP tip columns), a stage's source
#: column (``health_insurance_premiums_without_medicare_part_b``), declared
#: outputs nothing on Route A writes (the ``scf_*`` and education-debt
#: columns), and ``takes_up_snap_if_eligible``. SNAP state take-up does
#: rewrite that one on every SPM unit and raise on a null
#: (U/snap_state_take_up.py:282-290), but the SNAP take-up stage before it
#: passes ACS nulls through (U/snap_take_up.py:231-239, :274-277), and the
#: release's SNAP take-up gate (``us_snap_take_up_signal_gate(base_frame)``,
#: tools/build_us_fiscal_refresh_release.py:16463) grades that frame through
#: ``to_numpy(dtype=bool)``
#: (U/snap_take_up.py:319), which reads a NaN as take-up, a None as no
#: take-up and refuses a ``pd.NA``: it misgrades ACS nulls.
#:
#: Also left out (lead call L1): ten producer outputs an engine pass loads
#: before their producer writes them. policyengine-core 3.32.5 refuses a NaN
#: or None in a float or integer input at load (``holder.py`` :257,
#: :276-291, a static read), so a null in ``cps_race``, ``hourly_wage``,
#: ``fsla_overtime_premium``, ``detailed_occupation_recode``,
#: ``detailed_industry_recode``, ``major_industry_recode`` (org_wages,
#: ``with_us_org_wages_inputs``, tools/build_us_fiscal_refresh_release.py:17328),
#: ``selected_marketplace_plan_benchmark_ratio`` (aca_marketplace_inputs,
#: ``_with_aca_marketplace_source_outputs``, :17360),
#: ``other_health_insurance_premiums`` or ``health_insurance_premiums``
#: (other_health_insurance_premiums, ``with_us_other_health_insurance_inputs``,
#: :17467) stops the build at the SSI December pass
#: (``_ssi_person_uncapped_amount``, :17060). The ACA, Medicaid and
#: SNAP-state passes (``_with_aca_marketplace_source_outputs``, :17360;
#: ``_with_medicaid_take_up_outputs``, :17395;
#: ``_with_snap_state_take_up_outputs``, :17430) also compute from
#: ``health_insurance_premiums`` and ``is_self_employed`` (the self-employed
#: health insurance deduction into AGI and MAGI, and medical expenses), and
#: a boolean null there reads silently as True or False. Staging writes the
#: three integer recodes natively on ACS rows (U/acs_inputs.py:105-120,
#: ``ACS_WORK_EXPERIENCE_NATIVE_MAPPINGS``) and none of the other seven.
#:
#: And ``takes_up_medicaid_if_eligible`` (medicaid_take_up,
#: ``_with_medicaid_take_up_outputs``, :17395; moved by the lead-call
#: review, fail closed): the ACA stage's pass
#: (``_with_aca_marketplace_source_outputs``, :17360) computes
#: ``medicaid_enrolled`` from it before Medicaid writes it, by two
#: routes in policyengine-us 2.2.1. One is the ineligible-coverage sum's
#: qualifying non-marketplace coverage; the other is CHIP's disqualifying
#: coverage, into ``is_chip_eligible_child`` and so ``is_chip_eligible``,
#: which the ineligible-coverage sum and Basic Health Program eligibility
#: read. Both are masked today: ``medicaid_enrolled`` is defined for
#: ``is_medicaid_eligible``, which sits in the same sum, and
#: ``is_chip_eligible_child`` and Basic Health Program eligibility each
#: require not ``is_medicaid_eligible``. L1 let only the two
#: self-employed-premium inputs argue a harmless read, and nothing here
#: checks the masking, so the column is filled like the others.
RELEASE_WRITTEN_INPUTS: Mapping[str, str] = MappingProxyType(
    {
        "bank_account_assets": _SCF_ASSETS_EVIDENCE,
        "bond_assets": _SCF_ASSETS_EVIDENCE,
        "stock_assets": _SCF_ASSETS_EVIDENCE,
        "net_worth": (
            "scf_wealth stage: with_us_scf_wealth_inputs re-imputes net_worth "
            "on every household when any value is not finite "
            "(U/scf_wealth.py:1048, :1129; rewrite, :1157). "
            f"{_NO_EARLIER_READER} (sipp_vehicles reads it later, "
            "with_us_sipp_vehicle_inputs, "
            "tools/build_us_fiscal_refresh_release.py:17198)"
        ),
        "meets_ssi_disability_criteria": (
            "ssi_disability_criteria stage: with_us_ssi_disability_criteria "
            "passes through only when every value is finite, 0/1 and equal to "
            "its prediction, else writes the prediction on every person "
            "(U/ssi_disability_criteria.py:1160, :1167). "
            f"{_NO_EARLIER_READER} (the SSI December pass computes from it, "
            "after this stage)"
        ),
        "takes_up_head_start_if_eligible": (
            "sipp_head_start stage: with_us_sipp_head_start_input passes "
            "through only when every value is finite, 0/1 and equal to its "
            "prediction, else writes the prediction on every person "
            f"(U/sipp_head_start.py:740, :747). {_NO_EARLIER_READER}"
        ),
        "takes_up_ssi_if_eligible": (
            "ssi_take_up stage: with_us_ssi_take_up passes through only a "
            "boolean column with no null equal to its assignment, else writes "
            "the assignment on every person (U/ssi_take_up.py:1260-1261, "
            f":1268). {_LOADED_NOT_COMPUTED}; the pass's docstring says the "
            "same (_ssi_person_uncapped_amount, "
            "tools/build_us_fiscal_refresh_release.py:4564-4565)"
        ),
        "would_file_taxes_voluntarily": (
            "voluntary_filing_input stage: with_us_voluntary_filing_input "
            "passes through only when every value is finite, 0/1 and equal to "
            "its prediction, else writes the prediction on every tax unit "
            f"(U/voluntary_filing.py:934, :941). {_LOADED_NOT_COMPUTED}"
        ),
        **dict.fromkeys(
            (
                "is_hispanic",
                "worked_last_year",
                "has_never_worked",
                "is_military",
                "is_computer_scientist",
                "is_executive_administrative_professional",
                "is_farmer_fisher",
                "is_paid_hourly",
                "is_union_member_or_covered",
            ),
            _ORG_EVIDENCE,
        ),
        "takes_up_aca_if_eligible": (
            "aca_marketplace_inputs stage: _with_aca_marketplace_source_outputs "
            "raises if the output misses a tax unit and writes it on every tax "
            "unit (tools/build_us_fiscal_refresh_release.py:4422, :4428). "
            f"{_LOADED_NOT_COMPUTED}; the stage's own pass reads the potential "
            "aca_ptc, not the take-up flag "
            "(_aca_source_tax_unit_table_from_simulation, :4216-4222)"
        ),
    }
)

#: The kinds an :func:`absences` entry carries. Only "owner-declared" needs
#: an owner's decision (N-2). "release-written" is a column of
#: :data:`RELEASE_WRITTEN_INPUTS`; "raw source column" a column the
#: stored-input rule ignores (not model-named, such as the upper-case survey
#: columns); "reviewed default fill" a value prepare's fill wrote in place
#: of a null, reviewed by staging's ``reviewed_engine_input_nulls`` register
#: (``tools/_legacy/build_us_acs_multispine_base.py``, summary key at
#: :2099, built at :2336-2392); "zero written" a zero staging wrote in place
#: of a measure, reviewed by the receipt the caller cites; "origin-native" a
#: column of :data:`ACS_NATIVE_AMOUNTS` CPS rows lack (d2283 lead call (a),
#: 11-18).
ABSENCE_KINDS = (
    "owner-declared",
    "release-written",
    "raw source column",
    "reviewed default fill",
    "zero written",
    "origin-native",
)
(
    _OWNER_DECLARED,
    _RELEASE_WRITTEN,
    _RAW_SOURCE,
    _DEFAULT_FILL,
    _ZERO_WRITTEN,
    _ORIGIN_NATIVE,
) = ABSENCE_KINDS

_ACS_TRANSFER_READER = (
    "acs_transfer (a recipient-side predictor, read on ACS rows only; "
    "acs_transfer._RECIPIENT_COMBINED_SOURCES)"
)

#: The six ACS-native reported amounts, each with its entity and the
#: release's readers: the ``acs_*`` keys of
#: ``operator_boundary._ACS_NATIVE_INPUT_CONTRACTS``. A test pins the two by
#: reading that module's source. The import would cost nothing extra
#: (``us_runtime``'s ``__init__`` already imports ``operator_boundary``); the
#: copy stays by d2283 lead call L8. Staging maps them from ACS source
#: columns on ACS rows only, and the stored-input register files each as
#: "not a model input", so CPS rows lack them by origin: :func:`absences`
#: files that as "origin-native", with no owner's decision. The ACS transfer
#: reads the three combined amounts as recipient-side predictors on ACS
#: rows; no release producer reads the three housing amounts, which are
#: source lineage the native-input receipt records.
ACS_NATIVE_AMOUNTS: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "acs_social_security_income": ("person", (_ACS_TRANSFER_READER,)),
        "acs_retirement_income": ("person", (_ACS_TRANSFER_READER,)),
        "acs_interest_dividend_rental_income": ("person", (_ACS_TRANSFER_READER,)),
        "acs_monthly_contract_rent": ("household", ()),
        "acs_monthly_gross_rent": ("household", ()),
        "acs_annual_property_tax": ("household", ()),
    }
)
_ORIGIN_NATIVE_EVIDENCE = (
    "an ACS-native reported amount, not a model input "
    "(operator_boundary._ACS_NATIVE_INPUT_CONTRACTS; the stored-input "
    "register's acs_ entries): staging maps it from ACS source columns on ACS "
    "rows only, so CPS rows lack it by origin"
)

_HOUSEHOLD_TYPE_READER = "household_type (#1610's household-type materializer)"

#: Raw source columns the release reads, each with its readers, so the
#: absences file never files them as harmless. The fourteen ACS release
#: predictors (``acs_release_predictors._MODEL_PREDICTORS``, :992-1006; a
#: test pins the two) feed these always-on producers; the join writes them
#: only on channel ``acs`` rows, which this module never writes, so on the
#: joined frame each producer reads them null on every row that lacks them.
#: ``TYPEHUGQ``, ``HRHTYPE``, ``A_EXPRRP`` and ``A_MARITL`` are what
#: PolicyEngine/microcosm#1610's household-type materializer
#: (``household_type_targets.household_type_household_values``, not on main
#: at this module's commit) reads and refuses on.
RAW_COLUMN_READERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                "PEDISDRS",
                "PEDISEAR",
                "PEDISEYE",
                "PEDISOUT",
                "PEDISPHY",
                "PEDISREM",
            ),
            ("ssi_disability_criteria",),
        ),
        **dict.fromkeys(
            ("PRDTRACE", "PRDTHSP"),
            ("org_wages", "scf_auto_loans", "scf_wealth"),
        ),
        "SPM_TENMORTSTATUS": ("sipp_vehicles",),
        "PEIOOCC": ("sipp_tips",),
        **dict.fromkeys(("POCCU2", "WEIND", "WEMIND", "WKSWORK"), ("org_wages",)),
        **dict.fromkeys(
            ("TYPEHUGQ", "HRHTYPE", "A_EXPRRP", "A_MARITL"),
            (_HOUSEHOLD_TYPE_READER,),
        ),
    }
)

#: The ACS RELSHIPP role rule and its SPM partition, as staging's receipt
#: states them: copies of ``acs_inputs.ACS_SPM_ROLE_RULE`` and
#: ``acs_inputs.ACS_SPM_ROLE_PARTITION``, which a test pins by reading that
#: module's source. The import would cost nothing extra (``us_runtime``'s
#: ``__init__`` already imports ``acs_inputs`` through ``operator_boundary``);
#: the copies stay by d2283 lead call L8.
_ACS_SPM_ROLE_RULE = (
    "RELSHIPP == 20 OR RELSHIPP in {21, 23} OR RELSHIPP in {37, 38}, "
    "on the one-SPM-unit-per-household ACS partition"
)
_ACS_SPM_ROLE_PARTITION = "household (ACS PUMS carries no SPM_ID)"
#: That rule's ``RELSHIPP`` codes: the role holders (20 the reference
#: person, 21/23 the spouse, 37/38 the sole group-quarters person) and the
#: codes it reads at all (20 to 38). Copies of ``acs_inputs``'
#: ``_ACS_REFERENCE_PERSON``, ``_ACS_SPOUSE_OF_REFERENCE``,
#: ``_ACS_GROUP_QUARTERS_PERSON`` and ``_ACS_RELSHIPP_CODES``, which a test
#: pins, by lead call L8.
_RELSHIPP = "RELSHIPP"
_ACS_ROLE_RELSHIPP = frozenset({20, 21, 23, 37, 38})
_ACS_RELSHIPP_CODES = frozenset(range(20, 39))

#: The keys staging's ACS role receipt carries
#: (``acs_inputs.with_acs_spm_independence_role``, :478-511; a test pins the
#: two), and the values that identify it as the ACS RELSHIPP rule's.
_ACS_ROLE_RECEIPT_KEYS = frozenset(
    {
        "column",
        "entity",
        "source_columns",
        "rule",
        "partition",
        "asec_rule",
        "adult_rule",
        "provenance",
        "persons",
        "spm_units",
        "role_true_persons",
        "role_branches",
        "persons_aged_15_to_17",
        "independent_minor_persons",
        "housing_units_classified_only_by_role",
        "group_quarters_units_without_classified_adult",
        "weighted_role_share",
        "weighted_minor_role_share",
    }
)
_ACS_ROLE_RECEIPT_IDENTITY: Mapping[str, object] = MappingProxyType(
    {
        "column": NATIVE_SPM_ROLE,
        "entity": US_SCHEMA.person_entity,
        "source_columns": ["RELSHIPP"],
        "rule": _ACS_SPM_ROLE_RULE,
        "partition": _ACS_SPM_ROLE_PARTITION,
        "asec_rule": SPM_ROLE_RULE,
        "adult_rule": SPM_ADULT_RULE,
        "provenance": "acs_2024_1yr_native",
    }
)
#: The receipt's counts and the frame's ACS counts they must cover: the
#: receipt counts every staged ACS person, the frame only the selected ones.
_ACS_ROLE_RECEIPT_COVERAGE = (
    ("persons", "persons"),
    ("spm_units", "spm_units"),
    ("role_true_persons", "true_role_persons"),
)
#: The keys :func:`~microcosm.build.us_runtime.spm_independence_role.us_spm_independence_role_signal_gate`
#: compares with the frame (``spm_independence_role.py`` :574-586).
_ROLE_GATE_COUNT_KEYS = (
    "persons_joined",
    "table_rows",
    "native_spm_units",
    "true_role_persons",
)
_ROLE_GATE_DIGEST_KEY = "person_role_binding_sha256"
_ROLE_SOURCE_COLUMN = "source_column"

#: The top-level keys and per-fill keys of the reviewed default fill's
#: manifest (``tools/build_us_acs_local_release.py``, ``fill_reviewed_nulls``,
#: :1135-1144 and :1151-1165).
_FILL_MANIFEST_KEYS = frozenset(
    {
        "period",
        "register_entries",
        "columns_filled",
        "total_values_filled",
        "count_mismatch_warnings",
        "unregistered_violations",
        "fills",
    }
)
_FILL_KEYS = frozenset(
    {
        "entity",
        "column",
        "filled_rows",
        "rows",
        "fill_value",
        "fill_kind",
        "register_missing_rows",
        "missing_rows_by_spine",
    }
)
#: The absences file's layout version.
_ABSENCES_SCHEMA_VERSION = 1


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


#: Why :func:`_complex_or_temporal`'s refusals refuse, in their messages.
_READ_AS_NUMBERS = (
    "the contract reads only integer, real-number and string data as numbers, "
    "since a conversion would drop an imaginary part, turn a time into a count "
    "or misread other data."
)


def _is_real_number(value: object) -> bool:
    """``value`` is a real-number scalar, and not a boolean or a numpy time.

    Numpy registers ``timedelta64`` as an integer (it derives it from
    ``signedinteger``), so an ``isinstance`` test against :mod:`numbers`
    takes it for one. Comparisons read it as its count of time units for
    every unit, and for some units ``int()`` does too
    (``np.timedelta64(1, "ns") in (1, 2, 3)`` holds). A ``timedelta64`` and a
    ``datetime64`` are never numbers here (PR #1685 review r3, should-fix 2).
    """

    return isinstance(value, numbers.Real) and not isinstance(
        value, bool | np.bool_ | np.timedelta64 | np.datetime64
    )


def _is_integer(value: object) -> bool:
    """``value`` is an integer scalar: a real number (:func:`_is_real_number`)
    of an integral type."""

    return _is_real_number(value) and isinstance(value, numbers.Integral)


def _readable_value(value: object) -> bool:
    """``value`` may sit in an object column read as numbers.

    A string, a real number (booleans included, which ``pd.to_numeric``
    reads as 0 and 1, and ``decimal.Decimal``), or a null the null checks
    after this one count. Anything else refuses: a complex value, a time,
    and any other object ``pd.to_numeric`` might read as a number (a
    zero-dimensional numpy array of a ``timedelta64`` or a complex value
    reads as its count or its real part).
    """

    return (
        value is None
        or value is pd.NA
        or isinstance(value, str | bytes | bool | np.bool_ | decimal.Decimal)
        or _is_real_number(value)
    )


def _complex_or_temporal(series: pd.Series) -> str | None:
    """What ``series`` holds other than integers, real numbers or strings.

    Returns None when the data may be read as numbers. The name is older
    than the rule: review r2 found complex and temporal data read as
    numbers, and review r3 an Arrow dictionary of times (PR #1685,
    should-fix 1 each), so the check now names the data it accepts and
    refuses the rest. It reads the value type behind a categorical's
    categories and behind an Arrow dictionary, then accepts an integer,
    unsigned or floating type, a string type, or an object column whose
    every value passes :func:`_readable_value`. Any other dtype refuses by
    name: complex, datetime, timedelta, period, interval, boolean, Arrow
    decimal and the rest.
    """

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        found = _complex_or_temporal(pd.Series(dtype.categories))
        return None if found is None else f"categories of {found}"
    if isinstance(dtype, pd.ArrowDtype):
        import pyarrow as pa

        value_type = dtype.pyarrow_dtype
        if pa.types.is_dictionary(value_type):
            value_type = value_type.value_type
        readable = (
            pa.types.is_integer(value_type)
            or pa.types.is_floating(value_type)
            or pa.types.is_string(value_type)
            or pa.types.is_large_string(value_type)
        )
        return None if readable else f"{dtype} values"
    if isinstance(dtype, pd.StringDtype) or dtype.kind in ("i", "u", "f"):
        return None
    if not pd.api.types.is_object_dtype(dtype):
        return f"{dtype} values"
    values = series.to_numpy()
    unreadable = {
        kind
        for kind, value in {type(value): value for value in values}.items()
        if not _readable_value(value)
    }
    if not unreadable:
        return None
    found = [value for value in values if type(value) in unreadable]
    return (
        f"{len(found)} value(s) that are not real numbers or strings, such as "
        f"{found[:3]!r}"
    )


def _integral_ids(series: pd.Series, *, refusal: str) -> np.ndarray:
    """``series`` as int64 if every value is a non-null integer, else refuse.

    Integer dtypes pass as they are; float and object values must be finite,
    integral and inside int64 (staging's pooled form turns an origin's
    integer column into float64 when the other origin lacks it). Object
    values go through ``pd.to_numeric``, so digit strings (``"601"``,
    ``"6.0"``) and booleans held in an object column pass as their numbers.
    A boolean dtype refuses, and so does any data but integers, real
    numbers and strings, as a dtype or as values, before that conversion
    reads it (:func:`_complex_or_temporal`).
    """

    if pd.api.types.is_bool_dtype(series.dtype):
        raise OnePopulationContractError(f"{refusal}: it holds booleans, not ids.")
    found = _complex_or_temporal(series)
    if found is not None:
        raise OnePopulationContractError(
            f"{refusal}: it holds {found}, not integers; {_READ_AS_NUMBERS}"
        )
    nulls = int(series.isna().sum())
    if nulls:
        raise OnePopulationContractError(f"{refusal}: {nulls} value(s) are null.")
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.to_numpy(dtype=np.int64)
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_integer_dtype(numeric.dtype):
        # Object-held Python ints: exact, never through float64.
        return numeric.to_numpy(dtype=np.int64)
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    bad = (
        ~np.isfinite(values) | (values != np.floor(values)) | (np.abs(values) >= 2**63)
    )
    if bad.any():
        examples = series[bad].head(5).tolist()
        raise OnePopulationContractError(
            f"{refusal}: {int(bad.sum())} value(s) are not integers, such as "
            f"{examples!r}."
        )
    return values.astype(np.int64)


def _origin_tables(
    tables: Mapping[str, pd.DataFrame], *, origin: str
) -> dict[str, pd.DataFrame]:
    """The six US entity tables of one origin, or a refusal."""

    if not isinstance(tables, Mapping):
        raise TypeError(
            f"{origin} tables must be a mapping of entity to table, got "
            f"{type(tables).__name__}."
        )
    expected = set(US_SCHEMA.entities)
    missing = sorted(expected - set(tables))
    unexpected = sorted(str(name) for name in set(tables) - expected)
    if missing or unexpected:
        raise OnePopulationContractError(
            f"{origin} rows lack complete support metadata: the tables must be "
            f"the six US entities; missing {missing}, unexpected {unexpected}."
        )
    for entity in US_SCHEMA.entities:
        if not isinstance(tables[entity], pd.DataFrame):
            raise TypeError(
                f"{origin} {entity!r} table must be a DataFrame, got "
                f"{type(tables[entity]).__name__}."
            )
    return {entity: tables[entity] for entity in US_SCHEMA.entities}


def _require_support_columns(
    table: pd.DataFrame, entity: str, *, origin: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Presence and completeness of one table's support metadata (P5).

    Returns the integral ids, source ids and clone indices.
    """

    source, channel, clone, spine_source, spine = _provenance_columns(entity)
    id_column = US_SCHEMA.entity_id_column(entity)
    rule = f"{origin} rows lack complete support metadata"
    missing = [column for column in (source, channel, clone) if column not in table]
    if missing:
        raise OnePopulationContractError(
            f"{rule}: {entity!r} is missing {missing}; every entity needs its "
            "source id, support channel and clone index."
        )
    if id_column not in table:
        raise OnePopulationContractError(
            f"{rule}: {entity!r} is missing {id_column!r}."
        )
    if spine_source in table:
        raise OnePopulationContractError(
            f"{origin} rows already carry {spine_source!r}: the contract writes "
            "the spine source id, so its input must not hold one."
        )
    for column in (source, channel, clone):
        nulls = int(table[column].isna().sum())
        if nulls:
            raise OnePopulationContractError(
                f"{rule}: {column!r} has {nulls} null value(s)."
            )
    return (
        _integral_ids(
            table[id_column], refusal=f"{origin} {id_column!r} must hold ids"
        ),
        _integral_ids(
            table[source], refusal=f"{origin} {source!r} must hold source ids"
        ),
        _integral_ids(
            table[clone], refusal=f"{origin} {clone!r} must hold clone indices"
        ),
    )


def _unexpected(values: pd.Series, allowed: Collection[object]) -> list[str]:
    """Sorted reprs of the values outside ``allowed`` (strings only allowed)."""

    good = values.map(lambda value: isinstance(value, str) and value in allowed)
    return sorted({repr(value) for value in values[~good.astype(bool)]})


def _require_cps_support_metadata(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on the parent export's rows: Route A's labels, complete."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        _, channel, clone, _, spine = _provenance_columns(entity)
        _, _, clones = _require_support_columns(table, entity, origin=_CPS)
        bad_clones = sorted(set(clones.tolist()) - set(_CPS_CLONE_INDICES))
        if bad_clones:
            raise OnePopulationContractError(
                "CPS clone indices must be 0, 1 or 2 (Route A's native rows, PUF "
                f"tax-detail copies and capital-gains tail copies); {clone!r} "
                f"holds {bad_clones[:5]}."
            )
        bad_channels = _unexpected(table[channel], _CPS_CHANNELS)
        if bad_channels:
            raise OnePopulationContractError(
                f"CPS rows must carry channel {BASE_ASEC_SUPPORT_CHANNEL!r} or "
                f"{PUF_TAX_DETAIL_SUPPORT_CHANNEL!r} (Route A's labels); "
                f"{channel!r} holds {bad_channels[:5]}."
            )
        native = table[channel].eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
        disagree = int(np.count_nonzero(native != (clones == 0)))
        if disagree:
            raise OnePopulationContractError(
                f"CPS rows must carry channel {BASE_ASEC_SUPPORT_CHANNEL!r} exactly "
                f"on clone index 0; {disagree} {entity!r} row(s) disagree."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(CPS_SPINE_TAG)
            if wrong.any():
                raise OnePopulationContractError(
                    f"CPS rows carry a spine tag other than {CPS_SPINE_TAG!r} in "
                    f"{spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}; "
                    "staging tags the donor's rows "
                    f"{CPS_SPINE_TAG!r}."
                )


def _require_acs_support_metadata(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on staging's ACS rows: staging's labels, complete, one id offset."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        _, channel, clone, _, spine = _provenance_columns(entity)
        ids, sources, clones = _require_support_columns(table, entity, origin=_ACS)
        bad_channels = _unexpected(table[channel], (ACS_SUPPORT_CHANNEL,))
        if bad_channels:
            raise OnePopulationContractError(
                f"ACS rows must carry channel {ACS_SUPPORT_CHANNEL!r} (staging's); "
                f"{channel!r} holds {bad_channels[:5]}."
            )
        bad_clones = sorted(set(clones.tolist()) - {0})
        if bad_clones:
            raise OnePopulationContractError(
                f"ACS rows must carry clone index 0; {clone!r} holds {bad_clones[:5]}."
            )
        offsets = sorted(set((ids - sources).tolist()))
        if len(offsets) > 1:
            raise OnePopulationContractError(
                f"ACS {entity!r} ids must sit one constant offset from staging's "
                f"native source ids (staging shifts each entity's ids by one "
                f"amount); found offsets {offsets[:5]}."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(ACS_SUPPORT_CHANNEL)
            if wrong.any():
                raise OnePopulationContractError(
                    f"ACS rows carry a spine tag other than {ACS_SUPPORT_CHANNEL!r} "
                    f"in {spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}."
                )


def _keyed_weights(values: object, *, origin: str) -> dict[object, float]:
    """One origin's household weights, by key, before they meet its table.

    Only a keyed input is taken (d2283 lead call L3's check): PR 1's ACS
    starting weights come in selection order and staging's ACS households
    in id order, so a weight given by position cannot be tied to its
    household, and no later clause would see the permutation. The weights
    pass the shared guard (:func:`_complex_or_temporal`) before the float64
    conversion, which drops a complex weight's imaginary part and reads a
    ``timedelta64`` or ``datetime64`` as its count (PR #1685 review r3,
    should-fix 2).
    """

    key = _WEIGHT_KEYS[origin]
    if isinstance(values, pd.Series):
        keys, given = values.index.tolist(), values.to_numpy()
        found = _complex_or_temporal(values)
    elif isinstance(values, Mapping):
        keys, given = list(values), list(values.values())
        found = _complex_or_temporal(pd.Series(given, dtype=object))
    else:
        raise OnePopulationContractError(
            f"{origin} household weights must be keyed by {key} (a pandas Series "
            f"or a mapping), not a {type(values).__name__}: a weight given by "
            "position cannot be tied to its household."
        )
    if found is not None:
        raise OnePopulationContractError(
            f"{origin} household weights must be real numbers; they hold {found}: "
            f"{_READ_AS_NUMBERS}"
        )
    try:
        array = np.asarray(given, dtype=np.float64)
    except (TypeError, ValueError):
        array = None
    if (
        array is None
        or array.ndim != 1
        or not np.isfinite(array).all()
        or (array < 0).any()
    ):
        raise OnePopulationContractError(
            f"{origin} household weights must be finite and non-negative."
        )
    seen: set[object] = set()
    repeated = []
    for k in keys:
        if k in seen:
            repeated.append(k)
        seen.add(k)
    if repeated:
        raise OnePopulationContractError(
            f"{origin} household weights repeat a key: {repeated[:5]!r}."
        )
    return dict(zip(keys, array.tolist(), strict=True))


def _aligned_weights(
    weights: Mapping[object, float], keys: list[object], *, origin: str
) -> np.ndarray:
    """``weights`` in the order of ``keys``, each household's by its key."""

    missing = [k for k in keys if k not in weights]
    extra = sorted(repr(k) for k in set(weights) - set(keys))
    if missing or extra:
        raise OnePopulationContractError(
            f"{origin} household weights must name each {origin} household by its "
            f"{_WEIGHT_KEYS[origin]} exactly once: {len(missing)} household(s) "
            f"have no weight (such as {missing[:3]!r}) and {len(extra)} weight(s) "
            f"name no household (such as {extra[:3]})."
        )
    return np.array([weights[k] for k in keys], dtype=np.float64)


def _acs_weight_keys(household: pd.DataFrame) -> list[object]:
    """Each ACS household's weight key, ``acs_unit_key(SERIALNO)``, in row order."""

    try:
        return [acs_unit_key(serial) for serial in household[_SERIALNO]]
    except ValueError as exc:
        raise OnePopulationContractError(
            f"ACS household weights need each household's SERIALNO key: {exc}"
        ) from exc


def _object_column(value: object, index: pd.Index) -> pd.Series:
    return pd.Series(value, index=index, dtype=object)


def _assembled_cps(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """P1 and Q3: keep Route A's labels, add the spine source id and tag."""

    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source, _, _, spine_source, spine = _provenance_columns(entity)
        table = tables[entity].copy()
        table[spine_source] = table[source].to_numpy(copy=True)
        table[spine] = _object_column(CPS_SPINE_TAG, table.index)
        result[entity] = table
    return result


def _assembled_acs(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """P2: ``assemble_spines``'s rule on staging's rows, channel unchanged.

    The spine source id takes staging's native source id and the source id
    the pooled ``<entity>_id``; channel ``acs_2024_1yr`` and clone index 0
    stay. Nothing writes the predictor join's ``acs`` channel (P3).
    """

    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source, _, _, spine_source, _ = _provenance_columns(entity)
        table = tables[entity].copy()
        table[spine_source] = table[source].to_numpy(copy=True)
        table[source] = table[US_SCHEMA.entity_id_column(entity)].to_numpy(copy=True)
        result[entity] = table
    return result


def _require_spine_tags(origins: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    """Q3: no row of the joined frame may carry a null ``<entity>_spine``."""

    for origin, tables in origins.items():
        for entity in US_SCHEMA.entities:
            spine = _spine_column(entity)
            table = tables[entity]
            nulls = len(table) if spine not in table else int(table[spine].isna().sum())
            if nulls:
                raise OnePopulationContractError(
                    f"A null spine tag: {nulls} {origin} {entity!r} row(s) carry no "
                    f"{spine!r} after the rewrite; every row needs its origin's tag "
                    f"({CPS_SPINE_TAG!r} on CPS rows, {ACS_SUPPORT_CHANNEL!r} on "
                    "ACS rows)."
                )


def _require_disjoint_ids(origins: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    """P6b: each entity's ids unique across origins, memberships resolved.

    Every person membership must resolve to exactly one group row of its own
    origin, and every group row must hold a person of its origin. Staging
    offsets ACS ids clear of its own donor only, so an ACS frame staged on
    another donor than the parent export collides here. Each membership
    column passes the shared guard (:func:`_complex_or_temporal`) before it
    is matched: ``isin`` matches a complex ``5+0j`` with the id 5 (PR #1685
    review r3, nit 2). This is the one place the five person links resolve,
    on both paths, before any other clause reads them.
    """

    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        ids = {
            origin: _integral_ids(
                tables[entity][id_column],
                refusal=f"{origin} {id_column!r} must hold ids",
            )
            for origin, tables in origins.items()
        }
        joined = pd.Series(np.concatenate(list(ids.values())))
        repeated = joined[joined.duplicated(keep=False)]
        if len(repeated):
            across = sorted(set(ids[_CPS].tolist()) & set(ids[_ACS].tolist()))
            where = "across origins" if across else "within one origin"
            raise OnePopulationContractError(
                f"Frame ids collide: {id_column!r} repeats {where} "
                f"({sorted(set(repeated.tolist()))[:5]}); each entity's id must be "
                "unique across the joined CPS and ACS rows (an ACS frame staged on "
                "another donor than the parent export does this)."
            )
    for origin, tables in origins.items():
        person = tables[US_SCHEMA.person_entity]
        for group in US_SCHEMA.group_entities:
            membership_column = US_SCHEMA.membership_column(group)
            group_ids = tables[group][US_SCHEMA.id_column(group)]
            if membership_column not in person:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {origin} persons lack "
                    f"{membership_column!r}."
                )
            membership = person[membership_column]
            found = _complex_or_temporal(membership)
            if found is not None:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {origin} persons' "
                    f"{membership_column!r} holds {found}, not ids; "
                    f"{_READ_AS_NUMBERS}"
                )
            unresolved = int((~membership.isin(group_ids)).sum())
            if unresolved:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {unresolved} {origin} person(s)' "
                    f"{membership_column!r} match no {origin} {group} row; every "
                    "person membership must resolve to exactly one group row of its "
                    "own origin."
                )
            empty = int((~group_ids.isin(membership)).sum())
            if empty:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {empty} {origin} {group} row(s) "
                    f"hold no {origin} person."
                )


def _require_nested_groups(origins: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    """P6c: every group unit inside one household; each ACS household one SPM unit.

    A tax unit, SPM unit, family or marital unit whose persons sit in two
    households resolves every id and keeps every channel, so P6b and the
    assembly validator pass it, but the release reads it as one unit split
    across households (the ACA pass refuses one, and a tax unit whose
    households carry unequal weights has no single weight). ACS rows must
    also keep staging's partition, one SPM unit per household
    (``acs_inputs.ACS_SPM_ROLE_PARTITION``), the only partition the ACS role
    rule and its receipt are defined on (PR #1685 review, blocking 3). It
    runs after P6b, which resolves every membership within its origin.
    """

    household_column = US_SCHEMA.membership_column("household")
    for origin, tables in origins.items():
        person = tables[US_SCHEMA.person_entity]
        if not len(person):
            continue
        households = person[household_column].to_numpy()
        for group in US_SCHEMA.group_entities:
            if group == "household":
                continue
            units = pd.DataFrame(
                {
                    "unit": person[US_SCHEMA.membership_column(group)].to_numpy(),
                    "household": households,
                }
            )
            spans = units.groupby("unit", sort=True)["household"].nunique()
            split = spans.index[spans.to_numpy() > 1].tolist()
            if split:
                raise OnePopulationContractError(
                    f"Group units do not nest in households: {len(split)} {origin} "
                    f"{group} unit(s) hold persons of more than one household, "
                    f"such as {split[:5]}; every tax unit, SPM unit, family and "
                    "marital unit lies inside one household."
                )
        if origin != _ACS:
            continue
        spm_units = pd.DataFrame(
            {
                "household": households,
                "unit": person[US_SCHEMA.membership_column("spm_unit")].to_numpy(),
            }
        )
        per_household = spm_units.groupby("household", sort=True)["unit"].nunique()
        several = per_household.index[per_household.to_numpy() > 1].tolist()
        if several:
            raise OnePopulationContractError(
                "ACS SPM units must be the households (staging's one-SPM-unit-"
                f"per-household partition): {len(several)} ACS household(s) hold "
                f"more than one SPM unit, such as {several[:5]}."
            )


def _require_unique_source_ranks(
    origins: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    """P6: no (source id, rank) pair repeats anywhere in the joined frame.

    The rank is :func:`support_copy_rank_series`'s, the clone index, read on
    the rewritten (assembled) rows.
    """

    for entity in US_SCHEMA.entities:
        source = support_source_id_column(entity)
        pairs = pd.concat(
            [
                pd.DataFrame(
                    {
                        "source_id": _integral_ids(
                            tables[entity][source],
                            refusal=f"{origin} {source!r} must hold source ids",
                        ),
                        "rank": support_copy_rank_series(
                            tables[entity], entity=entity
                        ).to_numpy(),
                        "origin": origin,
                    }
                )
                for origin, tables in origins.items()
            ],
            ignore_index=True,
        )
        repeated = pairs[pairs.duplicated(["source_id", "rank"], keep=False)]
        if len(repeated):
            examples = sorted(
                {
                    (int(row.source_id), int(row.rank))
                    for row in repeated.itertuples(index=False)
                }
            )
            origins_hit = sorted(set(repeated["origin"]))
            raise OnePopulationContractError(
                f"A (source id, rank) pair repeats in {entity!r}: {examples[:5]} "
                f"({len(examples)} pair(s), on {origins_hit} rows); each support "
                "copy needs its own source id and clone index."
            )


#: Staging's lineage for the selected ACS persons: each person's native
#: (household id, person id) to its (``SERIALNO``, ``SPORDER``).
_Lineage = dict[tuple[int, int], tuple[str, int]]


def _integral_value(value: object) -> int | None:
    """``value`` as an int when it is a non-boolean integral number, else None.

    A ``numpy.timedelta64`` registers as an integer (numpy derives it from
    ``signedinteger``), and for some units ``int()`` reads it as its count of
    time units (``"ns"``; ``"us"`` raises), so it is refused like a boolean
    (PR #1685 review r2, should-fix 1), through the shared
    :func:`_is_real_number` (review r3, should-fix 2).
    """

    if not _is_real_number(value):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    as_float = float(value)
    return int(as_float) if as_float.is_integer() else None


def _lineage_rows(staging_lineage: object) -> _Lineage:
    """Staging's lineage, keyed by each ACS person's native ids, checked.

    A set of (``SERIALNO``, ``SPORDER``) pairs is refused: it cannot tie a
    ``SERIALNO`` to its household, so two households that swap serial
    numbers would keep the set and trade weights (PR #1685 review,
    blocking 2).
    """

    rule = "ACS lineage does not survive"
    if not isinstance(staging_lineage, Mapping):
        raise TypeError(
            "staging_lineage must map each staged ACS person's native (household "
            "id, person id) to its (SERIALNO, SPORDER), got "
            f"{type(staging_lineage).__name__}."
        )
    rows: _Lineage = {}
    for key, value in staging_lineage.items():
        household_id = person_id = None
        if isinstance(key, tuple) and len(key) == 2:
            household_id, person_id = (_integral_value(part) for part in key)
        if household_id is None or person_id is None:
            raise OnePopulationContractError(
                f"{rule}: staging's lineage has the key {key!r}, not a native "
                "(household id, person id) pair of integers."
            )
        pair = value if isinstance(value, tuple) and len(value) == 2 else None
        sporder = None if pair is None else _integral_value(pair[1])
        if (
            pair is None
            or not isinstance(pair[0], str)
            or not pair[0]
            or sporder is None
        ):
            raise OnePopulationContractError(
                f"{rule}: staging's lineage maps {key!r} to {value!r}, not a "
                f"({_SERIALNO}, {_SPORDER}) pair; {_SERIALNO} must be a non-empty "
                f"string and {_SPORDER} an integer."
            )
        native = (household_id, person_id)
        if native in rows:
            raise OnePopulationContractError(
                f"{rule}: staging's lineage repeats the native key {native!r}."
            )
        rows[native] = (pair[0], sporder)
    return rows


def _require_acs_lineage(
    tables: Mapping[str, pd.DataFrame],
    staging: Mapping[tuple[int, int], tuple[str, int]] | None,
) -> None:
    """P7: every ACS row's ``SERIALNO``/``SPORDER`` lineage survives.

    ``SERIALNO`` is a non-empty string, unique per ACS household; ``SPORDER``
    an integer, with (``SERIALNO``, ``SPORDER``) unique per ACS person through
    the household link; ``source_household_id`` an integer equal to the
    linked household's ``household_spine_source_id`` (staging's native id);
    ``source_person_id`` the string ``str(SPORDER)``; ``source_year`` the
    integer 2024. A null in any of them refuses before a comparison reads it
    (pandas' nullable dtypes hold ``pd.NA``, which a sum skips and a
    comparison cannot decide). When ``staging`` is given, each ACS person's
    (``SERIALNO``, ``SPORDER``), keyed by its native (household id, person
    id), must equal staging's, person by person: the same keys, and the
    same pair under each key.
    """

    rule = "ACS lineage does not survive"
    household = tables["household"]
    person = tables[US_SCHEMA.person_entity]
    for name, table, columns in (
        ("household", household, (_SERIALNO,)),
        (
            "person",
            person,
            (_SPORDER, _SOURCE_YEAR, _SOURCE_HOUSEHOLD_ID, _SOURCE_PERSON_ID),
        ),
    ):
        missing = [column for column in columns if column not in table]
        if missing:
            raise OnePopulationContractError(
                f"{rule}: the ACS {name} table lacks {missing}."
            )
    serials = household[_SERIALNO]
    bad = ~serials.map(lambda value: isinstance(value, str) and bool(value)).astype(
        bool
    )
    if bad.any():
        raise OnePopulationContractError(
            f"{rule}: every ACS household needs a non-empty string {_SERIALNO}; "
            f"{int(bad.sum())} hold {serials[bad].head(5).tolist()!r}."
        )
    if serials.duplicated().any():
        raise OnePopulationContractError(
            f"{rule}: {_SERIALNO} repeats across ACS households: "
            f"{sorted(set(serials[serials.duplicated()]))[:5]}."
        )
    sporder = _integral_ids(
        person[_SPORDER],
        refusal=f"{rule}: every ACS person needs an integral {_SPORDER}",
    )
    household_id = US_SCHEMA.id_column("household")
    by_household = household.set_index(household_id)
    link = person[US_SCHEMA.membership_column("household")]
    person_serial = link.map(by_household[_SERIALNO]).to_numpy(dtype=object)
    lineage = pd.DataFrame({"serial": person_serial, "sporder": sporder})
    repeated = lineage.duplicated(keep=False)
    if repeated.any():
        raise OnePopulationContractError(
            f"{rule}: ({_SERIALNO}, {_SPORDER}) repeats on "
            f"{int(repeated.sum())} ACS person(s)."
        )
    household_spine = spine_source_id_column("household")
    native = _integral_ids(
        link.map(by_household[household_spine]),
        refusal=f"{rule}: every ACS person's household needs an integral "
        f"{household_spine}",
    )
    # Nullable dtypes hold pd.NA, which a comparison cannot decide and a sum
    # skips, so every lineage value is parsed or null-checked before it is
    # compared (PR #1685 review, should-fix 2).
    source_household = _integral_ids(
        person[_SOURCE_HOUSEHOLD_ID],
        refusal=f"{rule}: every ACS person needs an integral {_SOURCE_HOUSEHOLD_ID}",
    )
    differs = int(np.count_nonzero(source_household != native))
    if differs:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_HOUSEHOLD_ID} differs from the linked household's "
            f"native id ({household_spine}) on {differs} ACS person(s)."
        )
    nulls = int(person[_SOURCE_PERSON_ID].isna().sum())
    if nulls:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_PERSON_ID} is null on {nulls} ACS person(s)."
        )
    expected = [str(value) for value in sporder.tolist()]
    actual = person[_SOURCE_PERSON_ID].tolist()
    differs = sum(
        1
        for want, got in zip(expected, actual, strict=True)
        if not (isinstance(got, str) and got == want)
    )
    if differs:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_PERSON_ID} differs from str({_SPORDER}) on {differs} "
            "ACS person(s)."
        )
    years = _integral_ids(
        person[_SOURCE_YEAR],
        refusal=f"{rule}: every ACS person needs an integral {_SOURCE_YEAR}",
    )
    wrong_year = int(np.count_nonzero(years != _ACS_SOURCE_YEAR))
    if wrong_year:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_YEAR} must be {_ACS_SOURCE_YEAR} on every ACS person; "
            f"{wrong_year} differ."
        )
    if staging is None:
        return
    person_spine = spine_source_id_column("person")
    native_person = _integral_ids(
        person[person_spine],
        refusal=f"{rule}: every ACS person needs an integral {person_spine}",
    )
    selected = dict(
        zip(
            zip(native.tolist(), native_person.tolist(), strict=True),
            zip(lineage["serial"].tolist(), lineage["sporder"].tolist(), strict=True),
            strict=True,
        )
    )
    if len(selected) != len(person):
        raise OnePopulationContractError(
            f"{rule}: a native (household id, person id) repeats on the ACS "
            "persons, so staging's lineage cannot be matched person by person."
        )
    _require_staging_lineage(selected, staging)


def _require_staging_lineage(
    selected: Mapping[tuple[int, int], tuple[str, int]],
    staging: Mapping[tuple[int, int], tuple[str, int]],
) -> None:
    """P7's last rule: each selected ACS person's pair is staging's, by key.

    The key is the person's native (household id, person id), so a
    ``SERIALNO`` swapped between two households, or a ``SPORDER`` swapped
    between two persons, is a moved pair even when the set of pairs holds
    (PR #1685 review, blocking 2).
    """

    missing = sorted(set(staging) - set(selected))
    extra = sorted(set(selected) - set(staging))
    moved = sorted(
        key for key in set(selected) & set(staging) if selected[key] != staging[key]
    )
    if missing or extra or moved:
        examples = [
            f"{key}: {selected[key]} here, {staging[key]} in staging"
            for key in moved[:3]
        ]
        raise OnePopulationContractError(
            "ACS lineage does not survive: the selected ACS persons' "
            f"({_SERIALNO}, {_SPORDER}) lineage differs from staging's, keyed by "
            f"native (household id, person id); {len(missing)} missing (such as "
            f"{missing[:3]}), {len(extra)} extra (such as {extra[:3]}), "
            f"{len(moved)} moved (such as {examples})."
        )


def _joined_tables(
    origins: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """CPS rows then ACS rows per entity, group tables sorted by id.

    A zero-row table is left out of the join, so an empty origin changes no
    dtype of the other's columns. Returns the tables and the household
    order applied to the concatenated rows.
    """

    tables: dict[str, pd.DataFrame] = {}
    household_order = np.arange(0)
    for entity in US_SCHEMA.entities:
        parts = [table[entity] for table in origins.values() if len(table[entity])]
        if not parts:
            parts = [origins[_CPS][entity]]
        table = pd.concat(parts, ignore_index=True, sort=False)
        if entity in US_SCHEMA.group_entities:
            order = np.argsort(
                table[US_SCHEMA.id_column(entity)].to_numpy(), kind="stable"
            )
            table = table.iloc[order].reset_index(drop=True)
            if entity == "household":
                household_order = order
        tables[entity] = table
    return tables, household_order


def to_assembled_provenance(
    cps_tables: Mapping[str, pd.DataFrame],
    acs_tables: Mapping[str, pd.DataFrame],
    *,
    cps_household_weights: pd.Series | Mapping[object, float],
    acs_household_weights: pd.Series | Mapping[object, float],
    staging_lineage: Mapping[tuple[int, int], tuple[str, int]],
) -> Frame:
    """Join CPS and ACS rows in the assembled provenance form (d2283 K-1).

    CPS rows (the parent export's) keep Route A's source id, channel and
    clone index (``asec`` on clone 0, ``puf_tax_detail`` on clones 1 and 2),
    gain ``<entity>_spine_source_id`` equal to the source id, and are tagged
    ``<entity>_spine = asec_puf``, staging's donor tag (d2283 Q3). ACS rows
    (staging's) follow
    :func:`~microcosm.build.us_runtime.spine_assembly.assemble_spines`'s rule,
    re-implemented because that function refuses inputs that already carry
    provenance: the spine source id takes staging's native source id, the
    source id takes the pooled ``<entity>_id`` (staging offset it clear of
    the donor's ids), the channel stays ``acs_2024_1yr`` and the clone index
    0, and staging's ``<entity>_spine`` tag stays. No row is labelled
    ``acs``, so the ACS predictor join passes the frame through. Ids are
    never re-offset (d2283 Q9): an ACS frame staged on another donor than
    the parent export is refused.

    The clauses run in order, each refusing with its own rule's message:

    - complete support metadata on each origin's input (P5). CPS: source id,
      channel and clone index on all six entities, no nulls, channel
      ``asec`` or ``puf_tax_detail``, ``asec`` exactly on clone index 0,
      clone index 0, 1 or 2, no spine source id, and no spine tag other than
      ``asec_puf``. ACS: the same three columns, no nulls, channel
      ``acs_2024_1yr``, clone index 0, ``<entity>_id - <entity>_source_id``
      one constant per entity, no spine source id, and no spine tag other
      than ``acs_2024_1yr``;
    - household weights keyed, each key once, finite and non-negative
      (d2283 lead call L3's check);
    - a non-null ``<entity>_spine`` on every row after the rewrite (Q3);
    - each entity's id unique across the joined rows, every person
      membership resolving to exactly one group row of its own origin, and
      every group row holding a person (P6b);
    - no (source id, rank) pair repeated anywhere, the rank being
      :func:`~microcosm.build.us_runtime.support_provenance.support_copy_rank_series`'s
      (P6);
    - the ACS rows' ``SERIALNO``/``SPORDER`` lineage, against
      ``staging_lineage`` person by person, keyed by native ids, before any
      weight is looked up by ``SERIALNO`` (P7; see
      :func:`_require_acs_lineage`);
    - one weight per household, found by its key: no household without a
      weight, no weight without a household;
    - after the frame and its record pass the assembly validator, every
      tax unit, SPM unit, family and marital unit inside one household of
      its origin, and each ACS household one SPM unit (staging's partition;
      see :func:`_require_nested_groups`).

    The spine assembly record (P8) is
    :func:`~microcosm.build.us_runtime.support_provenance.spine_assembly_manifest`
    over every entity's clone-0 rows, declaring :data:`ASSEMBLED_CHANNELS`
    (``puf_tax_detail`` with a clone-0 count of 0), so
    :func:`~microcosm.build.us_runtime.support_provenance.validate_assembly_provenance`
    accepts the clone rows. The frame is checked against it before return.

    Args:
        cps_tables: The parent export's six entity tables (every CPS row:
            native rows and their clones).
        acs_tables: Staging's six entity tables for the selected ACS
            households. A table may have no rows.
        cps_household_weights: PR 1's CPS starting weights keyed by
            ``household_id``: a :class:`pandas.Series` indexed by it, or a
            mapping, one entry per row of ``cps_tables["household"]``, in
            any order.
        acs_household_weights: PR 1's ACS starting weights keyed by each
            household's
            :func:`~microcosm.build.us_runtime.one_population_selection.acs_unit_key`
            of its ``SERIALNO`` (``pd.Series(starting.acs,
            index=plan.acs_keys)``), one entry per row of
            ``acs_tables["household"]``, in any order; empty when no ACS
            household is selected. A weight given by position is refused:
            ``StartingWeights.acs`` comes in selection order and staging's
            households in id order, and no later clause would see the
            permutation (d2283 lead call L3's check).
        staging_lineage: Staging's lineage for the persons of the selected
            ACS households, read from staging's output, not from
            ``acs_tables``: each person's native (household id, person id),
            that is staging's ``household_source_id`` of the person's
            household and the person's ``person_source_id``, mapped to its
            (``SERIALNO``, ``SPORDER``). Ids and ``SPORDER`` may be integral
            floats (staging's pooled form). Empty when no ACS household is
            selected. A set of pairs is refused (TypeError): it cannot tie a
            ``SERIALNO`` to its household.

    Returns:
        A :class:`~microcosm.frame.Frame` of the CPS rows then the ACS rows,
        each group table sorted by its id. Household weights are the two
        inputs, each household's by its key, carried through that sort, as
        :attr:`~microcosm.frame.WeightKind.IMPORTANCE` weights (a pool
        assembly's starting weights, the kind staging gives its pooled
        weights). ``metadata`` holds only the spine assembly record, under
        ``SPINE_ASSEMBLY_MANIFEST_KEY``; the mass log is empty. The H5
        writer does not keep the record, so PR 4 writes it into the
        population manifest and PR 5's loader restores it.

    Raises:
        OnePopulationContractError: Naming the first clause the inputs break,
            or if the joined tables fail the frame's invariants or the
            assembly validator.
        TypeError: If the tables are not a mapping of DataFrames, or
            ``staging_lineage`` is not a mapping.
    """

    cps = _origin_tables(cps_tables, origin=_CPS)
    acs = _origin_tables(acs_tables, origin=_ACS)
    _require_cps_support_metadata(cps)
    _require_acs_support_metadata(acs)
    cps_keyed = _keyed_weights(cps_household_weights, origin=_CPS)
    acs_keyed = _keyed_weights(acs_household_weights, origin=_ACS)
    staging = _lineage_rows(staging_lineage)

    origins = {_CPS: _assembled_cps(cps), _ACS: _assembled_acs(acs)}
    _require_spine_tags(origins)
    _require_disjoint_ids(origins)
    _require_unique_source_ranks(origins)
    _require_acs_lineage(origins[_ACS], staging)
    # After P7, which tied each ACS household's SERIALNO to staging's by
    # native id, so a weight keyed by SERIALNO lands on its own household.
    cps_weights = _aligned_weights(
        cps_keyed,
        cps["household"][US_SCHEMA.id_column("household")].tolist(),
        origin=_CPS,
    )
    acs_weights = _aligned_weights(
        acs_keyed, _acs_weight_keys(acs["household"]), origin=_ACS
    )

    tables, household_order = _joined_tables(origins)
    record = spine_assembly_manifest(
        {
            entity: table.loc[
                pd.to_numeric(table[support_clone_index_column(entity)]).eq(0)
            ]
            for entity, table in tables.items()
        },
        channels=ASSEMBLED_CHANNELS,
    )
    weights = np.concatenate([cps_weights, acs_weights])[household_order]
    try:
        frame = Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(weights, WeightKind.IMPORTANCE)},
            metadata=record,
        )
        validate_assembly_provenance(frame, boundary=_BOUNDARY)
    except ValueError as exc:
        raise OnePopulationContractError(
            f"The joined one-population frame is not valid: {exc}"
        ) from exc
    _require_nested_groups(origins)
    return frame


def _declared_acs_persons(record: object) -> int | None:
    """The ACS person count a spine assembly record declares.

    None when ``record`` is None or does not list
    :data:`ACS_SUPPORT_CHANNEL`. The count rule is R1's
    (:func:`_require_assembly_record`): a nonnegative integer, so 24.5,
    ``"24"`` and ``True`` refuse rather than read as counts.
    """

    if record is None:
        return None
    channels = record.get("channels") if isinstance(record, Mapping) else None
    counts = record.get("native_row_counts") if isinstance(record, Mapping) else None
    if (
        not isinstance(record, Mapping)
        or record.get("version") != 1
        or isinstance(channels, str | bytes)
        or not isinstance(channels, Collection)
        or not isinstance(counts, Mapping)
    ):
        raise OnePopulationContractError(
            f"Malformed spine assembly record: {SPINE_ASSEMBLY_MANIFEST_KEY!r} "
            "must hold version 1, a channel list and native row counts."
        )
    if ACS_SUPPORT_CHANNEL not in channels:
        return None
    by_channel = counts.get(US_SCHEMA.person_entity)
    count = (
        by_channel.get(ACS_SUPPORT_CHANNEL) if isinstance(by_channel, Mapping) else None
    )
    if not (_is_count(count) and count >= 0):
        raise OnePopulationContractError(
            "Malformed spine assembly record: its person count for "
            f"{ACS_SUPPORT_CHANNEL!r} must be a nonnegative integer, got {count!r}."
        )
    return count


def one_population_declares_acs_rows(frame: Frame) -> bool:
    """Whether ``frame`` declares ACS rows in the one-population form (d2283).

    A d2283 child holds ACS rows beside the CPS rows. This says whether a
    frame declares such rows, so a release gate can tell a child from a
    Route A frame. It answers from two things the frame carries and requires
    that they agree:

    - declared: the spine assembly record under ``SPINE_ASSEMBLY_MANIFEST_KEY``
      lists :data:`ACS_SUPPORT_CHANNEL` with a positive person count. No
      record, a record without that channel, or a count of 0 declares none;
    - live: some person row's support channel holds
      :data:`ACS_SUPPORT_CHANNEL`. A person table without the column holds
      none.

    So it returns False on a Route A frame (no record and no ACS row), on an
    assembled frame with no ACS row (its record lists the channel with a
    count of 0) and on a stacked pool (its ACS rows carry the channel
    ``acs``), and True on an assembled frame with ACS rows. Nothing calls it
    yet.

    It reads the person channel column alone, and only if the person table
    has it. It calls no role reader, no support provenance check and not
    :func:`~microcosm.build.us_runtime.support_provenance.validate_assembly_provenance`,
    which requires the record's clone-0 counts to equal the live rows and
    would refuse a Route A frame, which has no record. The release tool's
    frozen-support selection prunes the base frame before its ESI premium
    gate reads it, so the live rows can fall below the record's counts. This
    module's plan for the later PRs (module docstring) vouches for the record
    on a loaded frame: PR 4 writes the record into the population manifest,
    and PR 5's loader restores it into ``Frame.metadata`` and checks it on
    the loaded frame through that validator, counts and all.

    Args:
        frame: The frame to read.

    Returns:
        True when the record declares ACS persons and the frame holds some;
        False when it does neither.

    Raises:
        OnePopulationContractError: If the record lacks version 1, a
            channel list or a count mapping, or gives an ACS person count
            other than a nonnegative integer, if person rows carry
            :data:`ACS_SUPPORT_CHANNEL` the record does not declare, or if the
            record declares ACS persons and no person row carries the channel.
        TypeError: If ``frame`` is not a :class:`~microcosm.frame.Frame`.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    record = frame.metadata.get(SPINE_ASSEMBLY_MANIFEST_KEY)
    count = _declared_acs_persons(record)
    declared = count is not None and count > 0
    person = frame.table(US_SCHEMA.person_entity)
    column = support_channel_column(US_SCHEMA.person_entity)
    live = int(person[column].eq(ACS_SUPPORT_CHANNEL).sum()) if column in person else 0
    if live and not declared:
        if record is None:
            state = "the frame has no spine assembly record"
        elif count is None:
            state = "the frame's spine assembly record does not list that channel"
        else:
            state = "the frame's spine assembly record counts no ACS person"
        raise OnePopulationContractError(
            f"Undeclared ACS rows: {live} person row(s) carry the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}, and {state}."
        )
    if declared and not live:
        raise OnePopulationContractError(
            f"Missing ACS rows: the frame's spine assembly record declares {count} "
            "ACS person(s), and no person row carries the support channel "
            f"{ACS_SUPPORT_CHANNEL!r}."
        )
    return declared


# ------------------------------------------------------- engine variables


@runtime_checkable
class EngineVariableIndex(Protocol):
    """What the contract asks of the engine's variable metadata, at 2024.

    The answers must cohere: the inputs at 2024 are exactly the known
    columns that are neither computed at 2024 nor structural, so no input is
    left out of R6's null check. The contract refuses an index whose answers
    do not.
    """

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        """The names among ``columns`` the engine defines, formulas included."""
        ...

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """The tables' columns the engine computes at 2024."""
        ...

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """Every known, non-structural column of the tables not computed at 2024."""
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
    # The other direction: R6 reads only the inputs, and R10 skips every
    # known column, so an input the index leaves out would be checked by
    # neither (PR #1685 review, should-fix 4).
    omitted = known - computed - STRUCTURAL_COLUMNS - inputs
    if omitted:
        problems.append(
            "inputs omit known columns that are neither computed at 2024 nor "
            f"structural: {sorted(omitted)[:5]}"
        )
    if problems:
        raise OnePopulationContractError(
            "The engine variable index answers do not cohere (the inputs must be "
            "exactly the known columns not computed at 2024 and not structural): "
            + "; ".join(problems)
            + "."
        )
    return _EngineColumns(known=known, computed=computed, inputs=inputs)


# ----------------------------------------------------------- the contract


def _frame_tables(frame: Frame) -> dict[str, pd.DataFrame]:
    return {entity: frame.table(entity) for entity in US_SCHEMA.entities}


def _origin_of_channel(channel: object) -> str | None:
    if channel in _CPS_CHANNELS:
        return _CPS
    if channel == ACS_SUPPORT_CHANNEL:
        return _ACS
    return None


def _require_assembled_columns(tables: Mapping[str, pd.DataFrame]) -> None:
    """R1, first part: every entity assembled, provenance complete, channels known."""

    rule = "Incomplete assembled provenance"
    assembled = [
        entity
        for entity in US_SCHEMA.entities
        if spine_source_id_column(entity) in tables[entity]
    ]
    if not assembled:
        raise OnePopulationContractError(
            f"{rule}: the frame is not assembled; no entity carries its "
            "'<entity>_spine_source_id' (build it with to_assembled_provenance)."
        )
    if len(assembled) != len(US_SCHEMA.entities):
        unassembled = [e for e in US_SCHEMA.entities if e not in assembled]
        raise OnePopulationContractError(
            "Mixed assembled and unassembled provenance: "
            f"{unassembled} lack their spine source id while {assembled} carry "
            "it; every entity of the frame must be assembled."
        )
    for entity in US_SCHEMA.entities:
        table = tables[entity]
        try:
            require_assembled_support_provenance(table, entity=entity)
        except ValueError as exc:
            raise OnePopulationContractError(f"{rule}: {exc}") from exc
        _, channel, clone, spine_source, _ = _provenance_columns(entity)
        for column in (channel, clone, spine_source):
            nulls = int(table[column].isna().sum())
            if nulls:
                raise OnePopulationContractError(
                    f"{rule}: {column!r} has {nulls} null value(s)."
                )
        unknown = _unexpected(table[channel], ASSEMBLED_CHANNELS)
        if unknown:
            raise OnePopulationContractError(
                f"{rule}: rows must carry one of the assembled channels "
                f"{list(ASSEMBLED_CHANNELS)}; {channel!r} holds {unknown[:5]}."
            )


def _split_origins(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Each entity's CPS rows (``asec``, ``puf_tax_detail``) and ACS rows."""

    origins: dict[str, dict[str, pd.DataFrame]] = {_CPS: {}, _ACS: {}}
    for entity in US_SCHEMA.entities:
        table = tables[entity]
        origin = table[support_channel_column(entity)].map(_origin_of_channel)
        for name in (_CPS, _ACS):
            origins[name][entity] = table.loc[origin.eq(name).to_numpy()]
    return origins


def _require_assembled_cps_values(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on the frame's CPS rows: Route A's labels, spine id = source id."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        source, channel, clone, spine_source, spine = _provenance_columns(entity)
        sources = _integral_ids(
            table[source], refusal=f"CPS {source!r} must hold source ids"
        )
        clones = _integral_ids(
            table[clone], refusal=f"CPS {clone!r} must hold clone indices"
        )
        spine_sources = _integral_ids(
            table[spine_source], refusal=f"CPS {spine_source!r} must hold source ids"
        )
        bad_clones = sorted(set(clones.tolist()) - set(_CPS_CLONE_INDICES))
        if bad_clones:
            raise OnePopulationContractError(
                "CPS clone indices must be 0, 1 or 2 (Route A's native rows, PUF "
                f"tax-detail copies and capital-gains tail copies); {clone!r} "
                f"holds {bad_clones[:5]}."
            )
        native = table[channel].eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
        disagree = int(np.count_nonzero(native != (clones == 0)))
        if disagree:
            raise OnePopulationContractError(
                f"CPS rows must carry channel {BASE_ASEC_SUPPORT_CHANNEL!r} exactly "
                f"on clone index 0; {disagree} {entity!r} row(s) disagree."
            )
        differs = int(np.count_nonzero(spine_sources != sources))
        if differs:
            raise OnePopulationContractError(
                f"CPS rows' spine source id must equal their source id (the "
                f"parent export's rows are their own spine records); {differs} "
                f"{entity!r} row(s) differ in {spine_source!r}."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(CPS_SPINE_TAG)
            if wrong.any():
                raise OnePopulationContractError(
                    f"CPS rows carry a spine tag other than {CPS_SPINE_TAG!r} in "
                    f"{spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}; "
                    "staging tags the donor's rows "
                    f"{CPS_SPINE_TAG!r}."
                )


def _require_assembled_acs_values(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on the frame's ACS rows: staging's labels, source id = pooled id."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        source, _, clone, spine_source, spine = _provenance_columns(entity)
        id_column = US_SCHEMA.entity_id_column(entity)
        ids = _integral_ids(
            table[id_column], refusal=f"ACS {id_column!r} must hold ids"
        )
        sources = _integral_ids(
            table[source], refusal=f"ACS {source!r} must hold source ids"
        )
        clones = _integral_ids(
            table[clone], refusal=f"ACS {clone!r} must hold clone indices"
        )
        natives = _integral_ids(
            table[spine_source], refusal=f"ACS {spine_source!r} must hold source ids"
        )
        bad_clones = sorted(set(clones.tolist()) - {0})
        if bad_clones:
            raise OnePopulationContractError(
                f"ACS rows must carry clone index 0; {clone!r} holds {bad_clones[:5]}."
            )
        differs = int(np.count_nonzero(sources != ids))
        if differs:
            raise OnePopulationContractError(
                f"ACS rows' source id must equal their pooled id "
                f"({id_column!r}), as assembly writes it; {differs} {entity!r} "
                "row(s) differ."
            )
        offsets = sorted(set((ids - natives).tolist()))
        if len(offsets) > 1:
            raise OnePopulationContractError(
                f"ACS {entity!r} ids must sit one constant offset from staging's "
                f"native source ids (staging shifts each entity's ids by one "
                f"amount); found offsets {offsets[:5]}."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(ACS_SUPPORT_CHANNEL)
            if wrong.any():
                raise OnePopulationContractError(
                    f"ACS rows carry a spine tag other than {ACS_SUPPORT_CHANNEL!r} "
                    f"in {spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}."
                )


def _require_assembly_record(frame: Frame) -> None:
    """R1, last part: the clone-0 record, three channels, validated."""

    record = frame.metadata.get(SPINE_ASSEMBLY_MANIFEST_KEY)
    channels = record.get("channels") if isinstance(record, Mapping) else None
    if record is not None and (
        isinstance(channels, str | bytes)
        or not isinstance(channels, Iterable)
        or tuple(channels) != ASSEMBLED_CHANNELS
    ):
        raise OnePopulationContractError(
            f"Incomplete assembled provenance: the spine assembly record must "
            f"declare the channels {list(ASSEMBLED_CHANNELS)}, got {channels!r}."
        )
    # The validator reads each count through int(), so 24.5, "24" or 24.0
    # would pass for 24 live rows; the writer emits integers, so anything
    # else is refused here first (PR #1685 review, nit 1). A malformed
    # mapping is left to the validator's own message.
    counts = record.get("native_row_counts") if isinstance(record, Mapping) else None
    if isinstance(counts, Mapping):
        coerced = sorted(
            f"{entity}.{channel}={value!r}"
            for entity, by_channel in counts.items()
            if isinstance(by_channel, Mapping)
            for channel, value in by_channel.items()
            if not (_is_count(value) and value >= 0)
        )
        if coerced:
            raise OnePopulationContractError(
                "Incomplete assembled provenance: the spine assembly record's "
                "native row counts must be nonnegative integers; "
                f"{coerced[:5]} are not."
            )
    try:
        validate_assembly_provenance(frame, boundary=_BOUNDARY)
    except ValueError as exc:
        raise OnePopulationContractError(
            f"Incomplete assembled provenance: {exc}"
        ) from exc


def _require_provenance(
    frame: Frame,
    tables: Mapping[str, pd.DataFrame],
    staging: Mapping[tuple[int, int], tuple[str, int]] | None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """R1: the assembled form, complete; P5's value rules, Q3, P6b, P6, P7, P6c.

    Returns each origin's rows.
    """

    _require_assembled_columns(tables)
    origins = _split_origins(tables)
    _require_assembled_cps_values(origins[_CPS])
    _require_assembled_acs_values(origins[_ACS])
    _require_spine_tags(origins)
    _require_disjoint_ids(origins)
    try:
        _require_unique_source_ranks(origins)
    except OnePopulationContractError:
        raise
    except ValueError as exc:
        raise OnePopulationContractError(
            f"Incomplete assembled provenance: {exc}"
        ) from exc
    acs = origins[_ACS]
    if len(acs["household"]) or len(acs[US_SCHEMA.person_entity]):
        _require_acs_lineage(acs, staging)
    elif staging is not None:
        # No ACS row to read, but staging's lineage still has to match.
        _require_staging_lineage({}, staging)
    _require_assembly_record(frame)
    # After the validator, so a cross-grain move reports the validator's
    # reason first; nesting reads memberships P6b resolved.
    _require_nested_groups(origins)
    return origins


def _require_one_location(tables: Mapping[str, pd.DataFrame]) -> None:
    """R11: no table carries a location-clone index column."""

    marked = sorted(
        f"{entity}.{column}"
        for entity, table in tables.items()
        for column in table.columns
        if isinstance(column, str) and column.endswith(_LOCATION_CLONE_INDEX_SUFFIX)
    )
    if marked:
        raise OnePopulationContractError(
            f"A household with more than one location: {marked[:5]} mark location "
            "clones; the contract allows one location per household, so no table "
            f"may carry a '*{_LOCATION_CLONE_INDEX_SUFFIX}' column."
        )


def _require_selected_keys(
    origins: Mapping[str, Mapping[str, pd.DataFrame]],
    manifest: SelectionManifest,
) -> None:
    """R2: one native row per CPS key, one row per ACS household, as selected."""

    acs_households = origins[_ACS]["household"]
    try:
        acs_keys = (
            [acs_unit_key(serial) for serial in acs_households[_SERIALNO]]
            if len(acs_households)
            else []
        )
    except (KeyError, ValueError) as exc:
        raise OnePopulationContractError(
            f"ACS households need a SERIALNO key: {exc}"
        ) from exc
    counts = pd.Series(acs_keys, dtype=object).value_counts()
    repeated = sorted(counts.index[counts.gt(1)].tolist())
    if repeated:
        raise OnePopulationContractError(
            f"An ACS household with more than one row: {repeated[:5]} repeat."
        )
    if (len(acs_keys), keys_sha256(acs_keys)) != (
        manifest.n_acs,
        manifest.selected_acs_keys_sha256,
    ):
        raise OnePopulationContractError(
            "ACS keys do not match the selection manifest: the frame holds "
            f"{len(acs_keys)} ACS household key(s) with digest "
            f"{keys_sha256(acs_keys)}; the manifest selected {manifest.n_acs} with "
            f"digest {manifest.selected_acs_keys_sha256}."
        )

    cps = origins[_CPS]
    if len(cps[US_SCHEMA.person_entity]):
        try:
            units = cps_unit_ids(cps[US_SCHEMA.person_entity])
        except OnePopulationContractError as exc:
            raise OnePopulationContractError(
                f"A CPS key without exactly one native row: {exc}"
            ) from exc
    else:
        units = pd.DataFrame(
            {"income_year": [], "unit_id": [], "key": []},
            index=pd.Index([], dtype=np.int64, name="household_id"),
        )
    # Ids and clone indices through the integral parser R1 applies, so a
    # representation R1 accepts (digit strings, integral floats; lead call
    # L5) joins cps_unit_ids' int64 household index (PR #1685 review,
    # should-fix 3).
    household = cps["household"]
    household_ids = _integral_ids(
        household[US_SCHEMA.id_column("household")],
        refusal="CPS 'household_id' must hold ids",
    )
    clone_column = support_clone_index_column("household")
    clones = _integral_ids(
        household[clone_column],
        refusal=f"CPS {clone_column!r} must hold clone indices",
    )
    clone = pd.Series(clones, index=pd.Index(household_ids)).reindex(units.index)
    native = units.loc[clone.eq(0).to_numpy()]
    counts = native["key"].value_counts()
    repeated = sorted(counts.index[counts.gt(1)].tolist())
    if repeated:
        raise OnePopulationContractError(
            "A CPS key without exactly one native row: "
            f"{len(repeated)} key(s) have more than one clone-0 household, such "
            f"as {repeated[:3]}."
        )
    orphaned = sorted(set(units["key"]) - set(native["key"]))
    if orphaned:
        raise OnePopulationContractError(
            "A CPS key without exactly one native row: "
            f"{len(orphaned)} clone household key(s) have no clone-0 household, "
            f"such as {orphaned[:3]}."
        )
    by_year = tuple(
        (int(year), int(n))
        for year, n in native["income_year"].value_counts().sort_index().items()
    )
    observed = (len(native), by_year, keys_sha256(native["key"]))
    expected = (
        manifest.n_cps,
        manifest.n_cps_by_income_year,
        manifest.cps_keys_sha256,
    )
    if observed != expected:
        raise OnePopulationContractError(
            "CPS keys do not match the selection manifest: the frame's clone-0 "
            f"households give (count, by income year, digest) {observed}; the "
            f"manifest selected {expected}."
        )


def _require_congressional_districts(household: pd.DataFrame) -> None:
    """R12: a positive, integral CD with its state's prefix on every household."""

    rule = "A household without a congressional district"
    cd_column = "congressional_district_geoid"
    missing = [c for c in (cd_column, "state_fips") if c not in household]
    if missing:
        raise OnePopulationContractError(
            f"{rule}: the household table lacks {missing}."
        )
    cd = _integral_ids(
        household[cd_column],
        refusal=f"{rule}: {cd_column!r} must be an integral geoid on every household",
    )
    states = _integral_ids(
        household["state_fips"],
        refusal=f"{rule}: 'state_fips' must be integral on every household",
    )
    if (cd <= 0).any():
        raise OnePopulationContractError(
            f"{rule}: {int((cd <= 0).sum())} household(s) hold a non-positive "
            f"{cd_column!r}."
        )
    wrong = (cd // 100) != states
    if wrong.any():
        raise OnePopulationContractError(
            f"{rule}: {int(wrong.sum())} household(s) hold a {cd_column!r} whose "
            "state prefix (geoid // 100) differs from state_fips, such as "
            f"{cd[wrong][:3].tolist()}."
        )


def _require_acs_household_type(acs_household: pd.DataFrame) -> None:
    """T1: ``TYPEHUGQ`` 1, 2 or 3 on every ACS household.

    TODO(#1610): once #1610 lands, add its wider rules: every CPS household
    carries a valid ``HRHTYPE`` once any does, no household carries both
    codes, and every person carries valid ``A_EXPRRP``, ``A_MARITL``, ``age``
    and ``is_female``.
    """

    if not len(acs_household):
        return
    rule = (
        f"{_TYPEHUGQ} must be 1, 2 or 3 on every ACS household (#1610's "
        "household materializer reads it to leave group quarters out)"
    )
    if _TYPEHUGQ not in acs_household:
        raise OnePopulationContractError(f"{rule}; the household table lacks it.")
    values = acs_household[_TYPEHUGQ]
    bad = ~values.map(
        lambda value: _is_real_number(value) and value in _ACS_HOUSEHOLD_TYPES
    ).astype(bool)
    if bad.any():
        raise OnePopulationContractError(
            f"{rule}; {int(bad.sum())} ACS household(s) hold "
            f"{values[bad].head(5).tolist()!r}."
        )


def _require_esi_premium(frame: Frame) -> None:
    """R4: the ESI employer premium present on persons and never null."""

    if US_ESI_EMPLOYER_PREMIUM_COLUMN not in frame.table(US_SCHEMA.person_entity):
        raise OnePopulationContractError(
            f"A null ESI premium: the person table lacks "
            f"{US_ESI_EMPLOYER_PREMIUM_COLUMN!r}, so every person's premium is "
            "missing (the release's ESI signal gate fails without it)."
        )
    try:
        refuse_unassigned_us_esi_premiums(frame, consumer=_CONSUMER)
    except SourceRuntimeError as exc:
        raise OnePopulationContractError(f"A null ESI premium: {exc}") from exc


def _require_classified_adults(frame: Frame) -> None:
    """R5: every SPM unit has a classified adult (FAIL and errors refuse)."""

    rule = "An SPM unit has no classified adult"
    try:
        result = check_spm_composition(frame)
    except ValueError as exc:
        raise OnePopulationContractError(
            f"{rule}: the composition check cannot run: {exc}"
        ) from exc
    if result.status != "PASS":
        failures = "; ".join(str(failure) for failure in result.failures)
        raise OnePopulationContractError(f"{rule}: {result.summary}. {failures}")


def _null_counts(
    tables: Mapping[str, pd.DataFrame], columns: Collection[str]
) -> dict[str, int]:
    return {
        f"{entity}.{column}": int(table[column].isna().sum())
        for entity, table in tables.items()
        for column in sorted(set(columns).intersection(table.columns))
        if table[column].isna().any()
    }


def _refused_nulls(values: pd.Series) -> tuple[int, list[object]]:
    """The nulls R6 refuses in a release-written column: ``pd.NA`` and the rest.

    Returns how many values are ``pd.NA`` and the other refused nulls. Only
    None and a float NaN (a Python float or a numpy floating value) pass, so
    a numpy float column, whose only null is NaN, passes by its dtype before
    any value is read. A nullable ``Float32``/``Float64`` column's masked
    null is ``pd.NA``; a NaN it holds unmasked is not a null to pandas.
    """

    if isinstance(values.dtype, np.dtype) and values.dtype.kind == "f":
        return 0, []
    pd_na, others = 0, []
    for value in values[values.isna()].tolist():
        if value is None or (
            isinstance(value, float | np.floating) and math.isnan(value)
        ):
            continue
        if value is pd.NA:
            pd_na += 1
        else:
            others.append(value)
    return pd_na, others


def _require_filled_inputs(
    tables: Mapping[str, pd.DataFrame], inputs: frozenset[str]
) -> None:
    """R6: no null in an engine input at 2024, outside the release-written list.

    A release-written column may be null only as None or a float NaN (a
    Python float or a numpy floating value, so any null of a numpy float
    column); every other value pandas counts as null refuses, on any row of
    either origin. ``pd.NA`` (a pandas nullable dtype's null) refuses with
    its own message: policyengine-core 3.32.5 loads a boolean None as False
    and a NaN as True but cannot cast a ``pd.NA``, and the release's first
    engine pass, the SSI December pass, loads twelve of these columns before
    their producers write them (PR #1685 review, blocking 1). ``pd.NaT``,
    numpy's ``NaT`` and any other null refuse too, fail closed: nothing here
    traces how the release loads them (review r2, nit 1). The rule covers
    all of :data:`RELEASE_WRITTEN_INPUTS` present: the other six columns'
    producers (``scf_wealth``, ``ssi_disability_criteria``,
    ``sipp_head_start``) run before that pass and read each value through
    ``pd.to_numeric(...).to_numpy(dtype=np.float64)``, so a ``pd.NA``
    there is probably rewritten like a NaN, and the refusal is stricter
    than those stages need.
    """

    gaps = _null_counts(tables, inputs - set(RELEASE_WRITTEN_INPUTS))
    if gaps:
        raise OnePopulationContractError(
            f"A null in an engine input at 2024: {gaps} (null rows per column). "
            "Prepare must fill every engine input on every row; only the "
            "always-on producers' outputs in RELEASE_WRITTEN_INPUTS may be left "
            "null."
        )
    refused = {
        f"{entity}.{column}": _refused_nulls(table[column])
        for entity, table in tables.items()
        for column in sorted(set(RELEASE_WRITTEN_INPUTS).intersection(table.columns))
    }
    pd_na = {name: rows for name, (rows, _) in refused.items() if rows}
    if pd_na:
        raise OnePopulationContractError(
            f"A pd.NA in a release-written input: {pd_na} (pd.NA rows per column). "
            "A column in RELEASE_WRITTEN_INPUTS may be null only as None or NaN; "
            "the release's first engine pass loads twelve of them before their "
            "producers write them, and policyengine-core cannot cast a pd.NA."
        )
    others = {name: values for name, (_, values) in refused.items() if values}
    if others:
        counts = {name: len(values) for name, values in others.items()}
        examples = [values[0] for values in others.values()][:3]
        raise OnePopulationContractError(
            f"A null other than None or NaN in a release-written input: {counts} "
            f"(rows per column), such as {examples!r}. A column in "
            "RELEASE_WRITTEN_INPUTS may be null only as None or a float NaN; "
            "pd.NaT, numpy's NaT and any other null refuse."
        )


def _require_tail_columns(tax_unit: pd.DataFrame) -> None:
    """R7: the six PUF tail provenance columns on every tax unit."""

    rule = "A null PUF tail column on a tax unit"
    missing = [c for c in PUF_TAIL_TAX_UNIT_COLUMNS if c not in tax_unit]
    if missing:
        raise OnePopulationContractError(f"{rule}: the tax-unit table lacks {missing}.")
    gaps = _null_counts({"tax_unit": tax_unit}, PUF_TAIL_TAX_UNIT_COLUMNS)
    if gaps:
        raise OnePopulationContractError(f"{rule}: {gaps} (null rows per column).")


def _tail_donor_set(parent_tail_donor_ids: object) -> frozenset[int]:
    if isinstance(parent_tail_donor_ids, str | bytes | Mapping) or not isinstance(
        parent_tail_donor_ids, Collection
    ):
        raise TypeError(
            "parent_tail_donor_ids must be a set of int donor ids, got "
            f"{type(parent_tail_donor_ids).__name__}."
        )
    # Through the shared predicate, which refuses a numpy timedelta64 that
    # numbers.Integral accepts (PR #1685 review r3, should-fix 2).
    if not all(_is_integer(value) for value in parent_tail_donor_ids):
        raise TypeError("parent_tail_donor_ids must hold int donor ids only.")
    return frozenset(int(value) for value in parent_tail_donor_ids)


def _require_parent_tail_donors(tax_unit: pd.DataFrame, parent: frozenset[int]) -> None:
    """R8: the applied tail donor ids equal the parent's, each applied once,
    and no ACS tax unit applies one."""

    rule = "The tail donor set differs from the parent's"
    applied_values = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN]
    is_flag = applied_values.map(
        lambda value: (
            isinstance(value, bool | np.bool_)
            or (_is_real_number(value) and value in (0, 1))
        )
    ).astype(bool)
    if not is_flag.all():
        raise OnePopulationContractError(
            f"{rule}: {PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN!r} holds non-flag "
            f"values {applied_values[~is_flag].head(3).tolist()!r}."
        )
    applied = applied_values.astype(bool).to_numpy()
    acs_applied = applied & (
        tax_unit[support_channel_column("tax_unit")].eq(ACS_SUPPORT_CHANNEL).to_numpy()
    )
    if acs_applied.any():
        raise OnePopulationContractError(
            f"{rule}: {int(acs_applied.sum())} ACS tax unit(s) apply a tail "
            "donor; the parent's tail draws apply only to CPS recipients, so "
            f"every ACS tax unit holds {PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN!r} "
            "False (d2283 lead call L4)."
        )
    donors = _integral_ids(
        tax_unit.loc[applied, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN],
        refusal=f"{rule}: applied donor ids must be integers",
    )
    repeated = pd.Series(donors).duplicated(keep=False).to_numpy()
    if repeated.any():
        raise OnePopulationContractError(
            f"{rule}: donor id(s) {sorted(set(donors[repeated].tolist()))[:5]} are "
            "applied on more than one tax unit; each parent donor applies to one "
            "tax unit (the one-to-one rule "
            "assert_puf_capital_gains_tail_survives_selection keeps)."
        )
    found = frozenset(int(value) for value in donors.tolist())
    if found != parent:
        missing = sorted(parent - found)
        extra = sorted(found - parent)
        raise OnePopulationContractError(
            f"{rule}: {len(missing)} parent donor(s) missing (such as "
            f"{missing[:3]}) and {len(extra)} donor(s) the parent never applied "
            f"(such as {extra[:3]}). The comparison is of donor ids only, never "
            "weights."
        )


def _declarations(declared_absences: object, *, name: str) -> Mapping[str, str]:
    """The owner's declarations, checked; ``name`` is the caller's argument."""

    if declared_absences is None:
        return {}
    if not isinstance(declared_absences, Mapping):
        raise TypeError(
            f"{name} must map each column to its owner's decision, got "
            f"{type(declared_absences).__name__}."
        )
    for column, decision in declared_absences.items():
        if not isinstance(column, str) or not isinstance(decision, str):
            raise TypeError(
                f"{name} must map str columns to str decisions, got "
                f"{column!r}: {decision!r}."
            )
        if not decision.strip():
            raise OnePopulationContractError(
                f"A declared absence needs its owner's decision; {column!r} has none."
            )
    return declared_absences


def _origin_gap(
    null: np.ndarray, rows: np.ndarray, other_rows: np.ndarray
) -> np.ndarray | None:
    """The rows of one origin a column lacks, when they make a gap.

    The one gap predicate R10 and :func:`absences` share (d2283 lead call
    L2): a column is a gap of an origin when it is null on some row of that
    origin (``rows``) while some row of the other origin (``other_rows``)
    holds a value. Returns the origin's null rows, or ``None`` when there is
    no gap: no null on the origin's rows, or no value on the other origin's.
    """

    lacking = null & rows
    if not lacking.any() or not (~null & other_rows).any():
        return None
    return lacking


def _origin_rows(channels: Mapping[str, np.ndarray], origin: str) -> np.ndarray:
    """The rows of ``origin``, from one mask per assembled channel."""

    return np.logical_or.reduce([channels[name] for name in _ORIGIN_CHANNELS[origin]])


def _require_declared_absences(
    tables: Mapping[str, pd.DataFrame],
    engine: _EngineColumns,
    register: Mapping[str, str],
    declared: Mapping[str, str],
) -> None:
    """R10: a register column ACS rows lack while CPS rows hold it is declared.

    Engine inputs at 2024 are R6's (it refuses any null), so this covers
    model-named columns no engine variable names: register entries. It uses
    :func:`absences`' gap predicate (:func:`_origin_gap`): null on some ACS
    row while some CPS row holds a value. So one declaration answers both
    checks, and a column partly null on CPS rows still needs one. A column
    null on every row of both origins is no gap. What still differs:
    :func:`absences` also reads the reverse direction (CPS rows lacking a
    column ACS rows hold) and every model-named column, where this check
    reads register entries in the CPS-to-ACS direction only. A declaration
    this check does not use is ignored, as :func:`absences` records it.
    """

    from microcosm.data.stored_inputs import is_model_named

    gaps: dict[str, str] = {}
    for entity in US_SCHEMA.entities:
        table = tables[entity]
        channels = _channel_masks(
            table, entity, rule="A register column check needs every row's origin"
        )
        cps_rows, acs_rows = _origin_rows(channels, _CPS), _origin_rows(channels, _ACS)
        for column in table.columns:
            if (
                not isinstance(column, str)
                or not is_model_named(column)
                or column in engine.known
                or column not in register
                or column in RELEASE_WRITTEN_INPUTS
                or column in declared
            ):
                continue
            lacking = _origin_gap(table[column].isna().to_numpy(), acs_rows, cps_rows)
            if lacking is not None:
                gaps[f"{entity}.{column}"] = (
                    f"{int(lacking.sum())} of {int(acs_rows.sum())} ACS rows"
                )
    if gaps:
        raise OnePopulationContractError(
            "A model-named column held on some CPS row is null on ACS rows "
            f"without a declared absence: {gaps}. Fill it on ACS rows, or declare "
            "it in the absences file with its owner's decision."
        )


def require_one_population(
    frame: Frame,
    manifest: SelectionManifest,
    *,
    parent_tail_donor_ids: frozenset[int],
    declared_absences: Mapping[str, str] | None = None,
    staging_lineage: Mapping[tuple[int, int], tuple[str, int]] | None = None,
    variable_index: EngineVariableIndex | None = None,
    register: Mapping[str, str] | None = None,
) -> None:
    """Refuse a joined frame that breaks any clause of the contract.

    The clauses run in a fixed order, so a corrupted cell refuses for its own
    reason, and each message names its clause's rule. An input that breaks
    two clauses reports the earlier one. Two such inputs are expected: a
    real location clone
    (:func:`~microcosm.build.us_runtime.block_location.clone_frame_for_location`)
    remaps every pooled id and keeps the provenance ids, so it refuses at
    R1 (P5's ACS source id = pooled id) before R11 reads its
    ``*_location_clone_index`` columns, and R11 itself reports a marker on a
    frame R1 accepts; and a ``SERIALNO`` repeated across ACS households
    refuses at R1 (P7) before R2's repeated-key branch, which stays as a
    guard: no input reaches it through this function, since ``acs_unit_key``
    is one-to-one on the serial numbers P7 accepts (PR #1685 review, nit 2).

    - R1, incomplete assembled provenance: every entity assembled (none
      mixed), complete source id, channel, clone index and spine source id,
      only the three assembled channels; P5's value rules per origin (CPS:
      clone index 0, 1 or 2, ``asec`` exactly on clone 0, spine source id =
      source id, tag ``asec_puf``; ACS: clone index 0, source id = pooled id,
      one id offset from the native id, tag ``acs_2024_1yr``); a non-null
      spine tag on every row; P6b (ids and memberships within an origin); P6
      (no repeated (source id, rank)); P7 (the ACS rows' ``SERIALNO`` and
      ``SPORDER`` lineage, no null among its values, and, when
      ``staging_lineage`` is given, each person's pair equal to staging's
      under its native (household id, person id); on a frame with no ACS
      row it must then be empty); the clone-0 spine assembly record
      declaring the three channels, with nonnegative integer counts (the
      validator would truncate 24.5 to 24), checked by
      :func:`~microcosm.build.us_runtime.support_provenance.validate_assembly_provenance`;
      and P6c, after it (every group unit inside one household, and each
      ACS household one SPM unit).
    - R11, more than one location: no table carries a
      ``*_location_clone_index`` column. Location cloning is the only
      multi-location mechanism, and geography columns are household-level.
    - R2, one native row per CPS key and one row per ACS household: ACS
      households' :func:`~microcosm.build.us_runtime.one_population_selection.acs_unit_key`
      keys unique; each CPS key (:func:`cps_unit_ids`) on exactly one clone-0
      household, clones only beside it; both key sets equal to ``manifest``'s
      by count, by income year (CPS) and
      :func:`~microcosm.build.us_runtime.one_population_selection.keys_sha256`.
    - R12, a congressional district on every household: non-null, positive,
      integral ``congressional_district_geoid`` whose state prefix
      (geoid // 100) is the household's ``state_fips``, as staging checks
      donor districts. The release's CD-vintage preflight needs only one
      positive district (``_assert_cd_vintage_support_matches``, called at
      tools/build_us_fiscal_refresh_release.py:15502-15506, its check at
      :3822-3826), and its
      every-household rule (``_integer_geography_codes``) runs only when CD
      targets are materialized, which Route A's national surface drops. The
      clause keeps every household's district for a future CD surface and
      for the local lane, which reads the same H5.
    - T1: ``TYPEHUGQ`` 1, 2 or 3 on every ACS household (see
      :func:`_require_acs_household_type` for #1610's wider rules).
    - R3: no column the engine computes at 2024 (the release's export guard,
      ``computed_at(2024)``, not the lane's year-blind rule).
    - R9: no stored lower-case column that is neither an engine variable
      (inputs and formulas alike) nor a key of ``register``.
    - R4: the ESI employer premium on the person table, never null
      (:func:`~microcosm.build.us_runtime.esi_premiums.refuse_unassigned_us_esi_premiums`).
    - R5: a classified adult in every SPM unit
      (:func:`~microcosm.build.us_runtime.spm_composition.check_spm_composition`
      returning FAIL, or unable to run, refuses).
    - R6: no null in an engine input at 2024 (known, not structural, not
      computed at 2024, so a variable whose only formula starts after 2024
      counts) on any row, except :data:`RELEASE_WRITTEN_INPUTS`, which may
      be null only as None or a float NaN, never ``pd.NA``, ``pd.NaT``,
      numpy's ``NaT`` or any other null.
    - R7: the six :data:`PUF_TAIL_TAX_UNIT_COLUMNS` on every tax unit, never
      null.
    - R8: the ids of the tail donors applied on the frame's tax units equal
      ``parent_tail_donor_ids`` as a set, and no donor is applied on two tax
      units (the one-to-one rule of
      ``assert_puf_capital_gains_tail_survives_selection``). No weight is
      compared, so that function (whose weight check fires on the child's
      new starting weights) is not used. No ACS tax unit applies a donor
      (the parent's tail draws apply only to CPS recipients; d2283 lead
      call L4); which CPS tax units carry the applied flags is not checked.
    - R10: a model-named register column (no engine variable; R6 owns
      engine inputs) null on some ACS row while some CPS row of its entity
      holds a value, unless ``declared_absences`` names it: the gap
      predicate :func:`absences` uses (d2283 lead call L2), so one
      declaration answers both. The release-written list is the only
      exemption, never a stage's declared outputs whole. A declaration R10
      does not use is ignored, so the same mapping passes :func:`absences`,
      which records it as unused or needless. :func:`absences` also reads
      the reverse direction and every model-named column, so it refuses
      some frames R10 passes.

    Args:
        frame: The joined frame, built by :func:`to_assembled_provenance`.
        manifest: PR 1's selection manifest for the same selection.
        parent_tail_donor_ids: The parent export's applied capital-gains tail
            donor ids (PR 4 reads them from the export's tax-unit columns).
        declared_absences: Column to the owner's decision, for each register
            column ACS rows lack by decision: the mapping :func:`absences`
            takes as ``declared``.
        staging_lineage: Staging's lineage for the selected ACS persons, in
            :func:`to_assembled_provenance`'s keyed form (native household
            and person id to (``SERIALNO``, ``SPORDER``)); ``None`` checks
            the lineage's form only.
        variable_index: The engine's variable classes at 2024. ``None`` uses
            :class:`PolicyEngineUSVariableIndex`, built at call time.
        register: The stored non-variable register. ``None`` uses
            ``microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS``,
            imported at call time.

    Raises:
        OnePopulationContractError: Naming the first clause the frame breaks.
        TypeError: If an argument has the wrong type.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    if frame.schema != US_SCHEMA:
        raise OnePopulationContractError(
            "Incomplete assembled provenance: the frame must use the US schema."
        )
    if not isinstance(manifest, SelectionManifest):
        raise TypeError(
            f"manifest must be a SelectionManifest, got {type(manifest).__name__}."
        )
    parent_donors = _tail_donor_set(parent_tail_donor_ids)
    declared = _declarations(declared_absences, name="declared_absences")
    staging = None if staging_lineage is None else _lineage_rows(staging_lineage)
    if register is None:
        from microcosm.data.stored_inputs import US_STORED_NON_VARIABLE_COLUMNS

        register = US_STORED_NON_VARIABLE_COLUMNS

    tables = _frame_tables(frame)
    origins = _require_provenance(frame, tables, staging)  # R1 (P5-P7, P6b)
    _require_one_location(tables)  # R11
    _require_selected_keys(origins, manifest)  # R2
    _require_congressional_districts(tables["household"])  # R12
    _require_acs_household_type(origins[_ACS]["household"])  # T1

    engine = _engine_columns(tables, variable_index)
    if engine.computed:  # R3
        raise OnePopulationContractError(
            "A column the engine computes at 2024 is stored: "
            f"{sorted(engine.computed)}; the engine would overwrite it, so the "
            "export must not carry it."
        )
    from microcosm.data.stored_inputs import undefined_stored_inputs

    undefined = undefined_stored_inputs(  # R9
        _present_columns(tables),
        engine_variables=engine.known,
        register=register,
    )
    if undefined:
        raise OnePopulationContractError(
            "A stored lower-case column is neither an engine variable nor "
            f"registered: {list(undefined)}; the engine would ignore it."
        )
    _require_esi_premium(frame)  # R4
    _require_classified_adults(frame)  # R5
    _require_filled_inputs(tables, engine.inputs)  # R6
    _require_tail_columns(tables["tax_unit"])  # R7
    _require_parent_tail_donors(tables["tax_unit"], parent_donors)  # R8
    _require_declared_absences(tables, engine, register, declared)  # R10


# ------------------------------------------------------- the role record


def _plain(value: Any) -> Any:
    """``value`` as plain JSON builtins (frame metadata and the files hold these)."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _channel_masks(
    table: pd.DataFrame, entity: str, *, rule: str
) -> dict[str, np.ndarray]:
    """One row mask per assembled channel, or a refusal naming ``rule``."""

    column = support_channel_column(entity)
    if column not in table:
        raise OnePopulationContractError(
            f"{rule}: the {entity!r} table lacks {column!r}, so its rows have no "
            "origin."
        )
    channel = table[column]
    unknown = _unexpected(channel, ASSEMBLED_CHANNELS)
    if unknown or channel.isna().any():
        raise OnePopulationContractError(
            f"{rule}: {column!r} must hold one of {list(ASSEMBLED_CHANNELS)} on "
            f"every row; it holds {(unknown or ['null'])[:5]}."
        )
    return {name: channel.eq(name).to_numpy() for name in ASSEMBLED_CHANNELS}


def _is_count(value: object) -> bool:
    # The shared predicate refuses a boolean and a numpy timedelta64, which
    # numbers.Integral accepts (PR #1685 review r3, should-fix 2).
    return _is_integer(value)


def _role_counts(
    person: pd.DataFrame, spm_units: int, role: pd.Series
) -> dict[str, object]:
    """The role gate's counts and digest over ``person`` (its own expressions)."""

    return {
        "persons_joined": len(person),
        "table_rows": len(person),
        "native_spm_units": int(spm_units),
        "true_role_persons": int(role.dropna().sum()),
        _ROLE_GATE_DIGEST_KEY: _role_binding_sha256(person),
    }


def _require_cps_role_record(
    cps_record: Mapping[str, Any], recount: Mapping[str, object]
) -> None:
    """The CPS role record states the gate's counts and digest of the CPS rows."""

    rule = "The CPS role record does not match the frame's CPS rows"
    keys = (*_ROLE_GATE_COUNT_KEYS, _ROLE_GATE_DIGEST_KEY)
    missing = [key for key in keys if key not in cps_record]
    if missing:
        raise OnePopulationContractError(
            f"{rule}: it lacks the role gate's keys {missing} (PR 4 re-derives it "
            "with with_us_spm_independence_role on exactly those rows)."
        )
    malformed = [key for key in _ROLE_GATE_COUNT_KEYS if not _is_count(cps_record[key])]
    if not isinstance(cps_record[_ROLE_GATE_DIGEST_KEY], str):
        malformed.append(_ROLE_GATE_DIGEST_KEY)
    if malformed:
        raise OnePopulationContractError(
            f"{rule}: {malformed} must be integer counts and a hex digest."
        )
    differs = {
        key: {"record": _plain(cps_record[key]), "frame": recount[key]}
        for key in keys
        if cps_record[key] != recount[key]
    }
    if differs:
        raise OnePopulationContractError(
            f"{rule}: {differs}. The record must be derived on the CPS rows in "
            "the joined frame's order and dtypes (the digest hashes the ordered "
            "person id, SPM unit membership, age and role)."
        )


def _require_staging_receipt(
    staging_receipt: Mapping[str, Any], acs_counts: Mapping[str, int]
) -> None:
    """Staging's ACS role receipt: its keys, its rule, and counts that cover."""

    missing = sorted(_ACS_ROLE_RECEIPT_KEYS - set(staging_receipt))
    if missing:
        raise OnePopulationContractError(
            f"Staging's ACS role receipt lacks keys {missing} "
            "(acs_inputs.with_acs_spm_independence_role writes them)."
        )
    wrong = {
        key: _plain(staging_receipt[key])
        for key, expected in _ACS_ROLE_RECEIPT_IDENTITY.items()
        if _plain(staging_receipt[key]) != expected
    }
    if wrong:
        raise OnePopulationContractError(
            "Staging's ACS role receipt is not the ACS RELSHIPP role rule's: "
            f"{wrong} differ from {dict(_ACS_ROLE_RECEIPT_IDENTITY)}."
        )
    short = {}
    for receipt_key, frame_key in _ACS_ROLE_RECEIPT_COVERAGE:
        value = staging_receipt[receipt_key]
        if not _is_count(value) or value < acs_counts[frame_key]:
            short[receipt_key] = {
                "receipt": _plain(value),
                "selected": acs_counts[frame_key],
            }
    if short:
        raise OnePopulationContractError(
            "Staging's ACS role receipt covers fewer rows than the frame's ACS "
            f"rows: {short}. The receipt counts every staged ACS person, so each "
            "count must be an integer at least the selected one."
        )


def _require_acs_roles_by_rule(
    person: pd.DataFrame, acs_rows: np.ndarray, role: pd.Series
) -> None:
    """Each ACS person's stored role is staging's ``RELSHIPP`` rule's, row by row.

    Staging's receipt states counts over every staged person, which two
    persons trading roles keep, so the receipt alone cannot tie a role to a
    person. Re-deriving each selected person's role from the ``RELSHIPP``
    staging kept on the row can (PR #1685 review, should-fix 1).
    """

    if not acs_rows.any():
        return
    rule = "The ACS persons' SPM roles are not staging's RELSHIPP rule's"
    if _RELSHIPP not in person:
        raise OnePopulationContractError(
            f"{rule}: the person table lacks {_RELSHIPP!r}, which the rule reads "
            "(prepare keeps staging's raw column on ACS rows)."
        )
    relshipp = person.loc[acs_rows, _RELSHIPP]
    # Before the conversion below, which reads 20+1j as code 20 and a
    # timedelta64 of 20 units as 20 (PR #1685 review r2, should-fix 1), and
    # an Arrow dictionary of durations as their counts (review r3,
    # should-fix 1).
    found = _complex_or_temporal(relshipp)
    if found is not None:
        raise OnePopulationContractError(
            f"{rule}: {_RELSHIPP} on ACS rows holds {found}, not integer codes; "
            f"{_READ_AS_NUMBERS}"
        )
    codes = pd.to_numeric(relshipp, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    off_domain = ~np.isin(codes, sorted(_ACS_RELSHIPP_CODES))
    if off_domain.any():
        raise OnePopulationContractError(
            f"{rule}: {int(off_domain.sum())} ACS person(s) hold a blank or "
            f"off-domain {_RELSHIPP} (the rule reads the codes 20 to 38)."
        )
    derived = np.isin(codes, sorted(_ACS_ROLE_RELSHIPP))
    wrong = derived != role[acs_rows].to_numpy(dtype=bool)
    if wrong.any():
        raise OnePopulationContractError(
            f"{rule}: {int(wrong.sum())} of {int(acs_rows.sum())} ACS person(s) "
            f"hold a role their {_RELSHIPP} does not give ({_ACS_SPM_ROLE_RULE}). "
            "Staging's receipt states counts only, so it cannot vouch for which "
            "persons hold the role."
        )


def combined_spm_role_record(
    cps_record: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    frame: Frame,
) -> dict[str, object]:
    """The SPM independence role record for the joined frame (F-17, K-6; Q6).

    The parent's release keeps no role record, so each origin brings its own
    evidence and the record recounts the frame:

    - ``cps_record`` is a record over exactly the frame's CPS rows (channels
      ``asec`` and ``puf_tax_detail``), which the prepare step derives with
      ``with_us_spm_independence_role`` from the raw ASEC inputs. Its counts
      and ``person_role_binding_sha256`` must equal a recount of those rows
      by the gate's rule (the digest is the gate's own
      ``_role_binding_sha256``, over the rows in the frame's order and
      dtypes).
    - ``staging_receipt`` is staging's ACS role receipt
      (``acs_inputs.with_acs_spm_independence_role``, :478-511). It counts
      every staged ACS person, not the selection, so it is checked for its
      keys, for the RELSHIPP rule's identity (column, entity, source column,
      the ACS rule and its partition, the ASEC and adult rules, provenance)
      and for counts at least the selected ACS rows', never for equality.
      Counts cannot tie a role to a person, so each ACS person's role must
      also be the one its ``RELSHIPP`` gives under the receipt's rule (a
      code from 20 to 38 on every ACS person; the role holders 20, 21, 23,
      37 and 38).
    - The frame's role must come from its source column (``role_source ==
      "source_column"`` under
      :func:`~microcosm.build.us_runtime.spm_composition.spm_independence_role`),
      non-null and boolean on every person.

    The result holds the gate's keys
    (:func:`~microcosm.build.us_runtime.spm_independence_role.us_spm_independence_role_signal_gate`)
    recounted over the whole frame, the role source, the counts per support
    channel and per origin (with each origin's digest), and both inputs as
    plain builtins under ``evidence``. The gate still grades the bands and
    the classified-adult rule; this function records, it does not grade.
    The H5 writer does not keep frame metadata, so PR 4 writes the record
    into the population manifest and PR 5's loader restores it; the loader
    must also recompute it after the H5 round trip, since a dtype change
    moves the digest.

    Args:
        cps_record: The CPS rows' role record in the gate's keys.
        staging_receipt: Staging's ACS role receipt.
        frame: The joined frame, built by :func:`to_assembled_provenance`.

    Returns:
        The record, to store under the frame metadata key
        ``spm_independence_role``.

    Raises:
        OnePopulationContractError: If the frame's role is not its source
            column's, either origin's evidence disagrees with the frame, or
            an ACS person's role is not the one its ``RELSHIPP`` gives.
        TypeError: If an argument has the wrong type.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    for name, value in (
        ("cps_record", cps_record),
        ("staging_receipt", staging_receipt),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    if frame.schema != US_SCHEMA:
        raise OnePopulationContractError(
            "The SPM role record needs a frame in the US schema."
        )
    person = frame.table(US_SCHEMA.person_entity)
    spm_unit = frame.table("spm_unit")
    _, role_source, _ = spm_independence_role(person)
    if role_source != _ROLE_SOURCE_COLUMN:
        raise OnePopulationContractError(
            f"The SPM role must come from the source column (role_source "
            f"{_ROLE_SOURCE_COLUMN!r}); the frame's role source is "
            f"{role_source!r}: the person table lacks {NATIVE_SPM_ROLE!r}."
        )
    if "age" not in person:
        raise OnePopulationContractError(
            "The SPM role record needs 'age' on the person table (the role "
            "digest binds it)."
        )
    role = person[NATIVE_SPM_ROLE]
    if role.isna().any() or pd.api.types.infer_dtype(role, skipna=False) != "boolean":
        raise OnePopulationContractError(
            f"The SPM role record needs a non-null boolean {NATIVE_SPM_ROLE!r} on "
            f"every person; {int(role.isna().sum())} null(s), inferred type "
            f"{pd.api.types.infer_dtype(role, skipna=False)!r}."
        )
    rule = "The SPM role record needs every row's origin"
    persons = _channel_masks(person, US_SCHEMA.person_entity, rule=rule)
    units = _channel_masks(spm_unit, "spm_unit", rule=rule)

    by_channel = {
        channel: {
            "persons": int(persons[channel].sum()),
            "spm_units": int(units[channel].sum()),
            "true_role_persons": int(role[persons[channel]].sum()),
        }
        for channel in ASSEMBLED_CHANNELS
    }
    origins: dict[str, dict[str, object]] = {}
    for origin, channels in ((_CPS, _CPS_CHANNELS), (_ACS, (ACS_SUPPORT_CHANNEL,))):
        rows = np.logical_or.reduce([persons[channel] for channel in channels])
        counts = _role_counts(
            person.loc[rows],
            sum(by_channel[channel]["spm_units"] for channel in channels),
            role[rows],
        )
        origins[origin] = {
            "channels": list(channels),
            "persons": counts["persons_joined"],
            "spm_units": counts["native_spm_units"],
            "true_role_persons": counts["true_role_persons"],
            _ROLE_GATE_DIGEST_KEY: counts[_ROLE_GATE_DIGEST_KEY],
        }
        if origin == _CPS:
            _require_cps_role_record(cps_record, counts)
    _require_staging_receipt(staging_receipt, origins[_ACS])
    _require_acs_roles_by_rule(person, persons[ACS_SUPPORT_CHANNEL], role)

    return {
        **_role_counts(person, frame.n("spm_unit"), role),
        "column": NATIVE_SPM_ROLE,
        "role_source": role_source,
        "counts_by_channel": by_channel,
        "origins": origins,
        "evidence": {
            "cps_record": _plain(cps_record),
            "staging_receipt": _plain(staging_receipt),
        },
        "rule": (
            "d2283 one-population role record: the gate's counts and digest "
            "recounted over the joined frame; the CPS rows' record matched by "
            "recount, staging's ACS receipt matched by keys, rule and coverage."
        ),
    }


# ------------------------------------------------------------- absences


def _zero_written_columns(zero_written: object) -> Mapping[str, str]:
    if not isinstance(zero_written, Mapping):
        raise TypeError(
            "zero_written must map each column to the receipt that records its "
            f"written zeros, got {type(zero_written).__name__}."
        )
    for column, receipt in zero_written.items():
        if not isinstance(column, str) or not isinstance(receipt, str):
            raise TypeError(
                "zero_written must map str columns to str receipts, got "
                f"{column!r}: {receipt!r}."
            )
        if not receipt.strip():
            raise OnePopulationContractError(
                f"A zero-written column needs its receipt; {column!r} has none."
            )
    return zero_written


def _fills(
    fill_manifest: object,
    tables: Mapping[str, pd.DataFrame],
) -> list[Mapping[str, Any]]:
    """The fill manifest's fills, checked against the frame they describe."""

    rule = "The fill manifest does not describe this frame"
    if not isinstance(fill_manifest, Mapping):
        raise TypeError(
            "fill_manifest must be the reviewed default fill's manifest mapping, "
            f"got {type(fill_manifest).__name__}."
        )
    missing = sorted(_FILL_MANIFEST_KEYS - set(fill_manifest))
    if missing:
        raise OnePopulationContractError(f"{rule}: it lacks keys {missing}.")
    fills = fill_manifest["fills"]
    if isinstance(fills, str | bytes | Mapping) or not isinstance(fills, Collection):
        raise OnePopulationContractError(f"{rule}: 'fills' must be a list of fills.")
    violations = fill_manifest["unregistered_violations"]
    if violations:
        raise OnePopulationContractError(
            f"{rule}: it records unregistered violations {_plain(violations)!r}; "
            "the fill refuses those, so a manifest holding them is not a fill's."
        )
    fills = list(fills)
    seen: set[tuple[str, str]] = set()
    total = 0
    for fill in fills:
        if not isinstance(fill, Mapping) or _FILL_KEYS - set(fill):
            raise OnePopulationContractError(
                f"{rule}: each fill needs the keys {sorted(_FILL_KEYS)}; got "
                f"{_plain(fill)!r}."
            )
        entity, column = fill["entity"], fill["column"]
        if entity not in tables or column not in tables[entity]:
            raise OnePopulationContractError(
                f"{rule}: it fills {entity!r}.{column!r}, which the frame lacks."
            )
        if (entity, column) in seen:
            raise OnePopulationContractError(
                f"{rule}: it fills {entity!r}.{column!r} twice."
            )
        seen.add((entity, column))
        table = tables[entity]
        by_spine = fill["missing_rows_by_spine"]
        if not _is_count(fill["filled_rows"]) or fill["filled_rows"] < 1:
            raise OnePopulationContractError(
                f"{rule}: {entity!r}.{column!r} fills {fill['filled_rows']!r} rows."
            )
        if not _is_count(fill["rows"]) or fill["rows"] != len(table):
            raise OnePopulationContractError(
                f"{rule}: its {entity!r} table had {fill['rows']!r} rows; the "
                f"frame's has {len(table)} (prepare fills the joined frame)."
            )
        # Each count a nonnegative integer before the sum: a negative count
        # can balance the sum and would be filed as an entry's rows (PR
        # #1685 review, should-fix 5).
        if (
            not isinstance(by_spine, Mapping)
            or set(by_spine) - {CPS_SPINE_TAG, ACS_SUPPORT_CHANNEL}
            or not all(_is_count(value) and value >= 0 for value in by_spine.values())
            or sum(by_spine.values()) != fill["filled_rows"]
        ):
            raise OnePopulationContractError(
                f"{rule}: {entity!r}.{column!r}'s rows by spine "
                f"{_plain(by_spine)!r} must count its {fill['filled_rows']!r} "
                f"filled rows, in nonnegative integers, under the spine tags "
                f"{CPS_SPINE_TAG!r} and {ACS_SUPPORT_CHANNEL!r}."
            )
        if table[column].isna().any():
            raise OnePopulationContractError(
                f"{rule}: {entity!r}.{column!r} still holds "
                f"{int(table[column].isna().sum())} null(s) after its fill."
            )
        total += int(fill["filled_rows"])
    if (fill_manifest["columns_filled"], fill_manifest["total_values_filled"]) != (
        len(fills),
        total,
    ):
        raise OnePopulationContractError(
            f"{rule}: it says {fill_manifest['columns_filled']!r} columns and "
            f"{fill_manifest['total_values_filled']!r} values filled; its fills "
            f"are {len(fills)} columns and {total} values."
        )
    return fills


#: The spine tag each origin's rows carry on the joined frame (Q3).
_ORIGIN_OF_SPINE_TAG = MappingProxyType(
    {CPS_SPINE_TAG: _CPS, ACS_SUPPORT_CHANNEL: _ACS}
)
_ORIGIN_CHANNELS = MappingProxyType({_CPS: _CPS_CHANNELS, _ACS: (ACS_SUPPORT_CHANNEL,)})


@dataclass(frozen=True)
class _EntityMass:
    """One entity's rows and weights by channel, for the absences file."""

    channels: Mapping[str, np.ndarray]
    weights: np.ndarray
    total_weight: float

    def origin_rows(self, origin: str) -> np.ndarray:
        return _origin_rows(self.channels, origin)

    def share(self, rows: np.ndarray) -> float | None:
        if not self.total_weight > 0:
            return None
        return float(self.weights[rows].sum() / self.total_weight)

    def counted(self, rows: np.ndarray, origin: str) -> dict[str, object]:
        """Rows and entity-weight shares of ``rows`` per channel of ``origin``."""

        channels = _ORIGIN_CHANNELS[origin]
        by_channel = {c: int((rows & self.channels[c]).sum()) for c in channels}
        shares = {c: self.share(rows & self.channels[c]) for c in channels}
        return {
            "rows": int(rows.sum()),
            "origin_rows": int(self.origin_rows(origin).sum()),
            "rows_by_channel": by_channel,
            "entity_weight_share": self.share(rows),
            "entity_weight_share_by_channel": (
                None if None in shares.values() else shares
            ),
        }


def _entity_mass(frame: Frame, entity: str, table: pd.DataFrame) -> _EntityMass:
    channels = _channel_masks(
        table, entity, rule="The absences file needs every row's origin"
    )
    weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
    return _EntityMass(
        channels=channels, weights=weights, total_weight=float(weights.sum())
    )


def _entry(
    entity: str,
    column: str,
    kind: str,
    origin: str,
    counted: Mapping[str, object],
    *,
    rows_rule: str,
    evidence: str,
    readers: Iterable[str] = (),
    detail: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "entity": entity,
        "column": column,
        "kind": kind,
        "origin": origin,
        **counted,
        "rows_rule": rows_rule,
        "evidence": evidence,
        "readers": list(readers),
        "detail": dict(detail or {}),
    }


def _absence_kind(
    entity: str, column: str, origin: str, declared: Mapping[str, str]
) -> tuple[str, str, tuple[str, ...], dict[str, object]] | None:
    """The kind of a column ``origin``'s rows lack, or ``None`` if it has none.

    A kind that needs no decision (release-written, raw source column,
    origin-native) wins over a declaration, which files only a gap no such
    kind covers.
    """

    from microcosm.data.stored_inputs import is_model_named

    if column in RELEASE_WRITTEN_INPUTS:
        return (_RELEASE_WRITTEN, RELEASE_WRITTEN_INPUTS[column], (), {})
    if not is_model_named(column):
        readers = RAW_COLUMN_READERS.get(column, ())
        evidence = "a raw source column the stored-input rule ignores; " + (
            f"the release reads it: {', '.join(readers)}"
            if readers
            else "no release reader is listed in RAW_COLUMN_READERS"
        )
        return (_RAW_SOURCE, evidence, readers, {})
    native = ACS_NATIVE_AMOUNTS.get(column)
    if origin == _CPS and native is not None and native[0] == entity:
        return (_ORIGIN_NATIVE, _ORIGIN_NATIVE_EVIDENCE, native[1], {})
    if column in declared:
        return (
            _OWNER_DECLARED,
            declared[column],
            (),
            {"decision": declared[column]},
        )
    return None


def absences(
    frame: Frame,
    *,
    fill_manifest: Mapping[str, Any],
    zero_written: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, object]:
    """What one origin carries as a default fill, a written zero, or not at all.

    Returns the content of :data:`ABSENCES_FILENAME` (write it with
    :func:`write_absences`): one entry per (entity, column, kind, origin)
    with the rows it counts, per support channel, and their share of the
    entity's total weight on the frame
    (:meth:`~microcosm.frame.Frame.resolve_weights`), per channel too. Origins
    are keyed by support channel (``asec`` and ``puf_tax_detail`` for CPS,
    ``acs_2024_1yr`` for ACS), never by ``<entity>_spine``. Each entry
    carries one of :data:`ABSENCE_KINDS`:

    - A column null on some row of one origin while the other origin holds a
      value somewhere (both directions: what CPS rows fill and ACS rows
      lack, and the reverse) is "release-written" if it is in
      :data:`RELEASE_WRITTEN_INPUTS`, "raw source column" if it is not
      model-named (the entry names the release's readers from
      :data:`RAW_COLUMN_READERS`, so the predictor-join columns and #1610's
      four household-type columns are never filed as harmless),
      "origin-native" if CPS rows lack one of :data:`ACS_NATIVE_AMOUNTS` on
      its own entity (the entry names the ACS transfer where it reads the
      amount), and "owner-declared" if ``declared`` names it, with the
      owner's decision. Those kinds are tried in that order, so a
      declaration files only a gap no other kind covers. Any other such
      column refuses, all of them in one message: prepare must fill it or
      its owner declare it. The gap is :func:`_origin_gap`, the predicate
      :func:`require_one_population`'s R10 shares (d2283 lead call L2), so
      one declaration answers both. This file is still the stricter of the
      two: R10 reads only register entries, and only ACS rows lacking a
      column CPS rows hold, where this reads every model-named column in
      both directions.
    - "reviewed default fill": each fill of ``fill_manifest``, the reviewed
      default fill's manifest (``tools/build_us_acs_local_release.py``
      ``fill_reviewed_nulls``, :1151-1165), checked against the frame
      (table rows, no null left, per-spine counts nonnegative integers adding
      up, spine tags known). The
      fill writes in place and the manifest counts rows by spine tag without
      naming them, so per-channel rows and weighted shares are given only
      when the fill covered every row of that origin; otherwise they are
      ``None`` and ``rows_rule`` says why.
    - "zero written": each column of ``zero_written``, which maps it to the
      receipt recording staging's written zeros. Its rows are the ACS rows
      holding zero (a null, in any numeric dtype, is not a zero), an upper
      bound: a measured zero is not told apart.

    ``declared`` is the mapping :func:`require_one_population` takes, so a declaration this frame does not use is recorded, never
    refused (d2283 lead call (b), 11-18): ``needless_declarations`` lists
    each declared column that needs no decision here (release-written, not
    model-named, named by ``fill_manifest`` or ``zero_written``, whose
    receipt files it, or whose every gap on this frame took a kind that
    needs none, such as an origin-native amount), and ``unused_declarations``
    each other declared column no origin lacks on this frame (a stale
    declaration; on a frame with no ACS row, every such declaration). Each
    list holds ``{"column", "decision"}`` entries sorted by column. A
    declaration for an engine input is filed like any other, although
    :func:`require_one_population`'s R6 refuses that input's nulls on the
    same frame whatever is declared. A column named by both
    ``fill_manifest`` and ``zero_written`` refuses: an absence has one kind.

    Args:
        frame: The joined frame after prepare's fill.
        fill_manifest: The reviewed default fill's manifest for this frame.
        zero_written: Column to the receipt recording its written zeros.
        declared: Column to its owner's decision, the mapping
            :func:`require_one_population` takes as ``declared_absences``.

    Returns:
        The absences record: plain builtins, entries and both declaration
        lists sorted.

    Raises:
        OnePopulationContractError: If an input does not describe the frame,
            or a column one origin lacks has no kind.
        TypeError: If an argument has the wrong type.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    if frame.schema != US_SCHEMA:
        raise OnePopulationContractError(
            "The absences file needs a frame in the US schema."
        )
    declared = _declarations(declared, name="declared")
    zero_written = _zero_written_columns(zero_written)
    tables = _frame_tables(frame)
    fills = _fills(fill_manifest, tables)

    filled_columns = {str(fill["column"]) for fill in fills}
    both = sorted(filled_columns & set(zero_written))
    if both:
        raise OnePopulationContractError(
            f"An absence has one kind: {both} are named by both fill_manifest "
            "and zero_written."
        )
    # A declared column a receipt also names is needless: the receipt files it.
    receipted = filled_columns | set(zero_written)
    masses = {
        entity: _entity_mass(frame, entity, tables[entity])
        for entity in US_SCHEMA.entities
    }
    entries: list[dict[str, object]] = []
    used: set[str] = set()
    gapped: set[str] = set()
    gaps: dict[str, str] = {}
    for entity in US_SCHEMA.entities:
        table, mass = tables[entity], masses[entity]
        for column in table.columns:
            if not isinstance(column, str):
                continue
            null = table[column].isna().to_numpy()
            if not null.any():
                continue
            for origin, other in ((_CPS, _ACS), (_ACS, _CPS)):
                lacking = _origin_gap(
                    null, mass.origin_rows(origin), mass.origin_rows(other)
                )
                if lacking is None:
                    continue
                gapped.add(column)
                classified = _absence_kind(entity, column, origin, declared)
                if classified is None:
                    gaps[f"{entity}.{column}"] = (
                        f"null on {int(lacking.sum())} {origin} row(s)"
                    )
                    continue
                kind, evidence, readers, detail = classified
                if kind == _OWNER_DECLARED:
                    used.add(column)
                entries.append(
                    _entry(
                        entity,
                        column,
                        kind,
                        origin,
                        mass.counted(lacking, origin),
                        rows_rule="null on these rows; the other origin holds values",
                        evidence=evidence,
                        readers=readers,
                        detail=detail,
                    )
                )
    if gaps:
        raise OnePopulationContractError(
            f"A column one origin lacks has no absence kind: {gaps}. Each is "
            "model-named and not release-written, and the other origin holds "
            "values in it, so prepare must fill it or its owner must declare "
            "the absence with a decision."
        )
    from microcosm.data.stored_inputs import is_model_named

    needless = sorted(
        column
        for column in set(declared) - used
        if column in gapped
        or column in receipted
        or column in RELEASE_WRITTEN_INPUTS
        or not is_model_named(column)
    )
    unused = sorted(set(declared) - used - set(needless))

    for fill in fills:
        entity, column = str(fill["entity"]), str(fill["column"])
        mass = masses[entity]
        for tag, filled in sorted(fill["missing_rows_by_spine"].items()):
            if not filled:
                continue
            origin = _ORIGIN_OF_SPINE_TAG[tag]
            rows = mass.origin_rows(origin)
            if filled > int(rows.sum()):
                raise OnePopulationContractError(
                    f"The fill manifest does not describe this frame: it fills "
                    f"{filled} {origin} rows of {entity!r}.{column!r}; the frame "
                    f"holds {int(rows.sum())}."
                )
            if filled == int(rows.sum()):
                counted = mass.counted(rows, origin)
                rows_rule = "every row of the origin (the fill's count equals them)"
            else:
                counted = {
                    "rows": int(filled),
                    "origin_rows": int(rows.sum()),
                    "rows_by_channel": None,
                    "entity_weight_share": None,
                    "entity_weight_share_by_channel": None,
                }
                rows_rule = (
                    "the fill manifest's count; it names no rows, so the "
                    "channels and weighted shares are unknown"
                )
            entries.append(
                _entry(
                    entity,
                    column,
                    _DEFAULT_FILL,
                    origin,
                    counted,
                    rows_rule=rows_rule,
                    evidence=(
                        "prepare's reviewed default fill, reviewed by staging's "
                        "reviewed_engine_input_nulls register"
                    ),
                    detail={
                        "fill_value": _plain(fill["fill_value"]),
                        "fill_kind": _plain(fill["fill_kind"]),
                        "register_missing_rows": _plain(fill["register_missing_rows"]),
                    },
                )
            )

    for column, receipt in sorted(zero_written.items()):
        holders = [entity for entity in US_SCHEMA.entities if column in tables[entity]]
        if not holders:
            raise OnePopulationContractError(
                f"A zero-written column the frame lacks: {column!r}."
            )
        entity = holders[0]
        values = tables[entity][column]
        if pd.api.types.is_bool_dtype(values.dtype) or not (
            pd.api.types.is_numeric_dtype(values.dtype)
        ):
            raise OnePopulationContractError(
                f"A zero-written column must be numeric: {entity!r}.{column!r} "
                f"is {values.dtype}."
            )
        mass = masses[entity]
        # A plain boolean mask: on a nullable dtype (Float64, Int64) eq(0) is
        # a nullable boolean whose to_numpy() is an object array, which the
        # weight lookup cannot index with; a null is not a zero (PR #1685
        # review, should-fix 6).
        zero = values.eq(0).to_numpy(dtype=bool, na_value=False) & mass.origin_rows(
            _ACS
        )
        entries.append(
            _entry(
                entity,
                column,
                _ZERO_WRITTEN,
                _ACS,
                mass.counted(zero, _ACS),
                rows_rule=(
                    "ACS rows holding zero: an upper bound, since a measured "
                    "zero is not told apart from a written one"
                ),
                evidence=receipt,
                detail={"receipt": receipt},
            )
        )

    entries.sort(
        key=lambda entry: (
            US_SCHEMA.entities.index(entry["entity"]),
            entry["column"],
            ABSENCE_KINDS.index(entry["kind"]),
            entry["origin"],
        )
    )
    return {
        "schema_version": _ABSENCES_SCHEMA_VERSION,
        "file": ABSENCES_FILENAME,
        "kinds": list(ABSENCE_KINDS),
        "channels": list(ASSEMBLED_CHANNELS),
        "share_denominator": (
            "each entity's total weight on the frame (Frame.resolve_weights)"
        ),
        "rows_by_channel": {
            entity: {
                channel: int(mask.sum())
                for channel, mask in masses[entity].channels.items()
            }
            for entity in US_SCHEMA.entities
        },
        "fill_manifest_period": _plain(fill_manifest["period"]),
        "entries": entries,
        "unused_declarations": [
            {"column": column, "decision": declared[column]} for column in unused
        ],
        "needless_declarations": [
            {"column": column, "decision": declared[column]} for column in needless
        ],
    }


def _absences_json(record: Mapping[str, object]) -> str:
    """The absences file's bytes: PR 1's canonical JSON, one trailing newline."""

    return (
        json.dumps(
            _plain(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_absences(record: Mapping[str, object], directory: str | Path) -> Path:
    """Write :func:`absences`' record as ``directory/``:data:`ABSENCES_FILENAME`.

    The JSON is deterministic: sorted keys, no whitespace, ASCII, and
    ``allow_nan=False`` (a non-finite share refuses rather than writes).

    Returns:
        The written file's path.
    """

    if not isinstance(record, Mapping):
        raise TypeError(f"record must be a mapping, got {type(record).__name__}.")
    path = Path(directory) / ABSENCES_FILENAME
    path.write_text(_absences_json(record), encoding="ascii")
    return path


def _is_peridnum(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= _CPS_UNIT_ID_LENGTH
        and value.isascii()
        and value.isdigit()
    )


def cps_unit_ids(person: pd.DataFrame) -> pd.DataFrame:
    """Each CPS household's income year, unit id and selection key.

    The unit id is ``PERIDNUM[:20]`` (H_IDNUM), on which every person of the
    household must agree, as the support-mix bake-off reads it
    (``tools/bakeoff_us_support_mix.py`` ``_row_index_columns``). It comes
    from ``PERIDNUM`` when the column is present, else from
    ``source_person_id`` (the ASEC pool writes ``PERIDNUM`` there). Either
    way every value must be a string of at least 20 ASCII digits: the ASEC
    pool's integer fallback, a number cast to text and a null are refused,
    never sliced (unlike the bake-off's ``astype(str)``). The income year is
    ``source_year``, an integer on which the household's persons agree. The
    key is
    :func:`~microcosm.build.us_runtime.one_population_selection.cps_unit_key`,
    so the prepare step's selection keys (PR 4 builds them with this
    function) and the contract's key check share one rule; digest a key set
    with :func:`~microcosm.build.us_runtime.one_population_selection.keys_sha256`.

    Every physical household gets a row: a source household's PUF clone
    households carry its PERIDNUM, so they share its key. The selection's
    ``cps_keys`` are the clone-0 households' keys.

    Args:
        person: CPS person rows with ``person_household_id``,
            ``source_year`` and ``PERIDNUM`` or ``source_person_id``.

    Returns:
        One row per household id (``person_household_id``), sorted, in an
        index named ``household_id``, with columns ``income_year`` (int64),
        ``unit_id`` and ``key``.

    Raises:
        OnePopulationContractError: If an id or year is missing or malformed,
            or a household's persons disagree.
    """

    if not isinstance(person, pd.DataFrame):
        raise TypeError(f"person must be a DataFrame, got {type(person).__name__}.")
    household_column = US_SCHEMA.membership_column("household")
    if _PERIDNUM in person:
        column = _PERIDNUM
    elif _SOURCE_PERSON_ID in person:
        column = _SOURCE_PERSON_ID
    else:
        raise OnePopulationContractError(
            f"CPS unit ids need {_PERIDNUM} or {_SOURCE_PERSON_ID} on the persons."
        )
    missing = [c for c in (household_column, _SOURCE_YEAR) if c not in person]
    if missing:
        raise OnePopulationContractError(f"CPS unit ids need {missing} on the persons.")
    values = person[column]
    bad = ~values.map(_is_peridnum).astype(bool)
    if bad.any():
        raise OnePopulationContractError(
            f"CPS unit ids need {column} as a string of at least "
            f"{_CPS_UNIT_ID_LENGTH} digits on every person (the ASEC PERIDNUM); "
            f"{int(bad.sum())} person(s) hold values such as "
            f"{values[bad].head(3).tolist()!r}."
        )
    households = _integral_ids(
        person[household_column],
        refusal=f"CPS unit ids need integral {household_column!r}",
    )
    years = _integral_ids(
        person[_SOURCE_YEAR], refusal=f"CPS unit ids need an integral {_SOURCE_YEAR}"
    )
    per_person = pd.DataFrame(
        {
            "household_id": households,
            "income_year": years,
            "unit_id": [value[:_CPS_UNIT_ID_LENGTH] for value in values.tolist()],
        }
    )
    grouped = per_person.groupby("household_id", sort=True)
    for name, label in (
        ("unit_id", f"{column}[:{_CPS_UNIT_ID_LENGTH}] (H_IDNUM)"),
        ("income_year", _SOURCE_YEAR),
    ):
        split = grouped[name].nunique()
        if (split != 1).any():
            raise OnePopulationContractError(
                f"CPS unit ids: persons of one household disagree on {label} in "
                f"household(s) {split.index[split != 1].tolist()[:5]}."
            )
    result = grouped[["income_year", "unit_id"]].first()
    try:
        keys = [
            cps_unit_key(int(year), unit)
            for year, unit in zip(
                result["income_year"].tolist(), result["unit_id"].tolist(), strict=True
            )
        ]
    except ValueError as exc:
        raise OnePopulationContractError(
            f"CPS unit ids: a household's key is malformed: {exc}"
        ) from exc
    return pd.DataFrame(
        {
            "income_year": result["income_year"].to_numpy(dtype=np.int64),
            "unit_id": pd.Series(result["unit_id"].tolist(), dtype=object).to_numpy(),
            "key": pd.Series(keys, dtype=object).to_numpy(),
        },
        index=pd.Index(result.index.to_numpy(dtype=np.int64), name="household_id"),
    )
