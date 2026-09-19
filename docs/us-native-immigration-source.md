# Native immigration source projection

`current_survey_immigration_source` provides the original evidence needed for
the humanitarian immigration consumer reviewed at Microcosm PR #779 head
`7ee36ac1bea125218912996b1030b77a208db963`. It assigns no status and runs no
stock draw, fit, calibration or country model. It is not attached to the graph.

The new sibling source intentionally continues native integration
`cf04225c2c2c5c2e3eafe8ef5f49aa4c5feabd8f`; the freshly fetched main lacked the
native owner APIs. No old PR source tree, source-stage manifest, schema or seed
identity was adopted. The original humanitarian checkout was inspected
read-only. Subfleet's live-session inventory had no matching humanitarian owner;
its recent ledger contained only a completed #779 gate review. No message was
sent and no other checkout changed.

## API and evidence

An authorized native build can call:

```python
from microcosm.build.us_runtime.current_survey_immigration_source import (
    borrow_cloned_immigration_evidence,
    qualify_current_survey_immigration,
)

qualified = qualify_current_survey_immigration(preparation)
columns = borrow_cloned_immigration_evidence(qualified, receiving_frame.person)
```

`preparation` must be the live issued `AuthenticatedSurveyPopulationPreparation`.
The qualifier authenticates the retained current ASEC CSV and ACS person ZIP,
exhausts their bounded records, verifies exact digests and counts, and rechecks
the owners after I/O. It returns private tables plus an aggregate receipt:

- `asec_full_raw`: all income-2024 ASEC persons from the issued coverage roster,
  indexed by the original parent `source_person_id`. Exact `PERIDNUM`, `PH_SEQ`
  and `A_LINENO` remain strings. The full native-key roster and ages must match
  the independent coverage owner. Only identity fields and the exact 23-field
  #779 readset are retained. This full table survives smaller support selections
  so a later national stock assignment cannot accidentally run on selected rows.
- `asec_selected_raw` and `acs_raw`: original literals for selected people,
  indexed by preparation person ID, bound to the separate `origins` table.
  ACS retains `CIT`, `POBP`, `YOEP`, `AGEP` and its original keys. These support
  #779's source-aware profile; they do not supply an ASEC-UA indicator set or
  authorize ACS status assignment.
- `person_evidence`: namespaced `immigration_source_*` literals and observation
  year. Blank source tokens remain blank strings; another survey's columns are
  missing. Codes, decimals, whitespace and native string IDs are not filled or
  coerced. Allocation/response provenance is not inferred.

The exact ASEC readset replaces the older consumer's `WSAL_VAL` / `SEMP_VAL`
proxies with `A_LFSR`; its order was checked against the exact PR source by AST.
`A_LFSR` is current labor-force evidence at the **2025 interview**, whereas
income refers to **2024**. ACS observation year is **2024**. The qualifier does
not silently apply the old arrival-code midpoints or stock-year controls to
these literals. No prior-income columns or person-weight column are read here.

The clone function verifies the existing clone-0/1 ancestry, source channel and
native IDs, then returns detached column Series without changing the source or
receiver. It refuses column collisions. It performs no draws. Exact output and
receiver seals are checked after the last source validation. A graph host must
still authenticate its own receiving population; these columns and the JSON
receipt are not receiving or release authority. Keep the qualified object and
call `validate()` after subsequent relevant I/O before consuming its tables.

## Required next integration

The receipt's `missing_rule_admissions` lists the concrete unfinished steps:

1. Check 2025 ASEC observation/arrival-code semantics against the exact rules.
   Reconcile the rule period and stock year explicitly.
2. Bind the full ASEC native-key roster to its authoritative household design
   weights and approved controls. Run the reviewed deterministic status draw
   once on original sources, then select and transport its results. The full
   literal table alone does not qualify the national stock mass or weight basis.
3. Admit the ACS transfer/assignment method with its own source eligibility and
   cohort rules; do not populate absent ASEC fields with defaults.
4. Add the graph attachment under the retained native source and receiving
   owners, preserving original status assignments across clones and replay.
5. Qualify country-model/export inputs and run the written-H5 immigration
   composition and four HR1 smoke probes. A source projection cannot satisfy
   their national stock bands or dollar floors.

`status_assignment_performed` and `national_stock_alignment_qualified` describe
this projection's actual scope. They are not a permanent veto on a later
separately qualified consumer. The existing immigration implementation and
legacy release gates remain unchanged.

Tests use real source issuers and original-member capture paths over invented
archives. They exercise full-versus-selected coverage, literal/unknown-state
preservation, exact clone transport, copied-owner rejection and mutations during
final I/O. No real Census/ACS data or country model was used in this slice.
