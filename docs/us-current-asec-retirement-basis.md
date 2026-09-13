# ASEC retirement candidate ledger

`current_asec_retirement_basis.py` describes potential retirement-bridge support
from the accepted retirement-detail and income-routing projections. It preserves
source observations, route uncertainty and published accounting differences.
It produces no fiscal inputs, fitted model, graph host or source authority.

## API

`build_asec_retirement_basis` accepts keyword arguments `retirement_detail`,
`income_routing`, `original_household_membership`,
`original_household_design_weights`, and `assumptions`. The two DataFrames must
have identical ordered int64 `person_id` indices, exact native person IDs, and
original interview ages. Shared amount observations and pension receipt codes
must agree. A joint permutation is accepted; a one-sided permutation refuses.
The current contract requires at least one selected person.

It returns `AsecRetirementBasis` with six descriptive fields:

- `person`: source identifiers, observed routed subtotals, candidate family
  intervals, accounting differences and candidate-support flags.
- `slots`: ten records per person, retaining six pension/disability/survivor
  slots and four applicable/off-route distribution slots, source codes,
  published amounts, knownness, source statuses and bridge routes.
- `provenance`: detached, prefixed copies of both complete supplied projections,
  including allocation and disclosure flags.
- `exclusions`: uncertainty and route diagnostic masks, including cases that
  remain numerically usable. These masks are not all applied as donor filters.
- `summary`: counts and supplied original DESIGN-weight support for each
  candidate tier and diagnostic mask.
- `assumptions_payload`: canonical immutable bytes identifying the choices.

The source projection dataclasses and these outputs contain mutable descriptive
tables. Neither their type nor a digest grants source admission. A future host
must retain and requalify the actual original source owners, complete person
scope, table seals and original DESIGN weights around its relevant I/O. This
pure function does not write to either input, calibrate weights or attach clones.

## Required choices

Every `RetirementCandidateAssumptions` field must be supplied. No country caller
or development scenario is enabled by this change.

| Field | Supported values |
| --- | --- |
| `pension_annuity_regularity` | `unresolved`, `assume_regular` |
| `disability_pension_eligibility` | `unresolved`, `assume_qualifying` |
| `survivor_annuity_overlap` | `unresolved` |
| `withdrawal_regularity_netting` | `unresolved` |
| `aggregate_accounting` | `exact_visible_balance_only` |

`assume_regular` is a measurement scenario about pension/annuity regularity.
`assume_qualifying` explicitly assumes both the disability-pension scope and
regularity needed for the candidate bridge. Neither is inferred from an annual
amount, age, `DIS_HP`, or `DIS_CS`. The latter source answers remain provenance.
There are no private-pension, regular-withdrawal or taxable fractions. Changing
either supported scenario changes `to_bytes()`; unsupported choices refuse.

## Routes and candidate amounts

The accepted [retirement-detail source](us-current-asec-retirement-detail-source.md)
and [income-routing source](us-current-asec-income-routing-source.md) retain the
published codebooks. This ledger imports those definitions and applies a separate
bridge route classification:

| Family | Candidate routes | Separate or unresolved routes |
| --- | --- | --- |
| Pension | Codes 1–6 | 7 Railroad; 8 unresolved |
| Disability | Codes 2–5, subject to the explicit scenario | 6 Railroad; 1/7/8/9 other compensation; 10 unresolved |
| Survivor | Codes 1–4 are visible candidate subtotals | 5 Railroad; 8 property; 9 annuity overlap; 6/7/10 unresolved |
| Distribution | Applicable account codes 1–7 retained individually | Regularity/netting remain unknown; account 4 is a type of IRA |
| Other income | 2/13 indicate additional retirement candidates | 1 Social Security overlap; 8 property overlap; 19 unspecified |

The retirement ledger never consumes account-earnings amounts or adds Railroad
amounts to Social Security. Other-income amounts are retained separately and
never added to the candidate sum. Categories 2/13/19 leave combined candidate
scope unresolved; known other categories can remain outside that sum. These
routes do not establish any person's tax treatment.

A source-known amount and a published literal remain separate. A declared
positive slot contributes to its observed routed subtotal. A readable unused
NIU slot can contribute an explicitly labeled structural zero to a numerical
comparison, but its source-known amount remains missing. The subtotal is only
a sum of the readable known slots; it never asserts that unknown slots are zero.
Yes-receipt dollar zeros stay ambiguous where the source domain says none/NIU.
Annuity preserves its distinct valid dollar zero and published `-1` NIU encoding.

The four published total-minus-slot comparisons are recomputed and checked
against R1. `DBTN_VAL` compares the main distribution slots only; the applicable
known distribution total is supplied independently by the routing owner. No
comparison is allocated to another source or attributed causally to disclosure.

For pension/disability candidate bounds, all relevant slots must be readable and
receipt-consistent, and their exact integer-dollar sum must equal the aggregate.
Positive discrepancy leaves additional scope unresolved; negative discrepancy
is contradictory accounting. Neither is clipped, used as a cap, or silently
allocated. Readable outside routes are subtracted only under this exact balance.
With unresolved regularity/scope the lower bound is zero; under the named
scenario it includes only known candidate-route amounts. Unresolved source
routes can widen the candidate interval without becoming identified labels.

Positive survivor receipt keeps full survivor bounds unknown, even when its
published comparison balances. Visible slots do not prove the scope of sources
3/4. Only qualified no receipt with two NIU slots and zero total supplies a
zero survivor family amount. Source 9 alongside positive annuity is explicitly
flagged; no overlap priority or max/sum assumption is applied.

A fully resolved applicable distribution composition produces a zero-to-total
candidate interval. Missing/unreadable account types, ambiguous active zeros or
off-route answers/dollars prevent that interval. The routing owner's known total
is still preserved when the stricter candidate composition test fails. Account
subtotals are finite only when the complete account composition is known.

Under-15 people retain their source evidence with outside, unresolved-outside or
contradictory-outside status and no candidate zeros. Combined bounds are available
only when every family and other-income scope passes. `candidate_basis_eligible`
means this conservative numerical support exists. `candidate_point_under_assumptions`
means the candidate bounds coincide; `point_identified` and
`fiscal_outputs_produced` remain false. No training label is created merely by
calling a quantity an interval endpoint.

## Source and validation boundaries

The [2025 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
provides the source categories and domains. The [2024 ACS subject definitions,
page 95](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf#page=95)
provide the target's regularity and disability scope. The ledger's conservative
accounting and scenario choices are separate decisions. It does not equate the
surveys' reference periods, allocate joint income, or derive source regularity
or tax treatment from those documents.

Tests use the actual pure R1/routing projectors on invented records. They check
source routes, exact accounting, unknowns, assumption identities, interview-age
routes, native IDs above `2**53`, one-sided mutations, permutations, detached
provenance and descriptive design-weight support. Person mass uses the supplied
original household weight for each included person; household mass counts each
included household once. The Series has no independent weight-kind authority.
Allocation/disclosure flags do not filter candidates. No native data, source
capture, engine, fitting, calibration, PUF or release acceptance is involved.
