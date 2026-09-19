# Private completion observation retention

Locally verified, 2026-09-19. This continues native9af; main reconciliation
is separate. No native run, default switch, push, merge, or publication is
part of this change. The approved plan remains byte-identical as the review
record, including its initial narrower RangeIndex wording.

## Admission and direction

`run_atomic_survey_financial(..., _population_retention="compact")` selects
private compact retention. The default remains `"all"`; child recursion
forwards the explicit choice. Normative receipts do not include retention.
The executor/store and original replay comparator remain unchanged.

Witnesses admit exact Frame/Population/schema classes, NumPy primitive bool,
integer and floating dtypes (1/2/4/8 bytes), object cells accepted by the
existing typed scalar codec, exact pandas masked integer/bool arrays, and
exact Python/Arrow StringDtype storage with NA/nan policies. Arrays and cells
are hashed with bounded column/chunk buffers; immutable canonical JSON bytes
contain no live Frame, Series, array or caller object aliases.

Axis admission is narrower than column admission: exact plain Index with
native integer/unsigned values, or nonmissing exact strings in object/StringDtype
storage; and exact RangeIndex. Names are None or exact strings. Both exact
index classes must retain `_comparables == ["name"]`, otherwise compact
observation refuses. All DatetimeIndex, TimedeltaIndex, PeriodIndex,
CategoricalIndex, MultiIndex, float axes and subclasses refuse.

The initial 0..N-only RangeIndex admission refused an actual invented native
financial graph. Inspection of pandas3.0.3 shows RangeIndex.equals compares
Python ranges by sequence and Index.identical adds name/type/dtype. The
implemented witness binds the arithmetic sequence (length, start, step, with
canonical empty/singleton forms) separately from the exposed integer array.
This also catches a changed private range hidden by pandas' stale array cache.
It admits valid arbitrary start/stop/step while preserving empty/singleton
equivalences. Differential
tests cover negative/nonunit/reverse/empty/singleton cases; this is an explicit
profile clarification, not silent omission of axis metadata.

Masked comparison is directional: masks and present bytes must agree, and
full backing bytes must agree unless the *actual* missing backing is allzero.
Completion compares the base issuer observation as expected with the new
union observation as actual. No normalized equivalence or transitivity rule
is used. Canonical context comparison must distinguish booleans/integers,
float representations and ledger scalar types, as the untouched oracle does.

Unsupported or malformed actual state may refuse during observation, earlier
than the original later comparison. Mixed-defect error-code precedence is
therefore intentionally not promised; admitted well-formed comparisons use
the same semantic checks. Bytes alone confer no source or population authority.
Only the actual private financial issuance entry can supply completion's base
witness and receipt-input capsule.

## Retained consumers

Financial compact runs retain the actual financial attach, optional property
attach and optional tax gate snapshots. Full completion retains every extension
observation (10 without household roles;12 with roles), and no union base
observation. Child DONOR/RECIPIENT/FIT/DRAW still use the special declaration-
bound support stamp. Real receiving/role/child/tax verifiers consume live
extension objects and keep existing identity/mutation/refusal checks.

Completion's other base consumer, `_states`, uses frozen independently checked
population-derived inputs (structural column roster, mass and design-cap receipt
projection). It rederives current keys, implementations, capabilities, typed
contracts, seeds, artifact keys and writer obligations in union order. Issued
old base state alone is not accepted as evidence of the union. Source checks,
parent checked_view, expected mutation seals, actual store hashes and attached
manifest frame checks remain mandatory.

## Later graph capacity is separate

This opt-in change ends at the full survey completion graph. The later
`graph_survey_puf55.run_survey_puf55` and `graph_us_survey_enrichment` observers
currently retain every observed Population in their own graphs. Those later
snapshots can dominate memory even when the completion parent uses compact
retention. No downstream observer change is included here, and successful
full51 qualification must not be treated as PUF or enrichment capacity
qualification. Their retention, consumers and numerical admission need
separate assessment before larger builds.

## Validation status

Synthetic tests only. The source runtime was temporarily deleted by another
owner and subsequently restored; these implementation tests continue in the
preserved `microcosm-verify-once/.venv-noengine` fallback to avoid environment
churn. The receipt records exact interpreter/package versions and module
origins/hashes. No native performance comparison or RSS benefit is claimed.

The final production source is frozen in `source-fingerprint-v3.json`, with
test bytes in `test-fingerprint-v3.json`. The independent source reviewer
checked all six hashes and found no additional actionable issue. The separate
Fable request never reached model service: Subfleet capacity and subsequent
service/credential recovery blocked it. No Fable approval is claimed.

The required matrix passed 322 distinct named cases on this freeze: 59 witness,
17 compact financial, 196 original oracle/support/executor, three full 45/49/51
completion cases, and 47 unchanged default-path cases. Each completion case
executes all-cold, all-required, compact-cold and compact-required and compares
full outputs, receipts, lineage and artifacts while checking retained custody.

The default suite was scheduled in three exact disjoint partitions. The first
command was deliberately interrupted after 22 named passes and returned 2;
that whole command is not a successful run, and its unnamed interruption XML
element is not a test. The standalone default45 and remaining default51/rest
commands passed 1 and 24 cases. Their union matches all 47 collected IDs exactly.
`validation.json` records the independently reconciled XML/command hashes and
outcomes. The original witness XML is retained without a separate v3 command
JSON; no missing command metadata is invented.

After this matrix finished, the compact test received formatting only. The
executable AST and all 20 collected test IDs remain identical, as recorded in
`test-format-validation.json`; `test-fingerprint-final.json` separately records
the final bytes. Production bytes remain identical to the reviewed V3 freeze.
Local command receipts, pytest XML/logs, runtime inventories and superseded
partial runs remain preserved beside this document and are ignored by Git.

The declaration-cache integration trial runs on a separate candidate and does
not replace these results. Actual native execution and memory/capacity admission
remain outstanding; no full dataset or release qualification is claimed here.
