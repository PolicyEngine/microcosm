# Native other-disability completion

This optional development fragment estimates `disability_benefits` for selected
original ACS people aged 15 or older. The target comes from the existing qualified
ASEC two-slot other-disability adapter: workers' compensation and Social Security
disability remain separate leaves. Under-15 people and unresolved ASEC reports
remain unknown. This is not consumer-completeness or release certification.

`qualify_current_asec_retirement_detail(..., full_original=True)` and
`qualify_current_asec_other_disability(..., full_original=True)` expose the complete
current ASEC source through their actual retained owner. The default remains the
selected-source view. The implementation bytes change; this feature preserves
the default semantic/source values, not historical file hashes.

`qualify_current_survey_other_disability_completion` uses original ASEC DESIGN
weights, age, current WSAL employment income, and current SEMP self-employment
income. Known positive and source-qualified zero amounts train the maintained
zero-aware regime-gated QRF. Unknown targets are excluded explicitly; unknown
required predictors on an otherwise eligible donor or recipient refuse rather
than silently truncate the sample. No prior wages, PUF income, geography,
poverty targets or guessed disability-status predicates enter the fit.

The evidence records every source reason's count and DESIGN mass, including
affirmed recipients whose dollar amount remains unresolved, and their mass
relative to known-positive donors. Excluding these observations can bias modeled
incidence. The diagnostic does not authorize a correction. Transport from ASEC's
2024 calendar-year dollars to ACS's rolling prior-12-month income concept also
needs scientific qualification.

The graph has eight nodes when recipients exist: full-source CREATE, model
columns, donor FILTER, recipient projection, QRF fit, QRF apply, receiver
keep-all version, and attachment. The no-recipient path has four nodes and no
model fit. Donor and recipient populations are separate; the apply node never
sits on the ancestor of the donor FILTER. A visible version barrier precedes
the canonical write. Fit cache invalidation is currently conservative because
source-custody parameters include preparation/recipient bindings; this fragment
does not claim fit reuse across changed receiving selections.

One draw per original ACS person is copied exactly to its two clones, with
joins checking original ID, native ID, source channel and clone indices. Draws
are ordered by ascending original person ID. Receiving-row permutations do not
change assignment, but the sequential QRF stream is not invariant to changing
the original recipient roster. The later AGI `clone2` extension needs its own
reviewed variable-clone attachment contract before these features can integrate;
this fragment deliberately continues to require exactly clones 0 and 1.

## Explicit graph storage projection

Frame columns are `disability_benefits`, `survey_other_disability_known`,
`survey_other_disability_reason`, `survey_other_disability_model_applicable`,
`survey_other_disability_canonical_known`, and `survey_other_disability_value_origin`.
An ACS modeled zero remains source-unknown, with origin `modeled`. Known ASEC
positive/zero values have origin `observed`; unresolved source values have origin
`source_unresolved`; under-15 values have origin `outside_model_universe`.

The graph currently does not support nullable `Float64` or `Int16` columns.
Original source reports are therefore retained in the declared
`microcosm.us.other_disability_attachment@1` artifact. Its `source_reports`
section preserves selected-original ASEC axes, dtypes, numeric backing bytes,
null masks and string values; `artifact_only_source_fields` enumerates fields
that are absent from the Frame projection. Its physical seal binds the original
source representation independently of the canonical output. Graph inspection
can locate those reports at `survey_other_disability.attach`; this is an explicit
projection, not a claim of unchanged report storage in Frame columns.

New canonical storage is float64 with NaN for unknown. An incumbent float32 or
float64 column can be replaced only in the new version and only if every known
or modeled value round-trips exactly, including signed zero. Unsupported dtype,
report ownership collision, or lossy conversion refuses. No unknown is filled
with zero for compatibility.

## Integration boundary

`_CompletionBoundary` is private borrowed custody, not an issuer. The calling
host retains and validates the genuine preparation **and** the actual receiving
run before graph/store I/O and after final I/O, including required replay, with
pure checks last. It supplies the receiving version and exact typed terminal
artifact binding. A detached Frame, JSON receipt or fitted model confers no
source authority.

This change supplies an optional fragment and registry; it does not switch on a
production host, modify default enrichment, or integrate later PUF placement.
Invented fragment tests use a genuine invented source owner and an explicitly
invented receiver. Host integration and actual-data qualification require
separate review and bounded admission.
