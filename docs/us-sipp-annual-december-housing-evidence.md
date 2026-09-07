# SIPP annual and December housing evidence

`microcosm.build.sipp_annual_housing_source` combines annual panel evidence with
an existing December housing projection in a private canonical artifact. Its
format is `sipp_annual_december_housing_evidence_v1`; its recipe version is
`sipp-annual-december-housing-evidence/1`. This is a bounded supplied-row codec.
It performs no source extraction, fit, draw, recipient transfer, charge
estimation, publication, or source authentication.

The implementation preserves all five accepted housing module files from
`bed98d54308e67ee5960a423934e43c2361bba88`. Their byte hashes remain bound by the
housing receipt. Annual module version `sipp-annual-panel/2` extracts the
December household calculation from the accepted annual module at
`02e7b2933035ae609187d1c715531ca300ed50d7`; aggregate diagnostics reduce those
same records. The existing diagnostic schema and scientific definition remain
unchanged. The accepted annual version 1 source and its historical receipt
retain their original identities.

## API and external expectations

The reusable annual calculation is
`annual_december_households(panel, source_sha256)` in `sipp_annual_panel.py`.
It returns ordered records with household, member and contributor-month keys,
annual availability and reasons, December mean member weight, and source
identity. An optional internal `state` allows diagnostics and the evidence codec
to reuse the already validated panel groups.

The evidence codec has two entry points:

```python
project(
    panel,
    *,
    housing,
    expected_housing_receipt_sha256,
    source,
    selection_binding,
    housing_rows,
)

decode_artifact(
    payload,
    *,
    expected_sha256,
    expected_source_binding,
    expected_recipe_sha256,
    expected_selection_binding,
    expected_housing_receipt_sha256,
)
```

`housing` is an actual `sipp_housing_projection.HousingProjectionResult` from
exactly the supplied `housing_rows`. Creation recomputes that projection and
checks its receipt, measurements, predictors and diagnostics. Both entry points
return `AnnualHousingEvidence`, whose `artifact_json` and `receipt_json` are
canonical bytes with separate SHA-256 values. The `households` property exposes
private joined identities; the `quality` property exposes count summaries.
The API returns bytes and does not introduce a content store or persistence
framework. A caller owns private storage and any later explicit codec consumer.

`SourceBinding` carries facts supplied from outside the artifact:

- `declared`: the frozen housing parent's `SippSource`, including source and
  public-document identities, survey year 2023 and income year 2022.
- `panel_stream_sha256`, `panel_stream_bytes`, `panel_stream_rows`, and
  `panel_stream_passes`: declarations matching both retained annual stream
  audits; exactly two passes are required.
- `index_december_person_rows`, `index_eligible_households`, and
  `index_eligible_ssuids`: declarations matching the annual identifier-index
  counters. Their ordering and relationship to selected IDs and stream rows
  are checked.
- `evidence_origin`: either `invented_only` or the default
  `supplied_rows_unverified`.

For `supplied_rows_unverified`, the declared parent data SHA and panel stream SHA
must agree. `invented_only` permits different digests so invented rows and an
invented stream audit can exercise the unchanged housing parent declaration.
The artifact records that distinction and the digest agreement result.
Neither origin authenticates supplied rows. A matching digest declaration,
index count or receipt is insufficient to prove a relationship to source bytes.
`authenticated_source_bytes_established_here` remains false. A future external
runner must establish the source chain and independently retain its evidence.

`SelectionBinding` requires the annual rule, annual limit, exact sorted tuple of
selected canonical SSUIDs, and housing mode. The annual rule remains
`first32_by_canonical_14_digit_lexical_SSUID_from_eligible_December_ID_index` with
limit 32. Housing retains `first_32_ssuids`, which establishes only that the
supplied projection contains at most 32 SSUIDs. Selected identities must match
both inputs; the selector declarations are retained separately. This codec
cannot authenticate the source-wide first-32 claim from supplied rows.
Decoding requires the expected selection and parent receipt independently,
alongside source and recipe expectations; it does not derive those expectations
from the payload it is checking.

## Measurement and exact membership

The annual measure follows persons who are at least 15 in December through all
their available months and earlier residences. It sums usable `TPTOTINC` only
when that contributor evidence is complete. Under-15 December members remain
in the roster and weight denominator; their amounts do not enter this annual
measure. Annual once-per-person lump-sum fields remain separate lexical
evidence and diagnostic quantities. The annual measure is not a declaration
of equivalence to ACS `HINCP` or ASEC `HTOTVAL`.

Join identity is canonical 14-digit SSUID, December `ERESIDENCEID`, and exact
ordered member/person-month identity. Multiple December residences under one
SSUID remain separate households. Shared annual/housing columns must agree
lexically, including their original flags and whitespace. A mover's earlier
months follow the person into the annual evidence for their December residence.

The annual calculation can describe a December membership defect as unavailable
evidence. The combined artifact uses a conservative stronger join condition:
**any December membership reason refuses creation or decoding** with
`ANNUAL_HOUSING_INCOMPLETE_DECEMBER_MEMBERSHIP`. This includes missing or
unpaired self references, omitted named members, inconsistent published counts,
roster disagreement and an occupied suppressed-tail slot. It does not manufacture
a complete member axis from such rows. This remains a published-file roster
check, not proof of physical household completeness outside the file.

Unavailable income coverage is different from a roster refusal. Missing
contributor months, blank income, flag 0, unknown status, unknown age or an
unresolved income universe retain a null annual amount and explicit reasons.
Existing monthly diagnostics may skip a child with unknown SSI receipt without
removing that child from the December roster or annual weight denominator.
Signed amounts and observed zero remain values. Allocation codes and code 9
remain distinct evidence; code 9 does not assert component allocation quality.
Unknown member weights keep the mean unknown, and zero-weight members remain
in the denominator. There is no positive-weight denominator substitution.

Every annual December row is partitioned into anchored membership or unresolved
coverage. Inactive rows and live rows without a usable residence stay in the
retained annual lexical rows and `join.unresolved_coverage`; neither is counted
as a joined or complete household member. The supplied housing row axis must
match exactly the anchored December person-month axis. Adding an unresolved row
to housing or omitting an anchored child refuses the join. Unresolved coverage
is disclosed separately from joined household counts.

December rent, utilities, tenure, assistance and predictor measurements retain
the frozen housing definitions and admission ladder. They are distinct from the
annual income evidence. Mortgage evidence does not become renter rent;
unresolved assistance stays unresolved; a lower-bound maximum age does not
become a known maximum. This codec applies no annualization or price rebasing.

## Closed private transport and count receipt

The private artifact retains original lexical annual and December rows,
ordered identities, source and selection declarations, stream audits, public
contract/document and module identities, both sets of derived measurements,
join coverage, and quality counts. Lexical preservation permits recomputation
without replacing unknown values with monetary zero.

The decoder verifies the expected artifact digest and canonical representation,
uses a closed schema, and reconstructs the annual panel and housing rows. It
then reruns the actual annual calculation and frozen housing projection. The
entire regenerated artifact must match the original bytes. Recomputing a local
digest after editing a derived amount, count, membership, implementation binding
or extra schema field cannot bypass that comparison. Duplicate object keys,
duplicate person-months, noncanonical identities, wrong types, nonfinite JSON
constants and malformed lexical evidence refuse. Source and selection facts
that cannot be reconstructed from selected rows must match the external
expectations described above.

Transport ceilings are 16 MiB per artifact, 20,000 annual rows, 2,000 housing
rows, 512 joined households, 2,000 unresolved December rows, and 32 selected
SSUIDs. JSON depth is capped at 32, total nodes at 2,000,000, and string length
at 4,096; individual lexical values are ASCII and capped at 64 bytes. These
are refusal bounds, not preallocated buffers or a forecast for genuine data.

The separate quality receipt carries content and implementation hashes,
selection/source declarations, and counts of availability, reasons, sign and
weight classes, housing categories and unresolved coverage. It contains no
individual identifiers, scientific amounts or weight sums. The full frozen
housing diagnostics are not copied into that public/count receipt, because
some parent diagnostics contain scientific totals. The private artifact and
annual aggregate diagnostics require private handling. Neither their contents
nor test fixture values belong in execution logs or public count reports.

`fit_admission` and `release_admission` are null. The artifact also declares no
selected fit weight, bound model input, transfer, charge estimate, draw, source
authentication, or launch readiness. These are evidence facts, not a new
parallel approval mechanism.

## Field union, verification and scope

`FIELD_UNION` and the recipe's `future_runner_field_union` enumerate 95 fields:
the annual 67 plus 28 additional December housing fields. This is metadata for
a future reviewed runner. The existing 67-field source receipt does not verify
the additional fields or their genuine lexical behavior. The seven separate
weight/reference additions are excluded; this task selects no fit-weight
candidate. There is no 256-SSUID extension, and `all_ssuids` is not a workaround
for either recipe's current 32-SSUID boundary.

The affected tests use invented rows and exercise the real annual and housing
implementations, diagnostic and parent-receipt parity, exact member linkage,
unavailable states, strict refusals and canonical decode/recompute. Verification
must record the final committed source head and tree, actual loaded files and
Git byte agreement. A compilation inventory alone is not loaded-source proof.

The bounded recovery execution uses the existing pinned Python 3.14.4
interpreter at
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python`.
Each parent explicitly uses `-B -X pycache_prefix=<fresh-absolute-absent-path>`;
each child also explicitly uses `-I`. Every process has its own absent prefix.
No environment-only bytecode setting substitutes for those command-line flags.
No dependency install, large speculative allocation, or genuine source,
store, checkpoint or projection read/stat/hash is part of this integration.

Invented success authorizes no genuine execution proposal. Root owns genuine
execution decisions after independent source/evidence review. Cross-survey
income harmonization, donor allocation/coverage policy, fit-weight choice,
recipient eligibility, estimand and any modeled full-charge contract remain
separate scientific work. This seam adds no targets, held-out analysis,
uncertainty expansion, prior wages, PolicyEngine calculations, data export,
fit, push, PR, merge, deployment or publication.
