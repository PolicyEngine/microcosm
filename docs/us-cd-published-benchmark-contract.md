# US CD published-benchmark contract

`microcosm.build.cd_benchmark` is the executable form of the approved
congressional-district published-benchmark protocol
`us-cd-published-benchmark-v4`, SHA-256
`f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea`, 86,626
bytes. The approved bytes ship as a packaged resource beside the package and are
hash-verified every time they load.

## What this is not

The approval in force covers **invented-fixture implementation only**. Candidate
scoring and production ledger registration are not approved, and this package
does neither. It reads no candidate, population, HDF5, ACS or Census artifact; it
downloads nothing; it creates and appends to no exposure ledger; it takes no
lock; and it emits no release, eligibility or publication verdict. `evaluate_scope`
refuses to treat implementation approval as scoring approval, so a run under the
in-force approval reaches `scope_incomplete` with `protocol_unapproved` even
when every measured block conforms.

A `benchmark_conforming` verdict is operational published-benchmark conformity
for the stated scope. It does not prove population truth or policy accuracy, is
not a hypothesis test of truth error, and is not independent-source evidence.
`BenchmarkEvaluation.claims_not_established` carries that list at runtime, and
the class has no field that reads as a release verdict.

## The document is the source of truth

Thresholds, vocabularies, payload rosters and pinned digests are compiled out of
the approved bytes rather than retyped. `compile_protocol` refuses a float where
the document writes an exact decimal string, and `canonical_protocol` refuses to
return a protocol whose closed reason or exposure vocabulary differs from the
code's. So the code cannot drift from the contract without a test failing.

Paths named inside the document — the legacy design parents, the reference
authority entries, the source-code locators — are locators recording what was
inspected when the contract was written. Nothing here opens them.

## Numerical rules

Bounds are exact `Decimal` values and comparisons are the document's inclusive
`<=`. Measurements convert to `Decimal` exactly, so nothing is inflated on the
way in, and all ratios evaluate in a fixed 120-digit context.

| Block | Rule | Bound |
| --- | --- | --- |
| `population_total` | relative point error | 0.02 |
| `occupied_household_total` | relative point error | 0.03 |
| `age_distribution` | per bin / total variation | 1 pp / 0.03 |
| `income_distribution` | per bin / total variation | 2 pp / 0.05 |
| `tenure_distribution` | per bin / total variation | 3 pp / 0.03 |

A zero reference requires a zero candidate for point conformity and leaves the
relative discrepancy unavailable rather than substituting a ratio. Negative or
nonfinite masses, and booleans, are `invalid_reducer_input`: no absolute value,
clipping, renormalization or drop.

Each distribution carries a protocol-constructed `unclassified` residual bin with
a reference share of exactly zero — a protocol construct, not a publisher cell.
`sum(m_i) + U` must equal `D` exactly; a partition that does not close is
`invalid_reducer_input`. Known bins are never renormalized to absorb residual
mass, the residual bin counts in both the per-bin bound and the total variation,
and a residual fraction above 0.005 adds `unclassified_fraction_exceeded` while
the measured discrepancies are retained. Bin edges are lower-inclusive and
upper-exclusive with `None` meaning infinity, and candidate monetary amounts are
not rounded.

## Reference precision is a separate axis

Precision never widens a point bound and is never a test of an unknown true
value. Classification of published cells is not reimplemented: it calls
`microcosm.build.us_runtime.cd_reference.classify_acs_value`, so sentinels,
annotation precedence and annotation/sentinel conflicts stay owned by the
shipped adapter, whose own class string is retained on every result.

A controlled estimate is its own precision class. The point comparison stays
valid, no variance or nonsampling-error claim follows, and — this is the
important part — no adequacy is asserted either. A zero estimate leaves the
relative margin undefined and is recorded as unavailable, not as a perfect one.
The age block reads its published percentage margins as printed; income and
tenure use the frozen descriptive subset-share approximation
`100*sqrt(Mx^2-(X/Y)^2*My^2)/Y` with the documented plus fallback on a negative
radicand. A missing or controlled input yields no approximation at all.

## Origin support

Contributions are summed by authenticated original source-household key before
any count, effective sample size or maximum share, so clones and aliases of one
source household count once. Denominators carry all three bounds: at least 1,000
positive-contribution origins, Kish effective sample size at least 500, and a
largest origin share at most 0.005. Positive published bins carry the carrier
floor of 50 origins and a maximum share of 0.05; a zero candidate bin mass fails
the carrier floor without dividing by zero, and the residual bin has no carrier
floor. Any in-scope record without an authenticated origin — zero-weight records
included — is `unresolved_source_lineage`.

A satisfied floor is not an independence or unbiasedness result and a breached
one is not proof of bias. An origin key and an effective sample size do not
establish independent households across surveys or independent imputation donors.

## Origin keys

`SHA256(UTF8("microcosm.statistical_source_household.v1") + NUL + payload)`,
where the payload is canonical UTF-8 JSON of four ordered typed pairs. The two
arms are named separately so a survey vintage cannot be filed as an income
cohort year:

- ACS: `["string","ACS_PUMS"]`, the member object, the **survey vintage** as a
  JSON integer, and the **raw `SERIALNO` string** with no normalization,
  trimming or coercion.
- ASEC: `["string","ASEC"]`, the member object, the **income cohort year**, and
  the **native integer `H_SEQ`**.

Member identity comes from the authenticated source manifest. Physical paths,
recompressed archive hashes and clone or cache filenames are not identities, and
a generated `source_household_id` cannot replace this key.

## Source universe and the two bridges

ACS and ASEC membership come from the adapters the contract pins by digest;
`verify_pinned_adapters` checks the installed files against those pins, because a
changed adapter is a scope review rather than a pin to refresh in passing. Three
refusals stay distinct:

- an out-of-domain source code makes the adapter refuse the whole batch —
  `source_domain_refused`, with no fabricated classification or unclassified mass;
- membership or a denominator that cannot be established is
  `denominator_unknown`, regardless of mass. **A retained ACS `TYPEHUGQ=1` row
  with `NP=0` and no approved bridge is this case**: the pure adapter leaves
  occupancy unresolved and asserts no vacancy;
- an absent or unapproved required mapping is `mapping_unapproved`, again
  regardless of mass. Both may be retained together.

*Dropping* NP=0 rows as known vacant is different: `vacancy_bridge_missing`. An
unknown drop history is `unresolved_source_lineage` and cannot imply zero drops;
a known zero drop count waives nothing else.

The GQ weight bridge is an external required predicate. The shipped ACS loader's
policy of substituting `PWGTP` for a zero `WGTP` on a GQ household placeholder is
a representation policy, not observed occupied-HU weighting, and this package
does not adopt it: with no bridge injected, population and age report
`gq_weight_bridge_missing`.

## Approval, registration and the exposure ledger

The document requires each approval artifact to state APPROVE, the exact protocol
SHA, the reviewer identity and the approved scope, but fixes no byte format. This
package defines a compact one as an implementation detail; it does not amend the
contract. Activation needs three independent things: canonical protocol bytes, a
`protocol_registered` claim naming the same protocol, parent, reference
inventory, block definitions and approval digests, and an injected authority that
resolves one root and one peer APPROVE covering the requested action. A
registration naming an approval digest the authority does not hold is refused —
a ledger line cannot authorize itself.

There is no default and no production `ApprovalAuthority`, and no
`ProducerAuthenticator` ships at all. A filled binding slot is caller evidence,
never authentication: with no authenticator injected, every binding resolves to
`unapproved` however complete it looks.

Ledger records are validated in memory only. Sequences start at zero and stay
consecutive, `previous_record_sha256` is null only at sequence zero and otherwise
covers the complete previous record, each line must be exactly its own canonical
serialization plus one LF, and a partial or non-canonical tail is
`ledger_integrity_unknown` rather than a truncation back to a valid prefix. Each
attempt takes exactly one terminal record: a completion after an interruption, an
interruption after a completion, a second interruption and a duplicate completion
are all refused, so an interrupted attempt can never regain freshness by a
retroactive completion. Evaluation identifiers are lowercase canonical UUIDv4 and
must be unique across the ledger and every linked recovery history.

Exposure precedence runs in the document's order: incomplete history, reservation
or ancestry is `exposure_history_unknown` first; otherwise a nonreserved family
(S0101 population and age) is ordinary conformity with no holdout claim;
otherwise a reserved family with any earlier related attempt or external exposure
is post-exposure — a started attempt counts even if it failed or was interrupted;
only the remaining reserved first-attempt case may be fresh, and even then it is
same-source, never independent-source, evidence. The occupied-household total
carries both `B19001` and `B25003`, so exposure of either marks it.

## Where this package's reading fills a gap

Four things the document leaves open are decided here and named as
implementation choices rather than contract text.

1. **Approval artifact format.** Defined as
   `microcosm.cd_benchmark_approval.v1` with a three-member scope enum.
2. **Four ledger payload rosters** — `protocol_amended`,
   `evaluation_interrupted`, `external_exposure_recorded` and `ledger_recovery`
   — are described in prose, not enumerated. Closed rosters are defined in
   `ledger.py`.
3. **Binding-slot reason classification.** The document lists twenty-one slots
   without assigning a reason code to each; `binding.py` partitions them into
   identity, lineage, mapping and exposure so a missing slot reports its own
   axis.
4. **Per-bin effective sample size.** The 500 floor is required "for every
   district and relevant denominator", and only the carrier count and maximum
   share are named for bins, so the floor is not applied per bin. The per-bin
   value is still computed and returned as a diagnostic.

Two smaller readings: the adapter's `moe_open_interval` and `open_interval`
classes both map to the document's single `open_interval`, with the adapter's own
string retained; and a zero reference estimate makes relative precision
unavailable, which the document does not state either way.
