# SIPP nominal December-2022 equity: first source-named fit

This consumer turns one already accepted SIPP December-2022 source artifact
into a donor Frame, fits the real weighted sign-regime QRF over a single
predictor, and draws for a caller-supplied recipient population.

## Scope, and what this is not

The target is `sipp_all_vehicle_equity_dec2022`: the published all-vehicle
equity recode at residence/household grain, signed, in nominal December 2022
dollars. It is a **known-target subset** defined by an explicit admission
rule, not a US population estimate, not a PolicyEngine input, not a 2024
stock, and not a validated year transfer. Donor weights are SIPP source
design weights; they are not calibrated recipient weights. A request for any
other concept, or for 2024, refuses before any source access — the year
bridge remains outstanding work.

## Admission

Every December residence and its complete member roster is reconstructed
first, and its mean `WPFINWGT` computed over all members, before any target
or weight filtering. Only then does the target-specific rule apply:

1. a known, household-copy-agreed equity measurement;
2. a known complete-member mean weight;
3. a strictly positive mean weight.

Nothing is reweighted. An unknown weight stays unknown, never zero. Flag 9
(`component_derived_allocation_unresolved`) is the genuine source's only
observed `AHEQ_VEH` flag; admission retains those residences and never
relabels the quality as reported. Vehicle count is a diagnostic, not an
admission requirement or a predictor. Excluded residences are removed
*before* donor CREATE, so the FIT receipt's own positive-weight exclusion
count is zero; every one of them remains in the full coverage report with its
mutually exclusive reason.

For the accepted all-SSUID artifact the packaged contract expects 17,588
admitted residences out of 17,646, with 20 known-equity zero-weight and 38
conflicting-equity residences retained outside the fit.

## Modules

- `microcosm.build.sipp_equity_contract` — the packaged pin contract
  (`sipp_equity_contract_v1.json`) and the concept/period refusal.
- `microcosm.build.sipp_december_readback` — a public sibling of the December
  producer that reads the two closed objects without rereading the 3.7 GB
  parent. It reuses the producer's private `_bounded_object`, `_counts`,
  `_membership`, `_measurement_digest` and `_selected_ssuids` helpers as an
  explicit, tested internal dependency. The producer module itself is byte-for-byte unchanged,
  so its accepted historical output identity still holds. The historical
  producer binding and contract are validated **by comparison**; the receipt
  and projection key formulas are recomputed only as a consistency check.
- `microcosm.build.sipp_equity_graph` — the prepared read-only source store,
  the `sipp-december-equity-v1` Frame codec, the three kernels and the graph
  builder.

## Graph shape

    donors (CREATE)  ──> fit (QRF train) ──┐
        │  donor_support + donor_evidence   │ model artifact
        ▼                                  ▼
    recipients (CREATE, owns              draw (QRF apply)
      roster_household_size)                  ▲
        └──> support (owns household_size) ───┘

The support node reads `roster_household_size` and the typed donor-support
artifact and **owns** `household_size`, copying values only after every
requested size passes the support check. DRAW's Slice is exactly the model
predictors, so the compiler derives a real predecessor edge from column
ownership — not from lexical node ordering. A refusal inside the validator
rejects the node and aborts the run before DRAW, in cold, fresh and warm
paths alike.

## Model configuration

Two trees, seed 0, `zero_atol=0.0`, `max_samples_leaf=None`, predictor
`household_size` only, no extrapolation and no default fill. Exact source
`Decimal` weights are preserved in Frame metadata beside the checked float64
conversion performed at CREATE, because typed Frame weights are float64.

## Genuine execution

Not included here. Everything above is verified against invented fixtures
only. A separately reviewed private rung may fit the full admitted subset and
draw a small explicit recipient fixture; that fixture would be an engineering
recipient population, not a US population estimate. Engine mapping, the 2024
bridge and scientific validation remain separate outstanding work.

## Consumer contract identity in graph keys

The packaged contract is a JSON resource, so no module `source_hash` covers
it. Its identity — a domain-separated SHA-256 over the canonical contract
document — is therefore declared in the `contract_identity` param of the
donor, recipient and support nodes, which `node_key` binds through
`normative(node)`. Each of those kernels compares that declared identity
against the contract it actually loads and refuses `CONTRACT_IDENTITY_MISMATCH`
on a mismatch, so a hand-built node cannot execute under a contract other than
the one it declares.

The consequence is the one the root review asked for: a changed source pin,
admission policy or model setting moves all three keys, and through the
column and typed-artifact edges it moves fit and draw as well. A cache entry
written under an older contract cannot satisfy a node declaring a newer one.
The FIT and DRAW nodes keep the shipped QRF kernels' own params unchanged;
they invalidate transitively, not by declaring the contract themselves.

## Supplied recipient household sizes

Recipient CREATE derives and owns `roster_household_size` from the linked
person roster. A caller may also supply a `household_size` claim; it is
compared to the derived roster **exactly, before any cast**, and dropped once
it agrees. Casting first would truncate a fractional 1.5 to 1 and let it agree
with a one-person roster. A fractional, nonfinite, boolean or non-numeric
claim is a clean refusal (`RECIPIENT_SIZE_NOT_INTEGER`,
`RECIPIENT_SIZE_NOT_FINITE`, `RECIPIENT_SIZE_TYPE:*`); a disagreeing but
otherwise valid claim is `RECIPIENT_SIZE_CONFLICT`. The derived roster remains
the membership identity in every case.

## Boundary hardening

- The prepared source store is copied object by object, bounded by each
  object's pinned byte length and a small cap on `meta.json`, rather than by
  an unbounded `copyfile`.
- The closed-roster check refuses a symlinked `objects/`, an object parked
  under a shard other than its key prefix, and a duplicate.
- Readback refuses a scratch namespace equal to the store root, not only one
  beneath it.
- The signed target has its own float64 conversion, which refuses a nonfinite
  value while admitting negative and exact-zero equity. The strictly-positive
  predicate stays where it belongs, on the design weight.
- The three kernels declare the `numpy` and `pandas` versions their behavior
  depends on, so a behavior-bearing upgrade moves their implementation hash.

## Where the evidence lives, and why

`ContentStore` replay reconstructs a population from its tables, weights and
strata. It does **not** persist `Frame.metadata`, so everything the codec
attaches to the donor Frame — the coverage report, the exact `Decimal` weights
and targets, the household and person lineage — is gone the moment a run is
served from cache. Two surfaces carry it instead, and both survive a warm hit
in a fresh process.

**The donor receipt** carries the complete coverage report and an
identifier-free admission summary: the rule id, the denominators, the admitted
and excluded counts, the mutually exclusive reasons, and the expected-versus-
observed reconciliation. Receipts are rendered verbatim into the explain report
(`explain.py`), so nothing that identifies a residence or a person may enter
one. The receipt also carries the SHA-256, byte length and record counts of the
evidence artifact, so the two are bound.

**The `donor_evidence` typed artifact** carries what the receipt may not: each
admitted residence with its SSUID, December residence id, household size and
its exact `Decimal` source weight and signed target as strings, plus the full
ordered person lineage, plus the whole admission record including each excluded
residence's own key. It is declared as a second `ArtifactOutput` of the donor
CREATE node in the production graph builder, so the executor stores it beside
the node's outputs and replays it on a hit. It is private evidence: the explain
report shows only its key and digest. The support artifact the validator reads
carries the same digest, so the two donor outputs cannot drift apart, and the
validator's own receipt records it.

The lineage digest is recomputable from the artifact alone — the donor records
and the person list rebuild exactly the document it hashes.

## Expected versus observed admission

The packaged contract pins 17,588 admitted, 20 known-equity zero-weight and 38
conflicting-equity residences out of a pinned roster of 17,646. Those were
inert: nothing compared them to what admission actually produced.
`reconcile_admission` now compares four numbers — the denominator, the admitted
count, the zero-weight exclusions and the unknown-target exclusions — and
refuses `ADMISSION_RECONCILIATION:<fields>` on any disagreement, before the
donor Frame is built. The result, expected beside observed, is recorded in the
receipt. Because the three pinned counts sum exactly to the pinned roster, and
the readback has already refused a roster of any other size, agreement on those
four numbers leaves no room for an exclusion reason the contract does not name.
Invented fixtures carry their own expected counts through the same field, so
the reconciliation is exercised rather than bypassed.

## `selected_ssuid_digest`

The contract pinned it and nothing checked it. The readback now recomputes it
with the producer's **own** selector and key formula — `_selected_ssuids` and
`_key` over the reconstructed roster's identity tuples — and compares the
result to both the packaged pin and the accepted receipt's own field. The
recomputation is only sound under the all-SSUID selection, where the
reconstructed roster is the producer's full roster, so a narrower selection is
refused (`SELECTION_NOT_ALL_SSUIDS`) rather than compared incomparably.

## The recipient population's column constraint

`CREATE` must load exactly its declared columns: the executor compares the
returned Frame's non-structural columns to the node's `outputs` and rejects any
missing or extra one. The recipient node declares only
`roster_household_size`. A recipient population may therefore carry **only**
its entity ids, its person-to-household membership, its weights, and
optionally an exactly agreeing `household_size` claim, which is compared and
then dropped. Any other column — a region label, a diagnostic — rejects the
node. Adding one means declaring it as an `Owned` output of a node, not
smuggling it through the source.

A household with no members is refused before the kernel: `csv-tables` builds
the Frame first, and the Frame requires the group ids to be exactly the
distinct membership values.

## What the refusal types do and do not promise

`EquityGraphError` and `ReadbackError` are each raised only by their own
module, and their messages are closed codes rather than paths or payload bytes.
Neither wraps what happens beneath it: a `ContentStore` integrity failure, a
`StoreMiss`, or a producer-helper `SourceError` propagates as itself, with that
layer's own message. Code that needs a single exception type must catch those
too — the tests do.

## Scratch

The codec's bounded-copy namespace is a private temporary directory, created
and removed inside the codec call. It is outside the source store — the
readback refuses a scratch equal to or beneath the store root — and it writes
nothing into any caller-owned directory.
