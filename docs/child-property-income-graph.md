# Child property-income graph candidate

This source-only fragment connects the source-qualified ASEC age 15–17 donor
projection to the real paired empirical model and original-coordinate draws.
It depends on separately reviewed numerical, source-qualifier and generic graph
adapter revisions. Its tests are authored and unrun. It supplies no production
support default and is not connected to the default financial, tax or PUF host.

Six declared nodes expose the complete model-support schema:

1. `child_property.donors` creates a private donor support population. Original
   source coordinates are separate from its minted index IDs. The explicit
   original-household design-weight column enters the fit; unit index weights
   exist only to satisfy the support Frame contract.
2. `child_property.recipients` creates a private population containing every
   selected original recipient, its exact coordinates, source age, component
   knownness, eligibility and reason. Keeping this population nonempty permits
   zero eligible children without a sentinel or an empty weight vector. It
   emits the separate recipient projection and complete scenario artifact.
3. `child_property.fit` is the reusable typed empirical-fit node. Full-source
   donors are independent of target sampling and clone expansion. The donor
   projection excludes the selected preparation digest; changing target
   sampling can change custody keys but not the fitted paired donor law.
4. `child_property.draw` is the reusable keyed-draw node. Its declared Boolean
   eligibility input selects the original recipients internally. The full
   source coordinate plus `child-property-v1` and `pattern`/`donor` suffixes
   determine uniforms. Clone IDs, row order and scenario hashes do not.
5. `child_property.attach` reconstructs the real model and draws from the
   retained source projections, verifies exact typed artifacts, and fills only
   jointly unknown eligible O/D cells on both initial clones. It writes
   `child_property_completion_status` and `child_property_imputed`; a model zero
   has status `imputed_zero`, never source nonreceipt. Existing source-knownness,
   all nonrecipient amounts, other columns, groups and geography are preserved.
6. `child_property.verify` requalifies the source again and compares all table
   values/dtypes/missingness, effective weights and strata. It emits typed
   verification evidence that still requires the host's complete-population
   check, including owners and the ledger.

`ChildPropertyBoundary` requires the actual authenticated preparation, retained
receiving `Population`, explicit options, typed host edges and exact artifact
pins. Its `require_parent` callback must recheck the host's issued parent and
return that exact Population. The fragment checks the callback's code, defaults
and closure identity, retained source issuance, complete source projections,
configuration and parent physical seals before and after callbacks. The real
source qualifier executes on every fresh check; serialized projections alone
cannot create a boundary. A detected execution/materialization failure revokes
the boundary for future checks.

The country host must call `verify_materialized` on both cold and required
replay. Required cache hits skip kernel bodies. This method requires the actual
receiving population, both support populations, all verification-node input
`ArtifactValue`s and its output verification `ArtifactValue`. It re-executes the
real empirical fit and keyed draw independently of those artifacts, compares
all three complete populations with maintained replay equality, repeats source
and parent verification, and checks final output seals. Caller-owned artifact
descriptors and the verification output are detached before the first owner
borrow, then compared after the first and final borrows; their complete fields
and exact support-mapping keys must remain unchanged. The reconstructed artifact
and verification contracts are checked again after the final callback. The returned document
is descriptive evidence; it is not an issued financial, tax-complete or release
handle.

Root-owned host integration remains explicit: insert after the complete survey
clones, atomic geography and current-property attachment, before property-tax
rebasing/gates and PUF. The receiving version must have a structural base so
the two amount writes are declared rewrites. Supply the real preparation,
allocation, frame-context and ordering edges/pins; retain the actual parent
owner in the callback and in the common host's live/source/replay seals. Bind
the actual run manifest's producer keys, implementation identities and source
identities when resolving artifact descriptors. This fragment reconstructs
payloads and checks typed paired producers; it does not independently issue
the common host's parent or validate an arbitrary claimed run manifest. Bind
the checked completion evidence into the tax gate without changing its
all-person completeness requirement. No unresolved adult class is completed by
this child candidate. The disabled/default path must remain unchanged.

Tests separate pure descriptive fixtures from actual owner execution. Pure
fixtures test the real numerical law and graph patch, exact large IDs, clone
joins, zero stress, zero eligible children, known-incumbent refusal, exact
artifact bindings and drift anywhere in the receiving state. Actual-owner tests
reuse tiny authenticated survey fixtures whose sole teenager donor is absent
from the selected target sample. They run an invented receiving prefix and the
actual six-node fragment, including cold/required replay and first/final-owner
mutation controls for receiving values, artifact replacement, support-key
addition/removal, and verification payload/producer metadata. Refusal permanently
revokes the boundary. They do not claim actual financial/tax/PUF host acceptance.

The accepted receiving Frame profile has no explicit links; the maintained
whole-population comparison refuses nonempty links rather than dropping them.

Source projections and draw/completion JSON are private and bounded at 64 MiB;
selected originals are capped at 600,000. The generic model also has its 64 MiB
limit. JSON size checks follow object construction. No measured bytes-per-row,
peak memory or national capacity is claimed. This first implementation repeats
source requalification and model reconstruction at validation boundaries; a
separately reviewed sizing ladder and possible compact artifact representation
are required before a large candidate. Source-only tests cannot establish that
runtime performance or scientific donor support is adequate for release.
