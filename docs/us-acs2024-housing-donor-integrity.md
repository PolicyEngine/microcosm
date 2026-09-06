# ACS2024 housing donor identity

The ACS2024 reported-rent donor is separate from the legacy ACS2022 graph. Its
authenticated loader requires both packaged archive pins and the full source.
It copies each local archive to a private disk snapshot in bounded 1 MiB reads,
closes the writer, verifies the snapshot's bytes, and parses that same snapshot.
Original-path replacement during parsing cannot change its input. The snapshots
are read-only, stay inside a private temporary directory, and are removed on
success or failure. Temporary disk space must accommodate both compressed
archives; archive copies are not retained in RAM. This protects ordinary source
cache concurrency, not arbitrary hostile code running as the same user.

`Acs2024HousingDonor` is issued only by the loader or synthetic projector.
Its immutable receipt binds native projected table content, ordered columns,
index, dtype details, recipe, source identity, document references and exclusions.
The table remains inspectable, but changing it makes preparation fail before
sampling. Returned `provenance` dictionaries are detached views of canonical
receipt bytes: editing them does not alter the donor's evidence or convert a
synthetic fixture into authenticated data. Direct dataclass construction is
refused. This is an application integrity contract, not a security boundary
against Python code deliberately calling private issuance helpers.

`prepare_acs_2024_reported_rent` first takes an owned copy and verifies it against
that receipt. It consumes that exact copy, so later changes to the exposed donor
table cannot alter preparation. Its derived receipt binds the donor receipt,
training identity, ordered predictors, target, design-weight role and feature
schema. Metadata views are detached at this stage too.

Fit consumers must call `prepared.verified_training()` immediately at consumption.
It verifies the native training identity and predictor order and returns an owned
copy. The inspectable `prepared.training` attribute is mutable and does not itself
perform this check. Future graph or fit adapters must bind the derived receipt and
actual implementation closure; a bare QRF/DataFrame API cannot enforce an external
donor receipt on its own. The tiny QRF interface test exercises the verified
consumption method. There is no production recipient adapter in this slice.

The shared `table_values_sha256` helper is the existing canonical-scalar codec
factored from the US late-producer runtime. Legacy digest behavior is unchanged.
The new `native_table_identity` additionally binds an explicit closed dtype
descriptor, including pandas string storage and missing-value representation,
plus both axis names and column order. It does not construct a Frame or normalize
the input. It supports numeric, boolean and string tables with a flat index;
unsupported categorical, temporal or custom extension dtypes require another
explicit contract. Missing positions are preserved; distinct floating NaN payload
bits have the same logical missing identity.

The source-graph implementation inventory binds the actual shared helper module
in both affected scopes, `assembly_prepare` and `assembly_harmonize`. Its ordinary
imports are already-covered serialization, NumPy and pandas dependencies. The other
three source-graph stage module lists are unchanged. Legacy late-artifact receipts
still describe the codec and semantic controls without attesting implementation
bytes; they did not bind the original inline helper's bytes either. That inherited
limit is preserved here. New graph consumers must bind the helper's actual bytes;
the old late semantics, generator and pins are not upgraded by this donor change.

The sampler is unchanged: its joint sample is capped at 10,000 rows and reserves
up to 5,000 per target. With disjoint renter and owner universes that both fill
their share, rent preparation retains 5,000 renters after dropping tax-only rows.
Category levels come from the final rent-fit rows after this sample; their sorted
first level is the omitted baseline. The feature schema declares that a future
recipient encoder must refuse unseen categories. This does not establish a mapping
from reported contract rent to pre-subsidy rent.

The packaged authority records the preserved 2024 ACS Subject Definitions SHA-256
and page 16. That document is the publisher's description of respondent instructions;
it is not a separately authenticated 2024 respondent-help pamphlet or proof that
every respondent followed the instruction. These software receipts certify neither
predictive quality nor release readiness. Only invented archives and tiny fits test
this slice; no real archive, recipient population or calibration was run.
