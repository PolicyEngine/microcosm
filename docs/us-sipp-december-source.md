# Authenticated December SIPP evidence

`microcosm.build.sipp_december_source` authenticates a fixed 2023 SIPP parent and
reconstructs December household, vehicle and bank measurements with the actual
[pure measurement API](us-sipp-measurements.md). It lives outside `us_runtime`,
which has an inherited eager engine initialization. This implementation neither
imports a country engine nor fits a model, maps recipient variables, transfers
years, computes policy outcomes or changes the six existing SIPP consumers.

Only invented-fixture implementation checks are authorized here. A genuine
`first_32_ssuids` run needs separate review and approval; `all_ssuids` is a closed
API option, not permission to scale. No dataset, calibration, national/CD
acceptance or launch result follows from these software tests.

## Public boundary and closed source

Use `produce_sipp_december_source(parent_path, store, selection=...)` with an
existing `microcosm.graph.store.ContentStore`. Read through
`load_sipp_december_source(parent_path, store, receipt_key,
expected_selection=...)`. The reader authenticates a fresh parent snapshot and
reconstructs expected membership, measurements, keys and bytes before returning
`SippDecemberSource`. Caller receipts, locators and `SippSource` instances are not
authority. There is no public override of the source pins.

The reviewed code pins the 3,726,010,471-byte parent SHA-256
`5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2`,
source revision `21280dca5995e978d706740a8a4b9b7860cfd7b6`, master SHA-256
`b62876a0763a0b45f2c4873b055b35c264d3a67640b8a8b3590b650aa1db679e`,
registry SHA-256
`085658b4a9c8cba0e59624a64ae998c3e07d63ceba2243ab92904a0bef5f8402`
and dictionary SHA-256
`0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f`.
The source name is `sipp_2023_pu2023`; survey 2023 refers to reference 2022.
Mutable reports and runtime network requests supply no authority.

The projection contains exactly sixteen lexical fields in this order:

```
SSUID PNUM MONTHCODE ERESIDENCEID WPFINWGT
THEQ_VEH AHEQ_VEH THVAL_VEH AHVAL_VEH THDEBT_VEH AHDEBT_VEH
TVEH_NUM AVEH_NUM TVAL_BANK AVAL_BANK TAGE_EHC
```

The original is opened with nonblocking/no-follow flags and checked as a regular
file by descriptor. A bounded, hash-checked copy enters a private owned snapshot;
source size and descriptor/path identity must agree before and after copying.
The actual graph `source_content_key` verifies that persisted snapshot, using its
closed source name. This helper is used only on the owned regular file, because
it is not a secure nonblocking opener for an untrusted original.

## Parsing and complete December selection

The bounded binary parser rejects physical records above 1 MiB, UTF-8 BOMs,
invalid UTF-8, blank records, NULs and embedded carriage returns. LF and CRLF line
endings are explicit. The pipe reader uses `csv.QUOTE_NONE`: quotes are literal
characters. Its field-size limit is recorded in the producer contract and checked
again before publication. Header names must be nonempty, unique and include all
sixteen names. Every physical row must have the full header width, including
columns outside the projection. No short-row padding or extra-field recovery is
permitted. All retained tokens remain strings; an empty field remains `""`.
The omitted fields' meanings are outside this validation scope.

Invalid or missing `MONTHCODE` anywhere refuses. Valid months 1–11 are counted
and discarded before the December person/residence audit. Their person keys are
outside this producer's scope. By contrast, the standalone bank helper validates
all supplied person/month keys before skipping valid non-December rows.

The complete December roster passes through actual `december_households` before
selection. Duplicate normalized person-months, missing/invalid residences or
person keys anywhere in December refuse, including records beyond the first 32.
The original lexical tokens are preserved. Person order is raw SSUID string then
integer PNUM; household order is raw SSUID string then integer ERESIDENCEID.
Numeric-looking SSUIDs use plain string sorting.

Only canonical household-key tuples enter the selector. `first_32_ssuids` takes
the first min(32,n) SSUID strings and all their December people and residences;
`all_ssuids` takes all. Values, flags and weights never select households.
Unknown, conflicting, under-age and zero/unknown-weight members remain present.
Selected households are reconstructed with copied fields THEQ_VEH, THVAL_VEH,
THDEBT_VEH and TVEH_NUM, plus bank measurements for every selected person.

The fixed parent must reconcile to 39,513 December people, 17,646 residences,
17,314 distinct SSUIDs, 318 SSUIDs with multiple residences and 29 people with
numeric zero December weight before selection. These are prior source-audit
assertions, not output claims from this implementation. A mismatch refuses.

## Values, status and deterministic evidence

Vehicle copy agreement and each member's December weight use the existing API.
A missing weight makes the household mean unknown while preserving its full
membership. A valid zero participates in the mean. Person and residence weight
summaries retain separate known denominators, zero/unknown counts and sums.
No monetary weighted totals or effective-sample claims are computed.

Bank `TVAL_BANK` includes CDs. Its separate `sipp_december_bank_assets_v1` rule
requires a valid amount, monthly age at least 15, applicable flag and no
contradiction for a known value. Unknowns remain measurements, and flag 9 retains
unresolved component-allocation quality. A usable edited recode is not observed-only
or eligible for fitting merely because it has a value. Reference-year-end stocks
remain nominal 2022 dollars; vehicle counts retain reference-period ownership.

Canonical serialization enumerates the supported source, measurement, observation,
raw-field, membership and weight fields explicitly. Finite Decimals use
`format(value, "f")`, preserving encoded scale and sign without float conversion
or normalization. Nonfinite Decimals refuse. Sums and means use the existing
fixed 80-digit arithmetic context, independent of caller precision, rounding,
traps and exponent settings. Original lexical tokens remain separately available.

The receipt contains exact source/code/dependency/grammar identities, field order,
selection and membership digests, counts, measurement digest, projection identity
and aggregate status/flag/quality/lexical diagnostics. Reasons are nonexclusive
counts. It contains no individual source keys or amounts. The private lexical
payload and returned immutable measurement tuples necessarily contain them and
must not be printed in an aggregate report.

## Existing content store and verification

Versioned domains of actual graph `canonical_json` and `sha256_domain` bind the
producer, projection, receipt, membership, measurements and run information.
The producer binding includes closed source/master/registry/document pins,
parser/selector/serializer/rule contracts, exact API implementation module hashes,
dependency versions and `source_hash`. It is rechecked before publication.
Projection and receipt keys additionally bind their expected bytes and lengths.
Receipt identity excludes itself and runtime information, avoiding a hash cycle.

Both objects use actual `ContentStore` writes with `node_key=producer_binding`
and `verify_existing=True`. The latter validates an incumbent; it does not compare
the proposed payload. This producer therefore compares independently reconstructed
expected bytes with every incumbent and with post-write readback. A forged but
self-consistent receipt, changed projection or different payload under the same
caller key refuses.

Before generic store metadata/JSON/byte loaders run, the adapter copies exactly
`meta.json` and the expected payload to an owned object snapshot. A third entry
is enough to refuse a larger tree. Metadata is bounded to 1 MiB and payloads to
64 MiB. Links, nonregular files including FIFOs, extra paths, oversized/growing
files, unstable identities, wrong kinds/node bindings and unexpected payload sets
refuse. Generic unbounded reads operate only on these owned bounded files.

All cooperating producers and readers hold the fixed nonblocking exclusive
`sipp-december-source-v1/producer.lock` in the store. Each object is atomically
published by the existing store; the receipt follows the projection as the
completion reference. There is no cross-object transaction. An interruption may
leave an unreferenced projection or, after receipt publication, an artifact whose
run-information write did not finish. No failed call returns a verified result;
a future reader must reconstruct it. Existing or failed evidence is not deleted.
Hostile mutation of owned snapshots and writers outside the namespace lock are
outside the local single-writer contract. Hashes are not external witnessing.

The returned receipt does not install a graph kernel. A future importer must bind
the authenticated full-parent content key, projection, selection and producer
identity and use this verified reader; a plain raw-byte load is insufficient.

## Cost accounting and tests

One producer copies/hashes the original once, checks its persisted source key,
then parses the snapshot twice at chunk sizes 8192 and 4096. It drops the first
measurement graph before the second reconstruction and compares exact bytes.
This is a same-code chunk-consistency check, not independent semantic verification.
A reader separately copies/hashes the original, checks the snapshot key and parses
once. Producer plus one reader therefore uses seven parent byte traversals,
three parses and two snapshot writes. Success deletes only its own parent
snapshot. Failed snapshots remain with sanitized reason/count information.

Preflight requires 8 GiB estimated available RAM and additional free disk of
`parent_bytes + 4 * 64 MiB + 1 GiB`. Retained own snapshot count/bytes and actual
free disk are recorded. Occupied bytes are already reflected in free disk and
are not subtracted again. This is an operational resource check, not a forecast
of peak use. Run information has a separate content-derived key and records
phase durations, process peak RSS with units, available-memory estimate, disk
context and actual loaded Microcosm module paths/hashes. It is private diagnostic
evidence and does not alter canonical source identity.

The flat test modules `test_us_sipp_december_source.py` and
`test_us_sipp_december_bank_assets.py` use only invented parents and private
test pin replacement. They exercise actual store writes/readback and secure-file
boundaries, forged bytes/metadata, whole-SSUID selection, unknown/zero weights,
bank eligibility, fixed Decimal behavior, locking and interrupted publication.
The original measurement regression module remains unchanged. A separately
guarded test receipt records source/model/network refusal and loaded module blobs.
