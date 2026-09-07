# Durable CD candidate boundary

`microcosm.build.cd_benchmark_io` adds explicit local metadata and ledger I/O
beside the unchanged, pure `cd_benchmark` package. It implements the filesystem
part of the frozen US CD v4 contract. It does not measure a candidate, read ACS
references, calculate a discrepancy, approve a scientific mapping or authorize
publication. The implementation has been exercised with invented metadata only.

## Admission before exposure

The integrating runner must provide exact external pins for the candidate
admission manifest, every referenced producer/source/mapping metadata artifact,
and root and peer approvals. A pin includes an absolute locator, bounded byte
count and SHA-256. The loader checks regular files, every path component,
descriptor identity, exact bytes and canonical JSON. It rejects mutation before
decoding and rechecks loaded metadata before beginning an attempt. Hashes
provided by a candidate cannot select their own expected pins.

The coordinating admission format is `microcosm.cd_candidate_io.v1`: exact
protocol, candidate/population/producer identity, artifact/dependency roster,
all required v4 binding slots, income scope, and the history artifact ID. It is
an I/O format, not a replacement source receipt. Every slot also requires an
explicit producer adapter for its actual schema and scientific contract. No
such genuine adapter ships here. The invented test adapter accepts only its
invented schema. A generic envelope cannot turn missing universe, period,
income, geography, typed reducer or native-origin evidence into an admission.

Source-grain evidence must remain at source grain. A final-population adapter
must traverse the actual source/selection/clone/prune/export edges, authenticate
all in-scope rows including zero-weight rows, and bind the final content and
semantic snapshot. For example, a pre-transfer `survey_origins.clones` digest
cannot be relabeled as the final PUF-attachment population digest. The caller
must bind that later placement and its new owned columns separately.

Root and peer records use the existing `microcosm.cd_benchmark_approval.v1`
format. Exact expected bytes, reviewer identities, protocol digest and both
`candidate_scoring` and `production_ledger_registration` scopes are required.
The adapter does not authenticate people or external signatures; the reviewed
runner supplies those expected pins. An in-memory `ApprovalAuthority` alone is
not accepted by the filesystem adapter. No default approvals are supplied.

## One writer and durable exposure

The caller creates a dedicated directory and takes `DurableLedger.locked()`.
Acquisition uses a stable advisory `flock`, with an explicit wait between zero
and five seconds. The lock is held through registration, attempt, future
comparison, result, completion, saved prefix and quality envelope. The lockfile
is never unlinked. Directory/descriptor/path replacements refuse. Local locks
and hashes provide auditable provenance, not a tamperproof external witness.

Registration requires explicit action and exact approvals. A first registration
uses exclusive creation; repeated registration requires the exact existing
snapshot. Both the new file and its directory reach fsync. A missing, empty,
replaced or stale ledger cannot satisfy an expected history snapshot. Partial
or corrupt bytes are preserved and refuse the entire ledger: no truncation or
automatic recovery from a good prefix is provided.

The complete `microcosm.cd_exposure_audit_io.v1` history artifact binds the
candidate, expected ledger snapshot, original population ancestors, shared
tuning roots, reservation and external exposure audit. Its three completeness
fields must be true and authenticated by the integrating producer. The new
ledger cannot establish absence of unrecorded prior history by itself. Known
external derivatives must carry canonical v4 family roots; unknown or renamed
family labels refuse rather than become fresh. Either B19001 or B25003 exposure
marks their shared occupied-household total. Shared immutable survey input alone
does not imply shared tuning ancestry.

Before an attempt, the adapter rechecks metadata and the exact latest snapshot,
authenticates registration, and interrupts any prior unfinished attempt. It then
appends and fsyncs the attempt before yielding an identity-only permit. The
attempt includes both reserved families, all block-definition hashes, mapping
pins, candidate/population identity, roots, scope and pre-attempt exposure.
Candidate renaming, failed completion and interruption do not reset exposure.
Newly linked descendants inherit exposure from authenticated ancestor content
identities or shared tuning roots.

The adapter deliberately refuses protocol amendments, linked corrupt-ledger
recovery and external ledger events requiring a separate typed ancestry
adapter. An externally pinned complete history can carry prior external
exposures, but no generic event dictionary establishes its completeness.

## Results and interrupted work

A future comparator must execute only inside the permit context. This package
supplies no comparator and imports no metric evaluator. `finish` accepts only
nonnumeric verdict/reason metadata for the exact block roster. A failed result
retains explicit `uncomputed` states; completed results cannot contain them.
This format is an attachment boundary, not a replacement for the future
scientific quality producer and its detailed measured results.

The adapter exclusively creates and fsyncs the content-addressed result before
appending/fsyncing its completion. It then saves the exact ledger prefix through
completion and a quality envelope bound to both snapshots and every identity.
Each new directory entry is fsynced. Existing output files are read and compared,
never overwritten. A failed envelope write can be reattached from the same
immutable result and completion prefix without another comparison or attempt.
A torn output refuses attachment and is preserved for explicit investigation.

Leaving a permit without completion records an interruption. A raised original
exception survives an interruption-write failure; the unfinished durable attempt
continues to represent possible exposure. A hard process exit releases the OS
lock and leaves the attempt visible for the next authenticated writer. An
interrupted attempt cannot receive a retroactive completion.

## Remaining integration

Before any genuine CD comparison, implement and independently review the actual
producer adapters, complete all scientific mapping and scope admissions, provide
exact-action root/peer approval and external expected pins, and authenticate the
complete reservation/history audit. Wire the actual quality comparator and
reader to the permit and its snapshots. No new production ledger, candidate
scoring approval, genuine evaluation, national-reference claim or launch verdict
is created by this source change. B19001/B25003 and their derivatives remain
reserved. Protocol f7574a2a00734ce64ceb9c101ebc52469a176d37ee32b02c140d3ea2e6c568ea
and every threshold are unchanged.

Retries fsync retained identical files and their directory again; their
presence alone proves no prior fsync.
First-time registration cannot reuse any already-open ledger. The loader stores
its manifest pin separately from producer artifact IDs, rejects the reserved
historical `__manifest__` name, and rechecks the complete dependency roster.
Attachment also compares the declared pre-attempt snapshot to its actual prefix.
Exact snapshot comparisons use canonical bytes: Python's equality between
`true`, `1` and `1.0` cannot satisfy a typed count. External exposure population
identities must be content SHA-256 digests; malformed identifiers cannot be
silently classified as unrelated history.

The complete local ledger is limited to 8 MiB, using the same bound as immutable
metadata/snapshot reads. A supported completion prefix therefore cannot exceed
its attachment reader's capacity. An oversized canonical ledger refuses without
truncation; larger or linked history requires a separately reviewed extension.
