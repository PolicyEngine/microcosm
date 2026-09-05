# Authenticated ASEC current-money and tax units

`asec_current_money_source.load_authenticated_current_money_source(parent_path,
attachment_path)` verifies the reviewed full v4 checkpoint and its household
observation attachment. These are fixed source pins, not a public registration
or caller-hash API. The adapter performs no discovery, download, sampling,
calibration, policy calculation, or release evaluation.

The parent SHA256 is
`e2f2b7495bfcf1448dfb0acb0a17e93f86a6ab8bef8a70ec0a2981983028cbb5`;
the augmented attachment SHA256 is
`f7f086e262d9a1d9ec6fc1a0b7c32e577128578a4fc5b3f14e13fb8711f7ec2d`.
Both files are streamed into private temporary files while hashing. Only those
verified private copies are decoded. Replacing an original path after staging
cannot change the bytes decoded by that call; a subsequent call checks the
replacement against the fixed pin.

The actual v4 loader validates its registered source-only contract. The new
adapter also checks every unmodified parent table, index, dtype, numeric bytes,
weight, stratum, and mass-log record against the augmented checkpoint. The
seven household observations must match the actual attachment receipt,
household IDs, native H_SEQ join, income years, and cohort hashes. All three
income years 2022/2023/2024 and their registered survey-year 2023/2024/2025
ED_VAL archive/member pins are required. This authenticates their transitive
provenance from the pinned checkpoint metadata. It does not reopen or rehash
the original Census archives or cohort HDFs.

The issued `AuthenticatedCurrentMoneySource` exposes `.frame`, `.scope`,
`.source`, `.spec`, `.views()`, `.ready()`, and `.validate()`. Scope and input
digests are derived from actual ordered IDs, memberships, source years, native
keys, pandas indices, and the 33 owned raw money inputs. A full loaded-Frame
fingerprint also detects changes outside those money views. Verification binds
the actual loader/helper/Frame code, runtime versions, fixed pins, and live
source-operator ownership projection. Live source mutation or implementation
change refuses before subsequent source use. A caller cannot transfer authority
to a differently ordered or edited projection by retaining the original scope.

`.ready()` follows the original decoder, restater, and completeness checks. It
returns an immutable `ReadyCurrentMoney` with
`source_authentication=checkpoint_bytes_verified`. Synthetic sources still
produce only `SyntheticReadyCurrentMoney` with `synthetic_unverified`; changing
a namespace string cannot promote them. The existing codec preserves that
distinction. Authenticated replay requires both the exact expected binding and
an independently issued `ReadyCurrentMoney` supplied as
`decode_current_money(payload, bindings, expected=ready)`. That expected object
comes from the verified source's `.ready()` or a previously checked replay.
Every parsed field buffer must equal its expected counterpart before a new
state is issued. A binding-only authenticated call refuses; synthetic replay
keeps its binding-only API and refuses an expected object. The header binds
source inputs, not the restated body, and a transport checksum alone never
established the body's derivation. Readiness here
means all declared current-money amounts passed this contract. It does not
mean all live nominal consumers have been replaced or a release is ready.

## Actual tax-only reconstruction

`asec_current_money_units.current_money_microunit_view(source, ready)` projects
the reviewed structural and student/disability controls plus exactly thirteen
corrected money inputs. Missing columns, null, Boolean, nonintegral, or nonfinite
controls refuse before execution. The transient view uses a fresh positional
RangeIndex, retaining an explicit positional association with the authenticated
person order. This prevents duplicate original pandas labels from expanding
microunit's label selection. It neither changes the source index nor creates a
partial Frame. Undeclared raw `_VAL` columns and harmonized income aliases are
absent from the transient view.

`reconstruct_current_money_tax_units(source, ready)` runs the actual existing
Frame unit adapter and microunit **0.1.0**, with `year=2024` and
`mode="policyengine"`. Its execution identity verifies the package entry point,
core, diagnostics, registry, tax constructor, rule helpers, and dependent-income
YAML bytes, their imported origins, adapter source, and dependency versions.
ANN NIU is already normalized to positive zero; signed employment losses remain
signed. No additional price conversion or parallel tax rule is implemented.

Before constructing the transient view or invoking the engine, the consumer
calls the source's `.ready()` and compares its bindings and complete field
buffers with the supplied money. This reuses the source's validation pass and
adds one classification and restatement per reconstruction. The post-engine
source validation remains in place. No retained source-content cache or
caller-provided expected hash substitutes for these checks.

The result replaces only the tax-unit table and three person outputs:
`person_tax_unit_id`, `tax_unit_role_input`, and
`is_related_to_head_or_spouse`. All other person values and US entities, source
indices, weights, strata, metadata, and mass-log records are retained and
checked. The new tax table uses the existing canonical table string boundary
before Frame construction and receipt sealing, so checkpoint reload does not
change its filing-status representation or content signature. Numeric values
are not coerced by this string boundary. The receipt binds the original and resulting Frame, unchanged money
artifact, actual execution closure, and old/new member partitions identified by
person IDs. `result.validate()` detects later output or implementation changes,
including the Frame-core and shared-loader closure already bound by the source
evidence in its money spec. The receipt also carries `money_content_sha256`, a
domain-separated digest of the exact encoder parts without their transport
checksum. It covers the header plus all amount, status, validity and zero-origin
buffers, using the same serialization as the encoder. Result validation checks
this content digest in addition to the header and Frame digests; the receipt
preserves both through checkpoint persistence. The content digest is not an
input to the pre-restatement header or source evidence.
The resulting Frame is not a new raw-source authority and cannot be fed back as
the original authenticated source. Both `release_eligible` and
`all_current_money_consumers_wired` remain false in its receipt.

## Verification and operational bounds

Tests use six invented people, three households, and all three cohorts. Their
private pin substitutions are explicitly test-only and are not evidence of
Census authentication. The real pinned constructor executes. In its existing
policyengine mode, the test's minor-relative fallback retains the partitions
while current-income qualification changes one filing status. This fixture
checks adapter behavior; it does not validate every legal dependency rule or
modify the runtime's adult-relative behavior.

This implementation retains both decoded source Frames during parent comparison
and performs several full-Frame fingerprint passes. Staging writes each source
file once, sequentially, with at most the larger staged file present at one
time; scalar hashing can allocate one column-sized byte buffer. These costs are
unmeasured on the real full source, including the added classification,
restatement and content-hashing passes. The content hash streams existing
immutable buffers without joining a second full transport payload. Prior wages
remain excluded from this corrected current-money interface. Only tiny invented fixtures are authorized
for this slice. Existing checkpoint string canonicalization is preserved;
metadata already omitted by that checkpoint format cannot be recovered here.
The sealed types prevent accidental misuse in trusted Python, not hostile
reflection or private-token access. Graph placement, remaining current-leaf
consumers, and end-to-end release validation require separate integration.
