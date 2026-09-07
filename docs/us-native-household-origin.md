# Original household origins across the combined population

The native-origin extension identifies original survey household records across
assembly and the first full two-role clone. It changes no population cells,
weights, strata or design anchors. It fits no donor, invokes no policy engine,
reads no reference targets and activates no benchmark protocol.

`native_household_origin.py` reuses the v4 `SourceMember`, `AcsHouseholdOrigin`,
`AsecHouseholdOrigin` and `origin_key` encoding. Original member identity and its
SHA-256, source vintage/cohort and native household identifier define an origin.
Assembly source IDs and clone roles track descendants; they never substitute
for that source identity. Source records sharing a native number in different
cohorts are distinct statistical origins. This does not establish independent
real households across years, surveys or shared imputation donors.

## Source authority

ACS calls the existing `produce_acs_housing_source` reader. Its authenticated
member inventory supplies the original household CSV name and bytes digest;
its lexical projection supplies each raw `SERIALNO` and member name. The
canonical member identifier is
`census/acs/pums/2024/household/{original_member_name}`. Physical staging aliases
do not change the identity. No caller-provided `SourceMember` or fabricated
DataFrame can enter through this public source API.

ASEC has a separate `AsecNativeMemberPin` registry. The accepted prepared source
pins derived household HDFs and original **person** CSV members, so those pins
cannot authenticate an original household CSV member. The three original
household members are registered after a separate source audit and root review.
An empty registry or an unregistered cohort refuses before opening a source.

The source audit resolved `hhpub23.csv`, `hhpub24.csv`, `hhpub25.csv`
in the already-pinned `asecpub23csv.zip`, `asecpub24csv.zip`, `asecpub25csv.zip`
archives, confirming their central-directory names, archive/member byte hashes,
row counts and unique positive `H_SEQ` keys. The source audit and root independently
matched all 168,852 accepted attachment households and the 16 ASEC households in
the 169-household combined candidate. Their zero-weight ASEC counts are both zero:
that genuine coverage check is vacuous, while invented zero-weight tests remain
required. This identifier bridge certifies no other source values or full
population contents. The acceptance record is
`us-asec-native-household-root-acceptance.json`, SHA-256
`f997bbdc3573ed346d0e84d29f90c7b7ded9a12720c036d2789a7d561a04d602`.

The existing archive loader's local cache directory is
`~/.cache/microcosm/cps/asec_education`; the archive identities and income-to-survey
cohort map are already in `education_assistance_source.py`. No new download is
performed by the origin producer.

For any future original household member, record the exact archive and member SHA-256,
member byte count, row count, header digest, parser profile and verified archive
membership. Verify the original `H_SEQ` roster against all accepted prepared
household keys and their income cohorts, including zero-weight rows. Resolve
selected person membership through that same cohort/native household key. Keep
the exact mappings private. Publish counts, digests and refusal codes. A reviewed
registry revision then records `income_year`, `survey_year`,
`canonical_member_id`, `member_name`, `archive_sha256`, `member_sha256`,
`size_bytes` and `rows`. The registered canonical IDs are
`census/cps/asec/{survey_year}/household/{original_member_name}`.

The ASEC reader snapshots each registered member using the existing bounded
capture helper, checks its digest, parses positive integer `H_SEQ` tokens with
the strict CSV reader, checks uniqueness and row counts, and rechecks the
snapshot. No columns other than the native key are interpreted. Registry
declarations alone grant no authority: only matching source bytes issue the
private source capsule. Adding genuine pins requires the independent source
audit and exact scope review; never infer them from a person-member digest.

## Population and clone binding

`bind_population_origins` joins original source evidence to the carried raw
household identity and person cohort, including every zero-weight household.
Canonical integer text carried by union assembly can represent ASEC `H_SEQ`;
only its lossless positive decimal form is admitted, then matched to the
independently authenticated integer source roster. Floats, rounded values,
whitespace and noncanonical integer text refuse. ACS strings remain verbatim.

The private output records every household origin, entity source ID and clone
role, all person/group memberships, ordered entity-ID digests, all source
payload digests, and the optional prior population-binding digest. Its content
fingerprint covers every declared cell, ordered axis, effective typed weight
and stratum; column names are canonicalized while row order stays significant.
It deliberately excludes `Frame.metadata`, the graph mass ledger
and original design anchors, which are not available in `KernelContext`.
This fingerprint is the defined graph-view content identity, not a claim to
reproduce an HDF file hash or the executor's opaque population key.

`verify_materialized_population_origins` is the mandatory integrating-runner
boundary. It loads the actual typed outputs from the verified content store,
checks producer receipts and live implementation identity, compares the binding
with the complete `manifest.population(...)` Frame, and rederives lineage from
the carried source artifacts and explicit parent. An omitted or modified source
column refuses here. A graph consumer's projected context cannot establish its
own completeness, and its output alone does not claim full-population identity.
The verifier executes no kernels and reopens no original source files.

An expanded output needs its native parent binding. Each parent must have both
roles, unchanged origin and remapped source memberships, and two exactly
representable half-weight contributions conserving current mass on every
entity. Native selection also preserves each retained effective weight. Evidence
at the original row grain cannot be relabelled as expanded evidence. Native
whole-household selection can retain a parent binding; cutting individual
members from a retained household refuses.

Original design anchors remain executor-owned. Each clone inherits its anchor
unchanged; their aggregate doubles while current importance pair mass is
conserved. This module does not validate later calibration caps or allocate
design anchors. Later calibration, pruning of an already-cloned population,
partial attachment and additional clone roles need explicit lineage contracts.

## Graph identity and cache transport

`graph_native_household_origin.py` declares two source-only producers and a
population consumer. The source-only nodes name a population version because
the graph requires it to disambiguate structural branches; they request no
population columns. The consumer requests the complete column inventory and
uses the actual typed artifact aliases. It carries the executor's source and
parent producer/artifact keys into its output.

Cache transport verifies payload shape, typed keys and parent relationships.
It retains the original producer edges; it does not independently reauthenticate
original source bytes on a consumer cache hit. A nominal artifact type or a
caller-built graph cannot create external source authority. Production admission
still requires the reviewed graph/implementation, authenticated source receipts,
and exact candidate binding required by the unchanged v4 benchmark contract.

`graph_native_origin_implementation.py` validates the existing composition and
ACS source scopes, then the separate `native_origin_graph_inventory.json`.
Only these new kernels use it. The shared v1 implementation service, shared
inventory, accepted source modules and clone operator remain unchanged. The
new identity includes the actual additional modules, dependency/import/resource
contracts, source-codec identity, ASEC member registry and CSV runtime limit.

Invented cold and fresh REQUIRED-warm graph tests exercise source producers,
native binding, the real EXPAND+claim pair and expanded binding. A successful
replay requires zero kernel calls; it does not mean zero source/cache hashing
or I/O. Public receipts contain only counts and digests. All row-level lineage
and native source identifiers remain in the private typed artifacts.
