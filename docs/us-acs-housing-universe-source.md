# Authenticated ACS housing source and graph selection

This development route preserves ACS2024 source evidence for occupied housing
units, vacancies and both group quarters classes. It authenticates the two fixed
national PUMS archives and builds the existing native Frame from the same private
copies. It does not certify calibrated descendants, physical group quarters
facilities, congressional district accuracy or release eligibility.

The existing ACS codec, native mapper, donor helper and pure housing classifier
are unchanged. The new source API is
`microcosm.build.us_runtime.acs_housing_universe_source`; typed graph transport is
in `graph_acs_housing_universe`.

## Source contract

`acs_2024_1yr_sources.json` remains the closed authority for the exact compressed
archive bytes. The source directory must contain only `csv_hus.zip` and
`csv_pus.zip`. No public API accepts replacement hashes, arbitrary members or a
caller-authored source receipt. Invented tests patch private pins only.

Each operation captures regular files using `O_NOFOLLOW | O_NONBLOCK`, checks
every path component, and verifies exact size, SHA256 and descriptor/path
stability. A unique mode0700 child of an explicit snapshot directory holds
exclusive, fsynced, mode0400 copies. Closed-file hashes verify persisted copies.
This is an owned local operator boundary, not isolation from a malicious process
running as the same user.

The complete ZIP central directory is checked before the legacy Frame builder
can run. Duplicate or case-colliding names, paths, directories, special files,
encryption, unsupported compression and misleading data prefixes refuse. Every
member is consumed with bounded expanded size and its CRC checked by the ZIP
reader. The receipt binds member name, compressed and expanded lengths, CRC,
decompressed SHA256, raw header bytes and their hash, parsed header and row count.
Ordinary non-data members are inventoried but never projected.

Every physical data record must be strict UTF8, have the complete header width,
and fit the record and field caps. Full ordered headers must agree within a
role. Projected input fields stay strings, with no trimming or NA inference:

| Role | Source fields |
|---|---|
| Household | SERIALNO, TYPEHUGQ, NP, TEN, WGTP, PUMA, ST and/or STATE |
| Person | SERIALNO, SPORDER, PWGTP |

Generated lineage adds the original member name and one-based data row ordinal.
The receipt supplies the member's digest. SERIALNO retains the original ACS arm
and vintage; it is never replaced by an ASEC or PUF source key. SPORDER retains
its raw token while a separate integer value supports joins: `01` and `1` both
mean line1, so a duplicate after normalization refuses. Whitespace identifiers
refuse. ST/STATE must agree when both exist. The national source scope is the
50 states and DC; code72 refuses as a scope mismatch rather than being silently
discarded.

The full source requires unique households, no orphan persons, unique normalized
person keys and exact NP/person counts. The existing `_validate_person_counts`
helper runs before any selection or vacancy exclusion. Its surrounding guards
also catch orphans and duplicates that the helper alone cannot detect.

| Source case | Rule and retained interpretation |
|---|---|
| Occupied housing unit | TYPEHUGQ1, NP1–20, WGTP1–9999; blank TEN remains unknown tenure |
| Vacant housing unit | TYPEHUGQ1, NP0, no persons, WGTP1–9999, blank TEN; retained separately in source evidence |
| GQ placeholder | TYPEHUGQ2/3, NP1, WGTP0, blank TEN, exactly one linked person with PWGTP1–9999 |

Nonblank TEN must be1–4. The source's vacancy definition is separate from the
unchanged pure classifier, whose NP0 `occupied_hu` remains unknown. Source
vacancies never become occupied units. GQ placeholders represent source persons
at DESIGN weight; they do not establish a count of physical facilities. Occupied
unknown tenure remains in the occupied-unit denominator with unclassified tenure.
Model `tenure_type` never resolves a missing raw TEN.

The dictionary authority, page locators and named source definitions are in
`acs_2024_housing_universe.json`. The dictionary is the
[2024 ACS PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf),
SHA256 `929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff`.

## Source-only publication and reconstruction

`produce_acs_housing_source(source_dir, snapshot_root=..., output_dir=...,
serialnos=None)` authenticates the full parents, then publishes canonical
`projection.json` and `receipt.json` into a new directory. `None` means all
source households. An explicit tuple selects whole raw SERIALNO keys; unknown,
duplicate or malformed selections refuse. Sorting is canonical and independent
of caller order. A vacancy-only selection is valid source evidence.

`load_acs_housing_source(source_dir, output_dir, snapshot_root=..., serialnos=...)`
freshly captures and reconstructs the full parents. It privately captures both
candidate files and compares exact lengths and bytes by SHA256 before returning
the reconstructed source. Candidate JSON is never decoded as authority. Changed
metadata, rewritten projection values, reparenting and crossed selections refuse.
Both operations return immutable byte-backed `AuthenticatedACSHousingSource`;
DataFrame access creates owned copies.

Capture and output writes never replace an existing path. Successful captures
and output directories must sit outside the closed original source directory;
overlap refuses before copying source bytes. Successful captures
remain available for local audit. Failed captures remain private with a sanitized
`failure.json`; a failed publication can leave a partially written, exclusively
owned output directory. Neither source inputs nor earlier artifacts are cleaned
up or modified. Retrying requires a fresh output directory. Retained captures
consume disk and need a separate, explicit cleanup decision.

Operational caps are8GiB per source projection,1MiB per receipt,64 members per
archive,8GiB per expanded member and16GiB expanded per archive. Records are capped
at1MiB and fields at64KiB. The disk preflight reserves compressed parent sizes
plus two capped projections, two receipts and1GiB reserve, accounting for full
projection and selected publication/candidate coexistence. These are refusal
bounds, not measured forecasts of genuine source size or memory use. No genuine
source run is authorized by a passing invented test.

## Same-snapshot native Frame

`prepare_acs_housing_population(source_dir, snapshot_root=..., serialnos=None)`
preflights the private snapshots and passes those exact paths to the unchanged
`build_acs_pums_unit_frame`, with no household cap. Native mapping and the existing
operator-free source check run before whole-household key selection. Thus input
authentication and group construction remain full-source work even when the
resulting Frame is tiny. A vacancy-only Frame request refuses.

The bridge checks original SERIALNO, normalized SPORDER, source-person and
source-household coordinates, native memberships, year, geography, raw TYPEHUGQ,
NP and observed/missing TEN. The builder does not retain a WGTP cell: lexical
WGTP stays in the artifact and is compared through the Frame's exact typed DESIGN
weights. GQ weights are the single linked source person's PWGTP. This is the
existing source weight construction, not manual aggregation of PolicyEngine
entity weights. There is no IMPORTANCE/calibrated-weight transition here.

The preparation receipt binds all Frame cells, ordered axes, dtypes, missingness,
metadata, strata and weights before and after explicit string-storage promotion.
Promotion checks string values and missingness. Every other column and all native
memberships remain part of the Frame storage identity. No HU result is inserted
as a model input or a promoted alias.

## Typed CREATE and SELECT

`acs_housing_graph(columns, serialnos=None, selected_serialnos=None)` creates two
real graph nodes. Use `acs_housing_registry(snapshot_root=...)` and the existing
`us_source_codecs()` registry. The new codec name is
`us-acs-housing-universe-2024-v1`; its public loader lives in the source module.
An explicit `ContentStore` loader call uses the dedicated
`acs-hu-source-captures-v1` child, never store staging or object directories.
The graph executor content-keys declared source paths; CREATE performs the
authenticated Frame construction.

CREATE emits Frame context, preparation receipt and native HU evidence as typed
artifacts from one producer. The complete HU payload has its own64MiB cap,
separate from the existing raw-byte source cap. It contains bounded canonical
header/receipt/projection sections, eight ordered native uint8 code arrays and a
checksum. Decoding checks exact section lengths before parsing and recomputes
all native codes from the lexical projection. `BoundACSHousingEvidence` is graph
transport, not a newly authenticated source object.

SELECT consumes the same producer's original artifacts and all declared Frame
cells. Type, artifact key, producer, original context, source receipt and complete
Frame storage identity must agree. Structural-only group ID tables are rebuilt
from declared person memberships and checked against the producer's ordered-ID
digest; implicit IDs are never declared as invented owned data cells.

Whole-household selection preserves exact DESIGN weights without normalization.
It emits selected context and a selection artifact binding the original payload,
receipt, producer, person/household positions, ordered IDs for every entity, and
selected Frame/projection hashes. Original CREATE HU evidence remains available
as ancestry. SELECT does not mint a second source authority or assert that its
pruned population is representative. The mass change is explicitly declared.

The new `acs_housing_universe_2024` implementation scope binds source, mapper,
classifier and transport modules, actual Frame/graph/microunit inventories, rule
YAML, source/definition resources, codec identity, operator-boundary projections,
numpy/pandas/microunit/PyYAML/pyarrow versions and pandas string options. Its fifth
codec selector follows codec-map insertion order. Existing stage rosters remain
unchanged; the new modules receive actual stage/contract entries, not a fictitious
build-package roster.

Invented tests exercise source reconstruction and rejection, same-snapshot real
Frame construction, direct/graph equality, nonidentity selection and zero-kernel
warm replay. These establish code and custody contracts only. A later reviewed
run must measure genuine source/member inventory, projection size and resources
before authorizing scale, calibrated descendant bridges or CD evaluation.
