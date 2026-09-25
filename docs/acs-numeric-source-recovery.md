# Recovering ACS source hours from the dense Build P parent

`tools/acs_numeric_source_recovery.py` creates a local, numeric source-hours
sidecar. It preserves raw `WKHP`, `WKL` and `FWKHP` with blank tokens represented
by NaN. It does not alter the parent, fill defaults, calculate hours, construct
SPM units, calibrate or publish.

The committed `tools/acs_numeric_source_recovery_contract.json` fixes the dense
parent SHA256, national counts and four numeric HDF block schemas. Its source
manifest digest binds the existing `acs_2024_1yr_sources.json`; the CLI accepts
no alternative parent or source pins. Every input is hash-checked before its
contents are interpreted. NumPy and h5py are the only third-party imports.

## Linkage contract and provenance limit

This is **reviewed sort/rank reconstruction**, followed by complete checks of
accessible numeric anchors. The release name identifies packaging revision
`592ae5d67a615c6d8afcf4e09da4cb3288b6088b`; it does not prove the revision or
`max_households` setting that produced the original staging file. The receipt
therefore retains `original_staging_revision: null`.

The reviewed source at that revision is:

- `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:172`:
  sorted text SERIALNO; occupied-household selection; numeric SPORDER sorting;
  household rank starting at one and person source-row rank starting at zero.
  Vacant housing units are removed before ranking. GQ placeholders are retained
  with NP=1 and WGTP=0. ST takes precedence over STATE.
- The same file at lines 264–279 and 458–462: numeric source identities and
  PH_SEQ/A_LINENO/A_AGE structural copies. Numeric identities were introduced
  in ancestor `136ffc9169fd70fa3e5029f9aa63c68ffa196653`.
- `packages/microcosm-build/src/microcosm/build/us_runtime/base_pool.py:707`:
  source IDs and clone index zero precede the optional current-ID offsets.
  The offset rule is at line 822; an offset is not a donor row count.
- `tools/build_us_acs_local_release.py:1090`: the exporter retains source
  identities and raw anchors from staging. Lines 1297–1311 derive the
  packaging revision independently of staging.

Full mode requires 3,422,888 source people and 1,531,614 occupied households:
1,348,408 housing units and 183,206 GQ placeholders. It checks a bijection on
`(2024, source_row_id)` and agreement with the separately unique
`(2024, source_household_id, SPORDER)` key. It also checks every retained
person anchor (13 columns), household anchor (7 columns plus numeric state),
structural copies, household membership counts, clone indices and constant
nonnegative current/source ID offsets. Current person and household IDs must
be unique across the whole parent, including the unmodified donor portion.

Finite SPORDER and NP nominate ACS candidates. They do not prove membership by
themselves: full coverage, independent keys, counts and anchors must agree.
Paired missing values match; numerical differences are never hidden by a
tolerance or null fill.

H5 access is limited to byte-string column labels and plain numeric values in
`person/block8`, `person/block9`, `household/block15` and `household/block16`.
No pandas HDF reader, PyTables, pickle, attributes, object blocks or recursive
payload walk is used. Required blocks cannot be external links, virtual
datasets or external storage. Inaccessible SERIALNO, PUMA, ST and spine
strings are not compared; numeric `state_fips` supplies the state check. The
receipt explicitly says `puma_compared: false`.

Complete checks close practical recovery under this stated reconstruction.
They do not establish unavailable original producer provenance or distinguish
a hypothetical permutation of records with identical accessible anchors.

## Pilot and full use

Run from this checkout with its existing environment. Source archives and the
parent must already exist locally; the tool performs no acquisition.

```sh
uv run --no-sync python tools/acs_numeric_source_recovery.py \
  --household-zip /absolute/path/csv_hus.zip \
  --person-zip /absolute/path/csv_pus.zip \
  --parent-h5 /absolute/path/dense-parent.h5 \
  --output-dir /absolute/path/new-acs-pilot \
  --pilot-households 64
```

The spread pilot selects deterministic occupied ranks across the national
order, including both housing units and GQ when both exist. Explicit selection
uses `--household-ranks 1,123,456` instead. All selected household members are
checked; global household/person source ranks are preserved. Pilot receipts
always set `full_source_bijection: false`, even if an explicit selection happens
to include every household. Person duplicates and multiplicities outside the
selected pilot households are not certified.

Both modes stream all source CSV members. The pilot reduces retained source
arrays and anchor comparisons, not source archive scan size. Both scan the
parent's four declared numeric blocks in bounded chunks and check global
current-ID uniqueness. Household sorting uses a temporary SQLite index;
source arrays are disk-backed. `--chunk-rows` defaults to 8192. Full mode uses
additional scratch disk for the national numeric arrays.

Omit both pilot selectors for full validation. An existing output directory
is refused. Validation failures remove only the tool's temporary directory;
successful output contains exactly:

- `acs-source-hours.npy`: a structured array of numeric fields, ordered by
  ascending parent person row. It carries that row index, current person and
  household IDs, source year/household/row, SPORDER, WKHP, WKL and FWKHP.
- `RECOVERY.json`: mode, checked counts/anchors, selected ranks for a pilot,
  provenance limit, input/contract/tool digests and the sidecar digest/size.

Consumers must require a complete receipt and verify its parent and sidecar
digests, then use `np.load(path, mmap_mode="r", allow_pickle=False)`. A later
hours repair must make its source-to-input rule explicit: the parent reportedly
contains the default 40 for all ACS usual hours, so replacement is a default
repair rather than filling missing exported values.

## Full-source qualification on September 14, 2026

The pinned dense parent passed full recovery after 64-household and
4,096-household pilots. Both source-key bijections and all accessible numeric
anchors matched across 3,422,888 ACS people and 1,531,614 occupied households.
The full run took 75.513 seconds with peak RSS of 1,330,905,088 bytes, using
8,192-row chunks. It retained the reviewed sort/rank reconstruction assumption,
unknown original staging revision and unverified PUMA identity described above.

The local sidecar contains 1,767,132 observed WKHP values. WKL identifies
1,105,858 additional people as not working in the past year; the remaining
549,898 people are under 16 and outside the source question universe. The
aggregate validation found no source contradictions or unresolved cases aged
16 and older. This qualifies source recovery only: it created no repaired H5,
country-calculated or calibrated dataset and establishes no country-model 2.x
compatibility.

The measured tool SHA256 was
`bf71851dc1ca48bf547040fbb8f3049da6bd3f3e1d5ab08eec2adf8f68b5f59d`.
The 273,831,424-byte sidecar SHA256 is
`2002eb20456df9b50d0c28383e30bc7730e91101b6f2f569beee64cf96e9e143`;
its recovery receipt SHA256 is
`4f3803c24c251c20caf19ef4bd9ca29b33f79d71f7c74477bce9fdaba27ca14b`.
These local artifacts and all source payloads remain outside the repository.

## Synthetic verification

The flat build-shard test file is
`packages/microcosm-build/tests/test_us_acs_numeric_source_recovery.py`. It
creates only invented CSV ZIPs and plain numeric H5 files, including an unused
external object-block trap. The finite suite checks ordering (text SERIALNO;
SPORDER 2 before 10), vacancies/GQ, donor collisions and clones, current-ID
offsets, complete pilot households, duplicate/orphan/missing keys, anchor
permutations and missingness, source pin failures, unsafe labels and output
preservation. Fixture contract overrides exist only in the library API; the
CLI uses the committed production contract.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest --noconftest \
  packages/microcosm-build/tests/test_us_acs_numeric_source_recovery.py
```

This verification imports the standalone tool and its numeric dependencies;
it does not import country models or the build shard's broader test fixtures.
