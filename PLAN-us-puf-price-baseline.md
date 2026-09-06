# Plan — developmental PUF price-restatement baseline preparation

Frozen before implementation. Branch `us-puf-price-baseline-20260906`, forked
from `origin/main` and fast-forwarded to the accepted source-projection base
`688d00ca24060dbbe04e489ee99b7527750f9628`. Root plan:
`architecture-reviews/microcosm-20260904/us-puf-price-baseline-root-plan.md`.
Adjudication: `puf-growth-baseline-adjudication-fable51-capture.md`.

**Nothing here is a genuine run.** No PUF CSV, demographic file, status or
projection artifact payload, store path, or HH/person/return array is read or
statted. Every execution below is on invented fixtures. `root_approved` is
false in every emitted receipt.

## Load-bearing constraint: the accepted twelve-field key must not move

`USPufMonetarySourceKernel.implementation_hash()` hashes the *bytes* of
`puf_monetary_source.py` and `puf_raw_source.py`
(`microcosm.graph.kernel.source_hash`), and `_all_node_keys` folds that hash
into the node store key. Editing either module would move the accepted
twelve-field artifact's store key
(`us-puf-monetary-v3-root-acceptance.json:artifact_store_key`).
`test_us_puf_monetary_source.py:1803` also pins
`sha256(tools/build_us_puf_source_layout.py)` to the packaged document's
`generation.generator_sha256`, so that generator is equally frozen.

**Therefore: zero edits to `puf_monetary_source.py`, `puf_raw_source.py` and
`tools/build_us_puf_source_layout.py`.** The thirteen-field increment is
additive: a new generator that imports the frozen one, a new packaged
document, a new artifact type, a new kernel. The preserved implementation
hash is measured before and after and reported.

## Steps

1. **Pin the primary CPI resource.** `download.bls.gov/pub/time.series/cu`
   flat files (`cu.data.1.AllItems`, `cu.series`, `cu.period`, `cu.area`,
   `cu.item`, `cu.seasonal`, `cu.base`) plus the keyless BLS public API v1
   POST for `CUUR0000SA0` 2015-2024. Record URL, HTTP status, byte length,
   full-body sha256, the exact extracted `M13` records with their line
   offsets and per-line sha256, and the metadata rows that authenticate
   `0000 = U.S. city average`, `SA0 = All items`, `U = Not Seasonally
   Adjusted`, `M13 = Annual Average`, base `1982-84=100`. The frozen
   seasonally adjusted `cpi_u.yaml` is **not** cited as the annual-average
   source. → `R/us-puf-price-baseline-cpi-resource.json`.

2. **Developmental authority in `puf_growth.py`.** Add
   `FactorAuthority.DEVELOPMENTAL_PUBLIC_RESOURCE`, its required provenance
   key shape (the reviewed shape plus `resource_sha256` and `retrieved_utc`),
   the required literals `population_divisor == "none"` and
   `index_or_total == "index"`, and a refusal that a developmental table whose
   digest is in `REVIEWED_FACTOR_TABLE_DIGESTS` cannot compile. Release
   eligibility stays keyed on `REVIEWED_RESOURCE` alone, so no allowlist entry
   can flip a developmental table to release-eligible.

3. **Thirteen-field source projection.**
   `tools/build_us_puf_agi_source_layout.py` imports the frozen generator,
   asserts its sha256 equals the packaged document's pinned
   `generator_sha256`, rebuilds the twelve-field document and asserts it is
   byte-identical to the packaged file, then extends `PROJECTION` with
   `E00100` (reported AGI, delivered index 73, publisher position 31,
   marker-signed) and removes its `NOTABLE_SOURCE_ONLY` entry with a recorded
   supersession. Output:
   `us_runtime/puf_2015_monetary_agi_source_projection.json`.
   `puf_monetary_agi_projection.py` carries a distinct artifact type
   (`microcosm.us.puf_2015_monetary_agi_source_projection` v1) and reuses the
   frozen module's envelope helpers rather than restating them; the
   twelve-field magics are refused **by name**. Same grammar, same
   ordinary-integer versus disclosure-aggregate lexical split, same sentinel
   and same `amount_known` rule. No derived AGI, no prior wages.

4. **Money-view adapter and price baseline.** `puf_price_baseline.py`:
   the pinned CPI endpoints, the factor-table document builder, the recipe
   (thirteen money fields on one `nominal_price_restatement` series, `RECID`
   identifier, `S006` declared with an explicit identity design-weight series
   so the no-mass-transition choice is recorded, not implied), and
   `adapt_projection_to_money_view` which joins the projection to the status
   artifact on `RECID`, removes only the aggregate rows, records the excluded
   audit identity, and returns `FLPDYR`, `demographic_status` and the
   unaltered integer `S006` separately. Growth is the existing
   `apply_puf_growth`; nothing reimplements it.

5. **Tests.** New `test_us_puf_monetary_agi_projection.py` and
   `test_us_puf_price_baseline.py`; targeted additions to
   `test_us_puf_growth.py`. Old twelve-field and old growth behavior is shown
   unchanged by running those suites unmodified except for the added
   authority cases. Roundtrip, tamper and boundary cases for the new artifact.
   `tools/ci_test_groups.py --verify` stays green.

6. **Invented end-to-end receipt.** Build an invented delivery, produce the
   thirteen-field artifact, adapt, grow, and record: exact rational
   `313689/237017`, the float64 ratio bit pattern, unrounded float64 output,
   sign and zero preservation, scaled exact sums per column, `amount_known`
   agreement, unique `RECID`, status join equality, no sentinel in any numeric
   input, pairwise violation counts equal before and after, `S006` bytes
   unchanged, `release_eligible` false.

7. **Report.** `R/us-puf-price-baseline-preparation-REPORT.md` and `.json`
   with clean head, exact file hashes, git-verified loaded code identity, the
   CPI resource pins, test and invented-output receipts, the planned genuine
   call shape with `root_approved: false`, and the sensitivity gaps against
   the five draft category ratios stated as gaps, not as error.

## Out of scope

No genuine run, no remote write, no push, no PR, no merge, no Modal, no
package install, no nested agent, no population cloning, no fit, no
tax/benefit calculation, no allowlist promotion.
