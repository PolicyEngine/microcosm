# Re-pin the US Chronicle consumer artifact

`us/chronicle_feed.json` pins the Chronicle source commit, fact rows, artifact
manifest and scope consumed by the US fiscal target registry.
`us/chronicle_feed_scope.json` names the source package and build year for every
retained record-set/period pair. The builder exports source packages, selects
those pairs, and validates the result through Chronicle's consumer-artifact
builder. No emitted source row is patched.

## The pin

| Field | Value |
|---|---|
| Chronicle commit | `b571381fcd875393ea0dabc326558cfa2ca8e8fa` ([Chronicle #278](https://github.com/PolicyEngine/chronicle/pull/278)) |
| Fact rows | 39,158 |
| Facts SHA-256 | `4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f` |
| Manifest SHA-256 | `38ec5bf1efe5a0bd017ec5279065e2ea7645b37da237197f03ae2fbca28cadac` |
| Artifact wrapper | `policyengine_ledger.consumer_artifact.v2` |
| Row schema | `chronicle.consumer_fact.v3` |
| Row-schema SHA-256 | `bdb51e2a8115634633ba7448c4005930fd9c0bfbade5e1b079b6bc24da485d3d` |
| Scope | 586 record-set/period pairs; 62 package runs across build years 2020–2029 |

The artifact stays outside Git at
`~/PolicyEngine/_buildh-runtime/inputs/chronicle_us_b571381/artifact/`.
It contains `consumer_facts.jsonl` and `manifest.json`. The previous bare feed
`consumer_facts_us_c5e5bf8.jsonl` remains beside it, unchanged. The manifest
binds the fact and row-schema hashes, not the output path, scope file or source
commit. Microcosm separately pins the scope's exact bytes and source commit;
the builder's clean-checkout check and build receipt establish which Chronicle
source produced the artifact.

## Rebuild and verify

Use a clean Chronicle checkout at the pinned commit. From Microcosm:

```bash
uv run python tools/build_us_chronicle_feed.py \
  --chronicle-root ~/PolicyEngine/chronicle \
  --out /tmp/us-chronicle-artifact
```

The tool checks the Chronicle commit and clean tree, then runs one targeted
`chronicle build-bundle` command per build year. It keeps each row only from the
package/year assigned to its pair, refuses missing pairs and conflicting fact
keys, sorts by `aggregate_fact_key`, and invokes
`chronicle build-consumer-artifact`. `receipt.json` records all commands, the
scope hash, row count, source commit and both artifact hashes. The default
Chronicle command is `uv run --frozen chronicle`; `--chronicle-command` can name
an already-locked interpreter and `-m policyengine_chronicle.cli` explicitly.

Compare the generated artifact hashes with the declaration before placing its
two files in the external input directory. Then regenerate and test the
consumer resources:

```bash
uv run python tools/build_us_target_parity_manifest.py
uv run pytest packages/microcosm-build/tests/test_us_chronicle_feed.py \
  packages/microcosm-build/tests/test_release_target_parity.py
```

The generator refuses a facts hash that differs from the declaration. Its
resources preserve 32 compiled families, 52 reviewed exclusions and 81 feed
families. The local artifact tests and the two feed-dependent parity tests
must execute when qualifying the input; a CI skip is not that qualification.

For a release's `--exact-k` arm, pass the artifact **directory** as
`--ledger-facts`, with both `--ledger-facts-sha256` and
`--ledger-manifest-sha256` from the declaration. Passing the JSONL alone loads a
bare feed and cannot satisfy the manifest pin. The `--base-h5` arm continues
to accept a bare feed; it does not thereby run the exact-k improvement gate.

## Source-authority repair

The preceding #955 pin used Chronicle `c5e5bf8` and facts SHA-256
`b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801`.
Its 994 source-label rows lacked the authority required by the existing v3
schema, so the canonical artifact validator refused that feed. Chronicle #278
records the actual publisher in the source packages: CMS for 515 rows, Census
for 468, and JCT for 11. It does not claim an Axiom alignment or relax a schema.

The 19 September 2026 qualification rebuilt all 62 package runs from the
pinned source commit. It retained every one of the preceding pin's 39,158
cells with an identical value, adding or removing no cells. The 586 scoped
pairs are unchanged. Exactly 994 rows change only these provenance fields:

- `concept_alignment.authority`;
- `concept_alignment.concept_alignment_key`;
- `layout.record_set_spec_hash`;
- `legacy_fact_key`.

Aggregate and semantic fact keys, source references, lineage, labels,
observations and all other row fields remain identical. Both feeds compile
through the period-2024 fiscal registry, age targets, packaged CD crosswalk
and reviewed Medicaid substitutions to the same 32,867 target identities in
32 families. Every target value, measure, filter, entity, period, tolerance,
citation and hierarchy remains identical. The compiled metadata gains an
authority on 166 targets and updates legacy keys on 165; the Rhode Island
substitution deliberately drops per-fact keys. No other TargetSpec field
changes. The registry version consequently changes from `b74d86d94a76` to
`749a7b0627ce`.

The evidence preserves the strict full-TargetSpec equality failure alongside
this explicit provenance-only comparison. It does not report byte-identical
registries. See `experiments/us-chronicle-feed-repin/artifact_qualification.json`.
Reproduce the comparison against the preserved public feeds with:

```bash
uv run python experiments/us-chronicle-feed-repin/qualify_artifact.py \
  --old ~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl \
  --new ~/PolicyEngine/_buildh-runtime/inputs/chronicle_us_b571381/artifact \
  --out /tmp/us-chronicle-comparison.json
```

The helper checks complete source rows and TargetSpecs, refusing any difference
outside the four source-provenance paths and two target-metadata paths above,
or any deviation from their documented counts. It pins both facts hashes and
the new manifest hash before comparing rows, refusing different inputs. It also
checks the added publisher authorities, source-cell identity, scope, target
identity and values. It requires the changed targets to be exactly 155 in
`cms_medicaid.state_enrollment` and 11 in `jct.tax_expenditures`, and enforces
the previously qualified registry versions and full canonical TargetSpec hashes.
Its report includes full target equality separately; adding
`--require-identical-targets` exits 1 for this repair even when the narrower
qualification passes. It loads public Chronicle facts, never population data.

The full export took 578.58 seconds and peaked at 1.75 GB RSS. The canonical
loader validated all rows, and independent artifact re-packaging produced
byte-identical facts and manifest. One full export was run for this authority
repair; the repeated check covers packaging, not a second source export.
Cross-package label and duplicate-semantic-key warnings remain in the bundle
reports; all package runs were valid with none skipped.

## Why the labelled feed replaced the July pin

The earlier July feed (`consumer_facts_buildn_v9_4.jsonl`, `b3c08356…`) lacked
Chronicle-owned dimension labels, which the current target compiler requires.
The first labelled export in #955 kept exactly its 586 record-set/period pairs.
Scoping by record set alone would incorrectly add Medicaid months.

The source-cell comparison in
`experiments/us-chronicle-feed-repin/value_diff.json` records that first move:
all 37,399 prior cells retained the same value; 1,759 cells were added within
existing record sets; six duplicate CBO cells were represented once. This
source-authority repair preserves that labelled feed's complete cell surface.
New source families remain a separate compile-or-reviewed-exclusion decision.

A labelled Medicaid hierarchy also exposed a consumer defect corrected in
#955: the Rhode Island substitute inherited a neighbouring state's hierarchy.
`_substituted_hierarchy` now uses Rhode Island's identity and reviewed state
label. That calculation/selection behavior does not change in the authority
repair.

This artifact qualification addresses the Chronicle input contract. It does
not certify a population, pass the release improvement gate or authorize
publication. Chronicle #278 and the Microcosm consumer change retain their
independent code-review and CI requirements.
