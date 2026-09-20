# PR #916 merge-readiness audit

Objective: resolve the PR's engineering, integration and statistical issues to
reach merge-ready quality while retaining draft status. This audit does not
authorize publication or replacing the default population. Completion is not
yet established.

## Current audit, September 20, 2026

The starting PR head is `d2601c7a0749b134822086f4783e03501c750e32`.
Its 24 GitHub checks passed. The current integration base is
`18c6d39a`, 65 main-branch commits beyond the PR's previous merge base.
Maria's September 15 review is the latest external review; no follow-up approval
is recorded. Passing tests below are engineering evidence, not statistical
acceptance.

| Requirement | Current evidence and remaining work |
| --- | --- |
| Current main, clean merge and safe PR branch | Merge commit `8d74be68` reconciles README and serializer-test conflicts. At `4f746222`, local HEAD, upstream tracking and push refs match `fork/fix/us-childcare-attendance`; GitHub reports mergeable and draft. |
| C1: exact attendance values and source recipe bound together | The 573-test affected suite passes with no skips. Fresh verification of the real 166,321-person candidate through both native loaders validates the exact source/code/runtime/settings binding and preservation of all original values and weights. |
| A1: row-complete final attendance | The affected suite covers the final rowwise boundary (null days, out-of-range days, incoherent zeros and fractional monthly days), unbound nondegenerate attendance refusal, native tampering, receipt reuse and fiscal-builder checks. The complete national fiscal run remains separate evidence below. |
| A2: valid source household identities | Missing, blank, stringified-null and unresolved household identities are rejected; regression coverage exists. |
| A3: household intensity and larger-sibship validation | Joint days/hours and 3+ child diagnostics exist. Their reported screen failures remain substantive; adding diagnostics alone does not qualify the model. |
| S1: noncalendar transport and sensitivity | Observed regular hours remain fixed; paired sensitivity holds donor identities fixed. State flags and missing-calendar selection remain unresolved statistical evidence. |
| S2: explicit operation provenance | Receipts identify calendar derivation, dependence fitting, noncalendar bridge, target harmonization, joint transfer and outside-domain baseline policy. |
| Fiscal builder and exact-k integration | Repair metadata preservation and exact-k source configuration have targeted tests. The release rule requires a build from raw sources and certification from this PR's tree; a supplied-parent fiscal run is an intermediate integration check only. |
| Annual projection compatibility | The regression reproduced the original receipt loss. All 42 annual/serializer tests now pass: the 2024 base and 2025/2026 projections retain their original receipt, changed attendance is refused before finalizing an output, and the combined serializer inventory has ten writers. |
| Source mapping and population integrity | Licensed DS4/DS5 files, pinned ASEC cache and BuildP parent are available locally. Current candidate evidence preserves original population values/weights. Recheck any affected evidence after implementation or runtime changes. |
| Statistical qualification | Existing experiments fail provisional subgroup/household screens. Do not redefine those failures as acceptance, relax thresholds after seeing results, or call inspected partitions independent validation. A defensible model/assumption treatment and stronger validation remain required. |
| Final tests, packaging, evidence and handoff | The affected engine and wheel suites pass as detailed below; the real candidate report is reproduced. Source microdata and per-person hashes remain local. The PR retains its feature-branch push target, draft status and VS Code PR file-tree view. Full-build and statistical evidence remain separate requirements. |

## September 20 engineering checks

- Initial merged attendance, launcher and serializer run: 139 passed, one failed.
  The failure was the serializer count: the combined branches contain ten
  serializers, while each independently expected nine. The expectation was
  corrected; the subsequent 42-test annual/serializer run passes.
- Specification coverage remains current: 42,159/42,159 fields and 41/41 inventory
  checks. No checksum regeneration is needed for this merge.
- Repository lint and the tracked CI test inventory passed after conflict
  resolution.
- The new annual receipt regression failed on the original annual writer at
  native reload, proving the uncovered integration defect. The final annual and
  serializer run passes all 42 tests with no skips. Repository-wide lint and
  changed-file formatting also pass after the fix.
- VS Code recognizes the worktree as PR #916 and shows the GitHub Pull Request
  sidebar with its clickable file tree. Its Sync target is the fork's feature
  branch, not main.
- At `4f746222`, all 573 affected source, receipt, coverage, fiscal-builder,
  exact-k-launcher and L0-export tests pass, with no skips.
- All workspace wheels build and install into a separate constrained environment.
  The five package imports resolve inside that environment, PolicyEngine-US is
  absent as required by the base-wheel lane, and the annual exporter matches
  the committed source bytes. The affected wheel suite passes 172 tests; its
  40 skips comprise 26 checks requiring PolicyEngine-US and 14 requiring
  optional PyTables. The engine-enabled tests above cover those boundaries.
- Fresh native candidate verification reproduces
  [the committed report](review-fixes-verification.json) byte for byte
  (SHA256 `765ef64f53b7cad45153bc96ffae65b6156f7c23259f004ad722c828edea165e`).
  Both native loaders preserve all 166,321 people's original values and weights,
  and the exact attendance binding. No recipe change or population rebuild was
  necessary for the main merge and annual receipt fix.

## Full-build preparation and constraints

The [US release build rule](../../docs/us-release-build-rule.md) requires a
from-scratch build and certification receipt from the PR's own tree for a change
that alters how the dataset is built. Reusing the July BuildP population does
not satisfy that requirement, even if its fiscal refresh passes. Default
promotion additionally requires the exact-k frozen-register improvement gate;
publication remains a separate human decision.

The pinned DS4/DS5 files, raw ASEC CSV cache and original BuildP parent are
available locally. These are not the complete processed inputs for the raw
base builder. The inspected feed and source-stage remedies are still in open
upstream PRs as of September 20:

- [#952](https://github.com/PolicyEngine/microcosm/pull/952) pins and fetches the
  three processed ASEC H5 inputs from their public mirror.
- [#955](https://github.com/PolicyEngine/microcosm/pull/955) reproduces the
  approved Chronicle feed's 586 recordset/period pairs with labelled facts.
  [#961](https://github.com/PolicyEngine/microcosm/pull/961), stacked on #955,
  pins the validated consumer artifact produced with
  [Chronicle #278](https://github.com/PolicyEngine/chronicle/pull/278).
- [#948](https://github.com/PolicyEngine/microcosm/pull/948) diagnoses unresolved
  SPM composition, and [#959](https://github.com/PolicyEngine/microcosm/pull/959),
  stacked on #948, materializes the Census independence role in raw builds.

Those implementations have not been incorporated into this PR. The approved
target scope spans multiple years. Both exploratory Chronicle builds were
deliberately stopped: the default run included other countries, and the US-only
2023 run did not reproduce the approved multi-period scope. Their partial
outputs are not accepted consumer artifacts. A generic current-year export
must not silently replace the reviewed feed.

This host has 16 GiB RAM. The release runbook records a raw base build peaking at
72.47 GB on a 128 GiB machine under the older 1.819.0 engine; it also reports a
July fiscal comparator at about 85 GB. These historical measurements are not a
guarantee for the present larger build, but they rule out treating a generic
32 GiB minimum as adequate preparation here. An appropriately provisioned build
host and the licensed raw inputs are still needed.

Once the upstream prerequisites are integrated, validation must build a fresh
raw-source base/pool and run the normal preflight and certification gates from
the resulting PR tree, retaining its code, input and artifact identities. The
July frozen household selection cannot be reused for the rebuilt base; the
release runbook explains its donor-loss failure and the permitted new lineage.
No coverage or statistical gate overrides are an acceptance substitute.

The invocation below is only a prepared **supplied-parent integration check**,
not the required from-scratch build and not a completed run. `/local` denotes
the operator's input/output directory; use a fresh output directory, the
recorded parent hash, and replace the quoted SHA256 placeholders with the actual
qualified consumer-artifact hashes.

```bash
uv run --no-sync python tools/build_us_fiscal_refresh_release.py \
  --base-h5 /local/populace_us_2024.h5 \
  --ledger-facts /local/chronicle-us-consumer-artifact \
  --ledger-facts-sha256 "CONSUMER_FACTS_SHA256" \
  --ledger-manifest-sha256 "CONSUMER_MANIFEST_SHA256" \
  --asec-2023-weeks-unemployed-source /local/asec/asecpub23csv.zip \
  --childcare-attendance-household-tsv /local/39466-0005-Data.tsv \
  --childcare-attendance-calendar-tsv /local/39466-0004-Data.tsv \
  --childcare-attendance-asec-cache /local/asec \
  --childcare-attendance-inherit-outside-domain-baseline \
  --seed 915 --no-staging --out /local/pr916-fiscal-validation
```

That integration check should retain its final H5, source-coverage receipt and
release-gate results, with attendance checked on native reload. Completion of
the merge requirement needs equivalent evidence from the raw-source build and
certification as well. Neither an authenticated target registry nor the
supplied-parent integration result alone meets the requirement.

## Independent attendance evidence

Recent SIPP files are not a replacement hours benchmark. The Census Bureau's
[2023 SIPP comparability note](https://www.census.gov/programs-surveys/sipp/tech-documentation/user-notes/2023-usernotes/compr-2023-sipp-chld-care-data-prev-yrs.html)
describes the change from each child's arrangement hours in the 2008 and earlier
panels to the redesigned arrangement questions. The
[2024 SIPP variable catalog](https://api.census.gov/data/2024/sipp/variables.html)
provides arrangement and payment variables, not the joint days/hours calendar
needed here. It could support a separately harmonized participation comparison;
it cannot close the schedule-intensity or noncalendar-transport evidence gap.

The existing NSECE development folds and incomplete-calendar bounds remain useful
diagnostics. They are not a new independent sample, and missing schedules cannot
be turned into observed validation outcomes. The subgroup and household screen
failures therefore remain open after the engineering checks pass.

The full fiscal run is not replaced by the standalone attendance-preparation
command or by mocked builder tests. Publication, survey transport validity and
maintainer approval are distinct from serializer and code-contract checks.
