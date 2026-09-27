# US graph build: review guide

This branch consolidates the US graph build work for review. The intended file
supports national and 119th congressional district analysis from one population.
It is still a development build: complete enrichment, fiscal calibration and
release verification have not passed together.

## Architecture and review order

The intended construction order is ACS + CPS ASEC → complete survey multispine
including its initial support/PUF clones → household Census block assignment →
derived larger geographies → donor enrichment → rules evaluation and calibration
→ verified full and pruned exports. Each resulting household retains its assigned
location through the later stages. The PUF
operator acts on the combined survey frame. A separate ASEC–PUF base is not the
intended architecture.

National and local analysis share one fully constructed, enriched Frame. The
analysis branch begins after harmonization, cloning, atomic geography assignment
and imputation. Calibration changes weights; L0 controls sparsity. An export can
select a geographic scope and remove zero-weight households, but must preserve
every retained record's input values, knownness, entity relationships and assigned
geography. There is no separate national or local enrichment operation. A source
or mapping correction produces a new common parent Frame for both views.

Release verification must bind each full or pruned export to that parent's
identity and its calibration target/weight specification, compare retained
inputs by stable entity ID, and reject changed inputs, dangling entity links or
regenerated geography. Weight and scope changes are explicit exceptions; record
values are not. This is the required contract, not a claim that all of these
checks have already passed on the release candidate. The separate legacy
ACS-local hours omission in [#765](https://github.com/PolicyEngine/microcosm/issues/765)
illustrates the drift this common construction path removes.

Pre-calibration source quality and post-calibration export integrity are distinct
checks. Joint calibration may change an origin's contribution or set its weights
to zero. A pruned export must not be required to make every survey origin
independently resemble a national population, or to rerun imputation to satisfy
such a requirement.

The supplied-parent export comparison now passes 36 focused checks, including
actual checkpoint write/readback, complete household selection, unchanged input
values and missingness, signed and unsigned stable IDs, and full/pruned/local
weight and scope handling. A household with zero original weight can receive
positive calibrated weight without mutating its parent or other entity weights.
The comparison binds the full ordered weight vector and specification, including
rows outside the selected scope. It deliberately reports calibration ancestry
and release eligibility as unverified: the owning build must establish those
and seal its inputs across export I/O. See the
[scoped export acceptance](../experiments/us-common-frame-export36-20260910.json).

The earlier ten-node survey prefix assigns a block before cloning, and its
thirteen-node age-calibration extension passes cold execution and required
replay on invented originals. A twenty-node extension now carries atomic
geography through current financial imputation and required replay. Native
Delaware and complete national block support pass source normalization;
national/CD acceptance of the enriched population remains pending. The earlier joint tract/district
operator is a separate path. See
[Geography assignment](geography-assignment.md) for the country contracts,
implementation and source boundaries.

The 10 September sequencing correction now passes 70 inventoried controls on
invented inputs, across seven separately guarded runs, plus one separate default
compatibility case. The nine-node prefix completes its initial clones before
block assignment; the nineteen-node financial extension and twelve-node age
extension retain that geography through cold execution and required replay.
Assignment uses the qualified original-source key and stable clone discriminator,
so changing row order or numeric household coordinates does not redraw the same
clone. Observed state/PUMA constraints, input knownness, entity links, complete
population ownership and weights remain checked. Different clones may draw
different blocks; an occasional shared block is valid.

The financial donor FILTER requires the exact typed geography-validation artifact
and retains original ASEC DESIGN weights. The shared gate emits this artifact
only when requested, preserving its default API for UK consumers. Two test-only
corrections repair the CREATE source declaration and isolate the obsolete-order
budget counterexample from unrelated serialized storage differences; production
matches the independently reviewed candidate. See the
[scoped postclone acceptance](../experiments/us-postclone-geography-70-controls-20260910.json).

The earlier ten/twenty-node native runs retain their original order. Native
postclone execution, the new-order PUF host, held-out model quality, national/local
fiscal calibration and full release acceptance remain pending. The age control
uses invented targets and does not establish that complete calibration path.

| Area | Main source entry points | What to review |
| --- | --- | --- |
| Graph execution and storage | `packages/microcosm-graph/src/microcosm/graph/{decl,executor,manifest,store,codecs,schema}.py` | Typed inputs and outputs, ownership, artifact ancestry, cache identity and nullable round trips |
| Survey source preparation | `packages/microcosm-build/src/microcosm/build/us_runtime/{acs_population_catalogue,asec_population_catalogue,survey_population_preparation}.py` | Source identity, entity links, source universes and current income transformations |
| Combined survey and clone | `us_runtime/{graph_composed_population,graph_survey_population,graph_combined_clone}.py` | ACS and ASEC composition before the clone; native versus detail channels |
| Enrichment | `us_runtime/{full_puf_enrichment,graph_full_puf_enrichment,graph_current_survey_puf_transfer}.py` | Target ordering, conditioning, observed-value preservation and complete replay |
| Conditional models | `packages/microcosm-fit/src/microcosm/fit/{qrf_target,graph_legacy_train,graph_legacy_apply_matrix}.py` | Reusable model artifacts, deterministic draws and target regimes |
| Geography | `atomic_geography.py`; `us_runtime/{atomic_block_support,atomic_block_api_sources,survey_atomic_geography,graph_atomic_survey_population}.py` | Atomic block assignment after complete initial cloning, observed-source constraints, versioned mappings, stable clone identities and the optional typed validation prerequisite |
| Survey mass and calibration | `us_runtime/{survey_origin_budget,graph_survey_budget,graph_survey_calibration}.py`; `packages/microcosm-calibrate/src/microcosm/calibrate/{group_bounds,solve}.py` | Original survey mass, grouped bounds, fixed support and existing ungrouped solver behavior |
| Compatibility | `packages/microcosm-build/src/microcosm/build/{frame_checkpoint,us_runtime/__init__}.py` | Current-main APIs, checkpoint metadata and existing country consumers |

Paths beginning `us_runtime/` are relative to
`packages/microcosm-build/src/microcosm/build/`.
The branch contains a substantial consolidation; review each area and its tests
before deciding how to land it. Shared model work overlaps with
[#873](https://github.com/PolicyEngine/microcosm/pull/873). Anthony considers his
runtime proposal [#885](https://github.com/PolicyEngine/microcosm/pull/885)
superseded by this direction; it is reference material, not a prerequisite to
retain or merge separately. This does not imply that its GitHub PR is closed or
that this draft has received his approval.

## Parallel review and dependencies

Completion of this integration draft is not a prerequisite for all UK work.
Depend on identified shared changes where necessary, with independently reviewed
PRs and compatible interfaces. Keep the US native build, UK candidate repair,
source/target reconciliation and shared runtime work moving in parallel.

| Reviewer | Review focus in this draft | Work that can continue independently |
| --- | --- | --- |
| Anthony | Shared graph/Frame and calibration compatibility; target hierarchy and metadata flow into diagnostics/dashboard, including #855 | Target/dashboard changes follow #900; UK staging #896 proceeds independently of full UK graph conversion and native US PUF integration |
| Maria | Register each existing UK pipeline stage as a graph node using this branch's shared machinery; verify UK behavior and coverage | #900 lands before graph conversion; UK imputation, support/fit diagnosis and candidate validation continue; graph-node registration stacks on #893, with stage reordering in a later separate PR |
| US integration owner | Resolve overlap, maintain exact compatibility evidence, complete the US population and release checks, and extract shared changes agreed in review | US-specific source preparation, imputation, quality diagnostics and progressive local builds |

Coordination updated 10 September: Maria is deliberately stacking her UK
graph-node registration on #893. That specific dependent work should pin the
parent revision and coordinate shared-interface changes here. Her source/target
repairs and Anthony's staging work do not have to wait for the whole draft.
PR #900 merged at 14:54 UTC and is incorporated here from main at
`6f7571e1ab8c516154289bd8ebf75cb2646f03b5`. The combination preserves both parent
histories and does not rerun or recertify the frozen US controls.
Anthony's hierarchy/diagnostics [#855](https://github.com/PolicyEngine/microcosm/pull/855)
and its dashboard consumer follow the target reconciliation in
[#900](https://github.com/PolicyEngine/microcosm/pull/900). Registering UK graph
nodes and changing their order are separate review steps. The US integration
work must not build a competing full UK graph; its shared atomic-area adapter
is an input to Maria's owned integration.

Record shared dependencies as named changes with pinned revisions. Extract a
smaller shared prerequisite where useful, while preserving compatibility with
Maria's intentional stack. Proposed sequencing does not certify or merge any PR.

The shared post-clone geography interface now has a tested additive validation
artifact contract. The default gate API remains unchanged; UK registration can
adopt the optional typed prerequisite independently of the US stage ordering.
Existing target/source repairs can proceed on their reviewed base. Passing this
draft's invented controls does not certify Maria's UK candidate, and the final
US release need not gate her independent source and calibration work. No
whole-PR merge is implied by this review map.

## Verification and review scope

The first native twenty-node survey, block-geography and financial cold pilot
now passes at 1/1000: 1,584 source households and 3,464 people become 3,168
household records and 6,928 person records after cloning. Ten financial nodes ran
cold over the stored ten-node prefix, which retains the block assigned before
cloning. The composition retains location inheritance and attaches seven financial
fields while preserving the other population fields. It took 110.12 minutes and
11.45 GB peak memory. The final receipt and independent source/resource
postcheck pass. This is the original frozen 491-source implementation; newer
cache and budget changes are separate revisions. Required native cached replay
now also passes: all twenty final nodes and ten prefix nodes were cache hits,
the four non-manifest exports matched the cold bytes, and the portable manifest
retained the same content and stored-artifact identities. Replay took 74.17
minutes and 11.31 GB peak memory. Independent postchecks verified the recorded
result and exact code/resource identities without reopening native inputs or
exported payloads. PUF augmentation, model quality and calibration remain pending.
See the [scoped replay result](../experiments/us-native-atomic-financial-required-replay-20260910.json) and
[scoped native cold result](../experiments/us-native-atomic-financial-cold-20260910.json).

The previously published head `a85c9cbb79081a980e7d3afe8916c80153de5bf8`
passed 36 integration controls and 57 enrichment/replay controls on invented
inputs. The 36 cover eight solver, eleven snapshot/storage, nine geography and
eight facade cases. The 57 include cold execution, required replay, reconstruction
through a fresh Frame store, inherited tax-unit values and malformed ancestry
refusals. Source/control, model-declaration and resource pins matched before,
after and in external postchecks; the tax-benefit engine was not executed.

Those receipts are identified by SHA-256
`ad8ab6d8761db3d21343bc1fc60023c0dfa305e1367622630a5de6c68892d62a`
and `85a192f8c5c851648875d66b90bea6b8a4ee11e86e67d1875c3ca34a076ee014`.
These checks do not certify subsequent source additions or the entire repository.

CI exposed test-helper collection failures and a pool/registry circular import.
This revision adds the test helper directory to pytest's path and moves the
registry import after the pool functions it validates. Two fresh-process import
regressions are assigned to the US-engine CI lane; their runtime result remains
pending. No import-order result is inferred from an AST check.

The corrected checkpoint test selection passed all 58 invented controls against
the current integration checkpoint source, including nullable integer widths,
missing-value masks, deterministic round trips and malformed v4 refusals. It
took 5.36 seconds wall time and 348.5 MB peak RSS. All 45 source/control files,
three provider pins and two code resources matched in the external postcheck.
Receipt: `f8009ec6d07d076eff6a90ec97061d538756293547a21758ab2bc38fa5f7aab8`.

Fable's earlier replay review closed seven findings. A separate review of the
published consolidation identified the checkpoint test gap and additional
grouped-bound, metadata-store and facade coverage work. The portable grouped-bound,
fixed-support and metadata-store selection now passes all 117 tests with no skips.
The fixes reject nonfinite stored JSON as `StoreCorrupt` and normalize the deprecated
`apg` alias before grouped-mode validation. Source/control, resource and provider
bytes match before, after and in an external postcheck. Receipt:
`3674af9d610417897539da6eae766726d0f582ae6dd3bc2c3e550fefa45bab68`.
The historical platform fixture and complete facade import coverage remain separate.
Environment-specific prepatch byte fixtures do not establish portable CI byte
parity. Existing v1 graph stores remain preserved; v2 execution uses a new store
rather than silently promoting old frames.

Private runtime records and generated population artifacts stay outside this
source PR. Receipt identifiers are audit references, not reproducibility inputs.

The composed 156-case test run against source commit `8157300` at documentation
head `78d84f9` passed 152 cases and failed four current-survey cases with
`PRODUCER_CHANGED`. Its execution guard also recorded unexpected test-bytecode
and installed-model cache-directory probes. Source/control, model-source and
resource bytes remained unchanged. This is a failed integration run, not a
replacement for the earlier passing component evidence. Its receipt is
`67ff10fef45c58d2d6ea15c72b4311a4daa901dd2aae263fbb863d38c13bc7c8`.
The successor now passes all 314 selected invented integration and identity
checks with no skips or unexpected refusals. This includes the original 156
cases, five metadata-copy/mutation regressions and 153 encoder controls; seven
synthetic-issuance controls remain explicitly outside this selection. Shallow
copy, deepcopy and pickle previously added a class cache entry captured by
Python 3.14 annotation closures. An explicit immutable-metadata representation
avoids that mutation without weakening producer checks. Two stale ACS pins were
updated only after verifying AST equality with their accepted source revisions.
All 476 source/control, 5,983 model-source and 12 resource entries matched before,
after and in the external postcheck. The run took 253.5 seconds and 738.7 MB peak
RSS. Receipt:
`d940913f918af4455dab2b9e680d55f9efc7bb922f72ff3d8fe26b0530b070f7`.

A separate 16-case current seven-node age-development run initially passed 13
cases and failed three on exact diagnostic reconstruction. The checker omitted
two solver-option fields added by the consolidated solver. Reconstructing the
grouped solver's closing-state selection and empty selection receipt fixes that
mismatch while retaining exact whole-document comparison. All 16 tests now pass,
including actual seven-node execution and late target/successor mutation checks,
in 170.3 seconds at 541.9 MB peak RSS. All 469 source/control, 5,983 model-source
and 12 resource entries matched before/after/current checks. Receipt:
`5849dd42e914e4d8304a5d290ff451da45910e7ac9a61e76fff36468949ffe3a`.
Broader CI's missing helpers and stale schema, seed and runtime classification
expectations remain separate from these selected passing suites.

## Component evidence and remaining work

Separate component work has verified the national Census joint-geography source
artifact: 2,462 PUMAs, 87,841 joint cells, all 50 states plus DC and 436 districts
including DC. PUF source ingestion retained 207,692 ordinary records and excluded
four disclosure records. A corrected SCF wage-code interpretation passed source
preparation and model-mechanism checks. Those results belong to their respective
component revisions; they do not establish a calibrated result on this branch.

The corrected canonical-donor codec v2 was rebuilt from the same accepted typed
PUF source. All 207,692 returns and 59 outputs passed selected-cohort replay and
byte/value serialization checks in 24.37 seconds, using 1.95 GB peak RSS. The 481
source/control files and four resources matched their before/after/current pins.
Receipt: `6b334ca8df6c6ba721b60f84d28c43151a1e1b54e097616a3c0fa208daeb8cba`.
This run constructs the donor; it does not fit or place it onto survey recipients.

The identity encoder passed two 40-case preflights and four 40-case
timing runs in both execution orders. The 36 tiny invented cases retain exact
identity bytes and mutation checks. Nine of the 72 case/trial comparisons were
slower, including the small Python-string Frame in both orders; larger numeric
fixtures improved substantially. These timings do not establish genuine-build
speed. It is now integrated and passes the current 314-case selection, with the
newer current-wage projection preserved byte for byte. That projection was absent
from the benchmark's accepted baseline.

The following work remains open:

The [US and UK release path](us-uk-release-path.md) puts these implementation
items in the complete population-quality, calibration and consumer-release
sequence. Passing component controls is one step in that sequence.

1. Extend the accepted native twenty-node atomic and financial cold/replay
   pilot to the complete survey scale. Its
   invented cold/required replay, complete retained
   fields and geography, default three-feature versus explicit five-feature
   conditioning, and both final mutation refusals pass on invented inputs.
   Two test-code mistakes in the initial four-case run were corrected and
   rerun separately; the passing mutation cases were preserved. See the
   [composition acceptance](../experiments/us-atomic-financial-composition-acceptance-20260909.json).
   Native fit quality and admission of the financial result into subsequent
   calibration remain unassessed.
2. Integrate the adopted Social Security conditioning measurement and qualify
   recipient return roles before native PUF55 fitting. Thirty-one invented
   controls pass the authenticated filer/joint-spouse report-sum proxy, source
   preservation, missingness and refusal checks. An incomplete selected report
   uses a separately declared eight-predictor route; a known total uses nine.
   The helper uses the preparation's retained modeled roles, whose relationship
   to current-money tax-unit construction still needs explicit qualification.
   See the [measurement decision](puf55-survey-ss-measurement-decision.md) and
   [scoped control result](../experiments/us-puf55-survey-ss-measurement-31-controls-20260909.json).
   Route-aware graph fitting and attachment remain pending. The genuine
   207,692-return canonical donor has already passed its 55-output projection;
   that does not establish recipient matching. Complete survey
   beneficiary/component modeling, and supply SCF loan inputs. Separate component runs passed
   128 profile/compatibility controls and 26 current-survey predictor controls;
   see [the implementation status](current-survey-puf59-progress.md) and
   [donor construction and growth](puf2015-canonical59-and-growth.md).
   The genuine 207,692-return donor construction does not establish survey fit
   quality, complete enrichment or a releasable population.
3. Verify every applicable model input on each source/clone channel. The proposed
   national/CD profile retains 161 required model inputs, with state and county
   but without requiring the engine to consume block and tract. Independently,
   the build must retain one assigned Census block and derive its larger
   geographies. Prior wages remain excluded. A source operator's existence is
   not complete cell coverage.
4. Retain the earlier seven-node age-development result as historical evidence;
   the twenty-node cold/replay pilot in item 1 is the newer native milestone.
   The seven-node run passed at 1/1000: 1,584 source households,
   3,168 records after the support clone, final export/readback and owner/target
   verification, 71.1 minutes and 8.26 GB peak RSS. Its source is frozen at
   `a9895d8a5`, before the new Social Security, PUF55 and atomic-geography work.
   The prior two-hour failure and stale-derived-attachment failure remain failed
   historical runs. [The scoped acceptance record](../experiments/us-survey-age-development-20260909.json)
   does not certify enrichment, full national/CD calibration or a release.
5. Run real model evaluation, national/CD calibration, holdout checks and complete
   export/replay verification, followed by a dashboard bound to that exact file.
   Check every pruned analysis file separately.

These are active workstreams. No release, merge or deployment is implied by this
draft, and the source changes do not relax the outstanding acceptance checks.

The two-route numerical finalization layer now passes 46 invented controls,
including 110 actual target fits, 220 applications and comparisons against the
maintained whole-cohort finalizer for both routes together and either route
alone. The tests also refuse changed donor row order, weights and late raw or
donor mutations. The v2 numerical receipt seals the complete finalized candidate
before return and refuses a change during receipt encoding. This verifies the
numerical layer; typed model ancestry and complete survey attachment remain
separate graph checks.
The first canonical donor CREATE attempt failed during kernel construction
because its live-code checker followed a circular function closure. The
[failed observation](../experiments/us-puf55-canonical-create-failed-20260910.json)
is preserved. The cycle correction passed its ten new marker controls, but
the [39-case follow-up](../experiments/us-puf55-canonical-cycle39-failed-20260910.json)
still refused before CREATE because the code-state snapshot changed during
initialization. The changing member was the serialized Python code object;
the corrected marker retains the actual code object and its immutable fields.
After correcting three obsolete test accesses to the public PopulationView API,
all 47 canonical CREATE/converter controls pass, including cold execution,
required replay and teardown. The corrected numerical and canonical sources are
adopted locally. These checks use invented donors; the native full PUF host
remains pending. See the
[46-control numerical evidence](../experiments/us-puf55-numerical-output-seal-46-controls-20260910.json)
and [47-control canonical evidence](../experiments/us-puf55-canonical-create-47-controls-20260910.json).

All 149 selected calibration controls now pass on the combined US/UK solver,
including grouped bounds, fixed zero support, informed gates, budget search,
ordinary best-iterate behavior and strict diagnostic payloads. The run took
4.57 wall seconds at 394 MB peak memory; independent checks confirm the exact
34 source files, copied fixtures and all 149 unique test results. The preceding
run stopped before tests because two still-denied import probes exceeded their
reporting ceilings; the fresh invocation changed only those reporting ceilings
and run identities. No data access was added. This is invented-data solver
verification, separate from native national/CD calibration; see the
[calibration evidence](../experiments/us-calibration-consolidation-149-20260910.json).

Recipient qualification now covers all 28 distinct original public controls:
27 passed in the original run and the five-case corrected detached-output run
passed, with four overlapping controls. The original run remains failed because
one NaN mutation test did not change its input bits; its correction flips a bit
in the actual float64 storage. This is a compound result across two source
snapshots, not a single passing 28-case run. Exact original-source issuers and
the 22-node cold/required graph are included. See the
[recipient evidence](../experiments/us-puf55-public-recipient-controls-20260910.json).

The population-only Census API adapter passes thirty-five invented controls.
The native Delaware source control preserves all 15,317 populated blocks and
reconciles them to the independent Census state total of 989,948. It counts
4,881 zero-population blocks, checks CD119/PUMA joins and exactly reads back its
79,769-byte support artifact. The run took 10.94 seconds and 1.735 GB peak RSS.
The first attempt correctly refused an incomplete ZIP-member declaration;
a separate metadata probe established the six-member archive roster, and the
corrected check still decompresses only `NationalCD119.txt`. The
[accepted source control](../experiments/us-atomic-native-de-corrected-1-control-20260909.json)
preserves the original failed evidence. This is not national population or
release acceptance. Delaware-only support must not enter the national survey
graph.

The subsequent population-only national acquisition and normalization now pass
for all fifty states and DC. The 104 exact sources produce 5,769,942 populated
blocks, independently reconciling every state to a total of 331,449,281 people.
Every block retains its identity and population, with complete CD119/PUMA joins.
The full support artifact is 28,862,508 bytes and passes exact serialization and
readback. Normalization took 82.97 seconds and 6.814 GB peak RSS. Its SHA-256 is
`5edc0e77471ba31d550a1eed416d5b46ada0a35425718eb87cfabe4d66fe4960`;
see the [national source control](../experiments/us-atomic-native-national-1-control-20260909.json).
Postchecks reauthenticated code and compared the 104 native pins from bounded
receipts without reopening source bodies. National support normalization and
household assignment in the small native financial pilot now pass. Evaluating
geographic fit and calibrating the enriched survey remain separate acceptance
steps.

The financial successor's unchanged positive control now passes with both
original fixture teardowns and final code/resource checks: 647.19 CPU seconds,
651.21 wall seconds and 570.9 MB peak RSS. This separately reviewed invocation
used a 1,200 CPU-second budget; the earlier 600-second attempt remains recorded
as incomplete. No memo or profiler was used. The [positive control evidence](../experiments/us-financial-successor-positive-20260910.json)
checks financial replay and a subsequent weight-only change. The [seven remaining refusal and callback controls](../experiments/us-financial-successor-remaining-controls-20260910.json)
now also pass with their original shared fixture and teardowns: 798.61 CPU seconds,
802.89 wall seconds and 563.6 MB peak RSS. All eight original admission controls
are accepted on the same source snapshot. A subsequent source review found an
additional detached-view callback gap in the budget and weight-only views.
The correction is adopted after four targeted controls passed in 543 wall
seconds, including nine callback branches and independent source/resource checks.
The original eight and corrected four are separate source-version checks; see
the [correction evidence](../experiments/us-budget-detached-view-correction-20260910.json).
The native pilots exclude this overlay, and native calibration remains separate.

A separate, bounded profile of the unchanged invented cold/required financial
fixture passes its test, original teardown and all final code/resource checks:
303.90 CPU seconds, 308.26 wall seconds and 594.3 MB peak RSS. This measures the
fixture rather than the additional financial and weight-only admission work.
See the [profile evidence](../experiments/us-financial-fixture-profile-20260910.json).
Loaded ACS code verification accounts for 180.20 cumulative seconds inside the
298.15-second profiled fixture. Its nested function checker runs 2,009,108 times
and uses 91.05 self seconds; complete Frame identity uses 7.31 cumulative seconds.
Cumulative timings overlap and profiling adds overhead. The invocation-local
code comparison memo now passes all 18 focused invented controls, including
changed source, aliases, globals, closures, mutable constants and the historical
catalogue caller. Source and resource checks pass before, after and in the
external postcheck. See the [control evidence](../experiments/us-acs-loaded-code-controls-20260910.json).
The paired full-fixture run also passed correctness and final checks, but took
319.79 wall seconds versus 308.26 for the baseline. The memo added eligibility
work and demonstrated no speed improvement, so its implementation was not
adopted. The exact experimental patch and [paired comparison](../experiments/us-acs-code-memo-comparison-20260910.json)
remain recorded. A separate [one-call diagnostic](../experiments/us-acs-producer-profile-20260910.json)
now passes without constructing a fixture: the original producer takes 0.656
profiled wall seconds. Its 157 compiler calls take 0.238 seconds, including
0.122 seconds inside 103 AST parses; those overlapping times must not be added.
A bounded bytecode compilation cache now passes 25 invented controls and final
code/resource checks, with fresh source reads and every loaded-function check
retained. No native-data speedup is established. The compilation cache has separately passed
the complete invented fixture and is adopted; the native financial cold and
required-replay pilots passed with their original frozen code. The earlier
startup refusal remains preserved and confers no data acceptance.

Current CI separately reports stale source-attested spec/seed fingerprints.
The worker resource identity test now follows the actual lazy import closure
and includes an invented opened-JSON regression; its new CI result is pending.
Neither the component passes nor this profile establishes a green consolidation.

## Related reviews

- [Microcosm → Orrery exporter, #888](https://github.com/PolicyEngine/microcosm/pull/888):
  already published separately. It preserves the supplied schema metadata and
  exact large integers; graph visibility does not infer domain verdicts.
- [Orrery review index, #6](https://github.com/TheAxiomFoundation/orrery/pull/6)
  and [search improvements, #4](https://github.com/TheAxiomFoundation/orrery/pull/4).
- [Dynamics graph integration, #420](https://github.com/PolicyEngine/microcosm-dynamics/pull/420):
  owned by the separate retirement-model task and tested on synthetic inputs.
- [Candidate methods and paper updates, site #72](https://github.com/PolicyEngine/microcosm.institute/pull/72):
  public component evidence and remaining release checks.

The accepted local shared viewer remains a separate consumer. There is no package
upgrade or competing generic graph shell in this consolidation.
