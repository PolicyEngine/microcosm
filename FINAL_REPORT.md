# Final report: pkg3 r4 — main merge resolved by porting the calibration policy into the F0 spec

2026-08-23. This section reports the r4 continuation (the `origin/main`
merges and the F0 home for the post-transfer calibration policy). The
receipt-validation report it supersedes as the branch's latest result
remains below as history.

## Outcome

PR #742's source conflict against the locally available `origin/main` is
resolved by three merge commits: `e66074ad` (main at `b4dfa0e7`, carrying the
F0 policy port), `e0947020` (main at `2aa96795`, #733), and `1fc9055e` (main at
`055dcfaf`, #740). `origin/main` at `055dcfaf` is an ancestor of the final
tree. Commit `87e638bd` closes the three deterministic merge-union guards
exposed by the complete build shard. The full five-shard inventory,
repository-wide Ruff, coverage attestation, and final wheel boundary are
green.

The branch's post-transfer calibration policy declaration — formerly in
`specs/us_imputation_lineage.yaml`, which main's F0 migration deleted —
now lives where imputation models are authored:

- a closed, typed exact variant of `regime_gated_qrf_model` carrying the
  `post_transfer_calibration_policy_v1` payload
  (`packages/microcosm-build/src/microcosm/build/spec_engine/schema/imputation.schema.json:746-807`);
- the authored declaration in
  `packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:94-192`,
  with its generator in `tools/us_bundle_generation/imputation.py:269,2337`;
- a ninth `post_transfer_calibration` stacked-authority component projected
  at authority binding version 11
  (`spec_engine/stacked_authority_semantics.py:414,466,618-622`;
  `us/spec/battery.yaml:824-826`);
- the spec-matches-code identity test re-ported to the generated-bundle
  boundary in
  `packages/microcosm-build/tests/test_imputation_lineage_spec.py:99-103`,
  combined with main's producer-registry test, holding the authored payload
  byte-equal to `post_transfer_calibration_policy_identity`
  (`us_runtime/post_transfer_calibration.py:301`) in code.

No comparator, band, threshold, seed, fold, gate, or sample contract
changed. No host pool build ran. Nothing was published and PR #742 was not
merged. This report-closing commit is the payload for the single permitted
push to `origin/battery-pkg3-two-part`.

## Anti-rot chain

The port adds 92 authored configuration fields. Every count, sha, and pin
site the chain covers was walked and is test-green on the final tree:

- `EXPECTED_CONFIGURATION_FIELD_COUNT = 41_471`
  (= 32,252 authored + 9,219 resolved;
  `spec_engine/field_usage.py:29-31`), with the mode/effect counts at their
  matching literals and the `imputation_models` usage claim moving from 4
  fields to 96 with claim SHA-256
  `e4d6b6b747fcec1c027e0f1c2d1905274c0426217a61383b02e75baadb93db4d`
  (`field_usage.py:359-362`).
- The pointer-inventory sha and the regenerated
  `docs/evidence/spec-engine/us-f0-coverage.json`:
  `tools/spec_engine_coverage.py --check` reports 41,471/41,471
  configuration fields and 40/40 inventory checks with no drift. The pinned
  inventory SHA-256 is
  `2daa3ee07ac2e5d5ab731348edbca8c7a58438e9819d1ac3707070c7500a1c63`
  (`tools/spec_engine_coverage.py:42-45`).
- Test literals in `test_spec_engine_field_usage.py` and
  `test_spec_engine_coverage_tool.py`.
- Bundle identity pins. The US pin
  `d3de6760727cfcb6800209670d37e02b373d8dcda19f8ad054aa9d410e0efbb0` in
  `test_us_multispine_pool_tool.py` was recomputed via
  `load_bundle("us").spec_sha256`. The loader golden vector moved to
  `b1ab6ab000689cc03e1088d422d7b6328b9bff2878f39448a529e5513c03ed14`
  (`test_spec_engine_loader.py:237-239`); single-edit bisection proved the
  mover is the branch's edits to seed-attested kernel modules (the seed
  protocol digests attested kernel bytes into the resolved bindings inside
  the hashed envelope, `spec_engine/loader.py:404-418`), not the shared
  schema edit — the legitimate identity movement `CLAUDE.md` documents.

## Merge resolutions

Merge 1 (`e66074ad`, main at `b4dfa0e7`): `specs/us_imputation_lineage.yaml`
stays deleted (lineage derives from the authored US bundle);
`tools/build_us_multispine_pool.py` keeps main's immutable-safe direct
dataclass walk plus this branch's omission of empty `target_regimes` in both
the mapping and dataclass paths (`tools/build_us_multispine_pool.py:3826-3847`);
root journals are unioned and historicized; the port and chain above are
folded in. `test_imputation_lineage_spec.py` combines main's generated
producer-registry assertion with the branch's policy-identity assertion.

Merge 2 (`e0947020`, main at `2aa96795`): sole conflict was the UK entry in
`test_spec_engine_country_bundles.py`. Both parents' pins were computed on
trees missing the other side's envelope movers (this branch's
stacked-authority version-11 binding vs #733's UK sources/runtime
retarget), so the union pin was recomputed fresh via `load_bundle("uk")`:
`bb7110699a9cb7a346ecd478f55b7a1c57bdc0c3f3283705b8a9b51830207193`. BE
(`bf022118…`) and US (`d3de6760…`) recomputed identically on the union —
the `b4dfa0e7..2aa96795` range touches no BE- or US-side envelope input,
and its `sources.schema.json` edit sits in the grammar receipt outside the
hashed envelope (`spec_engine/loader.py:227-229,401`). Main's re-cut UK
gate-battery digests and `microcosm-data` contract pins merged clean and
pass unchanged.

Merge 3 (`1fc9055e`, main at `055dcfaf`): the sole conflict was again the UK
country-bundle identity after #740 added the E8 UK spec stages. Fresh
union-tree `load_bundle` calls produced BE `bf022118…`, UK `8bf62b6e…`, and
US `d3de6760…`; the UK test pin was resolved with the exact union value
`8bf62b6e47583da1bdad1b71be1e705f424e6e245880e90f4411aba57fa5eb93`.
Main's UK gate/data pins merged cleanly.

The complete build shard then exposed three exact union misses, fixed in
`87e638bd`: the direct dataclass serializer now applies the empty
`target_regimes` omission (`build_us_multispine_pool.py:3841-3847`); the
authored-SHA audit allows only the exact policy-identity SHA path in addition
to the two external-asset pins (`test_us_spec_bundle.py:707-770`); and the
closed runtime import graph is pinned at 66 after the branch-added
`post_transfer_calibration.py` module (`test_us_spine_blindness.py:3270-3315`).

## Verification on the final tree

- The final focused policy/count/pin/gate/data/serializer/SHA/import-graph
  suite passes, as do the two unchanged trade-publication crash tests after
  shared-host load fell below the point where their child CLI import exceeded
  the existing 60-second bound. No timeout was changed.
- Full five-shard suite on the final union. Fit, calibrate, frame, and data ran
  in one pytest process each. The build root's single-process behavioral run
  was green (6,248 passed, 39 skipped) but reached 15.989 GiB, so that resource
  receipt was rejected. The authoritative capped rerun covered all 262 build
  test files in 17 fresh pytest processes, reproduced the exact 6,248/39
  aggregate with exit 0, and peaked at 10.284 GiB; no guard split or
  intervention was needed.

  | Shard | Result | Peak RSS |
  | --- | --- | ---: |
  | `microcosm-fit` | 93 passed, exit 0 | 0.862 GiB |
  | `microcosm-calibrate` | 203 passed, exit 0 | 0.449 GiB |
  | `microcosm-frame` | 294 passed, 36 skipped, exit 0 | 6.492 GiB |
  | `microcosm-data` | 275 passed, 1 skipped, exit 0 | 11.052 GiB |
  | `microcosm-build` | 6,248 passed, 39 skipped, exit 0 | 10.284 GiB |

  Build authoritative receipt: `files=262 batches=17 passed=6248 skipped=39
  exit=0 max_rss_bytes=11042193408 guard_gib=12.0`.
- Repository-wide `uv run ruff check .` and `git diff --check`: pass.
- Final wheel boundary: all five shard wheels build offline. Reinstalled into
  the existing clean, lock-constrained wheel venv, all five namespaces import
  from its site-packages prefix with `policyengine_us` absent;
  `tools/spec_envelope_digests.py be uk` runs there; and installed-wheel
  `load_bundle` reproduces exact BE `bf022118…`, UK `8bf62b6e…`, and US
  `d3de6760…` identities. A second brand-new offline venv could not resolve
  uncached third-party packages, so no network-dependent claim is made.

## Boundaries

This lane ran no pool build (the host queue owns those builds; the 1% baseline
at `_buildo-runtime/out/battery-verify/baseline1pct/` stands for
before/after diffs). Certification, publication, and release-chain
mutation remain with their existing owners. PR #742 is not merged.

---

# Final report (historical): pkg3 post-transfer receipt validation failure #2

## Outcome

Fixed at executable commit `a932974f`.

The supplied weeks-unemployed failure was a receipt-generation bug, not an
invalid carrier model and not a count-target exception. Candidate capacity and
prefix selection reduced the same ordered weights through different float64
paths, so the prefix exceeded its declared candidate mass by one ULP. The
required cross-target audit then exposed the same class of bug one level up:
the first repair composed maximum capacity from independently rounded
partition endpoints, putting both child-support maxima 42 ULP above the whole
recipient mass.

The complete fix changes both generating relationships:

- one immutable `_PrefixSchedule` now supplies each candidate endpoint and all
  prefix-selection evidence (`post_transfer_calibration.py:282-287,471-515,844-872,891-928`);
- maximum capacity is generated from the row union
  `fixed_positive | allowed_positive | zero_candidates`, zero-masked onto the
  recipient-weight vector with identical length, order, and reduction topology
  to `recipient_total` (`post_transfer_calibration.py:823-885`).

The exact validator inequalities are unchanged: maximum must not exceed its
recipient superset, and a reported prefix must not exceed its candidate set
(`post_transfer_calibration.py:1457-1475,1499-1529`). No tolerance, threshold,
band, gate, comparator, seed, fold, target, or carrier constraint changed.

Both SHA-pinned no-build harnesses now validate. All five package test roots,
repository-wide Ruff, touched-file formatting, and whitespace checks are
green. This lane ran no host build and made no push. The serial host owner owns
the next 1% rerun against `a932974f`.

## No-build checkpoint reproduction

The read-only stage root was:

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/pkg3/pool.checkpoints/stacked/8f5077d6a1d5440b241f22fe4d20ad1d889924a27d094cb669e1035f9306546b
```

The current no-build checks are:

```text
uv run python tools/reproduce_us_post_transfer_weeks_checkpoint.py \
  --checkpoint-stage-root <stage-root-above> --expect valid
uv run python tools/audit_us_post_transfer_child_support_checkpoints.py \
  --checkpoint-stage-root <stage-root-above> --expect valid
```

### Weeks-unemployed failure

`tools/reproduce_us_post_transfer_weeks_checkpoint.py` validates the assembled
Frame plus unemployment-compensation and weeks target file, identity, and raw-
draw hashes. It reconstructs only the native clone-0 vectors and calls the live
kernel and strict validator; it performs no fit, DAG execution, artifact write,
or build (`reproduce_us_post_transfer_weeks_checkpoint.py:1-6,27-58,76-143,146-254`).

Pinned artifacts:

| Artifact | File SHA-256 | Identity SHA-256 | Raw-draw SHA-256 |
| --- | --- | --- | --- |
| `assembled.checkpoint.h5` | `5ce1815fc44dc43c7c24ccf27526852b8f1bddbdfe371255410a22f9b56ac015` | whole file pinned | n/a |
| `019__unemployment_compensation.h5` | `dc6637936ed4bd0322d38eaa3a4920fd137565f314387db3b3fdc7dfd6bc3086` | `708722093ca610426175998d50bbb6663585b07ffef912899f17adc90520f51f` | `e32d1559668e10b24abad8e1d639e4dbade964a712925bfe8f56d3136b839840` |
| `000__weeks_unemployed.h5` | `898397733aa3e5d8ec7d6679cb16a0504e826e25d23ca2c788f4397e0e061a43` | `d0d554ba05045e39a07f0f9515c83bbf754f067df12b8247f4bf3866162c4bdd` | `0214c8dcbc118676336069b906a07ee6145f2178542b6c5b4fb5899ad62d09f3` |

The production owner selects ASEC clone-0 reference rows, ACS clone-0
recipient rows, transferred nonnull mutable cells, and positive-UC mutable rows
for both the weeks allowed and addition masks
(`stacked_spine.py:8960-8971,8995-9032`). The pinned replay contains:

- 38,604 native person rows;
- 4,311 reference and 34,293 recipient/mutable rows;
- 134 positive reference rows;
- 24 initial recipient positives, all disallowed; and
- 32 positive-UC addition candidates
  (`reproduce_us_post_transfer_weeks_checkpoint.py:146-193`).

At reproduction commit `4cc41652`, the harness with `--expect invalid` exits
zero only for the exact supplied failure. Candidate capacity is
`85,676.23791782455`; the ID-ordered upper prefix is
`85,676.23791782456`; the excess is `1.4551915228366852e-11`. The sole false
relationship is:

```text
upper_prefix_mass <= addition_candidate_mass
```

It raises exactly:

```text
ValueError: Frame post-transfer calibration person/source_operator_weeks_unemployed/weeks_unemployed: match-reference carrier capacity relationships are invalid.
```

The harness pins that predicate, both floats, and the complete error rather
than accepting any aggregate validation failure
(`reproduce_us_post_transfer_weeks_checkpoint.py:210-231,257-286`).

Against `a932974f`, `--expect valid` reports candidate and upper prefix both
`85,676.23791782456`, zero prefix/candidate delta, no failed relationships,
and a valid receipt. The attainable-union maximum under the recipient
reduction topology is `85,676.23791782453`.

### Child-support cross-target reproduction

`tools/audit_us_post_transfer_child_support_checkpoints.py` reconstructs the
same native clone-0 support and half weights for both child-support targets.
It pins each target's whole-file, identity, and raw-draw hashes, then requires
both receipts—not merely one—to match the requested exact red or green state
(`audit_us_post_transfer_child_support_checkpoints.py:1-80,91-207,210-302`).

| Artifact | File SHA-256 | Identity SHA-256 | Raw-draw SHA-256 |
| --- | --- | --- | --- |
| `000__child_support_expense.h5` | `d119075e19fb767f3d8d24c7c0149d0df1ed963774a4b93d96974a72b3ac9bfe` | `41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab` | `8b2845aff0aa0695d98ae30828523bf6bca9c5d4ed5d2d91d2d1a636bb917600` |
| `001__child_support_received.h5` | `66120896d5793f3d737f9ffac2058e2196992e357f8d869f4b31b259d041b3aa` | `41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab` | `ea7f2eebb430b654acc639ef6ee6ed482207ffd74d54ba3a47cb55056813a381` |

Against the incomplete candidate-only repair `d7b12bab`, both receipts fail
only:

```text
maximum_attainable_mass <= recipient_total
```

For each target, recipient total is `79,926,522.10879111`; the partition-
composed maximum is `79,926,522.10879174`; the excess is
`6.258487701416016e-07`, or 42 ULP. The underlying endpoints are:

| Target | Allowed-positive mass | Addition-candidate mass | Old composed maximum |
| --- | ---: | ---: | ---: |
| `child_support_expense` | `71,696.09739141785` | `79,854,826.01140033` | `79,926,522.10879174` |
| `child_support_received` | `180,209.75664861224` | `79,746,312.35214312` | `79,926,522.10879174` |

Against `a932974f`, both strict receipts validate. Their attainable-union
maximum equals recipient total exactly, `79,926,522.10879111`, while the
independently rounded diagnostic partition sum remains
`79,926,522.10879174`. This demonstrates that the generating set relationship,
not the validator, was repaired.

## Root cause and semantic decision

Float64 addition is order-sensitive. The original code used a masked
`ndarray.sum` for candidate capacity and a separately ordered `np.cumsum` for
prefix selection. The first repair correctly unified those two values, but it
then added independently rounded fixed, existing-positive, and zero-candidate
endpoints to describe a different claim: the maximum mass of their row union.

Both strict invariants are semantically correct:

1. a prefix cannot exceed the candidate set from which it was selected; and
2. an attainable subset cannot exceed its recipient superset.

The final maximum implementation retains the recipient vector's length and
order and replaces unattainable entries with zero. With finite nonnegative
weights, each attainable leaf is less than or equal to its corresponding
recipient leaf, and the identical reduction topology preserves that ordering
through every floating-point addition (`post_transfer_calibration.py:775-783,823-885`).
It does not use `min`, `nextafter`, a tolerance, or a post-hoc clamp.

The regressions lock both numerical mechanisms for every late
`match_reference` declaration:

- the exact 32 production weeks weights distinguish masked sum from ordered
  prefix by one ULP;
- a four-weight case makes independently rounded capacity partitions exceed
  their whole set by one ULP; and
- a constrained proper-subset case makes a compressed subset sum
  `0x1.433526fbe1946p+48`, or `0.0625`, greater than its superset
  `0x1.433526fbe1945p+48`; the same-topology union validates exactly
  (`test_us_post_transfer_calibration.py:544-753`).

Weeks remains a valid positive-carrier calibration target. Its source accepts
only integer `-1` or `0..52` and maps `-1` to zero; its QRF path rounds, clips,
positive-UC-gates, and revalidates `0..52`; its carrier event is `weeks > 0`
(`weeks_unemployed.py:791-800,911-983,1218-1222`). Post-transfer amount mapping
uses only positive reference-donor support (`post_transfer_calibration.py:577-626,690-705`).
Count-valued support therefore does not invalidate weighted carrier capacity.

The ACS runtime's explicit discrete-numeric set contains only two mortgage-year
targets; other numeric targets use the ordinary numeric encoding
(`acs_transfer.py:129-138,3035-3117`). QRF's at-most-32-value “near-discrete”
branch is a leaf-storage optimization, not a carrier semantic type
(`microcosm-fit/qrf.py:388-401,482-503`). Annual child-support and disability
amounts also entered that optimization in the host log, which independently
rules out treating it as a weeks-specific count exception.

## Seven-target late-transfer audit

The immutable registry declares seven late targets. Six use
`match_reference`; disability benefits uses `preserve_recipient` and never
emits capacity or prefix evidence
(`post_transfer_calibration.py:208-258,840-932,1319-1335`).

| Late target | Evidence and verdict |
| --- | --- |
| `child_support_expense` | Covered. Nonnegative annual `CHSP_VAL`, not a count (`child_support.py:166-201,369-383`). Its pinned checkpoint fails the old whole-capacity relationship and validates the final union mechanism. |
| `child_support_received` | Covered. Nonnegative annual `CSP_VAL`, not a count (`child_support.py:166-201,369-383`). Its pinned checkpoint has the same red/green proof. |
| `disability_benefits` | Inapplicable to this capacity bug. It is a nonnegative annual two-slot amount excluding workers' compensation (`disability_benefits.py:184-220,382-395`) and uses `preserve_recipient`; its inspected checkpoint keeps before/after carrier mass at `42,658.57948297383` with `capacity=None` and `selection=None`, as required by the preserve-mode receipt branch (`post_transfer_calibration.py:1319-1335`). |
| `weeks_unemployed` | Covered. Sole semantic count target, integer `0..52`, with carrier additions constrained to positive-UC mutable rows (`weeks_unemployed.py:791-800,911-983,1218-1222`; `stacked_spine.py:8995-9008`). Exact pinned red/green replay proves reducer order caused the failure. |
| `workers_compensation` | Covered. Nonnegative annual `WC_VAL`, not a count (`workers_compensation.py:143-184,337-355`). It uses the default mutable carrier/addition masks (`post_transfer_calibration.py:786-812`) and the shared six-spec regressions. |
| `spm_unit_energy_subsidy` | Covered. Nonnegative measured `SPM_ENGVAL`, checked within unit and reduced to SPM-unit float64 (`energy_subsidy.py:157-233,537-557`). Its entity grain changes the weights, not the set/reduction mechanism; the shared regressions cover its declaration. |
| `pre_subsidy_care_expenses` | Covered. Nonnegative monetary care expense. ACS reconciliation restricts carriers to qualifying people and at most one per tax unit; the late owner admits one stable zero candidate per empty unit (`acs_transfer.py:660-739,1277-1299`; `stacked_spine.py:8728-8746,8977-8986`). The proper-subset six-spec regression covers this constrained structure. |

Current zero-based late-DAG positions are child support 24, disability 25,
weeks 30, workers' compensation 31, energy subsidy 32, and adult care 34.
Registry scheduling and stacked execution are deterministic, and each group
calibrates before returning (`us_late_producer_registry.py:1338-1396,2013-2019`;
`stacked_spine.py:10054-10095,10927-10931`). The failed host run produced child,
disability, and weeks checkpoints but stopped before workers, energy, and adult
care. Verdicts for those later targets are therefore source/mask proofs plus
shared-kernel regressions, not claims of nonexistent checkpoint replay.

## Verification

On the exact tree committed as `a932974f`, these commands ran serially under
the owner-provided memory guard and exited zero:

```text
uv run pytest packages/microcosm-fit/tests -q
uv run pytest packages/microcosm-calibrate/tests -q
uv run pytest packages/microcosm-data/tests -q
uv run pytest packages/microcosm-frame/tests -q
uv run pytest packages/microcosm-build/tests -q
uv run ruff check .
uv run ruff format --check <four touched Python files>
git diff --check
```

The focused post-transfer file passed all 47 cases. The complete build root
also covered stacked-spine, late-DAG, multispine pool, H5, pool-tool, terminal
receipt, and owner-mask paths. Only established skips and warnings appeared.

Both current checkpoint commands exited zero with `--expect valid`. Detached
temporary worktrees proved the hardened red side: weeks against `4cc41652` and
both child receipts against `d7b12bab`. The temporary worktrees were removed.

No host build ran from this lane. The read-only host log and checkpoint tree
were not modified. The owner-provided untracked `.codex-memory-guard.py` and
`_BUILD-FAILURE-1PCT.txt` remain unchanged. Nothing was pushed.

## Commit lineage and handoff

- `b533bc61` — open and commit the progress journal;
- `4cc41652` — add the SHA-pinned weeks reproduction and red regression;
- `d7b12bab` — bind candidate capacity to its ordered selection schedule;
- `47742720` — record the cross-target child-support escalation;
- `a932974f` — generate maximum capacity from the attainable row union, add
  the child red/green harness and exact proper-subset regressions, complete the
  seven-target audit, and record the green suite; and
- the following documentation-only commit — close `PROGRESS.md` and publish
  this report without changing the tested executable tree.

The next authorized action is the serial host owner's 1% rerun at executable
commit `a932974f`. Certification, publication, and release-chain mutation stay
outside this lane.
