# Battery package 3 lane notes

## Main/F0 merge continuation — 2026-08-22 02:09Z

- Starting tracked revision: `c22e5d37`; local `origin/main`: `b4dfa0e7`.
- This continuation owns only the main merge, the F0 home for the existing
  post-draw calibration policy declaration, its spec/code identity test, and
  the resulting closed-world anti-rot updates. It does not run a host build,
  change a comparator/gate/threshold, certify an artifact, publish a release,
  or merge PR #742.
- Main's generated F0 bundle supersedes `specs/us_imputation_lineage.yaml`, so
  the old file will remain deleted. The policy itself must survive by becoming
  a closed, typed exact variant at the authored imputation-model location.
- Earlier notes below remain source-cited history. Their branch-state and
  handoff language is not current for this continuation.
- Salvage audit: `ea21353d` has parent `bb94f789` and matches the current
  tracked merge worktree byte-for-byte. It preserved literal conflict markers
  and accidentally tracked `.codex-memory-guard.py` plus
  `_BUILD-FAILURE-1PCT.txt`; it contains no F0 schema/spec/anti-rot edit. The
  live automatic merge of `tools/build_us_multispine_pool.py` is retained
  because it carries main's F0 adapter path and this branch's omission of empty
  `target_regimes`. No salvage path was checked out or cherry-picked; the
  marker snapshots, old YAML, and accidental diagnostic additions were
  discarded while the owner files remain untracked and untouched.
- Salvage audit, second and third snapshots (2026-08-22 continuation):
  `e2402fd4` and `abcf7dc1` also have parent `bb94f789`. The worktree at
  recovery time already equaled `abcf7dc1` for every tracked path (verified
  with `git diff abcf7dc1 --stat`: the only differences were the two
  salvage-only debris files), so the third snapshot's merge resolution, F0
  policy port, and anti-rot edits were recovered in place — nothing was
  checked out or cherry-picked. Taken: the staged `b4dfa0e7` merge
  resolution; the closed `regime_gated_qrf_model` exact variant with the
  `post_transfer_calibration_policy_v1` payload
  (`packages/microcosm-build/src/microcosm/build/spec_engine/schema/imputation.schema.json`);
  the authored declaration and its generator
  (`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml`,
  `tools/us_bundle_generation/imputation.py`); the ninth
  `post_transfer_calibration` authority component and version-11 binding
  (`packages/microcosm-build/src/microcosm/build/spec_engine/stacked_authority_semantics.py`,
  `.../us/spec/battery.yaml`); and the walked anti-rot literals
  (field_usage/inventory/coverage/test pins). Discarded:
  `.codex-memory-guard.py` (carpool tooling debris) and
  `_BUILD-FAILURE-1PCT.txt` (stale r3 continuation note; its 1% failure was
  already closed on-branch by `a932974f`..`c22e5d37`).
- Chain site the salvage had not reached: the loader golden vector and every
  bundle `spec_sha256` pin move on this merge, and the generating mechanism was
  verified by single-file bisection, not assumed. `spec_sha256` hashes only the
  spec envelope — country, manifest schema version, normative projections, and
  resolved bindings (`spec_engine/loader.py:404-418`, `canonical.py:299-319`);
  the schema-set receipt lives in the grammar receipt outside that envelope
  (`loader.py:227-229,401`), which is why main's `sources.schema.json` property
  additions moved no pin on main. The resolved bindings embed the selected seed
  protocol's wire form, whose kernel attestations digest the bytes of the
  attested kernel modules (`spec_engine/seeds.py:40-71,92-97,189-199,309-370`),
  and the attested list includes `microcosm.build.us_runtime.acs_transfer`,
  `weeks_unemployed`, `puf_qrf_chain`, and `microcosm.fit.qrf` — modules this
  branch's calibration repair changed. Reverting each of this continuation's
  schema/semantics/tool edits one at a time left the synthetic golden value
  unchanged at `b1ab6ab0…`, confirming the branch's attested-kernel edits (not
  the shared-schema edit) are the mover, exactly the legitimate identity
  movement `CLAUDE.md` documents. Pin re-cut in
  `packages/microcosm-build/tests/test_spec_engine_loader.py::test_semantic_hash_has_golden_vector_and_surface_separation`.
- Verified unaffected: the UK gate-battery mirrors recompute from the live UK
  gates manifest, not the schema registry — all ten
  `packages/microcosm-build/tests/test_gate_battery_contract_pins.py` tests
  pass on this tree with the imputation-schema variant present, so the
  `microcosm-data` contract pins from `b4dfa0e7` stand here without a re-cut.

## Second merge: `origin/main` at `2aa96795` — 2026-08-22 headless run

- The first merge (with the recovered port, the re-cut loader golden vector,
  and the changelog fragment) is commit `e66074ad`; the second merge is
  commit `e0947020`. `origin/main` had advanced past the staged
  `b4dfa0e7` MERGE_HEAD by `1d066e5d` (#733 review fixes), `d70ea39c`
  (post-#735 UK spec / gate-battery digest re-cut), and merge `2aa96795`
  (#733 UK FRS 2024-25 retarget).
- Sole textual conflict: the UK entry of `EXPECTED_RESOURCES` in
  `packages/microcosm-build/tests/test_spec_engine_country_bundles.py`. Both
  parents' pins were computed on trees missing the other side's envelope
  movers — this branch's `aa32c4c9…` includes the stacked-authority
  version-11 binding but not #733's UK sources/runtime retarget; main's
  `e12a2cb8…` the reverse. `spec_sha256` hashes the spec envelope — country,
  manifest schema version, normative projections, resolved bindings
  (`spec_engine/loader.py:404-418`) — so the union pin was recomputed fresh
  via `load_bundle("uk")`: `bb711069…`, a third value, as required.
- BE (`bf022118…`) and US (`d3de6760…`) recomputed identically on the union:
  the `b4dfa0e7..2aa96795` range touches no BE- or US-side envelope input.
  Its only shared-schema edit, `sources.schema.json`, sits in the grammar
  receipt outside the hashed envelope (`spec_engine/loader.py:227-229,401`),
  the mechanism already proven when main's property additions moved no pin.
- Main's re-cut UK gate-battery digests (`uk/gates.json`, the
  `microcosm-data` contract pins in `contract.py`/`test_contract.py`) merged
  clean — this branch touches neither side of them — and the ten
  gate-battery pin tests plus `test_contract.py` pass unchanged on the
  union, inside the 339-test affected run.
- Union-tree receipts before the full suite: `tools/spec_engine_coverage.py
  --check` reports 41,471/41,471 configuration fields (main's 41,379 plus
  this branch's 92 policy fields) and 40/40 inventory checks with no drift;
  the loader golden vector holds (main's range touches neither
  `loader.py` nor `canonical.py`); repository-wide Ruff and
  `git diff --check` pass.
- Wheel gate (the merge moves `microcosm-build/pyproject.toml` and packaged
  spec data): all five shard wheels build; a clean venv installed from those
  wheels under the exported lock constraints imports every shard from the
  venv prefix with `policyengine_us` absent;
  `tools/spec_envelope_digests.py be uk` runs from the installed wheels; and
  wheel-venv `load_bundle` identities reproduce BE `bf022118…` / UK
  `bb711069…` byte-for-byte, proving the authored spec and schema files ship
  in the wheel (`packages/microcosm-build/pyproject.toml` data inclusion).

## Build-shard merge-union guards — 2026-08-23 headless continuation

- The first complete build-shard run reached 100% and exposed three
  deterministic union misses. The branch-side serializer omitted an empty
  opt-in `target_regimes` field by routing dataclasses through mappings; main
  replaced `dataclasses.asdict()` with direct field walking so immutable
  `MappingProxyType` authority records serialize. The merged mapping path had
  the omission filter, but the direct dataclass path did not. The union now
  applies the same filter during direct walking
  (`tools/build_us_multispine_pool.py:3826-3847`), preserving both intents.
- Main's authored-imputation SHA audit treated every `*sha256` field as an
  external-asset pin. The F0 port legitimately authors one derived policy
  identity digest. The audit now remains closed: its only non-asset allowance
  is the exact
  `models/regime_gated_qrf/post_draw_calibration/sha256` path, while the two
  existing asset pins remain exact
  (`packages/microcosm-build/tests/test_us_spec_bundle.py:707-770`).
- Direct import-graph enumeration reaches 66 classified US runtime modules;
  the delta from main's 65 is exactly the branch-added
  `post_transfer_calibration.py`. The cardinality pin now says 66, while the
  existing required-module, retired-module, unclassified-module, and
  source-spine-blind checks remain unchanged and pass
  (`packages/microcosm-build/tests/test_us_spine_blindness.py:3270-3315`).
- All three exact deterministic tests pass, as do touched-file Ruff,
  formatting, and whitespace checks. GitNexus query/context tools were not
  exposed in this session, so parent-source comparison, direct import-graph
  enumeration, and exact source reads supplied the debugging evidence.
- Two unrelated `test_us_trade_imdb_bulk.py` crash-publication cases timed out
  while their child process imported `build_us_import_entry_margins.py`, before
  reaching publication logic. They failed both in the complete shard and in
  unchanged isolation under shared-host load. A one-thread, import-only probe
  completed in 73.82 seconds while load was about 79, already beyond the
  tests' unchanged 60-second bound. This lane will retry after contention
  falls; it does not tune the timeout or any product threshold.

## Third merge: `origin/main` at `055dcfaf` — 2026-08-23

- The shared remote-tracking ref advanced six commits while the complete build
  shard ran: UK E8 CGT structure, salary sacrifice, and student-loan stages;
  their review fixes; the US fiscal-refresh memory-canary isolation; and merge
  `055dcfaf` (#740). This range does not touch the F0 imputation policy, its
  anti-rot ledger, or the US spec envelope.
- Sole textual conflict: the UK entry of `EXPECTED_RESOURCES` in
  `packages/microcosm-build/tests/test_spec_engine_country_bundles.py`, for the
  same two-parent reason as the prior merge. This branch's `bb711069…` omits
  E8; main's `1f163cbf…` omits the version-11 stacked-authority binding. Fresh
  union-tree `load_bundle` results are BE `bf022118…`, UK `8bf62b6e…`, and US
  `d3de6760…`; the conflict is resolved with the UK union value.
- Main's updated UK gate-battery and `microcosm-data` contract pins merged
  cleanly. The new shared `sources.schema.json` variants remain in the grammar
  receipt outside the hashed envelope (`spec_engine/loader.py:227-229,401`),
  while E8's authored UK sources legitimately move only the UK envelope.
- Newest-union verification before committing the merge:
  `tools/spec_engine_coverage.py --check` reports 41,471/41,471 fields and
  40/40 inventory checks; the focused policy identity, country bundle, field
  usage, coverage, inventory, gate/data contract, serializer, authored-SHA,
  and 66-module spine-blind graph tests all pass in one process.

## Final newest-union verification — 2026-08-23

- Re-ran the two unchanged `test_us_trade_imdb_bulk.py` crash-publication
  cases after shared-host load fell. Both pass with the existing 60-second
  child-process bound and one-thread numerical-library environment. The
  earlier failures were confined to child CLI import under contention; no
  timeout, comparator, gate, or product threshold changed.
- Repository-wide `ruff check .` and `git diff --check` pass on the final
  `055dcfaf` union.
- Four shard receipts, each in one pytest process with peak child RSS from
  `resource.getrusage`: fit 93 passed, exit 0, 925,466,624 bytes (0.862 GiB);
  calibrate 203 passed, exit 0, 481,968,128 bytes (0.449 GiB); frame 294
  passed / 36 skipped, exit 0, 6,971,260,928 bytes (6.492 GiB); data 275
  passed / one skipped, exit 0, 11,867,324,416 bytes (11.052 GiB).
- The complete build root first passed behaviorally in one process (6,248
  passed / 39 skipped) but reached 17,167,810,560 bytes (15.989 GiB). That
  receipt violates this lane's `<15 GiB` constraint and is explicitly
  rejected; it is not the resource evidence for completion.
- The authoritative build rerun enumerated all 262 `test_*.py` files and ran
  them in 17 fresh pytest processes under a 12 GiB/20 ms hard guard. Every
  batch passed without a split or guard intervention. The exact aggregate is
  6,248 passed / 39 skipped, exit 0, elapsed 6,966.45 seconds; maximum observed
  RSS is 11,042,193,408 bytes (10.284 GiB). This reproduces the complete
  single-process inventory while keeping the accepted receipt below 15 GiB.
- Refreshed the packaging boundary on the newest union. All five wheels built
  offline using the writable cache. A brand-new offline venv was created but
  could not resolve uncached third-party packages (`pytest`, then
  `huggingface-hub`), so it was not used as evidence. The five new local
  wheels were instead reinstalled without dependency resolution into the
  existing clean, lock-constrained wheel venv from the earlier gate. With
  `PYTHONPATH` removed and isolated mode enabled, all five `microcosm.*`
  shards import from that venv, `policyengine_us` is absent,
  `tools/spec_envelope_digests.py be uk` passes, and installed-wheel
  identities are BE `bf0221184046428782e7628dfad9b1a420bcc90c76ee88f4df373abecabff9d9`,
  UK `8bf62b6e47583da1bdad1b71be1e705f424e6e245880e90f4411aba57fa5eb93`,
  and US `d3de6760727cfcb6800209670d37e02b373d8dcda19f8ad054aa9d410e0efbb0`.
- A fresh `git fetch origin main battery-pkg3-two-part` remains blocked before
  transport by sandbox DNS (`Could not resolve host: github.com`). The shared
  local `origin/main` ref is `055dcfaf` and is an ancestor of HEAD. No push has
  been attempted; the sole permitted push remains the final transport step.

## Scope and frozen boundary

This lane owns 16 adjudicated FIX-CANDIDATE checks: 13 in
`source_operator_two_part_calibration`, two in
`adult_care_post_reconciliation`, and one in
`model_required_targeted_calibration`. The assigned baselines and named
remedies are the rows in the read-only reference
`../microcosm-arm-split/experiments/battery_burndown/adjudication.json`.

No comparator, band, threshold, seed, fold, or sample contract may change. All
builds in this lane are off-chain at `--sample-fraction 0.01` and
`--sample-seed 578`; they omit `--logbook-prev-row-digest` and do not touch
`logbook-pending-chain.txt`.

## Post-transfer receipt failure #2: ordered capacity evidence

### No-build reproduction

- The SHA-pinned replay harness reads the assembled Frame checkpoint and the
  unemployment-compensation and weeks-unemployed target-bank H5 files,
  validates their file, identity, and raw-draw digests, reconstructs only the
  native clone-0 vectors, and calls the calibration kernel and its strict
  receipt validator. It performs no target fit, DAG execution, Frame write, or
  build (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:1-6,74-112,115-250`).
- The stacked owner selects ASEC clone-0 reference rows, ACS clone-0 recipient
  rows, transferred-null mutable rows, and—for weeks—only mutable rows with
  positive unemployment compensation as allowed/addition candidates
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8960-8971,8995-9032`).
- The preserved replay contains 4,311 reference rows, 34,293 recipient/mutable
  rows, 134 positive reference rows, 24 initial recipient positives (all
  disallowed), and 32 positive-UC addition candidates. Its reference total is
  `80,851,529.27715749`, recipient total is `79,926,522.10879111`, reference
  positive mass is `2,762,659.3294707513`, and target positive mass is
  `2,731,052.2627107087` (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:144-206,230-250`).
- At reproduction commit `4cc41652`, capacity generation reduces the 32
  candidate weights with masked `ndarray.sum`, producing
  `85,676.23791782455`, while selection independently reduces the same
  ID-ordered weights with `np.cumsum`, producing `85,676.23791782456`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:463-493,817-880` at that commit).
  The exact validator relationship `0 <= lower <= upper <=
  expected_candidate_mass` therefore rejects the receipt by one float64 ULP,
  `1.4551915228366852e-11`; every other relationship passes
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:1389-1490` at `4cc41652`).

### Semantic adjudication

- The invariant is correct and target-type agnostic: a selected prefix cannot
  exceed its declared candidate capacity. No tolerance or target exception is
  authorized. The generating defect is that capacity and prefix evidence are
  derived by two reduction schedules for the same declared ordered carrier
  set (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:321-327,463-493,817-891` at `4cc41652`).
- Weeks is a valid positive-carrier target. The ASEC source accepts only
  integer `-1` or `0..52`, maps `-1` to zero, and defines the positive event as
  an in-range integer above zero
  (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:791-800,1218-1222`).
  Its dedicated QRF path rounds, clips, and revalidates predictions in that
  domain; post-transfer amount mapping then selects only positive
  reference-donor values and rejects support escape
  (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:911-983`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:588-626,702-705`).
- The semantics-preserving repair is one immutable ordered-prefix schedule per
  candidate set, consumed by both capacity and selection. Rewriting only the
  terminal cumulative value is invalid because `_nearest_prefix` requires a
  nondecreasing vector for its lower-mass tie break and `searchsorted`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:487-515`).

### Two generating defects and the complete repair

- `_PrefixSchedule` binds one immutable ordered-position vector to one
  float64 cumulative-mass vector. `_nearest_prefix` consumes that schedule and
  reports its terminal element as candidate mass
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:282-287,471-515`).
  Commit `d7b12bab` constructs the amount-descending/ID removal schedule and
  ID-ordered addition schedule once, so capacity and selection use the same
  candidate endpoint (`post_transfer_calibration.py:844-872,891-928`).
- The SHA-pinned weeks replay therefore changes its candidate mass from
  `85,676.23791782455` to the selection schedule's exact terminal
  `85,676.23791782456`; upper minus candidate becomes zero and strict
  validation succeeds. The harness now pins both the failing and repaired
  relationship values rather than accepting an arbitrary validator failure
  (`tools/reproduce_us_post_transfer_weeks_checkpoint.py:210-286`).
- The required cross-target replay found that `d7b12bab` still composed its
  whole maximum from independently rounded partition endpoints. Both actual
  child-support targets produced maximum `79,926,522.10879174` against
  recipient total `79,926,522.10879111`, so only the exact
  `maximum_attainable_mass <= recipient_total` relationship failed, by
  `6.258487701416016e-07`. Expense partitions were
  `71,696.09739141785 + 79,854,826.01140033`; received partitions were
  `180,209.75664861224 + 79,746,312.35214312`.
- The complete generating repair declares the attainable carrier set once as
  `fixed_positive | allowed_positive | zero_candidates`. It zero-masks that
  set onto the already ordered recipient-weight vector, retaining the same
  vector length and reduction topology used by `recipient_total`. For
  nonnegative weights, the exact subset bound is therefore structural; the
  maximum is neither a sum of rounded partition scalars nor a clamp
  (`post_transfer_calibration.py:823-885`).
- Both SHA-pinned child receipts now have maximum exactly equal to recipient
  total `79,926,522.10879111` while the historical partition sum remains
  `79,926,522.10879174`; every strict relationship passes. The child harness
  pins both targets' file/identity/raw hashes and requires the exact per-target
  state, error, relationship, and floats on the red and green sides
  (`tools/audit_us_post_transfer_child_support_checkpoints.py:1-80,91-207,210-302`).
- A proper-subset regression supplies weights for which compressed regrouping
  yields `0x1.433526fbe1946p+48`, `0.0625` above recipient total
  `0x1.433526fbe1945p+48`. The same-topology union yields the recipient value
  exactly and validates for every late `match_reference` declaration. Separate
  regressions cover the production weeks candidate bytes, independently
  rounded whole partitions, and the symmetric removal path
  (`packages/microcosm-build/tests/test_us_post_transfer_calibration.py:544-753`).
- The validator is unchanged: it still requires exact
  `maximum_attainable_mass <= recipient_total` and exact
  `0 <= lower <= upper <= candidate_mass`; its pre-existing approximate
  partition-additivity and boundary checks were not adjusted
  (`post_transfer_calibration.py:1457-1529`). No tolerance, threshold, band,
  gate, or target exception changed.

### Complete late-transfer target audit

The registry contains exactly seven late targets; six share the repaired
`match_reference` branch and one bypasses carrier selection by preserving
recipient carriers
(`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:208-258,786-812,840-932`).

| Late target | Source semantics / observed 1% shape | Carrier verdict |
| --- | --- | --- |
| `pre_subsidy_care_expenses` | Nonnegative monetary care expense. ACS reconciliation restricts it to qualifying people and at most one carrier per tax unit; the late owner admits existing qualifying positives and one stable zero candidate per empty unit (`acs_transfer.py:660-739,1277-1299`; `stacked_spine.py:8728-8746,8977-8986`). | Covered: constrained `match_reference`; not a count. Its proper attainable subset uses the same whole-union mechanism and proper-subset regression. Host had not reached it. |
| `child_support_expense` | Exact nonnegative annual `CHSP_VAL` carry (`child_support.py:166-201`); QRF reported six distinct donor values. | Covered: monetary `match_reference`. Its pinned checkpoint fails at `d7b12bab` and passes the complete union repair. |
| `child_support_received` | Exact nonnegative annual `CSP_VAL` carry (`child_support.py:166-201`); QRF reported 15 distinct donor values. | Covered: monetary `match_reference`. Its pinned checkpoint fails at `d7b12bab` and passes the complete union repair. |
| `disability_benefits` | Nonnegative annual two-slot sum excluding workers' compensation (`disability_benefits.py:184-220,558-560`); QRF reported ten distinct donor values. | Inapplicable: `preserve_recipient` emits neither capacity nor selection evidence (`post_transfer_calibration.py:230-236,1319-1335`). Its preserved checkpoint validates with before/after carrier mass `42,658.57948297383`. |
| `weeks_unemployed` | Integer `-1` or `0..52`, with `-1` mapped to zero (`weeks_unemployed.py:791-800,911-983,1218-1222`); QRF reported 12 distinct donor values. | Covered: sole semantic count; positive-UC-constrained `match_reference` (`stacked_spine.py:8995-9008`). The exact replay proves reducer order, not count support, caused the failure. |
| `workers_compensation` | Exact nonnegative annual `WC_VAL` carry (`workers_compensation.py:143-184,520-522`); host had not reached it. | Covered: monetary default-mask `match_reference`; shared-kernel regressions validate its declaration. |
| `spm_unit_energy_subsidy` | Measured nonnegative annual `SPM_ENGVAL`, checked within unit and reduced to SPM-unit float64 (`energy_subsidy.py:157-233,543-557`); host had not reached it. | Covered: monetary default-mask `match_reference`; shared-kernel regressions validate its declaration at its entity grain. |

The host log's near-discrete evidence appears at `build.log:1252-1266,1404-1408`.
Weeks does not rely on ACS's explicit discrete-numeric set, which contains only
two mortgage-year targets; ordinary numeric targets otherwise use the
continuous encoding. Its integer semantics come from the dedicated weeks
source/QRF checks and post-transfer donor-support mapping cited above
(`acs_transfer.py:129-138,3035-3117`). QRF's `<=32`-unique “near-discrete”
branch is a leaf-storage optimization, not a carrier-capacity semantic
distinction, which is why annual dollar targets also triggered it here
(`microcosm-fit/qrf.py:388-401,482-503`).

Current zero-based late-DAG positions are child support 24, disability 25,
weeks 30, workers' compensation 31, energy subsidy 32, and adult care 34.
The registry constructs and schedules those groups deterministically, stacked
execution enumerates them serially, and each production group applies its
post-transfer calibration before returning
(`us_late_producer_registry.py:1338-1396,2013-2019`;
`stacked_spine.py:10054-10095,10927-10931`). The failed host run had crossed
child support and disability but had not produced checkpoints for workers'
compensation, energy subsidy, or adult care. Their verdict is therefore a
source/mask proof plus the six-spec shared-kernel regressions, not a claim of
checkpoint replay.

## Source-cited mechanism record

- The battery computes positive and negative carrier incidence separately,
  compares ACS/ASEC weighted incidence to the frozen band, then computes five
  weighted conditional carrier quantiles when both sides have enough rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13351-13421`).
- Its quantile-envelope diagnostic is the maximum symmetric normalized
  separation across those five conditional quantiles
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13481-13512`).
- The fitted QRF draws a weighted sign gate and then a forest amount conditional
  on the selected sign; that is the two-part mechanism calibrated here
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:950-1003,1333-1429`).
- Ordinary ACS transfers partition recipients by optional-predictor
  availability and construct exact complete-donor model frames. Regime
  detection and fitted-model verification run only for an explicit
  owner-selected target subset; the existing pattern seed and draw surface do
  not change
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:919-1115,1297-1579`).
- The banked path applies the same explicit selection before the targetwise
  chain and verifies only selected returned target regimes before accepting
  their raw draws
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1621-1848`).
- Adult-care qualification is a fail-closed section-21 predicate, and the
  reconciliation clears nonqualifying mutable carriers, permits at most one
  qualifying mutable carrier per tax unit, and preserves pre-existing positive
  carriers
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:660-732`).
- The weeks-unemployed signal gate rejects positive PUF support without
  unemployment compensation, so carrier additions must respect that compatible
  capacity (`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:1328-1358`).
- The prior-year income gate explicitly requires the ASEC negative
  self-employment leg to survive, so this lane may calibrate only the positive
  mutable leg (`packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py:829-887`).

These mechanisms support an artifact-side correction; they do not justify a
comparator change.

## Implemented artifact mechanism

- The immutable nine-target policy declares the exact early/late owner, carrier
  mode, byte-exact negative leg, adult-care constraint, and weeks/UC constraint
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:123-263`).
- The kernel requires disjoint reference/recipient masks and mutable-subset
  scope, snapshots and byte-compares protected surfaces, computes the reference
  positive mass, and uses deterministic nearest-prefix removal/addition within
  a proven attainable-mass interval
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:445-547,727-922`).
- Mutable positive amounts are mapped only to reference positive support and
  are anchored at the frozen 10/25/50/75/90 percentiles; infeasible or
  conflicting anchors are recorded rather than hidden
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:577-724`).
- The kernel proves boundary saturation when capacity-limited and rejects any
  change to nonmutable, negative, negative-zero, or zero-weight bytes, any
  donor-support escape, or any preserve-carrier change
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:939-978`).
- The stacked owner derives ASEC clone-0 reference rows, ACS clone-0 recipient
  rows, and transferred-null mutable rows. Adult care uses qualifying rows plus
  one candidate per empty unit; weeks uses positive-UC mutable rows. The final
  adult-care reconciliation must be byte-identical/no-op
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8799-8955`).
- Schema-v2 receipts explicitly state that terminal validation cannot replay
  pre-calibration state; they separate live-replayable output claims from
  generation-transition evidence
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:43-119,980-1055`).
- Terminal validation independently recomputes live row/mask hashes, entity and
  output hashes, full weights, carrier masses, reference/recipient quantiles,
  QED, and coupled adult/weeks constraints
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3756-3818,3821-4025`).
- Only the nine declared calibration targets receive QRF pattern receipts.
  Those receipts persist ordered predictors, seeds, weights, row counts, and
  selected regimes, then validate canonical predictor/pattern/target order and
  exact record-family binding. All assigned and unassigned receipts validate
  the same legacy row-count schema and accounting before their scope branch.
  Unassigned transfers retain their evidence-free receipts and generic
  serializers omit the empty opt-in field. Receipt-only validation deliberately
  does not claim donor replay or out-of-sample verification
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4128-4445,4529-4795`).
- Canonical early and late production both use the certified maximum of eight
  targets per fit. Exact selected-family receipt binding therefore cannot be
  weakened by caller-selected batching; the non-production test seam retains
  smaller-width coverage.
- The two pinned 3.73 GB SIPP readers retain chunked selection and downstream
  explicit coercion while using streaming type inference. Guarded full-donor
  reruns observed much lower RSS and unchanged locked donor facts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-413`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/voluntary_filing.py:375-397`).

## Adjudication limits retained

- The named adjudication remedy calls for target/sign-scoped origin-aware
  cross-fitted carrier calibration and held-out nonregression. This branch does
  not have the adjudicated fold/comparator authority in main and does not invent
  one. Its carrier correction is deterministic terminal reference-margin
  matching, not cross-fitting or an out-of-sample estimate
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:839-922`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4211-4217`).
- The unemployment-compensation row likewise lacks the adjudicated money-OOS
  authority in this branch. The implementation freezes its carrier membership
  and calibrates only conditional positive amounts; no OOS nonregression claim
  is made
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:193-207,952-978`).
- Weeks-unemployed carrier matching is allowed to stop at the exact
  positive-UC-compatible capacity, but the receipt must prove the attainable
  interval and boundary saturation
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:839-955`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8863-8876`).
- Mutable masks, input hashes, before-state diagnostics, change counts, and
  byte-preservation proofs need the generation-time pre-frame. Terminal
  validation authenticates them through the enclosing execution authority and
  does not claim to reconstruct them from the final frame
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:43-119`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3925-3938`).

## Environment receipt

- `uv sync --all-packages --extra us`: failed before resolution because the
  sandbox denied writes to `/Users/maxghenis/.cache/uv`.
- Writable-cache retry: failed downloading `pandas==3.0.3` because sandbox DNS
  is unavailable.
- Exact-lock fallback: `uv.lock` SHA-1
  `6b213e740b114d008c0191fa492832a957a0a948` matches
  `../microcosm-707/uv.lock`; that environment imports NumPy 2.4.6, pandas
  3.0.3, and pytest 8.4.2 while `PYTHONPATH` points at this worktree.
- The pre-continuation PR test surface was green: all 225 `microcosm-build`
  test files; `microcosm-fit` 93 passed; `microcosm-calibrate` 201 passed;
  `microcosm-frame` 294 passed/36 skipped; and `microcosm-data` 275 passed/one
  skipped. Heavy files ran in fresh pytest processes. The final continuation
  tree also passes all 225 `microcosm-build` test files across fresh processes,
  including full ordinary/banked transfer, stacked binding, serializer, pool,
  H5 loader, and real late-executor files. The full rerun exposed one synthetic
  H5 target receipt with only `residual_null_rows`; its fixture was updated to
  the valid four-zero count block, its full file reran green, and a fixture
  scan found no other canonical partial blocks.
- `ruff check .`, touched-file `ruff format --check`, and `git diff --check`
  pass. Full-tree format checking reports 49 pre-existing files outside this
  lane's formatting scope.
- On the completed post-transfer receipt repair, all 47 focused calibration
  tests and all five package roots (`microcosm-fit`, `microcosm-calibrate`,
  `microcosm-data`, `microcosm-frame`, and the complete `microcosm-build`
  root) exited zero under the guard. Repository-wide Ruff, touched-file
  formatting, and whitespace checks also pass. Only established skips and
  warnings appeared; no host build ran.
- The final successful full-donor tests peaked at 0.485 GiB for vehicles and
  0.532 GiB for voluntary filing; the largest successful isolated build-test
  shard peaked at 6.531 GiB. An earlier 13.5 GiB/250 ms diagnostic guard
  observed one rapid parser spike at 15.424 GiB before termination. The guard
  was immediately tightened to 10 GiB/20 ms, the streaming reader was fixed,
  and all successful reruns stayed below the cap.

## Measurement ledger

The before build used `--sample-fraction 0.01 --sample-seed 578`, clone fraction
1 and clone seed 578, with no chain predecessor. Peak observed per-process RSS
was 10.733 GiB. Artifact receipts:

- `pool.h5`: SHA-256 `258891504201275f8006a1584b7d3e891890d15381724bc9f2b30b1f443d967f`
- `pool.gates.json`: SHA-256 `94ee914bb7490e7f513184e691cf15847d1e585693ef184977196485f86f1fee`
- `pool.manifest.json`: SHA-256 `953aaff72fac8cc17211959d0133f9bce8c22dd87b2957a0f6a89335fcc9c122`

The exact frozen-battery values are below. `unsupported` means the battery did
not emit QED because one side had fewer than five carriers; the parenthesized
number is the same five-grid weighted diagnostic computed manually without
changing that support rule.

| Assigned check | Before at 1% |
| --- | ---: |
| adult care positive incidence ratio | 0.561425035 |
| adult care positive QED | unsupported (manual 1.738865343) |
| unemployment compensation positive QED | 0.352941176 |
| child-support expense positive incidence ratio | 0.171280844 |
| child-support expense positive QED | 0.953846154 |
| child support received positive incidence ratio | 0.242882414 |
| child support received positive QED | 1.000000000 |
| disability benefits positive QED | 1.373534621 |
| prior-year self-employment positive incidence ratio | 1.376752468 |
| prior-year self-employment positive QED | 0.834720589 |
| weeks unemployed positive incidence ratio | 0.025384419 |
| weeks unemployed positive QED | 0.736842105 |
| workers' compensation positive incidence ratio | 0.047614655 |
| workers' compensation positive QED | unsupported (manual 1.918367347) |
| SPM-unit energy subsidy positive incidence ratio | 0.240477284 |
| SPM-unit energy subsidy positive QED | 0.666666667 |

Sparse 1% sampling also puts the otherwise frozen unemployment-compensation
and disability positive carrier ratios at 0.131859569 and 0.098471480. Their
adjudicated remedies freeze carrier membership, so this lane will improve only
their assigned amount-QED checks. The unassigned negative prior-year
self-employment ratio is 1.435953092 and is likewise deliberately untouched.

After values will be added after the calibrated 1% rebuild.
## Historical lane notes imported from `origin/main` — 2026-08-22

The following `one-target-surface` notes are preserved verbatim as merge
history. Any present-tense branch state or pending work below belongs to that
source lane's 2026-08-21 snapshot, not to this continuation.

# One-target-surface lane notes

## 2026-08-21 — baseline and doctrine

- Branch: `one-target-surface`, starting at `2c7a7218` (`origin/main`).
- User doctrine: all US calibrated artifacts compile one target surface;
  geography is a constraint dimension, while artifact scale changes only L0 /
  record count.
- The current split is explicit in
  `packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py`:
  `compile_us_fiscal_target_registry` accepts
  `include_congressional_district_targets`, forwards it through dynamic target
  dispatch, and uses it to omit SOI and ACS congressional-district facts.
- The target profile independently gates CD-to-state hierarchy reconciliation
  on the same flag in
  `packages/microcosm-build/src/microcosm/build/us/fiscal_target_references.json`.
- The SOI taxable-interest rebase doctrine remains distinct: CD aggregate rows
  are processing-window subsets and never act as national controls, as recorded
  in `_rebase_stale_soi_taxable_interest_distributions` and pinned by
  `test_stale_soi_taxable_interest_never_uses_congressional_district_controls`.
- [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353)
  explicitly names deletion of the flag as the one-surface outcome;
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569)
  records that the scorer's opt-in path is dead.
- Environment: default-cache sync failed because the sandbox cannot write
  `~/.cache/uv`; clean-cache sync then failed because network/DNS is disabled.
  The sibling `microcosm-spec-engine` checkout has the identical `uv.lock`
  (`895535...`) and a complete Python 3.14 environment, so its `.venv` was
  cloned copy-on-write. An offline editable reinstall still requires missing
  build-isolation metadata; test commands therefore use `UV_NO_SYNC=1` and put
  every current-worktree package `src` directory first on `PYTHONPATH`.
- GitNexus: repository analysis produced `.gitnexus/lbug`, then registration
  failed on the sandboxed `~/.gitnexus/registry.json`; dependency coverage is
  being checked with repository-wide `rg` plus focused tests.
- Build discipline: no calibration build has run; the off-chain / <=1% rule is
  intact and `logbook-pending-chain.txt` has not been touched.
- Validation baseline: `uv run pytest -q` advanced without a failure for
  1,136.97 seconds, then was interrupted while an unrelated PUF-QRF test was
  waiting on a subprocess
  (`test_puf_qrf_chain.py::test_primary_qrf_rejects_every_stale_schema_version`).
  Per-commit checks use the affected US target/compiler tests; the complete
  workspace suite will run against the final tree. `uv run ruff check .`
  passed.
- Affected-suite baseline: the target/compiler/parity/builder/scorer/spec set
  reached 100% with no failure in 349.59 seconds. `/usr/bin/time -l` itself
  exits 1 in this sandbox because `sysctl kern.clockrate` is forbidden (the
  same wrapper also exits 1 around `true`); later memory receipts use Python's
  child-resource accounting instead.

## 2026-08-21 — runtime surface unified

- The public compiler has no CD membership option and routes IRS SOI, ACS-CD,
  and PEP-CD facts unconditionally
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1010,2299-2390,2588-2625`).
- The row-level doctrine is unchanged: CD rows compile into the registry, but
  `_soi_taxable_interest_control_key_from_fact` rejects CD record sets as
  national controls
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:1478-1510,1726-1744`).
- Builder, fiscal scorer, state scorer, and ACS-local compilation all load the
  packaged source-to-current CD crosswalk by default; the production source
  aliases are active unconditionally
  (`tools/build_us_fiscal_refresh_release.py:1486-1497,8319-8350,11226-11230`;
  `tools/score_us_fiscal_targets.py:383-426`;
  `tools/score_us_state_files.py:313-345`;
  `tools/build_us_acs_local_release.py:141-170`).
- The diagnostic JCT deletion switch and its target-profile-gate bypass were
  deleted from all release/scorer entrypoints. The active registry is now the
  compiled registry in the builder and both scorers
  (`tools/build_us_fiscal_refresh_release.py:8435-8464`;
  `tools/score_us_fiscal_targets.py:415-432`;
  `tools/score_us_state_files.py:335-352`).
- The generated calibration contract declares `national`, `state`, and
  `congressional_district` in one `geography_layers` list and requires CD
  inclusion; there is no default-layer fork
  (`tools/us_bundle_generation/contracts.py:1322-1340`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/schema/calibration.schema.json:35-68`).
- The parity generator compiles with the canonical crosswalk unconditionally,
  and the regenerated manifest records 32 compiled / 52 reviewed families
  (`tools/build_us_target_parity_manifest.py:617-655,689-729`;
  `packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`).
- This deletes the dead scorer opt-in described by
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569) and the
  regime knob superseded by the one-surface decision in
  [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449#issuecomment-5002607353).
- Validation: the affected 10-file suite completed to 100% with exit 0 after
  the change; Ruff, Python byte compilation, and `git diff --check` pass. No
  build, push, chain operation, or pending-chain edit occurred.

## 2026-08-21 — parity doctrine and row-level invariant

- `irs_soi.congressional_district_2022` is a red-line compiled family, so the
  anti-rot validator refuses any future downgrade to a reviewed exclusion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/release_target_parity.py:88-123,579-586`).
- The shipped manifest entry is `compiled`, has no exclusion classification,
  reason, evidence, or fence, and states that there is no local-versus-national
  surface. Its header counts are pinned to the parsed 32 compiled and 52
  reviewed families
  (`packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:3-12,524-527`;
  `packages/microcosm-build/tests/test_release_target_parity.py:290-317`).
- The always-compiled SOI test exercises CD, state, and national rows
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:177-356`). The
  taxable-interest doctrine test additionally asserts that the CD aggregate is
  present in the registry, then proves it never supplies the rebase control
  metadata when a true Pub 1304 national control exists
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:2539-2625`).
- Validation: the standard 10-file affected suite completed to 100% with exit
  0; Ruff and `git diff --check` pass.

## 2026-08-21 — artifact-scale leak removed

- The hidden split was the compiler's per-run support-exclusion mapping: a
  caller could delete otherwise compiled source rows for one artifact. That
  parameter and its dynamic-dispatch branch are gone; the compiler now has
  exactly five inputs, none related to artifact size, sparsity, record count,
  support, inclusion, or diagnostic target deletion
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1005,2069-2090,2287-2300`).
- The release tool no longer parses or loads an artifact-specific exclusion
  file, passes no membership override into compilation, and reports only the
  standing surface-wide source exclusions
  (`tools/build_us_fiscal_refresh_release.py:897-923,7770-7815,8363-8376,11192-11198`).
  The obsolete
  `experiments/build_j_recert/sparse_zero_support_exclusions_buildj.json` was
  deleted and its shell/caller plumbing removed.
- The shared `US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` registry remains: it is a
  single source-row doctrine applied identically to all artifacts, not a scale
  input (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:554-660,2287-2295`).
- The identity regression compiles one CD-bearing fact set twice under nominal
  57,240-record sparse and 337,704-record dense labels; both the full `specs`
  tuple and content-addressed registry `version` must match, and the exact
  compiler signature is pinned
  (`packages/microcosm-build/tests/test_us_fiscal_targets.py:644-696`).
- Release/fiscal-scorer/state-scorer signature sets are pinned, and the release
  parser rejects all three deleted membership options
  (`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-97`;
  `packages/microcosm-build/tests/test_us_state_files_scorer.py:26-40`).
- Validation: six new identity/signature/parser tests pass; the standard
  10-file affected suite completed to 100% with exit 0; Ruff and
  `git diff --check` pass. No build ran.
