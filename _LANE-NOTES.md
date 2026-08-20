# Battery package 3 lane notes

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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:437-525,745-890`).
- Mutable positive amounts are mapped only to reference positive support and
  are anchored at the frozen 10/25/50/75/90 percentiles; infeasible or
  conflicting anchors are recorded rather than hidden
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:555-702`).
- The kernel proves boundary saturation when capacity-limited and rejects any
  change to nonmutable, negative, negative-zero, or zero-weight bytes, any
  donor-support escape, or any preserve-carrier change
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:908-956`).
- The stacked owner derives ASEC clone-0 reference rows, ACS clone-0 recipient
  rows, and transferred-null mutable rows. Adult care uses qualifying rows plus
  one candidate per empty unit; weeks uses positive-UC mutable rows. The final
  adult-care reconciliation must be byte-identical/no-op
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8799-8955`).
- Schema-v2 receipts explicitly state that terminal validation cannot replay
  pre-calibration state; they separate live-replayable output claims from
  generation-transition evidence
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:43-119,958-1024`).
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
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:817-890`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4211-4217`).
- The unemployment-compensation row likewise lacks the adjudicated money-OOS
  authority in this branch. The implementation freezes its carrier membership
  and calibrates only conditional positive amounts; no OOS nonregression claim
  is made
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:193-207,930-956`).
- Weeks-unemployed carrier matching is allowed to stop at the exact
  positive-UC-compatible capacity, but the receipt must prove the attainable
  interval and boundary saturation
  (`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:817-924`;
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
