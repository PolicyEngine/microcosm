# Stacked-pool to release CD-vintage provenance lane notes

## 2026-08-24 — lane start and environment

- Branch `pool-cd-vintage-provenance` starts at `origin/main` commit
  `7b90bb18`. This lane is limited to the authenticated producer/consumer
  contract described in the candidate-chain defect report: post-assembly
  household-CD assignment, checkpoint/manifest provenance, atomic nullable-H5
  attributes, format-aware release preflight, and a real tiny integration
  test (`tools/build_us_multispine_pool.py:441-577,3477-3786`;
  `tools/build_us_fiscal_refresh_release.py:2565-2661`).
- The ordered first `uv sync --all-packages --extra us` could not initialize
  the sandbox-inaccessible user cache. A writable-cache retry reached the
  locked Jinja2 URL but DNS is disabled. The exact recovery cloned the
  completed Python 3.14 environment from the merged PolicyEngine-US 1.819.0
  bump lane copy-on-write, then completed
  `UV_CACHE_DIR=/private/tmp/microcosm-scorecard-uv.0rntvY/cache uv sync
  --offline --all-packages --extra us`. uv rebuilt all five workspace wheels
  and replaced their sibling-worktree editable origins with this worktree.
- The GitNexus debugging skill applies to this cross-surface defect. No
  GitNexus query, context, graph, or resource tools are exposed in this
  session, so the fallback is direct source tracing, repository generators,
  and focused plus suite-wide tests.
- The producer must assign after the source boundary checks: preassembled
  `congressional_district_geoid` is an operator output and is rejected
  (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-353,372-406`;
  `tools/build_us_multispine_pool.py:4566-4584`). The release check remains
  unconditional and must continue to require matching crosswalk SHA, target
  vintage, and positive household support
  (`tools/build_us_fiscal_refresh_release.py:2565-2613,8571-8590`).
- No build, release, push, release-guard weakening, operator-boundary
  weakening, or `logbook-pending-chain.txt` access occurred at lane start.
- The pre-change suite is green. Fresh-process receipts are: calibrate 203
  passed; data 318 passed / 2 skipped; fit 93 passed; frame 295 passed / 36
  skipped; build partition excluding `test_us_[n-z]*.py` 4,127 passed / 36
  skipped; complementary build partition 2,177 passed / 3 skipped. Combined:
  7,213 passed / 77 skipped / 0 failed. The build split follows the CI
  fresh-process rationale (`.github/workflows/test.yml:24-34`) and the same
  previously accepted local partition; no assertion or behavior changed.

# Historical lane notes: PolicyEngine-US 1.819.0 lock bump

## 2026-08-23 — lane start and unchanged-lock environment

- Branch: `bump-policyengine-us`, starting at `31640b91` (`origin/main`).
- Owner ruling: the 25% replacement candidate must use
  `policyengine-us==1.819.0`, not the currently locked `1.764.6`. This lane is
  limited to the lock bump, compatibility repairs, identity re-pins, and PR-CI
  validation. It will not build a pool/release, push, or tune a gate,
  threshold, tolerance, or band.
- The ordered first command, `uv sync --all-packages --extra us`, reached no
  resolver or install action because the managed sandbox cannot initialize the
  user-wide uv cache. Retrying with a fresh writable cache reached the locked
  `jsonschema==4.26.0` wheel URL but DNS is disabled.
- Environment recovery reused only prior exact-lock local state: cloned the
  Python 3.14 `.venv` from `microcosm-scorecard` copy-on-write, then ran
  `UV_CACHE_DIR=/private/tmp/microcosm-scorecard-uv.0rntvY/cache uv sync
  --offline --all-packages --extra us`. uv rebuilt all five workspace wheels
  and replaced their sibling-worktree editable origins with this worktree.
- The GitNexus debugging skill applies to the expected failing-test diagnosis,
  but no GitNexus query, context, graph, or resource tools are exposed in this
  session. The fallback is `rg`, direct source reads, installed
  `policyengine_us` variable inspection, and focused tests.

## 2026-08-23 — lock upgrade and exact environment

- The binding resolver command completed offline against a task-local cache:
  `uv lock --upgrade-package policyengine-us --offline --cache-dir
  /private/tmp/microcosm-peus-lock-cache.kTEhCC`. The requested unfiltered
  version diff is exactly:

  ```text
  -version = "3.26.11"
  +version = "3.31.0"
  -version = "1.764.6"
  +version = "1.819.0"
  ```

  Thus the complete resolver movement is `policyengine-core 3.26.11 ->
  3.31.0` and `policyengine-us 1.764.6 -> 1.819.0`; NumPy remains 2.4.6 and
  Torch remains 2.12.0. The lock records the official PyPI 1.819.0 wheel at
  SHA-256 `525bdf8b238c3eb11cd60c5f4f7a7b0c57bc7eea5c1cf4346c261241b061be45`
  (`uv.lock:1364-1430`).
- `uv sync --all-packages --extra us` then completed with `--offline` and the
  same cache, resolving 123 packages and checking 100 installed distributions.
  The offline flag is environmental only: the resulting lock and environment
  use the official PyPI source and wheel identity above, not a local fork.

## 2026-08-23 — verified upstream compatibility repairs

- WIC was a verified rename, not a removal. PE-US 1.819.0 defines the monthly
  person input `takes_up_wic_if_eligible`, and `wic` gates on it before adding
  `wic_if_takes_up`
  (`.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/usda/wic/takes_up_wic_if_eligible.py:4-9`;
  `.../wic/wic.py:4-19`). Microcosm now exports that successor while retaining
  the historical `would_claim_wic` draw salt. Thus the Microcosm WIC
  source-stage Bernoulli draw remains unchanged for the same seed, stable key,
  and category rate; this does not claim that broader PE-US outcomes are
  unchanged across the release range
  (`packages/microcosm-build/src/microcosm/build/us_runtime/wic_claim.py:106-109,392-409`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:725-734`).
  The release-input builder carries an explicit historical alias instead of
  rewriting the frozen eCPS evidence
  (`tools/build_us_release_input_coverage_manifest.py:166-175`).
- TANF was also a verified successor. PE-US 1.819.0 makes
  `is_tanf_enrolled` formula-owned and returns the new monthly SPM-unit input
  `receives_tanf`
  (`.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/hhs/tanf/cash/eligibility/is_tanf_enrolled.py:4-16`;
  `.../cash/receives_tanf.py:4-8`). The measured ASEC `PAW_VAL`/`PAW_TYP`
  carry therefore writes `receives_tanf` without relaxing its TANF-specific
  source gate
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:127-160,266-289,485-500`).
- Three newly discovered input leaves — CA premium subsidy, CO premium
  assistance, and NM premium assistance take-up — each gate an upstream
  assigned-benefit formula. No reviewed Microcosm participation source exists,
  so all three are explicitly owned as `engine_default`/`rate_unsourced`; no
  rate was invented
  (`packages/microcosm-build/src/microcosm/build/us/spec/take_up.yaml:507-563`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/ca/hbex/premium_subsidy/assigned_ca_premium_subsidy.py:19-22`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/co/doi/premium_assistance/assigned_co_premium_assistance.py:19-22`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/nm/hca/premium_assistance/assigned_nm_premium_assistance.py:18-21`).
- The reported WIC source remains an adult-female reporter/carrier and does
  not identify a child or other beneficiary
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:352-365`).
  PE-US 1.819.0 now has six receipt consumers spanning household, SPM-unit,
  tax-unit, and direct-person use. The static source index now records both
  the direct reference receiver and any enclosing group aggregation, including
  implicit class-level `adds`; Microcosm preserves the carrier and freezes the
  exact six reviewed signatures across consumer, reference kind, output entity,
  direct receiver, and aggregation entity. Added/removed consumers or a changed
  receiver/aggregation fail closed
  (`packages/microcosm-frame/src/microcosm/frame/adapters/_policyengine_us_source_index.py:41-56,1694-1731,2004-2090`;
  `packages/microcosm-build/tests/test_us_pool_input_consumers.py:58-84,140-171,235-259,262-359`).
  Consequence: TX DART applies direct person use only to the carrier, and Pell
  aggregates only within the carrier's tax unit. Neither operation turns the
  carrier into beneficiary evidence.
- PE-US 1.804.1 made `home_mortgage_interest_tax_unit` the `add` consumer for
  first/second-home interest while balance and origination inputs remain
  direct inputs to `deductible_mortgage_interest_tax_unit`; 1.774.4 routes the
  Oklahoma subtraction through aggregate `taxable_pension_income`. The
  consumer-existence guards now assert those current paths
  (`packages/microcosm-build/tests/test_us_pool_input_consumers.py:491-575`).
- The installed input-surface comparison found nine old inputs absent from
  active discovery: `assessed_property_value`,
  `dc_ccsp_attending_days_per_month`, `dc_ccsp_child_category`,
  `ga_refundable_credits`, `has_marketplace_health_coverage`,
  `id_receives_aged_or_disabled_credit`, `is_tanf_enrolled`,
  `ne_child_care_subsidy_eligible_parent`, and `would_claim_wic`. The complete
  pool surface is constructed from transfer, deferred, primary-QRF, and
  take-up registries; only the TANF and WIC names in that surface needed the
  verified successors above
  (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:875-975`).
  `has_marketplace_health_coverage` was already rejected in favor of
  the non-engine survey fact `has_marketplace_health_coverage_at_interview`
  (`packages/microcosm-build/tests/test_us_puf_support.py:1083-1089`;
  `packages/microcosm-build/tests/test_us_multispine_pool.py:1294-1303`). No
  consumed variable was genuinely removed without a successor, so the binding
  owner-question stop condition did not trigger.
- The SNAP fixture failures combined two upstream changes: 1.769.0 multiplies
  deductible child-support expense by `snap_income_counted_share`, while
  1.794.3 changed missing `weekly_hours_worked_before_lsr` from 40 to zero. In
  the one-adult fixture, missing hours therefore fail the ABAWD work test,
  drive the counted share to zero, and erase the deduction. The fixture now
  supplies 40 hours explicitly and continues to assert the intended $200
  monthly deduction
  (`.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/usda/snap/income/deductions/snap_countable_child_support_expense.py:15-19`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/usda/snap/income/ineligible_members/is_snap_prorated_income_member.py:25-42`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/usda/snap/income/ineligible_members/snap_income_counted_share.py:15-19`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/usda/snap/eligibility/work_requirements/meets_snap_abawd_work_requirements.py:31-61`;
  `packages/microcosm-build/tests/test_us_child_support.py:448-489`;
  `CHANGELOG.md:263-268,968-972`).
- `weeks_unemployed` now directly feeds AL, NY, and OK unemployment insurance
  in addition to PA UC (1.809.0, 1.810.0, and 1.814.0). These paths remain
  structurally inert for the current pool: every independent claim-wage leaf
  required by all four formulas is formula-less and default-zero. The exact
  consumer and blocked-input sets are pinned so this rationale cannot drift
  silently
  (`packages/microcosm-build/tests/test_us_weeks_unemployed.py:695-751`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/al/dol/ui/al_ui.py:30-40`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/ok/oesc/ui/ok_ui.py:44-50`;
  `.venv/lib/python3.14/site-packages/policyengine_us/variables/gov/states/ny/dol/ui/ny_ui.py:24-38`;
  `CHANGELOG.md:60-64,114-133`).
- A stale source note incorrectly described every processed PUF person leaf as
  first-person placed and policy-neutral. The runtime actually aggregates PUF
  donor leaves to tax-unit QRF targets, redistributes predictions with copied
  ASEC shares or leaf-specific bases, and uses first-person placement only
  when no allocation mass exists. The corrected canonical note now reflects
  that person-sensitive caps and eligibility must remain distributable
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:424-442,1239-1252,1361-1369,1589-1598,3535-3583`).
- PE-US 1.794.3 changed the input default for
  `weekly_hours_worked_before_lsr` from 40 to zero. The frame adapter reads the
  engine's declared default rather than carrying a Microcosm override, so its
  scalar-default contract now expects zero
  (`.venv/lib/python3.14/site-packages/policyengine_us/variables/household/income/person/weekly_hours_worked.py:20-36`;
  `packages/microcosm-frame/tests/test_policyengine_us_adapter.py:105-119`).
- The published US population remains certified for PE-US 1.764.6 and must not
  be constructed under the new 1.819.0 lock. Package-version mismatch now has
  a private typed `ValueError`; the live network smoke skips exactly that
  expected certificate mismatch, while unit tests continue to prove the
  loader fails before dataset construction for either model or core mismatch
  (`packages/microcosm-data/src/microcosm/data/loader.py:63-64,425-445`;
  `packages/microcosm-data/tests/test_loader.py:326-356,403-419`). This does not
  weaken certified loading or treat malformed metadata as skippable.

## 2026-08-23 — compatibility note for 1.764.6 through 1.819.0

This is a deliberately short owner-facing scan, not an exhaustive policy
audit. Citations refer to the authoritative 1.819.0 release commit
`policyengine-us@350ecc766e4467dc05093d197a3404b53e0c9e4d/CHANGELOG.md`.
Changes in the range that can plausibly move values on this pool include:

- Receipt/take-up semantics: 1.777.0 adds `receives_X` and `X_if_takes_up`,
  moves SNAP take-up through the household API, and renames WIC; 1.778.0-1.778.1
  route reported SSI/TANF/SNAP receipt through many program rules and derive
  TANF enrollment from `receives_tanf`; 1.789.2 and 1.790.3 add WIC/SNAP and
  SSI/TANF receipt to categorical-eligibility lists
  (`CHANGELOG.md:694-719,319-351`).
- SNAP: 1.769.0 changes ineligible-member income/expense proration; 1.784.1
  fixes member filters, sanctions, and the OBBBA Medicaid-engagement SNAP
  status test; 1.784.2 fixes eligibility/allotment rounding; 1.785.0-1.789.1
  revise waiver/ABAWD routing; and 1.794.1 and 1.794.3 change BBCE schedules and
  missing-hours defaults (`CHANGELOG.md:968-976,512-525,432-436,354-358,263-281`).
- Major cash/health/housing formulas: 1.786.5 substantially revises Missouri
  TANF; 1.786.1 changes Medicaid categories and limits; 1.790.2 changes CHIP
  premiums; 1.767.2 applies HOTMA housing rules; and 1.768.3 adds Medicare Part
  A and Part D IRMAA to SPM medical expenses
  (`CHANGELOG.md:386-390,414-422,326-330,994-1044`).
- Tax and aggregate changes: 1.782.1/1.804.1 change mortgage-interest routing;
  1.787.0 repairs child-care/state-credit aggregates; 1.816.1 corrects New
  York IT-214; and 1.819.0 excludes Head Start/Early Head Start benefit values
  from household net income by default
  (`CHANGELOG.md:563-574,175-180,375-383,26-36,1-5`).
- 1.808.0 includes child support received in school-meal countable income, so
  the measured child-support leaf can now affect another benefit path
  (`CHANGELOG.md:143-150`).
- New programs can add simulated values where their state/program conditions
  bind: marketplace assistance including MD, CA, CO, NM, WA, MA, NJ, CT, and
  VT (1.791.0-1.803.0), plus a run of state child-care programs through 1.815.x
  (`CHANGELOG.md:53-87,143-172,189-253,284-317`).
- OBBBA-related follow-through in this range includes the 1.765.5 federal
  dependent-care/section-129 cap change, 1.766.5 static-conformity handling
  for state CDCCs, 1.784.1 Medicaid community-engagement/SNAP interaction,
  1.789.1 Alaska's ABAWD transition, and 1.817.0 Medicaid hardship exceptions
  (`CHANGELOG.md:1124-1129,1068-1075,519-525,354-358,15-23`). The large
  federal OBBBA current-policy implementation predates 1.764.6; this range
  carries these corrections and state-conformity consequences rather than
  introducing that whole policy baseline (`CHANGELOG.md:7773-7783`).

## 2026-08-23 — generated contracts and identity work

- The generated-source audit now binds PE-US 1.819.0. Of its seven attested
  source files, only `reforms/reforms.py` and `system.py` changed; the default
  engine still contributes exactly 110 dynamically generated variables
  (`packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py:114-152`).
- The engine input projection is now 924 inputs/defaults with SHA-256
  `b4b2041d221b6322a3143dd2d54dc95eb1b20d5f33f4a60908055d735ba07930`
  and defaults SHA-256
  `8718854d455ca536dba2e712aed3b2010becf909b8c61210f0456bcf732ec68c`.
  The SSI closure remains 55 input leaves but moves to 63 formula nodes / 187
  edges and SHA-256
  `e0a23d961c36526a10e56d80d51ca46760e92b4ea3653734014469da2394f702`.
  These are computed from the installed engine and validated by the loader,
  not hand-derived (`packages/microcosm-build/src/microcosm/build/spec_engine/engine_abi.py:292-371`;
  `packages/microcosm-build/src/microcosm/build/spec_engine/loader.py:376-417`).
- The terminal surface grows from 131 to 134 metrics because the CA/CO/NM
  take-up leaves are now explicit; the canonical partition still contains 48
  early and 70 late transfer targets
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2950-3000,8670-8730`).
- The source edits affect the globally attested direct-draw kernel because
  `wic_claim` and related producer modules are hashed into the seed protocol
  (`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:320-367`).
  NumPy and Torch versions did not move, but the attested module bytes did, so
  the seed protocol, BE/UK/US spec hashes, compiler coverage identities, and
  dependent authority/checkpoint hashes legitimately require tool-generated
  re-pins.

## 2026-08-24 — final tool-generated identities

The compatibility commit records the exhaustive 46-entry old-to-new mapping.
The final values below were computed by repository generators and tests, never
hand-derived:

- Lock artifacts: core sdist
  `4e7811d9f1668cb198ee68a460d7383add3443db1f0e32d591dc92ed4e5b4319`,
  core wheel
  `6537c2ecebd2a49a7a345dbe3d3d4f37fc3c3babc6bba1990a2eb99a9f55d67d`,
  US sdist
  `6e0e41887358dc5e80b8229b49a9027104ad2e49f32c2a8f08ce19a138556d9f`,
  and US wheel
  `525bdf8b238c3eb11cd60c5f4f7a7b0c57bc7eea5c1cf4346c261241b061be45`
  (`uv.lock:1366-1421`).
- Installed generated-source audit: `reforms/reforms.py`
  `9846915c2b03e776dc37cdd6d92566de117f6eebafbf5508179e196fce28d474`
  and `system.py`
  `820dadeb7d22ef14d9cb2c34607f2afe68ba5fae616e05e634c2319e61eb457d`
  (`packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py:114-152`).
- Frozen raw resources: source stages
  `dc58a0d700f0add7b658cec774df6e9587303beb58a1f432a35a18dcd1ac4097`
  and take-up contract
  `a9e70fb3e14b0af6cac5cc7935ef554f62dc3dca0377bc1fb57c0e6fa583e813`
  (`tools/generate_us_bundle_from_constants.py:120-134`).
- Engine ABI and remaining-input attestations: SSI closure
  `e0a23d961c36526a10e56d80d51ca46760e92b4ea3653734014469da2394f702`,
  projection
  `b4b2041d221b6322a3143dd2d54dc95eb1b20d5f33f4a60908055d735ba07930`,
  defaults
  `8718854d455ca536dba2e712aed3b2010becf909b8c61210f0456bcf732ec68c`,
  remaining manifest
  `98231086a18676778346fc3219bb9450f7eb85eb77791640598cba7a5ae66ef6`,
  and whole receipt
  `76de7320bd5f623c51f70326caee10c2f6a0b67d4a5641e10e4a196472a86a25`
  (`packages/microcosm-build/src/microcosm/build/us/engine_abi.lock.json:1`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:875-975`).
- Field-usage attestations: battery
  `55c2c0cd2d652216ed35f2d2667f52fe00229eb4336507bef9758397a1e24a17`,
  catalogs
  `a983c0d68e980a31bd1ad41e7ef3c0cb3db9a7ee4d73454ae46f81eb6d1bb427`,
  generated authorities
  `3f20975597d93f7313583a944eeb9d6437651c4ff20e67628bf6bf4c5aa9f004`,
  imputation family/concept
  `bdf40812604e7cd35d68093fe14b7cd1371cb8ab8658fd6002b254610b062781`,
  waivers
  `8075fc95d112a7442fa1d9e0f5a6a6d27ad081039147569d635993219eeac92a`,
  take-up surface
  `d53096196db4c34da260cce2f35af8e7ba67f978448656c602ee5e17529dc4e0`,
  and aggregate pointers
  `6d7353c6c42a6e1dbc6e3a227848e36864526fdc9533d5b284aa469c87dc064f`
  (`packages/microcosm-build/src/microcosm/build/spec_engine/field_usage.py:1422-1541`).
- Inventory/checkpoint attestations: authority
  `3a980927227704d0589f246eef9cd825c2ae84f3a4134ac835e0e5ed39a563ac`,
  early families
  `4aa9f736fd76e83955477ad1667e58f48f264783f05bdc7f0102cd32d61323bd`,
  full checkpoint
  `b6a47fac54d7de7aa42ce59dc1950c0765b7a67034f0174ad1531d5bbb06ceef`,
  gap-fill schedule
  `1c31f9868f7884347cc19cf1ff65da43f950b9114941a715bab168246db414a7`,
  producer graph
  `7125ad28ae2c69f22094a574bbf6ed2ddf1682a2c2c3b416f8f49304b7016ce7`,
  late families
  `d91f9ff0eb52f43e7b6eed3d5c58c37abe1620c3a11021da15dae9c10e16d382`,
  producer resources
  `3850554eb804fde5e4f86a34ac1bb8a7a07aafff7e8b48396a3d5fca844798e8`,
  late schedule
  `dcf3c6d2eade3449836c49a1dc4d3b8cd395aab9142db700c3c60598fa9c1c79`,
  seed-owner map
  `36a9d819ef196c312888591936d49c025b0407df9928440cceefabc5458f72af`,
  seed protocol
  `15840b380329410a7094f60b0f1dad453457fc785859f0c372f2c8e2d59b0246`,
  source semantics
  `cd5ba8924d64da5425ee14cca82a774e3f4b2bb5aabe06df291cc3cc457287a9`,
  and canonical take-up
  `fa186daea0f8dd641cc470e41d1a2953f887d45282ec990201298f47bedf8d4d`
  (`packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:347-382,599-1918`).
- Stacked-authority subcomponents: declared surface
  `5b5a4470e2612365f253e933833bb08b8f9c857ea0bc175b958ead9a74abee01`,
  gap plan
  `f41319a95750a441676bc6599b1de6bb49a87b45d83b9263d627c402cfe8e750`,
  late-producer schedule
  `9b15db577b85c796944e8eb267500d5d662f2a0eee77b25c1e4241c7d9620473`,
  metric registry
  `d75cb9b29f8b0a9a085471a11f4c19c32ba04cbe5419053df94ea81cbe6125a9`,
  and post-PUF transfer surface
  `a31e8a9512ec829c98745ed9ca2177e66d529bdc4b096ecfa2b4452f7bd41d73`
  (`packages/microcosm-build/tests/test_spec_engine_stacked_authority_semantics.py:70-100`).
- Specs and test goldens: US
  `3189d90dec95c8ea7090e41b5283fa52b1e6855bed4a776dfa02820f2bd11c62`,
  BE `86143c1c3f98980e34490c75706fe7dbf72e96e8accb8d7d4bbd2c1ae1a29b65`,
  UK `8f25ca46339a660b0022830228e39706fe872cdb0e5ca1d28b356f24fe6ec391`,
  loader vector
  `1ff676072985f104b9d80e3e5fa6e2078778969246246cf1371e982c772550cc`,
  late-schedule payload
  `5921cda83725b2801f2713242003e99ba54766851808b94a4f483666bce604c5`,
  target-name set
  `c792f12f0ef34f8a2ca9f16e68f5b306391eed56f120cda247bf778a95118c15`,
  legacy manifest
  `63c6e6973079f0b793d5435113aaae66184564b70271f8af120fecdbb5015f63`,
  coverage-document identity
  `4b39450dbdb8dafb83c3b627123b8026c6f82c660b66fe76f341a67c4f37c77b`,
  and report-only stacked observation
  `044d8a45c4fe42eec9f72f9bedbf403536b9734cf74d8d65b72caf3c6c1d60b7`
  (the respective assertions in `packages/microcosm-build/tests/`, including
  `test_us_multispine_pool_tool.py`, `test_spec_engine_country_bundles.py`,
  `test_spec_engine_loader.py`, `test_us_spec_bundle.py`, and
  `test_us_multispine_pool.py`; report-only observations in
  `docs/evidence/spec-engine/us-f0-coverage.json`).

## 2026-08-24 — final validation and memory receipts

- Regeneration/freshness is clean: the release-input builder reproduced 163
  required inputs, 7 reviewed exclusions, and 41 reform probes; the parity
  builder reproduced 32 compiled targets and 52 reviewed exclusions; the US
  bundle `--check` reproduced spec SHA-256
  `3189d90dec95c8ea7090e41b5283fa52b1e6855bed4a776dfa02820f2bd11c62`;
  and `tools/spec_engine_coverage.py --check` reports 42,096/42,096 fields and
  40/40 inventory checks. The generators enforce byte-for-byte checked-in
  agreement rather than tolerances
  (`tools/generate_us_bundle_from_constants.py:355-367,410-437`;
  `tools/spec_engine_coverage.py:378-398`).
- Package shards, run serially with `python -m pytest ... -p
  no:cacheprovider`: calibrate 203 passed; data 318 passed / 2 skipped; fit 93
  passed; frame 295 passed / 36 skipped. Their peak RSS values were 462,896,
  769,408, 872,432, and 6,904,032 KiB respectively.
- A canonical single-process build-shard run was behaviorally green (6,304
  passed / 39 skipped) but retained 18,548,960 KiB, so it was rejected under
  the binding 15 GiB ceiling. The same 6,341 collected tests were then proven
  to be an exact disjoint partition: 4,161 items excluding
  `test_us_[n-z]*.py`, plus 2,180 items in those 64 files. Serial fresh-process
  receipts were 4,127 passed / 36 skipped at 12,596,384 KiB and 2,177 passed /
  3 skipped at 13,363,984 KiB. Combined, they reproduce exactly 6,304 passed /
  39 skipped and every build test, with each process below 15 GiB. The process
  boundary follows the existing CI rationale that retained fixtures can
  outgrow a runner and each shard must have independent attribution
  (`.github/workflows/test.yml:24-34`). No assertion, gate, threshold,
  tolerance, or model input changed for this memory control.
- Complete accepted pytest total: 7,213 passed / 77 skipped / 0 failed. Ruff,
  generated-resource checks, source whitespace checks, and final repository
  status are recorded in `FINAL_REPORT.md` after the documentation pass.
- No pool or release build ran. Nothing was pushed, and no gate, threshold,
  tolerance, or band was changed.

## Historical replacement-scorecard notes

The sections below came from the merged replacement-scorecard lane and were
accurate when written. They are retained as history, not as current state.

## 2026-08-22 — lane start and environment

- Branch: `replacement-scorecard`, starting at `2aa96795` (post-#741
  `origin/main`).
- Owner ruling: publication evidence is a same-yardstick incumbent-versus-
  candidate comparison. This lane builds that yardstick and scores only the
  incumbent because the 25% bundle-mode candidate does not yet exist.
- Standing constraints recorded: no pushes, no pool builds, no gate/threshold/
  band tuning, green suite per commit, scoring below 20 GiB RSS, and a build
  queue check before scoring.
- Environment: the default-cache sync was denied by the managed sandbox, and a
  task-local empty cache could not download through disabled DNS. The sibling
  `microcosm-one-surface` checkout has the identical `uv.lock` (`895535...`)
  and a complete environment. Its `.venv` was cloned copy-on-write, after
  which a narrow writable cache of locked Hatch build requirements allowed
  `uv sync --offline --all-packages --extra us` to rebuild and relink all five
  Microcosm workspace packages to this worktree.
- The GitNexus exploration workflow was selected for the requested execution-
  path audit, but this session exposes no GitNexus resources or query tools.
  Repository-wide `rg`, symbol inspection, and focused tests are the fallback.
- The inherited one-target-surface notes below were accurate when written and
  are historical after #741; they are not current replacement-lane state.
- Journal-step validation: `uv run python -m pytest -q
  packages/microcosm-build/tests/test_us_state_files_scorer.py` passed (5/5),
  and `uv run ruff check .` passed. A complete workspace baseline is still
  running; the direct `python -m pytest` form avoids the cloned environment's
  stale console-script shebang.

## 2026-08-22 — yardstick audit

- One compiler surface: `compile_us_fiscal_target_registry` has no
  artifact-membership switch
  (`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-1017`),
  and the existing fiscal scorer's only divergent branch is loading before the
  common repair/materialize/score sequence
  (`tools/score_us_fiscal_targets.py:432-489`).
- Full-surface rule: the head-to-head must reject either
  `target_compilation.dropped_target_names` or `CalibrationResult.skipped`;
  both lower layers otherwise report and continue past an unmaterialized or
  uncompileable row
  (`tools/build_us_fiscal_refresh_release.py:4323-4347`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355`).
- Aggregate rule: use the production
  `sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct`
  constants, with no family multipliers. Target-scale square roots are
  normalized within amount/count basis, semantic concept groups receive one
  concept budget, and present bases receive equal total budgets
  (`tools/build_us_fiscal_refresh_release.py:344-348,5781-5814,6214-6290`).
  The aggregate is the weighted mean of capped target-scaled absolute errors
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:473-518`).
- Incumbent identity: PolicyEngine.py 4.15.0's bundled manifest names retired
  data package 1.115.5 and its historical HF model repo, immutable
  resolver revision `9531fe1d096244fe7eb45d791d52ef61b8a2a0a5`, filename
  `enhanced_cps_2024.h5`, and SHA-256
  `0a6b961ad363a421bde99f2c8e5d8f20370bcba45fd303050537a25bdd805b14`
  (`policyengine/data/release_manifests/us.json@4.15.0:12-27,37-42`).
  Microcosm's frozen parity reference pins revision `21280dca...` for the same
  hash (`packages/microcosm-build/src/microcosm/build/us/ecps_parity_reference.json:7-14`).
  Both cache-resolved files independently hash to `0a6b961a...`; the scorecard
  will retain both identities and call the former the package-resolved one.
  **[Superseded 2026-08-22, later session: 4.15.0 was tagged 2026-06-10 and is
  not the live package. See "incumbent identity corrected" below — the live
  policyengine.py 5.0.3 default US dataset is the buildp sparse artifact, not
  enhanced_cps_2024.]**
- Terminal-battery scope: all 131 marginal comparisons and the joint
  immigration comparison are by-origin-only. Each needs support-channel and
  clone-role columns to build separate ASEC and ACS masks
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11366-11378,11523-11533,11577-11580,11679-11739`).
  The incumbent has neither field, so every battery comparison is
  **inapplicable**, not failed or zero. A finished pool publishes an input-only
  H5 while sealing terminal results into its manifest and diagnostics
  (`tools/build_us_multispine_pool.py:3263-3334,3533-3550,3766-3786`); the
  manifest loader validates the H5, diagnostics, digests, run identity, and
  passing terminal alias before returning the frame
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:348-430,672-811`).
  **[Partially superseded 2026-08-22, later session: the by-origin-only
  classification and the pool-manifest receipt path stand, but the "incumbent
  has neither field" premise described the eCPS. The actual live incumbent
  (buildp sparse) carries both provenance columns; what it lacks is any
  ACS-origin row — see "incumbent identity corrected" below.]**
- Yardstick facts: the v9.4 feed at
  `/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
  has SHA-256 `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
  A read-only compile with the packaged current-vintage CD crosswalk, target
  period 2024, no aging, and the standing scoring period waiver produced
  32,842 specs at registry version `c4ac617743f2`. No artifact was built or
  scored during this compile.

## 2026-08-22 — incumbent identity corrected (fresh session, post-salvage)

- Environment: this session's `uv sync --all-packages --extra us` completed
  normally (network available; the earlier codex-sandbox venv cloning is
  historical).
- **The live policyengine.py US dataset is the microcosm buildp sparse
  artifact, not enhanced_cps_2024.** PyPI's latest `policyengine` is 5.0.3
  (tagged 2026-08-21; 4.15.0, the version the earlier audit read, was tagged
  2026-06-10). The 5.x bundle replaced the per-country release manifests:
  `get_release_manifest` reads the bundled
  `src/policyengine/data/bundle/manifest.json`, and
  `resolve_managed_dataset_reference(country, dataset=None)` returns
  `manifest.default_dataset_uri`
  (`policyengine.py@5.0.3 src/policyengine/provenance/manifest.py:301-320,540-561`);
  `default_dataset_uri` returns the certified artifact URI when
  `certified_data_artifact.dataset == default_dataset`
  (`.../provenance/manifest.py:181-187`), and dataset overlays are additive
  only — an overlay that shadows the default raises
  (`.../provenance/manifest.py:270-299`).
- The 5.0.3 bundle's US entry: `default_dataset: populace_us_2024`,
  `data_producer: populace`, build id
  `populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`,
  HF repo `policyengine/populace-us` (repo_type `dataset`), revision equal to
  the build id, filename `populace_us_2024.h5`, SHA-256
  `48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`,
  certified for policyengine-us 1.764.6 — the same engine version this
  workspace's `uv.lock` pins
  (`policyengine.py@5.0.3 src/policyengine/data/bundle/manifest.json`
  `data_releases.us.{default_dataset,certified_data_artifact,data_package,model_package}`).
- Local bytes verified: the HF cache ref for that revision points at commit
  `26dcad66867687f15735dc4926523e3741920836`, whose snapshot
  `populace_us_2024.h5` (462,915,783 bytes) hashes to `48b9d479...` exactly.
- Observed incumbent shape (read from those bytes this session): entity-table
  layout (six US entities + `_time_period` 2024), 57,240 households, all
  household weights positive; provenance columns present
  (`household_support_channel`, `household_support_clone_index`, person
  equivalents); channel counts `asec` 22,200 (clone 0) and `puf_tax_detail`
  35,040 (clone 1); **zero `acs`-channel rows**. The by-origin battery
  compares `asec` vs `acs` channels
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11577-11580,11695-11703`;
  channel constants `support_provenance.py:31` and `stacked_spine.py:263`),
  so every comparison is **inapplicable on observed evidence** — the
  incumbent predates the ACS stack and has no ACS origin to compare, which
  is a different (and observed) reason than the eCPS "no provenance columns"
  premise.
- CD provenance: the buildp H5 root attrs are PyTables boilerplate only — it
  predates the CD vintage crosswalk attributes, so
  `_assert_cd_vintage_support_matches` would fail on the attr comparison
  (`tools/build_us_fiscal_refresh_release.py:2519-2567,2570-2597`). The
  head-to-head scorer probes the attrs: strict when present, and otherwise
  records an explicit legacy waiver receipt while still requiring the
  household `congressional_district_geoid` lookup column to exist with
  positive values (observed present on the incumbent).
- Salvage adoption: the 1306-line sol draft was adopted after verifying every
  imported symbol and cited mechanism against the code this session. Rewrites
  beyond the identity block: entity H5s load through the canonical scorer's
  `release._load_frame` seam (`tools/score_us_fiscal_targets.py:436`,
  `tools/build_us_fiscal_refresh_release.py:2454-2471`), the
  dropped-target check now runs before scoring (clear failure instead of a
  loss-weight shape error), the battery inapplicability receipt is computed
  from observed origin-channel counts instead of asserted, and the Markdown
  scorecard renders rollups plus worst rows with the complete per-target
  table living in the JSON twin.

## 2026-08-22 — scorer landed; incumbent probe on real bytes (resumed session)

- Battery entities are `person`/`tax_unit`/`spm_unit` (114/9/8 single-column
  comparisons plus the joint person immigration comparison) — household is
  not a battery entity
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3011-3025`).
  The test fixture originally put provenance columns on household+person and
  its two battery tests failed; the fixture now provisions channel/clone
  columns on exactly the battery entities. The scorer needed no change.
- Real-bytes probe of the incumbent (the scorer's own probe path run against
  the cached `populace_us_2024.h5`, sha256 `48b9d479...`): layout
  `entity_tables`; `read_nullable_us_h5_metadata` raises "no artifact
  metadata" (caught → not a naked pool); CD vintage attrs both null with a
  usable `household.congressional_district_geoid` lookup (57,240 rows, 436
  positive unique geoids → the recorded legacy waiver path); frame loads as
  166,321 persons / 57,240 households / 79,729 tax units / 59,900 spm units
  with calibrated household weights. All three battery entities carry
  provenance columns; per-entity channel counts: person asec 66,001 +
  puf_tax_detail 100,320; tax_unit asec 30,974 + 48,755; spm_unit asec
  23,286 + 36,614; **zero `acs` rows on every entity** → the battery payload
  reports `inapplicable` with the observed empty-ACS-side reason. This
  verifies the earlier journal claim and extends it to tax_unit/spm_unit.
- Suite state at commit: head-to-head tests 7/7; state-files scorer,
  refresh-builder, pool-h5-io, fiscal-targets, and release-target-parity
  files all green (428 tests); ruff check + format clean.

## 2026-08-22 — newest salvage reconciled; candidate boundary made scoreable

- Inspected `refs/claude-salvage/replacement-scorecard-20260822-220105-73402`
  (`5577ee4c`) and verified that its scorer blob exactly matched the inherited
  uncommitted file. The salvage's household slicing was retained deliberately,
  but its full-pool measure-array assembly was replaced: a 25% dense pool at
  roughly 918,350 households would require about 56 GiB for one
  8,192-column float64 copy. The scorer now materializes and scores one fixed
  household slice at a time, checks weights plus target/scale/name/column
  contracts on every slice, accumulates the additive matrix/weight products in
  fixed order, frees the slice, and checks RSS before continuing
  (`tools/score_us_release_head_to_head.py:654-852,1399-1477`;
  `tools/build_us_fiscal_refresh_release.py:3730-3750`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/score.py:79-142`).
- The production pool loader remains a readiness boundary. A separate
  scorecard-evidence loader accepts the exact current stacked
  `status=gate_failed` / `simulation_ready=false` pair without weakening any
  manifest/diagnostics/H5 digest, schema, run-ID, materializer, terminal-alias,
  transition-authority, weight-kind, or row-count authentication
  (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:371-588,719-869`).
  This implements the owner's head-to-head ruling without relabeling a failed
  candidate as simulation-ready.
- Finished entity H5s do not retain the stacked assembly/tail manifests. Their
  battery path therefore evaluates the canonical authority, 132 comparisons,
  369 nominal scalar legs, clone-0 positive-weight scopes, metrics, support
  rules, and tolerances as artifact evidence, while explicitly reporting
  `production_receipt_authenticated=false`. The one structural ACS
  group-quarters scope is derived from retained origin/clone/`TYPEHUGQ`/
  membership/tenure columns and marks assembly authentication false
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7808-7897,11474-11492,11530-11898`).
- Validation: the combined focused run covered 19 scorer/pool/battery cases.
  Eighteen passed; the only failure was an assertion matching the wrong error
  wording, after which that corrected test reran green. Ruff check/format,
  Python byte compilation, and `git diff --check` pass. No pool build, push,
  gate edit, threshold edit, or band edit occurred.
- Mainline reconciliation: merged the six newer `origin/main` commits at
  `34d93846` with no conflict; the only US change among them isolates the
  fiscal-refresh memory canary. Post-merge validation passed all 13 scorecard
  tests and all 8 fiscal-memory tests (21/21).
- Required pre-score process check: `ps ax | grep
  build_us_multispine_pool` was attempted and denied by the managed macOS
  sandbox; `pgrep` and `top` are denied as well. The permitted `lsof -d cwd`
  process scan found no build-runtime working directory, and a full permitted
  file-descriptor scan found no open `build_us_multispine_pool`, pool H5,
  checkpoint, or pool-manifest path. The incumbent score may therefore start;
  no pool build was launched by this lane.

## 2026-08-23 — incumbent yardstick established

- Exact command run from this worktree after the empty queue scan:

  ```bash
  .venv/bin/python tools/score_us_release_head_to_head.py \
    --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
    --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
    --out-prefix experiments/replacement_scorecard/incumbent_48b9d479 \
    --maximum-microsim-batch-size 5000
  ```

- The scorer completed with exit 0 and peak RSS 18.666 GiB. It used registry
  `c4ac617743f2` (32,842 unique target rows), Ledger facts SHA-256
  `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`,
  and CD crosswalk SHA-256
  `c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec`.
  The fixed streaming plan used five registry chunks and twelve household
  slices per chunk, with a conservative live target-column payload bound of
  983,040,000 bytes (`tools/score_us_release_head_to_head.py:654-852,1399-1477`).
- Incumbent fiscal evidence: weighted loss
  `0.11462448275649702`; fraction within 10% `0.2669143170330674`; 57,240
  households and 57,240 nonzero shipped weights. The 32,842 per-target
  contributions sum to `0.11462448275649767`, agreeing with the canonical
  aggregate to floating-point summation tolerance. The production weighting
  and aggregate formulas are code-cited in the result
  (`tools/build_us_fiscal_refresh_release.py:344-348,481-516,5781-5814,6214-6290`;
  `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:471-537,576-600`).
- Incumbent battery evidence: 132/132 comparisons and all 369 nominal scalar
  legs are explicitly `inapplicable`; the artifact has 120,261 positive-weight
  clone-0 ASEC rows across the three battery entities and zero ACS rows. No
  zero, pass, or failure was synthesized for those by-origin-only legs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11644-11709,11824-11832,11948-12154`).
- Result identities: JSON SHA-256
  `b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8`;
  Markdown SHA-256
  `3f9171b8f63fcef61518a4af1c18a8555c4f449ac62e9283e41ac2fe9c779021`.
  Both record the incumbent artifact SHA-256 `48b9d479...`, exact HF repo,
  revision, resolved commit, filename, PolicyEngine.py bundle version/source
  commit, and certified policyengine-us version.

## Owner command when the 25% candidate exists

First confirm that the host builder has finished. On the owner host (where
process inspection is permitted), `ps ax | grep '[b]uild_us_multispine_pool'`
must print nothing before either scoring command. Set the two published
candidate paths, then run the dense pool and sparse-57k views separately on
the same frozen incumbent and Ledger yardstick:

```bash
CANDIDATE_POOL_MANIFEST=/absolute/path/to/25pct/pool.manifest.json
CANDIDATE_SPARSE_H5=/absolute/path/to/25pct/sparse-57k.h5
CANDIDATE_MANIFEST_SHA256="$(shasum -a 256 "$CANDIDATE_POOL_MANIFEST" | awk '{print $1}')"
CANDIDATE_POOL_SHA8="$(jq -r '.pool_h5.sha256[0:8]' "$CANDIDATE_POOL_MANIFEST")"
CANDIDATE_SPARSE_SHA8="$(shasum -a 256 "$CANDIDATE_SPARSE_H5" | awk '{print substr($1,1,8)}')"

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
  --candidate "$CANDIDATE_POOL_MANIFEST" \
  --candidate-manifest-sha256 "$CANDIDATE_MANIFEST_SHA256" \
  --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
  --out-prefix "experiments/replacement_scorecard/head_to_head_dense_48b9d479_${CANDIDATE_POOL_SHA8}" \
  --maximum-microsim-batch-size 5000

.venv/bin/python tools/score_us_release_head_to_head.py \
  --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5 \
  --candidate "$CANDIDATE_SPARSE_H5" \
  --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl \
  --out-prefix "experiments/replacement_scorecard/head_to_head_sparse_48b9d479_${CANDIDATE_SPARSE_SHA8}" \
  --maximum-microsim-batch-size 5000
```

“Better than the incumbent” on this yardstick means the owner compares each
candidate view with the incumbent’s exact `0.11462448275649702` weighted fiscal
loss and the reported target-by-target balance of lower, equal, and higher
absolute relative errors, while also inspecting every battery leg that is
computable for that candidate. There is no scorecard threshold and no automatic
conjunction or verdict. Because the incumbent’s ASEC-vs-ACS battery is
definitionally inapplicable, the candidate’s battery is standalone evidence—it
is not compared with a fabricated incumbent zero, pass, or failure. The owner
decides whether the dense and sparse evidence justifies the flip.

## Final validation and completion

- The prescribed full workspace suite was run through the current worktree
  interpreter because the copied `.venv/bin/pytest` console script retained a
  stale sibling-worktree shebang. The first correctly routed run reached 7,027
  passes and 76 skips; its only failure was the source-hygiene guard finding a
  retired-package literal in this journal. Commit `6847d245` removed that
  documentation-only literal, and the guard passed independently.
- Final full-suite receipt:

  ```text
  UV_CACHE_DIR=/tmp/microcosm-scorecard-uv-cache uv run python -m pytest
  7028 passed, 76 skipped, 1922 warnings in 5996.15s (1:39:56)
  ```

- Final static receipts: repository-wide `ruff check .` passed; all six Python
  files changed on this branch passed `ruff format --check`; scorer
  `py_compile` and `git diff --check` passed. A whole-tree format audit listed
  69 pre-existing mainline files, which this lane deliberately did not rewrite.
- Independent final audit found no deliverable-level gap. The implementation,
  incumbent JSON/Markdown, exact candidate commands, comparison doctrine,
  `PROGRESS.md`, and `FINAL_REPORT.md` are complete. The only next action is
  external: the host builds the candidate, then the owner scores its dense and
  sparse views and decides the flip.
- No pool build, publication, push, gate edit, threshold edit, tolerance edit,
  or band edit occurred in this lane.

---

# Historical: one-target-surface lane notes

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

## 2026-08-24 — stacked-pool CD-vintage provenance contract complete

### Recovery and environment

- The live worktree's tracked tree exactly matched salvage commit `ca26ea21`,
  so the salvaged implementation was retained and audited rather than
  restarted. `PROGRESS.md` and this root journal were already committed at lane
  start in `5925f808`.
- The required lock-exact environment command completed offline after the
  sandbox rejected the default uv cache:

  ```bash
  UV_CACHE_DIR=/private/tmp/microcosm-scorecard-uv.0rntvY/cache \
    uv sync --offline --all-packages --extra us
  ```

  It checked all 100 locked packages. Test execution used
  `uv run python -m pytest` so imports resolve to this worktree.

### Five-part repair

1. The stacked CLI now requires a national PUMA ladder and the canonical
   117th-to-119th congressional-district crosswalk as explicit path/SHA pairs.
   Missing arguments, noncanonical declared pins, or mismatching bytes fail
   before any source build
   (`tools/build_us_multispine_pool.py:563-594,897-1004`). The real assignment
   runs only after operator-free source validation and stack assembly, then
   before gap fill and clone operators
   (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:346-406`;
   `tools/build_us_multispine_pool.py:5290-5334`).

2. Configured and checkpoint identities bind both byte authorities, the
   117th-to-119th vintage declaration, algorithm
   `assign_us_puma_ladder.population_weighted_overlap.v1`, assignment order,
   seed site `legacy_puma_ladder`, stream `geography_legacy`, and value `0`.
   The ordered native-household assignment receipt persists through every
   resumable stage and the schema-9 terminal manifest
   (`tools/build_us_multispine_pool.py:814-928,1300-1469,1711-1930,3530-3635,3989-4027,4133-4215`).

3. Nullable H5 publication now accepts caller-owned root attributes, validates
   their names/values, writes them with the fixed entity tables in a temporary
   sibling, verifies their exact round trip, and replaces the destination only
   after successful verification. The stacked publisher writes the crosswalk
   SHA and `119th_congress`
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:1463-1699`;
   `tools/build_us_multispine_pool.py:4423-4519`). Schema-9 loading binds the
   manifest receipt to those physical attrs and recomputes native household ID
   and geography digests. It requires integral source/clone roles, exactly one
   native per source, positive integral CD values, and coherent PUMA/CD/county
   geography across clone roles
   (`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:671-923,1032-1193,1321-1364`).

4. Release preflight reads root attrs and the household frame within one
   HDFStore and delegates fixed/table representation handling to the shared
   reader. The guard's SHA equality, current-vintage equality, and positive-
   household-support checks remain intact
   (`tools/build_us_fiscal_refresh_release.py:2565-2677,8608-8612`).

5. The integration test calls the real stacked entry point with tiny
   fixture/stub sources, executes the real post-assembly geography assignment
   and publisher, proves fixed household storage and both H5 attrs, then calls
   the real release assertion and observes positive district support
   (`packages/microcosm-build/tests/test_us_multispine_pool_tool.py:2279-2335`).
   H5 consumer tests additionally reject attr drift, native-geography digest
   drift, missing clone lineage, and divergent clone geography; writer tests
   cover atomic attr round trip and pre-replacement failure
   (`packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1357-1440`;
   `packages/microcosm-build/tests/test_us_acs_multispine_base_builder.py:96-197`;
   `packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:1061-1184`).

### Assignment decision

A one-value deterministic PUMA-to-CD allocation is not defensible: 2020 PUMAs
and congressional districts do not nest, while the authority contains the
block-population overlap distribution. Selecting only the largest overlap
would erase supported within-PUMA geography. The existing ladder instead
preserves observed ACS PUMA, assigns missing ASEC PUMA proportional to 2020
PUMA population, and samples CD/county within PUMA proportional to block
population. Its stable state/PUMA order and single generator make output
reproducible from the bound seed
(`packages/microcosm-build/src/microcosm/build/us_runtime/puma_ladder.py:1-20,293-383,638-698`).
The seed ledger owns `legacy_puma_ladder` on `geography_legacy`, default `0`,
and the generated spine declaration carries the same draw contract
(`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:870-887`;
`packages/microcosm-build/src/microcosm/build/us/spec/spine.yaml:421-432`).

### Anti-rot and identity receipts

- Canonical US spec SHA:
  `3189d90dec95c8ea7090e41b5283fa52b1e6855bed4a776dfa02820f2bd11c62`
  to `5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e`.
- Field-pointer inventory SHA:
  `6d7353c6c42a6e1dbc6e3a227848e36864526fdc9533d5b284aa469c87dc064f`
  to `bc4a948ab632191954600da8474c5b011f977a65e24c399d126f3dc4a79f23e5`.
- Full checkpoint SHA:
  `b6a47fac54d7de7aa42ce59dc1950c0765b7a67034f0174ad1531d5bbb06ceef`
  to `a128a85f877fb32def9382b841b8b340f974e8a9148ac029c1f04becdc956c18`.
- Pool-code diagnostic SHA:
  `044d8a45c4fe42eec9f72f9bedbf403536b9734cf74d8d65b72caf3c6c1d60b7`
  to `91c65a9ff36839d575036264c4bf57ffde6457e8fd180a0856f9be712ada371d`.
- New geography-assignment SHA:
  `f49425ca8734ac559c73cf44f6458d86d3162a48956b98a27e6e758959361585`.
- Stacked checkpoint materializer moved 11 to 12, terminal manifest schema 8
  to 9, nullable H5 materializer 2 to 3, and geography receipt absent to
  schema 1. Pool-stage materializer 7, stacked authority 11, and late-registry
  schema 16 did not move; their documentation changes only correct stale prose
  (`tools/build_us_multispine_pool.py:332-360`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:75-105`;
  `docs/us-multispine-operator-ordering.md:1-120`).

The generated field ledger moved authored fields 32,331 to 32,351, resolved
bindings 9,765 to 9,769, configuration/consumed fields 42,096 to 42,120, and
claims 47 to 49. Compiler-semantic/front-end-validation/identity-only/legacy
modes moved 27,688/346/128/13,934 to 27,699/348/103/13,970; unused and
multiply-primary-used remain zero. Source records moved 7 to 8, vintage records
15 to 16, resolved references 325 to 334, full/static checkpoint components
12/9 to 13/10, and inventory checks 40 to 41
(`packages/microcosm-build/src/microcosm/build/spec_engine/field_usage.py:29-31,387-393,671-692,803-822`;
`packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:348-380,417-453,1649-1712`;
`docs/evidence/spec-engine/us-f0-coverage.json:1-15,775-788,1860-2060,2602-2612`).

Both generated freshness checks pass byte-for-byte:

```text
US bundle spec_sha256=5378bb9189aec96f50da22aac71e5bd2c3d919e9795f6ef2147e0bc9c739dd8e
spec-engine coverage: 42120/42120 configuration fields; 41/41 inventory checks
```

The new declarations and schema rules are generated in
`us/spec/{geography,sources,vintages,spine}.yaml`; the field-usage ledger and
inventory exact item are their fail-closed consumers
(`packages/microcosm-build/src/microcosm/build/us/spec/geography.yaml:3-37`;
`packages/microcosm-build/src/microcosm/build/us/spec/sources.yaml:69-86`;
`packages/microcosm-build/src/microcosm/build/us/spec/vintages.yaml:45-53`;
`packages/microcosm-build/src/microcosm/build/spec_engine/schema/geography.schema.json:108-150`;
`packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py:1649-1712`).

### Validation and operational discipline

- `microcosm-calibrate`: 203 passed.
- `microcosm-data`: 318 passed / 2 skipped.
- `microcosm-fit`: 93 passed.
- `microcosm-frame`: 295 passed / 36 skipped.
- `microcosm-build` partition A: 4,155 passed / 36 skipped.
- `microcosm-build` partition B: 2,177 passed / 3 skipped.
- Accepted total: **7,241 passed / 77 skipped / 0 failed**.
- Repository-wide Ruff, both generator freshness checks, smoke-script shell
  syntax, and `git diff --check` pass. The source-blind import-graph pin covers
  the exact 69 runtime modules now reached
  (`packages/microcosm-build/tests/test_us_spine_blindness.py:3270-3315`).

No production pool or release build was required: integration evidence is
fixture-sized. No push occurred, the release guard/operator boundary were not
weakened, and `logbook-pending-chain.txt` was not touched.

## Final candidate Stage-1 flags

Keep the current six authenticated source pairs, `--sample-fraction 0.25`,
`--sample-seed 578`, `--clone-attachment-fraction 1.0`,
`--clone-attachment-seed 578`, `--checkpoint-root`, and `--out`. Add these
exact runbook variables:

```bash
PUMA_LADDER="/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/build/us/us_puma_ladder_2020.npz"
PUMA_LADDER_SHA="39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8"
CD_CROSSWALK="$WT/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv"
CD_CROSSWALK_SHA="c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"
```

Add these exact `check_pool_inputs` calls:

```bash
check_sha256 puma-ladder "$PUMA_LADDER" "$PUMA_LADDER_SHA"
check_sha256 cd-crosswalk "$CD_CROSSWALK" "$CD_CROSSWALK_SHA"
```

Add these exact `POOL_COMMAND` arguments:

```bash
--puma-ladder "$PUMA_LADDER"
--puma-ladder-sha256 "$PUMA_LADDER_SHA"
--congressional-district-vintage-crosswalk "$CD_CROSSWALK"
--congressional-district-vintage-crosswalk-sha256 "$CD_CROSSWALK_SHA"
```

The parser and canonical-pin validator fail closed on any absent or different
value (`tools/build_us_multispine_pool.py:563-594,897-928`). Preserve the
runbook's wait loop until no other `build_us_*` process is running, invoke the
stage through `env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST`, pass no
`--logbook-prev-row-digest`, and never touch `logbook-pending-chain.txt`.

## Final checkpoint-reuse verdict

**No: the existing smoke pool checkpoints and pool H5 are not reusable.** The
configured namespace now binds both new SHA pins, the base identity binds the
complete geography contract, and the checkpoint materializer moved from 11 to
12; old namespaces are not selected, and a relocated stale manifest fails the
exact identity comparison
(`tools/build_us_multispine_pool.py:332-360,1300-1555`). The old schema-8 /
materializer-2 pool H5 also lacks the required physical attrs and household
support binding and is rejected by schema-9 loading
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:75-105,671-923,1032-1193`).
The six immutable raw source artifacts remain reusable, but Stage-1 stage
checkpoints, downstream banks, the terminal manifest, and the pool H5 must be
regenerated under the new identity.
