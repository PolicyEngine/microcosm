# Final report: PolicyEngine-US 1.819.0 lock bump

## Outcome

The branch is ready for the owner to open the PR. `uv.lock` now resolves
`policyengine-us==1.819.0` and `policyengine-core==3.31.0`; the complete
resolver version movement is exactly:

```text
policyengine-core 3.26.11 -> 3.31.0
policyengine-us   1.764.6 -> 1.819.0
```

NumPy remains 2.4.6 and Torch remains 2.12.0. The lock binds the official
PE-US 1.819.0 wheel SHA-256
`525bdf8b238c3eb11cd60c5f4f7a7b0c57bc7eea5c1cf4346c261241b061be45`
(`uv.lock:1366-1421`). The required upgraded
`uv sync --all-packages --extra us` completed for all five workspace packages
from that lock and task-local official PyPI artifacts.

## Compatibility repairs

- WIC's upstream input is now `takes_up_wic_if_eligible`; Microcosm writes
  that verified successor while retaining the historical draw salt, and the
  six current cross-entity consumers are frozen with direct-receiver and
  aggregation ownership
  (`packages/microcosm-build/src/microcosm/build/us_runtime/wic_claim.py:106-109,392-409`;
  `packages/microcosm-frame/src/microcosm/frame/adapters/_policyengine_us_source_index.py:1694-1731,2004-2090`;
  `packages/microcosm-build/tests/test_us_pool_input_consumers.py:235-359`).
- Reported TANF now writes the verified input successor `receives_tanf`, since
  PE-US owns `is_tanf_enrolled` as a formula
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:127-160,266-289,485-500`).
- New CA, CO, and NM premium-assistance take-up leaves remain explicit
  `engine_default`/`rate_unsourced` inputs because Microcosm has no reviewed
  participation source; no rate was invented
  (`packages/microcosm-build/src/microcosm/build/us/spec/take_up.yaml:507-563`).
- Mortgage-interest, Oklahoma pension, SNAP proration/missing-hours, and the
  expanded AL/NY/OK/PA `weeks_unemployed` consumers were updated or guarded
  against the installed graph
  (`packages/microcosm-build/tests/test_us_pool_input_consumers.py:491-575`;
  `packages/microcosm-build/tests/test_us_child_support.py:448-489`;
  `packages/microcosm-build/tests/test_us_weeks_unemployed.py:695-751`).
- The published US dataset remains certified for the prior engine lock.
  Certified loading still fails closed before construction on model/core
  mismatch; only the live smoke test recognizes that specific typed mismatch
  as an expected skip under this new development lock
  (`packages/microcosm-data/src/microcosm/data/loader.py:63-64,425-445`;
  `packages/microcosm-data/tests/test_loader.py:326-356,403-419`).

No pool-consumed upstream variable was genuinely removed without a successor,
so the owner-question stop condition did not trigger. The full installed-input
audit and mechanism-by-mechanism code citations are in `_LANE-NOTES.md` under
“verified upstream compatibility repairs.”

## Identities and compatibility note

Repository generators refreshed every affected raw-resource, generated-source,
engine-ABI, remaining-input, field-usage, inventory, seed-protocol, authority,
spec, coverage, and golden identity. Source bytes are part of the attested seed
protocol, so those module edits legitimately move dependent identities even
though NumPy and Torch did not move
(`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:320-367`;
`packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py:114-152`).
The final US spec is
`3189d90dec95c8ea7090e41b5283fa52b1e6855bed4a776dfa02820f2bd11c62`.

`_LANE-NOTES.md`, under “final tool-generated identities,” records all 46 final
values by mechanism. The compatibility commit body records every full old-to-
new digest mapping, including the four lock artifact hashes.

The requested short release-range note is in `_LANE-NOTES.md` under
“compatibility note for 1.764.6 through 1.819.0.” It flags receipt/take-up and
SNAP changes, OBBBA follow-through, major cash/health/housing and tax formula
changes, school-meal child-support treatment, and newly added state programs.
It is intentionally an owner-facing plausibility scan, not an exhaustive
policy audit.

## Verification

- `microcosm-calibrate`: 203 passed; peak RSS 462,896 KiB.
- `microcosm-data`: 318 passed / 2 skipped; peak RSS 769,408 KiB.
- `microcosm-fit`: 93 passed; peak RSS 872,432 KiB.
- `microcosm-frame`: 295 passed / 36 skipped; peak RSS 6,904,032 KiB.
- `microcosm-build`: exact complete inventory, 6,304 passed / 39 skipped. A
  canonical one-process run was green but retained 18,548,960 KiB and was not
  accepted under the 15 GiB ceiling. The exact 6,341 collected items were
  proven as disjoint 4,161-item and 2,180-item serial fresh-process partitions:
  4,127 passed / 36 skipped at 12,596,384 KiB, then 2,177 passed / 3 skipped at
  13,363,984 KiB. This follows the repository's documented fresh-process shard
  rationale (`.github/workflows/test.yml:24-34`) and changes no test assertion
  or model behavior.

Accepted total: **7,213 passed / 77 skipped / 0 failed**, with every accepted
process below 15 GiB RSS.

Final generated checks are green:

- release-input manifest: 163 required, 7 reviewed exclusions, 41 reform
  probes;
- target parity: 32 compiled, 52 reviewed exclusions;
- `tools/generate_us_bundle_from_constants.py --check`: US spec SHA above;
- `tools/spec_engine_coverage.py --check`: 42,096/42,096 configuration fields
  and 40/40 inventory checks. Both check paths reject stale bytes rather than
  using a tolerance
  (`tools/generate_us_bundle_from_constants.py:355-367,410-437`;
  `tools/spec_engine_coverage.py:378-398`).

Repository-wide `ruff check .` and `git diff --check` are green. No gate,
threshold, tolerance, or band was tuned. No pool or release build ran, no PR
was opened, and nothing was pushed.

## Commits and handoff

- `514964d4` — start the PolicyEngine-US 1.819.0 bump lane and commit the
  standing progress journal.
- The compatibility commit containing this report — lock bump, verified
  repairs, generated identities, complete digest mapping, and green receipts.

Next: the owner opens the PR from `bump-policyengine-us`.
