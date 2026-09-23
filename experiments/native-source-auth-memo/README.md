# Source-authentication memo evidence

The resumed 2026-09-23 session did **not** run the actual-archive ACS owner
benchmark or the 1/1000 native graph measurement. There are no successful
off/cold/warm benchmark receipts from this session and no measured speedup.
The validation below applies to the uncommitted lane worktree atop the
verified head `469b57b5b`. The sandbox refused creation of the shared Git
worktree `index.lock`, so these changes could not be committed in this session.

The first available-memory reading was about 79 GiB. However, `ps` failed with
`Operation not permitted`, preventing the required check that no other native
run was active. The task addendum explicitly prohibited benchmark and
actual-data runs when that check was unavailable. Memory alone did not admit
a run.

## Fixture validation in this resumed session

The final combined regression command passed **206 tests in 195.68 s**, with
56.94 GiB available immediately before launch:

```bash
.venv/bin/python -m pytest \
  packages/microcosm-build/tests/test_us_source_memo.py \
  packages/microcosm-build/tests/test_us_source_memo_codec.py \
  packages/microcosm-build/tests/test_us_source_memo_frames.py \
  packages/microcosm-build/tests/test_us_source_memo_parser_identity.py \
  packages/microcosm-build/tests/test_us_acs_person_coverage_authentication.py \
  --tb=short
```

This includes whole ACS+ASEC preparation equality with the memo off, cold and
warm; exact frame/receipt identity and trace-proven parser skips; object scalar,
missing-marker, string-storage and floating-bit preservation; invalidation and
refusal after live parser changes; and unsupported dtype/metadata rejection.

Two additional targeted commands passed:

```bash
.venv/bin/python -m pytest \
  packages/microcosm-build/tests/test_us_source_memo_codec.py \
  packages/microcosm-build/tests/test_us_acs_pums.py \
  packages/microcosm-build/tests/test_us_acs_person_coverage_columns.py
# 201 passed in 14.07 s; 74.52 GiB available before launch.
# This earlier run included 30 codec cases, which overlap the final suite.

.venv/bin/python -m pytest \
  packages/microcosm-build/tests/test_us_store_dependency_classification.py
# 14 passed; 56.72 GiB available before launch.
```

The first combined memo run exposed nine owner refusals from stale dependency
contracts and accepted preparation hashes in the inherited frame draft (102
other tests passed). The final run above follows the repair: three reviewed
contracts refreshed, source memo included in four additional ACS stages, and
the three changed owner-file hashes updated. The dependency tests cover all
ten maintained stage manifests and retain the stale-contract refusal checks.

Ruff lint and formatting checks passed for all eight changed/new Python files;
`git diff --check` passed. `tools/ci_test_groups.py --verify` passed for the
621 tracked test files. Because staging is blocked, a separate invocation of
the same verifier with the tracked inventory plus the three new test files
checked all 624 candidate files. The additions classify as fast `rest`, engine
`us-qs:build`, and `wheels`; none defaults to the shared group. The main session
must repeat the ordinary verifier after staging them.

Targeted tests exercise invented archives and frames. They establish code
contracts and memo equality on those fixtures; they do not measure actual
archive acceleration or certify a release.

## Inherited attribution

[`attribution-9af56aa8a-1-1000.json`](attribution-9af56aa8a-1-1000.json)
records attribution from the 2026-09-19 1/1000 run at
`9af56aa8a7a210e18674d484b11d36f417de6759`. Its source report SHA-256 is
`4e7be8adfe472a684cd8ff86a7139cffb42e8a3699d8072d02775fdd9554976d`.
[`attribute_outside_graph.py`](attribute_outside_graph.py) contains the
partitioning procedure. The resumed session inspected this checked-in evidence;
it did not reproduce the report or rerun that graph.

The inherited cold profile attributes 1,681.6 process CPU seconds to work
outside the graph, including 936.9 seconds across the two ACS issuers. These
are statistical estimates from sampled stacks. They locate prior costs; they
are not measurements of this memo implementation.

## Pending actual-archive benchmark

[`bench_acs_owners.py`](bench_acs_owners.py) issues the ACS catalogue and native
coverage for a deterministic approximately 1/1000 household selection. It
runs one phase per process and records digests, counts, timings and memo
counters. It does not select ASEC records, execute the native graph or build a
release.

Before any phase, an operator must confirm at least **40 GiB available** and
**no other native run active**. A failed or unavailable process check does not
satisfy that condition. The script's default memory threshold is 25 GiB, so
this lane must explicitly pass `--min-available-gib 40`.

After those checks succeed, run three separate processes with the same ACS
archives, seed, memo root and key. Use fresh work and memo paths, keep the key
outside the memo root, and preserve any existing results:

```bash
.venv/bin/python experiments/native-source-auth-memo/bench_acs_owners.py \
  --acs /path/to/staged/acs --work /path/to/new/owner-work \
  --phase off --memo-root /path/to/new/memo --memo-key /path/to/memo.key \
  --min-available-gib 40 --out /path/to/new/off.json
.venv/bin/python experiments/native-source-auth-memo/bench_acs_owners.py \
  --acs /path/to/staged/acs --work /path/to/new/owner-work \
  --phase cold --memo-root /path/to/new/memo --memo-key /path/to/memo.key \
  --min-available-gib 40 --out /path/to/new/cold.json
.venv/bin/python experiments/native-source-auth-memo/bench_acs_owners.py \
  --acs /path/to/staged/acs --work /path/to/new/owner-work \
  --phase warm --memo-root /path/to/new/memo --memo-key /path/to/memo.key \
  --min-available-gib 40 --out /path/to/new/warm.json
```

The cold phase requires a previously nonexistent memo root; the warm phase
requires the root populated by cold. Keep source code fixed through all three
phases because live code identity changes invalidate comparisons.

Compare the entire `catalogue` and `native_coverage` objects across all three
receipts. They must agree, including receipt, records, selected-key, payload
and frame digests and all counts. Record any failure instead of presenting an
incomplete receipt set as an equality result. Report CPU and wall seconds and
memo counters only after equality holds. The script records peak RSS using
`resource.getrusage(...).ru_maxrss`; its unit is platform-dependent despite the
JSON field name `peak_rss_bytes` (bytes on this macOS host).

The 1/1000 native graph measurement remains a separate pending run under the
same admission conditions. Owner timings cannot substitute for end-to-end
graph evidence. Poverty remains comparison-only.
