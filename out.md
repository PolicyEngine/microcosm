# PR #849 env-variable correction — final report

## What this was

PR #849's review comment (Fable, main) caught that `POPULACE_LEDGER_URL` /
`_KEY` / `_API_KEY` / `_EXPORT_KEY` are the **Logbook** store's credentials
(Supabase `logbook` schema, `logbook_writer` / `logbook_exporter` roles —
`packages/microcosm-build/src/microcosm/build/logbook.py`'s own docstring
and `logbook/README.md`), not Chronicle fact-store variables. "Ledger"
there is the generic build-ledger sense renamed to Logbook on 2026-08-08
(microcosm#632) specifically to stop colliding with Chronicle. The PR as
filed introduced `CHRONICLE_*` as the preferred spelling for these — the
wrong referent, recreating the exact collision #632 fixed. This task
renamed the preferred spelling to `LOGBOOK_*` everywhere and reframed the
docs/changelog/PR body accordingly. The epoch half of the PR
(`chronicle_epoch.py`, `ledger_artifact.py`, `import_entry_facts.py`,
`ledger_targets.py`) was approved as-is and left untouched except for two
stale module-path comments that had to follow the file rename.

## Branch state before starting

`chronicle-dual-accept` had 2 local commits ahead of `origin/chronicle-dual-accept`
(`d6cd24ca`, `1be2f18e`) that were unrelated follow-on work (journal exact-counts
edit, provenance docstring edit) not present on the PR head. Per the assignment,
the PR head (`origin/chronicle-dual-accept` = `4115573c`) is authoritative. Those
two commits were preserved on a new branch,
`backup/chronicle-dual-accept-pre-correction-1be2f18e`, before `chronicle-dual-accept`
was hard-reset to `4115573c`. `origin/main` was already an ancestor, so no rebase
was needed.

## New head SHA

**`a5121a1e0c4da8d7767fe8322bc948a957c7c5e7`** is the head of the substantive
code/docs/PR-body correction (all verification below was run against this
sha). One further commit, `b98edd88`, adds this report file (`out.md`,
following this repo's established per-task convention — see "out.md history"
below) on top, making `b98edd88` the actual current branch tip. Both were
pushed as fast-forwards of `origin/chronicle-dual-accept` (no force needed —
the branch was reset to the exact PR head before any new commits were added).

Three code/docs commits on top of `4115573c` (the original PR head):

1. `061544e1` — Rename the CHRONICLE_* env dual-read window to LOGBOOK_* (module
   rename + all symbol renames + every caller)
2. `ee73a624` — Reframe the env dual-read window as a Logbook cleanup, not
   chronicle#143 (changelog, README, journal correction note)
3. `a5121a1e` — Apply ruff import-sort and formatting to the renamed files

Plus the report commit:

4. `b98edd88` — Report the CHRONICLE_* to LOGBOOK_* correction on PR #849 (this file)

## Files changed (relative to 4115573c)

```
 PROGRESS-chronicle-dual-accept.md                                          |  29 +++++
 changelog.d/chronicle-dual-accept.added.md                                 |   4 +-
 logbook/README.md                                                          |   4 +-
 packages/microcosm-build/src/microcosm/build/__init__.py                   |  10 +-
 packages/microcosm-build/src/microcosm/build/logbook.py                    |  29 ++---
 .../build/{chronicle_env.py => logbook_env.py}                             |  91 ++++++------
 packages/microcosm-build/src/microcosm/build/uk_runtime/firm_generation.py |   2 +-
 packages/microcosm-build/src/microcosm/build/us_runtime/source_coverage.py |   2 +-
 .../tests/{test_chronicle_env.py => test_logbook_env.py}                   |  98 +++++++-------
 tools/logbook.py                                                           |  35 ++---
```

Plus the PR body itself, edited via `gh pr edit 849` (four scoped changes: the
`chronicle_env` → `logbook_env` framing paragraph, two audit-table rows naming
`chronicle_env`/`CHRONICLE_*`, and the `test_chronicle_env.py` → `test_logbook_env.py`
test-file mention — everything else verbatim, confirmed by diffing old vs. new body
text before posting).

### Rename map applied

| Old | New |
|---|---|
| `chronicle_env.py` | `logbook_env.py` |
| `chronicle_env()` | `logbook_env()` |
| `chronicle_env_names()` | `logbook_env_names()` |
| `describe_chronicle_env()` | `describe_logbook_env()` |
| `reset_chronicle_env_deprecation_warnings()` | `reset_logbook_env_deprecation_warnings()` |
| `CHRONICLE_URL_ENV = "CHRONICLE_URL"` | `LOGBOOK_URL_ENV = "LOGBOOK_URL"` |
| `CHRONICLE_KEY_ENV = "CHRONICLE_KEY"` | `LOGBOOK_KEY_ENV = "LOGBOOK_KEY"` |
| `CHRONICLE_API_KEY_ENV = "CHRONICLE_API_KEY"` | `LOGBOOK_API_KEY_ENV = "LOGBOOK_API_KEY"` |
| `CHRONICLE_EXPORT_KEY_ENV = "CHRONICLE_EXPORT_KEY"` | `LOGBOOK_EXPORT_KEY_ENV = "LOGBOOK_EXPORT_KEY"` |
| `CHRONICLE_ENV_LEGACY_NAMES` | `LOGBOOK_ENV_LEGACY_NAMES` |
| `test_chronicle_env.py` | `test_logbook_env.py` |

Unchanged (legacy names, same once-per-process `DeprecationWarning` behavior):
`LEGACY_URL_ENV = "POPULACE_LEDGER_URL"`, `LEGACY_KEY_ENV = "POPULACE_LEDGER_KEY"`,
`LEGACY_API_KEY_ENV = "POPULACE_LEDGER_API_KEY"`,
`LEGACY_EXPORT_KEY_ENV = "POPULACE_LEDGER_EXPORT_KEY"`. Warning message now
reads "...is the pre-rename name for LOGBOOK_...; the build ledger is now
Logbook (microcosm#632)..." — no chronicle#143 reference.

Untouched by design (per the reviewer's own note — these really do translate
Chronicle ids / pin a Chronicle commit, so they correctly stay `CHRONICLE_*`):
`CHRONICLE_ONS_TURNOVER_BANDS`, `CHRONICLE_ONS_EMPLOYMENT_BANDS`,
`CHRONICLE_HMRC_BANDS` (`firm_generation.py`) and
`CHRONICLE_US_SOURCE_COVERAGE_CONTRACT_COMMIT` (`source_coverage.py`). Only
their comments' dotted-path references to the old `chronicle_env` module were
updated to `logbook_env`, since that module no longer exists under the old name.

## Verification

### `uv sync --all-packages --extra us`
```
Resolved 125 packages in 15ms
Checked 102 packages in 68ms
```

### Targeted tests (exact command from the assignment)
```
uv run pytest packages/microcosm-build/tests/test_logbook_env.py \
  packages/microcosm-build/tests/test_chronicle_epoch.py \
  packages/microcosm-build/tests/test_logbook.py \
  packages/microcosm-build/tests/test_logbook_cli.py \
  packages/microcosm-build/tests/test_logbook_backfill.py \
  packages/microcosm-build/tests/test_logbook_adoption.py \
  packages/microcosm-build/tests/test_logbook_archive.py \
  packages/microcosm-build/tests/test_logbook_prediction_seed.py \
  packages/microcosm-build/tests/test_logbook_chain_scopes_pg.py -q
```
(`test_logbook*.py` expanded explicitly; `test_logbook_cli.py` is the only
other test file that imports `tools/logbook.py`, confirmed by grep.)

```
........................................................................ [ 46%]
........................................................................ [ 92%]
............                                                             [100%]
96 passed
```

Three `DeprecationWarning`s fired during the run, confirming the renamed
legacy fallback path and the new message text work end-to-end:
```
DeprecationWarning: POPULACE_LEDGER_URL is the pre-rename name for LOGBOOK_URL;
the build ledger is now Logbook (microcosm#632). Set LOGBOOK_URL instead —
POPULACE_LEDGER_URL stays honored only for the dual-read window.
```
(and the same for `POPULACE_LEDGER_KEY`/`LOGBOOK_KEY`,
`POPULACE_LEDGER_API_KEY`/`LOGBOOK_API_KEY`,
`POPULACE_LEDGER_EXPORT_KEY`/`LOGBOOK_EXPORT_KEY`.)

### `uv run ruff check .`
```
All checks passed!
```
(Two `I001` unsorted-import errors were introduced by the rename in
`build/__init__.py` and `tools/logbook.py`; fixed with `ruff check --fix`
and `ruff format`, scoped to only the files this task touched, committed
separately.)

### `uv run ruff format --check .`
116 files "would be reformatted" — all pre-existing and unrelated to this
task (confirmed: the count was 118 before the ruff --fix/format pass above
and dropped to exactly 116 after fixing the two files this task's edits
affected; the remaining 116 are files this task never touched, e.g.
`test_uk_firm_generation.py`, `tools/spec_engine_coverage.py`).

### `tools/ci_test_groups.py --verify` (the lint lane's own check)
```
packages/microcosm-build/tests/test_logbook_env.py -> shared-spec
...
verification=ok
```

### Full `microcosm-build` package suite
`uv run pytest packages/microcosm-build -q` was launched as supplementary
verification beyond the assignment's required command list. It was still
running in the background when this report was written (long-running —
includes engine-gated US/UK tests now that `--extra us` is synced). The
required, scoped verification above is fully green; this run is extra
thoroughness, not a gate for this deliverable.

### Grep sweep (exact command from the assignment)
```
grep -rn "CHRONICLE_URL\|CHRONICLE_KEY\|CHRONICLE_API_KEY\|CHRONICLE_EXPORT_KEY\|chronicle_env" \
  --include=*.py --include=*.md .
```
Result — 5 lines, all in `PROGRESS-chronicle-dual-accept.md`, all intentional:

```
PROGRESS-chronicle-dual-accept.md:39:- `microcosm/build/chronicle_env.py` — the env dual-read window, one helper,
PROGRESS-chronicle-dual-accept.md:51:- Tests: `test_chronicle_epoch.py`, `test_chronicle_env.py`, plus mixed-epoch
PROGRESS-chronicle-dual-accept.md:115:as-is. The env half only: `chronicle_env.py` renamed to `logbook_env.py`,
PROGRESS-chronicle-dual-accept.md:116:`CHRONICLE_*_ENV`/`chronicle_env`/`chronicle_env_names`/
PROGRESS-chronicle-dual-accept.md:117:`describe_chronicle_env`/`reset_chronicle_env_deprecation_warnings` renamed
```

Lines 39/51 are in the journal's original "Done" section, which the
assignment explicitly says not to rewrite ("append a dated correction note;
do not rewrite history") — they're the historical record of what this
branch originally built, before the correction. Lines 115-117 are inside
the new "Correction (2026-09-02)" section this task appended, which by
necessity names the old symbols it renamed away from. **Zero occurrences in
any `.py` file** and **zero occurrences outside this one journal file's
history/correction-note text** — i.e., nothing outside git history in any
file that matters to running code, docs a reader would act on, or the PR
body.

## What was NOT touched

- The epoch half (`chronicle_epoch.py`, `ledger_artifact.py`,
  `import_entry_facts.py`, `ledger_targets.py`, CD-vintage tests, band/commit
  aliases) — approved as-is by the review, left alone except for the two
  module-path comment fixes noted above.
- `CHRONICLE_*_BANDS` / `CHRONICLE_US_SOURCE_COVERAGE_CONTRACT_COMMIT` — real
  Chronicle aliases, correctly still `CHRONICLE_*`.
- No merge was performed. PR #849 remains open.

## out.md history

Root `out.md` is this repo's established ephemeral per-task report file
(`git log --all -- out.md` shows dozens of prior merged PRs overwriting it
with their own final report, same as this one). Its prior content — an
Armenia country-package report from PR #814, merged 2026-08-28 — remains
fully recoverable at `git show 305a13ed:out.md` and was not otherwise lost;
overwriting it here follows the same convention every prior task used.

## Backup branch

`backup/chronicle-dual-accept-pre-correction-1be2f18e` preserves the two
local-only commits that were on `chronicle-dual-accept` before this task
reset it to the PR head, in case that follow-on work (exact whole-feed
counts in the journal; per-row ids in the provenance docstring) is still
wanted — it was not part of PR #849 and was not re-applied.
