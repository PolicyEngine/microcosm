# Progress: defensive review of microcosm PR #743

## State (round 2)

- Current-head reverification is complete on the dedicated
  `review/pr-743-audit` branch; the PR branch remains untouched.
- GitHub's live PR record, the freshly written `FETCH_HEAD`, and
  `origin/uk-national-first-calibrated-candidate-623` all resolve to
  `74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`, the exact head audited in round
  1. The prompt's three named seam commits (`96e73047`, `971a9699`, and
  `e54e546b`) are ancestors of that head, not successors.
- The audit is defensive: every round-1 verdict will require a refreshed
  production-shaped probe plus current-head source citations. No build will be
  run.

## Done (round 2)

- Read the current PR conversation, including Vahid Ahmadi's four second-pass
  findings, and preserved the direct comment URL for the overlap analysis.
- Repeated the required setup attempt from a writable uv cache; dependency
  resolution succeeded but package download stopped on DNS. Verified the
  fallback sibling environment has the exact same `uv.lock` SHA-256 and placed
  this worktree's five source roots first on `PYTHONPATH`.
- Re-ran the focused seam, scorer, resolver, target-materialization, and exact
  no-resolver regression tests successfully.
- Refreshed probes for the documented builder path, production-shaped scorer,
  boolean child counts, Logbook scope/order, and release-candidate exclusion
  path. All five round-1 P0 failures reproduced at the live head.
- Added the round-2 verdict matrix to `REVIEW-743.md`: all five P0 findings are
  **SURVIVES**. Mapped Vahid Ahmadi's four second-pass findings: none supersedes
  ours; three are distinct/adjacent defects and the source-frame mutation
  concern is safe because the adapter copies its tables and links.

## Next (round 2)

- Publish this committed report only to the authorized
  `review/pr-743-audit` branch. Ordinary Git transport is currently blocked by
  host DNS; do not push any production or PR source branch.
- The PR owner should treat the five original P0 findings and Vahid's three
  confirmed distinct defects as open until fixes land on a genuinely new head,
  then rerun the same probes.

---

## Round 1 record

## State

Review complete on local branch `codex/review-743` at cached PR head
`74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`. `REVIEW-743.md` records a HOLD:
the documented held-run path is not runnable on its production target surface,
the decisive scorer cannot score exported production-shaped H5s or
authenticate the incumbent, boolean person-count measures collapse to
household indicators, release-candidate membership is operator-selectable,
and the new calibration attempt cannot satisfy the ratified Logbook contract.
The PR branch remains untouched; nothing was pushed or built.

## Done

- Read `CLAUDE.md` and the GitNexus PR-review workflow.
- Verified the worktree was clean and detached at `origin/main`
  (`5abda6145b06d55b255db87ecead2438d5d8fde7`).
- Attempted the requested `git fetch origin pull/743/head:pr-743`; recorded the
  DNS failure without mutating any existing ref.
- Verified the locally cached remote-tracking head for
  `uk-national-first-calibrated-candidate-623` was fetched at 08:59 local time
  today and points to `74b4d768f`; created local `pr-743` at that exact object,
  checked it out, and immediately created the separate review branch.
- Attempted `uv sync --all-packages --extra us` first with the default cache and
  again with `UV_CACHE_DIR=/private/tmp/microcosm-review-743-uv-cache`; the
  second attempt reached package acquisition before the host DNS restriction.
- Verified the cached PR head against its exact merge base and reviewed the
  complete 35-file, 4,881-addition/437-deletion triple-dot diff, including all
  added registers, fixtures, CLIs, runbook text, gate receipts, and provenance
  paths. The GitHub `gh pr diff 743` fallback also failed on host DNS.
- Used a complete sibling environment whose `uv.lock` SHA-256 exactly matches
  this worktree and forced all package imports through this worktree's source
  directories. A focused 36-test seam/scorer/materializer slice passed; an
  independent 49-test runtime slice and the exact no-resolver regression also
  passed. These green tests reproduce the false-green fixture shapes rather
  than covering the production failures.
- Reproduced the production-shaped scorer failure with
  `dwp/uc/households`: zero of the 397 packaged measure names survive as plain
  persisted columns, so scoring raises `No targets compiled` after calibration
  deliberately restores the pristine export tables.
- Reproduced arbitrary candidate/incumbent H5s being labelled
  `populace_uk_2023` and `enhanced_frs_2024_25` with no artifact hashes in the
  score receipt, and confirmed the target-row drift data is discarded.
- Reproduced a boolean `is_child` person measure for a three-child household
  resolving to `1.0` rather than the required `3.0`.
- Reproduced `uk-national-calibration` deriving the undeclared Logbook scope
  `uk/national`; the only ratified scopes are `uk/frs` and `us`. A conflicting
  predecessor probe also wrote staging H5, diagnostics, gates, and build record
  before refusing, with no Logbook row.
- Reproduced release-candidate parsing accepting a caller-selected
  `--measure-exclusions` file. All five committed exclusions omit approver,
  adjudication, approval date, and expiry, and the applied receipt retains only
  the reason.
- Verified the UK spec-bundle digest was correctly re-pinned and found no
  threshold/default-seed weakening, publication, or promotion side effect.
- Moved the generated local GitNexus analysis index out of the worktree to
  `/private/tmp/microcosm-review-743-gitnexus-index`.
- Wrote the ranked, code-cited defensive audit to `REVIEW-743.md`; validated all
  50 full-path citations against the checked-out PR files and ran
  `git diff --check` cleanly.
- Committed each coherent review checkpoint on the separate review branch:
  setup journal `88e72569`, evidence journal `b81b4ad0`, and review report
  `ccf2cdc7`.

## Next

- María should hold #743 until the documented run uses the resolver/exclusion
  seam; production-shaped scoring and count reductions are corrected; the
  scorer and parity receipts bind the incumbent and expose row drift; target
  membership has one owner-receipted surface; and Logbook scope, attempt IDs,
  predecessor ordering, and failure rows meet the ratified contract.
- After those changes, rerun the adversarial probes listed in `REVIEW-743.md`
  before relying on the claimed green suites.
