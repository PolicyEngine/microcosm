# Team-PR merge reviews: microcosm #702, #707, #576 (sol half of the merge gate)

You are the sol half of the two-model merge gate for three team PRs.
Claude's half is done (approvals with reasoning below). For each PR,
return MERGE-YES or MERGE-NO with blockers. These are colleagues' PRs:
review the CODE as merged-state (origin/<branch> vs origin/main), do
not edit anything, do not push. READ-ONLY plus targeted `uv run pytest
<paths>` spot checks (each PR's CI is already green four ways).

Repo checkout: /Users/maxghenis/PolicyEngine/_worktrees/microcosm-spec-engine
(fetch the PR branches; you are on spec-engine-schema — do NOT check
branches out here; review via `git diff origin/main...origin/<branch>`
and `git show`).

## PR #702 — Scope Logbook chains by country and dataset line (juaristi22)

Branch: logbook-chain-scopes (verify via `gh pr view 702`). Claude's
approval reasoning: scope DERIVED from pipeline (never stored) so every
existing row_digest stays valid and rows cannot move chains without
breaking their own digest; three legacy US pipelines grandfathered to
one mixed `us` chain forever; closed-world DECLARED_SCOPES ({"us",
"uk/frs"}) — genesis fail-closed against vocabulary (a typo'd pipeline
is refused rather than opening an unremovable stray chain on an
append-only table); Python mirror of the SQL chain_scope() with
cross-references; export requires explicit operator scope declaration.
Max's prior ruling (8/17): per-country chains right, global total
order buys nothing. Focus your skepticism on: the SQL migration's
claim-tail/append path under the new scope key (can two scopes hold
pending rows concurrently without interleaving?), the Python/SQL
mirror drift risk, and the archive-splitting of the 28 legacy spool
rows (renames only — verify no content changed).

## PR #707 — Adopt the UK national calibration contract (juaristi22)

Branch: uk-national-contract. This re-homes chronicle#164 per the
#166/#172 ruling (Chronicle facts-only; selection contracts live in
microcosm). The host resolved a merge conflict at d8c89386/74d788d4:
verify the resolution kept BOTH sides (main's was_wealth_support_bounds
+ regional_land_values AND the PR's uk_national_targets +
target_references, tuple test extended). Focus on: ledger_targets.py
changes (does the target-reference feed parse fail closed?), fixture
integrity, and whether anything in the contract adoption assumes the
flat country_package format that #698 will replace (flag as
non-blocking coordination if so).

## PR #576 — Staging telemetry on unless --no-staging (anth-volk)

Branch: staging-telemetry-563. Claude's approval reasoning: root cause
(env.get default defeated by exported empty string) fixed via
_env_default treating blank as unset; staging-with-nowhere-to-write is
parser.error not silent skip; manifest tri-state (absent key = no
staging path; enabled:false = declared opt-out; present-but-no-uploads
= undelivered) and publish_cli refuses undelivered staging without
--allow-missing-staging; uploads_succeeded counter separates intent
from delivery. The host merged origin/main into the branch (a06cd5fe)
to collapse ~90 stale byte-identical files out of the diff — verify
the true delta is the 10 files and the merge brought no behavior
change. Focus on: the module-level telemetry handle for the failure
path (thread-safety/reentrancy under the builder's actual call
pattern), and whether refusing publish on undelivered staging could
block a legitimate release path not covered by --no-staging or
--allow-missing-staging.

## Output (final message)

Per PR: VERDICT: MERGE-YES/MERGE-NO, blockers (file:line) if NO,
non-blocking findings, one EVIDENCE line per focus area.
