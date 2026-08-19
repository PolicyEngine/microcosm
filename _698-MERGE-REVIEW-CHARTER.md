# PR microcosm#698 — final merge review (sol adjudication half)

You are the sol half of a two-model merge gate ("Claude+sol agree →
merge", Max 2026-08-19: "k if you and sol agree, go for it"). Your
verdict decides whether spec-engine-schema (head b027dcb0) merges to
main TODAY. Be adversarial; a false YES ships defects to every
downstream lane, a false NO costs another merge-ref re-pin cycle
(main is hot — two UK-lane merges landed during this PR's CI alone).

Repo: /Users/maxghenis/PolicyEngine/_worktrees/microcosm-spec-engine
(branch spec-engine-schema @ 4d6b67f9 — two more mechanical main-folds
landed since this charter was first written; same pattern as the ones
you are reviewing). READ-ONLY: do not edit, do not run builds; `uv run
pytest <paths>` spot checks allowed (under 10 min total; CI is green
four ways on the current head).

## SCOPE CAP (added after runtime interruptions)

Host restarts have killed three long-form attempts; your context does
not survive them. Fit the review in ONE sitting (~45 min of tool
time). Concretely: do NOT read the full diff vs origin/main (the v3
design was already adversarially reviewed by you in round 2 and
approved). Review ONLY `git log --oneline 9becb709..HEAD` — the
post-approval operational commits — reading each commit's diff
directly (`git show <sha>`), and answer the six questions from those
plus targeted file reads. Verdict on that basis is sufficient; say so
in EVIDENCE.

## Context you must load first

- docs/spec-engine.md — the APPROVED v3 RFC this PR implements (F0).
- _698-SOL-REVIEW-R2.md — YOUR prior round-2 findings; v3 folded them.
  Your job now is NOT to re-review the design (Max approved it) but to
  verify the post-approval implementation and operational commits.
- git log 9becb709..HEAD — the post-approval commits: wheels-gate
  compatibility (single-source schemas, engine/tables importorskip
  guards, context-stable digests), lock-constrained wheels venv,
  per-shard CI test step, three merges of origin/main with conflict
  resolutions (uk/country_package.json typed rows ×2), UK stage-manifest
  regenerations, sources.schema.json closed-world variant additions
  (10 UK + 7 stochastic + seed field + public_aggregated_counts),
  identity re-pins (be/uk/us spec shas, seed protocol+map digests,
  loader golden), the spec_envelope_digests diagnostic tool, and the
  agent-guide update.

## Review questions (answer each explicitly)

1. **Conflict resolutions faithful?** For each merge of origin/main,
   did the resolution preserve BOTH sides' intent (main's new UK
   resources/stages present as typed rows; no main content dropped)?
   Diff uk/country_package.json and uk/spec/sources.yaml against
   origin/main's flat versions field-by-field.
2. **Schema variants honest?** The new sources.schema.json variants:
   closed-world (additionalProperties:false, typed maps only), fields
   match the actual UK stage JSON, no accidental widening of US/BE
   vocabularies.
3. **Identity pins defensible?** Every re-pin traces to an attested
   cause (main changing kernel-inventory modules / new UK stages), not
   to masking a real regression. Spot-check: recompute one country sha
   and one seed digest; confirm they match the pins.
4. **Test guards sound?** The importorskip guards (engine, tables)
   skip only when the dependency is genuinely absent — no guard
   swallows a real failure in CI's engine-present test job.
5. **CI changes safe?** Per-shard pytest loop (coverage-equivalent to
   testpaths=["packages"]?), lock-constrained wheels install, the
   diagnostic step — any way these hide failures or weaken the gate?
6. **Anything in the full diff vs origin/main** (~final check) that is
   NOT covered by the approved RFC or the operational fixes above —
   i.e., scope that snuck in.

## Output (final message, exactly this shape)

VERDICT: MERGE-YES or MERGE-NO
BLOCKERS: (numbered, with file:line, only if NO)
NON-BLOCKING FINDINGS: (numbered, file:line — filed, not gating)
EVIDENCE: one line per review question with what you checked.
