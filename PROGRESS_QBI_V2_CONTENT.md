# QBI v2 content progress

## State

- Branch: `qbi-v2-content`
- Base: local `repeal-validation-298` at `e45f797`
- Worktree: `.claude/worktrees/populace-wt-530`
- Mode: offline; no push, pull, fetch, PR, or stash
- Status: evidence intake complete; implementation not started

## Done

- Verified the dedicated checkout was clean, then created the requested branch
  from the local base HEAD.
- Read `AGENTS.md` and `CLAUDE.md`, including the spec-only package and
  historical root-journal contracts.
- Read the complete adjudicated evidence base under
  `/Users/maxghenis/ops/populace-qbi-port/research/`:
  - `FACTSHEET-sstb-crosswalk-draft.jsonl`
  - `FACTSHEET-199a-publications.jsonl`
  - `FACTSHEET-scf-business-vars.jsonl`
  - `FACTSHEET-soi-industry-tables.jsonl`
- Recorded the binding content decisions:
  - one live versioned resource containing both 2017-industry and
    2018-occupation maps;
  - occupation-primary wiring with no industry host column;
  - deterministic clear/absent probabilities and four adjudicated ambiguous
    tiers;
  - OTA/JCT-derived passive AGI priors;
  - Pub 4801/JCT-anchored REIT/PTP scale diagnostic;
  - evidence-specific qualification derivations and residual priors.

## Next

- Trace the v2 loader, post-QRF host transform, package resource contract, and
  placeholder fail-closed tests.
- Derive and document the passive-prior arithmetic directly from the
  221.00/162.76/183.46 OTA waterfall and JCT above-threshold split.
- Implement the live resource, assumptions changes, tests, changelog, and
  replay diagnostic in coherent committed steps.
- Run focused tests, Ruff, full workspace pytest with both restricted-data
  environment variables, then write the final report.
