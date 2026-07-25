# QBI v2 content progress

## State

- Branch: `qbi-v2-content`
- Base: local `repeal-validation-298` at `e45f797`
- Worktree: `.claude/worktrees/populace-wt-530`
- Mode: offline; no push, pull, fetch, PR, or stash
- Status: live SSTB crosswalk complete; assumptions content in progress

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
- Added and declared `us/sstb_crosswalk_v1.json`:
  - 27 2017 Census industry entries and 101 2018 Census occupation entries;
  - 10 industry and 11 occupation explicit non-SSTB documentation rows at
    probability zero;
  - exact 1.00/0.10/0.20/0.25/0.30 adjudicated tiers;
  - provisional §1.199A-5(c)(1) bases on every ambiguous entry;
  - preserved reputation-or-skill and occupation/industry wiring notes.
- Updated the strict loader and host classifier to accept the live schema,
  preserve leading-zero codes, consume code-level probabilities, prefer a
  configured industry signal, and resolve every unmapped code to zero.
- Pointed v2 assumptions at the live occupation-first resource while retaining
  the in-tree placeholder for explicit fail-closed tests.
- Added live-schema, probability-tier, occupation-format, explicit-zero, and
  real-crosswalk host-routing tests. Focused QBI, crosswalk, spec-only, and
  Ruff checks pass.

## Next

- Encode and test the two-band passive prior arithmetic directly from the
  221.00/162.76/183.46 OTA waterfall and JCT above-threshold split.
- Update qualification derivations, QRF exclusions, REIT/PTP scale and
  evidence metadata, and the restricted replay diagnostic.
- Add the changelog, run all focused validation, and commit each coherent step.
- Run focused tests, Ruff, full workspace pytest with both restricted-data
  environment variables, then write the final report.
