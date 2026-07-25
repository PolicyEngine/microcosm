# QBI v2 content progress

## State

- Branch: `qbi-v2-content`
- Base: local `repeal-validation-298` at `e45f797`
- Worktree: `.claude/worktrees/populace-wt-530`
- Mode: offline; no push, pull, fetch, PR, or stash
- Status: implementation complete; broader and full validation pending

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
- Filled all qualification derivations:
  - Schedule C and farm operations remain statutory derived sources;
  - farm rent/rental/estate retain documented 0.80/0.70/0.60 residual priors;
  - partnership/S-corporation income now uses the required 0.90 prior for
    guaranteed-payment and reasonable-compensation exclusions, anchored to
    JCT's 53% S-corporation and 17% partnership shares.
- Returned the partnership/S-corporation flag to the v2 QRF targets, yielding
  52 person and 61 total v2 targets.
- Derived the two passive SSTB priors from OTA WP-118:
  - `(221.00 - 162.76) / 221.00 = 58.24 / 221.00 = 0.2635`, rounded 0.264;
  - `(221.00 - 183.46) / 221.00 = 37.54 / 221.00 = 0.1699`, rounded 0.170.
- Documented JCT's `120.81 / 216.08 = 55.91%` above-threshold dollar share as
  evidence that the upper band matters, not as a conditional SSTB rate.
- Re-anchored the REIT/PTP model by changing the partnership/S-corporation
  receipt probability from 0.05 to provisional 0.09 while retaining the
  dividend exposure and Beta scales. Restricted fixed-seed replay:
  `$20.943037789B`, `0.993974x` the Pub 4801 `$21.07B` income anchor.
- Added strict provisional anchor metadata for Pub 4801's `$4.20B` component,
  JCT's `$2.90B` 2022 comparison, and the `[0.3x, 3x]` replay band.
- Retained the v1 BDC parameters with an explicit no-published-anchor finding;
  restricted replay is `$0.143270890B`.
- Focused QBI, crosswalk, QRF-chain, spec-only, restricted replay, and Ruff
  checks pass; the v1 golden replay is unchanged.

## Next

- Add the changelog and run the broader QBI/builder/manifest contract suites.
- Run full workspace pytest with both restricted-data environment variables,
  then full Ruff and repository hygiene checks.
- Write the final report, including arithmetic, replay result, and any
  unimplemented adjudication (currently none).
