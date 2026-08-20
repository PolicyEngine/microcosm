# Final report: US reviewed-exclusion machinery

## Outcome

Completed the machinery-only port on `battery-excl-machinery`, based on
`origin/main` at `164027e2`. The US stacked by-origin battery now has the UK
schema-2 reviewed-exclusion pattern with exact `leg#criterion` scope, strict
five-field receipts, authenticated overrides, lifecycle/anti-rot reporting,
and independent terminal-manifest validation.

The committed register is intentionally empty. No exclusion was approved, and
no band, threshold, seed, fold, comparison algorithm, or comparator failure
message changed.

## What shipped

- A declared package resource at
  `packages/microcosm-build/src/microcosm/build/us/by_origin_reviewed_exclusions.json`
  with schema version 2 and an empty `exclusions` object.
- An immutable US policy model and strict loader in
  `packages/microcosm-build/src/microcosm/build/us_runtime/reviewed_exclusions.py`.
  It accepts only canonical
  `entity/family/target[clone_N][/sign]#criterion` keys, rejects incompatible
  signed and joint criteria, requires exactly `reason`, `approved_by`,
  `adjudication`, `approved_on`, and `expires_on`, validates canonical dates
  and `expires_on > approved_on`, rejects duplicate/non-finite JSON, and seals
  a canonical content digest.
- A path-plus-raw-SHA override arming unit. Supplying only a path or only a
  digest fails before stacked build work; override flags are also refused on
  the legacy route.
- Exact criterion indexing in the existing battery paths: `#dead`,
  `#incidence`, `/positive|negative#incidence`,
  `/positive|negative#quantile`, and `#categorical`. Only an exact, failing,
  in-force criterion can be suppressed.
- Full `in_force`, `applied`, `stale`, `dormant`, `expired`, and `premature`
  reporting. Stale, expired, and premature receipts fail the gate; dormant
  receipts remain visible; unevaluated or structurally invalid comparisons are
  never silently waived.
- A terminal validator that independently reconstructs eligible failures from
  sealed comparison evidence, revalidates the policy source/digest and all
  five receipt fields, recomputes lifecycle buckets, and checks the exact
  applied failure text. `GateReport` then copies the validated report into the
  terminal receipt and final manifest.
- Typed-policy threading through cold and resumed stacked routes. The policy
  does not alter simulation inputs or checkpoint identity; its verdict and
  complete receipt are sealed only at the terminal gate.
- Publication and checkpoint collision protection for an authenticated
  override path, without contaminating simulation pins or checkpoint identity.

Primary implementation references are
`reviewed_exclusions.py:33-380`,
`stacked_spine.py:7066-7322,7698-7742,11644-11718,11775-11883,12202-12486`,
`gates.py:142-177,204-309`, and
`tools/build_us_multispine_pool.py:532-546,709-838,1039-1194,2866-2882,3293-3304,3413-3514,3740-3751,4097-4117,4168-4173,4357-4403,4471-4493`.

## 1% before/after

The before receipt is the existing 1%/seed-578 terminal artifact:

`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r10/logbook-receipts/populace-us-2024-stacked-f001-s578-asec1688-acs15316-20260813T041921Z-23bea977/terminal-gates.json`

Its SHA-256 is
`96f68ddac5c833cb3c6a3eb6ea3ceb0cf04a5dcbbf92ef56581479ed368a6ce5`.
The after measurement was an in-process rerun of the fresh terminal battery
and manifest validator from its authenticated saved simulated checkpoint,
using the committed empty policy. Peak RSS was 1,544,634,368 bytes (about
1.44 GiB); it did not produce a durable after receipt.

| Exact assigned check | Before | After | Delta |
| --- | --- | --- | --- |
| `person/puf_tax_itemization/sstb_self_employment_income_would_be_qualified[clone_0]#dead` | `dead_comparison`; ASEC/ACS incidence 0/0; rows 0/0; failed | Same values and failure; no receipt applied | None |
| `person/puf_tax_itemization/business_is_sstb[clone_0]#dead` | `dead_comparison`; ASEC/ACS incidence 0/0; rows 0/0; failed | Same values and failure; no receipt applied | None |

The battery remains failed with 127 failures. This is the only honest result
for an empty owner-controlled register. The adjudication explicitly classifies
both rows as `FIX-CANDIDATE`, rejects them as exclusions, and assigns a separate
per-target clone-role comparator correction. This machinery lane cannot make
that artifact correction and must not turn either row green with a receipt.
The literal assigned-check numerical-improvement condition is therefore unmet
in this machinery-only lane.

The measurable machinery delta is:

- before: no reviewed-exclusion policy/report in gate details;
- after: schema 2, committed source, policy digest
  `581b42d0f8ffdfe5a29671fbc9be4efc839a440da2c52951229f85d407d5758d`,
  evaluation date `2026-08-20`, and empty resolved/applied/lifecycle buckets;
- after manifest validation: accepted, proving that neither assigned failure
  was waived and that no receipt was forged or partially armed.

The replay used
`simulated.checkpoint.h5` SHA-256
`78721fa97cecf6f6befc83fd219abbce9453542da429e80432346d4e8272eb76`
(260,260,648 bytes). It ran off-chain, supplied no predecessor digest, wrote no
build/logbook artifacts, and did not touch `logbook-pending-chain.txt`.

## Verification

- `.venv/bin/python -m pytest -q` — complete workspace suite passed, exit 0.
- Complete exclusion-loader, stacked-battery, pool-driver, country-package,
  and spine-blindness test files passed. Coverage includes every eligible
  criterion form; in-force, stale, dormant, expired, and premature states;
  exact-leg isolation; both assigned checks; receipt/application tampering;
  source mismatch; partial CLI arming; cold/resume threading; and
  checkpoint-identity noninterference.
- Adversarial enumeration confirmed all 417 unique exact criterion IDs emitted
  by the canonical runtime surface pass the parser, while malformed shorthand,
  noncanonical clone indices, invalid signed/joint criteria, and duplicate
  joint members fail closed.
- A dual-failure monetary-leg regression confirms one receipt can suppress
  only its exact incidence or quantile criterion; the other failed criterion
  remains red.
- Override collision regressions confirm neither a publication path nor a
  checkpoint-managed path can overwrite the authenticated policy source.
- `.venv/bin/ruff check .` — passed.
- Ruff formatting check for all five hardened files — passed.
- `git diff --check` — passed.
- The committed policy digest was reloaded from the declared resource, and a
  direct JSON assertion confirmed schema version 2 with `exclusions == {}`.
- The unpushed history was corrected so runtime classification travels with
  the loader commit and the 65-module graph pin travels with the battery-import
  commit. Relevant loader/country/spine, battery/spine, and driver suites pass
  at those respective commit boundaries.

The required `uv sync --all-packages --extra us` was attempted first. The
managed sandbox rejected the default uv cache, and a private-cache retry could
not resolve/download `hf-xet` because DNS is disabled. A pre-existing locked
US environment was copied and rebound to this worktree for testing; neither
the lockfile nor dependencies were changed.

An isolated wheel build was also attempted, but could not resolve/download its
`hatchling` build dependency for the same network reason. Package-resource
declaration and loading are covered by the green tests, but a wheel artifact
could not be produced in this sandbox.

A cold 1% rebuild was not run because the saved stage profile records a
primary-QRF peak of 84.73 GB (78.91 GiB), which would violate the explicit
15 GiB RSS ceiling. The bounded terminal replay used the correct saved
1%/seed-578 simulation surface and stayed well below the limit.

## Local commits

- `bf670576` — start US exclusion machinery lane journal.
- `11e382c3` — record exclusion machinery design and baseline.
- `637788ab` — add empty US schema-2 exclusion register and classify its loader.
- `ac27ed1b` — integrate reviewed exclusions into the US battery and pin its
  runtime graph.
- `e6e09e56` — wire authenticated US exclusion policy into the pool driver.
- `48e61e81` — record honest 1% exclusion replay.
- `736a2ecb` — record the runtime-inventory audit in the lane journals.
- `08c4271f` — finalize the reviewed-exclusion machinery report.
- `a4d90b50` — harden reviewed-exclusion configuration boundaries.

Nothing was pushed. Owner review and the push remain external to this lane;
the two measured QBI rows remain separate non-exclusion fix work.
