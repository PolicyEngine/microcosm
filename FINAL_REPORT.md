# Final report: retirement model and data audit

## Outcome

Completed the adjudicated retirement blocker audit for all 16 failed criteria
across 11 physical legs. The result is **6 concept mismatches, 5 dense-rung
refits, and 0 derivation defects**. Every current ASEC source equation exactly
reproduces frozen clone 0, so there was no safe 1%-verifiable production
derivation fix to implement. No exclusion, pool build, gate/band/ceiling/fold/
seed change, push, publication, or chain operation occurred.

The detailed human audit is
[`experiments/retirement_model_and_data/AUDIT.md`](experiments/retirement_model_and_data/AUDIT.md),
the canonical machine proof is
[`f001_audit.json`](experiments/retirement_model_and_data/f001_audit.json), and
the blocked serial-host charter is
[`HOST_25PCT_PLAN.md`](experiments/retirement_model_and_data/HOST_25PCT_PLAN.md).

## Evidence delivered

`audit_frozen_artifacts.py` authenticates the raw ASEC, assembled/transferred,
baseline/pkg3, adjudication, and 11 target-bank artifacts before decoding. It
then proves:

- all 16 retirement source columns match across a unique, fully resolved
  4,311-row raw ASEC → assembled clone-0 → transferred clone-0 join;
- all 11 source equations bit-match the terminal ASEC clone-0 values;
- the actual early clone-0 and late clone-1 donor roles, carrier counts by
  sign, amounts, and QRF regimes are recomputed from frozen support;
- all selected baseline/pkg3 target arrays and gate records are identical;
- every ASEC-versus-ACS disagreement entry is exposed, including native
  combined `RETP`/`SSP` versus unobserved modeled leaves.

The canonical JSON SHA-256 is
`37e92f7358119c44670c104335d9452a8a4e9e22f28627a70c589691e4dc92bf`.
The source and transfer mechanisms are code-cited leg by leg in `AUDIT.md` and
inside the machine sidecar.

## Classification and action

| Classification | Legs | Required next action |
|---|---|---|
| Concept mismatch | taxable and tax-exempt private pension | Owner adjudicates private/public/annuity/railroad/mixed roles and the exact declared-absence equation before any model or derivation change. |
| Concept mismatch | Social Security retirement, disability, dependents, survivors | Owner adjudicates codes 7/8 and multi-reason rows using the exact declared-absence equation before any component refit. |
| Dense-rung refit | taxable IRA | Through the required typed post-compatibility overlay executor, add a single-target two-part overlay from exact clone-0 code-4 labels. Keep the current early bank and every other early output byte-identical. |
| Dense-rung refit | Keogh, taxable 401(k), taxable 403(b), taxable SEP | Through the same executor, add a four-output overlay from exact clone-0 labels: standalone Keogh, then 401(k)/403(b)/SEP with refit Keogh and the unchanged compatibility-pass tax-exempt IRA raw draw as fixed prefixes. Require that passing target's bank and terminal output remain byte-identical. |
| Derivation defect | none | No production patch. |

The pension owner equation marks 56 f001 rows/$1,804,558 ambiguous. The Social
Security owner equation marks 35 positive rows/$744,734 ambiguous. These are
declared-absence proposals for adjudication, not exclusions, and were not
implemented.

## Host handoff

The f025 charter pins the five-leg refit, removal commit
`1a8ad451c6eff17d405ef75cbdd014de72447153`, broad commit
`539e415defb27bf103a40081239f123ce9d76c6d`, all input and authority hashes,
the literal Darwin `lockf` entry, the no-extra-argument build argv, atomic
all-exit status requirements, refit/factorial 16-row ledgers, and exact Phase-P
runner recovery and checks.

Execution is deliberately blocked. Historical f025/f001 paths reached about
81/85 GB RSS, while this lane is bound below 15 GiB. Neither a reviewed
sub-14-GiB cold implementation nor an exact authenticated candidate resume
exists; `REFIT_SHA` and the six concept-owner decisions also do not exist. The
charter therefore records exact commands and prerequisites without falsely
authorizing a cold run that the memory guard would terminate.

## Verification

The corrected machine proof and an independent `--check` both returned zero.
Their guarded aggregate/individual RSS peaks were
`457,834,496/436,682,752` and `453,787,648/432,586,752` bytes; no descendant
escaped and both process groups were empty. The proof uses the battery's pinned
five-carrier QED minimum and independently reproduces the f001 Social Security
dependents/survivors distances `0.917093391855645` and
`0.2780886784330607`
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3026-3030,11912-11930`).

The whole frame, data, calibrate, and fit package suites returned zero. Peak
aggregate RSS was 6,995,214,336; 11,917,328,384; 498,696,192; and 863,649,792
bytes, respectively. The 259 tracked build test files ran exactly once across
six disjoint direct-pytest ranges; every accepted range returned zero, with a
maximum aggregate peak of 10,281,369,600 bytes. Accepted-command count, unique
count, and tracked file count are all 259 with an empty set difference. Every
final process group was empty.

The build suite intentionally exercises detached sessions. The initial
host-default range-1 supervisor therefore stopped at the expected escape; that
discarded receipt is not pytest evidence. The unchanged range passed with the
guard's explicit test-only escape tracking. Every disclosed escaped PID was
included in RSS and shutdown accounting and was independently verified gone.
No accepted command reached the 14 GiB fail-closed stop.

Final Ruff and staged whitespace checks pass. No pool build or network action
was used for this verification.
