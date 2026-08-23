# Candidate 25% legacy-arm audit, round 3

Date: 2026-08-23

Code authority: `d69131a3534a` (`origin/main` at round-3 start). The release
builder and its package sources in this branch are byte-identical to that
reference; branch-only differences before this receipt are journals.

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Status: **target-surface gate passed; stage-2 host-input audit in progress**.
No pool or release builder has run.

## 1. Unified target-surface verification

Verdict: **GO on target-surface identity.** The owner-ruled legacy dense arm
does not compile a different or smaller fiscal target surface than exact-k.

PR #741 is merge `c31e1525f099` ("Compile one target surface unconditionally;
remove membership flags"). Current tests require the parser to reject all three
former per-run membership controls:
`--include-congressional-district-targets`,
`--diagnostic-skip-tax-expenditure-targets`, and
`--zero-support-exclusions`
(`packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py:52-93`).
The current parser unconditionally resolves the packaged congressional-district
crosswalk for every run
(`tools/build_us_fiscal_refresh_release.py:1472-1478`).

Exact-k and legacy initially differ in base identity handling: exact-k
authenticates a pool manifest/H5, whereas legacy takes a bare `--base-h5` and
hashes it (`tools/build_us_fiscal_refresh_release.py:8213-8258`). Both paths
then reach the one unconditional Ledger loader and the same single call to
`compile_us_fiscal_target_registry`
(`tools/build_us_fiscal_refresh_release.py:8358-8371`). The same Medicaid
substitutions and target-parity gate follow before either calibration arm
branches (`tools/build_us_fiscal_refresh_release.py:8372-8402`).

The compiler signature has facts, period, crosswalk, aging, and period-waiver
inputs only; it has no exact-k, dense, geography-membership, JCT-skip, or
per-run exclusion argument
(`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:933-940`).
It builds the shared dynamic-fact and JCT-reference registry and applies the
same transforms, aging, and period contract
(`packages/microcosm-build/src/microcosm/build/us_runtime/fiscal_targets.py:962-1017`).
The checked-in parity authority states the one-surface rule directly: national,
state, and congressional-district facts compile for every artifact, while
record count changes only selection, never target membership
(`packages/microcosm-build/src/microcosm/build/us/target_parity_manifest.json:524-526`).

After the common source stages, both arms pass the same `target_specs` through
one `_load_or_materialize_target_frame` call
(`tools/build_us_fiscal_refresh_release.py:10070-10108`). Only then does exact-k
pass `registry.to_target_set()` to `calibrate_exact_k_ladder`
(`tools/build_us_fiscal_refresh_release.py:10207-10246`) while legacy dense
passes that same target set to `calibrate`
(`tools/build_us_fiscal_refresh_release.py:10300-10316`). A bare Ledger JSONL
and an artifact directory therefore compile the same facts bytes identically;
the envelope changes provenance and exact-k eligibility, not compiler
membership
(`packages/microcosm-build/src/microcosm/build/ledger_artifact.py:40-56,88-99,132-165`).

`--dense-default-dataset` itself:

- relaxes the L0-lambda validation and forbids a refit-L2 override
  (`tools/build_us_fiscal_refresh_release.py:1479-1490`);
- is mutually exclusive with exact-k
  (`tools/build_us_fiscal_refresh_release.py:1568-1571`);
- selects full-pool calibration and records `dense_no_l0` identity metadata
  (`tools/build_us_fiscal_refresh_release.py:10132-10140,10300-10325`); and
- activates the dense-arm SSI delivery enforcement fences after calibration
  (`tools/build_us_fiscal_refresh_release.py:10448-10462`).

Thus the stronger claim that the flag changes *only*
selection/identity/publication handling needs one explicit qualification: it
also changes post-calibration SSI release-gate enforcement. That does not
change fiscal target membership, so it does not trigger the owner's
different-surface stop rule, but it must remain in the release-arm label and
evidence.

Two controls must stay absent. `--selection-mass-protection` can append
synthetic run-scoped calibration targets after the parity gate
(`tools/build_us_fiscal_refresh_release.py:921-932,10032-10045`), and is not in
the owner ruling. `--gate-congressional-district-targets` changes only whether
unmaterializable CD rows are hard failures, not which materializable specs
enter the registry (`tools/build_us_fiscal_refresh_release.py:4323-4347,
6451-6474`). Materializability still depends on the actual frame, so a finished
legacy run must record and compare its `target_surface` count and SHA-256; the
code path itself has no legacy-only narrowing.
