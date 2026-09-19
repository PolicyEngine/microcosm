# US #958 full-vector AGI own-tail implementation

## State

2026-09-19: stopped before implementation on `us-958-full-vector-agi-tail`
under the task's explicit stop-on-unanticipated-design-conflict instruction.
This is a local implementation journal, not data certification evidence.
Final report output: `docs/implementation/us-958/FINAL_REPORT.md`.

## Done

- Read the supplied issue, prototype mapping, repository agent guide, design
  charter, shared-constants guidance, and US fact-to-target contract.
- Confirmed local-only execution: installed `.venv` tools, synthetic tests,
  no network, no base build/calibration/release, no publication.
- Read all 2,408 lines of the tail module and all 1,092 lines of its tests,
  plus all supplied prototype scripts; completed a parallel contract inventory.
- Found a reviewed final-owner conflict: clone-2 tuition and retirement fields
  must inherit/mirror clone-1 values, while the requested AGI full vector would
  make the PUF donor their owner. The retirement finalizer actually overwrites
  the two contribution fields; terminal preservation enforces clone-1 equality.
- Left production code, tests, generated specs and pins unchanged.
- Started focused existing contract tests and a synthetic overwrite probe.
- Repository lint passed (exit 0).

## Next

- Finish the contract checks and write the final report to the declared output.
- Implementation requires a resolution of final ownership: whether AGI-arm
  donor values supersede the clone-1 inheritance/mirroring doctrine for all PUF
  outputs. That requires arm-specific ownership, callback and terminal contracts.
  Do not bypass the existing checks or silently restore values after an owner
  write. Keep capital-gains-only ownership unchanged if this is authorized.

## Verification

Focused tests/probe are in progress; their exact commands and results will be
recorded in the final report. No base build was run. Real-data behavior remains
unverified.
