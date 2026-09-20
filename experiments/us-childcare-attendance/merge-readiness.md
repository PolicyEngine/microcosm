# PR #916 merge-readiness audit

Objective: resolve the PR's engineering, integration and statistical issues to
reach merge-ready quality while retaining draft status. This audit does not
authorize publication or replacing the default population. Completion is not
yet established.

## Current audit, September 20, 2026

The starting PR head is `d2601c7a0749b134822086f4783e03501c750e32`.
Its 24 GitHub checks passed. The current integration base is
`18c6d39a`, 65 main-branch commits beyond the PR's previous merge base.
Maria's September 15 review is the latest external review; no follow-up approval
is recorded. Passing tests below are engineering evidence, not statistical
acceptance.

| Requirement | Current evidence and remaining work |
| --- | --- |
| Current main, clean merge and safe PR branch | Merge commit `8d74be68` reconciles README and serializer-test conflicts, preserving both childcare and annual projection changes. The feature branch tracks and pushes to `fork/fix/us-childcare-attendance`; verify the final remote state after pushing. |
| C1: exact attendance values and source recipe bound together | Existing binding covers source pins, code/runtime, settings, identities, ages, household membership and values. Native ingress, reuse and export regressions exist; final affected-suite validation is still required. |
| A1: row-complete final attendance | Attendance-specific validation and binding checks run at release coverage and final write. Verify failures cannot be waived or lost through later transformations. |
| A2: valid source household identities | Missing, blank, stringified-null and unresolved household identities are rejected; regression coverage exists. |
| A3: household intensity and larger-sibship validation | Joint days/hours and 3+ child diagnostics exist. Their reported screen failures remain substantive; adding diagnostics alone does not qualify the model. |
| S1: noncalendar transport and sensitivity | Observed regular hours remain fixed; paired sensitivity holds donor identities fixed. State flags and missing-calendar selection remain unresolved statistical evidence. |
| S2: explicit operation provenance | Receipts identify calendar derivation, dependence fitting, noncalendar bridge, target harmonization, joint transfer and outside-domain baseline policy. |
| Fiscal builder and exact-k integration | Repair metadata preservation and exact-k source configuration have targeted tests. A complete fiscal build using the actual target bundle and survey inputs remains required. |
| Annual projection compatibility | The regression reproduced the original receipt loss. All 42 annual/serializer tests now pass: the 2024 base and 2025/2026 projections retain their original receipt, changed attendance is refused before finalizing an output, and the combined serializer inventory has ten writers. |
| Source mapping and population integrity | Licensed DS4/DS5 files, pinned ASEC cache and BuildP parent are available locally. Current candidate evidence preserves original population values/weights. Recheck any affected evidence after implementation or runtime changes. |
| Statistical qualification | Existing experiments fail provisional subgroup/household screens. Do not redefine those failures as acceptance, relax thresholds after seeing results, or call inspected partitions independent validation. A defensible model/assumption treatment and stronger validation remain required. |
| Final tests, packaging, evidence and handoff | Run relevant engine and wheel checks; inspect their actual coverage. Keep source microdata and per-person hashes out of commits. Preserve the PR's feature-branch push target, draft status and VS Code PR file-tree view. |

## September 20 engineering checks

- Initial merged attendance, launcher and serializer run: 139 passed, one failed.
  The failure was the serializer count: the combined branches contain ten
  serializers, while each independently expected nine. The expectation was
  corrected; the subsequent 42-test annual/serializer run passes.
- Specification coverage remains current: 42,159/42,159 fields and 41/41 inventory
  checks. No checksum regeneration is needed for this merge.
- Repository lint and the tracked CI test inventory passed after conflict
  resolution.
- The new annual receipt regression failed on the original annual writer at
  native reload, proving the uncovered integration defect. The final annual and
  serializer run passes all 42 tests with no skips. Repository-wide lint and
  changed-file formatting also pass after the fix.
- VS Code recognizes the worktree as PR #916 and shows the GitHub Pull Request
  sidebar with its clickable file tree. Its Sync target is the fork's feature
  branch, not main.

## Full-build preparation and constraints

The pinned DS4/DS5 inputs, ASEC source cache and original BuildP parent are
available locally. A 2023 Chronicle bundle build is in progress using Chronicle
commit `78466057401af48f9a53da41b87295241e4af1ba`; no completed target bundle or
successful fiscal build is claimed yet.

This host has 16 GiB RAM. The repository's measured system requirements put the
national build floor at 32 GiB, with approximately 15 GiB needed for SCF
imputation alone. A full national build needs an appropriately sized build host;
the small-fixture annual tests do not establish national-build feasibility.

The full fiscal run is not replaced by the standalone attendance-preparation
command or by mocked builder tests. Publication, survey transport validity and
maintainer approval are distinct from serializer and code-contract checks.
