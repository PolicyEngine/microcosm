# Native input profile inventory (2026-09-23)

A machine-readable classification of every input main's release requires, as
the graph-native US line stands at integration commit `47960af43` and its
2026-09-21 descendants. Epic PolicyEngine/microcosm#956.

This is a source-code classification. It does not count missing cells in any
actual build, verify source signal, or qualify a value scientifically.

## Files

| File | What it is |
| --- | --- |
| `inventory.json` | The result: 174 rows, one per name in main's `release_input_coverage_manifest.json` at `4305a7d34`. Each row has a class, a state, resolved `path:line@commit` citations and a note. |
| `classification.py` | The reviewed judgements. Citations are `(ref, path, token)` triples; no line number is typed by hand. |
| `build_inventory.py` | Resolves every citation with `git grep -F` at its pinned commit (and refuses if a token is missing), checks that all 174 names are classified exactly once, and writes `inventory.json`. |
| `collect_evidence.py` | Mechanical evidence: quoted-literal hits in native-added modules at `47960af43`, new hits on each 09-21 descendant, and main modules naming each input. Writes `evidence.json`. |
| `audit_fef2cb56a_native159.json` | The historical complete159 audit rows (root `fef2cb56a`, recovered from `native159-source-coverage-20260920/COVERAGE.json`, sha256 `ad28ac91…`). |
| `source_header_observations.json` | Header lines only of the pinned ASEC 2023–2025 person members and 2024 ACS PUMS members. It is used to test whether main's H5-based exclusions carry over to the native line. |

Regenerate from the repository root:

```bash
python3 experiments/native-input-profile-20260923/collect_evidence.py
python3 experiments/native-input-profile-20260923/build_inventory.py
```

## Classes

- `native_successor_exists`: a native graph owner produces or attaches the name. `state` says whether the current development request enables it, whether it is opt-in, partial, or on a descendant branch only.
- `port_needed`: there is no native canonical owner. `port_from` names main's maintained module. `native_source` lists any native source owner that already reads the needed literals.
- `source_absent`: no locked source has the observation, with evidence cited.
- `declared_scope_exclusion`: the native profile's declared scope excludes the name (engine omissions or the prior-year family). This is not missing data.

## Result

| Class | Names |
| --- | ---: |
| native successor exists | 84 |
| port needed | 83 |
| source absent | 3 |
| declared scope exclusion | 4 |

Of the 84 native successors, 28 are attached by the current development request. Another 40 are PUF clone-one outputs whose original-arm placement is only partial, 7 are ACS health flags with semantic gaps, 8 are opt-in owners the request leaves off (sex, race and Hispanic origin, immigration, workers' compensation, child support, veterans), and 1 (`disability_benefits`) exists only on a 09-21 descendant.

The historical audit found 76 names with no native successor. At `47960af43` plus the descendants, 6 of those now have one: `cps_race`, `is_hispanic`, `workers_compensation`, `child_support_received`, `veterans_benefits` and `disability_benefits`. All 6 are opt-in. The other 70 still need a port.

Findings that change earlier framing:

- **#959 needs no port.** Main derives `is_spm_independent_minor_role` through `spm_role_source.independent_minor_role`. That function body is unchanged between `47960af43` and main; the diff is zip-archive I/O only. The native SPM owner calls the same function on its own pinned member read.
- **#720 does not apply.** PR #991 fixes main's pooled H5 files, which lack 11 Census person columns. The native line reads those columns (`NOW_*`, `A_EXPRRP`, `PTOTVAL`, `A_ENRLW`, `A_FTPT`) straight from the pinned Census members.
- **Immigration clone attachment is integrated.** It is `d33d71bee`/`a4ed33447`/`15b7befff`, which are patch-equivalent to the 09-20 successor branches. It is an opt-in host option, and the current request passes `immigration_transfer=None`. The fragment still records `graph_fit_artifact_qualified=False` and `national_stock_alignment_qualified=False`.
- **Four of main's seven reviewed exclusions are not source-absent for the native line.** `survivor_benefits`, `financial_assistance`, `employer_sponsored_insurance_premiums` and `is_unmarried_partner_of_household_head` are excluded on main only because the pooled 2022–2024 H5 files lack `SRVS_VAL`, `FIN_VAL`, `NOW_OWNGRP`/`NOW_HIPAID`/`NOW_GRPFTYP` and `PERRP`. The pinned Census members that the native line reads do carry them. The native retirement-detail owner already qualifies `SRVS_VAL`. The other three exclusions (nutritional risk, DC PTC, Early Head Start) rest on the concept being absent from every source, so they stay source-absent.
- **Social Security.** The survey line has only a report-grain basis and a report-completion fragment that is not wired into the host. The ASEC-only five-node slice (`cps_carried_current`) already reuses main's reason-priority and age-62 rule. This branch adds `survey_social_security_canonical.py`, a canonical contract with explicit conventions (see `docs/us-social-security-canonical-contract.md`).
