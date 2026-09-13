# Current survey household reference-person roles

The canonical `is_household_head` leaf means the observed reference person of a surveyed household. `current_survey_household_roles` qualifies that role from the retained ACS/ASEC preparation and original source owners. `graph_current_survey_household_roles` supplies a private source projection and a binding node on the existing population after cloning. This fragment is not yet wired into the country host or validated on native inputs.

## Source meanings and unknownness

| Source evidence | Qualified household role |
| --- | --- |
| ASEC `A_EXPRRP` code 1 or 2, every relationship token named, exactly one reference person in the published household | Reference person is true; other members are false |
| ASEC missing/unnamed relationship, no reference person or multiple reference persons | Whole affected household remains unbound, with the owner's reason |
| ACS housing-unit roster with exactly one `RELSHIPP` code 20 and all relationship tokens admitted | Reference person is true; other members are false |
| ACS missing/invalid/mixed housing-unit roster or incorrect reference-person count | Household role remains unbound |
| ACS group-quarters codes 37 or 38 | Published code remains known; housing-unit reference-person role remains unbound |

The ASEC interpretation reuses `asec_demographic_source` and its pinned dictionary contract: `A_EXPRRP` has a 1–14 printed range, but code 6 has no named meaning. A value inside the range is therefore insufficient. The recorded field locators identify the 2023/2024/2025 dictionary pages 21/22/23 and the all-persons universe. Its owner proves the equivalence of published and prepared household memberships before classifying a role. `P_SEQ`, roster line number, family relationship, age and tax-unit position are not substitutes.

The ACS interpretation reuses `asec_demographic_source.acs_household_reference_states`, reading original `RELSHIPP` tokens from the retained catalogue's pinned person archive. It reads whole selected household rosters so a selected subset cannot conceal another reference person or an invalid relationship. A group-quarters person is not an observed non-reference member of a housing unit; the named group-quarters code is retained separately from the unresolved canonical leaf. Synthetic household construction and the harmonized `A_EXPRRP` recode cannot supply reference-person evidence.

These are the requirements already recorded by `graph_composed_asec_binding` for its unbound `is_household_head` column. This fragment pins that requirement and refuses changes. It does not relabel the composed stage as having supplied this observation.

Neither arm supplies relationship-allocation evidence through this fragment. The current ASEC readset has no allocation flag for `A_EXPRRP`; ACS `FRELSHIPP` is outside this fragment's readset. The public receipt records allocation provenance as unresolved and makes no respondent-reported or equivalent-evidence claim.

## Binding and inspection

The private projection retains exact original person identifiers, source relationship codes, role states, reasons, household verdicts and knownness. The binding fans each original onto its two initial clones using shared provenance columns with exact int64 identity. It writes only `is_household_head`; an existing leaf is an explicit rewrite on a population version below its CREATE node. The expected-output verifier reconstructs the full population through the maintained patch operation and compares all other cells, entities, relationships, weights, geography, provenance, owners and mass history through shared replay equality.

A known incumbent must have a qualified binding and agree with it. Contradictions and unsupported known incumbents both refuse without changing the original value. This includes an artificial group-quarters head: it cannot become source-confirmed headship. `household_role_reconciliation` reports aggregate disagreement and unsupported-incumbent counts for separate adjudication. A null incumbent is filled only when the source role is qualified. Source-unbound nulls stay null.

The qualifier accepts only the actual retained preparation type and registry entry. It checks source bytes, live functions, code objects, defaults, closures and contract constants around the first and final preparation/ACS owner callbacks. After the last callback and source identity read it checks retained buffers and independently reconstructs the projection. The graph kernels similarly bind the host callback, qualified inputs and detached output seals. These are bounded in-process integrity checks, not a general execution sandbox. Public artifacts contain only the declared aggregate receipt; source identities and rowwise evidence remain private.

## Independent-minor audit

This lane does not create a generic `independent_minor` or emancipation variable. Existing `spm_role_source.py` defines the distinct `is_spm_independent_minor_role`, reconstructed from `SPM_HEAD`, `A_FAMTYP` and `A_FAMREL`, before an age gate. Its documented consumer reconciliation uses SPM adult/child/person counts; it is not household reference-person status, tax dependency, filing status or statutory emancipation.

`tools/build_us_spm_role_enrichment.py` calls `derive_spm_role_source` for the separate legacy BuildP source-enrichment contract described in [the SPM runbook](us-native-spm-role-source-enrichment.md). Those retained source/release checks do not establish authority for the new combined graph or its ACS arm. A future SPM lane needs an explicit graph owner, current complete-unit source reconstruction, unit-count reconciliation, clone mapping, and verified consumer input semantics. This source-only audit did not import a country engine or inspect installed model definitions, so it makes no new consumer-compatibility claim. No age-only, head-only or tax-unit approximation is introduced.

## Validation scope

The proposed tests use invented observations through the real classifier, clone binder, graph compiler/executor, ContentStore and cold/required replay. They cover GQ unknownness, unsupported/conflicting incumbents, exact large integer IDs, complete population preservation, public receipt boundaries and mutations at first/final callbacks. A separate file uses the maintained tiny source fixture and real preparation, ASEC demographic and ACS catalogue owners, including both group-quarters types.

At this source checkpoint the tests are written but unrun. Source AST, lint and CI inventory checks establish neither execution success nor data certification. The runtime proposal requires separate review of source/resource maps and guard limits before either invented phase. Native qualification, country-host integration, calibration and release verification remain separate gates.

The finite roster has 63 pure cases, 50 graph cases and 19 actual-owner cases. Additional controls verify non-nullable Boolean storage, invalid incumbent dtypes, and detached coordinate/roster mismatches against actual retained ASEC and ACS owners. An unsupported known Boolean incumbent refuses before the final unresolved-Boolean defense; tests do not fabricate nullable cells inside non-nullable storage to force that later branch.

The **64 MiB serialized private-artifact cap is an operative capacity limit**, independent of the 2,097,152-original-person ceiling. Per-row JSON size varies with identifiers and values, so the person ceiling is not a supported national capacity claim. The maintained ASEC demographic owner also caps its own full input at 600,000 people. The first applicable source, row or byte limit governs, and none has been validated here against a full national universe. Larger builds need a reviewed storage and resource plan rather than silently increasing limits or assuming the row ceiling can be reached.

Roster lines supply coordinates only. The current 2025 ASEC dictionary prints `A_LINENO` as 01–16, while ACS `SPORDER` uses 1–20. The fragment's common 20-coordinate ceiling therefore does not establish the suggested failure for a legitimate current ASEC line above 20; this review found no such documented value and did not widen production bounds. The exact ASEC owner and coordinate comparison remain required. [ASEC 2025 dictionary, physical page 22](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf).
