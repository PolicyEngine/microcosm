# The SPM independence role on the ACS spine

`is_spm_independent_minor_role` is the one dataset source input the engine
declares (`policyengine_us.spm.DATASET_SOURCE_INPUTS`). spm-calculator counts a
15-to-17-year-old as an SPM adult only when it is true, and a single SPM unit
with no classified adult refuses the whole population's SPM measurement
(`SPM_COMPOSITION_REQUIRED`). The engine's declaration "permits source delivery;
it does not permit synthesizing a default value when data are absent".

ASEC rows get the role from the Census SPM fields
([`docs/us-spm-role-stage.md`](us-spm-role-stage.md)). This page covers the
ACS rows of the local-release staging (`tools/build_us_acs_multispine_base.py`).

## What failed, and why a default was the wrong fix

Since the adapter began classifying declared source inputs as input leaves
(2026-09-18), `PolicyEngineUSEngine.variables()` lists the role. The donor
transfer never gives it to ACS rows, so a staging H5 carries it on donor rows
and leaves it null on every `acs_2024_1yr` person. The local release's
`--stage materialize` then refused every existing staging artifact in
`fill_reviewed_nulls` (`UnregisteredNullError`).

Registering the null would have been worse than the refusal. The staging
builder's `_engine_input_null_audit` registers every null engine input, so a
rebuilt staging would have listed the role, and `fill_reviewed_nulls` fills a
registered column with the raw engine default, `False`. That default is
exactly what the engine forbids. It also strips the only classified adult from
the ACS housing units whose oldest member is a 15-to-17-year-old reference
person (__HU_ROLE_ONLY__ in the 2024 PUMS), so any SPM measurement of the
population would fail. A Modal smoke run on 2026-09-30 reached this state: its
staging summary registered the role, and the engine pass filled it `False` on
ACS rows. It was a smoke run, not a release.

## The rule

The ACS has no SPM fields, but it has no SPM partition either: PUMS carries no
`SPM_ID`, so `assign_us_unit_structure` makes each household one SPM unit. On
that partition the ASEC rule
`SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})` reads off
`RELSHIPP`:

| ASEC branch | On the ACS household partition | `RELSHIPP` |
| --- | --- | --- |
| `SPM_HEAD == 1` | the unit's head: the reference person, or a group-quarters record's only person | 20; 37, 38 |
| primary family, `A_FAMREL` 1 | the reference person of a family household | 20 |
| primary family, `A_FAMREL` 2 | the reference person's spouse | 21, 23 |
| unrelated subfamily (`A_FAMTYP` 4) | not identified by ACS PUMS | none |

An unmarried partner of the reference person (22, 24) is not a family member
(`A_FAMREL` 0), so holds no role, as in the ASEC. The rule never reads age:
like the ASEC role it holds before the adult rule
`age >= 18 OR (age >= 15 AND role)` is applied.

`microcosm.build.us_runtime.acs_inputs.with_acs_spm_independence_role` applies
it and refuses, naming the units, when the frame already carries the role;
`RELSHIPP` or `age` is missing, blank or off-domain; the SPM partition is not
the household partition; a unit has other than exactly one head, more than one
spouse, or a group-quarters person beside anyone else; `TYPEHUGQ` disagrees
with `RELSHIPP` about group quarters; or a housing unit is left without a
classified adult.

## Evidence

`experiments/acs_spm_role_asec_validation.py` measures both arms on pinned
public files and records aggregate counts only
([receipt](../experiments/acs-spm-role-asec-validation-receipt.json),
[write-up](../experiments/acs-spm-role-asec-validation.md)).

- **ASEC.** In households that are one SPM unit, the SPM head is the reference
  person in every household of every pinned vintage, and "reference person or
  spouse" (`A_EXPRRP` 1-4, the CPS encoding of the rule above) reproduces
  Census's own `SPM_NUMADULTS` and `SPM_NUMKIDS` in __ASEC_MATCH__. It is never
  true where the ASEC rule is false. The cases it misses are reference persons
  and spouses of unrelated subfamilies, almost all adults.
- **ACS.** The derivation run on the full pinned 2024 1-year PUMS:
  __ACS_SUMMARY__

## What the ACS spine cannot see

- **Unrelated subfamilies.** The ASEC rule's `A_FAMTYP 4` branch has no ACS
  counterpart. Among teens it moves at most one person per ASEC vintage in
  one-SPM-unit households.
- **Teens who would head their own Census SPM unit.** In the ASEC a
  15-to-17-year-old roommate or other nonrelative can head a separate SPM unit
  inside someone else's household. The ACS household partition places them in
  the household's unit as a child. That is a property of the partition, not of
  the rule; a reconstructed ACS SPM partition (the development work in
  microcosm#962) would have to supply its own role, and this derivation refuses
  any non-household partition rather than guess.
- **Group-quarters children.** A group-quarters person under 15 is a
  one-person unit with no classified adult under any role
  (__GQ_CHILDREN__ units). The source places ACS group quarters outside the SPM
  universe ([`docs/us-spm-measurement-universe.md`](us-spm-measurement-universe.md));
  until that universe is wired into a build, the derivation counts these units
  in its receipt instead of refusing the spine.

## Where it runs, and where it does not

`build_optional_acs_multispine` derives the role for the ACS spine exactly when
the base it joins carries one, before the transfer, so the pooled column is
complete on both spines and a structural refusal costs no fits. The staging
builder requires it end to end:

- `_require_donor_spm_independence_role` refuses a donor without a complete
  Boolean role before any source is fetched.
- `_require_pooled_spm_independence_role` refuses a staged frame whose role is
  missing, null on any row, non-Boolean, or not covered by the ACS receipt.
- `_engine_input_null_audit` refuses, rather than registers, a null declared
  source input.
- The summary's `reviewed_limitations` carries
  `acs_spm_independence_role_household_partition` with the receipt's counts.

The engine pass refuses too. `fill_reviewed_nulls` raises
`DeclaredSourceInputNullError` (a subclass of `UnregisteredNullError`) when a
register names a declared source input or a declared source input is null, and
names the staging rebuild as the remedy. Staging artifacts built before this
change carry the role only on donor rows and must be rebuilt.

The role is deliberately not part of `map_acs_native_inputs`. That mapping
also feeds the stacked pool (`tools/build_us_multispine_pool.py`), whose ASEC
arm carries no role and whose ACS-row rule is the open question Q1 in
[`docs/us-spm-role-stage.md`](us-spm-role-stage.md) §7. A test pins that the
shared mapping never emits it.

## Invariants and tests

For every valid ACS spine
(`packages/microcosm-build/tests/engine_free/us/test_us_acs_spm_role_properties.py`):

- the role is a complete numpy Boolean equal to the `RELSHIPP` rule, under any
  row order;
- changing any age never changes the role;
- only group-quarters persons under 15 lack a classified adult, and the
  independent composition checker agrees;
- the role equals the loader's own CPS relationship encoding (`A_EXPRRP` 1-4,
  or a group-quarters record's sole person), the analog the ASEC validation
  measured.

For every frame and register
(`test_us_acs_local_reviewed_nulls_properties.py`), the engine-pass fill never
defaults the role and never changes a role value. The example and refusal
tests sit beside them, and
`tests/engine_contract/us/test_us_declared_dataset_source_inputs.py` pins the
refused set to the installed engine's declaration.
