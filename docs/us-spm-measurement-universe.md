# The US SPM measurement universe

PolicyEngine/policyengine-us#9462 makes the SPM measurement universe a
**source declaration**. A dataset supplies `spm_unit_spm_universe_status` ∈
{`INCLUDED`, `OUTSIDE`, `UNRESOLVED`} for every SPM unit; `OUTSIDE` units get
missing SPM amounts and poverty indicators and never reach the threshold
provider, and an undeclared dataset is `UNRESOLVED` and raises
`SPM_UNIVERSE_REQUIRED`. The engine never infers scope from a unit's
composition, its tenure, its age or its geography.

This page covers the Microcosm side. The full design — the source evidence,
the registry survey, the consumer enumeration and the open questions — is the
design note `docs/us-spm-measurement-universe-producer.md` on branch
`us-spm-measurement-universe-producer-design`.

## The producer

`microcosm.build.us_runtime.spm_universe_source` declares the universe from
each record's own source record type, never from what the record contains:

| Source record | Status |
| --- | --- |
| ACS housing unit (`TYPEHUGQ == 1`) | `INCLUDED` |
| ACS group quarters (`TYPEHUGQ` ∈ {2, 3}) | `OUTSIDE` |
| ASEC record | `INCLUDED` |

Both arms come from Census SEHSD-WP2020-09, pages 6–7, and they are not
symmetric. The ACS exclusion is a *data limitation* Census states explicitly:
the public ACS PUMS cannot distinguish noninstitutional group quarters, so the
ACS SPM sample is limited to people living in households, and Microcosm's ACS
spine inherits exactly that limitation. The ASEC ruling is the opposite shape:
the CPS ASEC sample frame already equals the CPS SPM universe, so there is no
ASEC record outside it to declare. The ASEC ruling is therefore made at the
spine rather than per record, and it carries its own refusal — if a frame ever
arrives carrying an ASEC household record-type field, the producer refuses
rather than applying a ruling that was made on the premise that no such field
is read.

`UNRESOLVED` is the engine's *absence* sentinel, not a producer value. A
producer that cannot decide refuses.

### Never an inference

The ACS problem was discovered as SPM units with no classified adult, and the
tempting wrong design is to mark zero-adult units `OUTSIDE`. That would brand
a genuine household-unit data defect as out-of-universe and hide it, and it
would make the universe move whenever the role column moved. The input is
`TYPEHUGQ`, which the source fixes before any composition is computed.
`test_a_zero_adult_housing_unit_is_included` pins this, and
`test_the_declaration_is_invariant_to_age_and_role` pins it in both
directions.

### Never a target, never a gate, never an agreement surface

The status is not a calibration target, not a selection criterion, and not a
release gate. Poverty in this repository is a comparison only. The status is
also deliberately **not** a `spine_agreement` surface: it is *intended* to
differ by spine — that is its entire content — so registering it there would
assert that the ACS and ASEC arms should agree about a fact they are defined
to disagree about.

`OUTSIDE` changes no weight, no membership, and no record's presence in the
file. The fiscal population is unchanged; only its SPM measurability is
declared.

### Refusals

Every refusal is a named, fail-closed `ValueError`:

| Code | Condition |
| --- | --- |
| `SPM_UNIVERSE_COLUMN_EXISTS` | any entity table already carries a declaration |
| `SPM_UNIVERSE_UNDECLARED_SPINE` | a provenance channel with no universe ruling, an untagged frame with no caller declaration, or a caller declaration over a frame that already declares |
| `SPM_UNIVERSE_UNKNOWN_HOUSEHOLD_KIND` | an ACS household whose `TYPEHUGQ` ∉ {1, 2, 3}, missing included |
| `SPM_UNIVERSE_OFF_ARM_HOUSEHOLD_KIND` | a non-ACS household carrying an ACS record type |
| `SPM_UNIVERSE_MIXED_HOUSEHOLD` | one household id resolving to more than one kind |
| `SPM_UNIVERSE_GQ_MULTI_PERSON` | an ACS group-quarters placeholder with more than one native person |
| `SPM_UNIVERSE_ORPHAN_MEMBER` | a membership id matching no owning entity row |
| `SPM_UNIVERSE_EMPTY_UNIT` | an SPM unit with no member |
| `SPM_UNIVERSE_UNIT_SPANS_KINDS` | one SPM unit whose members sit in households of different kinds |
| `SPM_UNIVERSE_DEGRADED_PARTITION` | the ASEC arm's SPM partition is not the native one: `SPM_ID` absent, missing, or not one-to-one with the frame's units per support-clone copy |
| `SPM_UNIVERSE_ASEC_RECORD_TYPE_UNREVIEWED` | the frame carries an ASEC record-type field the spine-level ruling was not derived against |

`SPM_UNIVERSE_UNIT_SPANS_KINDS` is natively unreachable — ACS group-quarters
households are one-person placeholders — and earns its keep the moment a
future spine, a donor merge or a clone-index bug puts a group-quarters person
into a multi-person unit. Silently picking a majority kind there would be a
fabricated universe decision.

`SPM_UNIVERSE_DEGRADED_PARTITION` closes the largest silent-failure mode in
the area. `assign_us_unit_structure` uses the native `SPM_ID` when it is fully
present and falls back to `household_id` otherwise, and *any* missing `SPM_ID`
triggers that fallback for the whole frame — while `asec_pool` deliberately
tolerates a missing `SPM_ID` when globalizing. Declaring `INCLUDED` over a
degraded partition would attest to a unit structure that does not exist. The
check is ASEC-only on purpose: ACS PUMS supplies no `SPM_ID` at all, so the
ACS arm reaches `assign_us_unit_structure` without one and takes the household
fallback by construction — there, one SPM unit per household *is* the native
partition and carries no information.

The native unit key is `(support-clone copy, SPM_ID)`, not `SPM_ID` alone.
The PUF support clone (`puf_support._clone_entity_table`) deep-copies the
person table and remaps only the id and membership columns, so clone copy 1
carries the native `SPM_ID` under new SPM unit ids; keyed on `SPM_ID` alone,
every support-cloned ASEC frame would read as degraded.
`test_the_real_support_clone_operator_output_is_accepted` runs the real clone
operator on invented rows to pin that. A missing clone index is read as the
native copy, which can only add collisions, so it never relaxes this check or
the one-native-person group-quarters check.

## What is wired, and what is not

**Wired.** The producer, its refusals, and its tests. It is classified in the
spine-blindness inventory as a reviewed non-registry module and as a
source-spine provenance owner (the provenance tag is its input, not something
it must be blind to).

**Not wired, deliberately.**

- **No build path calls it.** Neither `tools/build_us_acs_local_release.py`,
  `tools/build_us_fiscal_refresh_release.py` nor the legacy multispine base
  invokes it. Adding a call site is separate work.
- **No export.** `microcosm.frame.adapters.policyengine_us` is untouched, so
  the column reaches no release H5. The adapter change is blocked upstream:
  see the design note's §4 and Q-A.
- **No enrichment-lane carry.** `microcosm.data.h5_enrichment` writes one
  Boolean person column at the HDF type level; carrying an `spm_unit`-entity
  string column is a separate, larger piece of work (design note §3).
- **No consumer change here.** `check_spm_composition` still grades every SPM
  unit with no universe predicate. `reform_validation._person_rate` already
  reads the poverty indicator as nullable (#976, merged 2026-09-22), so a
  person with a missing indicator leaves both numerator and denominator; on
  the pinned engine, where `in_poverty` is Boolean, nothing is missing.
- **No producer source pin.** The module is deliberately *not* added to
  `source_enrichment.PRODUCER_SOURCE_FILES`: that tuple is the SPM-role
  enrichment producer's own reviewed source closure, and
  `validate_source_enrichment_candidate` refuses a recorded inventory whose key
  set differs from it. This module is not in it until the enrichment lane
  carries the column (ruled 2026-09-22; see "Rulings" below).

## Open questions

These are the design note's, carried here so they are answerable from the
repository. The note states each in full.

- **Q-A — the formula-owned collision.** The engine variable as drafted cannot
  be exported by Microcosm at all, for a reason unrelated to the pin. Resolve
  upstream with a source-declaration marker, ask upstream to move the
  household default, or add a Microcosm-side allowlist?
- **Q-B — sequencing.** Ship the held-back frame column now, wait for the
  engine release, or both?
- **Q-C — group-quarters housing.** ACS group-quarters records are drawing a
  housing-receipt flag today with no housing unit. Force the flags `False` on
  the group-quarters mask, or supply an explicit reported-subsidy path? This
  is a defect on `main` independent of the universe rule.
- **Q-D — the published-rate discontinuity.** The 104 state SPM rates move
  when outside persons leave the denominator, correctly but visibly.
- **Q-E — the scope of the ASEC ruling.** Encoded here as a named, cited
  constant with its own refusal, per the note's recommendation; signed off
  2026-09-22 (below).

## Rulings (Max Ghenis, 2026-09-22)

- `spm_universe_source.py` stays out of
  `source_enrichment.PRODUCER_SOURCE_FILES`.
- The missing-`SPM_ID` refusal on the ASEC arm stays strict: an absent or
  partly missing `SPM_ID` refuses rather than falling back.
- `puf_tax_detail` stays unruled: a household whose only provenance tag is
  that clone-operator channel refuses with `SPM_UNIVERSE_UNDECLARED_SPINE`.
- The ASEC spine judgement is signed off as the named, cited constant
  `ASEC_SPM_INCLUDED_AUTHORITY` (Census SEHSD-WP2020-09, pages 6–7).

## Upstream

- Engine PR: PolicyEngine/policyengine-us#9462, fixing #9461.
- The dataset must carry `spm_unit_spm_universe_status` on `spm_unit` for
  **every year** SPM is requested; a single-year H5 resolves only its base
  year. Encode as member names.
