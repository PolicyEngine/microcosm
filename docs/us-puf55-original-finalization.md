# Original-arm PUF55 whole-arm finalization

The conservative development placement
([us-puf55-original-development-placement.md](us-puf55-original-development-placement.md))
places 8 of the 55 PUF55 outputs on the original (survey) arm. It covers only units
whose members are known for all twelve fixed inputs. Whole-arm finalization is its
opt-in successor. It covers the rest of the arm under an explicit numerical policy.
Neither path qualifies a release.

```python
run_us_survey_enrichment(
    parent,
    ...,
    original_application_seed=73,  # required, distinct from the clone-one seed
    original_finalization="microcosm.us.puf55-original-finalization.v1",
)
```

The option takes one exact policy name
(`puf55_original_finalization.POLICY`). A truthy value, an alias, bytes or a
policy without `original_application_seed` is refused before the parent run is
read. If you leave it out, every declaration, kernel and receipt field of the
conservative placement stays the same.

## What it attaches

Arm one introduces 52 person and 3 tax-unit outputs. Twelve person outputs are
fixed inputs that the original arm preserves: 8 survey-preserving financial
leaves and 4 recorded development assumptions. They are never rewritten. The
other **40 person and 3 tax-unit outputs** are candidates. They must pass the
same ownership checks as the conservative cut. Each must be absent from the
pre-PUF parent and introduced by the checked arm-one attachment. Its current
owner must be that attachment or the receiving structural version, with the
arm-one values carried unchanged. Its arm-zero cells must still be null.

### Mixed-known-member units

The observation-conditioned chain (`graph_legacy_apply_observed`) replaces a
target's raw draw with the qualified value only for rows where the observed
mask is known. For the original arm, that mask is the fixed-input qualifier's
`tax_unit_known`, which requires every member of the unit to be qualified
(`sum_all_modeled_tax_unit_members_only_when_every_member_qualified`). Later
targets condition on the merged value. If one member of a unit is unknown for
fixed input *f*, every target after *f* was conditioned on *f*'s raw draw. That
draw is not the preserved value.

The rule therefore works per target. A unit receives target *t* only if every
fixed input earlier than *t* in the chain is known for all its members. In the
current profiles the fixed inputs sit at chain positions 0–7 (financial) and
10, 11, 28 and 31 (pension, IRA, rental and farm). Both route profiles share
one target order. Known members keep their fixed values. Unknown members of a
mixed unit stay unresolved. The rule invents no residual, constrained total or
person-grain completion.

### Person allocation

A unit draw is split among the unit's arm-zero members by the maintained
arm-one helpers, `_write_person_tax_unit_totals` and
`_write_person_tax_unit_boolean_counts`. They use the maintained distribution
bases (`_PERSON_OUTPUT_DISTRIBUTION_BASIS`), in the maintained pass order.
Bases come from arm one: the fixed inputs, earlier candidates as placed in this
pass, and non-PUF columns such as `is_full_time_college_student`. A later
candidate contributes zero, as it does on arm one. Earnings-universe outputs
keep the maintained age-15 universe: under-15 members get the receipted
universe zero.

Arm zero is stricter than arm one in three cases. These units stay unresolved
instead of falling back to the first member or dropping mass:

| Case | Reason code |
| --- | --- |
| A multi-member unit, where no declared basis column exists | `allocation_basis_absent` |
| A multi-member unit, where a basis value is null for an allocation member | `allocation_basis_unresolved` |
| A nonzero earnings draw, where no member is aged 15 or older | `allocation_universe_empty` |

A basis that is present but zero keeps the maintained first-member fallback.
So does an output with no declared basis. The document counts these units per
output (`first_member_fallback_units`, `rank_ties_by_position_units`).

### Numerical policy

| Step | Arm one | This policy |
| --- | --- | --- |
| Nonnegative clipping | clips | a negative draw stays unresolved |
| Donor-value snapping | snaps sparse, boolean and discrete outputs | not applied |
| Tail-bound cap (`non_sch_d_capital_gains`) | 0.999 quantile | not applied |
| Donor positive-rate pruning | sparse outputs | not applied |
| Signed-mass alignment | farm operations, partnership SE earnings | not applied |
| Boolean QBI counts | snapped, then rounded and capped at allocation members | rounded and capped at allocation members (same helper) |

The maintained finalizer (`finalize_us_puf_tax_detail_predictions`) writes model
values only on the PUF clone arm, so no donor alignment step has an arm-zero
precedent. Each step aligns the whole receiving arm to a donor statistic. Arm
zero's modeled units are a subset of that arm, since mixed-known and
domain-unresolved units are excluded. The donor statistics are also not plumbed
to this arm. Whatever arm-one pruning corrects in its raw draws may also be
present in these draws. This policy leaves it uncorrected but measured.

For each output, the document records the modeled units, weighted units,
weighted positive share and weighted total over resolved units. These let a
later reviewed rule be compared with donor rates. They are comparison-only:
never a calibration target, tuning signal, selection criterion or release
gate.

## Own-tail copies (capital-gains/AGI tail)

PR #992 found that release stages on main keyed support copies by source ID and
role. The capital-gains own-tail copy (clone index 2) keeps its clone-one twin's
source IDs and PUF channel, so that key cannot tell the two apart. Main now
keys by clone index (`support_copy_rank_series`). The native line was checked
at `47960af43`:

- `graph_combined_clone` pairs copies by `(entity_source_id, clone_index)`. It
  refuses any clone index outside {0, 1} (`CLONE_ROLE_UNEXPECTED`).
- `puf55_survey_recipients._aligned` keys by `(source ID, clone index)`. It
  requires clone indices {0, 1} and exactly two copies per source unit
  (`CLONE_PAIRS`).
- `puf55_route_finalization._receiving_axis` requires clone indices {0, 1}
  (`CLONE_AXIS`).
- None of the four stages #992 changed is imported by a native `graph_*` or
  `puf55_*` module: `sipp_head_start`, `voluntary_filing`,
  `prior_year_income` and `ssi_disability_criteria`.

So the native PUF arm has no `(source ID, role)` key. These modules run on the
financial population, or on arm one's receiving version derived from it. If a
tail expansion came before them, they would refuse the copy; they never merge
it with its twin. The native tail expansion is not on this branch. It lives on
the unintegrated `native-agi-tail-*` branches, which copy whole clone-one
households to clone index 2.

The original-arm placement and finalization read the late receiving terminal,
so a tail expansion could come before them. Both now split tail copies off by
clone index (`_receiving_core`), in the pure results and in the graph
declarations and terminal observer alike, so a tail in the observed terminal
leaves the declared nodes unchanged. The copies must be whole households, closed
under every group membership. The remaining core must reproduce the arm-one ID
axis row for row, or the step is refused (`TAIL_MEMBERSHIP`, `CLONE_DOMAIN`,
`ID_AXIS`). Every tail cell is carried unchanged, and the document records the
per-entity copy counts. Placing values on the tail belongs to the tail owner.
The existing ownership checks still apply to the core. If a later
non-structural node owns a PUF output column, the step is refused
(`ARM_ONE_OUTPUT_OWNER`). Core values must still equal arm one's
(`ARM_ONE_OUTPUT_CARRIED`). Two-clone placement documents are byte-identical to
before.

## Graph declaration

The keep-all node (`survey_puf55.original.placement_receiving`) and the attach
node ID (`survey_puf55.original.placement_attach`) are the same for both
policies. Under finalization the attach node has its own kernel reference
(`us.survey_puf55.original_finalization_attach@1`) and artifact type
(`microcosm.us.puf55_original_finalization`). It carries the exact numerical
policy as a parameter and owns all 43 candidates (`rewrite=True`). It declares
reads of `age` and the present non-candidate bases, taken from arm one, so a
declaration made before the terminal matches the observed terminal. It consumes
the same 55-step model, training, raw-draw, conditioning and apply-state edges
as the conservative cut, through the same strict conditioning merger. The
enrichment receipt adds `finalization_policy` only when the option is set.

## Verification scope

`test_us_puf55_original_finalization.py` covers the following, on invented
Frames and synthetic 55-target envelopes (built from a real three-target
template):

- chain prefixes
- mixed-unit eligibility
- maintained-helper parity
- basis refusals
- boolean counts
- the earnings universe
- lossless dtype writes
- replay invariance
- tail-copy carry, appended and interleaved. Every candidate cell of a copy
  differs from its twin's, so a tail cell filled from its twin fails the tests
- tail refusals, including a tail cell changed inside the carry
  (`TAIL_CARRIED`)
- the arm-zero clone axis (`ARM_ZERO_AXIS`)
- declaration compile
- binding observation
- kernel policy routing
- the typed full55 result path

None of this is evidence of 55 fitted models, a genuine financial owner, a
whole-host run or a calibrated dataset.

Still open:

- a genuine host run on admitted sources
- a reviewed decision on whether, and how, arm zero should be pruned or
  aligned to donor statistics
- tail value placement by the tail owner
- a check that downstream consumers accept the 43 re-owned columns
