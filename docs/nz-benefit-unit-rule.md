# NZ benefit unit: the dependent-child rule

`packages/microcosm-build/src/microcosm/build/nz/benefit_unit_rule.json` declares how the NZ graph builds benefit units. This note records the statutory basis, the cases the NZ hub settled on 2026-10-12 (G8 method note B8, for package G8-u of #1183), and the hub's amendment of B8 on 2026-11-15. Every number lives in the spec; the shared unit-construction code reads it.

## Statutory basis

Social Security Act 2018, Schedule 2, as at 1 April 2026. The source is the PCO XML pinned in `TheAxiomFoundation/axiom-corpus` release `nz-rulespec-2026-07-25`, at `data/corpus/sources/nz/statute/2026-06-16-rulespec-nz-pco/act/public/2018/0032/wholeof.xml`. Its sha256 is recorded in the spec's `statutory_source.pco_xml`.

- **child** (LMS6902): a single person under 18, other than a 16- or 17-year-old who is financially independent.
- **dependent child** (LMS7234): a child whose care is primarily the person's responsibility, who is maintained as a member of their family and is financially dependent on them. It excludes children for whom a young parent payment, orphan's benefit or unsupported child's benefit is paid.
- **single** (DLM6784713) means not in a relationship. **in a relationship** (DLM6784537) means married, in a civil union, or in a de facto relationship.
- **financially independent** (DLM6784504): in full employment, or receiving a student allowance grant, payments under an analogous scheme, or a main benefit.
- **full employment** (DLM6784514): employment of no less than an average of 30 hours a week.

So the spec declares `age_limit_years: 18` (exclusive) and `partnered_person_is_dependent_child: false`. A 16- or 17-year-old whose donor record shows at least 30 usual weekly hours forms their own unit.

## Cases settled by the hub

The statute defines who is a dependent child. It does not say how a microsimulation should place people in units when the household data only give co-residence pointers. The hub settled those cases as follows.

- **Teen parents.** A 16- or 17-year-old with a co-resident child of their own forms their own unit with that child.
  - The basis is that Young Parent Payment is a main benefit for parents of that age, and receiving a main benefit makes a 16- or 17-year-old financially independent (DLM6784504, limb d), so not a child.
  - The rule infers this from the co-resident child pointer before the receipt layer runs; receipt itself is not observed here.
  - A parent under 16 with a co-resident parent stays a dependent child in their own parent's unit, and their child joins that same unit. A parent under 16 with no co-resident parent is placed as a child with no co-resident parent (below); if they are the household's reference person, they head their own unit and their child joins it.
  - These branches decide whether the teen parent is a dependent child. When the child's two co-resident parents end up in separate units, the child follows the declared parent-1 placement.
- **Two co-resident parents who are not partners.** The child joins parent 1's unit.
- **A child with no co-resident parent.** The child joins the unit that holds the household's reference person. If the reference person is a dependent child with a co-resident parent, that is the unit their parent chain leads to, usually their parent's. A reference person under 18 with no co-resident parent isn't treated as a dependent child, even if no partner, hours or parenthood test makes them independent: they head their own unit, and any other child with no co-resident parent joins it. This is the hub's amendment of 2026-11-15 (below); before it, construction refused these households.
- **Orphan's benefit and unsupported child's benefit.** The statute excludes these children from "dependent child", but the donor can't identify them, so they remain dependent children of the unit they join.

## Amendment of 2026-11-15

B8 said construction refuses a household whose reference person is a dependent child. On 2026-11-15 the NZ hub amended B8 and dropped that refusal: the spec now sets `refuse_dependent_reference_person: false` and records the amendment under `child_without_coresident_parent.amendments`.

- **What the refusal covered.** The shared constructor refused the whole frame in two cases:
  - (i) the reference person is under 18, has no co-resident parent and isn't independent by partner, hours or parenthood, so the constructor kept them dependent and treated them as their own parentless child;
  - (ii) the reference person is a dependent child with a co-resident parent, and another child in the household has no co-resident parent.

  Case (ii) is the refusal B8 states. Case (i) was the constructor's reading of it. The amendment drops both.
- **Why.** The strict reading refused real donor households, and one such household aborts the whole frame. The first real-data NZ transport run stopped on it. Literal B8 would still refuse case (ii), and keeping only that refusal would need a change to the shared constructor.
- **How many.** On the pinned donor, measured with the CREATE kernel's own reader and this spec, 9 of 57,240 households tripped the refusal, 0.0024% of household weight:
  - 6 in case (ii), each with a reference person aged 15 or 16 whose parent is present, and another child with no co-resident parent;
  - 3 in case (i), each with a reference person aged 16 or 17 with no co-resident parent.

  All 9 have a member aged 18 or over; none is a household of minors only. Without them the strict rule builds the frame.
- **What happens to them now.** The whole frame builds. Each case (i) reference person heads a one-person unit. In each case (ii) household, the child with no co-resident parent joins the unit of the reference person's parent, which no minor heads. Of the frame's 378 unit heads aged 17 or under, these 3 are the only ones heading a unit this way; the rest are partnered, independent teen parents or independent by hours.
- **Effect on the Accommodation Supplement.** A minor who heads a unit this way reads as a sole parent if the unit also holds another child with no co-resident parent, or their own child when they are under 16. The Accommodation Supplement's unit-based inputs (sole parent, has dependent children, two or more dependent children) and the rate bridge's sole-parent and child-count branches read that unit composition. The person-level inputs read parent pointers, so they don't count a sibling as that person's child. On this donor the 3 new heads are one-person units, so no minor reads as a sole parent through this branch.
- **What construction still refuses.** A dependent child whose caregivers never reach an adult is still refused, and that covers every household with no adult. A household of minors only now has an adult whenever its reference person has no co-resident parent, so it is refused only for unusual pointer data, such as a reference person who is a dependent child and whose parent 1 is under 16 with no co-resident parent or partner. No donor household reaches this refusal.
- **What the receipt counts.** The CREATE receipt gains two counts with this amendment: `parentless_reference_person_heads`, the minors who head a unit through this branch, and `younger_independent_teen_parents`, parents under 16 who aren't dependent because they head a unit this way or have a co-resident partner. With the second, the three teen-parent counts partition every parent under 18.

## Disclosed approximations

These are listed in the spec's `unobserved_statutory_limbs` and `financial_independence.approximation`. None is applied as a statutory exclusion.

- Usual weekly hours in worked weeks stand in for the statutory average. Employment contracts and the averaging period aren't observed.
- Student allowance grants, analogous schemes, contractual full-time exceptions and main-benefit receipt aren't observed when units are built.
- Single status is approximated by the co-resident partner pointer. A young person whose spouse or partner lives elsewhere is treated as single. MSD's section 8 determinations aren't observed.
- Care, maintenance and financial dependency are represented by household and parent placement, not by observed support.
- A reference person under 18 with no co-resident parent heads their own unit (the amendment above). The statute makes no exception for a household reference person: the pointers name nobody whose dependent child they would be, and who cares for them isn't observed.

## Reported counts

Unit construction reports, in the CREATE receipt:
- 16- and 17-year-olds made independent by hours;
- teen parents in each branch: independent at 16 or 17, not dependent under 16 (a partner, or a reference person with no co-resident parent heading their own unit), and dependent under 16;
- households with non-partnered co-parents;
- children placed with the reference person;
- reference persons under 18 with no co-resident parent who head their own unit (the amendment above).
