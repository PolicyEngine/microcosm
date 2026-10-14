# NZ benefit unit: the dependent-child rule

`packages/microcosm-build/src/microcosm/build/nz/benefit_unit_rule.json` declares how the NZ graph builds benefit units. This note records the statutory basis and the cases the NZ hub settled on 2026-10-12 (G8 method note B8, for package G8-u of #1183). Every number lives in the spec; the shared unit-construction code reads it.

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
  - A parent under 16 stays a dependent child in their own parent's unit, and their child joins that same unit.
  - These branches decide whether the teen parent is a dependent child. When the child's two co-resident parents end up in separate units, the child follows the declared parent-1 placement.
- **Two co-resident parents who are not partners.** The child joins parent 1's unit.
- **A child with no co-resident parent.** The child joins the unit of the household's reference person. Construction refuses the household if that reference person is themselves a dependent child.
- **Orphan's benefit and unsupported child's benefit.** The statute excludes these children from "dependent child", but the donor can't identify them, so they remain dependent children of the unit they join.

## Disclosed approximations

These are listed in the spec's `unobserved_statutory_limbs` and `financial_independence.approximation`. None is applied as a statutory exclusion.

- Usual weekly hours in worked weeks stand in for the statutory average. Employment contracts and the averaging period aren't observed.
- Student allowance grants, analogous schemes, contractual full-time exceptions and main-benefit receipt aren't observed when units are built.
- Single status is approximated by the co-resident partner pointer. A young person whose spouse or partner lives elsewhere is treated as single. MSD's section 8 determinations aren't observed.
- Care, maintenance and financial dependency are represented by household and parent placement, not by observed support.

## Reported counts

Unit construction reports, in the CREATE receipt:
- 16- and 17-year-olds made independent by hours;
- teen parents in each branch;
- households with non-partnered co-parents;
- children placed with the reference person.
