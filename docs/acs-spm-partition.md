# ACS SPM partition proposals

`microcosm.build.acs_spm_partition.reconstruct_acs_spm_partition` proposes
Supplemental Poverty Measure (SPM) resource-unit membership for complete American
Community Survey (ACS) households. It calls the canonical `spm_unit_id` assembler
and returns membership evidence, an old/new crosswalk and an inventory of old
unit-owned fields. It leaves the input tables, tax units, weights and unit amounts
unchanged. It does not calculate poverty or produce a replacement population.

## Dependency and scope

The implementation targets the corrected canonical source in
[spm-calculator PR 45](https://github.com/PolicyEngine/spm-calculator/pull/45),
commit `bcf45768003bb79addfafb0e6d9c7d2d5e547d9d`. Its `units.py` SHA256 is
`ce0d328d856ca81862b4e80947b6f6269a89bbd842cbdbb1569411b319da5a33`.
That source fixes parent-link ordering and the foster-child age boundary.
The current locked calculator is `1.0.0`; it lacks these required behaviors.
`1.0.0.post1` is metadata in the unshipped PR45 checkout, not a published
dependency qualification. A version string alone does not identify these bytes.

This adapter is a development component. Microcosm's existing `1.0.0` calculator
lock and country pins are unchanged. Dependency adoption, native source
qualification, policy acceptance and activation in a data builder remain separate
work. Tests exercise the real corrected assembler; they do not certify a dense
file. The module sits outside `us_runtime` because that package's eager registry
initialization imports the legacy country model.

## Source and decision contract

Inputs are person and old SPM tables, plus independently established
`household_person_counts` keyed by current household ID. The adapter checks that
every supplied household has exactly its declared count. The caller must
authenticate those counts and the source join; deriving counts from a partial
slice would not establish household completeness.

Required person columns are `person_id`, `person_household_id`,
`person_spm_unit_id`, `SPORDER`, `RELSHIPP`, `AGEP`, `MAR` and `TYPEHUGQ`.
Person and household/line-number pointer keys must be unique; IDs must remain
exact integers. The old SPM table must contain exactly the referenced unit IDs.
No values from its amount columns are read.

The raw `RELSHIPP` categories identify relationships to the reference person.
Spouse, partner and own/adopted/stepchild links to that person are recorded as
source observations. Other relatives join the reference-family component without
inventing their exact parentage. The adapter discards old SPM/family fallback IDs,
`A_EXPRRP`, `PEPAR1`, `PEPAR2` and Census assignment flags from the canonical input
view. In particular, ACS's existing `PEPAR2 = reference_spouse` construction is
not a raw observation. [ACS 2024 dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf)

`AcsSpmLink` supplies an explicitly approved inferred parent, spouse or partner
link. A parent link points from child to parent using global person IDs; the
adapter translates it to validated household-local `SPORDER`. Every link carries
the caller's `rule_id`. Missing endpoints, self-links, cross-household links,
duplicate links, excessive outgoing pointers and parent cycles are rejected.
Spouse and partner cardinality applies to both endpoints, so reversing an
inference cannot give an observed couple a second spouse. Observed and inferred
spouse links must agree with both people's raw married status.
The caller's inference policy must establish the substantive validity of each
approved link; this adapter does not implement the full Census/IPUMS pointer
inference ladder.

The canonical assembler joins reference relatives and partners, attaches foster
children below 22 and links approved parents before applying residual-child
attachment. An unrelated child below 15 with a resolved parent joins that parent's
component. A residual child without a resolved parent joins the reference unit.
The latter remains a construction assumption, not observed parental absence.
[Census unit methodology, printed pp.7–10](https://www.census.gov/content/dam/Census/library/working-papers/2011/demo/SEHSD-WP2011-22.pdf#page=8)

## Secondary relationships and unresolved cases

`secondary_candidate_screen_v1` is a review screen, not a claim that it has
recovered all family relationships. It examines non-reference partners,
roommates, foster children and other nonrelatives. A residual person's secondary
relationships need assessment if another such person has an age gap of at least
15 years or either person is below 18. The screen does not select a parent or
spouse. Marital status and similar ages neither establish a resident couple nor
quarantine an otherwise ordinary adult pair.

Straightforward adult roommates outside that screen receive the declared
separate-unit rule, labeled and counted as `modeled_residual_separation`.
An otherwise unattached child below 15 receives the canonical residual-child rule,
labeled and counted as `modeled_residual_child_attachment`. Neither label claims
an observed absence of family. The age-gap screen is an explicit approximation;
it can miss parent relationships outside its age criterion and unreported
secondary cohabitation. It does not screen relatives already in the reference
family as possible partners of nonrelatives: an adult child's unreported partner
can therefore receive the separation assumption. A caller can supply an approved
partner link or an ambiguous assessment for that case. Release acceptance must
assess these limitations; the adapter does not establish source-exact resolution.

In the default strict policy, `AcsSpmLinkAssessment(status="complete", rule_id=...)` records a reviewed
resolution, including the decision that no supplied link is appropriate.
`status="ambiguous"` retains uncertainty. A supplied link alone does not resolve
other candidate relationships. Any unassessed or ambiguous person makes the
whole household's proposed membership null, because a missing link can join
otherwise separate components. The result retains counts and statuses for those
households.

## Explicit development preset

Select `policy=ACS_SPM_DEVELOPMENT_POLICY` to use
`acs_spm_development_reconstruction_v1`. The default strict policy remains
available for source investigations. The development preset applies accepted
links first, then the canonical residual rules even when secondary relationships
remain unassessed or ambiguous. It retains those source statuses and the count
`source_unresolved_relationship_households`; they do not become pending
construction inputs under this preset.

Residual unrelated people aged 15 or older receive separate resource units.
An unrelated child below 15 without an accepted parent link pools with the
reference unit. The receipt labels this `parent_unknown_reference_pooling`;
it does not fabricate a parent pointer. `parent_link_status` reports the explicit
parent-link evidence sent to the assembler, while raw `RELSHIPP` remains
available for other observed household relationships. Foster children below 22
and the reference person's partner follow their distinct canonical attachment
rules. This residual treatment follows the published ACS research construction,
but omitting a full imputed-pointer ladder can leave different units from the
Census implementation. [Census ACS research method, printed pp.5–6](https://www.census.gov/content/dam/Census/library/working-papers/2015/demo/SEHSD-WP2015-09.pdf#page=5)

The preset completes minor roles after assembly:

- A residual singleton aged 15–17 receives a modeled unit-reference role.
- A unique parent aged 15–17 leading an accepted secondary parent component
  receives `accepted_parent_component_reference_v1`. Its parent-link evidence
  keeps its original source/inference classification.
- A component already containing a household reference or adult does not
  automatically give every minor parent another reference role. Multiple
  plausible minor references remain unresolved until explicit role decisions.
- The reference person's minor unmarried partner receives a modeled true role
  by default. Set `minor_partner_role=False` for the prespecified sensitivity;
  membership remains identical and the receipt records this parameter.
- Other nonreference minors receive a modeled child-role input. Under-15s
  cannot be promoted through a role decision. A residual under-15 parent can
  pool with the reference household through the canonical rule; their accepted
  child link remains visible. When both mechanisms operate, the receipt uses
  `accepted_sharing_and_parent_unknown_reference_pooling` and counts the
  applied pooling operation separately from the preserved child link.

These roles describe the selected SPM construction, not observed financial
independence. Exact 2024 adult-bit parity remains unverified. The preset must
undergo partition sensitivity and matched outcome comparisons before native
release acceptance.

```python
from microcosm.build.acs_spm_partition import (
    ACS_SPM_DEVELOPMENT_POLICY,
    reconstruct_acs_spm_partition,
)

proposal = reconstruct_acs_spm_partition(
    persons,
    old_spm_units,
    household_person_counts=authenticated_household_counts,
    policy=ACS_SPM_DEVELOPMENT_POLICY,
    minor_partner_role=True,
    links=approved_links,
)
proposal.require_resolved()
```

ACS provides only reference-person relationships. Census's ACS-SPM methodology
uses reconstructed pointers but does not publish a fully reproducible production
program. These local rule IDs do not claim Census or IPUMS equivalence. No IPUMS
licensed pointer data is consumed. [Census ACS methodology, pp.5–7](https://www.census.gov/content/dam/Census/library/working-papers/2020/demo/SEHSD-WP2020-09.pdf)

## Independent-minor roles and GQ

Membership and measurement roles are separate. A rule applied to raw reference-
head/spouse relationships supplies a true role primitive, labeled
`observed_relationship_rule`. This covers a strict subset of the ASEC source
independence rule, not observed financial independence. In strict mode, other household members aged 15–17
remain unclassified until the caller provides `AcsSpmRoleDecision` with a rule
and explicit boolean value. This includes minor unmarried partners, related
teenage parents and residual minor singleton units. Missing pointers do not
establish independence. For people outside the 15–17 role-sensitive range, the
adapter uses false for non-head/spouse inputs and labels it
`age_not_role_sensitive`, rather than claiming an observed dependent role.

`require_resolved()` requires no pending inputs under the selected construction
policy. Passing it does not establish observational completeness. It does not
validate threshold composition: a 14-year-old reference person retains their
head role but remains below the canonical adult-age rule. The canonical country
measurement path still owns its composition check and generic demographics.

Group quarters (GQ) retain their existing structural membership. They receive
`outside_acs_household_universe` measurement status and no inferred role.
Exclusion does not delete people or assign a nonpoor outcome.

## Crosswalk and downstream obligations

The proposed string IDs use each component's minimum global person ID. They are
stable under input row permutation and household-complete chunking. They are
proposal IDs; a caller may allocate engine-compatible integer IDs while retaining
the crosswalk. GQ ID spelling may change, but its member sets remain unchanged.

`crosswalk` counts people in each old/new unit pair, including null proposals.
`regrouping` inventories every non-ID column in the supplied old SPM table. A
changed resolved partition marks those fields `requires_regrouping`; unresolved
membership marks them `unresolved_partition`. This is deliberately conservative
at table level. The adapter cannot decide which monetary fields should be
recomputed, regrouped from authentic person observations, or separately imputed.
It never copies an old unit amount into each newly split unit.

For fresh ACS assembly, complete accepted `SPM_ID` can enter the existing
`assign_us_unit_structure` passthrough. For a dense repair, the caller must replace
only the accepted SPM membership/table and explicitly rebuild affected inputs.
Rerunning the full unit constructor would also change tax units and exceeds this
component's scope.

## Required qualification before activation

Root owns the secondary-inference policy, minor-role decisions, source-key
qualification and canonical dependency pin. The first empirical check should
compare reconstructed 2023 PUMS households with the same-vintage ACS-SPM extract:
partition signatures, unit sizes and adult/child counts, stratified by unrelated
families, partners, foster children and minors. The available 2023 extract cannot
be joined onto 2024 people. [ACS-SPM research files](https://www.census.gov/data/datasets/time-series/demo/supplemental-poverty-measure/acs-research-files.html)

Native regrouping, canonical measurement, geography checks, before/after
comparisons and release acceptance follow those decisions. No automatic builder
activation or release gate change accompanies this module.

## Compatibility and development receipt

`probe_acs_spm_assembler` checks the actual imported assembler's parent-link
ordering, foster-age boundary and diagnostics call contract. It returns typed
support/reason evidence and the real module path/hash when available. A missing
assembler is an ordinary unsupported result. Reconstruction raises the dedicated
`UnsupportedAssembler` only when an eligible household needs assembly; source
validation and preserved group quarters remain usable without the dependency.
These probes establish specific capabilities, not general correctness.

CI must exercise three environments: no assembler, locked `1.0.0`, and exact
reviewed PR45 source. Existing assembler-dependent tests use strict conditional
expected failures restricted to `UnsupportedAssembler`. All other tests execute
normally; all positive tests pass without expected failures on the exact PR45
source. A future separately qualified dependency update removes obsolete markers.

`build_acs_spm_source_receipt` in `microcosm.build.acs_spm_source_receipt` checks
supplied proposals by replaying the same pure partition/regroup transformations.
Its inputs include both minor-partner sensitivities, a caller-supplied pilot
identity registry with canonical labels, raw roster and household tables, explicit
legacy-default/care evidence and source-reference hashes. It binds input tables,
links, roles, crosswalks, ledger, policy provenance and actual assembler identity
to aggregate-only JSON. It preserves GQ membership and nullable roles, unchanged
unit IDs and one allocation ledger row per split old unit. Unresolved childcare
stays unresolved. The receipt never allocates national IDs or reads reference
files; their hashes are explicitly labeled supplied, not authenticated.

ACS head/spouse role outputs are labeled `observed_relationship_rule`, a strict
subset of the ASEC source independence rule. This describes a rule applied to
observed RELSHIPP, not observed financial independence. Observed relationship
links keep their existing `source_observed` label. Unit evidence is the weakest
of role evidence, construction assumptions, secondary-relationship uncertainty
and accepted internal links. Age-insensitive roles supply no stronger authority.
Unassessed/ambiguous links cannot become observed simply because a development
policy completed the membership. The receipt has fixed `development_source_only`
scope and makes no official-universe, consumer, or release claim.

Later structural pilots must pin every primitive and auxiliary input before
loading, including goldens, registry membership, defaults/null evidence and the
regroup report. Compare the person/household/old-unit/proposed-unit projection
separately from role labels. Record the explicit `source_observed` to
`observed_relationship_rule` role-label migration and both label hashes; the old
whole-file golden hash cannot authenticate renamed bytes. The original registry
remains immutable. No 512-household or country-consumer run is implied by this
source helper or synthetic test evidence.
