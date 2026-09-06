# Composed ASEC demographic binding

This stage binds the accepted ASEC sex and household-reference source projection
onto the ASEC rows of the combined ASEC–ACS survey population. It is a local
engineering intermediate, not a calibrated population or release candidate.
The source contracts remain in `us-asec-demographic-source-contract.md` and the
parent composition contracts remain in `us-composed-asec-binding.md`.

`graph_composed_asec_demographics.py` declares two actual graph kernels:

- `us.composed_population.asec_demographic_source@1` reads the exact declared
  projection file with the generic `raw-bytes-v1` codec. It checks the framed
  bytes, canonical header, source contract and implementation, and reconstructs
  the classification states and diagnostics from the carried observations.
- `us.composed_population.asec_demographic_bind@1` consumes that typed projection,
  its source document, and the existing composed arm/context, income observation,
  and prepared receipt artifacts. It produces masked nullable canonical columns,
  retained source states, a context document, and a binding receipt.

The demographic projection carries the original prepared parent's numeric person
IDs and income cohorts. These are not Census PERIDNUM strings. The graph binds
its full native-key roster digest to the already admitted income observations,
verifies the parent person/cohort/age correspondence, then resolves the sampled
ASEC arm by `(income year, parent person ID)`. It crosschecks the published
household coordinate, line number, and observed age for every selected person.
Repeated native household IDs across cohorts are distinct household keys. A
foreign native roster, missing or ambiguous key, wrong cohort, and merged or
split household partition refuse without dropping rows.

The source projection was authenticated against Census members by its separate
accepted producer. This binding does not reread Census microdata. Transport and
internal consistency checks are weaker than independent source reconstruction;
the externally reviewed exact artifact pin is still required. The projection
does not carry the parent's household membership array. Its recorded successful
bijection is inherited from the source producer; the binding independently
checks the actual composed household partition against published coordinates.

The two canonical cells already exist in the composed population and therefore
use explicit `Owned(..., rows="person_composed_asec_row", rewrite=True)`
declarations. Existing ACS values and missingness remain unchanged. Accepted ASEC
sex states 1 and 2 map to false and true `is_female`; household-reference states
1 and 2 map to true and false `is_household_head`. Unbound states become genuine
nullable boolean nulls, never false defaults. Allocation state and both unbound
reason codes remain separate nullable integer columns. The three new provenance
columns remain null on ACS.

Household head means the household reference person recorded by A_EXPRRP. It
does not mean tax-unit head or family head. P_SEQ and line number never supply a
headship verdict. Age remains the source observation; no cross-arm period
harmonization, national sex calibration, congressional-district validity,
housing fit, PUF transfer, or launch eligibility is claimed.

The integrated entry points are `composed_asec_demographic_population_graph` and
`composed_asec_demographic_registry`. Supply the actual source projection SHA-256,
row count and cohort row counts. `composed_asec_demographic_nodes` appends the
existing ASEC arm and measurement nodes plus the new source and binding nodes
when constructing a larger graph. The canonical incumbents must already be
nullable booleans; a different dtype or absent incumbent is a contract change,
not a license to fabricate a representation.

The test file uses invented source members exclusively. It checks real cold and
required warm graph execution, source classifier and direct binding parity,
sampling, nullable states, exact ACS/other-column/weight preservation, native
roster mismatch, cohort and coordinate refusals, source transport tampering,
and relevant cache-key invalidation. A separate root review must approve this
prepared stage before genuine source artifacts enter it.

## Scoped implementation identity

`graph_composed_asec_demographic_implementation.py` validates the existing
`composed_asec_binding_v1` implementation manifest and adds the three modules in
`asec_demographic_graph_inventory.json`. The original implementation service and
global inventory retain their accepted bytes. The separate inventory binds its
own bytes, exact dependency and module rosters, classified imports, and the
existing AST checks for unbound imports and resource access. Its manifest also
includes the live demographic source contract, member pins, transitive source
verifier identity, and dependency versions. Only the two new kernels consume
this extension manifest.

Adding this inventory does not change upstream keys. There is one independent
upstream change relative to the accepted composition: `e959b3503` corrected and
sealed the composed arm's unbound-demographic metadata. That actual receipt
change deliberately invalidates the arm binding and its two measurement nodes.
The preparation, harmonization, and four geography nodes retain their accepted
identities. Their cached source outputs can supply the three corrected receipt
nodes and the two new demographic nodes without rerunning source preparation.
The older receipts remain historical artifacts; this extension does not relabel
them or overwrite their recorded implementation hashes.

The local comparison uses accepted receipt/manifest metadata and invented
source bytes. A prepared-directory content key hashes raw packed file contents,
so individual file SHA pins alone cannot reconstruct that aggregate key.
Full genuine replay therefore remains a separate root-reviewed step. This
implementation does not read genuine source tables to perform the comparison.
When an authenticated run receipt supplies those aggregate source-content
keys, all nine node keys can be recomputed from metadata alone. The local
comparison uses the accepted clone extension's receipt to do this and confirms
the six unchanged keys and three deliberate changes described above.
