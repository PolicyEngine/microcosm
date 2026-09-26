# ACS SPM construction before source mapping

`microcosm.build.acs_spm_source_assembly` supplies an explicit development
partition for an ACS source Frame whose SPM table contains only `spm_unit_id`.
It preserves tax units, all other memberships and tables, household DESIGN
weights, strata and the mass log. It does not regroup amounts or run a country
model.

```python
from microcosm.build.acs_spm_partition import ACS_SPM_DEVELOPMENT_POLICY
from microcosm.build.acs_spm_source_assembly import AcsSpmSourceAssemblyOptions
from microcosm.build.us_runtime.acs_pums import build_acs_pums_unit_frame

frame, metadata = build_acs_pums_unit_frame(
    source,
    spm_construction=AcsSpmSourceAssemblyOptions(
        policy=ACS_SPM_DEVELOPMENT_POLICY,
        minor_partner_role=True,
    ),
)
```

The default `spm_construction=None` retains the existing constructor path. An
explicit request checks canonical assembler capability before opening source
archives and raises `UnsupportedAssembler` if unavailable or incompatible. The
locked calculator 1.0.0 lacks the required behavior; the passing synthetic test
environment uses exact calculator PR45 source
`bcf45768003bb79addfafb0e6d9c7d2d5e547d9d`, with `units.py` SHA256
`ce0d328d856ca81862b4e80947b6f6269a89bbd842cbdbb1569411b319da5a33`.
This change does not adopt a dependency or change a lockfile.

## Registry and evidence

The pure `assemble_acs_spm_source` function returns a Frame, partition projection,
numeric identity registry, crosswalk, unit evidence and aggregate receipt. Its
required `id_ceiling` must be an exact nonnegative int64 integer, excluding
Booleans. The ACS constructor supplies the existing reviewed clone-safe ceiling.
Old IDs survive unchanged member sets; split components receive distinct IDs
above the selected old-ID maximum, ordered by native person keys. Overflow is an
error.

Construct one registry from the whole selected roster. Pass the same registry
when applying the adapter to complete-household subsets. The registry binds
native `(SERIALNO, SPORDER)` keys, the supplied person and household identities,
raw relationship/age/marital/universe evidence, household counts, construction
policy and actual assembler hash. It rejects changed source records, incomplete
households, partition changes and identity collisions. Row order may differ;
identities do not become stable across different selection domains. These are
generated component identities, not published Census SPM identifiers.

The adapter temporarily joins household TYPEHUGQ and normalizes SPORDER for the
partition helper. It does not change the original stored tokens. Uncertainty
in secondary relationships remains visible in every affected household's unit
evidence, including a unit containing an observed reference person. The returned
nullable roles retain their original rule/source labels. Relationship allocation
provenance remains unresolved. No engine `is_spm_independent_minor_role` or annual
`spm_unit_spm_universe_status` is inferred or written.

The source constructor retains the aggregate receipt in its metadata and the
Frame's immutable metadata. The pure adapter returns the separate rowwise
projection for callers that need it; the constructor does not publish those rows
in its aggregate receipt. The receipt binds supplied tables, not authenticated
source files, and does not grant source issuance, consumer or release authority.

## Fresh SPM inputs and captured native construction

The hook runs after the one tax-unit construction and household source attachment,
before `map_acs_native_inputs`. Existing tenure mapping follows the new SPM
membership. Existing housing-participation routing awards only the actual
household head's SPM unit. Neither copies an old SPM monetary table. Any existing
SPM amount, flag or provenance column makes this adapter refuse; the dense-parent
regroup helper is a separate workflow.

The captured housing and native coverage owners accept an explicit
`spm_construction` option. They invoke the evidence-returning constructor against
their private, authenticated archive captures. The public constructor retains its
two-value return contract. Native owners retain canonical bytes produced from
the actual membership, links, partition crosswalk, regrouping summary, registry,
native crosswalk and unit uncertainty tables, including declared column rosters
and dtypes. They bind those bytes to the captured archive identities, options,
assembler and implementation identities. A 256 MiB limit bounds the development
evidence payload after serialization; it is not an allocation preflight or a
full-source memory qualification.

After capturing the actual evidence, the housing owner removes exactly the
`acs_spm_source_assembly` aggregate key from its mapped graph Frame and records
that transport in its receipt. All other metadata stays intact. The aggregate
receipt remains in the owned evidence, and direct constructor callers retain it
in Frame metadata. It becomes neither graph-context operator authority nor an
engine role. Native verification checks the retained bytes; the preparation's
nested seals also retain them. Missing or changed evidence refuses even though
the mapped Frame no longer carries that aggregate key. Defensive decoded views
are values, not new detached authority.

A version 1 `selection-request.json` keeps its exact closed shape and imports no
optional assembler. Version 2 requires the additional closed field:

```json
{
  "acs_spm_construction": {
    "policy": "acs_spm_development_reconstruction_v1",
    "minor_partner_role": true
  }
}
```

Use the exported `ACS_SPM_DEVELOPMENT_POLICY` value for the policy field and
`microcosm.us.survey-population-request.v2` for the protocol, retaining the existing
declaration, fraction and seed fields. No mode is enabled by default. The request
bytes bind the construction option into source and graph identities. Existing
survey callers read that request; financial/completion and retention host APIs
are unchanged. Source preparation checks capability before opening population
archives. The native owner additionally requires the exact reviewed PR45 defining
file and verifies the loaded callable during native verification. No global
package dependency or lockfile changes.

`AuthenticatedSurveyPopulationPreparation.acs_spm_construction_evidence` returns
a defensive view of the checked native records. Its IDs belong to the whole
selected native roster. The existing per-entity `receipt["origins"]` map relates
native person, household and SPM IDs to assembled IDs; cloning remains a later,
separate operation. No rowwise roles or universe status are inferred into model
columns.

The isolated branch updates the accepted parser/housing bytes and reviewed
helper inventory. These revisions change source/implementation identities even
for default requests; old receipts are not claimed byte-identical. The earlier
first slice at `fec3f9033ab47b752b2d5f8890507bcf06ad3428` intentionally left native
admission blocked. Its separate review and test receipts remain historical
source-only evidence. Frozen native performance and retention checkouts are not
changed by this continuation.

## Invented validation

At the first-slice source revision `fec3f9033ab47b752b2d5f8890507bcf06ad3428`,
the focused five-file battery passed 199 tests on exact PR45. Absent-assembler
and installed-1.0.0 environments each passed 151 tests with 48 strict expected
failures restricted to `UnsupportedAssembler`; neither had skips or unexpected
failures. The new assembly file contributes 42 passing PR45 cases, or 23 passing
and 19 expected unsupported cases. The imported partition/probe tests and existing
ACS parser/housing-participation tests retain their assertions.

Tests cover tax/non-SPM conservation, role sensitivity without membership drift,
GQ preservation, source refusals, ID overflow/collisions, native-key binding,
shuffled and complete-household chunk application, exact large integer IDs,
and real constructor/tenure/head-only participation functions. Guards recorded
zero country imports, network requests and population-file reads. All source
records are invented fixtures. Earlier unsuccessful attempts remain in the local
evidence; this battery does not replace historical PR962 helper-matrix or actual
512A reports. No actual population, 512B, full-source performance, calibration or
release qualification follows from these tests.

## Native request and issuer validation

The continuation passes 218 focused adapter/native tests on exact PR45, plus
95 existing ACS literal/native coverage regressions. The 19 new cases cover
actual invented archive issuance, exact options and capability refusal before
source access, the default path without importing the assembler, actual retained
rows and native-to-assembled identity maps, non-SPM/weight conservation, missing
or changed evidence, copied-token refusal, source mutation, candidate replay,
and cold/required graph replay. Changing the explicit option changes the graph
key and refuses reuse of the old required result. The existing `return_values`
path retains the checked preparation and its defensive evidence accessor.

Absent-assembler and installed-1.0.0 environments each pass 163 tests with
55 strict expected failures restricted to `UnsupportedAssembler`. Of those,
the new native file contributes 12 passing refusal/default cases and seven
unsupported cases. No test is skipped; unexpected exceptions remain failures.
Final completed runs record zero country-import, network and unapproved-file
read guard events. All archives, HDF fixtures and graph populations in these
tests are invented. The implementation and caller tests do not initialize a
country model.

Earlier failing fixture/assertion attempts, the graph metadata-boundary refusal
and a combined synthetic run stopped by its initial 120-second CPU limit remain
in the local evidence. The final focused runner uses a 600-second CPU ceiling
and segregated temporary fixture directories. These are development test
receipts, not source-scale memory or runtime measurements. No actual population
application, country512B, engine role delivery, annual-universe qualification,
dependency adoption or release claim follows from this continuation.
