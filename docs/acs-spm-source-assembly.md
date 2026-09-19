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

## Fresh SPM inputs and native integration hold

The hook runs after the one tax-unit construction and household source attachment,
before `map_acs_native_inputs`. Existing tenure mapping follows the new SPM
membership. Existing housing-participation routing awards only the actual
household head's SPM unit. Neither copies an old SPM monetary table. Any existing
SPM amount, flag or provenance column makes this adapter refuse; the dense-parent
regroup helper is a separate workflow.

The native source issuer still accepts the earlier parser hash
`893fd570cf74bf87c133d98e2a450793ad031ce60a40778fb5ea03b3cb9280c8`.
This branch intentionally leaves that admission pin and the graph implementation
inventory unchanged. Consequently native issuance is not qualified on this
branch, including for the default constructor. A reviewed follow-up must bind
the new option, source request, helper/dependency closure, receipts and parser
bytes before any native graph run. It must preserve the reviewed serial-lookup
optimization. No selection-request plumbing, completion-host or retention change
is included here.

## Invented validation

The focused five-file battery passed 199 tests on exact PR45. Absent-assembler
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
