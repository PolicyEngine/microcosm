# Original property-income source composition

`qualify_current_property_income_sources` composes the source branches for the
property-income model. It borrows the existing shared-predictor preparation,
qualifies the original ACS anchors and ASEC interest, income-routing and
dividend/survivor observations, and applies the reviewed pure ASEC donor basis.
It fits no model, supplies no tax split and performs no clone attachment.

The arguments match the existing shared predictor boundary: the actual
preparation, allocated population and clone population, with optional shared
demographic conditioning and geography configuration. The allocated/clone
objects satisfy that existing ownership contract. Their weights never supply
the donor weights. All donor membership and DESIGN weights come from the
original source Frame selected by the shared qualifier.

The result is descriptive `QualifiedPropertyIncomeSources`. Its source values,
source Frame and full origin document are retained for inspection and later
source/clone joins. The object has no issuer registry and cannot replace the
actual retained preparation. The complete descriptive seal is a mutation
check, not a credential. Downstream owners must qualify before consumption
and after their last relevant I/O before returning or exporting a successor.

## Selected branches

`donor_frame` is the original ASEC DESIGN Frame restricted to jointly eligible
persons, with referenced groups pruned by the existing Frame selector.
`donor_columns` contains the qualified shared features, reported property total
feature and four original components. It uses the same exact ordered person
IDs as that branch. The four component columns and reported total import their
canonical names from `property_income_constants.py`.

`recipient_frame` selects original ACS persons with a finite, known adult INTP
anchor. `recipient_columns` contains the same shared features plus the qualified
ADJINC-adjusted anchor under the reported-total feature name. The existing
recipient matrix codec preserves their ordered identities and float64 bits.
`recipient_matrix` is the encoded matrix. Source AGEP and canonical shared
feature age remain distinguishable; no raw age is overwritten by the feature.
Shared earnings features preserve the existing qualifier's declared NIU operator;
this composition adds no new zero completion for earnings or property anchors.

Adding these model columns to a graph population is a later visible graph
operation. The selected Frames themselves keep all original source columns,
weights, memberships, strata and metadata. The source composition does not
quietly change a Frame's tax leaves or model columns.

The complete original selected source axis is never reduced in `origins` or
`source_frame`. `origin_document` preserves all original entity origin records;
`donor_basis` contains every original ASEC row's component knownness, provenance,
exclusions and design-weight summaries. `recipient_diagnostics` preserves every
selected ACS row's raw literals and parsed anchor metadata, plus separate
under-15 and unknown-anchor exclusions and the eligible-recipient mask.

An empty eligible branch has an empty indexed columns table and `None` for
its Frame, and no recipient matrix when no ACS recipient is eligible. Complete
diagnostics remain available. The existing Frame selector and matrix codec
require nonempty inputs; the composition does not manufacture a record or zero
to satisfy them. A later model host must refuse fitting an empty branch or make
an explicitly reviewed completion decision.

## Exact identity and lifetime

The complete shared source Frame, its origins and the original preparation's
person origin records must agree in order, source channel and native identity.
Each ASEC qualifier must exactly match the selected ASEC person and native-ID
axes; the pure basis also checks their source ages. ACS anchors must match the
selected original ACS axis. Donor feature and recipient matrix axes are checked
before selection. No sorting, inner join or missing-key fill hides a mismatch.

The donor household design weights are mapped by exact original membership.
Their person mapping must match the original Frame's resolved DESIGN weights;
importance or clone weights are never substituted. This is source construction,
not engine microsimulation aggregation.

Each initial qualifier's complete physical description is sealed as soon as it
returns, and checked while subsequent qualifiers do I/O. After pure composition,
all five original source qualifications are repeated and compared to those
seals. Every returned value, including nullable data beneath missing masks,
is checked throughout. The actual preparation is then rechecked after the last
qualifier's I/O. The final checks are pure: retained issuer identity, nested
original source Frames, allocated/clone support, the exact optional geography
configuration payload and all complete derived seals.

The income-routing qualifier has no standalone physical-seal API, so this
composition uses the existing full table-stamp helper over its complete person
and original-literal tables plus evidence. Existing shared/ACS/interest/dividend
seal APIs are reused directly. No source-reader implementation or new authority
framework is introduced.

## Verification scope

The invented fixture creates closed source bytes before issuance, then executes
the actual survey source graph and source composition. It includes a negative
ACS anchor, malformed adult anchor, under-15 blank anchor, a resolved ASEC
ordinary/retirement-interest route, and excluded ASEC overlap/unknown cases.
Controls check ordered axes, weight kind, adjusted bits, all returned seal
surfaces and copied preparation refusal. A separate injected mutation isolates
the final original-source I/O fence using previously checked descriptive values;
it does not claim another complete source replay.

These checks run without native microdata, QRF, PUF or a country engine. Model
quality, donor conditional support, country financial integration, both-clone
fanout, replay and release acceptance remain later boundaries.

The initial composition acceptance used donor basis `2c959096eec0441e2acd975e63522aac7f1b7016`.
A subsequent primary-source review found that survivor slots 1/2 do not establish
complete absence of estate/trust overlap: the aggregate can include sources 3/4.
The country host must adopt the separate conservative donor-basis correction
before using this branch for fitting. The initial test's eligible donor reports
no survivor income, so its observed values do not rely on a yes-survivor clearance.
