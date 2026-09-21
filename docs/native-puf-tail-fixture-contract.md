# Native PUF tail fixture contract

`microcosm.build.us_runtime.native_puf_tail` implements the first pure slice of
native tail support. It accepts explicitly invented role projections only. It
does not admit actual PUF source rows, choose a modeled person allocation,
transfer values into a population, assign geography or qualify a release.

The donor selector adapts the union-arm and AGI-only thinning algebra from
[`puf_agi_tail.py` at PR #964 commit f7df78b2a00421f9b90305a9b7db192444075ae0](https://github.com/PolicyEngine/microcosm/blob/f7df78b2a00421f9b90305a9b7db192444075ae0/packages/microcosm-build/src/microcosm/build/us_runtime/puf_agi_tail.py).
This attribution is an immutable algorithm reference, not a claim that the
native implementation has qualified equivalent actual source projections. It
participates in the selection identity deliberately: an edit to the reference
this algebra came from is a change of provenance and must move the digest.
Two behaviours deliberately diverge from that reference and are marked below:
the residual placement in exact cell rescaling, and the arithmetic that decides
the AGI floor.

## APIs and ownership

- `native_puf_tail_profile()` derives 52 outputs from `PUF55_SURVEY_SS` and the
  existing three-field late-owner matrix. It preserves independent survey Social
  Security and detailed SCF mortgage ownership. It does not use #964's wider
  full65-minus-three projection.
- `declare_puf_tail_role_projection(...)` snapshots typed return/person cells,
  an all-known witness and invented head/spouse/dependent roles into immutable
  bytes. It requires physical person booleans, exact int64 identifiers, finite
  numeric amounts stored at exactly int64 or float64, and float64 return DESIGN
  weights. Unknown cells refuse; omitted cells do not become zero. Real
  source/model provenance modes refuse.
- `projection_tables(...)` revalidates the complete snapshot before returning
  detached tables. Public dataclasses and hashes are descriptive; neither is a
  source-authority capability. A caller may declare invented values, never use
  this contract to authenticate observed people.
- `select_native_puf_tail(...)` accepts two aligned, explicitly declared fixture
  masks for capital-gains support and eligibility. It binds both masks and the
  projection in a deterministic selection identity. It does not derive or
  qualify the capital-gains quantiles.

## Admitted cells

Filing-status codes come from the repository's existing PUF filing-status
authority, `puf_support._FILING_STATUS_CODES`, whose integer-keyed inverse view
`puf_capital_gains_tail._FILING_STATUS_BY_CODE` carries the same five entries.
All five are admitted, including `5 = SURVIVING_SPOUSE`; this contract does not
maintain a second local enumeration. The admitted domain appears in the receipt.
It is a validation gate rather than a selection parameter -- an out-of-domain
code refuses the whole declaration -- so it is not folded into the selection
identity, whose filing-status content already travels in the projection digest.

Money is stored at exactly int64 or float64. Narrower floats, narrower or
unsigned integers and non-native byte order refuse at declaration and again at
snapshot restore, rather than being reinterpreted at a precision this contract
has not demonstrated. Knownness is all-known only: a partially known
declaration refuses, so the knownness bytes are a constant all-true witness
that carries no information beyond the row count. Admitting partial knownness
would require threading the caller's real mask through the snapshot first; the
constant must not be read as measured knownness in the meantime.

## The AGI arm

The AGI arm uses an inclusive $5 million threshold on the declared 20-component
signed additive proxy. This is not calculated AGI. That 20-leaf list is the
donor-side build-period income locator and matches the reference selector's own
donor proxy; the seven-column `_RECIPIENT_AGI_PROXY_COLUMNS` in the sibling
`puf_capital_gains_tail` is a recipient-side SOI-band locator for a different
operator. Neither list substitutes for the other.

The floor decision does not route through a float64 total, which is the second
deliberate divergence from the reference. Each row's int64 components are
summed as Python integers, exactly at any magnitude; its float64 components are
summed with `math.fsum`, which returns the correctly rounded sum of the doubles
it was given -- a stable summation, not universally exact real arithmetic. The
decision then compares that float part against the exact integer remainder of
the floor, which is an exact comparison, so an int64 component above 2**53
cannot cross the floor by rounding. The float64 proxy total that the receipt
masses and the rank-decile order use is a diagnostic view of the same row; it
is labelled as such in the receipt and is not the threshold authority.

AGI donors require exactly one head, at most one spouse, representable roles and
matching person monetary sums. Those person sums are reconciled over Python
scalars, so signed integer reconciliation cannot wrap or round; they are also
person-row-order independent, because an eligible donor has exactly one head,
at most one spouse and exactly zero dependent money, leaving at most two nonzero
values in any column. Any nonzero dependent money skips that donor, including a
donor in both arms; it is not folded into the head. Dependent Boolean flags are
explicitly ignored, matching the reference rule.

`SelectedTailDonor.spouse_payload_nonzero` describes the declared cells and
nothing else: it reports whether any declared spouse cell of a role-screened
donor is nonzero. It is `None` for capital-gains-only donors, which are outside
the role-eligibility scope and were never examined -- unexamined is not the same
as examined and found zero. It never consults `filing_status_code`, and a joint
return with an all-zero spouse row is not a finding that the return needs no
spouse. Which household a donor becomes, and whether that household has a
spouse, remains entirely the later population owner's decision.

## Arms, thinning and weights

Capital-gains-only and both-arm donors remain unthinned and uncapped. AGI-only
support is capped at 3,000, using deterministic filing-status/rank-decile cells
and midpoint selection. Cell quotas assert their own postconditions -- the
quotas sum to the budget, every cell keeps at least one donor, and no cell keeps
more than it had. Those are inexpensive invariants; no counterexample to them is
known in this implementation.

Each thinned cell preserves its float64 return-weight sum exactly. The residual
correction is applied to a largest weight in the cell, which is the first
deliberate divergence from the reference: the reference forces the residual onto
the last row, and a cell with dispersed weights can have a last row orders of
magnitude smaller than the correction, which either drives that weight
non-positive or leaves the correction unable to move the sum at all. Positivity,
finiteness and exact conservation stay mandatory; a cell mass that still cannot
be represented refuses rather than being approximated.

Receipts report changes in proxy amount sums and weighted proxy sums; amount
mass is not asserted to be conserved, and those totals are float64 diagnostics,
not an exact amount reconciliation. Selected weights are IMPORTANCE support
weights at donor-return grain. They are not household population weights.
Conditional support selection does not retain DESIGN population semantics even
for a numerically unchanged weight, so every selected weight is IMPORTANCE;
each donor additionally carries a `weight_treatment` recording whether its
value was left unchanged or rescaled inside a thinned cell, and the receipt
totals both. A later owner reconciling design mass must read that field rather
than assuming the whole selection was resampled.

Role/monetary eligibility applies to AGI-only and both-arm donors. CG-only
selection inherits the declared mask without claiming full52 person fidelity;
its distinct transfer semantics remain a later-owner responsibility. The receipt
reconciles every arm: AGI candidates equal selected plus skipped plus thinned,
capital-gains candidates equal selected plus skipped, and both-arm candidates
equal selected plus skipped. A capital-gains donor can only be skipped when it
is also an AGI candidate, because only the AGI arms are role-screened.

Later population integration must resolve the scientific units and source
cohort behind donor-return-to-household normalization, and conserve host
household IMPORTANCE mass. No normalization formula is admitted by this slice.

The current contract bounds declarations to 10,000 invented returns and
100,000 invented persons. Those two bounds, the 3,000 AGI-only cap and the
inclusive 5,000,000 floor are declared development bounds: this slice states
them, and does not derive them. A source run must establish their basis
separately. They do not authorize an actual source run.
The receipts always state `source_admission_issued=false` and
`release_eligible=false`. Return/person values remain local fixture evidence.

## Qualification still required

Actual source-to-role ownership, stable source/growth identities, a reviewed
person-allocation method, structural EXPAND, full-spine geography placement,
late-owner replay and actual qualification remain separate work. The native
return-only donor envelope and its technical person rows are not observed PUF
people. Do not relabel them or import the legacy wrapper's rows as authority.

Administrative AGI distributions may supply independently admitted calibration
targets. Reform scores are diagnostics, not an optimization objective for this
selector. Survey poverty is comparison-only: do not tune weights, imputations,
take-up, candidate selection or target deferrals toward a survey poverty rate,
and do not gate release on proximity to that rate. Independent definition,
implementation and administrative-target correctness checks still apply.
