# Native PUF tail fixture contract

`microcosm.build.us_runtime.native_puf_tail` implements the first pure slice of
native tail support. It accepts explicitly invented role projections only. It
does not admit actual PUF source rows, choose a modeled person allocation,
transfer values into a population, assign geography or qualify a release.

The donor selector adapts the union-arm and AGI-only thinning algebra from
[`puf_agi_tail.py` at PR #964 commit f7df78b2a00421f9b90305a9b7db192444075ae0](https://github.com/PolicyEngine/microcosm/blob/f7df78b2a00421f9b90305a9b7db192444075ae0/packages/microcosm-build/src/microcosm/build/us_runtime/puf_agi_tail.py).
This attribution is an immutable algorithm reference, not a claim that the
native implementation has qualified equivalent actual source projections.

## APIs and ownership

- `native_puf_tail_profile()` derives 52 outputs from `PUF55_SURVEY_SS` and the
  existing three-field late-owner matrix. It preserves independent survey Social
  Security and detailed SCF mortgage ownership. It does not use #964's wider
  full65-minus-three projection.
- `declare_puf_tail_role_projection(...)` snapshots typed return/person cells,
  explicit knownness and invented head/spouse/dependent roles into immutable
  bytes. It requires physical person booleans, exact int64 identifiers, finite
  numeric amounts and float64 return DESIGN weights. Unknown cells refuse;
  omitted cells do not become zero. Real source/model provenance modes refuse.
- `projection_tables(...)` revalidates the complete snapshot before returning
  detached tables. Public dataclasses and hashes are descriptive; neither is a
  source-authority capability. A caller may declare invented values, never use
  this contract to authenticate observed people.
- `select_native_puf_tail(...)` accepts two aligned, explicitly declared fixture
  masks for capital-gains support and eligibility. It binds both masks and the
  projection in a deterministic selection identity. It does not derive or
  qualify the capital-gains quantiles.

The AGI arm uses an inclusive $5 million threshold on the declared 20-component
signed additive proxy. This is not calculated AGI. AGI donors require exactly
one head, at most one spouse, representable roles and matching person monetary
sums. Any nonzero dependent money skips that donor, including a donor in both
arms; it is not folded into the head. Dependent Boolean flags are explicitly
ignored, matching the reference rule. Nonzero spouse values require a spouse
when a later population owner constructs the transfer.

Capital-gains-only and both-arm donors remain unthinned. AGI-only support is
capped at 3,000, using deterministic filing-status/rank-decile cells and
midpoint selection. Each cell preserves its float64 return-weight sum exactly.
Receipts report changes in proxy amount sums and weighted proxy sums; amount
mass is not asserted to be conserved. Selected weights are IMPORTANCE support
weights at donor-return grain. They are not household population weights.
Role/monetary eligibility applies to AGI-only and both-arm donors. CG-only
selection inherits the declared mask without claiming full52 person fidelity;
its distinct transfer semantics remain a later-owner responsibility.
Later population integration must resolve the scientific units and source
cohort behind donor-return-to-household normalization, and conserve host
household IMPORTANCE mass. No normalization formula is admitted by this slice.

The current contract bounds declarations to 10,000 invented returns and
100,000 invented persons. These bounds do not authorize an actual source run.
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
