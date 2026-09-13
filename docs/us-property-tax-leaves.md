# Property tax leaf rebase

`graph_property_tax_leaves.py` provides three deterministic graph nodes for an
already checked population with property components and incumbent tax leaves.
It does not qualify sources, fit models, attach clones, grant a checked-run
handle or wire the country/PUF host. The owning host verifies the original
component observations, clone pairing and complete frame before using it.

## Numerical operation

`split_property_tax_leaves(person)` returns four float64 columns indexed by exact
int64 `person_id`. It imports the maintained fractions from `cps_carried`:

| Input component | Primary leaf | Complement leaf |
| --- | --- | --- |
| `property_ordinary_interest` (`O`) | `taxable_interest_income = O * 0.680` | `tax_exempt_interest_income = O - taxable_interest_income` |
| `property_dividends` (`D`) | `qualified_dividend_income = D * 0.448` | `non_qualified_dividend_income = D - qualified_dividend_income` |

The complement uses subtraction after computing the primary. This is a declared
operation order, including for the dividend complement that legacy code computes
by multiplying the complementary fraction. These fractions are modelling
assumptions; they do not establish a person's observed tax treatment.

Interest and dividend knownness are separate. A finite nonnegative input produces
its two leaves; a missing input produces two NaNs, replacing any earlier tax
predictions. Zero stays an observed numerical zero. A negative or infinite O/D
input, an infinite retirement component, incompatible dtype or ambiguous person
identity refuses. `property_retirement_interest` is retained as an auxiliary
component and never enters either tax-interest leaf. Its missing value does not
invalidate known O/D values. Broad property receipts, net anchor sign, source age,
donor-fit eligibility and earlier predictions never supply missing O/D values.
No under-15 zero completion or missing-adult prediction occurs here.

## Three graph nodes

Call `property_tax_leaf_nodes(frame, population=base_version,
projection=projection_input, reconciliation=reconciliation_input, atol=...,
rtol=...)`. Both tolerances must be explicit finite nonnegative Python floats.
The two `ArtifactInput` aliases must be `projection` and `reconciliation`; the
host supplies their actual producer IDs and nominal types. Register
`PropertyTaxReceivingKernel`, `PropertyTaxLeavesKernel` and
`PropertyTaxLeafGateKernel`.

1. `survey_property.tax_receiving` opens a new population version using
   FILTER-all. This inherits the prior owners and permits explicit replacement of
   the four incumbent leaves. The executor adds a real FILTER conservation record
   to the population mass ledger. It preserves the existing `Frame.mass_log`.
2. `survey_property.tax_leaves` replaces the four leaves on that version and
   emits `diagnostics` of type `microcosm.us.property_tax_leaf_diagnostics/v1`.
3. `survey_property.tax_leaf_gate` recomputes the leaves and their knownness,
   checks exact leaf bits, partition sums within the declared tolerances, and the
   complete diagnostics payload. It emits `verification` of type
   `microcosm.us.property_tax_leaf_verification/v1`.

The gate binds ordered person IDs without floating-point conversion, current
O/D/retirement bytes, all four leaf byte digests, fractions and operation order,
individual knownness, and the typed projection/reconciliation artifact keys and
payload hashes. It includes every physical person, including dependents and
zero-weight members. `complete` is false if any person lacks O or D; a successful
numerical check with missing inputs reports `evidence_absent`. Complete numerical
inputs report `pass`. Neither outcome establishes source, model, tax-validity or
release authority. The artifact explicitly leaves full-frame verification to
the host.

## Supported receiving shapes and host responsibilities

`Frame.select(all)` currently refuses link tables and normalizes group indices.
The declaration factory therefore refuses links, nondefault group indices,
orphan groups, empty person frames and groups lacking a readable non-ID column.
The last condition ensures the kernel context exposes group IDs for the
receiving check. Orphans are already invalid at Frame construction; the factory
also detects a later membership mutation. The kernel rechecks exposed group
membership and logical group indices immediately before returning its keep mask.
It never silently prunes a group or switches structural operation.

For supported frames, person rows/order/IDs, entity tables, weights and their
kinds, schema, strata, metadata and frame mass log are preserved, apart from the
four declared leaf replacements. The new population version, owners and
conservation record are intentional changes. Kernel contexts do not expose the
complete schema, metadata or population ledger, so the host must retain and
verify those independently. It must also qualify the complete supported shape
before adding these nodes, bind actual artifact producers and stored payloads,
and recheck the original owners and output after its final relevant I/O.

For independent expected-result reconstruction, the host can call each kernel's
`run(KernelContext)` on the retained current population and authenticated typed
artifact aliases. The existing executor `_project_context` and `_apply_result`
provide the canonical context projection and result application. FILTER results
must be materialized through the validated keep mask before `population.patch`;
passing its raw `keep` result directly to that lower-level patch function omits
the executor's frame selection. Ordinary rebase/gate results use the existing
patch path. These deterministic kernels consume no random draws. Expected
artifact bytes and receipts can then be compared with the actual store and node
states, and expected full populations with the host's retained observations.
This fragment adds no separate executor, authority registry or public run issuer.

Future PUF integration must depend on this verification artifact and require
complete tax leaves over every physical member the PUF consumer needs. The
fragment does not add that host dependency itself. Matching checked inputs give
matching outputs on paired clones; the fragment does not infer or repair clone
pairing. Capital gains retain their earlier conditioning and are not rebased by
this change.

## Verification scope

The 32 invented controls exercise a real four-node graph including its source,
ContentStore cold execution and required-cache replay. They cover negative,
zero-net and positive anchors with loss offsets, independent O/D unknowns,
known O/D with unknown retirement/broad components, integer IDs above `2**53`,
row permutations, empty pure input, malformed values, altered leaf bits,
diagnostics and artifact bindings, full-frame preservation, zero-weight records,
the population conservation record and unsupported receiving shapes. No native
survey, country engine, PUF execution, fitting or publication is part of this
acceptance.
