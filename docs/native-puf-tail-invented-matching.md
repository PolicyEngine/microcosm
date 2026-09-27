# Invented native AGI tail matching

`us_runtime.native_puf_tail_matching.match_invented_native_puf_tail` connects the
tested native selection algebra to a unique household assignment and the real
structural EXPAND declaration. It is a development contract over invented
inputs. It does not authenticate an actual PUF person allocation, housing
domain, recipient observation, or receiving host.

The API calls `select_native_puf_tail` itself. It accepts AGI-selected and
overlap-selected donors; any selected capital-gains-only donor refuses the
whole request. Source-adjusted AGI is a separately named input for exactly the
selected donor IDs. It is not the selection proxy. Donor and recipient inputs
declare equal annual USD income-year and price-year bases. This equality does
not establish observed equivalence of survey reporting windows.

Every clone1 person has an explicit known role, age and all seven maintained
recipient proxy components. Every clone1 tax unit has a known filing status.
The complete household-domain roster must agree with live source coordinates.
All six entity provenance tables and five memberships are checked, including
household closure. Missing or contradictory required evidence refuses; no
unknown value is converted to zero. The source/domain/status/role declarations
remain invented even when these checks pass.

Candidate occupied-housing parents must have one tax unit, exactly one head,
at most one spouse, eligible head/spouse ages, and positive household
IMPORTANCE mass with exactly representable equal halves. GQ, zero mass,
incompatible roles and unrepresentable halves have named exclusions. Extra
stored non-household weights refuse.

The head/spouse age floor of 15 deliberately follows the reference matching
method and reuses the ACS earnings-universe constant. This is a support
compatibility choice, not a tax-unit definition or evidence of observed roles.

Matching follows the reference implementation at commit
`f7df78b2a00421f9b90305a9b7db192444075ae0`: stable source-coordinate order before
seeded priorities, donors requiring a spouse first, donor-ID order, the same
filing status, then the nearest AGI band with lower-band tie breaking. Here
"requiring a spouse" means the tested declared projection has a nonzero spouse
payload; it does not imply observed household composition. Head-only payloads
can match a recipient with or without a spouse. No dependent-person amounts
are transferred by this API.

Capacity counts unique compatible parents, including spouse-specific demand.
An insufficient filing-status group is left wholly unassigned and recorded.
Another independently supported status can proceed. No parent is reused and
no cross-status fallback exists. An empty result has no EXPAND declaration.

The reference implementation's absolute donor-weight/household-weight
constraint is deliberately absent. The approved support-expansion policy
gives clone1 and clone2 one half each of the selected parent's household
IMPORTANCE weight. Donor support weights remain separately typed selection
metadata. They do not become household mass, and no claim of legacy #964
weighting equivalence is made.

The immutable receipt binds the selection, declared source AGI, comparison
basis, recipient inputs, provenance/membership/weight snapshot, seed,
candidate/exclusion lists, capacity counts, role coordinates, assignments and
structural declaration. It is descriptive: `source_admission_issued`,
`matching_qualified` and `release_eligible` are false. The matching receipt is
not yet an issued graph artifact or replay capability. The EXPAND independently
checks its live structural inputs but does not validate this matching receipt.
The role coordinates name clone1 parent records; newly minted clone2 IDs must
be obtained from the actual EXPAND lineage before any later monetary placement.

Invented tests cover selection, matching, refusals, capacity, role handling,
determinism and cold/required execution of the real six-entity EXPAND.
Test source alone is not a passing result; execution evidence is recorded
separately. Actual
person-projection qualification, actual housing-domain authority, native52
amount placement, variable-clone receiving joins and a pre-geography host
remain separate work. Atomic geography belongs after the completed spine.
Administrative AGI support informs this method; poverty is comparison-only and
does not choose assignments, weights, exclusions or release decisions.
