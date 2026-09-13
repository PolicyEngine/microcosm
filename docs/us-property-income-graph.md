# Property income model graph

`microcosm.build.us_runtime.graph_property_income.property_income_nodes`
declares ten numerical nodes: four conditional QRF fits, four chained draws,
one operation that writes the raw draws as columns, and one signed
reconciliation. It uses the existing QRF and reconciliation implementations.
The fragment consumes separately supplied original donor and recipient
populations; it does not qualify native sources or attach results to clones.

The shared columns in `property_income_constants.py` describe ordinary
interest, retirement-account interest, dividends and signed broad property
receipts. The first three are nonnegative; broad property receipts may be
negative. `property_reported_total` is a required predictor and the recipient
reconciliation anchor. The source bridge supplies the ASEC reported aggregate
for donors and the adjusted ACS INTP amount for recipients. It must preserve
their different reference periods and account for unresolved income routes.

The training wrapper requires finite float64 predictors and components,
original DESIGN weights, and a component sum equal to the supplied donor
aggregate. It delegates actual fitting to `LegacyQRFTrainKernel`; it does not
filter, repair or reconcile donor observations. The owning source preparation
must establish original household membership, design-weight mapping, eligible
donor support and source knownness before invoking this fragment. A weight
kind label alone does not establish original source authority.

Recipient draws preserve the complete target order and earlier raw draws as
conditioning inputs. The column operation binds every raw value to its apply
checkpoint, exact recipient index, sibling producer, and final model history.
The graph executor supplies the authenticated artifact edges. Direct calls or
copied checkpoint bytes do not authorize source or model admission. Person IDs
remain int64 independently of the pandas index used by QRF checkpoints.

Reconciliation retains raw draws in separate `draw_` columns and exposes each
component's adjustment, active bound, residual and objective. Scales and
tolerances are required parameters. Changing them recomputes reconciliation;
changing recipient anchors recomputes recipient draws while reusing donor fits.
Negative or zero net income never forces all positive components to zero:
positive interest and a property loss can coexist at either total.

Missing anchors and under-15 source blanks require an explicit upstream
decision; this fragment does not manufacture zeros. Original ASEC observations
are not reconciled to ACS respondents. The next country integration must join
the complete source-qualified basis, draw once per original ACS person, retain
the result on both initial clones, and derive the declared tax splits before
running the dependent PUF successor. Retirement-account earnings are not
ordinary taxable/exempt interest. Broad property receipts are not yet a
rental-income tax leaf. PUF-owned alternatives must not subsequently be forced
back onto the survey anchor.

The initial invented test uses 44 donor persons and four recipients, including
negative/zero totals, large int64 identities, original membership, metadata and
a zero-weight household. It executes the real twelve-node graph including two
source nodes, compares draws with direct weighted QRF, compares reconciled
values with the pure projection and reopens the store in required-cache mode.
Twenty-two tests also cover delegated training-code identity, dependency reuse, incorrect weight kinds,
unknown/inconsistent donors, altered draw history and invalid declarations.
They establish numerical and graph contracts, not native source acceptance,
statistical adequacy, tax correctness or release readiness.
