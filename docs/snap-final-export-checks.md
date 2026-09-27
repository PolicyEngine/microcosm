# SNAP final-export checks

`snap_export_acceptance` in `microcosm.build.us_runtime.snap_release_acceptance`
checks an explicitly identified final H5. It returns individual check statuses,
not an overall release certificate. It is not enabled in the builder by default
and has no CLI.

The interface requires the H5 path and expected SHA-256, an explicit nonempty
`ExportContract`, integer model year, US `TargetRegistry`, and typed
`TargetDiagnostic` rows. Optional `TargetFitRequirement` objects carry existing
reviewed fit decisions. A file hash identifies bytes; it does not authenticate
source custody, calibrated-weight issuance, the registry, or permission to
publish. The release owner must bind those authorities independently.

The wrapper requires exactly 51 state rows (50 states plus DC) for each of
`snap_households` and `snap_total`, following the current fiscal-target
materializer metadata. It checks diagnostic identities, target values and error
arithmetic. The final estimates come from a simulation reopened from the H5 by
the maintained `default_simulate_factory`, with the release SPM selection. The
interface accepts no precomputed simulation totals or caller-written passing
summary. Input-stage take-up fit and saturation never waive final-export misses.

## Count and period definitions

The current `fiscal_targets.py` mapping uses FNS fiscal-year average monthly
participating households. Its model counterpart is the weighted count of SPM
units with positive annual `snap`: the boolean is evaluated at the SPM grain
before summing. It is not a count of physical households with any recipient or
an annual-ever participation measure. Two positive SPM units within a household
count twice at that household's calibrated weight, regardless of how many
persons belong to either unit.

The benefits counterpart is the weighted sum of annual modeled `snap` at the
same SPM grain. The result records each target's source period, source citation
and model period separately. It does not relabel an older source observation as
a current-year observation or resolve the fiscal-year/calendar-year difference.
Both calculations retain `MicroSeries` weights and use `map_to="spm_unit"`.
Simulation SPM IDs and order must equal those read from the export.

The [FNS FY2022 eligible-person participation comparison](https://www.fns.usda.gov/research/snap/state-participation-rates/2022)
is explicitly advisory and pending. This slice does not import the old PR's
rate table or infer eligible people from positive-SNAP SPM-unit counts. A future
comparison needs a verified source table and an aligned eligible-person and SNAP
assistance-unit universe; SPM membership alone does not supply that universe.

## Tolerances and artifact checks

The wrapper reuses each `TargetSpec.tolerance` as an absolute tolerance and any
matching `TargetFitRequirement.max_abs_relative_error`. If both apply, both must
hold. A row with neither stays pending, even at zero measured error. Existing
fit requirements retain their minimum-match checks. The wrapper supplies no new
statistical tolerance and does not reuse the input-stage take-up tolerance as a
final-export tolerance.

The H5 passes the maintained denied-pool ingress boundary. Checks cover the
export contract, absence of persisted `snap`, entity IDs and memberships,
nonempty groups, nonnegative finite household weights, the model year,
nonmissing county/state FIPS, nonzero county suffixes, matching county/state
prefixes, and positive-weight support in all 51 areas. County codes are not authenticated against
a particular vintage's county roster. The file hash is checked again after all
engine calculation and teardown I/O, including when teardown raises. A changed file or failed teardown invalidates computed fit.
Missing engine dependencies or an absent final artifact stay pending; malformed
available evidence produces a failed check. Consumers must inspect every check,
including failures alongside pending checks.

## Validation and remaining proof

The flat invented suite writes genuine small H5 files with the maintained writer.
Its explicit test-only weighted-series double delegates sums to typed `Frame`
accounting. It exercises wrapper I/O and aggregation semantics, not actual
PolicyEngine formulas or a genuine release. It includes two SPM units in one
household, a multi-person SPM unit, missing/duplicate/mismatched targets,
malformed geography, stale diagnostics, missing and conflicting tolerances,
unweighted engine output, changed SPM order, and mutation during calculation or
teardown.

Before release qualification, run this wrapper on the exact final exported H5
with its authenticated release owner, exact hash, real consumer export contract,
model year, corresponding registry and diagnostics, and reviewed tolerance
decisions. Use the pinned PolicyEngine runtime to recompute all 102 state
estimates. Resolve every failed/pending required check and preserve the report
with the artifact and runtime identities. Separately retain the full release's
source, graph, calibration and publication checks. No actual population, model
execution or release approval was performed by this implementation slice.

This adapts the final-artifact intent of
[Microcosm PR #413](https://github.com/PolicyEngine/microcosm/pull/413), reviewed
at `c538c81d46661fe2d20876391ea1a4371d14cd8d`. Its old source/feed pins,
10% defaults, caller-supplied totals, manifest-summary approval and old-build
failure lists are not part of this interface.
