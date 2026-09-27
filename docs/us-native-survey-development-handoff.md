# Native survey development handoff

`tools/build_us_fiscal_refresh_release.py` now exposes
`prepare_native_survey_development_input(run, checkpoint_directory)`. Its input
is a **retained, issued `SurveyEnrichmentRun`** in the same process that ran
`run_us_survey_enrichment`. The output is a `NativeSurveyDevelopmentInput`
containing a usable `Frame` and a machine-readable coverage report.

```python
from tools.build_us_fiscal_refresh_release import prepare_native_survey_development_input
from microcosm.build.us_runtime.native_survey_handoff import (
    load_native_survey_development_checkpoint,
)

# enriched_run remains alive from the native PUF/enrichment continuation.
result = prepare_native_survey_development_input(
    enriched_run, "/owned/new/native-development-checkpoint"
)
frame = result.frame
missing_inputs = result.report["missing_inputs"]
remaining_gates = result.report["required_release_evidence"]

# Validated reuse while the original issuer remains alive:
reloaded = load_native_survey_development_checkpoint(
    "/owned/new/native-development-checkpoint", run=enriched_run
)
assert reloaded.owner_live_verified
```

The writer creates a fresh directory containing `population.h5` (the maintained
Frame checkpoint format) and `handoff.json`. It performs complete readback
comparison: entity tables and row order, schema/linkage, dtypes and knownness,
non-owned fields, geography, strata, typed weights and mass history. Frame
metadata uses the graph store's typed metadata encoding. Nullable values follow
the maintained replay comparator's documented null backing-buffer normalization;
known values and masks retain their meaning. No additional support clones,
weights, default inputs, fitted values or legacy assembly receipts are created.
The final owner check must succeed before the API returns. Existing output
directories are refused.

The loader also works without `run` for downstream development inspection. In
that case `owner_live_verified` is false. Byte identities bind the two files;
they do not issue source authority. The Frame checkpoint is not a
`USSingleYearDataset` H5 or simulation-ready multispine pool. Do not send it to
`--base-h5` or `--pool-manifest`. Its Frame can be used by existing Frame-based
consumers after those consumers' own input checks, while the source run is kept
alive whenever source qualification is required.

The separate `prepare_native_survey_engine_input(run, declaration=..., consumer=...)`
entry selects an exact, closed projection from a live owner. Its optional
`PolicyEngineUSEngine` consumer calls `validate_input_representation(frame,
period=...)` against that adapter's selected live registry and country source
declarations. This checks owning entities, selected-year formula ownership,
annual/eternity periods, actual enum names, knownness and physical dtype
compatibility, including lossless integer IDs and memberships. It copies the
entity tables and materializes the typed household weight for inspection; it
does not apply defaults, rewrite inputs, construct a dataset or calculate.

Success adds only `consumer_representation_compatible: true` to the projection
report. Runtime admission, source applicability, SPM universe qualification,
scientific gates, simulation readiness and release eligibility remain separate.
Ordinary float32 rounding and representable infinities are not scientific
certification; unsupported nulls, integer overflow, finite-to-infinite conversion
and string truncation refuse without filling or remapping. `UNRESOLVED` can be
a valid enum representation while still preventing SPM measurement. Omitting
the consumer preserves the previous import-free, unqualified projection path.
Both paths retain the final live-owner and exact-storage comparisons.

The builder's private `_calibrate_native_input_frame` bridge can then solve
already-materialized targets with the maintained dense or L0/refit calibrator.
It requires an explicit formula-ownership provider and a closed solve-only
option set; warm starts, checkpoints and exact-k identities are not accepted.
Numerical option validation remains with the maintained calibrator; this bridge
does not define another solver parameter policy.
Before solving, the target frame must retain every projected input column,
structural ID, membership, row order, stratum and typed initial weight. Target
materialization may add calculated columns; those columns never enter the
exported input frame.

The bridge works on detached copies, attaches weights using the maintained
helpers and verifies the result against its supplied parent. It reconstructs a
complete float64 weight vector in original household order. L0 keeps exactly
the maintained selected support, including zero post-refit weights, without a
second pruning threshold. The returned `comparison_specification` records the
exact solve options, registry and loss-weight identities, both frame identities,
and the supplied specification digest. The existing comparison `binding` hashes
those specification bytes, so the pair must be retained together. Target
materialization may have different metadata and mass history; the exported
input frame preserves its input parent's metadata and mass-history prefix.
Callback changes to caller configuration cannot
rewrite those captured inputs; changes to either supplied frame refuse.
The returned measured result and comparison binding do not issue native owner
authority, establish target/scientific qualification or authorize publication.

The report preserves the owner's receipt and source content identities, the
native population version and graph manifest identity, and the amount model's
source/draw flags. Its default inventory merges the maintained release-source
column roster and historical input profile, retaining each named missing or
unknown-containing input and its declarations. `input_inventory_scope` records
these two checked rosters and the historical profile's source-manifest identity.
The legacy pool input surface is **not checked** and the live take-up ABI
inventory remains **unqualified**. The handoff does not import the legacy pool,
whose eager agreement/take-up registries instantiate the country engine. This
descriptive inventory does not establish a complete engine input contract;
historical take-up names are names only, and missing fields without a declared
grain retain `entity=null`. Presence is not proof of
applicability, source validity or non-default statistical signal. An optional
`export_contract=` accepts the consuming adapter's actual `ExportContract` and
reports missing, forbidden, formula-owned and unexpected columns without
projecting or default-filling the Frame. No country engine is imported or run by
this default handoff. The explicit engine-consumer projection described above
retains its separate runtime qualification. Release gates, source/applicability
checks and `simulation_ready=false` / `release_eligible=false` are unchanged.

The approved native scope excludes the legacy prior-year income family:
`employment_income_last_year`, `self_employment_income_last_year` and
`previous_year_income_available`. The handoff does not generate, fit or require
these as native completion targets. Existing columns, if any, remain preserved
for inspection. The [input-coverage diagnostic](us-input-coverage-diagnostic.md)
now offers an explicit `USInputProfile.NATIVE_NATIONAL_CD` name profile with
159 inputs: it excludes the two prior-year fields still in the 161-name
national/CD roster; prior-year wages were already absent. This resolves only
the name-profile selection. Native scientific/source gates and historical
release requirements still need separate reconciliation; the handoff continues
to name that outstanding gate.

## Follow-on release integration

This closes the live-owner to exact development-Frame/checkpoint seam. Release
admission is still unsupported. Its concrete successors are:

1. Attach and qualify each missing input named by this candidate's report. The
   native hours source/transport work, SPM construction/role path, and remaining
   source families must become part of the receiving owner's checked lineage.
2. Bind applicability and source-signal gates to that owner, including the
   usual-hours gate, and reconcile the prior-year-income scope. The current
   report's `release_eligible=false` is descriptive, not an irrevocable decision
   about a future qualified owner.
3. Supply the selected engine's input projection and country runtime evidence.
   The full checkpoint deliberately preserves fields that may be forbidden or
   formula-owned in an engine export; drop them only in the owned projection.
4. Pass the resulting checked Frame directly into maintained target
   materialization and common-population calibration. Replace the native
   admission branch with checks for that actual evidence; do not route through
   legacy assembly/clone stages or mint their receipts.
5. Run the maintained Chronicle/model, national/CD target fit, export
   readback/input-mass/target parity, reform smoke, and matched incumbent/default
   improvement gates. The builder's existing exact-k arm additionally expects
   its agreement and ladder provenance, which this handoff does not fabricate.

Serialization proves none of those scientific or publication gates. This entry
writes no release manifest, certification or publication pointer.

## Development base and verification

This continuation starts from `5258a4c7f6d05cb1e68806d51b38996ea116bfb5` after
fetching and inspecting `origin/main` on September 19, 2026. Main lacked the
native `SurveyEnrichmentRun` API and associated owners; the branches diverged by
41 upstream commits and 407 continuation commits. Work occurs in a separate
worktree. The active native measurement source and harness are untouched.

Lightweight invented tests cover full storage readback, missing-input and
engine-contract inventories, unissued-owner refusal, changed artifacts and
output reuse. The genuine-owner integration test reuses the existing enrichment
suite's actual issued owner; run it with the compatible continuation source,
not against a frozen active measurement tree. No native data or country model
calculation is needed by the lightweight suite.

### Optional veterans’ benefits development route

The retained native enrichment host accepts `groups=("veterans_benefits",)` with
`full_original_amount_donors=True`. Existing default groups and donor selection
remain unchanged. This route qualifies `VET_VAL` and `VET_YN` from the exact
retained 2025 ASEC person member for 2024 income. Positive “yes” answers and
explicit “no” plus zero are known; NIU, recipient zero, missing answers and
contradictions remain unknown. Age 15 is the lower reporting boundary, without
an upper-age, veteran-status or VA-healthcare filter.

`I_VETYN` and `I_VETVAL` retain their literals, published codes, conditional flag
universes and allocation status. A zero flag outside its universe does not
establish an unallocated observation. Allocation provenance does not create
income zeros or discard otherwise known publisher-allocated amounts. The
six-digit veterans amount range does not change the five-digit UC/WC ranges.

The model uses full-original current ASEC DESIGN donors with qualified amounts
and predictors, including donors absent from the selected receiving support.
It draws once per eligible original ACS person and carries that draw to both
existing clones. ASEC originals and their clones retain the qualified source
amount and unknownness. The ASEC2025/income2024-to-ACS2024 transfer is an explicit
development assumption; held-out quality, donor adequacy and release equivalence
remain unverified.

When veterans is selected, `survey_amounts.canonical.version` keeps all receiving
support after prior enrichment, and `survey_amounts.canonical.attach` writes only
selected veterans and child-support canonical outputs. Child-only runs retain
their existing node IDs. An enabled canonical-state stage follows this version.
Carried canonical values establish no source authority: supported float32/float64
storage is replaced from qualified source/model values with exact-loss checks;
unsupported storage and unrelated ownership collisions refuse. The parent source
branch, other columns, entity axes, memberships, weights and ledger are preserved.

Invented-source and tiny real-fit/replay tests qualify this component boundary.
They do not establish full financial/PUF/enrichment-owner acceptance, a completed
native input profile, actual-source quality or publication eligibility.
