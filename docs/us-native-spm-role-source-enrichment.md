# Native SPM role source-enrichment release

This lane creates a **new H5** from the reviewed BuildP population by adding
`is_spm_independent_minor_role` to its native person table. It performs no
calibration, population aging, weight adjustment, geography assignment, or
membership reconstruction. The country and wrapper use their existing H5
loaders. The keyed CSV is immutable source evidence, not a runtime join.

The supported parent is
`populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z`, H5 SHA256
`48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e`.
The data contract pins its original release/build manifests, schema-5
calibration diagnostics, and source coverage evidence by SHA256. This is
explicit inheritance of those measurements. It does not upgrade diagnostics
to schema 6 or assert that they measured a new model's results. Ordinary
calibration releases still require schema 6. A different parent requires a
separately reviewed contract; there is no caller-supplied legacy-schema waiver.

The release type has one other reviewed operation: the reported-receipt child
of the national default (`add_reported_receipt_inputs`, pinned to
`populace-us-2024-spm-20260915`). It shares every gate below. It has its own
pinned parent, evidence and native inputs, and it never moves `latest.json`.
`source_enrichment.json`'s `operation` selects the lineage, and any other
value is judged, and refused, as this lane. See
[the reported-receipt runbook](us-reported-receipt-source-enrichment.md).

## Source reconstruction and exact preservation

`spm_role_source.py` reuses Microcosm's existing Census ASEC archive/member
pins from `education_assistance_source.py`. The complete source person CSVs
are `pppub23.csv`, `pppub24.csv`, and `pppub25.csv` (income years 2022–2024).
The release gate independently checks each survey year's exact archive SHA256,
official URL, person member and income year, alongside its existing CSV pin.
Rehashing provenance and enclosing manifests cannot authorize another archive.
The source role is:

```python
(SPM_HEAD == 1) | (A_FAMTYP.isin([1, 4]) & A_FAMREL.isin([1, 2]))
```

The stored Boolean is the role **before** the age gate. Reconciliation uses
`age >= 18 OR (age >= 15 AND role)` to reproduce `SPM_NUMADULTS`,
`SPM_NUMKIDS`, and `SPM_NUMPER` for all 176,039 source units and 59,900 native
BuildP units. This is an inference from documented relationships, exhaustively
checked for these sources; it is not a claim to possess the Census production
program. New source vintages must pass reconciliation independently.

The join uses income/source year and exact 22-digit string `PERIDNUM`.
Repeated source people across support clones are allowed; missing matches,
ambiguous source IDs, repeated people within one native SPM unit, partial
source units, and inconsistent raw fields are refused. Older missing
relationship cells remain missing. Existing model ages must equal observed
source ages. No weights enter the reconciliation.

BuildP uses a PyTables compound dataset at `/person/table`. Adding a native
column necessarily extends that compound type; a new unrelated H5 group would
be ignored by the existing country loader. The only permitted physical schema
changes are:

- Append one HDF bitfield byte (pandas Boolean) after every original record's
  bytes. All original field names, order, offsets, types, shapes, and bytes
  remain exact, including NaN payloads and signed zero.
- Append the new column's registration to the four `/person` attributes
  `data_columns`, `values_cols`, `non_index_axes`, and `info`, and add its five
  PyTables field attributes. Every existing registration entry remains intact.

All other H5 groups, datasets, indexes, types, shapes, attributes, and bytes
are compared against the actual parent. Thus logical pre-existing variable
identity is exact; the enclosing compound datatype is explicitly extended.
The builder copies the parent and does not repack it, so the new file can
retain unused space from the replaced person table. New HDF object timestamps
are disabled for deterministic output within the recorded runtime.

The generated three-column evidence CSV must also match independently reviewed
SHA256 `22b5968d90fecfeef7614583e493fe10cc16bda8b5be82e6f49a5bc2102d3ce5`.
The CSV is never used to derive the role. Parent and source evidence are copied
unchanged into the release bundle; H5 and evidence files are read-only on exit.
The H5 belongs only in `artifact_root`, for upload at its manifest-declared root
path. A same-named entry in the release directory is rejected before compatibility
probing or Hub client activity, including identical copies and symlinks: either
would otherwise change the publisher's upload destination.

## Build and validate a local candidate

Use the repository's installed environment. These commands do not fetch data,
calibrate, or publish. Populate the source cache through the existing pinned
Census source acquisition pipeline, or supply its directory explicitly.
The parent evidence directory contains the original `release_manifest.json`,
`build_manifest.json`, `calibration_diagnostics.json`, and
`us_source_coverage.json`.

```bash
python tools/build_us_spm_role_enrichment.py \
  --parent-h5 /path/to/certified/populace_us_2024.h5 \
  --parent-release-dir /path/to/buildp-release-source-receipts \
  --source-cache /path/to/pinned/census-person-csvs \
  --reference-evidence-csv /path/to/source-spm-person-independence-private.csv \
  --output-dir /path/to/new-candidate \
  --release-id populace-us-2024-buildp-spm-role-REVIEWED-NEW-ID

python -m microcosm.data.source_enrichment \
  --release-dir /path/to/new-candidate/releases/RELEASE_ID \
  --parent-h5 /path/to/certified/populace_us_2024.h5 \
  --artifact-root /path/to/new-candidate/artifacts
```

The candidate has `release_type: source_enrichment` and
`compatibility.status: pending`. It includes no inherited built-with or model
compatibility claims. An existing output directory is refused, and a failed
build removes its staging output. Its producer receipt identifies the actual
Git commit, dirty status, and source files. Rebuild from the reviewed clean
producer commit before certification; do not edit a dirty receipt to say clean.
The producer also records Python, HDF5, NumPy, pandas, h5py, PyTables, and data
package versions so serialization can be reproduced in the same runtime.

## Actual compatibility, preflight, and authorized publication

Install the actual country, Core, wrapper, and `spm-calculator` wheels in an
isolated qualification environment. These may be local candidate wheels; prior
registry publication or wrapper certification on the new H5 is not required.
Certification runs the real country and wrapper
native H5 loaders, verifies the role, IDs, memberships and existing weights,
checks that the country registers a person Boolean variable, and calculates
only the primitive on a small complete native household to prove explicit
inputs override its default formula. It does not run a population simulation.
Installed Python sources must match all four supplied wheels and their versions.
The country registers the native role class supplied by
`spm_calculator/policyengine_adapter.py`; its source is bound to the calculator
wheel, while country and wrapper loader sources bind to their respective wheels.

```bash
python -m microcosm.data.source_enrichment --certify \
  --release-dir /path/to/new-candidate/releases/RELEASE_ID \
  --output-dir /path/to/certified/releases/RELEASE_ID \
  --parent-h5 /path/to/certified/populace_us_2024.h5 \
  --artifact-root /path/to/new-candidate/artifacts \
  --compatibility-wheel /path/to/policyengine_us-EXACT.whl \
  --compatibility-wheel /path/to/policyengine_core-EXACT.whl \
  --compatibility-wheel /path/to/policyengine-EXACT.whl \
  --compatibility-wheel /path/to/spm_calculator-EXACT.whl

microcosm-publish-release /path/to/certified/releases/RELEASE_ID \
  --repo-id policyengine/populace-us \
  --parent-h5 /path/to/certified/populace_us_2024.h5 \
  --artifact-root /path/to/new-candidate/artifacts \
  --compatibility-wheel /path/to/policyengine_us-EXACT.whl \
  --compatibility-wheel /path/to/policyengine_core-EXACT.whl \
  --compatibility-wheel /path/to/policyengine-EXACT.whl \
  --compatibility-wheel /path/to/spm_calculator-EXACT.whl \
  --preflight-only
```

## Declaring a publisher compatibility range

Certification writes `compatible_model_packages` and `compatible_core_packages`
as exact pins on the versions the loader checks actually ran against. That is
the default and the safe answer: the bundle claims compatibility with exactly
what was measured.

The exact model pin makes every country release a swap rather than a widening.
A consumer pinned to the previous model version loses certification the moment
a re-certified bundle replaces the published one, and a country patch release
that changes nothing this lane measures still forces a new certified data
release even when the H5 bytes are identical. Where the publisher can stand
behind a range, declare it at certification:

```bash
python -m microcosm.data.source_enrichment --certify \
  --release-dir /path/to/new-candidate/releases/RELEASE_ID \
  --output-dir /path/to/certified/releases/RELEASE_ID \
  --parent-h5 /path/to/certified/populace_us_2024.h5 \
  --artifact-root /path/to/new-candidate/artifacts \
  --compatible-model-specifier 'policyengine-us>=2.0.1,<2.1' \
  --compatibility-claim-declared-by 'PolicyEngine data release owner, microcosm#NNN' \
  --compatibility-wheel ...
```

The claim is recorded in `source_enrichment.json` under
`compatibility.publisher_claims.model` and in `release_manifest.json` as the
single `compatible_model_packages` entry, both carrying
`"basis": "publisher_claim"` and the declarer. The specifier is stored exactly
as declared, save that PEP 508's optional parentheses are dropped. Validation
replays the claim at every later gate, publish preflight included: the manifest
entry must equal what the report declares, so a `release_manifest.json` widened
on its own is refused. Omitting the options leaves certification byte-identical
to an undeclared run — no `basis` key, no `publisher_claims` key.

Certification rewrites `compatible_model_packages` from these options every
time, so re-certifying a bundle that already declares a range without passing
them again reverts it to the exact pin. That is not silent: the run warns,
naming the lowest version the old claim covered and the new one does not, and
records the same under `compatibility.narrowed_claims` in the certified report.
Pass the options again to keep the range.

The record outlives the terminal that printed the warning. Every verdict that
touches the bundle carries it: `--certify` itself, validation
(`python -m microcosm.data.source_enrichment` without `--certify`) and publish
preflight (`microcosm-publish-release --preflight-only`) all print
`narrowed_claims` beside their verdict whenever the bundle records one, so an
operator publishing days later reads what an earlier run gave up rather than
just `passed`. Publication says the same on stderr, because reaching it does
not require running the preflight first — `tools/publish_release.sh` passes its
arguments straight through. All four read the record under one tolerance, so
none of them reports a bundle differently from the others, and a bundle that
gave nothing up prints nothing.

Only the **model** field may be widened. Core keeps the exact pin it has always
had, and a `core` key in `publisher_claims` is refused rather than honoured.
The coverage warning covers Core as defence in depth, and words it as a pin
moving rather than a claim narrowing, because there is no Core claim to narrow.
Nothing reaches that wording today: re-certification validates the input bundle
first, and that gate requires its recorded receipt to equal the current runtime,
so a Core version that moved is refused before the emitted pin could differ from
the carried one.

This is a record of who claimed what, not a tamper-proof seal. The report's
SHA256 lives in the manifest's own `artifacts` map, so widening a certified
bundle by hand takes two coordinated edits plus a hash refresh instead of one —
the same trust model as before, where the exact pin was equally editable. What
actually stands between an edited bundle and the Hub is
`_check_producer_source_identity`, the publish preflight, and the human
publication decision.

The tooling refuses a claim that:

- does not parse as a PEP 508 requirement, or carries a URL, extras or an
  environment marker — it must read `policyengine-us>=2.0.1,<2.1`;
- names a package other than the built-with model package;
- excludes the version certification tested, under the same PEP 440 containment
  the consumers apply (`microcosm.data.loader._package_certification` and
  policyengine.py's `provenance.manifest._specifier_matches`), so a claim that
  is accepted here is a claim they will honour;
- fails any of the three boundedness probes. Over a 2.0.1 build the guard asks
  whether the range still admits the next major version, `3.0.0`; whether it
  still admits a far-future `99999.0.0`; and whether it still admits `0`, the
  bottom of the tested version's epoch. `>=2.0.1`, `!=2.0.5` and
  `>=2.0.1,<3.0.1` fail the first; `>=2.0.1,!=3.0.0`, which excludes the next
  major by name while still certifying 4.x, fails the second; `<2.1` and
  `<=2.0.5`, bounded above but open below, fail the third — a bare `<2.1`
  certifies every release the package ever made, including ones predating the
  native-input loader path this qualification measures. State both bounds:
  `>=2.0.1,<2.1`, `~=2.0.1`, `==2.0.*` or `>=2.0.1,<3`. Probes bound a claim;
  they do not prove one is bounded, and the residue is symmetric: a specifier
  that names the probe versions and excludes them passes while admitting
  others. `>=2.0.1,!=3.0.0,!=99999.0.0` is accepted and admits `5.0`;
  `<2.1,!=0` is accepted and admits `0.9.0`. Both are pinned by test so the
  limit cannot quietly widen past what is written here. Declare real bounds
  rather than a hole-punched open range;
- arrives without `--compatibility-claim-declared-by`. A wider claim is the
  publisher's assertion rather than a measurement, so the bundle records who
  made it.

A **prerelease** built-with version narrows the options, for an ordering reason
rather than an exclusion one. `packaging` matches prereleases by default,
following PEP 440's recommendation, but a prerelease sorts below its own
release: over a `2.0.1rc1` build both `>=2.0.1,<2.1` and `~=2.0.1` exclude the
very version certification tested, and the containment check refuses them. A
range has to name the prerelease in its lower bound (`>=2.0.1rc1,<2.1`) or match
the series with a prefix (`==2.0.*`); both pass all three boundedness probes.
Declaring nothing leaves the exact `==2.0.1rc1` pin, which is the honest option
for a runtime still in prerelease anyway.

Where the tooling draws its line and where practice should draw one are not the
same place: the guard bounds a claim at the next major version, so `>=2.0.1,<3`
is accepted, while the recommended range stops at the next minor
(`>=2.0.1,<2.1`) — the span a publisher can actually read the diff for.

### When a range is appropriate

Declare a range over the model versions whose differences cannot reach what
certification measured — in practice a **country patch release that changes
neither the native H5 loader path, the person-role variable, nor the dataset
pin**. The native loader checks are `native_input_loading_only`; the claim is
about them and nothing else. Before declaring, read the diff between the tested
version and the upper bound and confirm it touches none of: the H5/dataset
loader, `DEFAULT_DATASET`, the `is_spm_independent_minor_role` registration, the
entity tables this release writes, or the SPM path that consumes them.

### When it is not

- **A minor or major bump** (2.0.x → 2.1, 2.x → 3). Re-certify instead.
- **Anything the certification did not test.** Native input loading is not
  numerical acceptance; a range never extends to SPM numerics, canonical model
  acceptance, or Axiom parity, which root owns separately.
- **A range used to avoid re-running certification** when the runtime under the
  upper bound was never installed anywhere. A claim the publisher cannot defend
  is worse than a new release.
- **Speculative headroom.** `>=2.0.1,<2.1` because 2.0.2 is expected is
  defensible; `>=2.0.1,<3` because a major bump seems far off is not.

Consumers record which basis they used. Measured against policyengine.py's
installed provenance code, a manifest declaring `>=2.0.1,<2.1` over `built_with`
2.0.1 accepts 2.0.1 on the exact build-time match, accepts 2.0.2 on the claim,
and refuses 2.0.0 and 2.1.0 — on both paths. They differ in what they say:

- **Bundle certification** (`provenance/certification.py::validate_release_manifest`)
  returns basis `compatible_model_packages` and a warning naming the claim and
  the version the data was built with. That warning is the intended cost of the
  wider binding, and an operator certifying a bundle sees it.
- **Runtime binding** (`provenance/manifest.py::certify_data_release_compatibility`)
  returns basis `legacy_compatible_model_package` and warns about nothing —
  that module issues no warnings at all. A user running the certified bundle on
  a version the claim covers gets no signal; the recorded basis is the only
  trace. Do not declare a range expecting the runtime to caveat it for you.

The range relaxes consumer-side certification only. It changes nothing about
the publisher's own gates: `--preflight-only` and the real publisher both re-run
the native-loader qualification in the current environment and require the
recomputed receipt to equal the one the bundle records, and that receipt names
the exact versions certification tested. A publish preflight therefore still
runs with the exact `built_with` model and Core versions installed, with the
four matching wheels to hand. A declared range never lets the publisher replay
a bundle against a runtime it did not measure.

### Follow-on: the other release types still pin exactly

This option exists on the source-enrichment lane only. The calibration release
assemblers — `tools/assemble_uk_release_dir.py`,
`tools/assemble_uk_dense_release_dir.py` and
`tools/build_us_fiscal_refresh_release.py` — each write
`compatible_model_packages` as `==<measured runtime version>` with no way to
declare a range, so a UK national, UK dense or US fiscal-refresh release stays
an exact swap for its consumers. Widening any of them is a separate change with
its own review.

Certification creates a separate bundle with measured compatibility; it leaves
the candidate H5 and source evidence unchanged. Both the preflight above and
the real publisher share local preparation: they invoke the source-enrichment
validator, replay the H5 and compatibility checks, and enforce release filenames,
root artifact paths/hashes, revision/tag pins and dataset-role/latest-pointer
eligibility. Preflight never constructs a Hub client or performs Hub activity;
publication constructs its client only after these checks pass. Pass the same
tag, extra-file and latest-pointer options to preflight as to publication.
The evidence-tier publisher cannot be used as an escape hatch. Pending
compatibility and a recorded dirty producer build are hard publication failures.

Coordinated order: build local candidate wheels, test them against the existing
immutable candidate H5, then let root commit the clean reviewed producer and
create fresh qualification receipts. Root separately verifies external package
publication and numerical/Fable acceptance before data publication and consumer
promotion. A wrapper candidate can load this local H5 before either is published,
so H5 qualification does not depend on an already certified wrapper release.
Qualification retains `external_package_publication: not_attested` and
`scope: native_input_loading_only`; it never substitutes for numerical acceptance.

These receipts deliberately state that external package publication and
canonical SPM numerical acceptance are **not attested**. Root owns exact Fable
review, canonical calculator/country/wrapper/Axiom parity, and the immutable
published package proof. Once those gates pass and root authorizes publication,
use `tools/publish_release.sh` with the same directory and arguments, remove
`--preflight-only`, and add `--tag-name RELEASE_ID`. The existing publisher
creates the new immutable Hugging Face tag; its standard invocation also
updates `latest.json`. An inspect-only tag uses `--no-latest --tag-only`.
Neither candidate construction nor certification authorizes either mutation.

Run certification and publisher commands from the Microcosm checkout. Publication
authenticates the six recorded producer source hashes against the recorded Git
commit, the checkout, and the executing data contract modules. A later docs-only
commit is allowed; an invented clean flag, missing commit, changed source, or
older installed producer module is refused.

## Local regression checks

```bash
python -m pytest \
  packages/microcosm-build/tests/test_us_spm_role_source.py \
  packages/microcosm-build/tests/test_us_spm_role_enrichment_builder.py \
  packages/microcosm-data/tests/test_h5_enrichment.py \
  packages/microcosm-data/tests/test_source_enrichment.py \
  packages/microcosm-data/tests/test_contract.py \
  packages/microcosm-data/tests/test_release.py \
  packages/microcosm-data/tests/test_publish_guard.py
python tools/ci_test_groups.py --verify
ruff check .
```

These tests use synthetic populations. Their success is not certification of
the full private artifact or compatibility with an unreleased model.
