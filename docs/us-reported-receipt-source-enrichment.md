# Reported-receipt source-enrichment release

This lane publishes the reported-receipt child of the US national default as
its own release, so the ACS local-area chain can stage from a **published**
donor with `--donor-release-manifest` (microcosm#978). The child is the
national default `populace-us-2024-spm-20260915` with three reported-receipt
inputs added: `person.receives_wic`, `spm_unit.receives_snap` and
`spm_unit.receives_tanf`.

It is not a new release type. It is a second operation of the existing
`source_enrichment` type, which already delivers the native SPM role
(`add_native_spm_independent_minor_role`, [runbook](us-native-spm-role-source-enrichment.md)).
The release manifest, calibration inheritance, exhaustive H5 replay,
native-loader qualification, certification, preflight and producer-identity
gates are the same functions. What differs is the pinned parent, the evidence
that proves the addition, and the native inputs the loader qualification
checks. The SPM-role lane's checks are unchanged, and its tests pass unmodified.

## Why a published donor is required

A local-area release names its donor in `us_source_coverage.json`
(`donor_release`). The staging tool pins that only when it is given the donor's
`release_manifest.json`, and it then requires that manifest to carry exactly
one microdata artifact whose SHA-256 equals the staging donor H5
(`_donor_release_identity`,
`tools/_legacy/build_us_acs_multispine_base.py:169-216`). The contract refuses
a local-area release without it (`microcosm/data/contract.py:4468-4486`).

The national default does not carry the three receipt inputs the ACS transfer
requires, so the local chain stages from the child that
`tools/build_us_acs_donor_receipt_qualification.py` produces
([qualification note](us-acs-donor-receipt-qualification.md)). That child had
no published manifest, so the full-scale local release built on it was refused
before any remote mutation (microcosm#978). This lane gives the child a
manifest whose single microdata artifact is the child itself.

## The pinned parent

Everything below is a constant in `microcosm/data/source_enrichment.py`, not a
caller-supplied pin (`:140-202`):

| Constant | Value |
| --- | --- |
| `RECEIPT_PARENT_BUILD_ID` | `populace-us-2024-spm-20260915`, the release `latest.json` names |
| `RECEIPT_PARENT_DATASET_SHA256` | `6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84` |
| `RECEIPT_PARENT_FILES["parent_release_manifest.json"]` | `d5c9e2a3…c01`, that release's `release_manifest.json` |
| `RECEIPT_PARENT_FILES["calibration_diagnostics.json"]` | `870449b4…40e`, Build P's schema-5 diagnostics |
| `RECEIPT_PARENT_FILES["us_source_coverage.json"]` | `6406c868…ff5`, Build P's coverage |
| `RECEIPT_DATASET_FILENAME` | `populace_us_2024_receipt_qualified.h5` |

The same H5 bytes are also published as `populace-us-2024-spm-20260909`. The
pinned id is `-20260915` because that is the release whose manifest bytes are
pinned, and the one `latest.json` names.

The pinned parent release manifest hash-binds the parent H5 and every other
parent file: its build manifest, SPM-role report, compatibility receipt and
role evidence. Anyone can fetch those at the parent tag and check them against
the carried manifest. The diagnostics and coverage bytes are Build P's own,
which the SPM-role release inherited unchanged, and a test pins them equal to
`PARENT_FILES`. So this lane grants no calibration an inheritance it did not
already have. The child's calibration is still Build P's schema-5
calibration, now two enrichment hops away, and the build manifest says so
explicitly (`calibration.mode: inherited`, `parent_build_id:
populace-us-2024-spm-20260915`).

A different parent needs a separately reviewed pin. Build P's own enrichment
lane stays pinned to Build P, as
[the release build rule](us-release-build-rule.md) requires.

## How provenance is carried

The qualification tool's receipt, `donor_receipt_qualification.json`, is copied
into the release directory byte-for-byte as source evidence. It records the
parent and child SHA-256, the producer's git commit, dirty flag and the SHA-256
of every source file that decided a receipt value, the exhaustive preservation
report, and aggregate true and false counts. It never records a row value. The
release report (`source_enrichment.json`) binds it by filename and SHA-256
under `source`, and declares `operation: add_reported_receipt_inputs`, the
pinned `parent` block, `dataset`, `added_variables` and `preservation`.

`validate_source_enrichment_candidate` (`source_enrichment.py:471`) selects the
lineage from `operation` (`:510`). Any other value, including a misspelling, is
judged as the SPM-role lane and refused there. For a receipt child it
checks:

- **Parent identity.** The parent H5 must have the pinned SHA-256. The report's
  `parent` block must equal the pinned identity exactly, and the three
  inherited files must carry the pinned bytes. The release id must be new: not
  Build P's id and not the parent's.
- **The child, replayed.** `compare_boolean_append` (`:635`,
  `microcosm/data/h5_boolean_append.py:189`) reruns the exhaustive comparison
  of the actual parent and candidate H5 files. Its result must equal the
  report's `preservation`, which must in turn equal the receipt's. This is the
  qualification's own verifier, moved into the data shard so the contract can
  run it. A hand-edited receipt is therefore not evidence.
- **The receipt** (`_check_receipt_qualification`, `:345`). It must be a
  schema-1 `add_reported_receipt_inputs` receipt naming the pinned parent and
  describing the enriched H5. It must add exactly the three inputs, in their
  append order. Each column's recorded rows and true count are recounted from
  the candidate H5 (`_receipt_column_counts`, `:329`). Each column must have a
  true value in the donor and inside the channel the staging gate selects. The
  receipt's per-year and per-role breakdowns are not recounted, because the
  data shard cannot rederive them.
- **One microdata artifact.** Exactly one microdata artifact, the child, at
  `populace_us_2024_receipt_qualified.h5`, selected by
  `default_datasets.national`. Any other filename is refused (`:602`), so no
  publication of this donor can overwrite the default's root
  `populace_us_2024.h5`. As in the role lane, the H5 lives only in the
  artifact root, never in the release directory.
- **Producers, at certification and publication**
  (`_check_receipt_producers`, `:431`). Two identities are authenticated
  against the checkout, each with `_check_producer_source_identity`
  (`:839`): the release assembler's build identity over
  `RECEIPT_RELEASE_PRODUCER_FILES`, and the receipt's identity over
  `RECEIPT_QUALIFICATION_SOURCE_FILES`, which a test holds equal to the
  qualification tool's `_PRODUCER_FILES`. Each must be a clean recorded commit
  that resolves in this checkout, and every listed file must have the recorded
  bytes both at that commit and in the checkout. Every data-contract module
  the list names must be the module that is executing.

## Native loader qualification

Certification runs `run_native_loader_compatibility` (`:1450`) with
`native_inputs=RECEIPT_NATIVE_INPUTS`: the SPM role the child inherits, plus
the three receipts. For each input the tested country must register a Boolean
of the right entity. Both the country and wrapper loaders must return the
column byte-identical to the H5. The complete-household probe
(`_check_native_input_precedence`, `:1621`) supplies all four inputs at once,
each on its own entity's rows, as `False` and then `True`, and Core must return
every one unchanged. The role's source must come from the `spm-calculator`
wheel. Each receipt variable's source must come from the `policyengine-us`
wheel (`_loaded_source_packages`, `:1437`). The same four wheels are required.
Without `native_inputs` the runner and its receipt are exactly the role lane's.

Measured in this change: the three receipt variables are registered in
policyengine-us 2.2.1 as Boolean inputs with no formula, on `person`,
`spm_unit` and `spm_unit`. A `requires_us` test supplies all three on a
synthetic household through the real country Core and reads them back
unchanged.

## Never the default pointer

The receipt inputs change results. policyengine-us 2.2.1 reads them in, for
example, WIC categorical eligibility
(`variables/gov/usda/wic/meets_wic_categorical_eligibility.py:15-19`) and SNAP
categorical eligibility
(`variables/gov/usda/snap/eligibility/meets_snap_categorical_eligibility.py:21`).
The published default omits them, so the model uses their `False` defaults.
The child is a donor for the local chain, not a new default. The publisher's
shared preparation therefore refuses to move `latest.json` to it
(`microcosm/data/release.py:234-247`). Publish it with `--no-latest`, and with
`--tag-only` so that `main` is untouched.

## Steps

Every step up to the preflight is local and never contacts the Hub.

1. **Qualify the parent** from a clean checkout. Pass the parent under its
   published name, so that the child is named
   `populace_us_2024_receipt_qualified.h5`:

   ```bash
   env -u UV_FROZEN uv run --no-sync python tools/build_us_acs_donor_receipt_qualification.py \
     --parent-h5 /path/to/spm-20260915/populace_us_2024.h5 \
     --output-dir /path/to/qualified
   ```

2. **Package the candidate.** `--parent-release-dir` holds the parent's
   `release_manifest.json`, `calibration_diagnostics.json` and
   `us_source_coverage.json`, verified against the pins:

   ```bash
   env -u UV_FROZEN uv run --no-sync python tools/build_us_receipt_enrichment_release.py \
     --parent-h5 /path/to/spm-20260915/populace_us_2024.h5 \
     --parent-release-dir /path/to/spm-20260915-evidence \
     --qualification-dir /path/to/qualified \
     --output-dir /path/to/candidate \
     --release-id populace-us-2024-spm-receipts-YYYYMMDD
   ```

   The tool derives nothing. It hard-links the child into
   `candidate/artifacts/`, copying it only across devices, copies the receipt
   and parent evidence, and writes the report and manifests. It runs the
   contract, which replays the H5 comparison, and only then exposes the
   directory atomically (`build_candidate`,
   `tools/build_us_receipt_enrichment_release.py:178`). Compatibility is
   `pending`.

3. **Validate** the pending candidate:

   ```bash
   python -m microcosm.data.source_enrichment \
     --release-dir /path/to/candidate/releases/RELEASE_ID \
     --parent-h5 /path/to/spm-20260915/populace_us_2024.h5 \
     --artifact-root /path/to/candidate/artifacts
   ```

4. **Certify** in an isolated environment with the four exact wheels (country,
   Core, wrapper, calculator), exactly as for the role lane. This step writes a
   new bundle, and certification selects the receipt inputs itself:

   ```bash
   python -m microcosm.data.source_enrichment --certify \
     --release-dir /path/to/candidate/releases/RELEASE_ID \
     --output-dir /path/to/certified/releases/RELEASE_ID \
     --parent-h5 /path/to/spm-20260915/populace_us_2024.h5 \
     --artifact-root /path/to/candidate/artifacts \
     --compatibility-wheel ... (four)
   ```

5. **Preflight**, which constructs no Hub client:

   ```bash
   microcosm-publish-release /path/to/certified/releases/RELEASE_ID \
     --repo-id policyengine/populace-us \
     --parent-h5 /path/to/spm-20260915/populace_us_2024.h5 \
     --artifact-root /path/to/candidate/artifacts \
     --compatibility-wheel ... (four) \
     --no-latest --tag-only --preflight-only
   ```

6. **Publish.** This is a deliberate human step, once root has authorized it.
   Use the same command through `tools/publish_release.sh`, without
   `--preflight-only`. It creates the immutable tag `RELEASE_ID` and leaves
   `main` and `latest.json` alone. Run it from a checkout in which every file
   in both producer lists has the recorded bytes, such as the producer commit
   itself.

7. **Stage the local chain** from the published donor. Pass `--base-h5` the
   child and `--donor-release-manifest` the certified bundle's
   `release_manifest.json`, then rerun the local build from staging.

## Local regression checks

```bash
python -m pytest \
  packages/microcosm-build/tests/test_us_receipt_enrichment_release.py \
  packages/microcosm-build/tests/test_us_acs_donor_receipt_qualification.py \
  packages/microcosm-data/tests/test_source_enrichment.py \
  packages/microcosm-build/tests/test_us_spm_role_enrichment_builder.py \
  packages/microcosm-data/tests/test_release.py
python tools/ci_test_groups.py --verify
```

These tests use invented populations and pins. Passing them does not certify
the real artifact, and it does not establish compatibility with a model that
has not been released.
