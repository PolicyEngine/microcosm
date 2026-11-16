# microcosm-data

The distribution end of the [microcosm](https://github.com/PolicyEngine/microcosm)
stack. The operator shards (`microcosm-frame`, `microcosm-fit`,
`microcosm-calibrate`) *build* populations; `microcosm-data` *serves* the ones
microcosm has published — a registry of `(country, year, variant)` pointers to
artifacts on the Hugging Face Hub, and a loader that returns each as its
PolicyEngine engine dataset.

```bash
pip install 'microcosm-data[us]'
```

```python
from policyengine_us import Microsimulation
from microcosm.data import load

sim = Microsimulation(dataset=load("us", 2024))
sim.calculate("household_net_income", 2024).sum()
```

`load("us")` (no year) loads the latest published compact year: the current
build of release `microcosm-us-2024-national`. Every UK and US release that
takes new builds publishes each build into its own folder,
`releases/<release>/<build>/`, and the repository's `releases.json` records
which build of each release is current. The loader reads that record, then
the current build's `release_manifest.json`, then the dataset it names, all
from the build's folder. It verifies the artifact SHA-256 and refuses model
or Core versions outside the build's certified compatibility specifiers.
`load("us", build="<build>")` or `load("uk", build="<build>")` loads a listed
build instead (a withdrawn one with a warning). A US national build's dataset
is `microcosm_us_2024.h5`, or `populace_us_2024.h5` for the builds the US
switch moved into folders; the loader accepts either name for a listed build
(`DatasetSpec.build_folder_filename`) and refuses any other.

`load("uk")` (the 2025 national release) and the 2025 dense variant read the
UK repository's record the same way. `load("uk", 2023)` reads release
`populace-uk-2023`: the UK repository's switch moved the June 2023 build into
it as its only and current build. An entry whose repository has no
`releases.json` follows its pointer file (`latest.json` or
`latest-<name>.json`) and reads the manifest at the build's tag; no
registered entry's repository is still in that state. `available()` lists
the default-variant `(country, year)` pairs (the ones a bare ``load(country, year)``
resolves); `available_variants()` lists every registered
`(country, year, variant)`, non-default lines included.

The old mutable-root behavior is an unsafe escape hatch,
`load("us", 2024, unverified_root=True)`, which emits a runtime warning because
it bypasses the release pointer, manifest, digest, and engine compatibility
checks. The loader refuses it for an entry whose repository records its builds
in `releases.json`, which every registered entry's repository does since the
UK and US switches. Builds published into folders write no root file, so the
US repository's root `populace_us_2024.h5` stopped changing at its switch,
and the switch tool's `delete-old-layout` step deletes it.

## Why a shard, not a repo per country

Publishing a new population is **one `DatasetSpec` entry** in
[`registry.py`](src/microcosm/data/registry.py) plus its uploaded artifact on the
Hub — never a new package or repository. The registry is the single source of
truth for what exists and where; nothing else hard-codes a repo id, filename, or
engine class.

Dataset variants let one country/year expose multiple scale contracts without a
new shard. `variant="compact"` is the default fast national microsimulation
artifact; a future UK local-geography build should register the same
country/year with `variant="local"` once its pooled-FRS, clone-and-assign, and
L0 calibration artifact is published.

The shard does not depend on the Frame kernel: a published population is an
engine-native dataset, so loading it needs only `huggingface_hub` and the
country engine. The engine is an optional extra (`microcosm-data[us]`,
`microcosm-data[uk]`, …) imported lazily, so the base install pulls neither torch
nor any country model until a load actually needs one.

## Published datasets

| Country | Year | Artifact | Engine |
| --- | --- | --- | --- |
| UK | 2023 compact | [`policyengine/populace-uk-private`](https://huggingface.co/datasets/policyengine/populace-uk-private) | `policyengine-uk` |
| US | 2024 compact | [`policyengine/populace-us`](https://huggingface.co/datasets/policyengine/populace-us) | `policyengine-us` |

The **microcosm-US** population is a calibrated synthetic microdataset that
loads as a PolicyEngine-US dataset — built from CPS ASEC structure and IRS PUF
tax detail (weight-aware imputation), calibrated to PolicyEngine's
administrative target surface with a hard per-record weight bound. Its
strengths and gaps are documented on the dataset card and the
[microcosm.institute dashboard](https://microcosm.institute/dashboard). Historical incumbent
benchmark comparisons live outside this package.

## Release contract

Each build published since its repository's switch to `releases.json`, and
each build the switch moved, lives in its own folder,
`releases/<release>/<build>/`, on the Hub dataset repo's `main` branch. A
build that keeps only its tag lives under `releases/<build_id>/` at that tag.
The five US releases are `microcosm-us-2024-national` (the default),
`microcosm-us-2024-local`, `microcosm-us-2024-receipt-donor`,
`microcosm-us-2024-evidence` and `microcosm-us-2024-annual`
(`microcosm.data.contract.US_RELEASES`).
Each build must include `build_manifest.json` and `release_manifest.json`;
US releases must also include `us_source_coverage.json`. A current release
manifest contains a typed `calibration_diagnostics` status. `available` binds a
schema-9 `calibration_diagnostics.json` file (schema 8 in releases published
before it) and its SHA-256. `failed` records
the expected schema, error code, and sanitized message and deliberately omits
the file and artifact entry. Publication logs a warning but continues, because
the diagnostics document reports a completed calibration and is not an input
to the dataset itself. Historical manifests without the status retain the old
required-file behavior. The release manifest records the build environment
as `built_with_model_package` and `built_with_core_package`, at its top level
in a build folder and under `build` in a build that keeps only its tag, and
separately records certified runtime
compatibility through `compatible_model_packages` and `compatible_core_packages`
using PEP 440 specifiers. By default those entries pin exactly the versions the
build measured. A model entry the publisher deliberately widened
instead carries `"basis": "publisher_claim"` and a `declared_by` naming who is
accountable for the claim; the loader honours any specifier that contains the
built version, whichever way the entry was produced.

Use `releases.json` to find each release's current build, and the build's
folder for its files; a build folder never changes once published. A build
published without being made current, such as a US exact-k ladder build,
does not replace the build the default `load("us")` resolution selects. Load
it by name with `load("us", build="<build>")` until a separate promotion
makes it the national release's current build. The exact-k build also runs
with `--no-staging`, so it sends no staging run files.
