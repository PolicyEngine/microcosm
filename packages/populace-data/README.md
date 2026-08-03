# populace-data

The distribution end of the [populace](https://github.com/PolicyEngine/populace)
stack. The operator shards (`populace-frame`, `populace-fit`,
`populace-calibrate`) *build* populations; `populace-data` *serves* the ones
populace has published — a registry of `(country, year, variant)` pointers to
artifacts on the Hugging Face Hub, and a loader that returns each as its
PolicyEngine engine dataset.

```bash
pip install 'populace-data[us]'
```

```python
from policyengine_us import Microsimulation
from populace.data import load

sim = Microsimulation(dataset=load("us", 2024))
sim.calculate("household_net_income", 2024).sum()
```

`load("us")` (no year) loads the latest published compact year. It resolves
`latest.json`, reads the selected release manifest at the immutable release
tag, verifies the artifact SHA-256, and refuses model or Core versions outside
the release's certified compatibility specifiers. `available()` lists
published `(country, year)` pairs; `available_variants()` lists every published
`(country, year, variant)`.

The old mutable-root behavior is available only as an unsafe escape hatch:
`load("us", 2024, unverified_root=True)`. It emits a runtime warning because it
bypasses the release pointer, manifest, digest, and engine compatibility checks.

## Why a shard, not a repo per country

Publishing a new population is **one `DatasetSpec` entry** in
[`registry.py`](src/populace/data/registry.py) plus its uploaded artifact on the
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
country engine. The engine is an optional extra (`populace-data[us]`,
`populace-data[uk]`, …) imported lazily, so the base install pulls neither torch
nor any country model until a load actually needs one.

## Published datasets

| Country | Year | Artifact | Engine |
| --- | --- | --- | --- |
| UK | 2023 compact | [`policyengine/populace-uk-private`](https://huggingface.co/datasets/policyengine/populace-uk-private) | `policyengine-uk` |
| US | 2024 compact | [`policyengine/populace-us`](https://huggingface.co/datasets/policyengine/populace-us) | `policyengine-us` |

The **populace-US** population is a calibrated synthetic microdataset that
loads as a PolicyEngine-US dataset — built from CPS ASEC structure and IRS PUF
tax detail (weight-aware imputation), calibrated to PolicyEngine's
administrative target surface with a hard per-record weight bound. Its
strengths and gaps are documented on the dataset card and the
[populace.dev dashboard](https://populace.dev/dashboard). Historical incumbent
benchmark comparisons live outside this package.

## Release contract

Published releases live under `releases/<release_id>/` in the Hub dataset repo.
Each release must include `build_manifest.json`, `release_manifest.json`, and
`calibration_diagnostics.json`; US releases must also include
`us_source_coverage.json`. The release manifest records the build environment
under `build.built_with_*_package` and separately records certified runtime
compatibility through `compatible_model_packages` and `compatible_core_packages`
using PEP 440 specifiers.

Use `latest.json` to discover the current release and its contract file paths;
use the release id/tag in artifact revisions when loading an immutable release.
An exact-k ladder candidate published with `--create-tag --no-latest` does not
update `latest.json` and therefore does not replace the release selected by the
default `load("us")` resolution. Inspect it through its explicit release id or
Hub tag until a separate promotion updates `latest.json`. The exact-k build
also runs with `--no-staging`, so it does not create or update a staging
pointer.
