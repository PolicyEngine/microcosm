# populace-data

The distribution end of the [populace](https://github.com/PolicyEngine/populace)
stack. The operator shards (`populace-frame`, `populace-fit`,
`populace-calibrate`) *build* populations; `populace-data` *serves* the ones
populace has published — a registry of `(country, year)` pointers to artifacts
on the Hugging Face Hub, and a loader that returns each as its PolicyEngine
engine dataset.

```bash
pip install 'populace-data[us]'
```

```python
from policyengine_us import Microsimulation
from populace.data import load

sim = Microsimulation(dataset=load("us", 2024))
sim.calculate("household_net_income", 2024).sum()
```

`load("us")` (no year) loads the latest published year. `available()` lists
every published `(country, year)`.

## Why a shard, not a repo per country

Publishing a new population is **one `DatasetSpec` entry** in
[`registry.py`](src/populace/data/registry.py) plus its uploaded artifact on the
Hub — never a new package or repository. The registry is the single source of
truth for what exists and where; nothing else hard-codes a repo id, filename, or
engine class.

The shard does not depend on the Frame kernel: a published population is an
engine-native dataset, so loading it needs only `huggingface_hub` and the
country engine. The engine is an optional extra (`populace-data[us]`,
`populace-data[uk]`, …) imported lazily, so the base install pulls neither torch
nor any country model until a load actually needs one.

## Published datasets

| Country | Year | Artifact | Engine |
| --- | --- | --- | --- |
| US | 2024 | [`policyengine/populace-us`](https://huggingface.co/datasets/policyengine/populace-us) | `policyengine-us` |

The **populace-US** population is a calibrated synthetic microdataset that
loads anywhere the enhanced CPS loads — built from CPS ASEC structure and IRS
PUF tax detail (weight-aware imputation), calibrated to PolicyEngine's
administrative target surface with a hard per-record weight bound. Its
strengths and its gaps (no wealth layer yet, degenerate housing tenure,
several un-imputed income streams) are documented on the dataset card and the
[populace.dev dashboard](https://populace.dev/dashboard); the provenance
snapshot lives in [`build/us/`](build/us).
