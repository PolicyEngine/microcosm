---
license: mit
pretty_name: populace-us
tags:
  - policyengine
  - microsimulation
  - synthetic-population
  - tax-benefit
  - united-states
---

# populace-us

The **populace-built US population**: a calibrated synthetic microdataset for
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us), built by the
[`populace`](https://github.com/PolicyEngine/populace) stack. It loads anywhere
the enhanced CPS loads (an API-compatible alternative population), with its own
calibrated weights — and its own strengths and gaps, both documented below.

## Load it

```bash
pip install 'populace-data[us]'
```

```python
from policyengine_us import Microsimulation
from populace.data import load

sim = Microsimulation(dataset=load("us", 2024))
sim.calculate("household_net_income", 2024).sum()
```

Or grab the H5 directly:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="policyengine/populace-us",
    filename="populace_us_2024.h5",
    repo_type="dataset",
)
```

## What it is

One HDF5 `USSingleYearDataset` per year. The population is generated from a
declarative spec: Current Population Survey ASEC provides household structure
(~half the records are PUF-derived clones, flagged per record), tax detail is
imputed from the IRS Public Use File with weight-aware quantile-forest models,
and the result is calibrated to PolicyEngine's administrative target surface
(3,704 IRS/Census/program targets) with a hard per-record weight bound
(`max_weight_ratio=50`), so no aggregate leans on a handful of super-weighted
records. Same source classes and hosting precedent as PolicyEngine's published
enhanced CPS.

## Validation

<!-- VALIDATION_TABLE -->

## Known gaps

We publish the misses with the hits:

- **No wealth layer yet.** Balance-sheet columns (net worth, vehicles, debts)
  are present in the schema but zero-valued; SCF imputation is on the roadmap,
  not in this artifact.
- **Housing tenure is degenerate** (no owner/renter split), so
  tenure-dependent analysis (renter credits, property-tax circuit breakers,
  shelter deductions) gets uninformative inputs in this release.
- **Several income streams are not yet imputed** and sit at zero: retirement
  account distributions and contributions, workers' compensation, veterans'
  benefits, tips.
- **Aggregate household net income runs low** relative to the enhanced CPS
  (the gap is concentrated in the un-imputed streams above and tail capital
  income); some program totals land closer to administrative actuals than the
  incumbent (e.g. SSI), others further. Results from the two populations are
  not interchangeable.

The dashboard at [populace.dev/dashboard](https://populace.dev/dashboard)
shows the full per-family calibration fit, the worst-fit targets by name, and
the weight distribution. Methodology and evidence:
[populace.dev](https://populace.dev); loader and registry:
[github.com/PolicyEngine/populace](https://github.com/PolicyEngine/populace)
(`packages/populace-data`).
