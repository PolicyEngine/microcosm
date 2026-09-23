# Native 19-node retention census (2026-09-23)

A structural memory census of `run_atomic_survey_financial` at this tree
(`47960af43`, native-release-integration-20260919), run on the existing
invented source-issued fixture, then scaled per row to real entity counts.
No actual data, no full-source run and no measured full-source RSS are
involved. The numbers below are **live-set estimates**, not peak-RSS
predictions.

## Run

```bash
uv run pytest -p conftest -s \
    experiments/native-memory-model-20260923/test_retention_census.py  # ~100 s
python experiments/native-memory-model-20260923/model.py               # prints tables, writes model-summary.json
```

The test is outside `testpaths`, so the default suite never collects it.

## What is measured and what is scaled

- Measured on invented data at this tree: which objects are live at three
  points, which of them share physical buffers, and bytes per row per entity
  table with the graph's real column set and dtypes.
- Measured on real data by earlier receipts: cloned entity counts at 1/15 and
  1/1000, and the documented full-source counts (`docs/us-native-row-ceilings.md`).
- Assumed: the stacked grain is half the cloned grain; the ASEC donor grain is
  3.47% of the stacked grain; opaque byte payloads do not scale (they are
  reported, not extrapolated); Python `str` cell sizes transfer from invented
  strings.

## Live set at full source (GiB, unique physical bytes)

| point | `all` retention | `compact` retention |
| --- | ---: | ---: |
| prefix runner, after its fresh geography reconstruction | 208.6 | 208.6 |
| financial runner, just after the 19-node `run_graph` returns | 251.6 | 137.1 |
| financial runner, after the independent replay (`atomic._states`) | 259.1 | 144.5 |
| returned run object | 197.1 | 82.5 |

Largest owners after the replay (`all` → `compact`): detached observations
126.4 → 11.9; the financial runner's own geography reconstruction 54.5; the
prefix result 45.4; manifest-attached frames 20.3; expected pool 7.5.
Inside the prefix: its detached observations 69.5, its fresh reconstruction
54.5, its expected pool 47.3. `compact` does not reach the prefix runner.

At 1/15 the same scaling gives 17.28 GiB after the replay (`all`). The
measured 1/15 whole-process high-water on `9af56aa8a` was 43.46 GiB and the
1/1000 high-water 11.08 GiB, so about half of the measured slope is not
explained by the live set. That remainder is unattributed.
