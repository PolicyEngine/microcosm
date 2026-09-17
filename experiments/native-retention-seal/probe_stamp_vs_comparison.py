"""US_SCHEMA version: _population_stamp vs same_replayed_population on a store round trip."""
import pathlib, tempfile
from dataclasses import replace
import numpy as np, pandas as pd
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph.population import Population
from microcosm.graph.store import ContentStore
from microcosm.build.us_runtime.survey_population_replay import same_replayed_population
from microcosm.build.us_runtime import survey_atomic_geography as geo
from microcosm.build.us_runtime import survey_population_preparation as prep

n = 3
person = pd.DataFrame({
    "person_id": np.arange(n, dtype=np.int64),
    "person_household_id": np.array([0, 0, 1], dtype=np.int64),
    "person_tax_unit_id": np.array([0, 0, 1], dtype=np.int64),
    "person_spm_unit_id": np.array([0, 0, 1], dtype=np.int64),
    "person_family_id": np.array([0, 0, 1], dtype=np.int64),
    "person_marital_unit_id": np.array([0, 1, 2], dtype=np.int64),
    "nullable_integer": pd.arrays.IntegerArray(
        np.array([11, 71, 73], dtype=np.int64), np.array([False, True, True])),
})
groups = {
    "household": pd.DataFrame({"household_id": np.array([0, 1], dtype=np.int64)}),
    "tax_unit": pd.DataFrame({"tax_unit_id": np.array([0, 1], dtype=np.int64)}),
    "spm_unit": pd.DataFrame({"spm_unit_id": np.array([0, 1], dtype=np.int64)}),
    "family": pd.DataFrame({"family_id": np.array([0, 1], dtype=np.int64)}),
    "marital_unit": pd.DataFrame({"marital_unit_id": np.array([0, 1, 2], dtype=np.int64)}),
}
frame = Frame({"person": person, **groups}, US_SCHEMA,
              {"household": Weights(np.array([1.5, 2.5]), WeightKind.IMPORTANCE)},
              pd.Series(["urban", "urban", "rural"], dtype=object))
tmp = pathlib.Path(tempfile.mkdtemp(prefix="stampgap2-"))
store = ContentStore(tmp / "store")
store.put_frame("a" * 64, frame)
loaded = store.load_frame("a" * 64)

expected = Population.from_frame(frame, "allocated")
actual = replace(expected, frame=loaded)
print("expected masked _data:", expected.frame.person["nullable_integer"].array._data.tolist())
print("actual   masked _data:", actual.frame.person["nullable_integer"].array._data.tolist())
try:
    same_replayed_population(expected, actual); print("same_replayed_population: ACCEPTS")
except ValueError as e: print("same_replayed_population: REFUSES", e)
se, sa = geo._population_stamp(expected), geo._population_stamp(actual)
print("_population_stamp equal :", se == sa, se[:12], sa[:12])
fe, fa = prep._frame_identity(expected.frame), prep._frame_identity(actual.frame)
print("_frame_identity  equal  :", fe == fa, fe[:12], fa[:12])
