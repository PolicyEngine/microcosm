"""Table-by-table exact comparison of two processed CPS ASEC HDF stores."""

import json
import sys

import pandas as pd

a, b = sys.argv[1], sys.argv[2]
out = {}
with pd.HDFStore(a, "r") as sa, pd.HDFStore(b, "r") as sb:
    ka, kb = sorted(sa.keys()), sorted(sb.keys())
    out["keys_equal"] = ka == kb
    out["keys"] = [ka, kb]
    for k in sorted(set(ka) & set(kb)):
        da, db = sa[k], sb[k]
        r = dict(
            shape=[list(da.shape), list(db.shape)],
            columns_equal_ordered=list(da.columns) == list(db.columns),
            only_a=sorted(set(da.columns) - set(db.columns)),
            only_b=sorted(set(db.columns) - set(da.columns)),
            index_name=[da.index.name, db.index.name],
            index_equal=bool(da.index.equals(db.index)),
            dtype_diffs={
                c: [str(da[c].dtype), str(db[c].dtype)]
                for c in set(da.columns) & set(db.columns)
                if str(da[c].dtype) != str(db[c].dtype)
            },
        )
        try:
            pd.testing.assert_frame_equal(da, db, check_exact=True)
            r["frame_equal_exact"] = True
        except AssertionError as e:
            r["frame_equal_exact"] = False
            r["first_diff"] = str(e)[:600]
            if da.shape == db.shape and list(da.columns) == list(db.columns):
                r["value_diff_columns"] = [
                    c for c in da.columns if not da[c].equals(db[c])
                ][:50]
        out[k] = r
print(json.dumps(out, indent=1, default=str))
