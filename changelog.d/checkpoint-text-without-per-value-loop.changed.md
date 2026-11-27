Frame checkpoints (`microcosm.build.frame_checkpoint`) now write and read a text column, string index or strata whose values are all strings, none missing, without a Python loop over each value. They produce and parse the same `object_scalars_v1` bytes as before: a tag byte before each value's UTF-8 bytes, and the end offset of every value after a leading 0.

- **pyarrow-backed `str` and `string` columns:** written from the column's Arrow buffers with NumPy, after widening every chunk to 64-bit offsets.
- **Object and python-backed columns:** a value takes the new path only if its type is exactly `str` or `np.str_`.
  - Values are encoded with `str.encode` through `map`, 8,192 at a time. Each batch is joined with `bytes.join`, and the batches are joined once.
  - Arrow is not used for these values, because it attaches a cached UTF-8 copy to every non-ASCII Python string it reads.
- **Reading:** checks the offsets as before, then moves the text in place in the array HDF5 fills, about 1 MiB at a time, dropping the tag bytes. It checks UTF-8 with Arrow and restores the declared dtype, string storage and NA marker.

These still use the per-value code, which raises the same errors as before:
- columns with a missing value, a value that is not a string, or a `str` subclass;
- a string that cannot be encoded as UTF-8, or a stored value that is not valid UTF-8;
- offsets outside the payload;
- a pyarrow-backed column that pandas refuses to build from one Arrow array.

Checkpoint files, format versions 2 and 3, and recorded file digests are unchanged. Loaded frames hold the same values, dtypes, string storage and NA markers.

Peak memory, measured with `tracemalloc`:
- **Writing Python strings:** at most the per-value code's (0.96–1.00 times).
- **Reading object and python-backed columns:** within 1% of before.
- **pyarrow-backed columns:** both directions now hold the text in NumPy arrays instead of Python strings plus an Arrow copy.

Timings on a synthetic copy of the 15 text series of one post-clone US base checkpoint (9.85 million values), on a 10-core Mac at load 14–17:
- **Converting the text:** writing went from 8.78 s to 0.59 s, and reading from 4.89 s to 0.39 s.
- **A whole 253 MB checkpoint of those series:** writing went from 7.76 s to 1.02 s, and loading from 6.67 s to 1.86 s.
