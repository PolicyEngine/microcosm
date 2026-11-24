The graph executor no longer deep-copies the tables a kernel reads (microcosm#1679). Each kernel gets new pandas objects whose buffers are read-only views of the memory they are projected from: the stored population's own memory for a node that reads every row, and the one copy `.loc` makes of the selected rows for a node with a row mask. Kernels see the same columns, order, dtypes, index, values and `attrs`, so no node key or output changes. The saving is in UK graph builds; the US release and base tools do not run the graph executor.

Measured with `tracemalloc` on an invented 1,000,000-row person table, 23 NumPy and nullable columns (131.6 MiB):

- a node reading every row: the context held 154.5 MiB before and 7.7 MiB after (the weights copy), and the projection peaked at 170.8 MiB before and 24.9 MiB after;
- a node reading half the rows: the context held 77.3 MiB before and 73.5 MiB after, and the projection peaked at 136.4 MiB before and 89.7 MiB after.

A pandas write into a context table, and an item write into an array a column hands out (`to_numpy()`, `.values`, `.array`, `np.asarray`), raises or changes only the kernel's own objects. `np.<ufunc>.at` ignores the read-only flag and writes into the population's memory; the executor's before/after comparison of the context rejects the node before its result is applied or stored, and the run stops. An item write into `np.asarray(table.index)` no longer reaches the population's index. Graph amendment 31 in `docs/graph-acceptance.md` records the design.
