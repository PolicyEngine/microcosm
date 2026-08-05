Classify YTD-union zero rows as agreement-by-absence in the import-margins
API cross-check (#615): the Census API returns all-zero month rows (and
`-` totals) for pairs active earlier in the statistical year that the bulk
detail legitimately omits; these are now counted per side instead of
gating, while one-sided cells or totals carrying any nonzero measure still
gate.
