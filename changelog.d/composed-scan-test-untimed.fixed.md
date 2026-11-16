Drop the wall-clock assert from
`test_precalibration_ancestry_composed_stage_scan_is_linear`. The test already
pins the composed pass's linear scan exactly (no JSON decoded from an offset);
the extra `elapsed < 1` check flaked on a loaded CI runner (2.13 s).
