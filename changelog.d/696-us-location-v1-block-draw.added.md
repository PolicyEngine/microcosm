Add US household location v1 (`us_runtime.block_location`), the rule Max set on 27 September 2026.

Each household draws one 2020 census block, in proportion to block population, within the finest geography its source provides. That is the PUMA for ACS records and the identified county for CPS records. A CPS record with no county code draws within its state, outside the counties that Census's List 4 for that ASEC year names and that the year's file actually codes. For a year whose List 4 drops the whole-county guarantee (ASEC 2026), it draws from the whole state.

Every other geography is then looked up from the block: tract, county, place, SLDU/SLDL, CBSA, PUMA, and the district of every attached congressional plan. Draws are keyed by household id, so they do not depend on row order. The draw also supports K clones at weight/K.

Each US line gains `--location-rule block_v1`. The default, `legacy`, stays byte-identical.

The block ladder builder now adds a per-block `puma` array; the schema-1 loader ignores it. Census's identified-county lists for ASEC 2023-2026 are packaged with their PDF digests and a per-year whole-county-guarantee flag.
