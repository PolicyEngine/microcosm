The UK atomic-area supports gain conditioned forms for the geography-first draw (#931), which only that move binds; today's build binds the registered supports, so no output moves.

Each conditioned support extends its system's registered support byte for byte, and adds:

- **`imd_decile`:** the area's deprivation decile within its nation, at the 2019 vintage the FRS 2024-25 deciles follow (IoD 2019, WIMD 2019, SIMD 2020v2, NIMDM 2017). Each area takes the plurality of its live residential postcodes in the May 2026 NSPL; a tie goes to the parent area's plurality, and an area without such a postcode takes its parent's decile.
- **Households by council-tax band, times 10^6, as exact integers:**
  - England and Wales take bands A to I from the area's LSOA mix in VOA CTSOP1.1; only Wales has band I.
  - Scotland takes bands A to H from its council's CTAXBASE 2025 chargeable dwellings, rounded half to even, so there the band moves mass only between councils.
  - Northern Ireland is unbanded.

`tools/build_uk_conditioned_area_supports.py` builds them from chronicle#412's raw-only registrations (PR chronicle#413), pinned by sha256. `uk/uk_atomic_area_conditioned_supports.provenance.json` records every input, the decile rule counts, the rounding and a precomputed table of zero-mass draw cells.

`sources.yaml` declares the supports under the new role `uk_atomic_area_conditioned_support`, with five new geography vintages. `spine.yaml` declares the FRS channel's `observed_constraints: [imd_decile]`, an optional field the shared spine schema now admits. `validate_uk_country_source_projection` refuses a spec in which the supports, their conditioned forms and that declaration disagree.

The country-neutral `microcosm.build.atomic_support_conditioning` provides the building blocks:

- `modal_area_codes`;
- `apportion_weights`;
- `extend_atomic_support`;
- `zero_mass_cells`.
