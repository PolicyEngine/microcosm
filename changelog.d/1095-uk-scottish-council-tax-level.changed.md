The FRS council tax stage now levels Scotland's bills to the Scottish Government's own measure of the 2024-25 liability before council tax reduction (S2(a) of the November event-data plan, microcosm#1095).

- **The measure:** the sum over the 32 council areas of CTAXBASE 2025's Band D equivalent dwellings before council tax reduction at September 2024 (the middle of 2024-25), times CTAS 2026's average Band D council tax for 2024-25. Both are vendored from the pinned feed into `scotgov_band_d_liability_facts.json`.
- **The level:** one factor, the measure over the Scottish households' derived bills at the spine design weights, applied to every Scottish household's `council_tax`.
- **Why the bills and not the weights:** the bound Scottish band stock already fixes the household mix, and the gap is in the amounts. `obr.council_tax_scotland` stays a held-out diagnostic.
- **Receipts:** the stage records the measure, the stored total, the factor and the Scottish households' support as a band.
- **Synthetic frames:** a frame with no Scottish bill to level keeps its bills as derived, and the receipt says so.
