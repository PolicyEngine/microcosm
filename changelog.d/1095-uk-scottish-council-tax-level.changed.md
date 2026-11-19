The FRS council tax stage now levels Scotland's bills to the Scottish Government's own measure of the 2024-25 liability before council tax reduction (S2(a) of the November event-data plan, microcosm#1095).

- **The measure:** the sum over the 32 council areas of CTAXBASE 2025's Band D equivalent dwellings before council tax reduction at September 2024 (the middle of 2024-25), times CTAS 2026's average Band D council tax for 2024-25. Both are vendored from the pinned feed into `scotgov_band_d_liability_facts.json`.
- **The level:** one factor, the measure over the Scottish households' derived bills at the spine design weights, applied to every Scottish household's `council_tax`.
- **Why the bills and not the weights:** the bound Scottish band stock already fixes the household mix, and the gap is in the amounts. `obr.council_tax_scotland` stays a held-out diagnostic.
- **Receipts:** the stage records the measure, the stored total, the factor and the Scottish households' support as a band.
- **Synthetic frames:** a frame with no Scottish bill to level keeps its bills as derived, and the receipt says so.
- **Dwellings no FRS household occupies** (Vahid's review of #1550; the stage notes say so, signed by María on 10 November 2026):
  - The level counts the liability of every chargeable dwelling, including about 21,600 second homes and 43,500 liable empty homes in September 2024 (Scottish Government, Second Homes and Empty Properties in September 2025). They are roughly 2 to 3% of the chargeable stock.
  - So the factor may carry 2 to 5% of the level that belongs to no FRS household, depending on their bands and on the second-home and empty-home premiums.
  - Netting them out needs CTAXBASE's counts by class in Chronicle, a follow-up.
