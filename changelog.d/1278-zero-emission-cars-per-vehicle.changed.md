The UK LCFS stage now draws the fuel-buyer flag per vehicle (microcosm#1278). Each of a household's vehicles is zero-emission at the battery-electric and fuel-cell share of the DfT VEH1103 licensed-car stock, so a household with `n` vehicles buys fuel unless all `n` are zero-emission: it is flagged at `1 - (1 - rate) ** n`, against the same identity-keyed uniform as before.

- One-vehicle households keep their flag.
- Households with several vehicles are no longer all one or the other, so they buy fuel more often.
- The declaration's `rate_rule` becomes `per_car_zero_emission_share` in the stage spec, the rendered stage roster and the sources schema.
- The fuel-flag receipt adds the flagged share among vehicle households and the share of vehicles in fuel-free households. The flag is per household, so a zero-emission car beside a petrol or diesel one keeps its household's fuel spend, and the second share sits below the per-car share the draw uses. The receipt's shares on each side difference into the households with and without a vehicle, crossed with fuel or none, so every share on a side is "<10" when any of those groups that is not empty rests on one to nine distinct positive-weight sources: the frame's source households, the donor's diaries.
- Who holds the zero-emission cars by income is still random (a later change).
- The release input coverage manifest and the H2 spine fixture are regenerated.
