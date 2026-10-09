The four UC payment-distribution families now read DWP's monthly award bands in whole pence (uk-data#530, microcosm#1095).

**Band edges.** DWP publishes the bands as "£0.01 to £100.00", "£100.01 to £200.00", up to "£2,400.01 to £2,500.00", with each band starting a penny above its predecessor's top. Each binding now declares `band_semantics: whole_pence_upper_closed`, so a band is the upper-closed interval (label lower − £0.01, next label lower − £0.01] × 12, compared in whole pence. The half-open reading was float-fragile on those edges:

- an award of a penny a month could fall below the bottom band;
- £100.01 a month (£1,200.12 a year) stayed in the bottom band, because 100.01 × 12 rounds just above 1,200.12.

The semantics are opt-in, and an unknown value is refused. The other 28 banded bindings keep the half-open reading.

**The top band stays unbound.** `band_upper_bound` closes the last bound band at £2,500.00 a month, so it never absorbs the separately published "£2,500.01 or over" category. Binding that category was measured and left out. With it unbound, the calibration holds 2.6 times DWP's count of lone parents in it and 2.0 times its count of couples with children, and reaches OBR's Universal Credit total. Bound to DWP's counts, the total falls 5.6% short, and couples without children have no household in the band. The awards below the top band are too low for DWP's counts and OBR's total to fit together.

The generator now counts the fan-out facts it leaves unnamed, rather than declaring 12: the current feed leaves 8, the "No payment" and "or over" facts. The national surface stays at 1,231 active references.
