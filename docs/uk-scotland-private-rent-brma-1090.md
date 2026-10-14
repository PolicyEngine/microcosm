# Scottish private rents: BRMA facts and council targets (#1090)

ONS's Price Index of Private Rents (PIPR) publishes Scottish rent levels for the
18 Broad Rental Market Areas (area codes `S33000001` to `S33000018`). The
`ons.rent.private_rent` target is bound at local-authority grain, where each
cell is 12 × the area's mean monthly rent × its private-renter households
(microcosm#355). Scotland has no council-grain rent facts, so its 32 cells are
signed deferred under `private_rent_pipr_scotland_brma_grain`.

This note records the measured overlap between the two geographies and the
decision it supports: BRMA rents are not translated into council targets.

## The overlap table

`uk/scotland_brma_la_overlap.json` holds households for every BRMA × council
intersection. Each 2022 census output area is placed in one BRMA by its
population-weighted centroid in the Scottish Government's BRMA polygons. Rent
Service Scotland's postcode lookup (FOI 202300368850) supplies the BRMA of nine
centroids outside every polygon and corrects 61 output areas around Balloch
where the newer lookup contradicts the 2015 polygons. Output areas nest in 2022
electoral wards, and wards nest in council areas.

The table carries two weights and says which is which:

- `private_rented_households` comes from Census 2022 tenure by ward, the finest
  level at which the census releases the private-rented split. A ward that
  crosses a BRMA boundary is split by its share of output-area households, so
  within those 36 wards private renters are assumed to be spread like all
  households. They hold 22,821 of the 322,995 private-rented households.
- `all_households` sums output-area household counts directly and needs no
  such assumption. Both weights occupy the same 63 cells and put every council
  in the same main BRMA.

Rebuild it from the five sha-pinned originals (the ward table is a manual
export; the tool's `SOURCES` gives each URL):

```bash
uvx --with geopandas --with pandas --with openpyxl --with pyogrio \
    python tools/build_uk_scotland_brma_la_overlap.py --cache <dir> --check
```

`UK_SCOTLAND_BRMA_LA_CACHE=<dir>` runs the same comparison as a test wherever
the originals are staged. PR CI pins the cells by digest and checks the
invariants listed in `test_uk_scotland_brma_la_overlap.py`.

Limits: one centroid places a whole output area; the polygons date from 2015,
the postcode lookup from 2023 and the census from 2022; census cells are
perturbed for disclosure control.

## What the table shows

- 18 of the 32 councils fall in more than one BRMA, and 16 of the 18 BRMAs
  cover more than one council.
- 29 councils and 17 BRMAs form one connected block. The three Ayrshire
  councils and the Ayrshires BRMA are the only group that separates, so no
  grouping finer than that makes BRMAs a union of whole councils.
- Much of the joining is slivers. Dropping an intersection that is under 1% of
  both its council and its BRMA leaves 11 blocks, but five BRMAs and seven
  councils still sit in one of them.
- Four councils coincide with one BRMA at 99% of private-rented households in
  both directions: Dumfries and Galloway, Fife, West Lothian and Scottish
  Borders.
- 14 councils sit wholly inside one BRMA; 11 of them share that BRMA with
  another such council.

## Decision: no translation into council targets

The candidate rule was the mean of BRMA rents weighted by each council's
private-rented households in each BRMA. It does not satisfy the parity
concern's fence, "preserve source geography without allocating BRMA averages
over local authorities", for three reasons.

1. **It is the allocation the fence names.** A council wholly inside one BRMA
   gets that BRMA's average unchanged, and a split council gets a blend of
   averages. Councils sharing a BRMA therefore get the same rent, whatever
   their own rents are. The parity register already rules out the same move at
   constituency grain (`local_devolved_constituency_rent_anchors`: a signed
   absence is preferred to an allocated local target).
2. **The differences it erases are large.** Rent Service Scotland's market
   evidence for the year to September 2021 (FOI 202200303624: 31,833 lettings
   with a BRMA, a rent and a postcode district) gives each council's own mean
   rent. The rule's value is up to 30.9% away from it:

   | Council | BRMA | Rule ÷ council's own mean |
   |---|---|---|
   | Midlothian | Lothian | +30.9% |
   | Clackmannanshire | Forth Valley | +26.8% |
   | East Lothian | Lothian | +23.1% |
   | Falkirk | Forth Valley | +20.5% |
   | Stirling | Forth Valley | −15.2% |
   | Orkney Islands | Highland and Islands | +13.6% |
   | Na h-Eileanan Siar | Highland and Islands | +13.3% |
   | East Renfrewshire | Greater Glasgow, Renfrewshire/Inverclyde | −12.5% |
   | Angus | Dundee and Angus | +11.1% |

   Across the 31 councils with at least 30 records, the median absolute gap is
   3.3%; 12 exceed 5% and 9 exceed 10%. The four coincident councils are
   within 0.1%. `experiments/1090-scotland-brma-la-rent-dispersion.py` rebuilds
   the receipt. Records are placed in councils by postcode district, which is
   approximate, and they are rent officers' market evidence rather than PIPR,
   so the receipt measures the size of the differences, not council rents.
3. **The cross-grain operator cannot carry it.** `CrossGrainBridge` declares
   that a set of higher-grain contract targets is the same additive total as a
   lower side, and `UK_CROSS_GRAIN_RULE` rescales lower rows onto that control
   over legs that partition the areas (`country > region > constituency > la`).
   A BRMA rent is a mean, not a total; BRMA is not a grain of the rule; and
   `leg_of_area` gives each council one leg, which 18 councils do not have.

The one thing the rule does preserve is the rent bill: renters × rent summed
over councils equals the same sum over BRMAs. That is a property of the totals
at BRMA grain, which is where the facts should bind.

## What would lift the deferral

- **BRMA target rows.** Keep PIPR's geography on the target side (12 × BRMA
  mean × BRMA private renters) and aggregate the model side with the overlap
  shares. This uses all 18 published rents and allocates nothing. The local
  surface has two area types today (`constituency`, `la`), so it needs a third
  through the rowwise matrix, the support checks and the receipts.
- **A signed exception for the four coincident councils.** For these the BRMA
  rent is the council's own rent up to the 1% on each side that lies elsewhere.
  Binding them lets a fact on the BRMA boundary set reach a local-authority
  cell, which the boundary-frame gate added in #795 refuses by design, so it
  needs a maintainer's ruling rather than a translation rule.

`test_the_surface_still_has_no_brma_grain` fails when either route lands, so
the concern's reason and fence are revisited then.
