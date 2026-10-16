# Scottish private rents: BRMA facts and council targets (#1090)

ONS's Price Index of Private Rents (PIPR) publishes Scottish rent levels for the
18 Broad Rental Market Areas (area codes `S33000001` to `S33000018`). The
`ons.rent.private_rent` target is bound at local-authority grain, where each
cell is 12 × the area's mean monthly rent × its private-renter households
(microcosm#355). Scotland has no council-grain rent facts, so its 32 cells are
signed deferred under `private_rent_pipr_scotland_brma_grain`.

This note records the measured overlap between the two geographies and the
decision it supports: BRMA rents are not translated into council targets.
microcosm#1092 measured the overlap first; microcosm#1262 rebuilt it on the
area-to-BRMA layer and Census 2022 output-area counts.

## The overlap table

`uk/scotland_brma_la_overlap.json` holds households for every BRMA × council
intersection:

- each 2022 census output area's BRMA comes from the committed area-to-BRMA
  layer (`uk/brma_area_layer.json`, grain `scotland_oa22`). The layer places
  an output area by its population-weighted centroid in the Scottish
  Government's BRMA polygons. Rent Service Scotland's postcode lookup
  (FOI 202300368850) supplies the BRMA of nine centroids outside every polygon
  and corrects the output areas around Balloch where the newer lookup
  contradicts the 2015 polygons (`tools/build_uk_brma_area_layer.py` records
  each one). The overlap and the households' BRMAs are therefore one copy of
  one method;
- its council comes from the committed Scotland atomic support, the one that
  assigns households their local authority;
- both weights are Scotland's Census 2022 UV404 counts by output area, read
  from the pinned Chronicle feed (PolicyEngine/chronicle#326):
  `private_rented_households` is "Private rented: Total" (a private landlord
  or letting agency, plus other private rented) and `all_households` is all
  occupied households. Output areas nest in both geographies, so each cell is
  an exact sum and no within-area assumption is made. UV404 prints `-` for an
  empty cell and the feed carries no fact for it; 4,910 output areas have no
  private renters and count zero.

microcosm#1092 split the census ward tenure table, then the finest release of
the private-rented split, across BRMAs by each ward's output-area households.
The rebuilt table has the same 63 cells and the same 2,508,542 households;
its private-rented total is 322,903 against the ward split's 322,995 (the
census perturbs every cell). The ward split put a few renters in one sliver
(Greater Glasgow × West Dunbartonshire) that holds none.

Rebuild or check it from the pinned feed and the mounted supports:

```bash
uv run --no-sync python tools/build_uk_scotland_brma_la_overlap.py \
    --ledger-facts <feed artifact>/consumer_facts.jsonl \
    --supports <directory of the atomic support NPZs> --check
```

`test_uk_scotland_brma_la_overlap.py` runs the same check wherever the feed
and the supports are present; PR CI pins the cells by digest and checks the
invariants listed in its docstring.

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

   | Council | Main BRMA | Rule ÷ council's own mean |
   |---|---|---|
   | Midlothian | Lothian | +30.9% |
   | Clackmannanshire | Forth Valley | +26.7% |
   | East Lothian | Lothian | +23.1% |
   | Falkirk | Forth Valley | +20.5% |
   | Stirling | Forth Valley | −15.2% |
   | Orkney Islands | Highland and Islands | +13.6% |
   | Na h-Eileanan Siar | Highland and Islands | +13.3% |
   | East Renfrewshire | Greater Glasgow | −11.8% |
   | Angus | Dundee and Angus | +11.1% |

   Across the 31 councils with at least 30 records, the median absolute gap is
   3.3%; 12 exceed 5% and 9 exceed 10%. The four coincident councils are
   within 0.1%. `experiments/1090-scotland-brma-la-rent-dispersion.py` rebuilds
   the receipt from Chronicle's raw registrations, the layer and the feed.
   Records are placed in councils by postcode district, which is approximate,
   and they are rent officers' market evidence rather than PIPR, so the
   receipt measures the size of the differences, not council rents.
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
  mean × BRMA private renters) and sum the model side by each household's
  BRMA. This uses all 18 published rents and allocates nothing. The local
  surface has two area types today (`constituency`, `la`), so it needs a third
  through the rowwise matrix, the support checks and the receipts.
- **A signed exception for the four coincident councils.** For these the BRMA
  rent is the council's own rent up to the 1% on each side that lies elsewhere.
  Binding them lets a fact on the BRMA boundary set reach a local-authority
  cell, which the boundary-frame gate added in #795 refuses by design, so it
  needs a maintainer's ruling rather than a translation rule.

microcosm#1262 takes the first route: placed households now carry the BRMA of
their assigned output area, so the model side needs no overlap shares.
