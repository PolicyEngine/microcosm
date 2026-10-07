Add engine-free transport operators: a reader for local populace-US donor
files pinned by size and SHA-256, stable source-ID seeds, one-step currency
conversion, and weighted signed quantile maps with component and
entity-aggregate support. The reader requires exactly one
`is_household_head` reference person per household. For a two-person parent
cycle, it drops the edge naming a younger parent, or both edges when ages are
equal or the donor has no age column, and returns the dropped-edge count.
Longer cycles and other inconsistent relationships are refused.
The full reader smoke test loads the pinned donor's 166,321 persons in
57,240 households, dropping two parent edges.
