Carry ASEC's native HRHTYPE through source pooling and exclude group quarters
from ACS housing-household targets. Historical frames without that field
remain usable, with their ASEC group quarters still counted by the roster;
that fallback now warns and the calibration diagnostics record a
household-universe receipt. A frame that carries HRHTYPE refuses a missing code
on an ASEC household or on any household without TYPEHUGQ, and refuses a
household that carries both codes.
