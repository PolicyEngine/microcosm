"""2023-24 property income of Scottish taxpayers by NSND band (microcosm#1326).

A project tool for one external request, kept off main. It reads a released UK
national file and the national target registry of the build that produced it,
and writes private outputs outside the repository:

1. a one-year back-cast of the 2024-25 file with policyengine-uk's own growth
   indices (``backcast``);
2. a bounded reweight to the SPI 2023-24 facts the build already binds
   (``registry``, ``measures``, ``rake``);
3. the two release tables with the records behind every band, and their
   disclosure statuses (``bands``, ``release``);
4. an illustrative 1p on the Scottish rates charged on property income
   (``reform``).

Run it with ``python -m tools.uk_scotland_property_income --help``.
"""
