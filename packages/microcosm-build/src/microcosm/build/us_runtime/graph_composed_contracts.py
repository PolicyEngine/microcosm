"""Source-free declarations shared by composed producers and their consumers.

These are the existing legacy producer IDs and artifact types. Importing their
names must not select legacy source assembly, evaluation, or country resources.
The defining producers re-export the same objects for compatibility; operations
and source authority remain in those producers.
"""

from microcosm.graph import ArtifactType

CREATE_NODE = "composed_population.prepare"
BIND_NODE = "composed_population.asec_bind"
LEAVES_NODE = "composed_population.asec_cps_carried_current"
REPORTED_INCOME_NODE = "composed_population.asec_reported_income"
US_COMPOSED_ASEC_BINDING_TYPE = ArtifactType("microcosm.us.composed_asec_binding", 1)
US_COMPOSED_ASEC_ARM_ROWS_TYPE = ArtifactType("microcosm.us.composed_asec_arm_rows", 1)
