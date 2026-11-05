"""Geographic clones of a UK test frame, made as the dense pool makes them.

The FRS-trained fits fold geographic clones into each survey record
(microcosm#931); a replicated fixture holds them to it. Two clones at half the
weight keep every weight exact, so the folded training rows carry the
original weights bit for bit.
"""

from __future__ import annotations

from microcosm.build.uk_runtime.national_frame import uk_time_period
from microcosm.build.uk_runtime.rowwise_dataset import expand_uk_geographic_pool
from microcosm.frame import Frame


def with_geographic_clones(frame: Frame, *, n_clones: int = 2) -> Frame:
    """``frame`` expanded to ``n_clones`` geographic clones per household.

    Clone 0 keeps every id, so its rows draw on the same keys as the frame's.
    """

    household = frame.table("household").copy()
    household["household_weight"] = frame.weights_for("household").values
    return expand_uk_geographic_pool(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        n_clones=n_clones,
        time_period=uk_time_period(frame),
        household_weight_kind=frame.weights_for("household").kind,
        mass_log=frame.mass_log,
    ).frame
