"""PolicyEngine-UK adapter import/protocol behavior."""

# ruff: noqa: F401

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import RulesEngine
from microcosm.frame.adapters.policyengine_uk import (
    UK_SCHEMA,
    PolicyEngineUKEngine,
)

__all__ = [name for name in globals() if not name.startswith("__")]
