import pytest
import sys, os
import typing as ty
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itpr.tables import TABLE
from itpr import calculate_ITPR


def test_itpr() -> None:
    """Test `calculate_ITPR`."""
    v = calculate_ITPR(TABLE["id"], TABLE["age_1"], direct=True)
    expected = 1
    assert v == pytest.approx(expected, rel=1e-2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
