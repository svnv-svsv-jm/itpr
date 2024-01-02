import pytest
import sys
import os
import typing as ty
from loguru import logger

from itpr.tables import TABLE
from itpr import calculate_ITPR


def test_itpr() -> None:
    """Test `calculate_ITPR`."""
    v = calculate_ITPR(TABLE["id"], TABLE["age_1"])
    expected = 1
    assert v == pytest.approx(expected, rel=1e-2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
