import pytest
import sys, os
import typing as ty
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
