import pytest
import sys
import os
import typing as ty
from loguru import logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itpr import calculate_entropy


def _uniform_x(n: int, n_values: int) -> pd.Series:
    """Create uniform X."""
    x = []
    for i in range(n_values):
        x += [i] * n
    return pd.Series(x)


def test_calculate_entropy_plot() -> None:
    """Test `calculate_entropy()` with plot."""
    entropies = []
    n = 50
    values = list(range(50))
    for n_values in values:
        h = calculate_entropy((_uniform_x(n, n_values)))
        entropies.append(h)
    # Plot
    plt.plot(values, entropies, linestyle="--", marker="*")
    plt.xlabel("Number of unique values [-]")
    plt.ylabel("Entropy [-]")
    plt.savefig(os.path.join("pytest_artifacts", "entropies.png"))
    # plt.show()


def test_calculate_entropy() -> None:
    """Test `calculate_entropy()`."""
    # Entropy of constant
    H = calculate_entropy(_uniform_x(10, 1))
    logger.info(f"Entropy: {H}")
    assert abs(H) < 1e-12
    # Entropy of uniform
    n = 100
    H = calculate_entropy(_uniform_x(n, 2))
    logger.info(f"Entropy: {H}")
    assert abs(H) == pytest.approx(1, abs=1e-6)
    H = calculate_entropy(_uniform_x(n, 3))
    logger.info(f"Entropy: {H}")
    assert abs(H) == pytest.approx(1.58, abs=1e-2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
