"""This file is run by PyTest as first file. Define fixtures here.
"""
import pytest
import os
import typing as ty
import pyrootutils

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)
