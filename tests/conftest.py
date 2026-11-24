"""Shared pytest configuration.

The reference comparisons run in float64, so JAX's 64-bit mode is enabled for
the whole suite, and jaxtyping checks every annotated array at runtime through
its import hook (`python-style`). Tests marked ``gridax`` are skipped when the
extra is not installed.
"""

from __future__ import annotations

import importlib.util

import jax
import pytest
from jaxtyping import install_import_hook


jax.config.update("jax_enable_x64", True)
# Installed before pytest imports tracerax for doctest collection, so every
# tracerax module imported afterwards is checked with beartype.
install_import_hook("tracerax", "beartype.beartype")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the tests that need gridax when it is not installed."""
    if importlib.util.find_spec("gridax") is not None:
        return
    missing = pytest.mark.skip(reason="gridax is not installed")
    for item in items:
        if item.get_closest_marker("gridax") is not None:
            item.add_marker(missing)
