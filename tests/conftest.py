"""Shared pytest configuration.

The reference comparisons run in float64, so JAX's 64-bit mode is enabled for
the whole suite, and jaxtyping checks every annotated array at runtime through
its import hook (`python-style`).
"""

from __future__ import annotations

import jax
from hypothesis import settings
from jaxtyping import install_import_hook


jax.config.update("jax_enable_x64", True)
# Installed before pytest imports tracerax for doctest collection, so every
# tracerax module imported afterwards is checked with beartype.
install_import_hook("tracerax", "beartype.beartype")
# The first example of a property test compiles, so a time limit per example
# fails on the compiler and not on the code.
settings.register_profile("jax", deadline=None)
settings.load_profile("jax")
