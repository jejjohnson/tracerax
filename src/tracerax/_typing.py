"""Shared type aliases.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+).
"""

from __future__ import annotations

from jaxtyping import Array, Float


__all__ = [
    "ScalarLike",
]


type ScalarLike = float | int | Float[Array, ""]
"""A time, a length or a rate: a Python number, or a JAX scalar, traced or not."""
