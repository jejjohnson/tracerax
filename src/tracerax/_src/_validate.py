"""Checks of concrete inputs, shared by the public functions."""

from __future__ import annotations

from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from tracerax.exceptions import ValidationError


__all__ = [
    "as_floating",
    "check_dimension",
    "check_count",
    "is_concrete",
]


def as_floating(value: Any) -> jax.Array:
    """The value as a floating array: integers become JAX's default float."""
    array = jnp.asarray(value)
    return array.astype(jnp.result_type(array, float))


def is_concrete(*values: Any) -> bool:
    """Whether no leaf of the values is traced, so that they can be compared."""
    leaves = jax.tree.leaves(values)
    return not any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def check_dimension(shape: tuple[int, ...], name: str) -> None:
    """Raise unless the last of the sizes is 2 or 3, the position columns.

    Raises:
        ValidationError: If it is neither, or there is none.
    """
    if not shape or shape[-1] not in (2, 3):
        raise ValidationError(f"'{name}' must have 2 or 3 columns, got shape {shape}")


def check_count(value: Any, name: str) -> None:
    """Raise unless the value is a Python integer of at least 1.

    Raises:
        ValidationError: If it is not.
    """
    integral = isinstance(value, int | np.integer) and not isinstance(value, bool)
    if not integral or value < 1:
        raise ValidationError(f"'{name}' must be an integer of at least 1, got {value}")
