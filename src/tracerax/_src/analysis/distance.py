"""Distances between positions, on the plane and on the sphere."""

from __future__ import annotations

from typing import Any, Literal

import einx
import jax.numpy as jnp
from geophysix.earth import Ellipsoid, great_circle_distance
from jaxtyping import Array, Bool, Float
from pipekit_array.arrays import safe_sqrt

from tracerax._src._sphere import check_sphere, checked_lonlat
from tracerax._src._validate import as_floating
from tracerax.exceptions import ValidationError


__all__ = [
    "Distance",
    "checked_positions",
    "distance_between",
    "known",
    "squared_distance",
]


type Distance = Literal["euclidean", "sphere"]
"""How far apart two positions are: in their own frame, or along a great circle."""

_KINDS = ("euclidean", "sphere")


def checked_positions(kind: str, earth: Ellipsoid, **positions: Any) -> tuple[Any, ...]:
    """The positions as floating arrays, checked to fit a known distance.

    Raises:
        ValidationError: If the kind is unknown, or the positions of a sphere are
            not longitudes and latitudes, or ``earth`` is not a sphere.
        EquinoxRuntimeError: If a traced latitude is beyond a pole.
    """
    if kind not in _KINDS:
        raise ValidationError(f"'distance' must be one of {list(_KINDS)}, got {kind!r}")
    arrays = {name: as_floating(position) for name, position in positions.items()}
    if kind == "sphere":
        check_sphere(earth)
        arrays = {
            name: checked_lonlat(position, name) for name, position in arrays.items()
        }
    return tuple(arrays.values())


def known(position: Float[Array, "*batch axis"]) -> Bool[Array, "*batch"]:
    """Whether every column of each position is finite; NaN is a missing one."""
    return einx.all("... [axis]", jnp.isfinite(position))


def squared_distance(
    left: Float[Array, "*batch axis"],
    right: Float[Array, "*batch axis"],
    kind: Distance,
    earth: Ellipsoid,
) -> Float[Array, "*batch"]:
    """d² between two sets of finite positions [m²]."""
    if kind == "euclidean":
        return einx.sum("... [axis]", (left - right) ** 2)
    return _great_circle(left, right, earth) ** 2


def distance_between(
    left: Float[Array, "*batch axis"],
    right: Float[Array, "*batch axis"],
    kind: Distance,
    earth: Ellipsoid,
) -> Float[Array, "*batch"]:
    """d between two sets of finite positions [m], with gradient 0 where d = 0."""
    if kind == "sphere":
        return _great_circle(left, right, earth)
    squared = squared_distance(left, right, kind, earth)
    # √, with a zero slope at 0; NaN stays
    return safe_sqrt(squared, jnp)


def _great_circle(
    left: Float[Array, "*batch axis"],
    right: Float[Array, "*batch axis"],
    earth: Ellipsoid,
) -> Float[Array, "*batch"]:
    return great_circle_distance(
        left[..., 0], left[..., 1], right[..., 0], right[..., 1], earth=earth
    )
