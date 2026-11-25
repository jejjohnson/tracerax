"""Positions in longitude and latitude, and velocities as vectors on the sphere."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from functax.geometry import lonlat_to_unit, tangent_to_cartesian, unit_to_lonlat
from geophysix.earth import Ellipsoid
from jaxtyping import Array, Bool, Float

from tracerax._src._validate import is_concrete
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "angular_rate",
    "check_lonlat",
    "check_sphere",
    "checked_lonlat",
    "normalise",
    "to_lonlat",
    "to_unit",
]


_LATITUDES = "must hold latitudes between −90° and 90° in its second column"


def check_sphere(earth: Ellipsoid) -> None:
    """Raise when the Earth's figure is known not to be a sphere.

    Raises:
        ValidationError: If it has a flattening.
    """
    if not isinstance(earth, Ellipsoid):
        raise ValidationError(
            f"'earth' must be a geophysix Ellipsoid, got {type(earth).__name__}"
        )
    if is_concrete(earth.flattening) and float(earth.flattening) != 0.0:
        raise ValidationError(
            f"'earth' must be a sphere, got flattening {earth.flattening}"
        )


def check_lonlat(
    position: Float[Array, "*batch axis"],
    name: str,
    active: Bool[Array, "*batch"] | None = None,
) -> None:
    """Raise unless the columns are a longitude and a latitude, in degrees.

    A latitude beyond a pole is refused when concrete: it is usually the two
    columns in the other order. Only what is active is looked at.

    Raises:
        ValidationError: If there are not two columns, or a concrete latitude is
            outside [−90°, 90°].
    """
    if position.ndim == 0 or position.shape[-1] != 2:
        raise ValidationError(
            f"'{name}' must have 2 columns, longitude and latitude in degrees, "
            f"got shape {position.shape}"
        )
    if not is_concrete(position, active):
        return
    latitude = np.abs(np.asarray(position)[..., 1])
    if active is not None:
        latitude = latitude[np.asarray(active)]
    if (latitude > 90.0).any():
        raise ValidationError(
            f"'{name}' {_LATITUDES}, got {np.nanmax(latitude)} in size"
        )


def checked_lonlat(
    position: Float[Array, "*batch axis"],
    name: str,
    active: Bool[Array, "*batch"] | None = None,
) -> Float[Array, "*batch axis"]:
    """The positions, checked as by `check_lonlat` and, if traced, when they run.

    The check of traced positions is a host callback, so it stays out of scans.

    Raises:
        ValidationError: As `check_lonlat` does.
        EquinoxRuntimeError: If a traced latitude is outside [−90°, 90°].
    """
    check_lonlat(position, name, active)
    if is_concrete(position, active):
        return position
    beyond = jnp.abs(position[..., 1]) > 90.0
    if active is not None:
        beyond = beyond & active
    return eqx.error_if(position, beyond, f"'{name}' {_LATITUDES}")


def to_unit(lonlat: Float[Array, "... 2"]) -> Float[Array, "... 3"]:
    """r = (cos φ cos λ, cos φ sin λ, sin φ), from degrees.

    A latitude beyond a pole gives NaN, where the formula would give another place.
    """
    latitude = lonlat[..., 1]
    latitude = jnp.where(jnp.abs(latitude) > 90.0, jnp.nan, latitude)
    return lonlat_to_unit(lonlat.at[..., 1].set(latitude), unit="degrees")


def to_lonlat(vector: Float[Array, "... 3"]) -> Float[Array, "... 2"]:
    """(λ, φ) in degrees, with λ in [−180°, 180°), of a vector of any length."""
    lonlat = unit_to_lonlat(vector, unit="degrees")
    # half open, as the cells and the box are
    longitude = jnp.where(
        lonlat[..., 0] >= 180.0, lonlat[..., 0] - 360.0, lonlat[..., 0]
    )
    return lonlat.at[..., 0].set(longitude)


def normalise(vector: Float[Array, "particle 3"]) -> Float[Array, "particle 3"]:
    """r / |r|, for vectors that are not zero."""
    norm = jnp.sqrt(einx.sum("particle [xyz]", vector**2))
    return einx.divide("particle xyz, particle -> particle xyz", vector, norm)


def angular_rate(
    velocity: Velocity,
    time: ScalarLike | Float[Array, " particle"],
    vector: Float[Array, "particle 3"],
    radius: ScalarLike,
) -> Float[Array, "particle 3"]:
    """dr/dt = (u ê_λ + v ê_φ) / R, the velocity read below each vector [s⁻¹].

    It depends on the direction of the vector and not on its length. Within √ε of
    a pole, with ε the machine epsilon (0.1 m in float64, 2 km in float32), it is
    read √ε from it.
    """
    lonlat = to_lonlat(_off_the_poles(vector))
    east_north = velocity(time, lonlat).astype(vector.dtype)
    # (u ê_λ + v ê_φ) / R
    spatial = tangent_to_cartesian(lonlat, east_north, unit="degrees")
    return spatial / jnp.asarray(radius, vector.dtype)


def _off_the_poles(vector: Float[Array, "particle 3"]) -> Float[Array, "particle 3"]:
    """The vectors, those nearer than √ε to a pole moved to that distance from it.

    A longitude changes by 1/cos φ for a step eastward, so at a pole a derivative
    through it is the difference of two numbers of size 1/ε. At √ε from the pole
    the rounding and the move each cost √ε. The move is along the meridian and has
    no derivative of its own, so the derivative is that of the moved point.
    """
    distance = jnp.sqrt(jnp.finfo(vector.dtype).eps)
    fixed = jax.lax.stop_gradient(vector)
    x, y = fixed[:, 0], fixed[:, 1]
    horizontal = jnp.hypot(x, y)
    least = distance * jnp.sqrt(einx.sum("particle [xyz]", fixed**2))
    near = horizontal < least
    # on the axis itself every meridian will do
    on_axis = horizontal == 0.0
    size = jnp.where(on_axis, 1.0, horizontal)
    along_x = jnp.where(on_axis, 1.0, x / size)
    along_y = jnp.where(on_axis, 0.0, y / size)
    # d (cos λ, sin λ) − (x, y)
    move_x = jnp.where(near, least * along_x - x, 0.0)
    move_y = jnp.where(near, least * along_y - y, 0.0)
    moved = vector.at[:, 0].add(move_x.astype(vector.dtype))
    return moved.at[:, 1].add(move_y.astype(vector.dtype))
