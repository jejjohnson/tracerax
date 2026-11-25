"""Dispersion: how far particles get from their start, and from each other."""

from __future__ import annotations

from typing import Any

import einx
import jax.numpy as jnp
import numpy as np
from geophysix.earth import SPHERE, Ellipsoid
from jaxtyping import Array, Bool, Float, Int

from tracerax._src._validate import is_concrete
from tracerax._src.analysis.distance import (
    Distance,
    checked_positions,
    known,
    squared_distance,
)
from tracerax.exceptions import ValidationError


__all__ = [
    "absolute_dispersion",
    "pair_dispersion",
]


def _mean_squared(
    left: Float[Array, "time member axis"],
    right: Float[Array, "time member axis"],
    counted: Bool[Array, "time member"],
    distance: Distance,
    earth: Ellipsoid,
) -> Float[Array, " time"]:
    """The mean over the counted members of d², NaN where there is none."""
    filled = "time member, time member axis, -> time member axis"
    left, right = (einx.where(filled, counted, side, 0.0) for side in (left, right))
    squared = squared_distance(left, right, distance, earth)
    zero = jnp.zeros((), squared.dtype)
    total = einx.sum("time [member]", jnp.where(counted, squared, zero))
    number = einx.sum("time [member]", counted.astype(squared.dtype))
    some = number > 0.0
    mean = total / jnp.where(some, number, 1.0)
    return jnp.where(some, mean, jnp.nan)


def _check_trajectory(trajectory: Any) -> None:
    """Raise unless the trajectory has times, particles and columns.

    Raises:
        ValidationError: If it has not.
    """
    if trajectory.ndim != 3 or trajectory.shape[0] < 1:
        raise ValidationError(
            f"'trajectory' must be (time, particle, axis), got {trajectory.shape}"
        )


def absolute_dispersion(
    trajectory: Float[Array, "time particle axis"],
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Float[Array, " time"]:
    """The mean squared distance of particles from where they started.

    A²(t) = (1/N_t) ∑_p d(x_p(t), x_p(0))², over the N_t particles whose position
    is known at t and at the start.

    Args:
        trajectory: The positions at each time, the starts first, as the
            ``trajectory`` of `tracerax.advect.integrate` [m], or (longitude,
            latitude) in degrees. A position that is NaN is missing: give a
            particle NaN before its release, or it counts as one that stays.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances,
            which are right across the date line.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        A² at each time [m²], NaN where no particle is known.

    Raises:
        ValidationError: If the trajectory has not three axes, or ``distance`` is
            unknown, or does not fit it.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import absolute_dispersion
        >>> start = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        >>> trajectory = jnp.stack([start, start + jnp.array([3.0, 4.0])])
        >>> absolute_dispersion(trajectory).tolist()
        [0.0, 25.0]

        Two steps of one degree on the equator, across the date line:

        >>> path = jnp.array([[[179.5, 0.0]], [[-179.5, 0.0]]])
        >>> spread = absolute_dispersion(path, distance="sphere")
        >>> bool(jnp.allclose(jnp.sqrt(spread[1]), 111_195.0, rtol=1e-5))
        True
    """
    _check_trajectory(trajectory)
    (trajectory,) = checked_positions(distance, earth, trajectory=trajectory)
    starts = jnp.broadcast_to(trajectory[:1], trajectory.shape)
    counted = known(trajectory) & known(starts)
    return _mean_squared(trajectory, starts, counted, distance, earth)


def pair_dispersion(
    trajectory: Float[Array, "time particle axis"],
    pairs: Int[Array, "pair 2"],
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Float[Array, " time"]:
    """The mean squared distance between the partners of each pair.

    D²(t) = (1/N_t) ∑_{(p, q)} d(x_p(t), x_q(t))², over the N_t pairs whose two
    positions are known at t.

    Args:
        trajectory: The positions at each time [m], or (longitude, latitude) in
            degrees. A position that is NaN is missing.
        pairs: The two particles of each pair. Pairs of particles that start
            close measure what two-particle theories are about; every pair of a
            cloud measures its width.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        D² at each time [m²], NaN where no pair is known.

    Raises:
        ValidationError: If the trajectory has not three axes, the pairs two
            columns, a concrete pair names a particle that is not there, or
            ``distance`` is unknown, or does not fit the trajectory.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import pair_dispersion
        >>> trajectory = jnp.array(
        ...     [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]]
        ... )
        >>> pair_dispersion(trajectory, jnp.array([[0, 1]])).tolist()
        [1.0, 25.0]
    """
    _check_trajectory(trajectory)
    (trajectory,) = checked_positions(distance, earth, trajectory=trajectory)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValidationError(f"'pairs' must be (pair, 2), got shape {pairs.shape}")
    n_particles = trajectory.shape[1]
    if is_concrete(pairs) and pairs.size:
        members = np.asarray(pairs)
        if members.min() < 0 or members.max() >= n_particles:
            raise ValidationError(
                f"'pairs' must name particles from 0 to {n_particles - 1}, "
                f"got {members.min()} to {members.max()}"
            )
    left, right = trajectory[:, pairs[:, 0]], trajectory[:, pairs[:, 1]]
    counted = known(left) & known(right)
    return _mean_squared(left, right, counted, distance, earth)
