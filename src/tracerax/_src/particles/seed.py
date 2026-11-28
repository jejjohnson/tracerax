"""Releases: particles at a point, along a line or in a box, and when they start."""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray
from pipekit_ops import is_concrete

from tracerax._src._validate import as_floating, check_count, check_dimension
from tracerax._src.advect.turbulence import TurbulenceModel
from tracerax._src.particles.state import ParticleState
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "along_line",
    "at_point",
    "in_box",
    "release_times",
    "with_turbulent_velocity",
]


def _corner(value: Any, name: str) -> Float[Array, " axis"]:
    """A point as a floating array: integers become JAX's default float.

    Raises:
        ValidationError: If it is not (2,) or (3,).
    """
    # an integer point promotes to the default float; a float one keeps its dtype
    point = as_floating(value)
    check_dimension(point.shape, name)
    if point.ndim != 1:
        raise ValidationError(f"'{name}' must be one point, got shape {point.shape}")
    return point


def _two_corners(
    first: Any, second: Any, names: tuple[str, str]
) -> tuple[Float[Array, " axis"], Float[Array, " axis"]]:
    """Two points of the same dimension and dtype.

    Raises:
        ValidationError: If either is not (2,) or (3,), or the two differ.
    """
    start, stop = _corner(first, names[0]), _corner(second, names[1])
    if start.shape != stop.shape:
        raise ValidationError(
            f"'{names[0]}' and '{names[1]}' must have the same shape, "
            f"got {start.shape} and {stop.shape}"
        )
    dtype = jnp.result_type(start, stop)
    return start.astype(dtype), stop.astype(dtype)


def _midpoints(n: int, dtype: Any) -> Float[Array, " particle"]:
    """(p + ½) / N for p = 0, …, N − 1."""
    return (jnp.arange(n, dtype=dtype) + 0.5) / n


def _released(
    position: Float[Array, "particle axis"], mass: Any, release_time: Any
) -> ParticleState:
    """The state of a release whose total mass is split equally.

    Raises:
        ValidationError: If the mass is negative or not a scalar, or the release
            times are neither one nor one per particle.
    """
    n_particles, dtype = position.shape[0], position.dtype
    if jnp.shape(mass) != ():
        raise ValidationError(
            f"'mass' must be the release's total, a scalar, got {jnp.shape(mass)}"
        )
    if is_concrete(mass) and not np.asarray(mass) >= 0.0:
        raise ValidationError(f"'mass' must be non-negative, got {mass}")
    if jnp.shape(release_time) not in ((), (n_particles,)):
        raise ValidationError(
            f"'release_time' must be a scalar or have one entry per particle, "
            f"({n_particles},), got {jnp.shape(release_time)}"
        )
    per_particle = (n_particles,)
    # m_p = M / N
    share = jnp.asarray(mass, dtype=dtype) / n_particles
    return ParticleState(
        position=position,
        velocity=jnp.zeros_like(position),
        mass=jnp.broadcast_to(share, per_particle),
        release_time=jnp.broadcast_to(
            jnp.asarray(release_time, dtype=dtype), per_particle
        ),
        active=jnp.ones(per_particle, dtype=bool),
    )


def at_point(
    point: Float[ArrayLike, " axis"],
    n: int,
    *,
    mass: ScalarLike = 1.0,
    release_time: ScalarLike | Float[ArrayLike, " particle"] = 0.0,
) -> ParticleState:
    """N particles at one point.

    x_p = x₀, m_p = M / N

    Args:
        point: x₀, (x, y) or (x, y, z) [m].
        n: N, the number of particles.
        mass: M, the total of the release [kg], split equally.
        release_time: t_r, one for all or one per particle, in seconds from the
            run's epoch. `release_times` spreads them over an interval.

    Returns:
        The state, with zero velocity and every particle active.

    Raises:
        ValidationError: If the point has not 2 or 3 columns, ``n`` is below 1,
            or the mass is negative.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import at_point
        >>> state = at_point(jnp.array([0.0, 0.0, 10.0]), 100, mass=5.0)
        >>> state.position.shape, bool(jnp.allclose(state.mass.sum(), 5.0))
        ((100, 3), True)

        A puff released at 60 s:

        >>> puff = at_point(jnp.array([0.0, 0.0]), 10, release_time=60.0)
        >>> bool((puff.release_time == 60.0).all())
        True
    """
    check_count(n, "n")
    source = _corner(point, "point")
    position = einx.id("axis -> particle axis", source, particle=n)
    return _released(position, mass, release_time)


def along_line(
    start: Float[ArrayLike, " axis"],
    stop: Float[ArrayLike, " axis"],
    n: int,
    *,
    mass: ScalarLike = 1.0,
    release_time: ScalarLike | Float[ArrayLike, " particle"] = 0.0,
) -> ParticleState:
    """N particles at the middles of N equal pieces of a line.

    x_p = start + (p + ½)/N (stop − start), so that the ends of the line are not
    sampled more than its inside.

    Args:
        start: One end of the line [m].
        stop: The other end [m].
        n: N, the number of particles.
        mass: M, the total of the release [kg], split equally.
        release_time: t_r, one for all or one per particle [s].

    Returns:
        The state, with zero velocity and every particle active.

    Raises:
        ValidationError: If an end has not 2 or 3 columns, the two ends differ
            in dimension, ``n`` is below 1, or the mass is negative.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import along_line
        >>> state = along_line(jnp.array([0.0, 0.0]), jnp.array([8.0, 0.0]), 4)
        >>> bool(
        ...     jnp.allclose(state.position[:, 0], jnp.array([1.0, 3.0, 5.0, 7.0]))
        ... )
        True
    """
    check_count(n, "n")
    begin, end = _two_corners(start, stop, ("start", "stop"))
    along = _midpoints(n, begin.dtype)
    # x_p = start + (p + ½)/N (stop − start)
    offset = einx.multiply("particle, axis -> particle axis", along, end - begin)
    return _released(begin + offset, mass, release_time)


def in_box(
    lower: Float[ArrayLike, " axis"],
    upper: Float[ArrayLike, " axis"],
    n: int,
    *,
    mass: ScalarLike = 1.0,
    release_time: ScalarLike | Float[ArrayLike, " particle"] = 0.0,
    key: PRNGKeyArray,
) -> ParticleState:
    """N particles drawn uniformly in a box.

    x_p = lower + ξ_p (upper − lower), ξ_p ~ 𝒰[0, 1)

    Args:
        lower: The lowest corner [m].
        upper: The highest corner [m].
        n: N, the number of particles.
        mass: M, the total of the release [kg], split equally.
        release_time: t_r, one for all or one per particle [s].
        key: The PRNG key of the draws.

    Returns:
        The state, with zero velocity and every particle active.

    Raises:
        ValidationError: If a corner has not 2 or 3 columns, the two differ in
            dimension, ``lower`` is not below ``upper``, ``n`` is below 1, or
            the mass is negative.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import in_box
        >>> lower, upper = jnp.array([0.0, 0.0]), jnp.array([2.0, 5.0])
        >>> state = in_box(lower, upper, 50, key=jax.random.key(0))
        >>> bool(((state.position >= lower) & (state.position < upper)).all())
        True
    """
    check_count(n, "n")
    low, high = _two_corners(lower, upper, ("lower", "upper"))
    if is_concrete(low, high) and not (np.asarray(low) < np.asarray(high)).all():
        raise ValidationError(f"'lower' must be below 'upper', got {low} and {high}")
    draws = jax.random.uniform(key, (n, *low.shape), dtype=low.dtype)
    # rounding can put lower + ξ (upper − lower) on upper, which is outside
    below = jnp.nextafter(*jax.lax.stop_gradient((high, low)))
    # x = lower + ξ (upper − lower)
    position = jnp.minimum(low + draws * (high - low), below)
    return _released(position, mass, release_time)


def release_times(
    start: ScalarLike, stop: ScalarLike, n: int
) -> Float[Array, " particle"]:
    """N release times at the middles of N equal pieces of an interval.

    t_p = start + (p + ½)(stop − start)/N, a continuous release.

    Args:
        start: The start of the release, in seconds from the run's epoch.
        stop: Its end [s].
        n: N, the number of particles.

    Returns:
        The release time of each particle [s].

    Raises:
        ValidationError: If ``n`` is below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import release_times
        >>> times = release_times(0.0, 60.0, 3)
        >>> bool(jnp.allclose(times, jnp.array([10.0, 30.0, 50.0])))
        True
    """
    check_count(n, "n")
    dtype = jnp.result_type(start, stop, float)
    begin, end = jnp.asarray(start, dtype=dtype), jnp.asarray(stop, dtype=dtype)
    # t_p = start + (p + ½)(stop − start)/N
    return begin + _midpoints(n, dtype) * (end - begin)


def with_turbulent_velocity(
    state: ParticleState, turbulence: TurbulenceModel, *, key: PRNGKeyArray
) -> ParticleState:
    """The state with its velocity drawn from the turbulence at its heights.

    u' ~ 𝒩(0, σ(z)²), the stationary law of the Langevin step, so that a run
    starts well mixed in velocity.

    Args:
        state: The state, in three dimensions.
        turbulence: The closure that gives σ(z).
        key: The PRNG key of the draws. Re-using the run's key would correlate
            the first velocities with the steps: split it.

    Returns:
        The state with its velocity replaced.

    Raises:
        ValidationError: If the state is in two dimensions.

    Examples:
        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import at_point, with_turbulent_velocity
        >>> class Isotropic(eqx.Module):
        ...     def at(self, height):
        ...         ones = jnp.ones((*height.shape, 3))
        ...         return 0.5 * ones, 60.0 * ones
        >>> state = at_point(jnp.array([0.0, 0.0, 10.0]), 8)
        >>> state = with_turbulent_velocity(
        ...     state, Isotropic(), key=jax.random.key(0)
        ... )
        >>> state.velocity.shape
        (8, 3)
    """
    shape = state.position.shape
    if shape[-1] != 3:
        raise ValidationError(
            f"'state' must have 3 position columns for a turbulent velocity, "
            f"got shape {shape}"
        )
    sigma, _ = turbulence.at(state.position[:, 2])
    draws = jax.random.normal(key, shape, dtype=state.position.dtype)
    # u' = σ(z) ξ
    velocity = (sigma * draws).astype(state.position.dtype)
    return eqx.tree_at(lambda particles: particles.velocity, state, velocity)
