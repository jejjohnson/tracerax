"""Finite-time Lyapunov exponents, from the Jacobian of the flow map."""

from __future__ import annotations

import math
from typing import Any

import diffrax as dfx
import einx
import jax
import jax.numpy as jnp
import numpy as np
from functax.geometry import enu_basis
from geophysix.earth import Ellipsoid
from jaxtyping import Array, Float
from pipekit_ops import is_concrete

from tracerax._src._sphere import (
    check_sphere,
    checked_lonlat,
    to_lonlat,
    to_unit,
)
from tracerax._src._validate import (
    as_floating,
    check_count,
    check_dimension,
)
from tracerax._src.advect.deterministic import RK4, Deterministic, check_explicit
from tracerax._src.advect.loop import check_epoch, integrate
from tracerax._src.advect.masks import stand_in
from tracerax._src.advect.sphere import UnitVectorStep
from tracerax._src.particles.state import ParticleState
from tracerax._src.particles.step import ParticleStep
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = ["ftle"]


def _check_run(duration: float, n_steps: int, t0: ScalarLike, dtype: Any) -> None:
    """Raise unless the run is one: a duration, a step, and a time that fits them.

    Raises:
        ValidationError: If the duration is not finite or is 0, there is no step,
            ``t0`` is not a scalar, or a concrete one is too far from its epoch
            for the dtype.
    """
    check_count(n_steps, "n_steps")
    if not is_concrete(duration) or not math.isfinite(duration) or duration == 0.0:
        raise ValidationError(
            f"'duration' must be a finite number other than 0, got {duration}"
        )
    if jnp.shape(t0) != ():
        raise ValidationError(f"'t0' must be a scalar, got shape {jnp.shape(t0)}")
    check_epoch(t0, "t0", abs(duration) / n_steps, dtype)


def ftle(
    seeds: Float[Array, "seed axis"],
    velocity: Velocity,
    *,
    t0: ScalarLike,
    duration: float,
    n_steps: int,
    solver: dfx.AbstractERK = RK4,
    n_windows: int = 1,
    earth: Ellipsoid | None = None,
) -> Float[Array, " seed"]:
    """The finite-time Lyapunov exponent at each seed.

        Λ(x₀) = (1/|T|) ln √λ_max(∇Φᵀ ∇Φ)

    with Φ the flow map over the duration T from t₀ and ∇Φ its Jacobian at the
    seed x₀: the rate at which two neighbours separate along the direction that
    separates them most. Forward in time its ridges repel, and backward
    they attract.

    Algorithm:
        1. Run the positions of every seed over a window with
           `tracerax.advect.integrate`, and push one tangent per position column
           through the run by forward mode. Each seed's velocity depends on its own
           position only, so they share it.
        2. Multiply the windows' Jacobians. After each, divide the product by its
           largest entry and add the logarithm, so that nothing overflows.
        3. Take the largest eigenvalue of the Cauchy–Green tensor.

    Args:
        seeds: x₀ [m], or (longitude, latitude) in degrees with ``earth``.
        velocity: u(t, x) [m s⁻¹]. Its gradient enters, and that of a linear read
            jumps at the faces of the cells, so the ridges depend on the scheme.
        t0: The start, in seconds from the run's epoch. It may be traced.
        duration: T [s], negative for a backward flow map. Static.
        n_steps: The number of steps over the whole duration.
        solver: An explicit Runge–Kutta solver, of diffrax or integrix, whose
            tableau is read.
        n_windows: In how many windows to split the duration. It must divide
            ``n_steps``. One is enough unless the exponent times the duration is
            large: e^{ΛT} overflows float32 at ΛT = 88.
        earth: The sphere the seeds are on, or ``None`` on the plane. The
            Jacobian is then taken in the local east–north frames at both ends.

    Returns:
        Λ [s⁻¹], NaN at a seed whose path is not finite. Within √ε of a pole, with
        ε the machine epsilon, the velocity is read at √ε from it. A reverse-mode
        gradient of the exponent runs through the steps; a path that a NaN of the
        velocity ends spoils it for every seed, as in `tracerax.advect.integrate`:
        give such a run a field that is defined where its paths go.

    Raises:
        ValidationError: If ``duration`` is 0, ``n_steps`` or ``n_windows`` is
            below 1 or the second does not divide the first, a concrete ``t0`` is
            too far from its epoch for the seeds' dtype, the seeds have not 2 or 3
            columns (2 on the sphere), ``earth`` is not a sphere, or the solver
            is not explicit Runge–Kutta.
        EquinoxRuntimeError: If a traced latitude is beyond a pole.

    References:
        Shadden, S. C., Lekien, F. and Marsden, J. E. (2005). Definition and
        properties of Lagrangian coherent structures from finite-time Lyapunov
        exponents in two-dimensional aperiodic flows. Physica D 212, 271–304.

    Examples:
        A strain field stretches at its own rate:

        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import ftle
        >>> def strain(t, position):
        ...     return 0.1 * position * jnp.array([1.0, -1.0])
        >>> seeds = jnp.array([[0.0, 0.0], [3.0, -2.0]])
        >>> exponent = ftle(seeds, strain, t0=0.0, duration=20.0, n_steps=200)
        >>> bool(jnp.allclose(exponent, 0.1, rtol=1e-5))
        True
    """
    seeds = as_floating(seeds)
    _check_run(duration, n_steps, t0, seeds.dtype)
    check_explicit(solver)
    check_count(n_windows, "n_windows")
    if n_steps % n_windows:
        raise ValidationError(
            f"'n_windows' must divide the number of steps, {n_steps}, got {n_windows}"
        )
    if seeds.ndim != 2:
        raise ValidationError(f"'seeds' must be (seed, axis), got shape {seeds.shape}")
    if earth is None:
        check_dimension(seeds.shape, "seeds")
    else:
        seeds = checked_lonlat(seeds, "seeds")
        check_sphere(earth)
    n_seeds, dim = seeds.shape
    # a seed at NaN would leave 0 × NaN in the reverse pass of every other one
    given = einx.all("seed [axis]", jnp.isfinite(seeds))
    origin = jnp.zeros(dim, seeds.dtype)
    seeds = stand_in(
        seeds, given, jnp.where(given.any(), seeds[jnp.argmax(given)], origin)
    )
    dtype = seeds.dtype
    window = duration / n_windows
    start_time = jnp.asarray(t0, dtype=dtype)
    if earth is None:
        step: ParticleStep = Deterministic(velocity, solver)
    else:
        step = UnitVectorStep(velocity, solver, earth)

    def flow(
        begin: ScalarLike, point: Float[Array, "seed coordinate"]
    ) -> Float[Array, "seed coordinate"]:
        """Φ over one window, of positions or, on the sphere, of unit vectors."""
        released = jnp.broadcast_to(jnp.asarray(begin, dtype), (n_seeds,))
        state = ParticleState(
            point,
            jnp.zeros_like(point),
            jnp.ones(n_seeds, dtype),
            released,
            jnp.ones(n_seeds, bool),
        )
        result = integrate(
            state,
            step,
            t0=begin,
            duration=window,
            # h = |T| / K
            step_size=abs(duration) / n_steps,
        )
        return result.state.position

    def frame_at(
        point: Float[Array, "seed coordinate"],
    ) -> Float[Array, "seed axis coordinate"]:
        """The orthonormal directions along which positions may move."""
        if earth is None:
            identity = jnp.eye(dim, dtype=dtype)
            return jnp.broadcast_to(identity, (n_seeds, dim, dim))
        # ê_λ and ê_φ: a turn of one radian along each
        return enu_basis(to_lonlat(point), unit="degrees")[:, :2, :]

    def advance(carry: Any, begin: ScalarLike) -> tuple[Any, None]:
        point, jacobian, log_scale = carry

        def push(direction: Float[Array, "seed coordinate"]) -> tuple[Any, Any]:
            return jax.jvp(lambda where: flow(begin, where), (point,), (direction,))

        # one tangent per column, which all share one path
        ends, tangents = jax.vmap(push, in_axes=1)(frame_at(point))
        end = ends[0]
        # (∇Φ)_ij = ê_i(end) · ∂Φ/∂x_j
        local = einx.dot(
            "seed i coordinate, j seed coordinate -> seed i j", frame_at(end), tangents
        )
        jacobian = einx.dot("seed i j, seed j k -> seed i k", local, jacobian)
        # the scale cancels from the exponent, so it needs no derivative; the
        # largest entry is safe to square where the entries are not
        largest = einx.max("seed [i j]", jnp.abs(jacobian))
        scale = jax.lax.stop_gradient(largest)
        jacobian = einx.divide("seed i j, seed -> seed i j", jacobian, scale)
        return (end, jacobian, log_scale + jnp.log(scale)), None

    initial = seeds if earth is None else to_unit(seeds)
    identity = jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (n_seeds, dim, dim))
    begins = start_time + jnp.asarray(window * np.arange(n_windows), dtype=dtype)
    (end, jacobian, log_scale), _ = jax.lax.scan(
        advance, (initial, identity, jnp.zeros(n_seeds, dtype)), begins
    )
    # C = ∇Φᵀ ∇Φ, up to the scale
    cauchy_green = einx.dot("seed k i, seed k j -> seed i j", jacobian, jacobian)
    largest = jnp.linalg.eigvalsh(cauchy_green)[:, -1]
    # Λ = (1/|T|) (ln s + ½ ln λ_max)
    exponent = (log_scale + 0.5 * jnp.log(largest)) / abs(duration)
    # a velocity that does not depend on the position has a Jacobian anyway
    finite = einx.all("seed [coordinate]", jnp.isfinite(end))
    return jnp.where(given & finite, exponent, jnp.nan)
