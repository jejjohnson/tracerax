"""Deterministic advection on the sphere, stepped as unit vectors."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from geophysix.earth import SPHERE, Ellipsoid
from integrix.march import erk_step
from jaxtyping import Array, Float
from pipekit_array.arrays import stand_in

from tracerax._src._sphere import (
    angular_rate,
    check_lonlat,
    check_sphere,
    normalise,
    to_lonlat,
    to_unit,
)
from tracerax._src.advect.deterministic import RK4, check_explicit
from tracerax._src.advect.masks import choose
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity


__all__ = [
    "DeterministicSphere",
    "UnitVectorStep",
]


class DeterministicSphere(eqx.Module):
    """A deterministic step on the sphere, as a `ParticleStep`.

    With r the unit vector of a position, δ the particle's own duration, d the
    direction and a, b, c the solver's tableau,

        k_i = (u ê_λ + v ê_φ)(t + d c_i δ, r + d δ ∑_{j<i} a_ij k_j) / R,
        r ← r + d δ ∑_i b_i k_i,  r ← r / |r|.

    Each step returns to longitude and latitude, which in float32 rounds by up to
    0.9 m, and by as much in the same direction at each step. So in float32 the
    displacement per step must be far above a metre: at 6 m per step a month's
    path was 4 % off, and at 1.5 m per step a particle did not move at all. Take
    longer steps, or float64.

    Attributes:
        velocity: (u, v)(t, x), eastward and northward [m s⁻¹], at positions
            (longitude, latitude) in degrees.
        solver: An explicit Runge–Kutta solver, whose tableau is read.
        earth: The sphere whose radius R = ``earth.semi_major`` is used.

    Raises:
        ValidationError: If the solver is not explicit Runge–Kutta, or ``earth``
            is not a sphere.

    Examples:
        Over the pole, at 100 m s⁻¹, from longitude 0° to longitude 180°:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import DeterministicSphere, integrate
        >>> from tracerax.particles import at_point
        >>> def over_the_pole(t, position):
        ...     longitude = jnp.deg2rad(position[:, 0])
        ...     latitude = jnp.deg2rad(position[:, 1])
        ...     east = -jnp.sin(longitude) * jnp.sin(latitude)
        ...     return 100.0 * jnp.column_stack([east, jnp.cos(longitude)])
        >>> result = integrate(
        ...     at_point(jnp.array([0.0, 60.0]), 1),
        ...     DeterministicSphere(over_the_pole),
        ...     t0=0.0,
        ...     duration=50_000.0,
        ...     step_size=500.0,
        ... )
        >>> longitude, latitude = result.state.position[0]
        >>> bool(jnp.abs(longitude) > 179.9), bool(latitude < 80.0)
        (True, True)
    """

    velocity: Velocity
    solver: dfx.AbstractERK = RK4
    earth: Ellipsoid = SPHERE

    def __check_init__(self) -> None:
        check_explicit(self.solver)
        check_sphere(self.earth)

    def noise_shape(self, state: ParticleState) -> None:
        """No noise: the step is deterministic.

        The loop calls this before a run, so concrete positions are checked here.

        Raises:
            ValidationError: If the state is not in longitude and latitude.
        """
        check_lonlat(state.position, "state.position", state.active)

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "..."] | None = None,
    ) -> ParticleState:
        """The state with its positions moved over each particle's duration.

        Args:
            state: The state to move, in longitude and latitude.
            start: The time at which each particle's move begins [s].
            duration: δ ≥ 0, one per particle [s].
            direction: d = +1 forward in time, −1 backward.
            noise: Unused.

        Returns:
            The moved state, with longitudes in [−180°, 180°). An inactive
            particle, and one with duration 0, is unchanged bit for bit. A traced
            latitude beyond a pole gives NaN, because a check that raises would be
            a host callback at every step of a run.

        Raises:
            ValidationError: If the state is not in longitude and latitude.
        """
        del noise
        check_lonlat(state.position, "state.position", state.active)
        moves = state.active & (duration > 0.0)

        def rate(
            time: Float[Array, " particle"], vector: Float[Array, "particle 3"]
        ) -> Float[Array, "particle 3"]:
            return angular_rate(self.velocity, time, vector, self.earth.semi_major)

        initial = to_unit(stand_in(state.position, moves, jnp))
        # h = d δ, one per particle
        vector = erk_step(self.solver, rate, initial, start, direction * duration)
        # r ← r / |r|, which the angles do not depend on
        position = choose(moves, to_lonlat(vector), state.position)
        return eqx.tree_at(lambda particles: particles.position, state, position)


class UnitVectorStep(eqx.Module):
    """The deterministic step on the sphere, on positions kept as unit vectors.

    r ← r + d δ ∑_i b_i k_i,  r ← r / |r|, with the stages of
    `DeterministicSphere`. A flow map that is differentiated keeps its positions
    as unit vectors, which have a derivative at the poles and across the date
    line, where a longitude has none.

    Attributes:
        velocity: (u, v)(t, (λ, φ)), eastward and northward [m s⁻¹].
        solver: An explicit Runge–Kutta solver, whose tableau is read.
        earth: The sphere, whose radius turns a velocity into a rate of turning.
    """

    velocity: Velocity
    solver: dfx.AbstractERK = RK4
    earth: Ellipsoid = SPHERE

    def __check_init__(self) -> None:
        check_explicit(self.solver)
        check_sphere(self.earth)

    def noise_shape(self, state: ParticleState) -> None:
        """No noise: the step is deterministic."""
        del state
        return None

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "..."] | None = None,
    ) -> ParticleState:
        """The state with its unit vectors moved over each particle's duration.

        Args:
            state: The state to move, with unit vectors for positions.
            start: The time at which each particle's move begins [s].
            duration: δ ≥ 0, one per particle [s].
            direction: d = +1 forward in time, −1 backward.
            noise: Unused.

        Returns:
            The moved state. An inactive particle, and one with duration 0, is
            unchanged bit for bit.
        """
        del noise
        moves = state.active & (duration > 0.0)

        def rate(
            time: Float[Array, " particle"], vector: Float[Array, "particle 3"]
        ) -> Float[Array, "particle 3"]:
            return angular_rate(self.velocity, time, vector, self.earth.semi_major)

        initial = stand_in(state.position, moves, jnp)
        # h = d δ, one per particle
        vector = erk_step(self.solver, rate, initial, start, direction * duration)
        # r ← r / |r|
        position = choose(moves, normalise(vector), state.position)
        return eqx.tree_at(lambda particles: particles.position, state, position)
