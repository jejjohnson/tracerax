"""Deterministic advection on the sphere, stepped as unit vectors."""

from __future__ import annotations

import math
from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from geophysix.earth import SPHERE, Ellipsoid
from jaxtyping import Array, Float

from tracerax._src._sphere import (
    angular_rate,
    check_lonlat,
    check_sphere,
    checked_lonlat,
    normalise,
    to_lonlat,
    to_unit,
)
from tracerax._src.advect.deterministic import (
    CHECKPOINT,
    RK4,
    PerParticleTerm,
    ReleasedTime,
    check_explicit,
    check_releases,
    check_run,
    runge_kutta,
    solve,
)
from tracerax._src.advect.masks import choose, stand_in
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike


__all__ = [
    "DeterministicSphere",
    "ProjectedSolver",
    "deterministic_sphere",
]


class ProjectedSolver(dfx.AbstractWrappedSolver):
    """A solver whose state is divided by its norm after each step.

    The rate depends on the direction of the state only, so what a solver keeps
    from the end of a step is the same before and after the division.
    """

    solver: dfx.AbstractSolver  # ty: ignore[incompatible-attribute-override]

    @property
    def term_structure(self) -> Any:  # ty: ignore[incompatible-property-override]
        return self.solver.term_structure

    @property
    def interpolation_cls(self) -> Any:  # ty: ignore[incompatible-property-override]
        return self.solver.interpolation_cls

    @property
    def scan_kind(self) -> Any:
        # diffrax sets this of a Runge–Kutta solver, wrapped or not
        return cast(Any, self.solver).scan_kind

    @property
    def root_finder(self) -> Any:
        # and reads these of an implicit one
        return cast(Any, self.solver).root_finder

    @property
    def root_find_max_steps(self) -> Any:
        return cast(Any, self.solver).root_find_max_steps

    @property
    def term_compatible_contr_kwargs(self) -> Any:  # ty: ignore[incompatible-property-override]
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: Any) -> int | None:
        return self.solver.order(terms)

    def strong_order(self, terms: Any) -> Any:
        return self.solver.strong_order(terms)

    def init(self, terms: Any, t0: Any, t1: Any, y0: Any, args: Any) -> Any:
        return self.solver.init(terms, t0, t1, y0, args)

    def step(
        self,
        terms: Any,
        t0: Any,
        t1: Any,
        y0: Any,
        args: Any,
        solver_state: Any,
        made_jump: Any,
    ) -> Any:
        y1, error, dense, solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        # r ← r / |r|
        return normalise(y1), error, dense, solver_state, result

    def func(self, terms: Any, t0: Any, y0: Any, args: Any) -> Any:
        return self.solver.func(terms, t0, y0, args)


def deterministic_sphere(
    state: ParticleState,
    velocity: Velocity,
    *,
    t0: ScalarLike,
    duration: float,
    n_steps: int,
    solver: dfx.AbstractSolver = RK4,
    save_at: Float[Array, " save"] | None = None,
    earth: Ellipsoid = SPHERE,
    adjoint: dfx.AbstractAdjoint = CHECKPOINT,
) -> tuple[ParticleState, Float[Array, "save particle 2"] | None]:
    """Carry particles along a velocity on the sphere, forward or backward.

    dr_p/dt = 1_p(t) (u ê_λ + v ê_φ) / R, with r the unit vector of a position
    and (u, v) the velocity read there. A path may cross the date line and the
    poles, where a step in longitude and latitude would fail.

    Algorithm:
        1. Lift the positions to unit vectors.
        2. Integrate over the offsets from 0 to ``duration`` in ``n_steps`` equal
           steps, and divide each vector by its norm after each step.
        3. Return to longitude and latitude.

    Args:
        state: The particles, with columns (longitude, latitude) in degrees. Their
            release times must lie on the step times, as in `deterministic`.
        velocity: (u, v)(t, x), eastward and northward [m s⁻¹], read at longitudes
            in [−180°, 180°). A global grid needs ``Axis(..., period=360.0)``.
        t0: The start of the run, in seconds from its epoch. It may be traced.
        duration: The length of the run [s], negative for a backward run. Static.
        n_steps: The number of steps.
        solver: The diffrax or integrix solver.
        save_at: The offsets from ``t0`` to save the positions at, as in
            `deterministic`.
        earth: The sphere whose radius R = ``earth.semi_major`` is used.
        adjoint: The diffrax adjoint. Forward mode needs
            ``diffrax.ForwardMode()``.

    Returns:
        The final state, and the positions at ``save_at`` or ``None``. Longitudes
        are in [−180°, 180°), so a path across the date line jumps by 360°: use
        great-circle distances. A particle that has not moved keeps its position
        bit for bit. In float32 a step rounds by up to 0.4 m, and by as much in the
        same direction at each step, so the displacement per step must be far
        above that.

    Raises:
        ValidationError: If the state is not in longitude and latitude, ``earth``
            is not a sphere, or as `deterministic` raises.
        EquinoxRuntimeError: If a traced release time is inside a step, or a traced
            latitude is beyond a pole.

    Examples:
        A day along the equator at 1 m s⁻¹, across the date line:

        >>> import jax.numpy as jnp
        >>> from geophysix.earth import great_circle_distance
        >>> from tracerax.advect import deterministic_sphere
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> state = at_point(jnp.array([179.9, 0.0]), 1)
        >>> east = uniform(jnp.array([1.0, 0.0]))
        >>> final, _ = deterministic_sphere(
        ...     state, east, t0=0.0, duration=86_400.0, n_steps=24
        ... )
        >>> longitude, latitude = final.position[0]
        >>> bool(longitude < -179.0)
        True
        >>> distance = great_circle_distance(179.9, 0.0, longitude, latitude)
        >>> bool(jnp.allclose(distance, 86_400.0, rtol=1e-4))
        True
    """
    check_run(duration, n_steps, t0, state.position.dtype)
    position = checked_lonlat(state.position, "state.position", state.active)
    state = eqx.tree_at(lambda particles: particles.position, state, position)
    check_sphere(earth)
    direction = math.copysign(1.0, duration)
    dtype = state.position.dtype
    start_time = jnp.asarray(t0, dtype=dtype)
    release_time = check_releases(
        state.release_time.astype(dtype),
        start_time,
        direction,
        abs(duration) / n_steps,
        n_steps,
    )
    # what an inactive particle has for a position may be NaN
    initial = to_unit(stand_in(state.position, state.active))

    def vector_field(
        offset: ScalarLike, vector: Float[Array, "particle 3"], args: Any
    ) -> Float[Array, "particle 3"]:
        del args
        time = start_time + jnp.asarray(offset, dtype=dtype)
        rate = angular_rate(velocity, time, vector, earth.semi_major)
        # 1_p (u ê_λ + v ê_φ) / R
        return choose(state.active, rate, jnp.zeros_like(vector))

    # r_p = d (t_{r,p} − t₀)
    released = ReleasedTime(direction * (release_time - start_time), direction)
    final, saved = solve(
        PerParticleTerm(vector_field, released),
        initial,
        duration=duration,
        n_steps=n_steps,
        solver=ProjectedSolver(solver),
        save_at=save_at,
        adjoint=adjoint,
    )

    def positions_at(
        vector: Float[Array, "particle 3"], offset: ScalarLike
    ) -> Float[Array, "particle 2"]:
        # a particle that has not moved keeps its own position
        elapsed = released.increment(jnp.zeros((), dtype), offset)
        moved = state.active & (elapsed != 0.0)
        return choose(moved, to_lonlat(vector), state.position)

    position = positions_at(final, jnp.asarray(duration, dtype=dtype))
    moved = eqx.tree_at(lambda particles: particles.position, state, position)
    if saved is None or save_at is None:
        return moved, None
    trajectory = eqx.filter_vmap(positions_at)(saved, jnp.asarray(save_at, dtype=dtype))
    return moved, trajectory


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

        initial = to_unit(stand_in(state.position, moves))
        vector = runge_kutta(self.solver, rate, initial, start, duration, direction)
        # r ← r / |r|, which the angles do not depend on
        position = choose(moves, to_lonlat(vector), state.position)
        return eqx.tree_at(lambda particles: particles.position, state, position)
