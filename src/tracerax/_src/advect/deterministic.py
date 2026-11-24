"""Deterministic advection: a solve that saves chosen times, and a step."""

from __future__ import annotations

import math
from typing import Any, cast

import diffrax as dfx
import einx
import equinox as eqx
import integrix.march
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from tracerax._src._validate import check_count, is_concrete
from tracerax._src.advect.domain import Box
from tracerax._src.advect.masks import choose, stand_in
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "Deterministic",
    "deterministic",
]

_RK4 = integrix.march.RK4Classic()
_CHECKPOINT = dfx.RecursiveCheckpointAdjoint()


class _ReleasedTime(eqx.Module):
    """C_p(s) = d max(d s − r_p, 0): how long particle p has been released for.

    As the control of a diffrax term it gives each particle its own step length,
    so a release at the end of a step does not reach the stage evaluated there.
    """

    release: Float[Array, " particle"]
    direction: float = eqx.field(static=True)

    def increment(self, t0: ScalarLike, t1: ScalarLike) -> Float[Array, " particle"]:
        """C_p(t₁) − C_p(t₀), signed like t₁ − t₀.

        In a backward run diffrax passes the two times in increasing order, so
        the run's own start of the interval is found here.
        """
        begin = jnp.minimum(self.direction * t0, self.direction * t1)
        end = jnp.maximum(self.direction * t0, self.direction * t1)
        # δ_p = clip(end − max(begin, r_p), 0, end − begin), by branches: a tie
        # would split the gradient in r_p
        length = end - begin
        remaining = length - (self.release - begin)
        partial = jnp.where(remaining > 0.0, remaining, 0.0)
        elapsed = jnp.where(self.release >= begin, partial, length)
        return jnp.sign(t1 - t0) * elapsed


class _PerParticleTerm(dfx.AbstractTerm):
    """dx_p = u_p dC_p: an ODE term whose step length is one per particle."""

    vector_field: Any
    control: _ReleasedTime

    def vf(self, t: Any, y: Any, args: Any) -> Any:
        return self.vector_field(t, y, args)

    def contr(self, t0: Any, t1: Any, **kwargs: Any) -> Any:
        return self.control.increment(t0, t1)

    def prod(self, vf: Any, control: Any) -> Any:
        # u δ, one δ per particle
        return einx.multiply("particle axis, particle -> particle axis", vf, control)


def _check_run(duration: float, n_steps: int, t0: ScalarLike) -> None:
    """Raise unless the duration is finite and not zero, and there is a step."""
    check_count(n_steps, "n_steps")
    if not is_concrete(duration) or not math.isfinite(duration) or duration == 0.0:
        raise ValidationError(
            f"'duration' must be a finite number other than 0, got {duration}"
        )
    if jnp.shape(t0) != ():
        raise ValidationError(f"'t0' must be a scalar, got shape {jnp.shape(t0)}")


_RELEASE_MESSAGE = (
    "'release_time' must lie on the step times of a deterministic solve; use "
    "Deterministic under the loop for a release inside a step"
)


def _check_releases(
    release_time: Float[Array, " particle"],
    start_time: Float[Array, ""],
    direction: float,
    step_size: float,
    n_steps: int,
) -> Float[Array, " particle"]:
    """The release times, checked to lie on the step times of the run.

    A release before the start or after the end is on no step. The tolerance is
    1e-6 of a step, and the rounding of the times in their dtype.

    Raises:
        ValidationError: If a concrete one is inside a step.
        EquinoxRuntimeError: If a traced one is.
    """
    xp = np if is_concrete(release_time, start_time) else jnp
    times, start = xp.asarray(release_time), xp.asarray(start_time)
    steps = xp.clip(direction * (times - start) / step_size, 0.0, float(n_steps))
    largest = xp.maximum(xp.abs(times), xp.abs(start))
    tolerance = 1e-6 + 4.0 * xp.spacing(largest) / step_size
    inside = xp.abs(steps - xp.round(steps)) > tolerance
    if xp is jnp:
        return eqx.error_if(release_time, inside, _RELEASE_MESSAGE)
    if inside.any():
        raise ValidationError(_RELEASE_MESSAGE)
    return release_time


def _check_saves(save_at: Float[Array, " save"], duration: float) -> None:
    """Raise when concrete save offsets are outside the run or out of order.

    Raises:
        ValidationError: If one is not between 0 and the duration, or they do not
            follow the run.
    """
    if not is_concrete(save_at):
        return
    offsets = math.copysign(1.0, duration) * np.asarray(save_at)
    # the duration as the offsets see it
    last = np.asarray(abs(duration), dtype=offsets.dtype)
    inside = (offsets >= 0.0) & (offsets <= last)
    if not inside.all() or (np.diff(offsets) < 0.0).any():
        raise ValidationError(
            f"'save_at' must hold offsets from 0 to {duration}, in that order, "
            f"got {np.asarray(save_at)}"
        )


def _check_tableau(solver: dfx.AbstractSolver) -> None:
    """Raise when a particle could stop inside a domain for ever.

    Such a solver gives its first stage no weight, so a particle whose other
    stages are outside does not move.

    Raises:
        ValidationError: If the solver's first weight is 0.
    """
    tableau = getattr(solver, "tableau", None)
    if isinstance(tableau, dfx.ButcherTableau) and tableau.b_sol[0] == 0.0:
        raise ValidationError(
            f"'solver' {type(solver).__name__} gives its first stage no weight, so a "
            "particle could stop inside 'domain'; use another, such as RK4Classic"
        )


def deterministic(
    state: ParticleState,
    velocity: Velocity,
    *,
    t0: ScalarLike,
    duration: float,
    n_steps: int,
    solver: dfx.AbstractSolver = _RK4,
    save_at: Float[Array, " save"] | None = None,
    domain: Box | None = None,
    adjoint: dfx.AbstractAdjoint = _CHECKPOINT,
) -> tuple[ParticleState, Float[Array, "save particle axis"] | None]:
    """Carry particles along a velocity, forward or backward in time.

    dx_p/dt = 1_p(t) u(t, x_p), with 1_p = 1 when particle p is active, released
    and inside the domain.

    Algorithm:
        1. Integrate the positions over the offsets s from 0 to ``duration``,
           with the right-hand side u(t₀ + s, x), in ``n_steps`` equal steps.
        2. Give each particle the time it has been released for as its own clock.
           The stages are evaluated at the step's own times, so the gradient in a
           release time is exact for a steady velocity only; `Deterministic` under
           the loop gives it for any.
        3. Set ``active`` to false outside the domain.

    Args:
        state: The particles. Their release times must lie on the step times. A
            particle is released once the clock has passed its release time in
            the run's direction, so a backward run takes the time it starts from.
        velocity: u(t, x) [m s⁻¹].
        t0: The start of the run, in seconds from its epoch. It may be traced,
            and never reaches diffrax, so one trace serves every start time.
        duration: The length of the run [s], negative for a backward run. Static.
        n_steps: The number of steps.
        solver: The diffrax or integrix solver.
        save_at: The offsets from ``t0`` to save the positions at, signed like
            ``duration`` and in the run's order; ``None`` saves none. Between
            step times they are the solver's interpolation.
        domain: The box a particle stops outside of, crossed at first order. A
            solver whose first stage has no weight is refused with one.
        adjoint: The diffrax adjoint. Forward mode needs
            ``diffrax.ForwardMode()``.

    Returns:
        The final state, and the positions at ``save_at`` or ``None``.

    Raises:
        ValidationError: If ``duration`` is 0, ``n_steps`` is below 1, a concrete
            release time is inside a step, a concrete save offset is outside the
            run, or the solver's first stage has no weight and there is a domain.
        EquinoxRuntimeError: If a traced release time is inside a step.

    Examples:
        A uniform wind, 10 s forward, saved half way:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import deterministic
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> state = at_point(jnp.zeros(2), 3)
        >>> wind = uniform(jnp.array([2.0, 1.0]))
        >>> final, saved = deterministic(
        ...     state,
        ...     wind,
        ...     t0=0.0,
        ...     duration=10.0,
        ...     n_steps=4,
        ...     save_at=jnp.array([5.0]),
        ... )
        >>> bool(jnp.allclose(final.position, jnp.array([20.0, 10.0])))
        True
        >>> saved.shape, bool(jnp.allclose(saved[0], jnp.array([10.0, 5.0])))
        ((1, 3, 2), True)

        Back to the start. In a backward run a particle is released once the
        clock has come down to its release time:

        >>> import equinox as eqx
        >>> final = eqx.tree_at(lambda s: s.release_time, final, jnp.full(3, 10.0))
        >>> back, _ = deterministic(final, wind, t0=10.0, duration=-10.0, n_steps=4)
        >>> bool(jnp.allclose(back.position, 0.0, atol=1e-5))
        True
    """
    _check_run(duration, n_steps, t0)
    direction = math.copysign(1.0, duration)
    dtype = state.position.dtype
    start_time = jnp.asarray(t0, dtype=dtype)
    release_time = _check_releases(
        state.release_time.astype(dtype),
        start_time,
        direction,
        abs(duration) / n_steps,
        n_steps,
    )
    starts_inside = state.active
    if domain is not None:
        _check_tableau(solver)
        starts_inside = starts_inside & domain.contains(state.position)
    # where the velocity is read for a particle that does not move
    reference = state.position[jnp.argmax(starts_inside)]

    def vector_field(
        offset: ScalarLike, position: Float[Array, "particle axis"], args: Any
    ) -> Float[Array, "particle axis"]:
        del args
        time = start_time + jnp.asarray(offset, dtype=dtype)
        moving = state.active
        if domain is not None:
            moving = moving & domain.contains(position)
        rate = velocity(time, stand_in(position, moving, reference))
        # 1_p u(t₀ + s, x)
        return choose(moving, rate, jnp.zeros_like(position))

    # r_p = d (t_{r,p} − t₀)
    release = direction * (release_time - start_time)
    term = _PerParticleTerm(vector_field, _ReleasedTime(release, direction))
    offsets = jnp.asarray(np.linspace(0.0, duration, n_steps + 1), dtype=dtype)
    if save_at is None:
        saveat = dfx.SaveAt(t1=True)
    else:
        _check_saves(save_at, duration)
        saveat = dfx.SaveAt(ts=jnp.asarray(save_at, dtype=dtype), t1=True)
    solution = dfx.diffeqsolve(
        term,
        solver,
        offsets[0],
        offsets[-1],
        None,
        state.position,
        stepsize_controller=dfx.StepTo(offsets),
        saveat=saveat,
        adjoint=adjoint,
        max_steps=n_steps,
    )
    positions = solution.ys
    final = positions[-1]
    active = state.active if domain is None else state.active & domain.contains(final)
    moved = eqx.tree_at(
        lambda particles: (particles.position, particles.active), state, (final, active)
    )
    return moved, None if save_at is None else positions[:-1]


class Deterministic(eqx.Module):
    """A deterministic step: one explicit Runge–Kutta step per particle.

    With δ the particle's own duration, d the direction and a, b, c the solver's
    tableau,

        k_i = u(t + d c_i δ, x + d δ ∑_{j<i} a_ij k_j),
        x ← x + d δ ∑_i b_i k_i.

    The stages are unrolled from the tableau, with δ one per particle, so forward
    and reverse mode both work and a release inside a step is exact.

    Attributes:
        velocity: u(t, x) [m s⁻¹].
        solver: An explicit Runge–Kutta solver, whose tableau is read.

    Raises:
        ValidationError: If the solver is not explicit Runge–Kutta.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Deterministic
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> step = Deterministic(uniform(jnp.array([2.0, 1.0])))
        >>> state = at_point(jnp.zeros(2), 2)
        >>> moved = step(state, jnp.zeros(2), jnp.array([0.0, 5.0]), 1.0, None)
        >>> bool(jnp.allclose(moved.position, jnp.array([[0.0, 0.0], [10.0, 5.0]])))
        True
    """

    velocity: Velocity
    solver: dfx.AbstractERK = _RK4

    def __check_init__(self) -> None:
        if not isinstance(self.solver, dfx.AbstractERK):
            raise ValidationError(
                "'solver' must be explicit Runge–Kutta (diffrax.AbstractERK), got "
                f"{type(self.solver).__name__}"
            )

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
        """The state with its positions moved over each particle's duration.

        Args:
            state: The state to move.
            start: The time at which each particle's move begins [s].
            duration: δ ≥ 0, one per particle [s].
            direction: d = +1 forward in time, −1 backward.
            noise: Unused.

        Returns:
            The moved state. An inactive particle, and one with duration 0, is
            unchanged bit for bit.
        """
        del noise
        # __check_init__ refuses any other solver
        tableau = cast(dfx.ButcherTableau, self.solver.tableau)
        moves = state.active & (duration > 0.0)
        position = stand_in(state.position, moves)
        dtype = position.dtype
        times = [float(tableau.c1), *(float(c) for c in tableau.c)]

        def combine(
            weights: Any, stages: list[Float[Array, "particle axis"]]
        ) -> Float[Array, "particle axis"]:
            # x + d δ ∑_j w_j k_j
            total = jnp.zeros_like(position)
            for weight, stage in zip(weights, stages, strict=True):
                if float(weight) != 0.0:
                    total = total + float(weight) * stage
            offset = einx.multiply(
                "particle axis, particle -> particle axis", total, direction * duration
            )
            return position + offset

        stages: list[Float[Array, "particle axis"]] = []
        for index, fraction in enumerate(times):
            before = [] if index == 0 else tableau.a_lower[index - 1]
            # k_i = u(t + d c_i δ, x + d δ ∑_{j<i} a_ij k_j)
            rate = self.velocity(
                start + direction * fraction * duration, combine(before, stages)
            )
            stages.append(rate.astype(dtype))
        # x ← x + d δ ∑_i b_i k_i
        moved = combine(tableau.b_sol, stages)
        position = choose(moves, moved, state.position)
        return eqx.tree_at(lambda particles: particles.position, state, position)
