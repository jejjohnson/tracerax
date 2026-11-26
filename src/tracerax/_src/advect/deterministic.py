"""Deterministic advection: one explicit Runge–Kutta step per particle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import diffrax as dfx
import einx
import equinox as eqx
import integrix.march
import jax.numpy as jnp
from jaxtyping import Array, Float

from tracerax._src.advect.masks import choose, stand_in
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity
from tracerax.exceptions import ValidationError


__all__ = [
    "Deterministic",
    "RK4",
    "check_explicit",
    "runge_kutta",
]

RK4 = integrix.march.RK4Classic()


def check_explicit(solver: Any) -> None:
    """Raise unless the solver is explicit Runge–Kutta, with a tableau to read.

    Raises:
        ValidationError: If it is not.
    """
    if not isinstance(solver, dfx.AbstractERK):
        raise ValidationError(
            "'solver' must be explicit Runge–Kutta (diffrax.AbstractERK), got "
            f"{type(solver).__name__}"
        )


def runge_kutta(
    solver: dfx.AbstractERK,
    rate: Callable[
        [Float[Array, " particle"], Float[Array, "particle axis"]],
        Float[Array, "particle axis"],
    ],
    initial: Float[Array, "particle axis"],
    start: Float[Array, " particle"],
    duration: Float[Array, " particle"],
    direction: float,
) -> Float[Array, "particle axis"]:
    """One step of an explicit Runge–Kutta solver, with a length per particle.

    k_i = f(t + d c_i δ, y + d δ ∑_{j<i} a_ij k_j),  y ← y + d δ ∑_i b_i k_i

    The stages are unrolled from the tableau, with δ one per particle, so forward
    and reverse mode both work and a release inside a step is exact.

    Args:
        solver: The solver whose tableau is read.
        rate: f(t, y), with one time per particle.
        initial: y.
        start: t, one per particle [s].
        duration: δ ≥ 0, one per particle [s].
        direction: d = +1 forward in time, −1 backward.

    Returns:
        y after the step.
    """
    # every explicit Runge–Kutta solver has one
    tableau = cast(dfx.ButcherTableau, solver.tableau)
    times = [float(tableau.c1), *(float(c) for c in tableau.c)]

    def combine(
        weights: Any, stages: list[Float[Array, "particle axis"]]
    ) -> Float[Array, "particle axis"]:
        # y + d δ ∑_j w_j k_j
        total = jnp.zeros_like(initial)
        for weight, stage in zip(weights, stages, strict=True):
            if float(weight) != 0.0:
                total = total + float(weight) * stage
        offset = einx.multiply(
            "particle axis, particle -> particle axis", total, direction * duration
        )
        return initial + offset

    stages: list[Float[Array, "particle axis"]] = []
    for index, fraction in enumerate(times):
        before = [] if index == 0 else tableau.a_lower[index - 1]
        # k_i = f(t + d c_i δ, y + d δ ∑_{j<i} a_ij k_j)
        stage = rate(start + direction * fraction * duration, combine(before, stages))
        stages.append(stage.astype(initial.dtype))
    # y ← y + d δ ∑_i b_i k_i
    return combine(tableau.b_sol, stages)


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
    solver: dfx.AbstractERK = RK4

    def __check_init__(self) -> None:
        check_explicit(self.solver)

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
        moves = state.active & (duration > 0.0)
        position = stand_in(state.position, moves)
        moved = runge_kutta(
            self.solver, self.velocity, position, start, duration, direction
        )
        position = choose(moves, moved, state.position)
        return eqx.tree_at(lambda particles: particles.position, state, position)
