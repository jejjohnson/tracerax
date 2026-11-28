"""Masking a particle so that neither its value nor its gradient spreads."""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from pipekit_array.arrays import stand_in

from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity


__all__ = [
    "choose",
    "mean_at_start",
]


def choose(
    mask: Bool[Array, " particle"],
    chosen: Float[Array, "particle axis"],
    other: Float[Array, "particle axis"],
) -> Float[Array, "particle axis"]:
    """``chosen`` where the mask holds and ``other`` elsewhere, in other's dtype."""
    return einx.where(
        "particle, particle axis, particle axis -> particle axis",
        mask,
        chosen.astype(other.dtype),
        other,
    )


def mean_at_start(
    velocity: Velocity,
    state: ParticleState,
    start: Float[Array, " particle"],
    duration: Float[Array, " particle"],
    direction: float,
) -> tuple[Float[Array, " particle"], Float[Array, "particle axis"]]:
    """When a stochastic step reads its mean velocity, and d u(t, x) then.

    Args:
        velocity: u(t, x) [m s⁻¹].
        state: The state to move.
        start: The time at which each particle's move begins [s].
        duration: δ ≥ 0, one per particle [s].
        direction: d = +1 forward in time, −1 backward.

    Returns:
        The physical start of each particle's interval, in both directions [s],
        and the velocity along the run there, read at a valid position for a
        particle that does not move.
    """
    # min(t, t + d δ)
    time = jnp.minimum(start, start + direction * duration)
    moves = state.active & (duration > 0.0)
    # d u(t, x)
    return time, direction * velocity(time, stand_in(state.position, moves, jnp))
