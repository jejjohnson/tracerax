"""Masking a particle so that neither its value nor its gradient spreads."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity


__all__ = [
    "choose",
    "mean_at_start",
    "stand_in",
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


def stand_in(
    position: Float[Array, "particle axis"],
    mask: Bool[Array, " particle"],
    reference: Float[Array, " axis"] | None = None,
) -> Float[Array, "particle axis"]:
    """The positions, with a valid one in place of each masked particle's.

    A field that is undefined at a masked particle's position gives NaN there,
    and masking the result alone leaves 0 × NaN in the reverse pass, which spoils
    every particle's gradient. So the field is read somewhere valid instead.

    Args:
        position: x [m].
        mask: True for the particles whose own position is read.
        reference: The position to read for the others, or ``None`` for that of
            the first particle in the mask. No gradient reaches it.

    Returns:
        The positions to read at.
    """
    if reference is None:
        reference = position[jnp.argmax(mask)]
    return einx.where(
        "particle, particle axis, axis -> particle axis",
        mask,
        position,
        jax.lax.stop_gradient(reference),
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
    return time, direction * velocity(time, stand_in(state.position, moves))
