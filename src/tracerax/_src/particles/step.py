"""The protocol of a step: what moves a particle state."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float

from tracerax._src.particles.state import ParticleState


__all__ = ["ParticleStep"]


@runtime_checkable
class ParticleStep(Protocol):
    """A step: it moves every particle over its own duration.

    A step returns the positions and the memory only. Releases, masks, domains
    and noise are the loop's, so one loop runs any step. A step with duration 0
    returns the particle unchanged, bit for bit: this is how an unreleased
    particle waits.

    Examples:
        A step that drifts at 1 m s⁻¹ along x, with no noise:

        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import ParticleStep, at_point
        >>> class Drift(eqx.Module):
        ...     def noise_shape(self, state):
        ...         return None
        ...
        ...     def __call__(self, state, start, duration, direction, noise):
        ...         moved = state.position.at[:, 0].add(direction * duration)
        ...         return eqx.tree_at(lambda s: s.position, state, moved)
        >>> isinstance(Drift(), ParticleStep)
        True
        >>> state = at_point(jnp.zeros(2), 3)
        >>> moved = Drift()(state, jnp.zeros(3), jnp.full(3, 5.0), -1.0, None)
        >>> bool(jnp.allclose(moved.position[:, 0], -5.0))
        True
    """

    def noise_shape(self, state: ParticleState) -> tuple[int, ...] | None:
        """The shape of one step's standard normal noise.

        Args:
            state: The state the step will move.

        Returns:
            The shape, or ``None`` for a deterministic step.
        """
        ...

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "..."] | None,
    ) -> ParticleState:
        """The state moved from ``start`` to ``start + direction × duration``.

        Args:
            state: The state to move.
            start: The model time at which each particle's move begins, in the
                run's direction [s].
            duration: δ ≥ 0, the length of each particle's move [s].
            direction: d = +1 forward in time, −1 backward.
            noise: ξ, standard normal, of shape ``noise_shape(state)``, or
                ``None`` for a deterministic step.

        Returns:
            The state with its positions, and its memory, moved.
        """
        ...
