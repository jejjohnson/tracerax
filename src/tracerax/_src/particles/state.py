"""The particle state: where the particles are, what they carry, when they start."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Bool, Float

from tracerax._src._validate import check_dimension
from tracerax.exceptions import ValidationError


__all__ = ["ParticleState"]


class ParticleState(eqx.Module):
    """N particles in D = 2 or 3 dimensions.

    A particle that has not been released yet is still active; the loop holds it
    in place until its release time. The age of a particle is |t − t_r| and is
    not stored.

    Attributes:
        position: x, with columns (x, y) or (x, y, z) [m], in a local frame.
        velocity: The step's memory [m s⁻¹]: the turbulent fluctuation u' for a
            Langevin step, the particle's own velocity for an inertial one, and
            zeros for the steps that have none.
        mass: m, the mass each particle carries [kg].
        release_time: t_r, in seconds from the run's epoch, in the positions'
            dtype. Count it from an epoch near the run: float32 resolves 128 s
            at Unix-epoch seconds.
        active: False once a particle has left the domain. A particle is masked
            and never dropped, so that shapes are static.

    Raises:
        ValidationError: If ``position`` is not (N, 2) or (N, 3), or another
            field's shape does not match it.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.particles import ParticleState, at_point
        >>> state = at_point(jnp.array([0.0, 0.0, 10.0]), 4, mass=2.0)
        >>> state.n_particles, state.position.shape
        (4, (4, 3))

        It is a pytree, so it is changed with ``equinox.tree_at``:

        >>> import equinox as eqx
        >>> moved = eqx.tree_at(lambda s: s.position, state, state.position + 1.0)
        >>> bool(jnp.allclose(moved.position[0], jnp.array([1.0, 1.0, 11.0])))
        True
    """

    position: Float[Array, "particle axis"]
    velocity: Float[Array, "particle axis"]
    mass: Float[Array, " particle"]
    release_time: Float[Array, " particle"]
    active: Bool[Array, " particle"]

    def __check_init__(self) -> None:
        shape = self.position.shape
        check_dimension(shape, "position")
        if len(shape) != 2:
            raise ValidationError(f"'position' must be (N, 2) or (N, 3), got {shape}")
        if self.velocity.shape != shape:
            raise ValidationError(
                f"'velocity' must have the shape of 'position', {shape}, "
                f"got {self.velocity.shape}"
            )
        for name in ("mass", "release_time", "active"):
            if getattr(self, name).shape != shape[:1]:
                raise ValidationError(
                    f"'{name}' must have one entry per particle, {shape[:1]}, "
                    f"got {getattr(self, name).shape}"
                )

    @property
    def n_particles(self) -> int:
        """N, the number of particles, active or not."""
        return self.position.shape[0]
