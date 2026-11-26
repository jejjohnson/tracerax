"""Random displacement: diffusion that keeps a uniform cloud uniform."""

from __future__ import annotations

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from tracerax._src.advect.diffusivity import Diffusivity
from tracerax._src.advect.masks import choose, mean_velocity, stand_in
from tracerax._src.advect.walls import Walls
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.derivatives import tangents, unit_columns
from tracerax._src.velocity.uniform import Velocity
from tracerax.exceptions import ValidationError


__all__ = [
    "RandomDisplacement",
    "displacement_step",
]


def _diffusivity_and_slope(
    diffusivity: Diffusivity,
    t: Float[Array, " particle"],
    position: Float[Array, "particle axis"],
) -> tuple[Float[Array, "particle axis"], Float[Array, "particle axis"]]:
    """K_i and ∂K_i/∂x_i at each particle, by forward mode along each axis."""

    def at(points: Float[Array, "particle axis"]) -> Float[Array, "particle axis"]:
        return diffusivity.at(t, points)

    # each particle's K depends on its own position only
    gradient = tangents(at, position, unit_columns(position))
    # ∂K_i/∂x_i
    slope = jnp.diagonal(gradient, axis1=1, axis2=2)
    return at(position), slope


def displacement_step(
    state: ParticleState,
    mean_velocity: Float[Array, "particle axis"],
    diffusivity: Diffusivity,
    t: Float[Array, " particle"],
    duration: Float[Array, " particle"],
    noise: Float[Array, "particle axis"],
    *,
    walls: Walls | None = None,
) -> ParticleState:
    """One random-displacement step over each particle's own duration.

        x_i ← x_i + (u_i + ∂K_i/∂x_i) δ + √(2 K_i δ) ξ_i

    The Itô step of particles that follow ∂c/∂t = −∇·(u c) + ∑_i ∂/∂x_i
    (K_i ∂c/∂x_i). Without the drift ∂K_i/∂x_i, particles gather where K is
    small, at every step size.

    The scheme takes K to be nearly linear over one step, so keep
    δ ≤ 0.1 / max |∂²K/∂x²| (Visser, 1997).

    Args:
        state: The particles, in two or three dimensions. Their ``velocity`` is
            not used.
        mean_velocity: u at each particle [m s⁻¹]. A backward step passes −u; the
            drift keeps its sign, because diffusion is its own adjoint.
        diffusivity: K. A value below zero, from a cubic read, is clipped.
        t: The time to read K at, one per particle [s].
        duration: δ ≥ 0, one per particle [s].
        noise: ξ, standard normal; nothing checks that it is.
        walls: Where the third column is reflected, or ``None``.

    Returns:
        The state with its positions moved. An inactive particle, and one with
        duration 0, is unchanged bit for bit.

    Raises:
        ValidationError: If there are walls in two dimensions.

    References:
        Visser, A. W. (1997). Using random walk models to simulate the vertical
        distribution of particles in a turbulent water column. Mar. Ecol. Prog.
        Ser. 158, 275–281.

    Examples:
        One metre of standard deviation after half a second at K = 1 m² s⁻¹:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import ConstantDiffusivity, displacement_step
        >>> from tracerax.particles import at_point
        >>> state = at_point(jnp.zeros(2), 2)
        >>> moved = displacement_step(
        ...     state,
        ...     jnp.zeros((2, 2)),
        ...     ConstantDiffusivity([1.0, 0.0]),
        ...     jnp.zeros(2),
        ...     jnp.array([0.5, 0.0]),
        ...     jnp.ones((2, 2)),
        ... )
        >>> bool(jnp.allclose(moved.position, jnp.array([[1.0, 0.0], [0.0, 0.0]])))
        True
    """
    dtype, dim = state.position.dtype, state.position.shape[-1]
    if walls is not None and dim != 3:
        raise ValidationError(
            f"'walls' reflect the third position column, and 'state' has {dim}"
        )
    moves = state.active & (duration > 0.0)
    position = stand_in(state.position, moves)
    # a particle that waits keeps finite gradients
    elapsed = jnp.where(moves, duration, 1.0)
    waiting = jnp.zeros((), dtype=dtype)
    masked = "particle, particle axis, -> particle axis"
    mean = einx.where(masked, moves, mean_velocity, waiting)

    # --- the drift ---
    value, slope = _diffusivity_and_slope(diffusivity, t, position)
    # K ← max(K, 0), where its slope is 0
    drift = jnp.where(value < 0.0, waiting, slope)

    # --- the noise ---
    scaled = "particle axis, particle -> particle axis"
    # 2 K δ
    variance = 2.0 * einx.multiply(scaled, value, elapsed)
    # √(2 K δ), whose slope is infinite at K = 0
    diffuses = variance > 0.0
    root = jnp.sqrt(jnp.where(diffuses, variance, 1.0))
    spread = jnp.where(diffuses, root, waiting)

    # --- move and reflect ---
    # x_i ← x_i + (u_i + ∂K_i/∂x_i) δ + √(2 K_i δ) ξ_i
    moved = position + einx.multiply(scaled, mean + drift, elapsed) + spread * noise
    if walls is not None:
        height, _ = walls.reflect(moved[:, 2], jnp.zeros_like(moved[:, 2]))
        moved = moved.at[:, 2].set(height)
    chosen = choose(moves, moved, state.position)
    return eqx.tree_at(lambda particles: particles.position, state, chosen)


class RandomDisplacement(eqx.Module):
    """The random-displacement step as a `ParticleStep`, with noise (N, D).

    The velocity and the diffusivity are read at each particle's position and at
    the physical start of its interval, min(start, start + d δ).

    Attributes:
        velocity: u(t, x) [m s⁻¹].
        diffusivity: K(t, x) [m² s⁻¹].
        walls: Where the third column is reflected, or ``None``.

    Examples:
        A surface drifter that diffuses at 10 m² s⁻¹:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import (
        ...     ConstantDiffusivity,
        ...     RandomDisplacement,
        ...     integrate,
        ... )
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> step = RandomDisplacement(
        ...     uniform(jnp.array([0.3, 0.0])), ConstantDiffusivity([10.0, 10.0])
        ... )
        >>> result = integrate(
        ...     at_point(jnp.zeros(2), 100),
        ...     step,
        ...     t0=0.0,
        ...     duration=86_400.0,
        ...     step_size=900.0,
        ...     key=jax.random.key(0),
        ... )
        >>> result.state.position.shape
        (100, 2)
    """

    velocity: Velocity
    diffusivity: Diffusivity
    walls: Walls | None = None

    def noise_shape(self, state: ParticleState) -> tuple[int, int]:
        """(N, D), one standard normal per particle and axis."""
        n_particles, dim = state.position.shape
        return n_particles, dim

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "particle axis"] | None,
    ) -> ParticleState:
        """The state moved over each particle's duration.

        Args:
            state: The state to move.
            start: The time at which each particle's move begins [s].
            duration: δ ≥ 0, one per particle [s].
            direction: d = +1 forward in time, −1 backward.
            noise: ξ, standard normal, of shape ``noise_shape(state)``.

        Returns:
            The moved state.

        Raises:
            ValidationError: If there is no noise, or there are walls in two
                dimensions.
        """
        if noise is None:
            raise ValidationError(
                "'noise' is required by a random-displacement step, got None"
            )
        time, mean = mean_velocity(self.velocity, state, start, duration, direction)
        return displacement_step(
            state, mean, self.diffusivity, time, duration, noise, walls=self.walls
        )
