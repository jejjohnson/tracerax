"""The well-mixed Langevin step: turbulence as a Markov-1 velocity."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from tracerax._src.advect.turbulence import TurbulenceModel
from tracerax._src.advect.walls import Walls
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity
from tracerax.exceptions import ValidationError


__all__ = [
    "Langevin",
    "langevin_step",
]

_GROUND = Walls()

#: ε, the floor of σ [m s⁻¹], so that a zero σ gives a zero drift and not NaN.
_SIGMA_FLOOR = 1e-12


def _check_three_columns(state: ParticleState) -> None:
    """Raise unless the state is in three dimensions."""
    if state.position.shape[-1] != 3:
        raise ValidationError(
            "'state' must have 3 position columns for a Langevin step, got shape "
            f"{state.position.shape}"
        )


def _variance_slope(
    turbulence: TurbulenceModel, height: Float[Array, " particle"]
) -> Float[Array, " particle"]:
    """∂σ_w²/∂z at each height, by autodiff of the closure."""

    def vertical_variance(level: Float[Array, ""]) -> Float[Array, ""]:
        sigma, _ = turbulence.at(level)
        # σ_w² = max(σ_w, ε)²
        return jnp.maximum(sigma[2], _SIGMA_FLOOR) ** 2

    return jax.vmap(jax.grad(vertical_variance))(height)


def langevin_step(
    state: ParticleState,
    mean_velocity: Float[Array, "particle 3"],
    turbulence: TurbulenceModel,
    duration: Float[Array, " #particle"],
    noise: Float[Array, "particle 3"],
    *,
    walls: Walls = _GROUND,
) -> ParticleState:
    """One Markov-1 Langevin step over each particle's own duration.

    Algorithm:
        1. Integrate the linear part exactly, with ξ standard normal:
           u' ← e^{−δ/τ_L} u' + σ √(1 − e^{−2δ/τ_L}) ξ.
        2. Add the well-mixed drift of Thomson (1987) to the vertical component,
           by an Euler step: w' ← w' + ½ (1 + w'²/σ_w²) ∂σ_w²/∂z δ.
        3. Move: x ← x + (u + u') δ.
        4. Reflect at the walls.

    The drift is first order, so keep δ small against τ_L near the ground. The
    gradient is pathwise at fixed noise, and one-sided at a reflection.

    Args:
        state: The particles, in three dimensions. Their ``velocity`` is u'.
        mean_velocity: u at each particle [m s⁻¹]. A backward step passes −u.
        turbulence: The closure that gives σ(z) and τ_L(z).
        duration: δ ≥ 0, one for all or one per particle [s].
        noise: ξ, standard normal; nothing checks that it is.
        walls: Where particles are reflected, the ground by default.

    Returns:
        The state with its position and velocity moved. An inactive particle, and
        one with duration 0, is unchanged bit for bit.

    Raises:
        ValidationError: If the state is in two dimensions.

    References:
        Thomson, D. J. (1987). Criteria for the selection of stochastic models of
        particle trajectories in turbulent flows. J. Fluid Mech. 180, 529–556.

    Examples:
        Without turbulence the particle follows the mean velocity:

        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import langevin_step
        >>> from tracerax.particles import at_point
        >>> class Calm(eqx.Module):
        ...     def at(self, height):
        ...         return jnp.zeros((*height.shape, 3)), jnp.ones(
        ...             (*height.shape, 3)
        ...         )
        >>> state = at_point(jnp.array([0.0, 0.0, 10.0]), 2)
        >>> wind = jnp.array([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        >>> moved = langevin_step(
        ...     state, wind, Calm(), jnp.array([0.0, 2.0]), jnp.ones((2, 3))
        ... )
        >>> bool(jnp.allclose(moved.position[:, 0], jnp.array([0.0, 6.0])))
        True
    """
    _check_three_columns(state)
    position = state.position
    dtype = position.dtype
    durations = jnp.broadcast_to(
        jnp.asarray(duration, dtype=dtype), (state.n_particles,)
    )
    moves = state.active & (durations > 0.0)
    # a particle that waits keeps finite gradients: √ has none at δ = 0
    elapsed = jnp.where(moves, durations, 1.0)
    waiting = jnp.zeros((), dtype=dtype)
    mean = einx.where(
        "particle, particle axis, -> particle axis", moves, mean_velocity, waiting
    )

    # --- the linear part, exactly ---
    sigma, timescale = turbulence.at(position[:, 2])
    # σ ← max(σ, ε)
    sigma = jnp.maximum(sigma, _SIGMA_FLOOR)
    # δ / τ_L
    ratio = einx.divide("particle, particle axis -> particle axis", elapsed, timescale)
    # σ √(1 − e^{−2δ/τ_L})
    spread = sigma * jnp.sqrt(-jnp.expm1(-2.0 * ratio))
    # u' ← e^{−δ/τ_L} u' + σ √(1 − e^{−2δ/τ_L}) ξ
    velocity = jnp.exp(-ratio) * state.velocity + spread * noise

    # --- the well-mixed drift ---
    vertical = velocity[:, 2]
    slope = _variance_slope(turbulence, position[:, 2])
    # ½ (1 + w'²/σ_w²) ∂σ_w²/∂z δ
    drift = 0.5 * (1.0 + (vertical / sigma[:, 2]) ** 2) * slope * elapsed
    velocity = velocity.at[:, 2].add(drift)

    # --- move and reflect ---
    # x ← x + (u + u') δ
    moved = position + einx.multiply(
        "particle axis, particle -> particle axis", mean + velocity, elapsed
    )
    height, vertical = walls.reflect(moved[:, 2], velocity[:, 2])
    moved = moved.at[:, 2].set(height)
    velocity = velocity.at[:, 2].set(vertical)

    chosen = "particle, particle axis, particle axis -> particle axis"
    return eqx.tree_at(
        lambda particles: (particles.position, particles.velocity),
        state,
        (
            einx.where(chosen, moves, moved.astype(dtype), position),
            einx.where(chosen, moves, velocity.astype(dtype), state.velocity),
        ),
    )


class Langevin(eqx.Module):
    """The Langevin step as a `ParticleStep`, with noise of shape (N, 3).

    The mean velocity is read at each particle's position and at the physical
    start of its interval, min(start, start + d δ), so a backward step is the
    discrete reverse of a forward one; backward, the step takes −u.

    Attributes:
        velocity: u(t, x) [m s⁻¹].
        turbulence: The closure that gives σ(z) and τ_L(z). Make it a pytree to
            differentiate through it.
        walls: Where particles are reflected, the ground by default.

    Examples:
        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Langevin, Walls
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> class Isotropic(eqx.Module):
        ...     def at(self, height):
        ...         ones = jnp.ones((*height.shape, 3))
        ...         return 0.5 * ones, 60.0 * ones
        >>> wind = uniform(jnp.array([3.0, 0.0, 0.0]))
        >>> step = Langevin(wind, Isotropic(), Walls(bottom=0.0, top=50.0))
        >>> state = at_point(jnp.array([0.0, 0.0, 10.0]), 4)
        >>> noise = jax.random.normal(jax.random.key(0), step.noise_shape(state))
        >>> moved = step(state, jnp.zeros(4), jnp.full(4, 5.0), 1.0, noise)
        >>> moved.position.shape, noise.shape
        ((4, 3), (4, 3))
    """

    velocity: Velocity
    turbulence: TurbulenceModel
    walls: Walls = _GROUND

    def noise_shape(self, state: ParticleState) -> tuple[int, int]:
        """(N, 3), one standard normal per particle and component.

        Raises:
            ValidationError: If the state is in two dimensions.
        """
        _check_three_columns(state)
        return state.n_particles, 3

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "particle 3"] | None,
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
            ValidationError: If there is no noise, or the state is in two
                dimensions.
        """
        if noise is None:
            raise ValidationError("'noise' is required by a Langevin step, got None")
        # the physical start of the interval, in both directions
        time = jnp.minimum(start, start + direction * duration)
        # d u(t, x)
        mean = direction * self.velocity(time, state.position)
        return langevin_step(
            state, mean, self.turbulence, duration, noise, walls=self.walls
        )
