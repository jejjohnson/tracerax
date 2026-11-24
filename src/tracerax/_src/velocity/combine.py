"""Velocities made of other velocities."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = ["WeightedSum"]


class WeightedSum(eqx.Module):
    """A weighted sum of velocities, each on its own grid.

    u(t, x) = ∑ₘ wₘ uₘ(t, x)

    With a current u_o, a 10 m wind u_w and a windage coefficient β, the drift of
    a floating object is u_o + β u_w. The weights are an array leaf, so a path's
    gradient reaches β.

    Attributes:
        velocities: The terms uₘ, each a velocity.
        weights: wₘ, one per term.

    Raises:
        ValidationError: If there is no term, or not one weight per term.

    Examples:
        A current and 3 % of the wind:

        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import WeightedSum, uniform
        >>> current = uniform(jnp.array([0.5, 0.0]))
        >>> wind = uniform(jnp.array([0.0, 10.0]))
        >>> drift = WeightedSum((current, wind), jnp.array([1.0, 0.03]))
        >>> bool(jnp.allclose(drift(0.0, jnp.zeros((1, 2))), jnp.array([0.5, 0.3])))
        True
    """

    velocities: tuple[Callable, ...] = eqx.field(converter=tuple)
    weights: Float[Array, " term"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        if not self.velocities:
            raise ValidationError("'velocities' must hold at least one term, got none")
        if self.weights.shape != (len(self.velocities),):
            raise ValidationError(
                f"'weights' must have one entry per term, ({len(self.velocities)},), "
                f"got shape {self.weights.shape}"
            )

    def __call__(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """The weighted sum at each particle.

        Args:
            t: One time, or one per particle [s].
            position: x, with columns (x, y) or (x, y, z) [m].

        Returns:
            u(t, x) [m s⁻¹], in the positions' dtype.
        """
        weights = self.weights.astype(position.dtype)
        total = jnp.zeros_like(position)
        for weight, velocity in zip(weights, self.velocities, strict=True):
            # u ← u + wₘ uₘ(t, x)
            term = velocity(t, position).astype(position.dtype)
            total = total + weight * term
        return total
