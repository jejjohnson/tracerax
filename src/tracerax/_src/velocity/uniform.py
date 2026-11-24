"""What a velocity is, and the velocities that do not vary in space."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from tracerax._src._validate import check_dimension
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "Velocity",
    "uniform",
    "uniform_in_space",
]


type Velocity = Callable[
    [ScalarLike | Float[ArrayLike, "*#particle"], Float[Array, "particle axis"]],
    Float[Array, "particle axis"],
]
"""u(t, x): one velocity per particle, at one time or one time per particle.

The argument order is diffrax's and `integrix.march`'s. Any callable of this
signature is a velocity, such as a rotation or a crosswind lift.
"""


class _Uniform(eqx.Module):
    """u(t, x) = u, one value for every particle and every time."""

    value: Float[Array, " axis"]

    def __call__(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        del t
        return jnp.broadcast_to(self.value.astype(position.dtype), position.shape)


class _UniformInSpace(eqx.Module):
    """u(t, x) = f(t), one value for every particle."""

    fn: Callable[[ScalarLike], Float[Array, " axis"]]

    def __call__(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        times = jnp.asarray(t, dtype=position.dtype)
        values = self.fn(times) if times.ndim == 0 else jax.vmap(self.fn)(times)
        values = jnp.asarray(values, dtype=position.dtype)
        return jnp.broadcast_to(values, position.shape)


def uniform(value: Float[ArrayLike, " axis"]) -> Velocity:
    """A velocity that is the same everywhere and at every time.

    u(t, x) = u

    Args:
        value: u, (u, v) or (u, v, w) [m s⁻¹]. It is an array leaf of the result,
            so a gradient reaches it.

    Returns:
        The velocity.

    Raises:
        ValidationError: If the value has not 2 or 3 components.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import uniform
        >>> wind = uniform(jnp.array([3.0, -1.0]))
        >>> bool(jnp.allclose(wind(0.0, jnp.zeros((2, 2))), jnp.array([3.0, -1.0])))
        True
    """
    velocity = jnp.asarray(value)
    velocity = velocity.astype(jnp.result_type(velocity, float))
    check_dimension(velocity.shape, "value")
    if velocity.ndim != 1:
        raise ValidationError(
            f"'value' must be one velocity, got shape {velocity.shape}"
        )
    return _Uniform(velocity)


def uniform_in_space(fn: Callable[[ScalarLike], Float[Array, " axis"]]) -> Velocity:
    """A velocity that varies in time and not in space.

    u(t, x) = f(t)

    Args:
        fn: f, from one time [s] to one velocity [m s⁻¹]. It is vmapped when the
            particles have a time each.

    Returns:
        The velocity.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import uniform_in_space
        >>> gust = uniform_in_space(lambda t: jnp.array([2.0 + 0.1 * t, 0.0]))
        >>> position = jnp.zeros((3, 2))
        >>> gust(10.0, position)[:, 0].tolist()
        [3.0, 3.0, 3.0]

        One time per particle:

        >>> gust(jnp.array([0.0, 10.0, 20.0]), position)[:, 0].tolist()
        [2.0, 3.0, 4.0]
    """
    return _UniformInSpace(fn)
