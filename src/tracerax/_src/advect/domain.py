"""The domain of a run: a particle that leaves it becomes inactive."""

from __future__ import annotations

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from tracerax._src._validate import check_dimension, is_concrete
from tracerax.exceptions import ValidationError


__all__ = ["Box"]


class Box(eqx.Module):
    """A box [lower, upper), one interval per position column.

    Attributes:
        lower: The lowest corner [m], inside.
        upper: The highest corner [m], outside.

    Raises:
        ValidationError: If a corner has not 2 or 3 columns, the two differ in
            shape, or ``lower`` is not below ``upper``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Box
        >>> box = Box([0.0, 0.0], [10.0, 5.0])
        >>> box.contains(jnp.array([[0.0, 4.9], [10.0, 1.0], [3.0, -1.0]])).tolist()
        [True, False, False]
    """

    lower: Float[Array, " axis"] = eqx.field(converter=jnp.asarray)
    upper: Float[Array, " axis"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        check_dimension(self.lower.shape, "lower")
        if self.lower.ndim != 1 or self.lower.shape != self.upper.shape:
            raise ValidationError(
                "'lower' and 'upper' must be two corners of the same dimension, "
                f"got shapes {self.lower.shape} and {self.upper.shape}"
            )
        if is_concrete(self.lower, self.upper):
            low, high = np.asarray(self.lower), np.asarray(self.upper)
            if not (low < high).all():
                raise ValidationError(
                    f"'lower' must be below 'upper', got {low} and {high}"
                )

    def contains(
        self, position: Float[Array, "particle axis"]
    ) -> Bool[Array, " particle"]:
        """Whether each particle is inside: 1[lower ≤ x < upper], all axes.

        Args:
            position: x [m].

        Returns:
            True where the particle is inside.
        """
        inside = (position >= self.lower) & (position < self.upper)
        return einx.all("particle [axis]", inside)
