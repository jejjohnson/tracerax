"""Walls that fold a particle back inside, any number of times."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from pipekit_ops import is_concrete

from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = ["Walls"]


class Walls(eqx.Module):
    """Two reflecting walls in height, or a ground alone.

    With h = top − bottom, r = z − bottom and m = ⌊r/h⌋, the fold is

        z ← bottom + r − m h          for m even,
        z ← bottom + h − (r − m h)    for m odd,

    and w' changes sign where m is odd, so that any number of crossings in one
    step ends inside. Without a top, z ← bottom + |r|, and w' changes sign where
    r < 0.

    Perfect reflection keeps a column well mixed where σ_w is uniform near the
    wall (Wilson and Flesch, 1993).

    Attributes:
        bottom: The height of the ground [m].
        top: The height of the lid [m], or ``None`` for no lid.

    Raises:
        ValidationError: If a height is not finite, or ``top`` is not above
            ``bottom``.

    Examples:
        A particle that crosses both walls, and one that crosses neither:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Walls
        >>> walls = Walls(bottom=0.0, top=50.0)
        >>> height, vertical = walls.reflect(
        ...     jnp.array([120.0, 20.0]), jnp.array([3.0, 1.0])
        ... )
        >>> height.tolist(), vertical.tolist()
        ([20.0, 20.0], [3.0, 1.0])

        The ground alone:

        >>> height, vertical = Walls().reflect(jnp.array([-2.0]), jnp.array([-1.0]))
        >>> height.tolist(), vertical.tolist()
        ([2.0], [1.0])
    """

    bottom: ScalarLike = 0.0
    top: ScalarLike | None = None

    def __check_init__(self) -> None:
        for name, height in (("bottom", self.bottom), ("top", self.top)):
            if height is None or not is_concrete(height):
                continue
            if not np.isfinite(np.asarray(height)):
                # the folds of an infinite layer are 0 · ∞
                raise ValidationError(
                    f"'{name}' must be finite, got {height}; "
                    "give no top, None, for no lid"
                )
        if self.top is None or not is_concrete(self.bottom, self.top):
            return
        if not np.asarray(self.top) > np.asarray(self.bottom):
            raise ValidationError(
                f"'top' must be above 'bottom', got {self.top} and {self.bottom}"
            )

    def reflect(
        self, height: Float[Array, " particle"], vertical: Float[Array, " particle"]
    ) -> tuple[Float[Array, " particle"], Float[Array, " particle"]]:
        """Fold the heights inside, and turn the vertical velocities.

        Args:
            height: z [m].
            vertical: w', the vertical velocity [m s⁻¹].

        Returns:
            z and w' after the reflections. A particle inside is unchanged, bit
            for bit.
        """
        bottom = jnp.asarray(self.bottom, dtype=height.dtype)
        # r = z − bottom
        above_ground = height - bottom
        if self.top is None:
            outside = above_ground < 0.0
            # z ← bottom + |r|
            folded = bottom - above_ground
            turned = outside
        else:
            top = jnp.asarray(self.top, dtype=height.dtype)
            depth = top - bottom
            outside = (height < bottom) | (height > top)
            # m = ⌊r/h⌋
            crossings = jnp.floor(above_ground / depth)
            odd = jnp.mod(crossings, 2.0) == 1.0
            # r − m h
            remainder = above_ground - crossings * depth
            inside = jnp.where(odd, depth - remainder, remainder)
            # rounding can leave the fold just outside
            folded = jnp.clip(bottom + inside, bottom, top)
            turned = outside & odd
        return (
            jnp.where(outside, folded, height),
            jnp.where(turned, -vertical, vertical),
        )
