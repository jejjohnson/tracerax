"""Diffusivities: the interface, and a constant and a gridded one."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from functax.geometry import RectilinearGrid
from functax.interp import Interpolant, Linear, Policies
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from pipekit_ops import is_concrete

from tracerax._src._validate import check_dimension
from tracerax._src.velocity.gridded import fit_components, read_components
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "ConstantDiffusivity",
    "Diffusivity",
    "GriddedDiffusivity",
]

_LINEAR = Linear()
_CLAMP = Policies(outside="clamp")


@runtime_checkable
class Diffusivity(Protocol):
    """A diagonal diffusivity K(t, x), one value per particle and axis.

    Any object with this method is one. Each particle's value may depend on its
    own position only, because the drift is found by differentiating ``at`` in
    all of them at once. A closure, such as Smagorinsky's, is computed on the
    model's grid and passed as a `GriddedDiffusivity`.

    Examples:
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Diffusivity
        >>> class Shallow(eqx.Module):
        ...     def at(self, t, position):
        ...         return 0.1 * jnp.ones_like(position)
        >>> isinstance(Shallow(), Diffusivity)
        True
    """

    def at(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """The diffusivity at each particle.

        Args:
            t: One time, or one per particle [s].
            position: x, with columns (x, y) or (x, y, z) [m].

        Returns:
            K_i(t, x) ≥ 0 [m² s⁻¹], one column per axis.
        """
        ...


class ConstantDiffusivity(eqx.Module):
    """A diffusivity that is the same everywhere: K_i(t, x) = K_i.

    Attributes:
        value: K, one per axis [m² s⁻¹]. An array leaf, so a gradient reaches it.

    Raises:
        ValidationError: If it has not 2 or 3 entries, or one is negative.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import ConstantDiffusivity
        >>> diffusivity = ConstantDiffusivity([10.0, 10.0, 0.01])
        >>> diffusivity.at(0.0, jnp.zeros((4, 3))).shape
        (4, 3)
    """

    value: Float[Array, " axis"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        check_dimension(self.value.shape, "value")
        if self.value.ndim != 1:
            raise ValidationError(
                f"'value' must hold one diffusivity per axis, got {self.value.shape}"
            )
        if is_concrete(self.value) and not (np.asarray(self.value) >= 0.0).all():
            raise ValidationError(f"'value' must be non-negative, got {self.value}")

    def at(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """K at each particle, in the positions' dtype."""
        del t
        return jnp.broadcast_to(self.value.astype(position.dtype), position.shape)


class GriddedDiffusivity(eqx.Module):
    """A diffusivity read from fields, as `GriddedVelocity` reads a velocity.

    Each axis has its own field and grid. A linear read has a drift that jumps at
    the nodes, and it stays well mixed; a cubic one can go below zero, where the
    step clips it.

    Attributes:
        components: One functax interpolant per axis; build them with `fit`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, RectilinearGrid
        >>> from tracerax.advect import GriddedDiffusivity
        >>> grid = RectilinearGrid(
        ...     (Axis(jnp.zeros(1)), Axis(jnp.arange(3.0)), Axis(jnp.arange(4.0)))
        ... )
        >>> field = jnp.broadcast_to(jnp.arange(4.0), (1, 3, 4))
        >>> diffusivity = GriddedDiffusivity.fit((grid, grid), (field, field))
        >>> diffusivity.at(0.0, jnp.array([[1.5, 0.5]])).tolist()
        [[1.5, 1.5]]
    """

    components: tuple[Interpolant, ...]

    @classmethod
    def fit(
        cls,
        grids: Sequence[RectilinearGrid],
        fields: Sequence[Float[ArrayLike, "..."]],
        *,
        scheme: Any = _LINEAR,
        policies: Policies = _CLAMP,
    ) -> GriddedDiffusivity:
        """Fit the samples of each axis's diffusivity on its own grid.

        Args:
            grids: One grid per axis, with the axes (time, z, y, x) or
                (time, y, x).
            fields: The samples of K_i [m² s⁻¹], shaped like their grids.
            scheme: The local functax scheme.
            policies: What to read outside the grid and at missing samples.

        Returns:
            The diffusivity.

        Raises:
            ValidationError: If there are not 2 or 3 fields, one grid per field,
                1 + D axes per grid, or a field shaped like its grid.
        """
        return cls(fit_components(grids, fields, scheme=scheme, policies=policies))

    def at(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """K at each particle, in the positions' dtype.

        Raises:
            ValidationError: If the positions have another dimension.
        """
        return read_components(self.components, t, position)
