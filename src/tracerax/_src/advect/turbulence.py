"""The interface of a turbulence closure; the closures themselves are plumax's."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float


__all__ = ["TurbulenceModel"]


@runtime_checkable
class TurbulenceModel(Protocol):
    """A turbulence closure: σ(z) and τ_L(z) of the velocity fluctuation.

    Any object with this method is one; nothing has to inherit from the class.
    To differentiate a run through a closure, make it a pytree, such as an
    ``equinox.Module``.

    Examples:
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import TurbulenceModel
        >>> class Isotropic(eqx.Module):
        ...     sigma: float
        ...     timescale: float
        ...
        ...     def at(self, height):
        ...         ones = jnp.ones((*height.shape, 3))
        ...         return self.sigma * ones, self.timescale * ones
        >>> isinstance(Isotropic(0.5, 60.0), TurbulenceModel)
        True
    """

    def at(
        self, height: Float[Array, "*batch"]
    ) -> tuple[Float[Array, "*batch 3"], Float[Array, "*batch 3"]]:
        """The standard deviations and the Lagrangian time scales at a height.

        Args:
            height: z [m], of any shape.

        Returns:
            σ [m s⁻¹] and τ_L [s], one column per velocity component (x, y, z).
        """
        ...
