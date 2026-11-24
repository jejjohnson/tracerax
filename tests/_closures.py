"""Two turbulence closures for the tests.

A closure is any object whose ``at(height)`` gives (σ, τ_L) with a trailing axis
of 3 for (u, v, w), which is what plumax's ``TurbulenceModel`` protocol asks for
and what tracerax's will (tracerax#19). So these run in both packages, and no
plumax closure is copied here. ``tests/data/write_plumax.py`` imports this file
in plumax's environment: it may need JAX and Equinox only.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


__all__ = ["Homogeneous", "LinearSigma"]


class Homogeneous(eqx.Module):
    """Uniform statistics: σ(z) = σ and τ_L(z) = τ, the same on the three axes.

    ∂σ_w²/∂z = 0, so the well-mixed drift is zero.

    Args:
        sigma: σ [m s⁻¹].
        timescale: τ [s].
    """

    sigma: float = 0.5
    timescale: float = 60.0

    def at(self, height: jax.Array) -> tuple[jax.Array, jax.Array]:
        ones = jnp.ones((*jnp.shape(height), 3))
        return self.sigma * ones, self.timescale * ones


class LinearSigma(eqx.Module):
    """σ_w(z) = σ₀ + s z, with σ_u, σ_v and τ_L constant.

    ∂σ_w²/∂z = 2 s σ_w, which is not zero, so a missing drift shows.

    Args:
        sigma_horizontal: σ_u = σ_v [m s⁻¹].
        sigma_ground: σ₀, the value of σ_w at the ground [m s⁻¹].
        slope: s [s⁻¹].
        timescale: τ [s].
    """

    sigma_horizontal: float = 0.4
    sigma_ground: float = 0.2
    slope: float = 0.001
    timescale: float = 60.0

    def at(self, height: jax.Array) -> tuple[jax.Array, jax.Array]:
        height = jnp.asarray(height)
        # σ_w(z) = σ₀ + s z
        sigma_vertical = self.sigma_ground + self.slope * height
        horizontal = jnp.broadcast_to(self.sigma_horizontal, height.shape)
        sigma = jnp.stack([horizontal, horizontal, sigma_vertical], axis=-1)
        return sigma, self.timescale * jnp.ones_like(sigma)
