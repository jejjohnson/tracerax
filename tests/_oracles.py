"""Closed forms of particle motion, and the stored plumax values.

Each form is a small object or function in SI units. The velocities are
callables ``velocity(t, position)``, as tracerax's are, so a test passes one to
the code under test and compares with its ``exact``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "PLUMAX_COMMIT",
    "PLUMAX_FAMILIES",
    "CosineDiffusivity",
    "DoubleGyre",
    "SolidBodyRotation",
    "plumax_values",
    "taylor_variance",
    "uniform_wind_position",
]


# --- stored plumax values ---------------------------------------------------

#: The plumax commit whose values are stored in ``tests/data``.
PLUMAX_COMMIT = "96449d2"

#: The families that have a file there.
PLUMAX_FAMILIES = (
    "binning",
    "footprint",
    "langevin",
    "residence",
    "trajectory",
    "wind",
)


def plumax_values(family: str) -> dict[str, np.ndarray]:
    """The values stored from plumax, by name (see ``data/write_plumax.py``)."""
    path = Path(__file__).parent / "data" / f"plumax_{family}_{PLUMAX_COMMIT}.npz"
    with np.load(path) as stored:
        return dict(stored)


# --- deterministic motion ---------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SolidBodyRotation:
    """Rotation at ω about the origin: u = −ω y, v = ω x, and w = 0 in 3-D.

    A particle returns to its start after one period 2π/ω.

    Args:
        rate: ω [rad s⁻¹].
    """

    rate: float

    @property
    def period(self) -> float:
        """T = 2π/ω [s]."""
        return 2.0 * np.pi / self.rate

    def __call__(self, t: Any, position: jax.Array) -> jax.Array:
        x, y = position[..., 0], position[..., 1]
        velocity = jnp.zeros_like(position)
        # u = −ω y, v = ω x
        velocity = velocity.at[..., 0].set(-self.rate * y)
        return velocity.at[..., 1].set(self.rate * x)

    def exact(self, position: jax.Array, duration: Any) -> jax.Array:
        """x(t + T) = R(ωT) x(t), for a duration of either sign."""
        angle = self.rate * duration
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        x, y = position[..., 0], position[..., 1]
        rotated = position.at[..., 0].set(cosine * x - sine * y)
        return rotated.at[..., 1].set(sine * x + cosine * y)


def uniform_wind_position(
    start: Any, wind: Any, release_time: Any, end_time: float
) -> np.ndarray:
    """x(T) = x₀ + u max(T − t_r, 0), in a uniform steady wind, forward in time.

    Args:
        start: x₀, shape (particle, dim) [m].
        wind: u, shape (dim,) [m s⁻¹].
        release_time: t_r, shape (particle,) [s].
        end_time: T [s].
    """
    elapsed = np.maximum(end_time - np.asarray(release_time), 0.0)
    return np.asarray(start) + np.multiply.outer(elapsed, np.asarray(wind))


@dataclasses.dataclass(frozen=True, slots=True)
class DoubleGyre:
    """The time-periodic double gyre on [0, 2] × [0, 1] (Shadden et al., 2005).

    With f(x, t) = ε sin(ωt) x² + (1 − 2ε sin(ωt)) x,

        u = −π A sin(πf) cos(πy),
        v =  π A cos(πf) sin(πy) ∂f/∂x.

    The flow is incompressible, so the flow map has det ∇Φ = 1.

    Args:
        amplitude: A.
        strength: ε.
        frequency: ω.
    """

    amplitude: float = 0.1
    strength: float = 0.25
    frequency: float = 2.0 * np.pi / 10.0

    def __call__(self, t: Any, position: jax.Array) -> jax.Array:
        x, y = position[..., 0], position[..., 1]
        oscillation = self.strength * jnp.sin(self.frequency * t)
        # f = ε sin(ωt) x² + (1 − 2ε sin(ωt)) x
        phase = oscillation * x**2 + (1.0 - 2.0 * oscillation) * x
        # ∂f/∂x
        slope = 2.0 * oscillation * x + 1.0 - 2.0 * oscillation
        scale = np.pi * self.amplitude
        u = -scale * jnp.sin(np.pi * phase) * jnp.cos(np.pi * y)
        v = scale * jnp.cos(np.pi * phase) * jnp.sin(np.pi * y) * slope
        return jnp.stack([u, v], axis=-1)


# --- stochastic motion ------------------------------------------------------


def taylor_variance(sigma: float, timescale: float, elapsed: Any) -> np.ndarray:
    """The displacement variance of homogeneous turbulence (Taylor, 1921).

    Var[x(t)] = 2 σ² τ (t − τ (1 − e^{−t/τ}))

    The position step is first order in Δt, so a run compared with it uses
    Δt ≤ τ/12 and the bound of `assert_variance`.

    Args:
        sigma: σ, the velocity standard deviation [m s⁻¹].
        timescale: τ, the Lagrangian time scale [s].
        elapsed: t [s].
    """
    elapsed = np.asarray(elapsed, dtype=np.float64)
    # 1 − e^{−t/τ}
    memory = -np.expm1(-elapsed / timescale)
    return 2.0 * sigma**2 * timescale * (elapsed - timescale * memory)


@dataclasses.dataclass(frozen=True, slots=True)
class CosineDiffusivity:
    """A diffusivity between two walls: K(x) = K₀ (1 + a cos(πx/L)).

    ∂K/∂x = 0 at both walls. With the drift ∂K/∂x a uniform cloud on [0, L]
    stays uniform; without it, particles gather where K is small.

    Args:
        mean: K₀ [m² s⁻¹].
        contrast: a, with |a| < 1.
        length: L [m].
    """

    mean: float
    contrast: float
    length: float

    def __call__(self, position: jax.Array) -> jax.Array:
        """K(x) [m² s⁻¹]."""
        phase = np.pi * position / self.length
        return self.mean * (1.0 + self.contrast * jnp.cos(phase))

    def gradient(self, position: jax.Array) -> jax.Array:
        """∂K/∂x = −K₀ a (π/L) sin(πx/L) [m s⁻¹]."""
        phase = np.pi * position / self.length
        scale = self.mean * self.contrast * np.pi / self.length
        return -scale * jnp.sin(phase)
