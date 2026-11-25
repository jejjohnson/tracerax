"""Closed forms of particle motion.

Each form is a small object or function in SI units. The velocities are
callables ``velocity(t, position)``, as tracerax's are, so a test passes one to
the code under test and compares with its ``exact``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "CosineDiffusivity",
    "DoubleGyre",
    "SolidBodyRotation",
    "SphereRotation",
    "taylor_variance",
    "uniform_wind_position",
]


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


@dataclasses.dataclass(frozen=True, slots=True)
class SphereRotation:
    """Rotation of the sphere at ω about an axis in the plane of longitude 0°.

    With n the axis and r the unit vector of a position, the velocity is
    ω R n × r, given eastward and northward, and the path is a circle about n.

    Args:
        rate: ω [rad s⁻¹].
        tilt: The angle from the north pole to n [rad].
        radius: R [m].
    """

    rate: Any
    tilt: float
    radius: float = 6_371_008.771415059

    @property
    def axis(self) -> jax.Array:
        """n."""
        return jnp.array([np.sin(self.tilt), 0.0, np.cos(self.tilt)])

    def __call__(self, t: Any, lonlat: jax.Array) -> jax.Array:
        longitude, latitude = jnp.deg2rad(lonlat[..., 0]), jnp.deg2rad(lonlat[..., 1])
        sin_lon, cos_lon = jnp.sin(longitude), jnp.cos(longitude)
        sin_lat, cos_lat = jnp.sin(latitude), jnp.cos(latitude)
        n_x, n_z = np.sin(self.tilt), np.cos(self.tilt)
        # (n × r) · ê_λ and (n × r) · ê_φ
        east = n_z * cos_lat - n_x * sin_lat * cos_lon
        north = n_x * sin_lon
        scale = self.rate * self.radius
        return (scale * jnp.stack([east, north], axis=-1)).astype(lonlat.dtype)

    def exact(self, lonlat: jax.Array, duration: Any) -> jax.Array:
        """The positions after a duration, by Rodrigues' formula."""
        longitude, latitude = jnp.deg2rad(lonlat[..., 0]), jnp.deg2rad(lonlat[..., 1])
        r = jnp.stack(
            [
                jnp.cos(latitude) * jnp.cos(longitude),
                jnp.cos(latitude) * jnp.sin(longitude),
                jnp.sin(latitude),
            ],
            axis=-1,
        )
        angle = self.rate * duration
        along = jnp.sum(r * self.axis, axis=-1, keepdims=True)
        # r cos θ + (n × r) sin θ + n (n · r)(1 − cos θ)
        turned = (
            r * jnp.cos(angle)
            + jnp.cross(jnp.broadcast_to(self.axis, r.shape), r) * jnp.sin(angle)
            + self.axis * along * (1.0 - jnp.cos(angle))
        )
        longitude = jnp.arctan2(turned[..., 1], turned[..., 0])
        latitude = jnp.arcsin(jnp.clip(turned[..., 2], -1.0, 1.0))
        return jnp.rad2deg(jnp.stack([longitude, latitude], axis=-1))


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
