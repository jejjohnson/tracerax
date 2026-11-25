"""Derivatives of a velocity at the particles, on the plane and on the sphere."""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from functax.geometry import enu_basis
from geophysix.earth import Ellipsoid
from jaxtyping import Array, Float

from tracerax._src._sphere import angular_rate, to_unit
from tracerax._src.velocity.uniform import Velocity


__all__ = [
    "Derivatives",
    "velocity_derivatives",
]


class Derivatives(eqx.Module):
    """A velocity and its first derivatives, one of each per particle.

    Attributes:
        value: v [m s⁻¹].
        tendency: ∂v/∂t [m s⁻²].
        gradient: G_ij = ∂v_i/∂x_j [s⁻¹]. On the sphere it is the covariant
            derivative, in the local eastward and northward directions.
    """

    value: Float[Array, "particle axis"]
    tendency: Float[Array, "particle axis"]
    gradient: Float[Array, "particle axis axis"]

    def following(
        self, carrier: Float[Array, "particle axis"]
    ) -> Float[Array, "particle axis"]:
        """∂v/∂t + (w · ∇) v, the rate of change of v along a velocity w."""
        carried = einx.dot(
            "particle i j, particle j -> particle i", self.gradient, carrier
        )
        return self.tendency + carried

    @property
    def vorticity(self) -> Float[Array, " particle"]:
        """ω = ∂v_2/∂x_1 − ∂v_1/∂x_2 [s⁻¹], in two dimensions."""
        return self.gradient[:, 1, 0] - self.gradient[:, 0, 1]


def velocity_derivatives(
    velocity: Velocity,
    t: Float[Array, " particle"],
    position: Float[Array, "particle axis"],
    *,
    earth: Ellipsoid | None = None,
) -> Derivatives:
    """Differentiate a velocity by forward mode, once in time and once per axis.

    Each particle's velocity depends on its own time and position only, so one
    tangent serves all of them.

    Args:
        velocity: v(t, x) [m s⁻¹].
        t: The time, one per particle [s].
        position: x, in metres, or (longitude, latitude) in degrees with
            ``earth``.
        earth: The sphere of the positions, or ``None`` on the plane.

    Returns:
        The velocity and its derivatives. Within √ε of a pole, with ε the machine
        epsilon, they are those at √ε from it.
    """
    if earth is None:
        point: Any = position
        n_particles, dim = position.shape
        directions = [jnp.zeros_like(position).at[:, k].set(1.0) for k in range(dim)]

        def at(times: Any, points: Any) -> Any:
            return velocity(times, points).astype(position.dtype)

    else:
        point = to_unit(position)
        frame = enu_basis(position, unit="degrees")[:, :2, :]
        # r(ξ) = r + (ξ_1 ê_λ + ξ_2 ê_φ) / R, to first order
        radius = jnp.asarray(earth.semi_major, position.dtype)
        directions = [frame[:, k, :] / radius for k in range(2)]

        def at(times: Any, points: Any) -> Any:
            # V(r) = u ê_λ + v ê_φ, as a vector in space
            vectors = radius * angular_rate(velocity, times, points, radius)
            # seen in the frame of the particle itself, which does not move
            return einx.dot(
                "particle axis xyz, particle xyz -> particle axis", frame, vectors
            )

        n_particles = position.shape[0]

    times = jnp.broadcast_to(jnp.asarray(t, position.dtype), (n_particles,))
    # v and ∂v/∂t
    value, tendency = jax.jvp(
        lambda when: at(when, point), (times,), (jnp.ones_like(times),)
    )
    columns = []
    for direction in directions:
        # ∂v/∂x_j
        _, column = jax.jvp(lambda where: at(times, where), (point,), (direction,))
        columns.append(column)
    return Derivatives(value, tendency, jnp.stack(columns, axis=-1))
