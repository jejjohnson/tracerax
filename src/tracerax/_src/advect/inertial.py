"""Particles of finite size and buoyancy at the sea surface (Maxey–Riley)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geophysix.earth import Ellipsoid
from jax.typing import ArrayLike
from jaxtyping import Array, Complex, Float
from pipekit_ops import is_concrete

from tracerax._src._sphere import check_lonlat, check_sphere
from tracerax._src._validate import as_floating
from tracerax._src.advect.masks import choose, stand_in
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.derivatives import Derivatives, velocity_derivatives
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "Inertial",
    "SlowManifold",
]

#: The terms of the Taylor series of φ₁ and φ₂: the first one left out is below
#: 1e-18 at |z| = 1.
_N_TERMS = 19


def _phi(z: Complex[Array, " particle"]) -> tuple[Any, Any]:
    """φ₁(z) = (e^z − 1)/z and φ₂(z) = (e^z − 1 − z)/z², to rounding at any z."""
    small = jnp.square(z.real) + jnp.square(z.imag) < 1.0
    inner = jnp.where(small, z, 0.0)
    outer = jnp.where(small, 1.0, z)
    # ∑ₖ zᵏ / (k + m)!, by Horner's rule
    first = jnp.zeros_like(z)
    second = jnp.zeros_like(z)
    for k in reversed(range(_N_TERMS)):
        first = first * inner + 1.0 / math.factorial(k + 1)
        second = second * inner + 1.0 / math.factorial(k + 2)
    # (e^z − 1)/z and (φ₁ − 1)/z, where nothing cancels
    first_outer = (jnp.exp(outer) - 1.0) / outer
    second_outer = (first_outer - 1.0) / outer
    return jnp.where(small, first, first_outer), jnp.where(small, second, second_outer)


def _rotate(vector: Float[Array, "particle 2"]) -> Float[Array, "particle 2"]:
    """J (a, b) = (−b, a)."""
    return jnp.column_stack([-vector[:, 1], vector[:, 0]])


def _scaled(
    factor: Float[Array, " particle"], vector: Float[Array, "particle 2"]
) -> Float[Array, "particle 2"]:
    return einx.multiply("particle, particle axis -> particle axis", factor, vector)


def _parameter(
    name: str, words: str, inside: Callable[[Any], Any]
) -> Callable[[Any], Float[Array, ""]]:
    """The converter of a scalar field: floating, and checked to be in its range.

    A traced value is checked when the program runs.
    """
    message = f"'{name}' must be {words}"

    def convert(value: Any) -> Float[Array, ""]:
        array = as_floating(value)
        if array.shape != ():
            raise ValidationError(f"'{name}' must be a scalar, got shape {array.shape}")
        if not is_concrete(array):
            return eqx.error_if(array, ~inside(array), message)
        # on the host, which a trace around a constant does not reach
        if not bool(inside(np.asarray(array))):
            raise ValidationError(f"{message}, got {float(array)}")
        return array

    return convert


_buoyancy = _parameter("buoyancy", "in (0, 1]", lambda x: (x > 0.0) & (x <= 1.0))
_response_time = _parameter("response_time", "positive", lambda x: x > 0.0)
_finite_response_time = _parameter(
    "response_time",
    "positive and finite for a slow manifold",
    lambda x: (x > 0.0) & (x < jnp.inf),
)
_air_weight = _parameter("air_weight", "in [0, 1]", lambda x: (x >= 0.0) & (x <= 1.0))
_coriolis = _parameter("coriolis", "finite", lambda x: abs(x) < jnp.inf)


def _optional_coriolis(value: Any) -> Any:
    return None if value is None else _coriolis(value)


class SlowManifold(eqx.Module):
    """The velocity of a small buoyant object at the sea surface, a `Velocity`.

    With v the water's velocity and ω its vorticity, v_a the air's,
    u = (1 − α) v + α v_a, f the Coriolis parameter, J the rotation
    J (a, b) = (−b, a) and D/Dt the derivative following the water,

        v_p = u + τ (R Dv/Dt + R (f + ω/3) J v − Du/Dt − (f + Rω/3) J u),

    where Du/Dt follows u itself. It is the solution of `Inertial` after a time
    long against τ, to order τ², and it has no memory, so it runs forward and
    backward: in the plane under `Deterministic` or any other step, and on the
    sphere, with ``earth``, under `DeterministicSphere`. The two velocities are
    read with one time per particle.

    Attributes:
        water: v(t, x) [m s⁻¹]. Its derivatives enter, so the samples of a grid
            need a scheme with a continuous gradient, such as
            `functax.interp.Cubic`: under a linear read the velocity jumps at
            cell faces.
        air: v_a(t, x) [m s⁻¹], at the height whose share α is meant.
        buoyancy: R ∈ (0, 1], 1 for an object that floats under the surface.
        response_time: τ > 0 [s], finite.
        air_weight: α ∈ [0, 1], the share of the air.
        coriolis: f [s⁻¹], or ``None`` for 2Ω sin φ, which needs ``earth``.
        earth: The sphere on which positions are (longitude, latitude) in degrees
            and velocities eastward and northward, or ``None`` on the plane.

    Raises:
        ValidationError: If a parameter is not a scalar or a concrete one is outside
            its range, ``earth`` is not a sphere, or neither ``coriolis`` nor
            ``earth`` is given.
        EquinoxRuntimeError: If a traced parameter is outside its range.

    References:
        Beron-Vera, F. J., Olascoaga, M. J. and Miron, P. (2019). Building a
        Maxey–Riley framework for surface ocean inertial particle dynamics. Phys.
        Fluids 31, 096602.

    Examples:
        In a uniform current and wind, with no rotation, the object moves with
        their weighted mean:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import SlowManifold
        >>> from tracerax.velocity import uniform
        >>> slow = SlowManifold(
        ...     uniform(jnp.array([0.5, 0.0])),
        ...     uniform(jnp.array([0.0, 10.0])),
        ...     buoyancy=0.6,
        ...     response_time=600.0,
        ...     air_weight=0.03,
        ...     coriolis=0.0,
        ... )
        >>> velocity = slow(0.0, jnp.zeros((1, 2)))
        >>> bool(jnp.allclose(velocity, jnp.array([0.485, 0.3])))
        True
    """

    water: Velocity
    air: Velocity
    buoyancy: Float[Array, ""] = eqx.field(converter=_buoyancy)
    response_time: Float[Array, ""] = eqx.field(converter=_finite_response_time)
    air_weight: Float[Array, ""] = eqx.field(converter=_air_weight)
    coriolis: Float[Array, ""] | None = eqx.field(
        default=None, converter=_optional_coriolis
    )
    earth: Ellipsoid | None = None

    def __check_init__(self) -> None:
        if self.earth is not None:
            check_sphere(self.earth)
        elif self.coriolis is None:
            raise ValidationError("'coriolis' must be given without 'earth', got None")

    def __call__(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """The object's velocity at each particle.

        Args:
            t: One time, or one per particle [s].
            position: x, with two columns.

        Returns:
            v_p [m s⁻¹], in the positions' dtype.

        Raises:
            ValidationError: If the positions have not two columns.
        """
        _check_two_columns(position, "position")
        dtype = position.dtype
        times = jnp.broadcast_to(jnp.asarray(t, dtype), position.shape[:1])
        buoyancy = self.buoyancy.astype(dtype)
        share = self.air_weight.astype(dtype)
        response = self.response_time.astype(dtype)
        if self.earth is not None:
            check_lonlat(position, "position")
        if self.coriolis is not None:
            coriolis = jnp.broadcast_to(self.coriolis.astype(dtype), times.shape)
        else:
            assert self.earth is not None
            # f = 2Ω sin φ
            rate = 2.0 * self.earth.rotation_rate
            coriolis = rate * jnp.sin(jnp.deg2rad(position[:, 1]))

        def at(velocity: Velocity) -> Derivatives:
            return velocity_derivatives(velocity, times, position, earth=self.earth)

        water, air = at(self.water), at(self.air)
        # u = (1 − α) v + α v_a
        mixed = jax.tree.map(lambda w, a: (1.0 - share) * w + share * a, water, air)
        spin = water.vorticity / 3.0
        # R Dv/Dt + R (f + ω/3) J v
        forcing = buoyancy * (
            water.following(water.value)
            + _scaled(coriolis + spin, _rotate(water.value))
        )
        # Du/Dt + (f + Rω/3) J u
        inertia = mixed.following(mixed.value) + _scaled(
            coriolis + buoyancy * spin, _rotate(mixed.value)
        )
        # v_p = u + τ (R Dv/Dt + R (f + ω/3) J v − Du/Dt − (f + Rω/3) J u)
        return mixed.value + response * (forcing - inertia)


def _check_two_columns(position: Float[Array, "particle axis"], name: str) -> None:
    """Raise unless there are two position columns.

    Raises:
        ValidationError: If there are not.
    """
    if position.shape[-1] != 2:
        raise ValidationError(
            f"'{name}' must have 2 columns for a particle at the sea surface, "
            f"got shape {position.shape}"
        )


class Inertial(eqx.Module):
    """An inertial step: a buoyant object at the sea surface, with a memory.

    The memory is the object's own velocity v_p, which follows the Maxey–Riley
    equation for the surface ocean. With v the water's velocity and ω its
    vorticity, v_a the air's, u = (1 − α) v + α v_a, f the Coriolis parameter,
    J (a, b) = (−b, a) and D/Dt the derivative following the water,

        dv_p/dt = b − v_p/τ − g J v_p,  dx/dt = v_p,
        b = R Dv/Dt + u/τ + R (f + ω/3) J v,  g = f + Rω/3.

    Algorithm:
        1. Write a vector (a₁, a₂) as a₁ + i a₂, which turns J into i, and let
           λ = −1/τ − i g. With b and λ frozen the step of length δ is exact:

               v_p ← e^{λδ} v_p + δ φ₁(λδ) b,
               x ← x + δ φ₁(λδ) v_p + δ² φ₂(λδ) b,

           with φ₁(z) = (e^z − 1)/z and φ₂(z) = (e^z − 1 − z)/z².
        2. Take half a step with b and λ at the start, evaluate them there, and
           take the whole step with those: second order, and stable for τ ≪ δ.

    It runs forward only, because backward the drag grows an error like e^{t/τ},
    and in the plane only. `SlowManifold` runs backward under `Deterministic`, and
    on the sphere, with its ``earth``, under `DeterministicSphere`.

    Attributes:
        water: v(t, x) [m s⁻¹]. Its derivatives enter, so the samples of a grid
            need a scheme with a continuous gradient, such as
            `functax.interp.Cubic`.
        air: v_a(t, x) [m s⁻¹].
        buoyancy: R ∈ (0, 1].
        response_time: τ > 0 [s]; infinite for no drag.
        air_weight: α ∈ [0, 1].
        coriolis: f [s⁻¹].

    Raises:
        ValidationError: If a parameter is not a scalar, or a concrete one is outside
            its range.
        EquinoxRuntimeError: If a traced one is.

    References:
        Beron-Vera, F. J., Olascoaga, M. J. and Miron, P. (2019). Building a
        Maxey–Riley framework for surface ocean inertial particle dynamics. Phys.
        Fluids 31, 096602.

    Examples:
        An object with no drag turns on a circle of radius |v_p| / f:

        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Inertial, integrate
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> still = uniform(jnp.zeros(2))
        >>> step = Inertial(
        ...     still,
        ...     still,
        ...     buoyancy=0.5,
        ...     response_time=jnp.inf,
        ...     air_weight=0.0,
        ...     coriolis=1e-4,
        ... )
        >>> state = at_point(jnp.zeros(2), 1)
        >>> state = eqx.tree_at(lambda s: s.velocity, state, state.velocity + 0.1)
        >>> result = integrate(
        ...     state, step, t0=0.0, duration=jnp.pi / 1e-4, step_size=300.0
        ... )
        >>> distance = jnp.linalg.norm(result.state.position[0])
        >>> bool(jnp.allclose(distance, 2.0 * 0.1 * 2**0.5 / 1e-4, rtol=1e-4))
        True
    """

    water: Velocity
    air: Velocity
    buoyancy: Float[Array, ""] = eqx.field(converter=_buoyancy)
    response_time: Float[Array, ""] = eqx.field(converter=_response_time)
    air_weight: Float[Array, ""] = eqx.field(converter=_air_weight)
    coriolis: Float[Array, ""] = eqx.field(converter=_coriolis)

    @property
    def slow_manifold(self) -> SlowManifold:
        """The velocity this step's memory tends to, to order τ²."""
        return SlowManifold(
            self.water,
            self.air,
            buoyancy=self.buoyancy,
            response_time=self.response_time,
            air_weight=self.air_weight,
            coriolis=self.coriolis,
        )

    def release(self, state: ParticleState, *, t0: ScalarLike) -> ParticleState:
        """The state with each memory on the slow manifold at its release.

        Without it a particle starts with the state's own velocity, at rest by
        default, and has a transient of length τ.

        Args:
            state: The particles, in two dimensions.
            t0: The start of the run [s]. A particle released before it is put on
                the slow manifold at ``t0``.

        Returns:
            The state with the velocity of its active particles replaced.

        Raises:
            ValidationError: If the state is not in two dimensions, or the
                response time is infinite, for which there is no slow manifold.
        """
        _check_two_columns(state.position, "state.position")
        dtype = state.position.dtype
        # max(t_r, t₀)
        begin = jnp.maximum(state.release_time.astype(dtype), jnp.asarray(t0, dtype))
        velocity = self.slow_manifold(begin, stand_in(state.position, state.active))
        velocity = choose(state.active, velocity, state.velocity)
        return eqx.tree_at(lambda particles: particles.velocity, state, velocity)

    def noise_shape(self, state: ParticleState) -> None:
        """No noise: the step is deterministic.

        Raises:
            ValidationError: If the state is not in two dimensions.
        """
        _check_two_columns(state.position, "state.position")

    def _coefficients(
        self, time: Float[Array, " particle"], position: Float[Array, "particle 2"]
    ) -> tuple[Float[Array, "particle 2"], Complex[Array, " particle"]]:
        """b and λ at each particle."""
        dtype = position.dtype
        buoyancy = self.buoyancy.astype(dtype)
        share = self.air_weight.astype(dtype)
        coriolis = self.coriolis.astype(dtype)
        rate = 1.0 / self.response_time.astype(dtype)
        water = velocity_derivatives(self.water, time, position)
        air = self.air(time, position).astype(dtype)
        spin = water.vorticity / 3.0
        # b = R Dv/Dt + u/τ + R (f + ω/3) J v
        mixed = (1.0 - share) * water.value + share * air
        forcing = (
            buoyancy * water.following(water.value)
            + rate * mixed
            + buoyancy * _scaled(coriolis + spin, _rotate(water.value))
        )
        # λ = −1/τ − i (f + Rω/3)
        turn = coriolis + buoyancy * spin
        return forcing, jax.lax.complex(-rate * jnp.ones_like(turn), -turn)

    def __call__(
        self,
        state: ParticleState,
        start: Float[Array, " particle"],
        duration: Float[Array, " particle"],
        direction: float,
        noise: Float[Array, "..."] | None = None,
    ) -> ParticleState:
        """The state moved over each particle's duration.

        Args:
            state: The state to move, in two dimensions.
            start: The time at which each particle's move begins [s].
            duration: δ ≥ 0, one per particle [s].
            direction: d = +1. A backward step is refused.
            noise: Unused.

        Returns:
            The state with its positions and its memory moved. An inactive
            particle, and one with duration 0, is unchanged bit for bit.

        Raises:
            ValidationError: If the step is backward, or the state is not in two
                dimensions.
        """
        del noise
        if direction < 0.0:
            raise ValidationError(
                "'duration' must be positive for an inertial step, whose drag grows "
                "errors backward; run its 'slow_manifold' under 'Deterministic'"
            )
        _check_two_columns(state.position, "state.position")
        moves = state.active & (duration > 0.0)
        position = stand_in(state.position, moves)
        velocity = stand_in(state.velocity, moves)

        forcing, rate = self._coefficients(start, position)
        half, _ = _exact_step(position, velocity, forcing, rate, 0.5 * duration)
        # b and λ in the middle of the step
        forcing, rate = self._coefficients(start + 0.5 * duration, half)
        moved, changed = _exact_step(position, velocity, forcing, rate, duration)
        return eqx.tree_at(
            lambda particles: (particles.position, particles.velocity),
            state,
            (
                choose(moves, moved, state.position),
                choose(moves, changed, state.velocity),
            ),
        )


def _exact_step(
    position: Float[Array, "particle 2"],
    velocity: Float[Array, "particle 2"],
    forcing: Float[Array, "particle 2"],
    rate: Complex[Array, " particle"],
    duration: Float[Array, " particle"],
) -> tuple[Float[Array, "particle 2"], Float[Array, "particle 2"]]:
    """The step with b and λ frozen: the new positions and velocities."""
    z = rate * duration
    first, second = _phi(z)
    memory = jax.lax.complex(velocity[:, 0], velocity[:, 1])
    driven = jax.lax.complex(forcing[:, 0], forcing[:, 1])
    # v_p ← e^{λδ} v_p + δ φ₁(λδ) b
    changed = jnp.exp(z) * memory + duration * first * driven
    # x ← x + δ φ₁(λδ) v_p + δ² φ₂(λδ) b
    path = duration * first * memory + jnp.square(duration) * second * driven
    moved = position + jnp.column_stack([path.real, path.imag])
    return moved, jnp.column_stack([changed.real, changed.imag])
