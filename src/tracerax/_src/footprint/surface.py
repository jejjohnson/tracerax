"""The surface footprint of a receptor, from particles run backward."""

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from integrix.adjoint import Checkpointed, Truncated
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from tracerax._src._validate import check_count, is_concrete
from tracerax._src.advect.langevin import Langevin
from tracerax._src.advect.loop import check_epoch, integrate
from tracerax._src.advect.turbulence import TurbulenceModel
from tracerax._src.advect.walls import Walls
from tracerax._src.bin.cells import Cells, cells_in
from tracerax._src.bin.deposit import check_kind
from tracerax._src.bin.residence import Residence
from tracerax._src.particles.seed import at_point, with_turbulent_velocity
from tracerax._src.particles.state import ParticleState
from tracerax._src.velocity.uniform import Velocity
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "Footprint",
    "surface",
]

_GROUND = Walls()


class Footprint(eqx.Module):
    """The sensitivity of one receptor to the surface fluxes of each cell.

    A receptor value is y = ∑_j F_j q_j, for a surface flux q_j in kg m⁻² s⁻¹.

    Attributes:
        sensitivity: F [s m² kg⁻¹].
        residence: R, the time the particles spent over each cell, in its surface
            layer, summed over them [s].
    """

    sensitivity: Float[Array, "x y"]
    residence: Float[Array, "x y"]


class _Below(eqx.Module):
    """1[z < z_s], with z_s the top of the surface layer."""

    top: Float[Array, ""]

    def __call__(self, state: ParticleState) -> Bool[Array, " particle"]:
        return state.position[:, 2] < self.top


def _check_static(cells: Cells, n_particles: Any, **lengths: Any) -> None:
    """Raise unless the static arguments of a footprint are usable.

    Raises:
        ValidationError: If one is not.
    """
    check_count(n_particles, "n_particles")
    if len(cells.edges) != 2:
        raise ValidationError(
            f"'cells' must have two axes (x, y), got {len(cells.edges)}"
        )
    for name, value in lengths.items():
        static = is_concrete(value) and math.isfinite(value)
        if not static or not np.asarray(value) > 0.0:
            # a footprint looks back over a positive duration
            raise ValidationError(
                f"'{name}' must be a positive, finite number of seconds, got {value}"
            )


def _check_receptor(receptor: Any, walls: Walls, **scalars: Any) -> None:
    """Raise unless the receptor is one point in the air and the rest are scalars.

    Raises:
        ValidationError: If one is not.
    """
    if jnp.shape(receptor) != (3,):
        raise ValidationError(
            f"'receptor' must be one point (x, y, z), got shape {jnp.shape(receptor)}"
        )
    for name, value in scalars.items():
        if jnp.shape(value) != ():
            raise ValidationError(
                f"'{name}' must be a scalar, got shape {jnp.shape(value)}"
            )
    if not is_concrete(receptor, walls.bottom, walls.top):
        return
    point = np.asarray(receptor)
    height, top = point[2], np.inf if walls.top is None else np.asarray(walls.top)
    if not np.isfinite(point).all() or not np.asarray(walls.bottom) <= height <= top:
        raise ValidationError(
            f"'receptor' must be finite and between the walls, got {point}"
        )


def surface(
    receptor: Float[Array, " 3"],
    velocity: Velocity,
    turbulence: TurbulenceModel,
    cells: Cells,
    *,
    n_particles: int,
    receptor_time: ScalarLike,
    duration: float,
    step_size: float,
    surface_depth: ScalarLike,
    air_density: ScalarLike,
    key: PRNGKeyArray,
    walls: Walls = _GROUND,
    kind: Literal["nearest", "linear"] = "nearest",
    adjoint: Checkpointed | Truncated | None = None,
) -> Footprint:
    """The surface footprint of one receptor, from particles run backward.

    F_j = 1/(N ρ h_s) ∑ₚ ∑ₖ δ_{p,k} 1[x_p(t_k) ∈ j, z_p(t_k) − z_0 < h_s], with
    ρ the air density, h_s the depth of the surface layer and z_0 the ground.

    There is no division by the cell area: F multiplies a flux in kg m⁻² s⁻¹.
    The sensitivity to an emission rate in kg s⁻¹ is F divided by the area.

    The backward run undoes the intervals of a forward one, and reads the velocity
    at the physical start of each, so that it reverses the mean wind exactly, also
    one that varies in time. It reverses homogeneous turbulence in law, because the
    Ornstein–Uhlenbeck process is time-reversible. In inhomogeneous turbulence the
    backward drift is not the forward one with the wind reversed (Flesch et al.,
    1995), and this one is approximate.

    Algorithm:
        1. Split ``key``: the first drives the run, the second draws the velocities.
        2. Release N particles at the receptor, with u' ~ 𝒩(0, σ(z_r)²).
        3. Run them backward with `Langevin`, and sum the time each spends over a
           cell with its height in the surface layer.
        4. Divide by N ρ h_s.

    Args:
        receptor: (x, y, z) [m].
        velocity: The mean wind, forward in time. It is not to be reversed.
        turbulence: The closure that gives σ(z) and τ_L(z).
        cells: The surface cells, with two axes (x, y).
        n_particles: N.
        receptor_time: The time of the receptor, in seconds from its epoch. It may
            be traced, so one trace serves every time.
        duration: T > 0, how far back to look [s]. Static.
        step_size: Δt > 0, the length of a step [s]. Static.
        surface_depth: h_s, the depth of the layer above the ground that surface
            fluxes mix into [m].
        air_density: ρ [kg m⁻³].
        key: The PRNG key.
        walls: The ground, and a lid if there is one; a ground at zero alone by
            default. ``walls.bottom`` is z_0.
        kind: ``"nearest"`` or ``"linear"``, as in `deposit`. Only the latter is
            differentiable in the horizontal wind and in the receptor's x and y.
            Whether a particle is below z_0 + h_s is a step that no gradient flows
            through, so the gradient in what moves particles across that height
            (the receptor's z, the vertical wind, the closure, h_s) is not that of
            the footprint.
        adjoint: How to differentiate the run, as in `integrate`.

    Returns:
        The footprint, in the receptor's dtype.

    Raises:
        ValidationError: If ``receptor`` is not one finite point between the walls
            (its values are checked when they are not traced), ``cells`` have not
            two axes or one is too narrow for the receptor's dtype, ``duration``,
            ``step_size``, ``surface_depth`` or ``air_density`` is not positive,
            ``receptor_time`` is not a scalar or too far from its epoch for the
            receptor's dtype, ``n_particles`` is below 1, or ``kind`` is unknown.

    Examples:
        In a layer of 50 m that is all surface layer, the footprint sums to
        T / (ρ h_s):

        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Walls
        >>> from tracerax.bin import Cells
        >>> from tracerax.footprint import surface
        >>> from tracerax.velocity import uniform
        >>> class Isotropic(eqx.Module):
        ...     def at(self, height):
        ...         ones = jnp.ones((*height.shape, 3))
        ...         return 0.5 * ones, 60.0 * ones
        >>> cells = Cells(([-2000.0, -300.0, 0.0, 2000.0], [-2000.0, 2000.0]))
        >>> footprint = surface(
        ...     jnp.array([0.0, 0.0, 10.0]),
        ...     uniform(jnp.array([3.0, 0.0, 0.0])),
        ...     Isotropic(),
        ...     cells,
        ...     n_particles=100,
        ...     receptor_time=0.0,
        ...     duration=200.0,
        ...     step_size=5.0,
        ...     surface_depth=50.0,
        ...     air_density=1.25,
        ...     walls=Walls(0.0, 50.0),
        ...     key=jax.random.key(0),
        ... )
        >>> footprint.sensitivity.shape
        (3, 1)
        >>> total = footprint.sensitivity.sum()
        >>> bool(jnp.allclose(total, 200.0 / (1.25 * 50.0)))
        True

        The wind is from the west, so the receptor saw the cells west of it:

        >>> bool(footprint.sensitivity[1, 0] > 100.0 * footprint.sensitivity[2, 0])
        True

    References:
        - Lin et al. (2003), J. Geophys. Res. 108, 4493: STILT.
        - Flesch, Wilson and Yee (1995), J. Appl. Meteor. 34, 1320–1332.
    """
    _check_static(cells, n_particles, duration=duration, step_size=step_size)
    check_kind(kind)
    _check_receptor(
        receptor,
        walls,
        receptor_time=receptor_time,
        surface_depth=surface_depth,
        air_density=air_density,
    )
    for name, value in (("surface_depth", surface_depth), ("air_density", air_density)):
        if is_concrete(value) and not np.asarray(value) > 0.0:
            raise ValidationError(f"'{name}' must be positive, got {value}")
    # --- release ---
    key_run, key_velocity = jax.random.split(key)
    state = at_point(receptor, n_particles, release_time=receptor_time)
    state = with_turbulent_velocity(state, turbulence, key=key_velocity)
    dtype = state.position.dtype
    check_epoch(receptor_time, "receptor_time", step_size, dtype)

    # --- run back ---
    # z_0 + h_s
    top = jnp.asarray(walls.bottom, dtype) + jnp.asarray(surface_depth, dtype)
    # the total is in the dtype of the edges
    cells = cells_in(cells, dtype)
    result = integrate(
        state,
        Langevin(velocity, turbulence, walls),
        t0=receptor_time,
        # an unsigned integer has no negative
        duration=-float(duration),
        step_size=step_size,
        key=key_run,
        accumulate=Residence(cells, kind, _Below(top)),
        adjoint=adjoint,
    )

    # --- normalise ---
    residence = result.accumulated
    # ρ h_s, the air under one square metre
    layer_mass = jnp.asarray(air_density, dtype) * jnp.asarray(surface_depth, dtype)
    # F = R / (N ρ h_s)
    return Footprint(residence / (n_particles * layer_mass), residence)
