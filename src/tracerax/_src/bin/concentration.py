"""Concentrations: a snapshot of a puff, and the steady field of a source."""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from tracerax._src._validate import check_count
from tracerax._src.bin.cells import Cells, cells_in, position_columns
from tracerax._src.bin.deposit import deposit
from tracerax._src.particles.state import ParticleState
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "column",
    "concentration",
    "steady_concentration",
]


def concentration(
    state: ParticleState,
    cells: Cells,
    *,
    t: ScalarLike,
    kind: Literal["nearest", "linear"] = "nearest",
) -> Float[Array, "*cells"]:
    """The concentration at one time, from the masses the particles carry.

    c_j = ∑ₚ mₚ 1[xₚ ∈ j] / V_j, over the active particles released by then, with
    V_j the volume of cell j. A continuous source is a release spread over time,
    with m_p = Q T / N.

    Args:
        state: The particles at ``t``, after a forward run.
        cells: The cells, with two axes (x, y) or three (x, y, z).
        t: The time of the state, in seconds from its epoch. A particle whose
            release is later adds nothing.
        kind: ``"nearest"`` or ``"linear"``, as in `deposit`.

    Returns:
        c [kg m⁻³], or [kg m⁻²] on two axes, in the positions' dtype.

    Raises:
        ValidationError: If the cells have one axis or more axes than the
            positions have columns, ``t`` is not a scalar, or ``kind`` is unknown.

    Examples:
        2 kg at one point, in cells of 8 m³, half of it still to be released:

        >>> import jax.numpy as jnp
        >>> from tracerax.bin import Cells, concentration
        >>> from tracerax.particles import at_point
        >>> cells = Cells(([0.0, 8.0, 16.0], [0.0, 1.0], [0.0, 1.0, 3.0]))
        >>> state = at_point(
        ...     jnp.array([5.0, 0.5, 0.5]),
        ...     2,
        ...     mass=2.0,
        ...     release_time=jnp.array([0.0, 60.0]),
        ... )
        >>> concentration(state, cells, t=30.0)[:, 0, 0].tolist()
        [0.125, 0.0]
        >>> concentration(state, cells, t=60.0)[:, 0, 0].tolist()
        [0.25, 0.0]
    """
    if jnp.shape(t) != ():
        raise ValidationError(f"'t' must be a scalar, got shape {jnp.shape(t)}")
    dtype = state.position.dtype
    columns = position_columns(cells)
    present = state.active & (state.release_time <= jnp.asarray(t, dtype))
    weight = jnp.where(present, state.mass, jnp.zeros((), state.mass.dtype))
    # ∑ₚ mₚ 1[xₚ ∈ j]
    mass = deposit(cells, state.position[:, columns], weight, kind=kind)
    # the volumes of the cells that the particles were binned in
    volumes = cells_in(cells, dtype).volumes()
    # c_j = M_j / V_j
    return mass / volumes


def steady_concentration(
    residence: Float[Array, "*cells"],
    rate: ScalarLike,
    n_particles: int,
    cells: Cells,
) -> Float[Array, "*cells"]:
    """The steady concentration of a continuous source, from one puff's residence.

    c_j = Q R_j / (N V_j), because a continuous source is a superposition of
    puffs, one per unit time. It needs N particles released at once, a steady
    velocity, and a run long enough for the particles to leave the cells.

    Args:
        residence: R, the time spent in each cell [s], from `Residence`.
        rate: Q, the source's rate [kg s⁻¹].
        n_particles: N, the number of particles of the run.
        cells: The cells of the residence.

    Returns:
        c [kg m⁻³], or [kg m⁻²] on two axes, in the dtype of ``residence``.

    Raises:
        ValidationError: If ``residence`` is not shaped like the cells, ``rate`` is
            not a scalar or ``n_particles`` not an integer of at least 1.

    Examples:
        A source of 0.5 kg s⁻¹ seen by 10 particles:

        >>> import jax.numpy as jnp
        >>> from tracerax.bin import Cells, steady_concentration
        >>> cells = Cells(([0.0, 2.0, 6.0], [0.0, 2.0]))
        >>> residence = jnp.array([[40.0], [40.0]])
        >>> steady_concentration(residence, 0.5, 10, cells)[:, 0].tolist()
        [0.5, 0.25]
    """
    _check_shape(residence, "residence", cells)
    check_count(n_particles, "n_particles")
    if jnp.shape(rate) != ():
        raise ValidationError(f"'rate' must be a scalar, got shape {jnp.shape(rate)}")
    dtype = residence.dtype
    # N V_j
    sampled = n_particles * cells_in(cells, dtype).volumes()
    # c_j = Q R_j / (N V_j)
    return jnp.asarray(rate, dtype) * residence / sampled


def column(values: Float[Array, "x y z"], cells: Cells) -> Float[Array, "x y"]:
    """The vertical column of a concentration.

    C_xy = ∑_z c Δz, with the vertical axis last.

    Args:
        values: c, one value per cell [kg m⁻³].
        cells: The cells, with three axes (x, y, z).

    Returns:
        C [kg m⁻²], in the dtype of ``values``.

    Raises:
        ValidationError: If the cells have not three axes, or ``values`` is not
            shaped like them.

    Examples:
        Two layers, of 1 m and 3 m:

        >>> import jax.numpy as jnp
        >>> from tracerax.bin import Cells, column
        >>> cells = Cells(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 4.0]))
        >>> column(jnp.array([[[2.0, 1.0]]]), cells).tolist()
        [[5.0]]
    """
    if len(cells.edges) != 3:
        raise ValidationError(
            f"'cells' must have three axes (x, y, z), got {len(cells.edges)}"
        )
    _check_shape(values, "values", cells)
    # Δz
    thickness = jnp.diff(cells.edges[2]).astype(values.dtype)
    # C_xy = ∑_z c Δz
    return einx.dot("x y z, z -> x y", values, thickness)


def _check_shape(values: Float[Array, "..."], name: str, cells: Cells) -> None:
    """Raise unless there is one value per cell.

    Raises:
        ValidationError: If there is not.
    """
    if values.shape != cells.shape:
        raise ValidationError(
            f"'{name}' must have one value per cell, {cells.shape}, "
            f"got shape {values.shape}"
        )
