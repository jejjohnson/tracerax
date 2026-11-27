"""Deposits of particle weights in cells, as functax binning plans."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from functax.geometry import Points
from functax.interp import Binning, Policies
from jaxtyping import Array, Float

from tracerax._src.advect.masks import stand_in
from tracerax._src.bin.cells import (
    Cells,
    cell_grid,
    check_columns,
    edges_in,
    first_centre,
    host_axes,
)
from tracerax.exceptions import ValidationError


__all__ = [
    "check_kind",
    "deposit",
]

_KINDS = ("nearest", "linear")

#: Linear weights out to the edges, so the outer half cells keep their mass.
_OVER_CELLS = Policies(extent="cells")


def check_kind(kind: str) -> None:
    """Raise unless the kind of a deposit is known.

    Raises:
        ValidationError: If it is not.
    """
    if kind not in _KINDS:
        raise ValidationError(f"'kind' must be one of {list(_KINDS)}, got {kind!r}")


def deposit(
    cells: Cells,
    position: Float[Array, "particle axis"],
    weight: Float[Array, " particle"],
    *,
    kind: Literal["nearest", "linear"] = "nearest",
) -> Float[Array, "*cells"]:
    """Deposit one weight per particle in the cells, losing none inside the edges.

    c = Hᵀ m, with H the matrix that reads the cells at the particles: one tap per
    particle for ``"nearest"``, which is a histogram, and 2ᴰ for ``"linear"`` on D
    axes, which is cloud-in-cell. Both keep the whole weight of every particle
    inside the edges, ∑_j c_j = ∑_{p inside} m_p, at the walls too.

    Algorithm:
        1. Give a particle outside the edges, or at NaN, the weight zero.
        2. Sum the weights with a functax `Binning` plan on the cells, whose
           axes carry the edges. Nearest bins between the edges,
           e_i ≤ x < e_{i+1}; linear, under ``extent="cells"``, reflects the end
           centres in their edges, so a particle in an outer half cell falls
           whole in the end cell.

    Args:
        cells: The cells, with D axes.
        position: x, with one column per axis of the cells [m]. For the footprint
            of a plume pass its first two columns, ``position[:, :2]``.
        weight: m, one per particle, such as a mass [kg] or a duration [s].
        kind: ``"nearest"``, whose gradient in the positions is zero, or
            ``"linear"``, which is differentiable in them between the cell
            centres and has the gradient zero in the outer half cells.

    Returns:
        c, the sum per cell, in the units of ``weight`` and the positions' dtype. A
        float32 sum loses digits with the number of equal weights in one cell: of
        the order of 1e-3 of it at 10⁵ and 1e-2 at 10⁶.

    Raises:
        ValidationError: If ``kind`` is unknown, ``position`` has not one column
            per axis of the cells or ``weight`` not one entry per particle, or a
            cell is too narrow for the positions' dtype.

    Examples:
        Three particles, the last one outside:

        >>> import jax.numpy as jnp
        >>> from tracerax.bin import Cells, deposit
        >>> cells = Cells(([0.0, 10.0, 20.0, 30.0],))
        >>> position = jnp.array([[2.0], [20.0], [30.0]])
        >>> mass = jnp.array([1.0, 2.0, 4.0])
        >>> deposit(cells, position, mass).tolist()
        [1.0, 0.0, 2.0]

        Linear splits a particle between two centres, and keeps the whole of one
        that is within half a cell of a wall:

        >>> deposit(cells, position, mass, kind="linear").tolist()
        [1.0, 1.0, 1.0]
    """
    check_kind(kind)
    position, weight = jnp.asarray(position), jnp.asarray(weight)
    check_columns(cells, position)
    edges = edges_in(cells, position.dtype)
    if weight.shape != position.shape[:1]:
        raise ValidationError(
            f"'weight' must have one entry per particle, {position.shape[:1]}, "
            f"got shape {weight.shape}"
        )
    dtype = position.dtype
    inside = cells.contains(position)
    kept = jnp.where(inside, weight.astype(dtype), jnp.zeros((), dtype))
    # what is outside may be NaN: it is read in the first cell, with weight zero
    position = stand_in(position, inside, first_centre(edges, dtype))
    # the plan builds axes of its own from the edges, which stay off the trace too
    with host_axes(edges):
        grid = cell_grid(edges)
        # c = Hᵀ m
        plan = Binning("sum", weights=kind).plan(
            Points(position), grid, policies=_OVER_CELLS
        )
    return plan(kept).reshape(grid.shape)
