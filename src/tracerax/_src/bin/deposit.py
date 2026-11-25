"""Deposits of particle weights in cells, as transposes of functax stencils."""

from __future__ import annotations

from typing import Any, Literal

import einx
import jax.numpy as jnp
import numpy as np
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Binning, Stencil, TransposedStencil
from jaxtyping import Array, Float

from tracerax._src.advect.masks import stand_in
from tracerax._src.bin.cells import Cells, cell_centres, edges_in, host_axes
from tracerax.exceptions import ValidationError


__all__ = [
    "check_kind",
    "deposit",
]

_KINDS = ("nearest", "linear")


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
        2. Nearest: i = max{j : e_j ≤ x} on each axis, one tap at the flat index.
        3. Linear: the nodes are the cell centres and one ghost node past each
           end, the reflection of the end centre in its edge. Deposit on them,
           then add each ghost to its neighbour, so that a particle in an outer
           half cell falls whole in the end cell.

    Args:
        cells: The cells, with D axes.
        position: x, with one column per axis of the cells [m]. For the footprint
            of a plume pass its first two columns, ``position[:, :2]``.
        weight: m, one per particle, such as a mass [kg] or a duration [s].
        kind: ``"nearest"``, whose gradient in the positions is zero, or
            ``"linear"``, which is differentiable in them between the cell
            centres and has the gradient zero in the outer half cells.

    Returns:
        c, the sum per cell, in the units of ``weight`` and the positions' dtype.

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
    edges = edges_in(cells, position)
    if weight.shape != position.shape[:1]:
        raise ValidationError(
            f"'weight' must have one entry per particle, {position.shape[:1]}, "
            f"got shape {weight.shape}"
        )
    dtype = position.dtype
    inside = cells.contains(position)
    kept = jnp.where(inside, weight.astype(dtype), jnp.zeros((), dtype))
    # what is outside may be NaN: it is read in the first cell, with weight zero
    corner = jnp.asarray([cell_centres(edge[:2])[0] for edge in edges], dtype=dtype)
    position = stand_in(position, inside, corner)
    if kind == "nearest":
        return _nearest(edges, position, kept)
    return _linear(edges, position, kept)


def _nearest(
    edges: tuple[Any, ...],
    position: Float[Array, "particle axis"],
    weight: Float[Array, " particle"],
) -> Float[Array, "*cells"]:
    """The histogram, for particles inside the edges.

    It becomes `Binning("sum")` with jejjohnson/functax#168, until which the
    latter ignores the edges of an axis.
    """
    shape = tuple(edge.shape[0] - 1 for edge in edges)
    flat = jnp.zeros(position.shape[0], dtype=int)
    for index, (edge, size) in enumerate(zip(edges, shape, strict=True)):
        # i = max{j : e_j ≤ x}
        cell = jnp.searchsorted(edge, position[:, index], side="right") - 1
        # the flat index, row-major
        flat = flat * size + cell
    n_cells = int(np.prod(shape))
    taps = einx.id("particle -> particle 1", flat)
    # H has one tap of weight 1 per particle
    read = Stencil(
        taps, jnp.ones(taps.shape, weight.dtype), n_source=n_cells, check=False
    )
    ones = jnp.ones(n_cells, weight.dtype)
    # c = Hᵀ m
    sums = TransposedStencil(read, ones, ones > 0.0).apply(weight)
    return sums.reshape(shape)


def _linear(
    edges: tuple[Any, ...],
    position: Float[Array, "particle axis"],
    weight: Float[Array, " particle"],
) -> Float[Array, "*cells"]:
    """Cloud-in-cell, for particles inside the edges.

    It becomes one `Binning` call with jejjohnson/functax#169, until which the
    latter drops what falls between an end centre and its edge.
    """
    with host_axes(edges):
        grid = RectilinearGrid([_with_ghosts(edge) for edge in edges])
    # c = Hᵀ m, on the centres and the ghosts
    plan = Binning("sum", weights="linear").plan(Points(position), grid)
    padded = plan(weight).reshape(grid.shape)
    for index in range(len(edges)):
        padded = _fold_ghosts(padded, index)
    return padded


def _with_ghosts(edge: Any) -> Axis:
    """The cell centres, with the end ones reflected in their edges."""
    centres = jnp.asarray(cell_centres(edge))
    # 2 e_0 − c_0
    first = jnp.asarray(2.0 * edge[0] - centres[0])
    # 2 e_n − c_{n−1}
    last = jnp.asarray(2.0 * edge[-1] - centres[-1])
    return Axis(einx.id(", node, -> (1 + node + 1)", first, centres, last))


def _fold_ghosts(padded: Float[Array, "..."], index: int) -> Float[Array, "..."]:
    """Add the first and last slice of one axis to their neighbours, and drop them."""
    before = (slice(None),) * index
    inner = padded[(*before, slice(1, -1))]
    inner = inner.at[(*before, 0)].add(padded[(*before, 0)])
    return inner.at[(*before, -1)].add(padded[(*before, -1)])
