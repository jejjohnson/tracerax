"""Cells from edges: their centres, their volumes and which particles are inside."""

from __future__ import annotations

import contextlib
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from functax.geometry import Axis, RectilinearGrid
from functax.quad import cell_area
from jaxtyping import Array, Bool, Float
from pipekit_ops import is_concrete

from tracerax.exceptions import ValidationError


__all__ = [
    "Cells",
    "cell_centres",
    "cell_grid",
    "cells_in",
    "check_columns",
    "edges_in",
    "first_centre",
    "host_axes",
    "position_columns",
]


def _as_edges(edges: Any) -> tuple[Any, ...]:
    """The edges as floating arrays: integers become JAX's default float.

    Traced edges are checked when the program runs.
    """
    edges = tuple(edges)
    with host_axes(edges):
        arrays = [jnp.asarray(edge) for edge in edges]
        arrays = [array.astype(jnp.result_type(array, float)) for array in arrays]
    checked = []
    for index, array in enumerate(arrays):
        if not is_concrete(array) and array.ndim == 1:
            ordered = jnp.all(jnp.diff(array) > 0.0) & jnp.all(jnp.isfinite(array))
            array = eqx.error_if(
                array,
                ~ordered,
                f"'edges[{index}]' must be finite and strictly increasing",
            )
        checked.append(array)
    return tuple(checked)


def cell_centres(edge: Any) -> Any:
    """½ (e_i + e_{i+1}) of one axis."""
    return 0.5 * (edge[:-1] + edge[1:])


def _increasing(edge: Any) -> bool:
    """Whether every centre lies strictly between its edges, in their dtype."""
    values = np.asarray(edge)
    centres = cell_centres(values)
    ordered = (values[:-1] < centres) & (centres < values[1:])
    return bool(np.isfinite(values).all() and ordered.all())


def host_axes(edges: Any) -> contextlib.AbstractContextManager[Any]:
    """The context in which to build functax axes from the edges.

    An axis checks its nodes. Under a trace that check is a host callback at every
    step of a run, unless the nodes are concrete and are kept so.
    """
    if is_concrete(edges):
        return jax.ensure_compile_time_eval()
    return contextlib.nullcontext()


class Cells(eqx.Module):
    """Cells between edges, along one, two or three axes.

    Cell i of an axis is [e_i, e_{i+1}), so that a particle on an inner edge
    belongs to the upper cell and one on the last edge is outside. The cells may
    be stretched. The axes are in the order of the position columns: (x, y, z).

    Attributes:
        edges: One increasing array per axis, with one more value than the axis
            has cells [m].

    Raises:
        ValidationError: If there are not 1, 2 or 3 axes, or the edges of one are
            not 1-D, finite and strictly increasing, with at least two values. Of
            traced edges the values are checked when the program runs.

    Examples:
        Two cells by three, the second axis stretched:

        >>> import jax.numpy as jnp
        >>> from tracerax.bin import Cells
        >>> cells = Cells(([0.0, 10.0, 20.0], [0.0, 1.0, 3.0, 9.0]))
        >>> cells.shape, cells.centres[1].tolist()
        ((2, 3), [0.5, 2.0, 6.0])
        >>> cells.volumes()[0].tolist()
        [10.0, 20.0, 60.0]

        The last edge is outside:

        >>> cells.contains(jnp.array([[0.0, 8.9], [10.0, 9.0]])).tolist()
        [True, False]
    """

    edges: tuple[Float[Array, " _edge"], ...] = eqx.field(converter=_as_edges)

    def __check_init__(self) -> None:
        if len(self.edges) not in (1, 2, 3):
            raise ValidationError(
                f"'edges' must hold 1, 2 or 3 axes, got {len(self.edges)}"
            )
        for index, edge in enumerate(self.edges):
            if edge.ndim != 1 or edge.shape[0] < 2:
                raise ValidationError(
                    f"'edges[{index}]' must be 1-D with at least two values, "
                    f"got shape {edge.shape}"
                )
            if is_concrete(edge) and not _increasing(edge):
                raise ValidationError(
                    f"'edges[{index}]' must be finite and strictly increasing, "
                    f"got {np.asarray(edge)}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of cells along each axis."""
        return tuple(edge.shape[0] - 1 for edge in self.edges)

    @property
    def centres(self) -> tuple[Float[Array, " _cell"], ...]:
        """½ (e_i + e_{i+1}), one array per axis [m]."""
        return tuple(cell_centres(edge) for edge in self.edges)

    def volumes(self) -> Float[Array, "*cells"]:
        """|C_j|, the product of the widths of each cell.

        Returns:
            One value per cell, in metres to the power of the number of axes.
        """
        with host_axes(self.edges):
            grid = cell_grid(self.edges)
            # |C_j| = ∏_d (e_{d,j+1} − e_{d,j})
            return jnp.asarray(cell_area(grid).weights)

    def contains(
        self, position: Float[Array, "particle axis"]
    ) -> Bool[Array, " particle"]:
        """Whether each particle is in some cell: e_0 ≤ x < e_n on every axis.

        Args:
            position: x, with one column per axis of the cells [m].

        Returns:
            True where the particle is inside; False at NaN.

        Raises:
            ValidationError: If ``position`` has another number of columns.
        """
        position = jnp.asarray(position)
        check_columns(self, position)
        edges = edges_in(self, position.dtype)
        inside = jnp.ones(position.shape[0], dtype=bool)
        for index, edge in enumerate(edges):
            column = position[:, index]
            # e_0 ≤ x < e_n
            inside = inside & (column >= edge[0]) & (column < edge[-1])
        return inside


def check_columns(cells: Cells, position: Float[Array, "particle axis"]) -> None:
    """Raise unless the positions have one column per axis of the cells.

    Raises:
        ValidationError: If they have not.
    """
    n_axes = len(cells.edges)
    if position.ndim != 2 or position.shape[1] != n_axes:
        raise ValidationError(
            f"'position' must have one column per axis of the cells, {n_axes}, "
            f"got shape {position.shape}"
        )


def edges_in(cells: Cells, dtype: Any) -> tuple[Any, ...]:
    """The edges in the positions' dtype, on the host unless they are traced.

    Raises:
        ValidationError: If a cell is too narrow for the dtype, which changes its
            width by more than a hundredth.
    """
    cast = []
    for index, edge in enumerate(cells.edges):
        if not is_concrete(edge):
            cast.append(edge.astype(dtype))
            continue
        values = np.asarray(edge).astype(dtype)
        widths = np.diff(np.asarray(edge))
        rounding = np.abs(np.diff(values) - widths)
        if not _increasing(values) or (rounding > 0.01 * widths).any():
            raise ValidationError(
                f"the cells of 'edges[{index}]' are too narrow for {dtype} positions; "
                "count positions and edges from a local origin"
            )
        cast.append(values)
    return tuple(cast)


def cells_in(cells: Cells, dtype: Any) -> Cells:
    """The cells as particles of a dtype see them, with the edges in it.

    Raises:
        ValidationError: If a cell is too narrow for the dtype.
    """
    edges = edges_in(cells, dtype)
    with host_axes(edges):
        return Cells(edges)


def cell_grid(edges: tuple[Any, ...]) -> RectilinearGrid:
    """The functax grid of the cells: their centres, with the edges explicit.

    Build it inside `host_axes` of the edges, so that concrete edges stay off the
    trace.
    """
    return RectilinearGrid([Axis(cell_centres(edge), edges=edge) for edge in edges])


def first_centre(edges: tuple[Any, ...], dtype: Any) -> Float[Array, " axis"]:
    """The centre of the first cell, where a particle outside the cells is read."""
    return jnp.asarray([cell_centres(edge[:2])[0] for edge in edges], dtype=dtype)


def position_columns(cells: Cells) -> slice:
    """The columns of a state's positions that the axes of the cells are.

    Raises:
        ValidationError: If the cells have one axis, which a state cannot tell
            from a height.
    """
    n_axes = len(cells.edges)
    if n_axes == 1:
        raise ValidationError(
            "'cells' must have two axes (x, y) or three (x, y, z) to bin a state, "
            "got 1; use 'deposit' on one column of its positions"
        )
    return slice(n_axes)
