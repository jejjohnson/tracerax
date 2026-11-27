"""Labelled regions: the time spent in each, and which are connected."""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from functax.geometry import Points
from functax.interp import Binning
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int

from tracerax._src._validate import check_count, is_concrete
from tracerax._src.advect.masks import stand_in
from tracerax._src.bin.cells import (
    Cells,
    cell_grid,
    edges_in,
    first_centre,
    host_axes,
    position_columns,
)
from tracerax._src.particles.state import ParticleState
from tracerax.exceptions import ValidationError


__all__ = [
    "RegionTime",
    "Regions",
    "connectivity",
    "visited_connectivity",
]


class Regions(eqx.Module):
    """Regions as a label per cell.

    L(x) is the label of the cell that x is in, and −1 in no region and outside
    the cells. A cell is [e_i, e_{i+1}) on each axis, as in `Cells`, so every
    quantity made of labels is piecewise constant in the positions.

    Attributes:
        cells: The cells, with two axes (x, y) or three (x, y, z). Two axes on
            particles in three dimensions label every height together.
        labels: The label of each cell, from 0 to ``n_regions`` − 1, or −1.
        n_regions: R, the number of regions. Static.

    Raises:
        ValidationError: If the cells have one axis, the labels are not integers
            of the cells' shape, ``n_regions`` is below 1, or a concrete label is
            outside its range.
        EquinoxRuntimeError: If a traced label is.

    Examples:
        A coast of two cells, and the open sea:

        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import Regions
        >>> from tracerax.bin import Cells
        >>> cells = Cells(([0.0, 10.0, 20.0, 30.0], [0.0, 5.0]))
        >>> regions = Regions(cells, jnp.array([[0], [0], [1]]), n_regions=2)
        >>> position = jnp.array([[3.0, 1.0], [20.0, 1.0], [30.0, 1.0]])
        >>> regions.at(position).tolist()
        [0, 1, -1]
    """

    cells: Cells
    labels: Int[Array, "*cells"]
    n_regions: int = eqx.field(static=True)

    def __init__(self, cells: Cells, labels: ArrayLike, n_regions: int) -> None:
        position_columns(cells)
        check_count(n_regions, "n_regions")
        array = jnp.asarray(labels)
        if array.shape != cells.shape:
            raise ValidationError(
                f"'labels' must have the shape of the cells, {cells.shape}, "
                f"got {array.shape}"
            )
        if not jnp.issubdtype(array.dtype, jnp.integer):
            raise ValidationError(f"'labels' must be integers, got dtype {array.dtype}")
        message = f"'labels' must be from −1 to {n_regions - 1}"
        if is_concrete(array):
            values = np.asarray(array)
            # a Python integer compares with any dtype
            if int(values.min()) < -1 or int(values.max()) >= n_regions:
                raise ValidationError(
                    f"{message}, got {values.min()} to {values.max()}"
                )
        else:
            array = array.astype(int)
            outside = (array < -1) | (array >= n_regions)
            array = eqx.error_if(array, outside, message)
        self.cells = cells
        # a dtype without a sign has no −1 to give outside the cells
        self.labels = array.astype(int)
        self.n_regions = n_regions

    def at(self, position: Float[Array, "particle axis"]) -> Int[Array, " particle"]:
        """L(x), the label at each particle.

        Args:
            position: x, with the columns of the cells, and any after them.

        Returns:
            The labels, −1 in no region, outside the cells and at NaN.

        Raises:
            ValidationError: If there are fewer columns than the cells have axes,
                or a cell is too narrow for the positions' dtype.
        """
        columns = position[:, position_columns(self.cells)]
        inside = self.cells.contains(columns)
        dtype = columns.dtype
        edges = edges_in(self.cells, dtype)
        # a particle outside has no cell: it is read in the first one
        columns = stand_in(columns, inside, first_centre(edges, dtype))
        with host_axes(edges):
            grid = cell_grid(edges)
            # l_p = (H l)_p: the adjoint of a sum reads the cell between the edges
            plan = Binning("sum").plan(Points(columns), grid)
        read = plan.adjoint(jnp.ravel(self.labels).astype(dtype))
        label = jnp.round(read).astype(self.labels.dtype)
        return jnp.where(inside, label, -1)


class RegionTime(eqx.Module):
    """The time each particle of a run spends in each region, an `Accumulator`.

    τ_{p,i} = ∑ₖ δ_{p,k} 1[L(x_p(t_k)) = i], with δ_{p,k} how long particle p
    moved for in step k and x_p(t_k) its position after the move: a
    right-endpoint rule in time, which is right to one step per crossing. The
    residence time of a region is the mean of a column over the particles
    released in it.

    Attributes:
        regions: The regions.
        n_particles: N, the number of particles of the run. Static.

    Raises:
        ValidationError: If ``n_particles`` is below 1.

    Examples:
        A drifter that crosses a region 10 m wide at 2 m s⁻¹:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Deterministic, integrate
        >>> from tracerax.analysis import RegionTime, Regions
        >>> from tracerax.bin import Cells
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> cells = Cells(([0.0, 10.0, 30.0], [0.0, 1.0]))
        >>> regions = Regions(cells, jnp.array([[0], [1]]), n_regions=2)
        >>> result = integrate(
        ...     at_point(jnp.array([0.5, 0.5]), 1),
        ...     Deterministic(uniform(jnp.array([2.0, 0.0]))),
        ...     t0=0.0,
        ...     duration=20.0,
        ...     step_size=0.5,
        ...     accumulate=RegionTime(regions, n_particles=1),
        ... )
        >>> result.accumulated.tolist()
        [[4.5, 10.0]]
    """

    regions: Regions
    n_particles: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        check_count(self.n_particles, "n_particles")

    def init(self) -> Float[Array, "particle region"]:
        """Zero for every particle and region, in the dtype of the edges."""
        dtype = jnp.result_type(*self.regions.cells.edges)
        return jnp.zeros((self.n_particles, self.regions.n_regions), dtype)

    def update(
        self,
        total: Float[Array, "particle region"],
        state: ParticleState,
        duration: Float[Array, " particle"],
    ) -> Float[Array, "particle region"]:
        """Add one step's contribution.

        Args:
            total: τ so far [s].
            state: The state after the move.
            duration: δ, how long each particle moved for [s].

        Returns:
            The new τ [s], in the dtype of ``total``.

        Raises:
            ValidationError: If the state has another number of particles.
        """
        if state.n_particles != self.n_particles:
            raise ValidationError(
                f"'n_particles' must be that of the run, {state.n_particles}, "
                f"got {self.n_particles}"
            )
        label = self.regions.at(state.position)
        counted = state.active & (label >= 0)
        # δ 1[active, in a region]
        spent = jnp.where(counted, duration, jnp.zeros((), duration.dtype))
        particle = jnp.arange(self.n_particles)
        region = jnp.where(counted, label, 0)
        return total.at[particle, region].add(spent.astype(total.dtype))


def _fractions(
    released: Float[Array, "particle region"], reached: Float[Array, "particle region"]
) -> Float[Array, "region region"]:
    """The fraction of each region's particles that reached each region."""
    # N_i
    number = einx.sum("[particle] region", released)
    pairs = einx.dot("[particle] i, [particle] j -> i j", released, reached)
    some = number > 0.0
    # C_ij = (1/N_i) ∑_{p ∈ i} 1[p reached j]
    safe = jnp.where(some, number, 1.0)
    fraction = einx.divide("i j, i -> i j", pairs, safe)
    return einx.where("i, i j, -> i j", some, fraction, jnp.nan)


def _released(
    start: Float[Array, "particle axis"],
    regions: Regions,
    counted: Bool[Array, " particle"] | None,
) -> Float[Array, "particle region"]:
    """1[L(x_p(t₀)) = i], for the particles that are counted.

    Raises:
        ValidationError: If ``counted`` has not one entry per particle.
    """
    label = regions.at(start)
    if counted is not None:
        if counted.shape != label.shape:
            raise ValidationError(
                f"'counted' must have one entry per particle, {label.shape}, "
                f"got shape {counted.shape}"
            )
        label = jnp.where(counted, label, -1)
    return _one_hot(label, regions, start.dtype)


def _one_hot(
    label: Int[Array, " particle"], regions: Regions, dtype: Any
) -> Float[Array, "particle region"]:
    """1[L = i], and zero for the label −1."""
    return jax.nn.one_hot(label, regions.n_regions, dtype=dtype)


def connectivity(
    start: Float[Array, "particle axis"],
    end: Float[Array, "particle axis"],
    regions: Regions,
    *,
    counted: Bool[Array, " particle"] | None = None,
) -> Float[Array, "region region"]:
    """The fraction of the particles of each region that end in each region.

    C_ij = (1/N_i) ∑_{p ∈ i} 1[L(x_p(t₁)) = j], with N_i the number of particles
    that start in region i. It is piecewise constant in the positions.

    Args:
        start: x(t₀), where each particle was released.
        end: x(t₁), where it is at the end.
        regions: The regions.
        counted: Which particles to count, or ``None`` for all.

    Returns:
        C. A row sums to one when no particle of its region ends outside every
        region, and to less otherwise. The row of a region without particles is
        NaN.

    Raises:
        ValidationError: If the two have another number of particles, or
            ``counted`` has not one entry per particle.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import Regions, connectivity
        >>> from tracerax.bin import Cells
        >>> cells = Cells(([0.0, 10.0, 20.0], [0.0, 1.0]))
        >>> regions = Regions(cells, jnp.array([[0], [1]]), n_regions=2)
        >>> start = jnp.array([[1.0, 0.5], [2.0, 0.5], [12.0, 0.5]])
        >>> end = jnp.array([[2.0, 0.5], [15.0, 0.5], [25.0, 0.5]])
        >>> connectivity(start, end, regions).tolist()
        [[0.5, 0.5], [0.0, 0.0]]
    """
    if start.shape[0] != end.shape[0]:
        raise ValidationError(
            f"'end' must have one position per particle of 'start', {start.shape[0]}, "
            f"got {end.shape[0]}"
        )
    released = _released(start, regions, counted)
    reached = _one_hot(regions.at(end), regions, start.dtype)
    return _fractions(released, reached)


def visited_connectivity(
    start: Float[Array, "particle axis"],
    time: Float[Array, "particle region"],
    regions: Regions,
    *,
    counted: Bool[Array, " particle"] | None = None,
) -> Float[Array, "region region"]:
    """The fraction of the particles of each region that ever visit each region.

    C_ij = (1/N_i) ∑_{p ∈ i} 1[τ_{p,j} > 0], with τ the total of a `RegionTime`
    of the run. A particle may visit several regions, so a row may sum to more
    than one. A `RegionTime` sums where particles are after each step, so the
    start itself is not a visit: a particle that leaves its region in the first
    step has not visited it.

    Args:
        start: x(t₀), where each particle was released.
        time: τ, the time each particle spent in each region [s].
        regions: The regions.
        counted: Which particles to count, or ``None`` for all.

    Returns:
        C. The row of a region without particles is NaN.

    Raises:
        ValidationError: If ``time`` has not one row per particle and one column
            per region, or ``counted`` not one entry per particle.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import Regions, visited_connectivity
        >>> from tracerax.bin import Cells
        >>> cells = Cells(([0.0, 10.0, 20.0], [0.0, 1.0]))
        >>> regions = Regions(cells, jnp.array([[0], [1]]), n_regions=2)
        >>> start = jnp.array([[1.0, 0.5], [2.0, 0.5]])
        >>> time = jnp.array([[30.0, 0.0], [10.0, 20.0]])
        >>> visited_connectivity(start, time, regions)[0].tolist()
        [1.0, 0.5]
    """
    shape = (start.shape[0], regions.n_regions)
    if time.shape != shape:
        raise ValidationError(
            f"'time' must have one row per particle and one column per region, "
            f"{shape}, got shape {time.shape}"
        )
    released = _released(start, regions, counted)
    visited = (time > 0.0).astype(start.dtype)
    return _fractions(released, visited)
