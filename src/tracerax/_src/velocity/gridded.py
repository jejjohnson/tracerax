"""Velocities on a grid, each component read on its own staggered nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Interpolant, Linear, Policies
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from tracerax._src._validate import is_concrete
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "GriddedVelocity",
    "cgrid_grids",
    "fit_components",
    "read_components",
]

_LINEAR = Linear()
_CLAMP = Policies(outside="clamp")


def fit_components(
    grids: Sequence[RectilinearGrid],
    fields: Sequence[Float[ArrayLike, "..."]],
    *,
    scheme: Any,
    policies: Policies,
) -> tuple[Interpolant, ...]:
    """One interpolant per position column, each on its own (time, …) grid.

    Args:
        grids: One grid per component, with the axes (time, z, y, x) or
            (time, y, x).
        fields: The samples of each component, shaped like its grid.
        scheme: The local functax scheme that fits each field.
        policies: What to read outside the grid and at missing samples.

    Returns:
        The interpolants, in the order of the fields.

    Raises:
        ValidationError: If there are not 2 or 3 components, one grid per
            component, 1 + D axes per grid, or a field shaped like its grid.
    """
    grids, fields = tuple(grids), tuple(fields)
    dim = len(fields)
    if dim not in (2, 3):
        raise ValidationError(f"'fields' must hold 2 or 3 components, got {dim}")
    if len(grids) != dim:
        raise ValidationError(
            f"'grids' must hold one grid per component, {dim}, got {len(grids)}"
        )
    components = []
    for index, (grid, field) in enumerate(zip(grids, fields, strict=True)):
        if len(grid.shape) != 1 + dim:
            raise ValidationError(
                f"'grids[{index}]' must have a time axis and {dim} axes in space, "
                f"got {len(grid.shape)} axes"
            )
        samples = jnp.asarray(field)
        if samples.shape != grid.shape:
            raise ValidationError(
                f"'fields[{index}]' must have the shape of its grid, {grid.shape}, "
                f"got {samples.shape}"
            )
        components.append(scheme.fit(grid, jnp.ravel(samples), policies=policies))
    return tuple(components)


def read_components(
    components: tuple[Interpolant, ...],
    t: ScalarLike | Float[ArrayLike, "*#particle"],
    position: Float[Array, "particle axis"],
) -> Float[Array, "particle axis"]:
    """Read every component at the particles: f_d(t, x_p), d = 1, …, D.

    Args:
        components: One interpolant per position column, from `fit_components`.
        t: One time, or one per particle [s].
        position: x, with columns (x, y) or (x, y, z).

    Returns:
        One value per particle and component, in the positions' dtype.

    Raises:
        ValidationError: If the positions have another dimension.
    """
    n_particles, dim = position.shape
    if dim != len(components):
        raise ValidationError(
            f"'position' must have {len(components)} columns, one per component, "
            f"got shape {position.shape}"
        )
    times = jnp.broadcast_to(jnp.asarray(t, dtype=position.dtype), (n_particles,))
    # (t, z, y, x): time first, then the position columns reversed
    targets = jnp.column_stack([times, position[:, ::-1]])
    # f_d = H_d(targets) c_d
    values = [component(targets) for component in components]
    return jnp.column_stack(values).astype(position.dtype)


class GriddedVelocity(eqx.Module):
    """A velocity on a grid, each component on its own staggered nodes.

    u_d(t, x) = ∑ₛ wₛ c_{d, iₛ}, d = 1, …, D, where the taps iₛ and the weights
    wₛ are those of component d's own grid. The default is multilinear in space
    and piecewise linear in time, and holds the nearest sample outside the grid,
    in space and in time. Reading u and v as if they sat where w does would shift
    them by half a cell; `cgrid_grids` gives the grids of a C-grid.

    The samples are array leaves, so a gradient reaches them.

    Attributes:
        components: One functax interpolant per component; build them with
            `fit`.

    Examples:
        A field that is linear in x comes back as x at any particle:

        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import GriddedVelocity, cgrid_grids
        >>> times, x, y = jnp.array([0.0, 60.0]), jnp.arange(4.0), jnp.arange(3.0)
        >>> grids = cgrid_grids(times, x, y)
        >>> u = jnp.broadcast_to(x + 0.5, (2, 3, 4))
        >>> velocity = GriddedVelocity.fit(grids, (u, jnp.zeros((2, 3, 4))))
        >>> velocity(30.0, jnp.array([[1.25, 0.5], [2.0, 1.5]])).tolist()
        [[1.25, 0.0], [2.0, 0.0]]
    """

    components: tuple[Interpolant, ...]

    @classmethod
    def fit(
        cls,
        grids: Sequence[RectilinearGrid],
        fields: Sequence[Float[ArrayLike, "..."]],
        *,
        scheme: Any = _LINEAR,
        policies: Policies = _CLAMP,
    ) -> GriddedVelocity:
        """Fit the samples of each component on its own grid.

        Args:
            grids: One grid per component, with the axes (time, z, y, x) or
                (time, y, x). A periodic axis has its ``period``, and its nodes
                must cover the period.
            fields: The samples of each component [m s⁻¹], shaped like its grid.
            scheme: The local functax scheme. `functax.interp.Cubic` has a
                continuous gradient, which an adaptive step needs.
            policies: What to read outside the grid and at missing samples. A
                field with NaN on land needs ``missing="renormalise"``.

        Returns:
            The velocity.

        Raises:
            ValidationError: If there are not 2 or 3 components, one grid per
                component, 1 + D axes per grid, or a field shaped like its grid.
        """
        return cls(fit_components(grids, fields, scheme=scheme, policies=policies))

    def __call__(
        self,
        t: ScalarLike | Float[ArrayLike, "*#particle"],
        position: Float[Array, "particle axis"],
    ) -> Float[Array, "particle axis"]:
        """The velocity at each particle.

        Args:
            t: One time, or one per particle, on the clock of the time axis [s].
            position: x, with columns (x, y) or (x, y, z) [m].

        Returns:
            u(t, x) [m s⁻¹], in the positions' dtype.

        Raises:
            ValidationError: If the positions have another dimension.
        """
        return read_components(self.components, t, position)


def _half_spacing(values: Float[Array, " node"], name: str) -> Float[Array, ""]:
    """Δ/2 of a uniform axis.

    Raises:
        ValidationError: If it has one node, or is concrete and not uniform.
    """
    size = values.shape[0]
    if size < 2:
        raise ValidationError(f"'{name}' must have at least 2 nodes, got {size}")
    # Δ = (a_{n−1} − a₀) / (n − 1)
    spacing = (values[-1] - values[0]) / (size - 1)
    if is_concrete(values):
        nodes = np.asarray(values)
        # in NumPy, because `spacing` is traced under a transformation
        steps = np.diff(nodes)
        mean_step = (nodes[-1] - nodes[0]) / (size - 1)
        # rounding of nodes as large as the largest, whatever the dtype
        tolerance = 8.0 * np.finfo(nodes.dtype).eps * np.abs(nodes).max()
        if np.abs(steps - mean_step).max() > max(tolerance, 1e-6 * abs(mean_step)):
            raise ValidationError(
                f"'{name}' must be uniformly spaced for a C-grid; give "
                "GriddedVelocity.fit the grids of a stretched one"
            )
    return 0.5 * spacing


def cgrid_grids(
    times: Float[ArrayLike, " time"],
    x: Float[ArrayLike, " x"],
    y: Float[ArrayLike, " y"],
    z: Float[ArrayLike, " z"] | None = None,
) -> tuple[RectilinearGrid, ...]:
    """The grids of the components of a uniform C-grid, for `GriddedVelocity.fit`.

    With T points (xᵢ, yⱼ, z_k), u sits at (xᵢ + Δx/2, yⱼ, z_k), v at
    (xᵢ, yⱼ + Δy/2, z_k) and w at the T points. A model that keeps w half a
    level higher passes its own grid for it.

    Args:
        times: The sample times [s], strictly increasing. One sample is read at
            every time.
        x: The T points along x [m], uniformly spaced.
        y: The T points along y [m], uniformly spaced.
        z: The T points along z [m], of any spacing, or ``None`` in two
            dimensions.

    Returns:
        The grids of (u, v) or (u, v, w), with the axes (time, z, y, x) or
        (time, y, x).

    Raises:
        ValidationError: If x or y has one node or is not uniformly spaced.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import cgrid_grids
        >>> grids = cgrid_grids(jnp.array([0.0]), jnp.arange(4.0), jnp.arange(3.0))
        >>> [grid.shape for grid in grids]
        [(1, 3, 4), (1, 3, 4)]
        >>> grids[0].axes[2].values.tolist(), grids[1].axes[1].values.tolist()
        ([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5])
    """
    time_axis = Axis(jnp.asarray(times))
    nodes_x, nodes_y = jnp.asarray(x), jnp.asarray(y)
    axis_x, axis_y = Axis(nodes_x), Axis(nodes_y)
    levels = () if z is None else (Axis(jnp.asarray(z)),)
    # u at xᵢ + Δx/2, v at yⱼ + Δy/2
    shifted_x = Axis(nodes_x + _half_spacing(nodes_x, "x"))
    shifted_y = Axis(nodes_y + _half_spacing(nodes_y, "y"))
    grid_u = RectilinearGrid((time_axis, *levels, axis_y, shifted_x))
    grid_v = RectilinearGrid((time_axis, *levels, shifted_y, axis_x))
    if z is None:
        return grid_u, grid_v
    return grid_u, grid_v, RectilinearGrid((time_axis, *levels, axis_y, axis_x))
