"""Velocities on a grid, each component read on its own staggered nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Interpolant, Linear, Policies
from jax.typing import ArrayLike
from jaxtyping import Array, Float

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
        >>> grids = cgrid_grids(times, (x + 0.5, y), (x, y + 0.5))
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


def cgrid_grids(
    times: Float[ArrayLike, " time"],
    u: Sequence[Float[ArrayLike, " _node"]],
    v: Sequence[Float[ArrayLike, " _node"]],
    w: Sequence[Float[ArrayLike, " _node"]] | None = None,
) -> tuple[RectilinearGrid, ...]:
    """The grids of the components of a C-grid, each on its own points.

    Each component is read on the axes (time, z, y, x), or (time, y, x), of the
    points where it sits. On a C-grid with T points (xᵢ, yⱼ, z_k), u sits half a
    cell along x and v half a cell along y; w sits on the T levels or half a
    level above them, as the model keeps it. The caller gives each position, so
    a stretched grid works and no shift is computed here.

    Args:
        times: The sample times [s], strictly increasing. One sample is read at
            every time.
        u: Where u sits, one array per axis, in the order (x, y) or (x, y, z) [m].
        v: Where v sits, along the same axes as ``u`` [m].
        w: Where w sits, along (x, y, z) [m], when ``u`` and ``v`` have three
            axes; ``None`` in two dimensions.

    Returns:
        The grids of (u, v) or (u, v, w), for `GriddedVelocity.fit`, with the
        axes (time, z, y, x) or (time, y, x).

    Raises:
        ValidationError: If ``u`` and ``v`` do not both have 2 or 3 axes, or
            ``w`` is missing in three dimensions, given in two, or has not 3 axes.

    Examples:
        A uniform C-grid with T points at x = 0, 1, 2, 3 and y = 0, 1, 2:

        >>> import jax.numpy as jnp
        >>> from tracerax.velocity import cgrid_grids
        >>> x, y = jnp.arange(4.0), jnp.arange(3.0)
        >>> grids = cgrid_grids(jnp.array([0.0]), (x + 0.5, y), (x, y + 0.5))
        >>> [grid.shape for grid in grids]
        [(1, 3, 4), (1, 3, 4)]
        >>> grids[0].axes[2].values.tolist(), grids[1].axes[1].values.tolist()
        ([0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5])

        A stretched axis is read where its points are:

        >>> z = jnp.array([0.0, 1.0, 5.0])
        >>> grids = cgrid_grids(
        ...     jnp.array([0.0]), (x + 0.5, y, z), (x, y + 0.5, z), (x, y, z + 0.5)
        ... )
        >>> grids[2].axes[1].values.tolist()
        [0.5, 1.5, 5.5]
    """
    u, v = tuple(u), tuple(v)
    n_axes = len(u)
    if n_axes not in (2, 3) or len(v) != n_axes:
        raise ValidationError(
            "'u' and 'v' must both have 2 axes (x, y) or 3 (x, y, z), got "
            f"{len(u)} and {len(v)}"
        )
    components = [u, v]
    if n_axes == 3:
        if w is None or len(tuple(w)) != 3:
            got = "None" if w is None else f"{len(tuple(w))} axes"
            raise ValidationError(
                f"'w' must have 3 axes (x, y, z) when 'u' and 'v' have 3, got {got}"
            )
        components.append(tuple(w))
    elif w is not None:
        raise ValidationError("'w' must be None when 'u' and 'v' have 2 axes")
    time_axis = Axis(jnp.asarray(times))
    grids = []
    for axes in components:
        # (x, y, z) → (z, y, x)
        spatial = [Axis(jnp.asarray(values)) for values in reversed(axes)]
        grids.append(RectilinearGrid((time_axis, *spatial)))
    return tuple(grids)
