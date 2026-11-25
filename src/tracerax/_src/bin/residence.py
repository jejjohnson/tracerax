"""The residence accumulator: the time particles spend in each cell of a run."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from tracerax._src.bin.cells import Cells, position_columns
from tracerax._src.bin.deposit import check_kind, deposit
from tracerax._src.particles.state import ParticleState


__all__ = ["Residence"]


class Residence(eqx.Module):
    """The time the particles of a run spend in each cell, an `Accumulator`.

    R_j = ∑ₚ ∑ₖ δ_{p,k} 1[x_p(t_k) ∈ j], with δ_{p,k} how long particle p moved for
    in step k and x_p(t_k) its position after the move: a right-endpoint rule in
    time. A particle adds nothing before its release, nor from the step that takes
    it out of the domain on.

    Attributes:
        cells: The cells, with two axes (x, y) or three (x, y, z). Two axes on
            particles in three dimensions bin every height together.
        kind: ``"nearest"`` or ``"linear"``, as in `deposit`. Only the latter is
            differentiable in the positions.
        where: A further condition on the state, such as a height, or ``None``. It
            is a step, so no gradient flows through it: what is differentiated is
            the sum over the particles that are counted.

    Raises:
        ValidationError: If the cells have one axis, or ``kind`` is unknown.

    Examples:
        A drifter that crosses two cells, at 2 m s⁻¹:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Deterministic, integrate
        >>> from tracerax.bin import Cells, Residence
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> cells = Cells(([0.0, 10.0, 30.0], [0.0, 1.0]))
        >>> result = integrate(
        ...     at_point(jnp.array([0.5, 0.5]), 1),
        ...     Deterministic(uniform(jnp.array([2.0, 0.0]))),
        ...     t0=0.0,
        ...     duration=20.0,
        ...     step_size=0.5,
        ...     accumulate=Residence(cells),
        ... )
        >>> result.accumulated.shape, float(result.accumulated[0, 0])
        ((2, 1), 4.5)

        Only what is in the first 20 m:

        >>> near = Residence(cells, where=lambda state: state.position[:, 0] < 20.0)
        >>> total = near.update(near.init(), result.state, jnp.array([0.5]))
        >>> bool((total == 0.0).all())
        True
    """

    cells: Cells
    kind: Literal["nearest", "linear"] = eqx.field(static=True, default="nearest")
    where: Callable[[ParticleState], Bool[Array, " particle"]] | None = None

    def __check_init__(self) -> None:
        check_kind(self.kind)
        position_columns(self.cells)

    def init(self) -> Float[Array, "*cells"]:
        """Zero in every cell, in the dtype of the edges."""
        return jnp.zeros(self.cells.shape, jnp.result_type(*self.cells.edges))

    def update(
        self,
        total: Float[Array, "*cells"],
        state: ParticleState,
        duration: Float[Array, " particle"],
    ) -> Float[Array, "*cells"]:
        """Add one step's contribution.

        Args:
            total: R so far [s].
            state: The state after the move.
            duration: δ, how long each particle moved for [s].

        Returns:
            The new R [s], in the dtype of ``total``.
        """
        counted = state.active
        if self.where is not None:
            counted = counted & self.where(state)
        # δ 1[active]
        weight = jnp.where(counted, duration, jnp.zeros((), duration.dtype))
        position = state.position[:, position_columns(self.cells)]
        # ∑ₚ δₚ 1[xₚ ∈ j]
        spent = deposit(self.cells, position, weight, kind=self.kind)
        return total + spent.astype(total.dtype)
