"""The skill of a trajectory against an observed one."""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geophysix.earth import SPHERE, Ellipsoid
from jaxtyping import Array, Bool, Float

from tracerax._src._validate import is_concrete
from tracerax._src.analysis.distance import (
    Distance,
    checked_positions,
    distance_between,
    known,
)
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "cumulative_separation",
    "separation",
    "skill_score",
]


def _checked_bound(name: str, value: ScalarLike, *, zero: bool) -> ScalarLike:
    """The number, checked to be one, finite, and above zero or on it.

    Raises:
        ValidationError: If it is not a scalar, or a concrete one is out of range.
        EquinoxRuntimeError: If a traced one is.
    """
    if jnp.shape(value) != ():
        raise ValidationError(
            f"'{name}' must be a scalar, got shape {jnp.shape(value)}"
        )
    words = "a finite length of at least 0" if zero else "finite and positive"
    concrete = is_concrete(value)
    number = np.asarray(value) if concrete else jnp.asarray(value)
    above = number >= 0.0 if zero else number > 0.0
    inside = above & (number < np.inf)
    if not concrete:
        return eqx.error_if(number, ~inside, f"'{name}' must be {words}")
    if not inside:
        raise ValidationError(f"'{name}' must be {words}, got {value}")
    return value


def _paired(
    trajectory: Any, reference: Any, distance: Distance, earth: Ellipsoid
) -> tuple[Any, Any, Bool[Array, "*batch time"]]:
    """The two, broadcast and with something finite at each missing observation.

    Raises:
        ValidationError: If they do not fit each other or the distance.
    """
    trajectory, reference = checked_positions(
        distance, earth, trajectory=trajectory, reference=reference
    )
    if trajectory.ndim < 2 or trajectory.shape[-2:] != reference.shape[-2:]:
        raise ValidationError(
            "'trajectory' and 'reference' must have the same times and columns, "
            f"got shapes {trajectory.shape} and {reference.shape}"
        )
    dtype = trajectory.dtype
    trajectory, reference = jnp.broadcast_arrays(trajectory, reference.astype(dtype))
    observed = known(reference)
    filled = "... time, ... time axis, -> ... time axis"
    trajectory = einx.where(filled, observed, trajectory, 0.0)
    reference = einx.where(filled, observed, reference, 0.0)
    return trajectory, reference, observed


def separation(
    trajectory: Float[Array, "*batch time axis"],
    reference: Float[Array, "*#batch time axis"],
    *,
    distance: Distance = "euclidean",
    earth: Ellipsoid = SPHERE,
) -> Float[Array, "*batch time"]:
    """The distance between a trajectory and an observed one, time by time.

    d_k = d(x_k, y_k)

    Args:
        trajectory: x, simulated [m], or (longitude, latitude) in degrees. Time
            comes before the columns: for the ``trajectory`` of
            `tracerax.advect.integrate` swap its first two axes.
        reference: y, observed at the same times; NaN where there is none.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        d [m], NaN at a missing observation and where the trajectory is not
        finite. Its gradient is 0 where d = 0.

    Raises:
        ValidationError: If the two have other times or columns, or ``distance``
            is unknown, or does not fit them.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import separation
        >>> observed = jnp.array([[0.0, 0.0], [10.0, 0.0], [jnp.nan, jnp.nan]])
        >>> simulated = jnp.array([[0.0, 0.0], [13.0, 4.0], [20.0, 0.0]])
        >>> separation(simulated, observed).tolist()
        [0.0, 5.0, nan]
    """
    trajectory, reference, observed = _paired(trajectory, reference, distance, earth)
    apart = distance_between(trajectory, reference, distance, earth)
    return jnp.where(observed, apart, jnp.nan)


def _length_travelled(
    reference: Float[Array, "*batch time axis"],
    observed: Bool[Array, "*batch time"],
    distance: Distance,
    earth: Ellipsoid,
) -> Float[Array, "*batch time"]:
    """l_k, the length of the observed path up to each time.

    A gap is bridged in a straight line, or along a great circle.
    """
    n_times = reference.shape[-2]
    index = jnp.where(observed, jnp.arange(n_times), -1)
    # the last observation up to each time, and up to the one before
    last = jax.lax.cummax(index, axis=index.ndim - 1)
    before = jnp.concatenate([jnp.full_like(last[..., :1], -1), last[..., :-1]], -1)
    taken = einx.id("... time -> ... time 1", jnp.maximum(before, 0))
    previous = jnp.take_along_axis(reference, taken, axis=-2)
    piece = distance_between(previous, reference, distance, earth)
    counted = observed & (before >= 0)
    piece = jnp.where(counted, piece, jnp.zeros((), piece.dtype))
    return jnp.cumsum(piece, axis=-1)


def cumulative_separation(
    trajectory: Float[Array, "*batch time axis"],
    reference: Float[Array, "*#batch time axis"],
    *,
    distance: Distance = "euclidean",
    min_length: ScalarLike = 0.0,
    earth: Ellipsoid = SPHERE,
) -> Float[Array, "*batch"]:
    """The normalised cumulative separation of a trajectory from an observed one.

        s = ∑ₖ d_k / ∑ₖ l_k

    with d_k the separation at time k and l_k the length of the observed path up
    to it, both summed over the times with an observation. It is 0 for a perfect
    trajectory, and about 1 for one whose errors grow like the path. It is also a
    loss: its gradient is finite where the two coincide.

    Args:
        trajectory: x, as in `separation`.
        reference: y, as in `separation`.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        min_length: A length added to each l_k [m], without which an observed
            path that does not move gives s = ∞, or NaN when d = 0 too.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        s, NaN without any observation. Where it is ∞ or NaN for want of a length
        its gradient is 0.

    Raises:
        ValidationError: As `separation` does, or if ``min_length`` is negative.

    References:
        Liu, Y. and Weisberg, R. H. (2011). Evaluation of trajectory modeling in
        different dynamic regions using normalized cumulative Lagrangian
        separation. J. Geophys. Res. 116, C09013.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import cumulative_separation
        >>> observed = jnp.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        >>> simulated = jnp.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        >>> float(cumulative_separation(simulated, observed))
        0.5
    """
    min_length = _checked_bound("min_length", min_length, zero=True)
    trajectory, reference, observed = _paired(trajectory, reference, distance, earth)
    apart = distance_between(trajectory, reference, distance, earth)
    length = _length_travelled(reference, observed, distance, earth)
    length = length + jnp.asarray(min_length, length.dtype)
    zero = jnp.zeros((), apart.dtype)
    # ∑ₖ d_k and ∑ₖ l_k
    total = einx.sum("... [time]", jnp.where(observed, apart, zero))
    scale = einx.sum("... [time]", jnp.where(observed, length, zero))
    # without a length, s is ∞, or NaN where there is no separation either
    travelled = scale > 0.0
    quotient = total / jnp.where(travelled, scale, 1.0)
    empty = jnp.where(total > 0.0, jnp.inf, jnp.nan).astype(total.dtype)
    return jnp.where(travelled, quotient, empty)


def skill_score(
    trajectory: Float[Array, "*batch time axis"],
    reference: Float[Array, "*#batch time axis"],
    *,
    tolerance: ScalarLike = 1.0,
    distance: Distance = "euclidean",
    min_length: ScalarLike = 0.0,
    earth: Ellipsoid = SPHERE,
) -> Float[Array, "*batch"]:
    """The skill of a trajectory: 1 for a perfect one, 0 for one beyond tolerance.

    1 − s/n where s ≤ n, and 0 elsewhere, with s the `cumulative_separation`.

    Args:
        trajectory: x, as in `separation`.
        reference: y, as in `separation`.
        tolerance: n, the separation beyond which there is no skill, usually 1.
        distance: ``"euclidean"``, or ``"sphere"`` for great-circle distances.
        min_length: As in `cumulative_separation`.
        earth: The sphere of ``distance="sphere"``.

    Returns:
        The skill, NaN without any observation. Beyond the tolerance its gradient
        is 0, so fit with the separation.

    Raises:
        ValidationError: As `cumulative_separation` does, or if ``tolerance`` is
            not positive.

    References:
        Liu, Y. and Weisberg, R. H. (2011). Evaluation of trajectory modeling in
        different dynamic regions using normalized cumulative Lagrangian
        separation. J. Geophys. Res. 116, C09013.

    Examples:
        >>> import jax.numpy as jnp
        >>> from tracerax.analysis import skill_score
        >>> observed = jnp.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        >>> simulated = jnp.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        >>> float(skill_score(simulated, observed))
        0.5
        >>> float(skill_score(simulated, observed, tolerance=0.4))
        0.0
    """
    tolerance = _checked_bound("tolerance", tolerance, zero=False)
    s = cumulative_separation(
        trajectory,
        reference,
        distance=distance,
        min_length=min_length,
        earth=earth,
    )
    limit = jnp.asarray(tolerance, s.dtype)
    beyond = s > limit
    within = jnp.where(beyond, jnp.zeros((), s.dtype), s)
    # 1 − s/n where s ≤ n, and NaN where s is
    return jnp.where(beyond, jnp.zeros((), s.dtype), 1.0 - within / limit)
