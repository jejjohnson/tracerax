"""The integration loop: it runs any particle step over a duration."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from integrix.adjoint import Checkpointed, Truncated, scan_for
from integrix.exceptions import ValidationError as IntegrixValidationError
from integrix.march import count_steps, step_lengths
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from pipekit_ops import is_concrete

from tracerax._src._validate import check_count
from tracerax._src.advect.domain import Box
from tracerax._src.advect.masks import choose, stand_in
from tracerax._src.particles.state import ParticleState
from tracerax._src.particles.step import ParticleStep
from tracerax._typing import ScalarLike
from tracerax.exceptions import ValidationError


__all__ = [
    "Accumulator",
    "Result",
    "check_epoch",
    "integrate",
]


@runtime_checkable
class Accumulator(Protocol):
    """What a run sums: a run of many steps then stores sums and no trajectory.

    Any object with these two methods is one.

    Examples:
        The time each particle has spent moving:

        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Accumulator
        >>> class Elapsed(eqx.Module):
        ...     size: int
        ...
        ...     def init(self):
        ...         return jnp.zeros(self.size)
        ...
        ...     def update(self, total, state, duration):
        ...         return total + duration
        >>> isinstance(Elapsed(4), Accumulator)
        True
    """

    def init(self) -> PyTree:
        """The total before the first step."""
        ...

    def update(
        self,
        total: PyTree,
        state: ParticleState,
        duration: Float[Array, " particle"],
    ) -> PyTree:
        """The total with one step's contribution added.

        Args:
            total: The total so far.
            state: The state after the move.
            duration: δ, how long each particle moved for in this step [s], and 0
                for one that did not move.

        Returns:
            The new total, of the same structure, shapes and dtypes.
        """
        ...


class Result(eqx.Module):
    """What `integrate` returns.

    Attributes:
        state: The state at the end of the run.
        accumulated: The accumulator's total, or ``None`` without one.
        trajectory: The positions of the input state and of every
            ``save_every``-th one after it, or ``None``.
        offsets: The offsets from ``t0`` they were saved at, signed like
            ``duration`` [s], or ``None``.
    """

    state: ParticleState
    accumulated: PyTree
    trajectory: Float[Array, "save particle axis"] | None
    offsets: Float[Array, " save"] | None


def _check_run(duration: Any, step_size: Any, t0: Any) -> None:
    """Raise unless the duration and the step are static, finite numbers."""
    for name, value in (("duration", duration), ("step_size", step_size)):
        if not is_concrete(value) or not math.isfinite(value):
            raise ValidationError(f"'{name}' must be a finite number, got {value}")
    if step_size <= 0.0:
        raise ValidationError(f"'step_size' must be positive, got {step_size}")
    if jnp.shape(t0) != ():
        raise ValidationError(f"'t0' must be a scalar, got shape {jnp.shape(t0)}")


def check_epoch(values: Any, name: str, step_size: float, dtype: Any) -> None:
    """Raise when concrete times are too coarse in the positions' dtype.

    Raises:
        ValidationError: If their spacing is more than a hundredth of a step.
    """
    if not is_concrete(values):
        return
    largest = np.max(np.abs(np.asarray(values, dtype=np.float64)), initial=0.0)
    spacing = float(np.spacing(np.asarray(largest, dtype=dtype)))
    if spacing > step_size / 100.0:
        raise ValidationError(
            f"'{name}' is too far from its epoch: {dtype} resolves {spacing:.3g} s at "
            f"{largest:.6g} s, and 'step_size' is {step_size:g} s; count times from "
            "an epoch near the run"
        )


def _noise_source(
    step: ParticleStep,
    state: ParticleState,
    key: PRNGKeyArray | None,
    noise: Float[Array, "..."] | None,
    n_steps: int,
) -> tuple[Any, Callable[[Any], Float[Array, "..."] | None]]:
    """What to scan over for the noise, and how to read one step's from it.

    Raises:
        ValidationError: If both or neither of ``key`` and ``noise`` are given
            for a step with noise, either is given for a step without, or the
            noise has another shape.
    """
    shape = step.noise_shape(state)
    if shape is None:
        if key is not None or noise is not None:
            raise ValidationError(
                "'key' and 'noise' must both be None for a step without noise"
            )
        return None, lambda draw: None
    dtype = state.position.dtype
    if noise is not None and key is None:
        if noise.shape != (n_steps, *shape):
            raise ValidationError(
                f"'noise' must have one array of shape {shape} per step, "
                f"{(n_steps, *shape)}, got shape {noise.shape}"
            )
        return noise.astype(dtype), lambda draw: draw
    if key is None or noise is not None:
        raise ValidationError(
            "exactly one of 'key' and 'noise' must be given for a step with noise"
        )
    keys = jax.random.split(key, n_steps)
    return keys, lambda draw: jax.random.normal(draw, shape, dtype=dtype)


def _position(particles: ParticleState) -> Any:
    return particles.position


def _moving(particles: ParticleState) -> Any:
    return particles.position, particles.velocity, particles.active


def _scan(adjoint: object) -> Callable[..., Any]:
    """The scan of an adjoint, refusing one the loop does not take."""
    try:
        return scan_for(adjoint)
    except IntegrixValidationError as error:
        raise ValidationError(str(error)) from error


def integrate(
    state: ParticleState,
    step: ParticleStep,
    *,
    t0: ScalarLike,
    duration: float,
    step_size: float,
    key: PRNGKeyArray | None = None,
    noise: Float[Array, "step ..."] | None = None,
    domain: Box | None = None,
    accumulate: Accumulator | None = None,
    save_every: int | None = None,
    adjoint: Checkpointed | Truncated | None = None,
) -> Result:
    """Run a particle step over a duration, forward or backward in time.

    Particle p is released at the offset r_p = d (t_{r,p} − t₀), with d the sign
    of ``duration``. In step k, which covers the offsets s_k to s_k + Δt_k in the
    run's direction, it begins its move at max(s_k, r_p) and moves for

        δ_{p,k} = clip(s_k + Δt_k − max(s_k, r_p), 0, Δt_k),

    so a release inside a step is exact.

    Algorithm:
        1. Split |``duration``| in steps of ``step_size``, the last one shorter.
           A backward run takes them in reverse order, so it undoes the intervals
           of the forward run.
        2. At each step, call ``step`` with start t₀ + d max(s_k, r_p), the
           durations δ (0 for an inactive particle), d and the noise.
        3. Deactivate the particles outside ``domain``, where they are. One that
           a step leaves at NaN, outside the data, keeps its position from before.
           A particle that starts outside never moves.
        4. Add the accumulator's contribution.

    Args:
        state: The particles. A particle is released once the clock has passed
            its release time in the run's direction, so a backward run usually
            takes the time it starts from.
        step: What moves the particles.
        t0: The start of the run, in seconds from its epoch. It may be traced, so
            one trace serves every start time.
        duration: T, the length of the run [s], negative for a backward run.
            Static.
        step_size: Δt > 0, the length of a step [s]. Static.
        key: The PRNG key of the noise, split up front into one key per step. Do
            not re-use the key of the release.
        noise: The noise itself, one array per step, in place of ``key``: for
            antithetic pairs, for common random numbers across runs, or for noise
            that was learned. It costs O(K N).
        domain: The box outside of which a particle becomes inactive. A step
            that leaves a particle at NaN, outside the data, puts it back and
            deactivates it, but the NaN has been computed and a reverse-mode
            gradient through it is NaN: give such a run a field that is defined
            outside the box, such as one clamped at its edge.
        accumulate: What to sum over the run.
        save_every: Save the positions every so many steps. It must divide the
            number of steps.
        adjoint: How to differentiate: `integrix.adjoint.Checkpointed` stores the
            carry every ``every`` steps, `integrix.adjoint.Truncated` cuts the
            gradient before the last ``k``; with ``save_every`` they count blocks
            of it. ``None`` stores every carry.

    Returns:
        The final state, the accumulated total and the saved positions.

    Raises:
        ValidationError: If ``duration`` or ``step_size`` is not a finite number,
            a concrete time is too far from its epoch for the positions' dtype,
            ``save_every`` does not divide the number of steps, not exactly one
            of ``key`` and ``noise`` is given for a step with noise, or
            ``adjoint`` is not one of the three it takes.

    Examples:
        A drifter in a uniform current, released 13 s into the run:

        >>> import jax.numpy as jnp
        >>> from tracerax.advect import Deterministic, integrate
        >>> from tracerax.particles import at_point
        >>> from tracerax.velocity import uniform
        >>> state = at_point(jnp.zeros(2), 1, release_time=13.0)
        >>> step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        >>> result = integrate(state, step, t0=0.0, duration=100.0, step_size=10.0)
        >>> bool(jnp.allclose(result.state.position, jnp.array([[174.0, 0.0]])))
        True

        Its positions every five steps, the input state's first:

        >>> result = integrate(
        ...     state, step, t0=0.0, duration=100.0, step_size=10.0, save_every=5
        ... )
        >>> result.trajectory.shape, result.offsets.tolist()
        ((3, 1, 2), [0.0, 50.0, 100.0])
    """
    _check_run(duration, step_size, t0)
    scan = _scan(adjoint)
    position = state.position
    dtype = position.dtype
    check_epoch(t0, "t0", step_size, dtype)
    check_epoch(state.release_time, "release_time", step_size, dtype)

    # --- the steps, on the host ---
    direction = math.copysign(1.0, duration) if duration != 0.0 else 1.0
    horizon = abs(float(duration))
    n_steps = count_steps(horizon, step_size)
    if save_every is not None:
        check_count(save_every, "save_every")
        if n_steps % save_every:
            raise ValidationError(
                f"'save_every' must divide the number of steps, {n_steps}, "
                f"got {save_every}"
            )
    lengths = step_lengths(horizon, step_size)
    if direction < 0.0:
        # the partial interval of the forward run is undone first
        lengths = lengths[::-1].copy()
    # s_k = Δt_0 + (k − 1) Δt, for k ≥ 1
    begins = np.arange(n_steps, dtype=np.float64) * step_size
    if n_steps:
        begins[1:] += lengths[0] - step_size

    # --- what the scan carries ---
    draws, draw_noise = _noise_source(step, state, key, noise, n_steps)
    start_time = jnp.asarray(t0, dtype=dtype)
    # r_p = d (t_{r,p} − t₀)
    release = direction * (state.release_time.astype(dtype) - start_time)
    if domain is not None:
        inside = state.active & domain.contains(position)
        state = eqx.tree_at(lambda particles: particles.active, state, inside)
    # where a field is read for a particle that does not move
    reference = position[jnp.argmax(state.active)]
    total = None if accumulate is None else jax.tree.map(jnp.asarray, accumulate.init())

    def advance(carry: Any, inputs: Any) -> tuple[Any, None]:
        particles, summed = carry
        offset, length, draw = inputs
        # --- when each particle moves, and for how long ---
        # δ_{p,k} = clip(s_k + Δt_k − max(s_k, r_p), 0, Δt_k), by branches: a tie
        # of the maximum or the clip would split the gradient in r_p, and
        # s_k + Δt_k − s_k is not Δt_k in float32
        waits = release >= offset
        remaining = length - (release - offset)
        partial = jnp.where(remaining > 0.0, remaining, 0.0)
        elapsed = jnp.where(waits, partial, length)
        elapsed = jnp.where(particles.active, elapsed, 0.0)
        moves = elapsed > 0.0
        # max(s_k, r_p)
        begin = jnp.where(waits & moves, release, offset)

        # --- move ---
        before = (particles.position, particles.velocity)
        read = stand_in(particles.position, moves, reference)
        moved = step(
            eqx.tree_at(_position, particles, read),
            start_time + direction * begin,
            elapsed,
            direction,
            draw_noise(draw),
        )
        after = (moved.position, moved.velocity)
        active = moved.active
        if domain is not None:
            # a step that reads outside the data can end on NaN: it did not move
            finite = einx.all("particle [axis]", jnp.isfinite(moved.position))
            elapsed = jnp.where(finite, elapsed, 0.0)
            moves = moves & finite
            active = active & finite
        position, velocity = (
            choose(moves, new, old) for new, old in zip(after, before, strict=True)
        )
        if domain is not None:
            active = active & domain.contains(position)
        moved = eqx.tree_at(_moving, moved, (position, velocity, active))
        if accumulate is not None:
            summed = accumulate.update(summed, moved, elapsed)
        return (moved, summed), None

    inputs = (
        jnp.asarray(begins, dtype=dtype),
        jnp.asarray(lengths, dtype=dtype),
        draws,
    )
    if save_every is None:
        (final, accumulated), _ = scan(advance, (state, total), inputs)
        return Result(final, accumulated, None, None)

    def advance_block(
        carry: Any, block: Any
    ) -> tuple[Any, Float[Array, "particle axis"]]:
        carry, _ = jax.lax.scan(advance, carry, block)
        return carry, carry[0].position

    # (K, …) → (K/m, m, …)
    blocks = jax.tree.map(
        lambda leaf: leaf.reshape(n_steps // save_every, save_every, *leaf.shape[1:]),
        inputs,
    )
    (final, accumulated), saved = scan(advance_block, (state, total), blocks)
    first = einx.id("particle axis -> 1 particle axis", position)
    trajectory = jnp.concatenate([first, saved])
    # the last offset is the whole horizon, without the rounding of a sum
    ends = np.append(begins[save_every::save_every], horizon) if n_steps else []
    offsets = jnp.asarray(direction * np.concatenate([[0.0], ends]), dtype=dtype)
    return Result(final, accumulated, trajectory, offsets)
