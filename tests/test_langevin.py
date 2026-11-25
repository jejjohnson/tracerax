"""Tests for the Langevin step, the turbulence protocol and the walls."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous, LinearSigma
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _reference import CLOSURES, reference_values
from _sampling import assert_uniform, assert_variance
from hypothesis import given, strategies as st

from tracerax import ValidationError
from tracerax.advect import Langevin, TurbulenceModel, Walls, langevin_step
from tracerax.particles import (
    ParticleState,
    ParticleStep,
    at_point,
    in_box,
    with_turbulent_velocity,
)
from tracerax.velocity import uniform, uniform_in_space


KEY = jax.random.key(0)
LID = 50.0
LAYER = Walls(bottom=0.0, top=LID)


def _layer_state(size: int, closure: Any = LinearSigma()) -> ParticleState:  # noqa: B008
    """Particles uniform in the layer, with the closure's velocities."""
    key_position, key_velocity = jax.random.split(jax.random.key(7))
    lower, upper = jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 10.0, LID])
    state = in_box(lower, upper, size, key=key_position)
    return with_turbulent_velocity(state, closure, key=key_velocity)


def _run(
    state: ParticleState, closure: Any, step_size: float, n_steps: int, walls: Walls
) -> ParticleState:
    """A run in a calm mean wind, without the loop."""
    calm = jnp.zeros_like(state.position)

    def body(carry: ParticleState, key: jax.Array) -> tuple[ParticleState, jax.Array]:
        noise = jax.random.normal(key, calm.shape)
        moved = langevin_step(
            carry, calm, closure, jnp.array([step_size]), noise, walls=walls
        )
        height = moved.position[:, 2]
        bottom, top = walls.bottom, walls.top
        return moved, jnp.sum((height < bottom) | (height > top))

    keys = jax.random.split(jax.random.key(3), n_steps)
    final, outside = jax.jit(lambda start: jax.lax.scan(body, start, keys))(state)
    assert int(outside.sum()) == 0
    return final


class TestTurbulenceModel:
    @pytest.mark.parametrize("closure", CLOSURES.values(), ids=CLOSURES)
    def test_the_test_closures_satisfy_it(self, closure: Any) -> None:
        assert isinstance(closure, TurbulenceModel)

    def test_something_else_does_not(self) -> None:
        assert not isinstance(object(), TurbulenceModel)


class TestWalls:
    # XLA compares a subnormal height as 0
    @given(
        height=st.floats(-1e4, 1e4, allow_subnormal=False),
        bottom=st.floats(-100.0, 100.0, allow_subnormal=False),
        depth=st.floats(1.0, 1e3),
    )
    def test_the_fold_is_the_triangle_wave(
        self, height: float, bottom: float, depth: float
    ) -> None:
        top = bottom + depth
        walls = Walls(bottom=bottom, top=top)
        folded, vertical = walls.reflect(jnp.array([height]), jnp.array([1.0]))
        assert bottom <= float(folded[0]) <= top
        # a second way to fold: by the distance from the nearest multiple of 2h
        depth = top - bottom
        wrapped = np.mod(height - bottom, 2.0 * depth)
        expected = bottom + min(wrapped, 2.0 * depth - wrapped)
        # rounding of heights as large as the largest
        scale = abs(height) + abs(bottom) + depth
        assert abs(float(folded[0]) - expected) <= 1e-14 * scale
        # in doubt only within rounding of a wall
        margin = min(wrapped % depth, depth - wrapped % depth)
        if margin > 1e-12 * scale:
            turned = wrapped > depth
            assert float(vertical[0]) == (-1.0 if turned else 1.0)

    def test_the_ground_alone(self) -> None:
        walls = Walls(bottom=2.0)
        height, vertical = walls.reflect(
            jnp.array([-1.0, 2.0, 1e6]), jnp.array([-3.0, 1.0, 2.0])
        )
        assert_matches_oracle(height, [5.0, 2.0, 1e6], rtol=0.0)
        assert_matches_oracle(vertical, [3.0, 1.0, 2.0], rtol=0.0)

    @pytest.mark.parametrize("walls", [Walls(0.1, 12.7), Walls(0.1)])
    def test_a_particle_inside_is_bit_for_bit(self, walls: Walls) -> None:
        height = jnp.concatenate(
            [jnp.array([0.1, 12.7]), 0.1 + 12.6 * jax.random.uniform(KEY, (50,))]
        )
        folded, vertical = walls.reflect(height, jnp.ones(52))
        assert_matches_oracle(folded, height, rtol=0.0)
        assert_matches_oracle(vertical, np.ones(52), rtol=0.0)

    def test_transforms_and_gradient(self) -> None:
        heights = jnp.array([[-70.0, 12.0, 51.0], [260.0, -1.0, 49.0]])
        assert_traces(lambda z: LAYER.reflect(z, jnp.ones(3)), heights)
        assert_grad_matches_fd(lambda z: LAYER.reflect(z, 2.0 * z), heights[0])

    def test_float32(self) -> None:
        heights = jnp.array([-7e5, 12.0, 3e6], jnp.float32)
        folded, vertical = LAYER.reflect(heights, heights)
        assert folded.dtype == vertical.dtype == jnp.float32
        assert bool(((folded >= 0.0) & (folded <= LID)).all())

    def test_the_top_is_above_the_bottom(self) -> None:
        with pytest.raises(ValidationError, match="'top' must be above"):
            Walls(bottom=5.0, top=5.0)

    @pytest.mark.parametrize(
        ("walls", "name"),
        [
            ((0.0, np.inf), "top"),
            ((-np.inf, 10.0), "bottom"),
            ((np.nan, None), "bottom"),
        ],
    )
    def test_the_walls_are_finite(self, walls: tuple[Any, Any], name: str) -> None:
        with pytest.raises(ValidationError, match=f"'{name}' must be finite"):
            Walls(*walls)


class TestLangevinStep:
    @pytest.mark.parametrize("name", CLOSURES)
    def test_it_is_the_step_in_numpy(self, name: str) -> None:
        stored = reference_values("langevin")
        size = stored["noise"].shape[0]
        state = at_point(jnp.zeros(3), size)
        state = eqx.tree_at(
            lambda s: (s.position, s.velocity),
            state,
            (
                jnp.asarray(stored[f"{name}_start_position"]),
                jnp.asarray(stored[f"{name}_start_velocity"]),
            ),
        )
        # the loop draws the noise so: the same key gives the same numbers
        noise = jax.random.normal(jnp.asarray(stored["key"]), (size, 3))
        moved = langevin_step(
            state,
            jnp.broadcast_to(stored["wind"], (size, 3)),
            CLOSURES[name],
            jnp.asarray(stored["step_size"])[None],
            noise,
            walls=Walls(bottom=0.0, top=float(stored["lid"])),
        )
        # G1-reference, with positions of order 50 m
        for field in ("position", "velocity"):
            assert_matches_oracle(
                getattr(moved, field),
                stored[f"{name}_{field}"],
                rtol=1e-12,
                atol=1e-12,
            )

    def test_durations_per_particle_are_separate_steps(self) -> None:
        state = _layer_state(5)
        durations = jnp.array([0.0, 0.5, 5.0, 20.0, 100.0])
        noise = jax.random.normal(KEY, (5, 3))
        wind = jnp.ones((5, 3))
        moved = langevin_step(state, wind, LinearSigma(), durations, noise, walls=LAYER)
        for p in range(5):
            alone = jax.tree.map(lambda leaf, p=p: leaf[p : p + 1], state)
            single = langevin_step(
                alone,
                wind[:1],
                LinearSigma(),
                durations[p : p + 1],
                noise[p : p + 1],
                walls=LAYER,
            )
            assert_matches_oracle(moved.position[p], single.position[0], rtol=1e-14)
            assert_matches_oracle(moved.velocity[p], single.velocity[0], rtol=1e-13)

    def test_a_zero_duration_is_bit_for_bit(self) -> None:
        # outside the walls too: a particle that waits is not reflected
        state = _layer_state(5)
        state = eqx.tree_at(lambda s: s.position, state, state.position - 30.0)
        moved = langevin_step(
            state,
            jnp.ones((5, 3)),
            LinearSigma(),
            jnp.zeros(5),
            jnp.ones((5, 3)),
            walls=LAYER,
        )
        assert_matches_oracle(moved.position, state.position, rtol=0.0)
        assert_matches_oracle(moved.velocity, state.velocity, rtol=0.0)

    def test_an_inactive_particle_stays_and_the_others_do_not_change(self) -> None:
        state = _layer_state(5)
        noise = jax.random.normal(KEY, (5, 3))
        active = jnp.array([True, False, True, True, False])
        masked = eqx.tree_at(lambda s: s.active, state, active)
        # an inactive particle may be anywhere, in a wind that is not a number
        wind = jnp.where(active[:, None], jnp.ones((5, 3)), jnp.nan)

        def step(particles: ParticleState, mean: jax.Array) -> ParticleState:
            return langevin_step(
                particles, mean, LinearSigma(), jnp.array([5.0]), noise, walls=LAYER
            )

        all_moved, some_moved = step(state, jnp.ones((5, 3))), step(masked, wind)
        for field in ("position", "velocity"):
            expected = jnp.where(
                active[:, None], getattr(all_moved, field), getattr(state, field)
            )
            assert_matches_oracle(getattr(some_moved, field), expected, rtol=0.0)

    def test_gradients_of_a_particle_that_waits_are_finite(self) -> None:
        state = _layer_state(3)
        noise = jax.random.normal(KEY, (3, 3))

        def loss(sigma: jax.Array, durations: jax.Array) -> jax.Array:
            closure = LinearSigma(sigma_ground=sigma)
            moved = langevin_step(
                state, jnp.ones((3, 3)), closure, durations, noise, walls=LAYER
            )
            return jnp.sum(moved.position**2)

        durations = jnp.array([0.0, 5.0, 0.0])
        for gradient in jax.grad(loss, argnums=(0, 1))(jnp.asarray(0.2), durations):
            assert bool(jnp.isfinite(gradient).all())

    @pytest.mark.parametrize("closure", CLOSURES.values(), ids=CLOSURES)
    def test_gradient_in_sigma_and_the_time_scale(self, closure: Any) -> None:
        state = _layer_state(4)
        noise = jax.random.normal(KEY, (4, 3))

        def move(scales: tuple[jax.Array, jax.Array]) -> Any:
            sigma, timescale = scales
            name = "sigma" if isinstance(closure, Homogeneous) else "sigma_ground"
            changed = eqx.tree_at(
                lambda c: (getattr(c, name), c.timescale), closure, (sigma, timescale)
            )
            moved = langevin_step(
                state, jnp.ones((4, 3)), changed, jnp.array([5.0]), noise, walls=LAYER
            )
            return moved.position, moved.velocity

        assert_grad_matches_fd(move, (jnp.asarray(0.3), jnp.asarray(50.0)), rtol=1e-6)

    def test_transforms(self) -> None:
        state = _layer_state(4)

        def move(position: jax.Array, noise: jax.Array, duration: jax.Array) -> Any:
            particles = eqx.tree_at(lambda s: s.position, state, position)
            moved = langevin_step(
                particles, noise, LinearSigma(), duration, noise, walls=LAYER
            )
            return moved.position, moved.velocity

        positions = LID * jax.random.uniform(KEY, (3, 4, 3))
        noises = jax.random.normal(KEY, (3, 4, 3))
        durations = 10.0 * jax.random.uniform(KEY, (3, 4))
        # the batched and the plain exponential round differently
        assert_traces(move, positions, noises, durations, rtol=1e-11)

    def test_a_calm_closure_gives_the_mean_velocity(self) -> None:
        state = at_point(jnp.array([1.0, 2.0, 3.0]), 3)
        wind = jnp.array([[2.0, -1.0, 0.5]] * 3)
        moved = langevin_step(
            state, wind, Homogeneous(sigma=0.0), jnp.array([4.0]), jnp.ones((3, 3))
        )
        # σ is floored at 1e-12 m/s
        expected = np.broadcast_to([9.0, -2.0, 5.0], (3, 3))
        assert_matches_oracle(moved.position, expected, rtol=0.0, atol=1e-10)

    def test_the_gradient_of_a_calm_closure_is_finite(self) -> None:
        state = at_point(jnp.array([1.0, 2.0, 3.0]), 3)

        def loss(sigma: jax.Array) -> jax.Array:
            moved = langevin_step(
                state,
                jnp.ones((3, 3)),
                LinearSigma(sigma_ground=sigma, slope=0.0),
                jnp.array([4.0]),
                jnp.ones((3, 3)),
            )
            return jnp.sum(moved.position)

        assert bool(jnp.isfinite(jax.grad(loss)(0.0)))

    def test_float32(self) -> None:
        point = jnp.array([0.0, 0.0, 10.0], jnp.float32)
        state = at_point(point, 4)
        noise = jnp.ones((4, 3), jnp.float32)
        moved = langevin_step(
            state, noise, LinearSigma(), jnp.array([5.0], jnp.float32), noise
        )
        assert moved.position.dtype == moved.velocity.dtype == jnp.float32

    def test_two_dimensions_are_refused(self) -> None:
        with pytest.raises((ValidationError, TypeError), match=r"3|noise"):
            langevin_step(
                at_point(jnp.zeros(2), 2),
                jnp.zeros((2, 3)),
                Homogeneous(),
                jnp.ones(1),
                jnp.zeros((2, 3)),
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("ratio", [0.1, 1.0, 10.0])
    def test_no_particle_is_outside_its_walls(self, ratio: float) -> None:
        """G1-inside: a single reflection per wall leaves 6.8 % outside at 0.3 τ."""
        closure = Homogeneous(sigma=1.5, timescale=100.0)
        state = _layer_state(5_000, closure)
        # _run counts the particles outside after every step
        final = _run(state, closure, 100.0 * ratio, 40, LAYER)
        # the fold keeps a uniform column uniform at any step size
        assert_uniform(final.position[:, 2], 0.0, LID)

    @pytest.mark.slow
    def test_a_uniform_column_stays_uniform(self) -> None:
        """G1-mixed: 30 min, about two mixing times of the layer."""
        state = _layer_state(20_000)
        final = _run(state, LinearSigma(), 5.0, 360, LAYER)
        assert_uniform(final.position[:, 2], 0.0, LID)

    @pytest.mark.slow
    def test_without_the_drift_it_does_not(self) -> None:
        class NoSlope(LinearSigma):
            def at(self, height: jax.Array) -> Any:
                return super().at(jax.lax.stop_gradient(height))

        final = _run(_layer_state(20_000), NoSlope(), 5.0, 360, LAYER)
        with pytest.raises(AssertionError, match="not uniform"):
            assert_uniform(final.position[:, 2], 0.0, LID)

    @pytest.mark.slow
    def test_the_velocity_variance_has_no_step_size_bias(self) -> None:
        closure = Homogeneous(sigma=0.5, timescale=60.0)
        state = with_turbulent_velocity(
            at_point(jnp.array([0.0, 0.0, 25.0]), 20_000), closure, key=KEY
        )
        final = _run(state, closure, 120.0, 5, LAYER)
        for column in range(3):
            assert_variance(final.velocity[:, column], 0.25)


class TestLangevin:
    WIND = staticmethod(uniform_in_space(lambda t: jnp.array([t, 2.0 * t, 0.0])))

    def test_it_is_a_particle_step(self) -> None:
        step = Langevin(self.WIND, Homogeneous(), LAYER)
        assert isinstance(step, ParticleStep)
        assert step.noise_shape(_layer_state(4)) == (4, 3)

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_the_wind_is_read_at_the_physical_start(self, direction: float) -> None:
        """Forward at start, backward at start − δ, where the wind is −u."""
        state = _layer_state(3)
        start = jnp.array([10.0, 20.0, 30.0])
        durations = jnp.array([1.0, 2.0, 5.0])
        noise = jax.random.normal(KEY, (3, 3))
        step = Langevin(self.WIND, LinearSigma(), LAYER)
        moved = step(state, start, durations, direction, noise)
        time = start if direction > 0.0 else start - durations
        mean = direction * jnp.stack([time, 2.0 * time, 0.0 * time], axis=-1)
        expected = langevin_step(
            state, mean, LinearSigma(), durations, noise, walls=LAYER
        )
        assert_matches_oracle(moved.position, expected.position, rtol=1e-14)
        assert_matches_oracle(moved.velocity, expected.velocity, rtol=1e-14)

    def test_the_default_wall_is_the_ground(self) -> None:
        step = Langevin(uniform(jnp.array([0.0, 0.0, -5.0])), Homogeneous(sigma=0.0))
        state = at_point(jnp.array([0.0, 0.0, 2.0]), 1)
        moved = step(state, jnp.zeros(1), jnp.ones(1), 1.0, jnp.zeros((1, 3)))
        # 2 − 5 = −3, reflected; the tiny velocity turns
        assert_matches_oracle(moved.position[0, 2], 3.0, rtol=1e-12)

    def test_noise_is_required(self) -> None:
        step = Langevin(self.WIND, Homogeneous())
        with pytest.raises(ValidationError, match="'noise'"):
            step(_layer_state(2), jnp.zeros(2), jnp.ones(2), 1.0, None)

    def test_two_dimensions_have_no_noise_shape(self) -> None:
        with pytest.raises(ValidationError, match="3 position columns"):
            Langevin(self.WIND, Homogeneous()).noise_shape(at_point(jnp.zeros(2), 2))
