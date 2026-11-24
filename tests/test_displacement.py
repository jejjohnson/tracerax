"""Tests for random displacement: the diffusivities, the step and its drift."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _sampling import assert_mean, assert_uniform, assert_variance
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Cubic

from tracerax import ValidationError
from tracerax.advect import (
    Box,
    ConstantDiffusivity,
    Deterministic,
    Diffusivity,
    GriddedDiffusivity,
    RandomDisplacement,
    Walls,
    displacement_step,
    integrate,
)
from tracerax.particles import (
    ParticleState,
    ParticleStep,
    at_point,
    in_box,
)
from tracerax.velocity import uniform, uniform_in_space


KEY = jax.random.key(0)
DEPTH = 50.0
COLUMN = Walls(0.0, DEPTH)
CALM = uniform(jnp.zeros(3))


class Ocean(eqx.Module):
    """K_z = 0.02 + 0.24 s (1 − s) e^{−2s}, s = z/h, with K′ ≠ 0 at both walls.

    K is between 0.020 and 0.048 m² s⁻¹, and |K″| ≤ 5.8e-4 s⁻¹.
    """

    scale: Any = 1.0

    def at(self, t: Any, position: jax.Array) -> jax.Array:
        s = position[:, 2] / DEPTH
        vertical = self.scale * (0.02 + 0.24 * s * (1.0 - s) * jnp.exp(-2.0 * s))
        return jnp.stack([0.0 * s, 0.0 * s, vertical], axis=-1)


class Wave(eqx.Module):
    """K_x = K₀ + K₁ sin(2πx/L), periodic, and no diffusion in y."""

    mean: float
    amplitude: float
    length: float

    def at(self, t: Any, position: jax.Array) -> jax.Array:
        phase = 2.0 * jnp.pi * position[:, 0] / self.length
        along = self.mean + self.amplitude * jnp.sin(phase)
        return jnp.stack([along, 0.0 * along], axis=-1)


class NoDrift(eqx.Module):
    """A diffusivity whose slope the step cannot see."""

    diffusivity: Any

    def at(self, t: Any, position: jax.Array) -> jax.Array:
        return self.diffusivity.at(t, jax.lax.stop_gradient(position))


class Inside(eqx.Module):
    """An accumulator: the time each particle spends in a periodic interval."""

    size: int
    low: float
    high: float
    length: float

    def init(self) -> jax.Array:
        return jnp.zeros(self.size)

    def update(
        self, total: jax.Array, state: ParticleState, duration: jax.Array
    ) -> jax.Array:
        x = jnp.mod(state.position[:, 0], self.length)
        return total + jnp.where((x >= self.low) & (x < self.high), duration, 0.0)


def _cloud(size: int = 5, dim: int = 3) -> ParticleState:
    lower, upper = jnp.zeros(dim), jnp.full(dim, DEPTH)
    return in_box(lower, upper, size, key=jax.random.key(9))


def _step(state: ParticleState, diffusivity: Any, **options: Any) -> ParticleState:
    size, shape = state.n_particles, state.position.shape
    defaults = {
        "mean_velocity": jnp.ones(shape),
        "t": jnp.zeros(size),
        "duration": jnp.full(size, 30.0),
        "noise": jax.random.normal(KEY, shape),
    }
    return displacement_step(state, diffusivity=diffusivity, **(defaults | options))


class TestDiffusivities:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_constant(self, dim: int) -> None:
        value = np.arange(dim) + 0.5
        diffusivity = ConstantDiffusivity(value)
        assert isinstance(diffusivity, Diffusivity)
        expected = np.broadcast_to(value, (4, dim))
        position = jax.random.normal(KEY, (4, dim))
        assert_matches_oracle(diffusivity.at(0.0, position), expected, rtol=0.0)
        small = position.astype(jnp.float32)
        assert diffusivity.at(0.0, small).dtype == jnp.float32

    @pytest.mark.parametrize(
        ("value", "match"), [([-1.0, 0.0], "non-negative"), ([1.0], "2 or 3")]
    )
    def test_an_invalid_constant(self, value: Any, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            ConstantDiffusivity(value)

    def test_something_else_is_not_one(self) -> None:
        assert not isinstance(object(), Diffusivity)

    @staticmethod
    def _gridded(scheme: Any = None) -> GriddedDiffusivity:
        """K_x = 1 + 0.2 x, K_y = 2 + 0.1 y, K_z = 3 + 0.3 z on three grids."""
        options = {} if scheme is None else {"scheme": scheme}
        grids, fields = [], []
        for column, (start, slope) in enumerate(((1, 0.2), (2, 0.1), (3, 0.3))):
            nodes = jnp.linspace(0.0, DEPTH, 6 + column)
            axes = [Axis(jnp.array([0.0, 10.0]))] + [Axis(nodes)] * 3
            grid = RectilinearGrid(axes)
            mesh = jnp.meshgrid(*[axis.values for axis in axes], indexing="ij")
            grids.append(grid)
            fields.append(start + slope * mesh[-1 - column])
        return GriddedDiffusivity.fit(grids, fields, **options)

    def test_a_gridded_linear_field_has_its_slope_as_the_drift(self) -> None:
        state = _cloud(20)
        diffusivity = self._gridded()
        assert isinstance(diffusivity, Diffusivity)
        still = jnp.zeros_like(state.position)
        moved = _step(state, diffusivity, mean_velocity=still, noise=still)
        drift = (moved.position - state.position) / 30.0
        expected = np.broadcast_to([0.2, 0.1, 0.3], (20, 3))
        # rounding of a 50 m position over a move of a few metres
        assert_matches_oracle(drift, expected, rtol=0.0, atol=1e-12)

    def test_a_gridded_field_is_read_at_the_particles(self) -> None:
        state = _cloud(20)
        values = self._gridded().at(3.0, state.position)
        expected = jnp.array([1.0, 2.0, 3.0]) + state.position * jnp.array(
            [0.2, 0.1, 0.3]
        )
        assert_matches_oracle(values, expected, rtol=1e-13)

    def test_a_cubic_read_below_zero_is_clipped(self) -> None:
        """Cubic overshoots next to a jump from 0 to 1."""
        nodes = jnp.arange(8.0)
        axes = (Axis(jnp.zeros(1)), Axis(nodes), Axis(nodes))
        grid = RectilinearGrid(axes)
        field = jnp.broadcast_to(jnp.where(nodes > 3.5, 1.0, 0.0), (1, 8, 8))
        diffusivity = GriddedDiffusivity.fit(
            (grid, grid), (field, field), scheme=Cubic()
        )
        state = at_point(jnp.array([2.5, 4.0]), 3)
        values = diffusivity.at(0.0, state.position)
        assert float(values[0, 0]) < -0.01
        still = jnp.zeros_like(state.position)
        moved = _step(
            state, diffusivity, mean_velocity=still, noise=jnp.ones_like(still)
        )
        assert_matches_oracle(moved.position[:, 0], np.full(3, 2.5), rtol=0.0)


class TestDisplacementStep:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_without_noise_and_drift_it_is_an_euler_step(self, dim: int) -> None:
        state = _cloud(5, dim)
        wind = jax.random.normal(KEY, state.position.shape)
        moved = _step(
            state,
            ConstantDiffusivity(jnp.full(dim, 2.0)),
            mean_velocity=wind,
            noise=jnp.zeros_like(wind),
        )
        assert_matches_oracle(
            moved.position, state.position + 30.0 * wind, rtol=1e-15, atol=1e-13
        )

    def test_it_is_the_ito_step(self) -> None:
        state = _cloud(50)
        noise = jax.random.normal(KEY, state.position.shape)
        moved = _step(state, Ocean(), noise=noise)
        z = np.asarray(state.position[:, 2])
        s = z / DEPTH
        value = 0.02 + 0.24 * s * (1.0 - s) * np.exp(-2.0 * s)
        # K′ = 0.24 e^{−2s} (1 − 4s + 2s²) / h
        slope = 0.24 * np.exp(-2.0 * s) * (1.0 - 4.0 * s + 2.0 * s**2) / DEPTH
        expected = z + (1.0 + slope) * 30.0
        expected = expected + np.sqrt(2.0 * value * 30.0) * np.asarray(noise[:, 2])
        assert_matches_oracle(moved.position[:, 2], expected, rtol=1e-13)
        assert_matches_oracle(
            moved.position[:, :2], state.position[:, :2] + 30.0, rtol=1e-14
        )
        assert_matches_oracle(moved.velocity, state.velocity, rtol=0.0)

    def test_the_walls_reflect_the_third_column(self) -> None:
        state = _cloud()
        wind = jnp.zeros_like(state.position).at[:, 2].set(-10.0)
        moved = _step(
            state,
            ConstantDiffusivity(jnp.zeros(3)),
            mean_velocity=wind,
            walls=COLUMN,
        )
        # down 300 m in a 50 m column: three periods of the fold
        assert_matches_oracle(moved.position, state.position, rtol=0.0, atol=1e-12)

    def test_walls_in_two_dimensions_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="third position column"):
            _step(_cloud(5, 2), ConstantDiffusivity([1.0, 1.0]), walls=COLUMN)

    def test_a_zero_duration_is_bit_for_bit(self) -> None:
        state = _cloud()
        moved = _step(state, Ocean(), duration=jnp.zeros(5), walls=Walls(20.0, 30.0))
        assert_matches_oracle(moved.position, state.position, rtol=0.0)

    def test_an_inactive_particle_stays_and_the_others_do_not_change(self) -> None:
        state = _cloud()
        active = jnp.array([True, False, True, True, False])
        masked = eqx.tree_at(lambda s: s.active, state, active)
        wind = jnp.where(active[:, None], jnp.ones((5, 3)), jnp.nan)
        every, some = _step(state, Ocean()), _step(masked, Ocean(), mean_velocity=wind)
        expected = jnp.where(active[:, None], every.position, state.position)
        assert_matches_oracle(some.position, expected, rtol=0.0)

    def test_durations_per_particle_are_separate_steps(self) -> None:
        state = _cloud()
        durations = jnp.array([0.0, 1.0, 5.0, 30.0, 120.0])
        noise = jax.random.normal(KEY, (5, 3))
        moved = _step(state, Ocean(), duration=durations, noise=noise)
        for p in range(5):
            alone = jax.tree.map(lambda leaf, p=p: leaf[p : p + 1], state)
            single = _step(
                alone,
                Ocean(),
                duration=durations[p : p + 1],
                noise=noise[p : p + 1],
            )
            assert_matches_oracle(moved.position[p], single.position[0], rtol=1e-14)

    @pytest.mark.parametrize("duration", [30.0, 0.0])
    def test_no_diffusivity_leaves_the_euler_path_with_finite_gradients(
        self, duration: float
    ) -> None:
        state = _cloud()

        def move(value: jax.Array, position: jax.Array) -> jax.Array:
            particles = eqx.tree_at(lambda s: s.position, state, position)
            diffusivity = ConstantDiffusivity(value)
            durations = jnp.full(5, duration)
            return _step(particles, diffusivity, duration=durations).position

        value = jnp.zeros(3)
        expected = state.position + duration
        assert_matches_oracle(move(value, state.position), expected, rtol=1e-15)
        for gradient in jax.jacrev(move, argnums=(0, 1))(value, state.position):
            assert bool(jnp.isfinite(gradient).all())

    def test_gradient_in_the_diffusivity(self) -> None:
        state = _cloud()
        assert_grad_matches_fd(
            lambda scale: _step(state, Ocean(scale)).position,
            jnp.asarray(1.3),
            rtol=1e-6,
        )
        assert_grad_matches_fd(
            lambda value: _step(state, ConstantDiffusivity(value)).position,
            jnp.array([0.5, 1.0, 2.0]),
            rtol=1e-6,
        )

    def test_gradient_in_the_position(self) -> None:
        state = _cloud()

        def move(position: jax.Array) -> jax.Array:
            particles = eqx.tree_at(lambda s: s.position, state, position)
            return _step(particles, Ocean()).position

        assert_grad_matches_fd(move, state.position, rtol=1e-6)

    def test_transforms(self) -> None:
        state = _cloud()

        def move(position: jax.Array, noise: jax.Array, duration: jax.Array) -> Any:
            particles = eqx.tree_at(lambda s: s.position, state, position)
            return _step(
                particles, Ocean(), noise=noise, duration=duration, walls=COLUMN
            ).position

        positions = DEPTH * jax.random.uniform(KEY, (3, 5, 3))
        noises = jax.random.normal(KEY, (3, 5, 3))
        durations = 10.0 * jax.random.uniform(KEY, (3, 5))
        # the batched and the plain exponential round differently
        assert_traces(move, positions, noises, durations, rtol=1e-11)

    def test_float32(self) -> None:
        state = at_point(jnp.ones(3, jnp.float32), 4)
        ones = jnp.ones((4, 3), jnp.float32)
        moved = _step(
            state,
            ConstantDiffusivity([1.0, 1.0, 1.0]),
            mean_velocity=ones,
            noise=ones,
            t=ones[:, 0],
            duration=ones[:, 0],
        )
        assert moved.position.dtype == jnp.float32


class TestRandomDisplacement:
    def test_it_is_a_particle_step(self) -> None:
        step = RandomDisplacement(CALM, Ocean())
        assert isinstance(step, ParticleStep)
        assert step.noise_shape(_cloud(4, 2)) == (4, 2)

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_the_fields_are_read_at_the_physical_start(self, direction: float) -> None:
        """Backward u changes sign and the drift does not."""

        class Ramp(eqx.Module):
            def at(self, t: Any, position: jax.Array) -> jax.Array:
                return (1.0 + t)[:, None] * (1.0 + 0.1 * position)

        state = _cloud()
        wind = uniform_in_space(lambda t: jnp.array([t, 2.0 * t, 0.0]))
        start, durations = 10.0 * jnp.arange(5.0) + 10.0, jnp.arange(5.0)
        still = jnp.zeros((5, 3))
        step = RandomDisplacement(wind, Ramp())
        moved = step(state, start, durations, direction, still)
        time = start if direction > 0.0 else start - durations
        mean = direction * np.stack([time, 2.0 * time, 0.0 * time], axis=-1)
        # ∂K/∂x = 0.1 (1 + t)
        drift = 0.1 * (1.0 + np.asarray(time))[:, None]
        expected = state.position + (mean + drift) * np.asarray(durations)[:, None]
        assert_matches_oracle(moved.position, expected, rtol=1e-13)

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("duration", [60.0, -60.0])
    def test_it_runs_under_the_loop(self, dim: int, duration: float) -> None:
        """Without diffusion it is the Euler step that the loop runs."""
        state = _cloud(5, dim)
        wind = uniform(jnp.arange(dim) + 1.0)
        options = {"t0": 0.0, "duration": duration, "step_size": 7.0}
        still = RandomDisplacement(wind, ConstantDiffusivity(jnp.zeros(dim)))
        result = integrate(state, still, key=KEY, **options)
        expected = integrate(state, Deterministic(wind), **options)
        assert_matches_oracle(
            result.state.position, expected.state.position, rtol=1e-13, atol=1e-12
        )
        mixed = RandomDisplacement(wind, ConstantDiffusivity(jnp.ones(dim)))
        moved = integrate(
            state,
            mixed,
            key=KEY,
            domain=Box(jnp.full(dim, -1e6), jnp.full(dim, 1e6)),
            **options,
        )
        assert bool(moved.state.active.all())
        assert float(jnp.abs(moved.state.position - result.state.position).min()) > 0

    def test_noise_is_required(self) -> None:
        with pytest.raises(ValidationError, match="'noise'"):
            RandomDisplacement(CALM, Ocean())(
                _cloud(), jnp.zeros(5), jnp.ones(5), 1.0, None
            )


@pytest.mark.slow
class TestStatistics:
    def test_the_variance_is_2kt_on_each_axis(self) -> None:
        value, duration = np.array([10.0, 3.0, 0.1]), 600.0
        step = RandomDisplacement(
            uniform(jnp.array([1.0, 2.0, 3.0])), ConstantDiffusivity(value)
        )
        result = integrate(
            at_point(jnp.zeros(3), 50_000),
            step,
            t0=0.0,
            duration=duration,
            step_size=60.0,
            key=KEY,
        )
        for column in range(3):
            assert_variance(
                result.state.position[:, column], 2.0 * value[column] * duration
            )
            assert_mean(result.state.position[:, column], (column + 1.0) * duration)

    @pytest.mark.parametrize("with_drift", [True, False])
    def test_a_column_between_walls_stays_uniform(self, with_drift: bool) -> None:
        """G1-diffusion: 6 h at Δt = 30 s, where 0.1 / max|K″| is 172 s."""
        diffusivity = Ocean() if with_drift else NoDrift(Ocean())
        key_start, key_run = jax.random.split(KEY)
        state = in_box(jnp.zeros(3), jnp.full(3, DEPTH), 20_000, key=key_start)
        step = RandomDisplacement(CALM, diffusivity, COLUMN)
        result = integrate(
            state, step, t0=0.0, duration=21_600.0, step_size=30.0, key=key_run
        )
        height = result.state.position[:, 2]
        if with_drift:
            assert_uniform(height, 0.0, DEPTH)
            return
        with pytest.raises(AssertionError, match="not uniform"):
            assert_uniform(height, 0.0, DEPTH)

    @pytest.mark.parametrize("with_drift", [True, False])
    def test_a_periodic_cloud_stays_uniform(self, with_drift: bool) -> None:
        """G1-diffusion: K₀t/L² = 0.3 at K₀Δt/L² = 2e-4; the rule allows 3e-3."""
        wave = Wave(mean=1.0, amplitude=0.8, length=1.0)
        diffusivity = wave if with_drift else NoDrift(wave)
        key_start, key_run = jax.random.split(KEY)
        state = in_box(jnp.zeros(2), jnp.ones(2), 20_000, key=key_start)
        step = RandomDisplacement(uniform(jnp.zeros(2)), diffusivity)
        result = integrate(
            state, step, t0=0.0, duration=0.3, step_size=2e-4, key=key_run
        )
        x = jnp.mod(result.state.position[:, 0], 1.0)
        if with_drift:
            assert_uniform(x, 0.0, 1.0)
            return
        with pytest.raises(AssertionError, match="not uniform"):
            assert_uniform(x, 0.0, 1.0)

    def test_reciprocity(self) -> None:
        """The time spent in B from A is the time spent in A, backward, from B."""
        length, size = 1000.0, 40_000
        step = RandomDisplacement(
            uniform(jnp.array([0.1, 0.0])), Wave(mean=5.0, amplitude=4.0, length=length)
        )
        box_a, box_b = (100.0, 150.0), (500.0, 550.0)

        def residence(
            source: tuple, receptor: tuple, duration: float, seed: int
        ) -> jax.Array:
            key_start, key_run = jax.random.split(jax.random.key(seed))
            lower, upper = jnp.array([source[0], 0.0]), jnp.array([source[1], 1.0])
            result = integrate(
                in_box(lower, upper, size, key=key_start),
                step,
                t0=0.0,
                duration=duration,
                step_size=5.0,
                key=key_run,
                accumulate=Inside(size, *receptor, length),
            )
            return np.asarray(result.accumulated)

        forward = residence(box_a, box_b, 3600.0, 3)
        backward = residence(box_b, box_a, -3600.0, 4)
        # s² = s_f²/N + s_b²/N
        error = np.sqrt(forward.var() / size + backward.var() / size)
        assert abs(forward.mean() - backward.mean()) <= 4.0 * error
        # 182 s ± 1.2 s was measured on this case
        assert abs(forward.mean() - 182.0) <= 4.0 * 1.2
