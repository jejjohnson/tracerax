"""Tests for the integration loop: releases, noise, sums, saves and adjoints."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous, LinearSigma
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import (
    SolidBodyRotation,
    taylor_variance,
    uniform_wind_position,
)
from _reference import CLOSURES, reference_values
from _sampling import assert_mean, assert_uniform, assert_variance
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Policies
from integrix.adjoint import Checkpointed, Truncated

from tracerax import ValidationError
from tracerax._src.advect.loop import n_steps_for_horizon, step_durations
from tracerax.advect import (
    Accumulator,
    Box,
    Deterministic,
    Langevin,
    Result,
    Walls,
    deterministic,
    integrate,
)
from tracerax.particles import (
    ParticleState,
    along_line,
    at_point,
    in_box,
    release_times,
    with_turbulent_velocity,
)
from tracerax.velocity import GriddedVelocity, cgrid_grids, uniform, uniform_in_space


KEY = jax.random.key(0)
ROTATION = SolidBodyRotation(2.0 * np.pi / 100.0)
WIND = uniform_in_space(lambda t: jnp.array([2.0 + jnp.sin(t / 100.0), 1.0, 0.0]))
STEP = Langevin(WIND, LinearSigma(), Walls(0.0, 50.0))


class Moving(eqx.Module):
    """An accumulator: how long each particle moved for, and its mean position."""

    size: int

    def init(self) -> Any:
        return {"elapsed": jnp.zeros(self.size), "path": jnp.zeros(())}

    def update(self, total: Any, state: ParticleState, duration: jax.Array) -> Any:
        return {
            "elapsed": total["elapsed"] + duration,
            "path": total["path"] + jnp.vdot(state.position[:, 0], duration),
        }


class Residence(eqx.Module):
    """An accumulator: the time spent in each cell, below a height."""

    edges: tuple[jax.Array, ...]
    below: float

    def init(self) -> jax.Array:
        return jnp.zeros([edge.size - 1 for edge in self.edges])

    def update(
        self, total: jax.Array, state: ParticleState, duration: jax.Array
    ) -> jax.Array:
        inside = state.position[:, 2] < self.below
        indices = []
        for column, edge in enumerate(self.edges):
            index = jnp.searchsorted(edge, state.position[:, column], side="right") - 1
            inside = inside & (index >= 0) & (index < edge.size - 1)
            indices.append(jnp.clip(index, 0, edge.size - 2))
        return total.at[tuple(indices)].add(jnp.where(inside, duration, 0.0))


def _cloud(size: int = 6, **options: Any) -> ParticleState:
    lower, upper = jnp.array([0.0, 0.0, 0.0]), jnp.array([10.0, 10.0, 50.0])
    return in_box(lower, upper, size, key=jax.random.key(9), **options)


def _released_at(state: ParticleState, time: Any) -> ParticleState:
    times = jnp.full(state.n_particles, time)
    return eqx.tree_at(lambda particles: particles.release_time, state, times)


def _noise(n_steps: int, size: int, key: jax.Array = KEY) -> jax.Array:
    """The noise the loop draws from the key."""
    keys = jax.random.split(key, n_steps)
    return jax.vmap(lambda one: jax.random.normal(one, (size, 3)))(keys)


class TestTheSteps:
    @pytest.mark.parametrize(
        ("horizon", "step_size", "expected"),
        [
            (2.1, 0.15, 14),
            (122.0, 5.0, 25),
            (120.0, 5.0, 24),
            (1.0, 5.0, 1),
            (1e-12, 5.0, 1),
            (0.0, 5.0, 0),
            (3600.0, 0.1, 36_000),
        ],
    )
    def test_count(self, horizon: float, step_size: float, expected: int) -> None:
        n_steps = n_steps_for_horizon(horizon, step_size)
        assert n_steps == expected
        lengths = step_durations(horizon, step_size, n_steps)
        assert lengths.shape == (expected,)
        assert_matches_oracle(lengths.sum(), horizon, rtol=1e-12)
        assert (lengths > 0.0).all()


class TestAgainstNumPy:
    @staticmethod
    def _wind(stored: dict[str, np.ndarray], kind: str) -> Any:
        constant, slope = jnp.asarray(stored["wind"])
        if kind == "uniform":
            return uniform_in_space(lambda t: constant + t * slope)
        # the same wind on a grid: constant in space and linear in time
        times, axis = jnp.array([0.0, 1000.0]), jnp.array([-1000.0, 1000.0])
        grids = cgrid_grids(times, axis, axis, jnp.array([0.0, 100.0]))
        samples = constant[:, None] + slope[:, None] * times
        fields = [
            jnp.broadcast_to(sample[:, None, None, None], (2, 2, 2, 2))
            for sample in samples
        ]
        return GriddedVelocity.fit(grids, fields)

    @pytest.mark.parametrize("kind", ["uniform", "gridded"])
    @pytest.mark.parametrize("name", CLOSURES)
    def test_a_trajectory(self, name: str, kind: str) -> None:
        stored = reference_values("trajectory")
        expected = stored[f"{name}_trajectory"]
        state = at_point(
            jnp.zeros(3), expected.shape[1], release_time=float(stored["t0"])
        )
        state = eqx.tree_at(
            lambda s: (s.position, s.velocity),
            state,
            (jnp.asarray(expected[0]), jnp.asarray(stored[f"{name}_start_velocity"])),
        )
        step = Langevin(
            self._wind(stored, kind), CLOSURES[name], Walls(0.0, float(stored["lid"]))
        )
        result = integrate(
            state,
            step,
            t0=float(stored["t0"]),
            duration=float(stored["duration"]),
            step_size=float(stored["step_size"]),
            key=jnp.asarray(stored["key"]),
            save_every=1,
        )
        # G1-reference, with positions of order 100 m
        assert_matches_oracle(result.trajectory, expected, rtol=1e-12, atol=1e-11)
        assert_matches_oracle(
            result.state.velocity, stored[f"{name}_velocity"], rtol=1e-12, atol=1e-12
        )

    @staticmethod
    def _release(stored: dict[str, np.ndarray], name: str, where: str, time: float):
        """The release at a point, and the key it leaves for the steps."""
        key, key_velocity = jax.random.split(jax.random.PRNGKey(int(stored["seed"])))
        state = at_point(stored[where], int(stored["size"]), release_time=time)
        return with_turbulent_velocity(state, CLOSURES[name], key=key_velocity), key

    @staticmethod
    def _edges(domain: np.ndarray) -> jax.Array:
        return jnp.linspace(domain[0], domain[1], int(domain[2]) + 1)

    @pytest.mark.parametrize("name", CLOSURES)
    def test_residence(self, name: str) -> None:
        stored = reference_values("residence")
        state, key = self._release(stored, name, "source", 0.0)
        edges = tuple(self._edges(stored[f"domain_{axis}"]) for axis in "xyz")
        result = integrate(
            state,
            Langevin(
                self._wind(stored, "uniform"),
                CLOSURES[name],
                Walls(0.0, float(stored["lid"])),
            ),
            t0=0.0,
            duration=float(stored["duration"]),
            step_size=float(stored["step_size"]),
            key=key,
            accumulate=Residence(edges, jnp.inf),
        )
        volume = np.prod([float(edge[1] - edge[0]) for edge in edges])
        # C = Q H / (N V)
        expected = (
            result.accumulated * stored["emission_rate"] / stored["size"] / volume
        )
        assert_matches_oracle(stored[f"{name}_concentration"], expected, rtol=1e-12)

    @pytest.mark.parametrize("name", CLOSURES)
    def test_footprint_backward(self, name: str) -> None:
        stored = reference_values("footprint")
        time = float(stored["receptor_time"])
        state, key = self._release(stored, name, "receptor", time)
        edges = tuple(self._edges(stored[f"domain_{axis}"]) for axis in "xy")
        depth = float(stored["pbl_fraction"] * stored["pbl_height"])
        result = integrate(
            state,
            Langevin(
                self._wind(stored, "uniform"),
                CLOSURES[name],
                Walls(0.0, float(stored["pbl_height"])),
            ),
            t0=time,
            duration=-float(stored["t_back"]),
            step_size=float(stored["dt"]),
            key=key,
            accumulate=Residence(edges, depth),
        )
        # F = H / (N ρ f_pbl h)
        expected = result.accumulated / stored["size"] / stored["air_density"] / depth
        assert_matches_oracle(stored[f"{name}_footprint"], expected, rtol=1e-12)


class TestReleases:
    @pytest.mark.parametrize("direction", [1.0, -1.0])
    @pytest.mark.parametrize("step_size", [30.0, 7.0])
    def test_a_release_inside_a_step_is_exact(
        self, direction: float, step_size: float
    ) -> None:
        wind = np.array([2.0, -1.0, 0.0])
        waits = np.array([0.0, 13.0, 29.9999, 30.0, 587.0, 600.0, 1e4])
        state = at_point(
            jnp.array([5.0, 5.0, 25.0]), 7, release_time=100.0 + direction * waits
        )
        step = Langevin(uniform(wind), Homogeneous(sigma=0.0))
        result = integrate(
            state,
            step,
            t0=100.0,
            duration=direction * 600.0,
            step_size=step_size,
            key=KEY,
            accumulate=Moving(7),
        )
        start = np.broadcast_to([5.0, 5.0, 25.0], (7, 3))
        expected = uniform_wind_position(start, direction * wind, waits, 600.0)
        assert_matches_oracle(result.state.position, expected, rtol=0.0, atol=1e-9)
        # a particle adds to a sum for as long as it moved
        moved = np.maximum(600.0 - waits, 0.0)
        assert_matches_oracle(
            result.accumulated["elapsed"], moved, rtol=1e-12, atol=1e-12
        )

    def test_a_release_before_the_run_moves_from_its_start(self) -> None:
        state = at_point(jnp.zeros(2), 1, release_time=-50.0)
        step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        result = integrate(state, step, t0=0.0, duration=10.0, step_size=3.0)
        assert_matches_oracle(result.state.position, [[20.0, 0.0]], rtol=1e-14)

    def test_the_velocity_is_read_from_the_release_on(self) -> None:
        """∫ t dt from t_r to T, which RK2 integrates exactly."""
        wind = uniform_in_space(lambda t: jnp.array([t, 0.0]))
        state = at_point(jnp.zeros(2), 2, release_time=jnp.array([3.5, 11.0]))
        result = integrate(
            state, Deterministic(wind), t0=2.0, duration=10.0, step_size=4.0
        )
        expected = 0.5 * (12.0**2 - np.array([3.5, 11.0]) ** 2)
        assert_matches_oracle(result.state.position[:, 0], expected, rtol=1e-13)

    def test_float32_is_exact_near_an_epoch(self) -> None:
        state = at_point(
            jnp.zeros(2, jnp.float32), 1, release_time=jnp.float32(100_013.0)
        )
        step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        options = {"duration": 600.0, "step_size": 30.0}
        result = integrate(state, step, t0=1e5, **options)
        assert result.state.position.dtype == jnp.float32
        # 2 (600 − 13) m
        assert_matches_oracle(result.state.position[0, 0], 1174.0, rtol=0, atol=1e-3)

    @pytest.mark.parametrize("name", ["t0", "release_time"])
    def test_float32_is_refused_far_from_one(self, name: str) -> None:
        times = {"t0": 0.0, "release_time": 0.0} | {name: 1.7e9}
        state = at_point(
            jnp.zeros(2, jnp.float32), 1, release_time=times["release_time"]
        )
        step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        with pytest.raises(
            ValidationError, match=f"'{name}' is too far from its epoch"
        ):
            integrate(state, step, t0=times["t0"], duration=600.0, step_size=30.0)

    @pytest.mark.parametrize(
        ("days", "step_size"), [(1.0, 0.7 * 8), (30.0, 33.3 * 8), (365.0, 330.0)]
    )
    def test_float32_loses_no_time_on_a_long_run(
        self, days: float, step_size: float
    ) -> None:
        """s + Δt − s is not Δt in float32 once s is large."""
        state = at_point(jnp.zeros(2, jnp.float32), 1)
        duration = days * 86_400.0

        class Total(eqx.Module):
            def init(self) -> jax.Array:
                return jnp.zeros(1, jnp.float64)

            def update(self, total: jax.Array, state: Any, duration: jax.Array):
                return total + duration

        result = integrate(
            state,
            Deterministic(uniform(jnp.zeros(2))),
            t0=0.0,
            duration=duration,
            step_size=step_size,
            accumulate=Total(),
        )
        # each step rounds once, to 6e-8 relative
        assert_matches_oracle(result.accumulated[0], duration, rtol=1e-7)

    def test_a_float64_release_time_of_a_float32_state(self) -> None:
        state = at_point(jnp.zeros(2, jnp.float32), 1)
        state = eqx.tree_at(lambda s: s.release_time, state, jnp.array([13.0]))
        step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        result = integrate(state, step, t0=0.0, duration=100.0, step_size=10.0)
        assert result.state.position.dtype == jnp.float32
        assert_matches_oracle(result.state.position[0, 0], 174.0, rtol=1e-6)

    def test_float64_is_not(self) -> None:
        state = at_point(jnp.zeros(2), 1, release_time=1.7e9 + 13.0)
        step = Deterministic(uniform(jnp.array([2.0, 0.0])))
        result = integrate(state, step, t0=1.7e9, duration=600.0, step_size=30.0)
        # the times themselves round to 2e-7 s
        assert_matches_oracle(result.state.position[0, 0], 1174.0, rtol=0, atol=1e-6)


class TestReleaseGradients:
    """A release on a step time is where a maximum and a clip would tie."""

    OPTIONS = {"duration": 10.0, "step_size": 5.0}  # noqa: RUF012
    CUBIC = Deterministic(uniform_in_space(lambda t: jnp.array([t**3, 0.0])))

    @staticmethod
    def _end(step: Any, t0: Any, release: Any, **options: Any) -> jax.Array:
        state = _released_at(at_point(jnp.zeros(2), 1), release)
        return integrate(state, step, t0=t0, **options).state.position[0, 0]

    @pytest.mark.parametrize("release", [3.0, 5.0, 7.0])
    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_in_the_release_time(self, release: float, direction: float) -> None:
        options = {"duration": direction * 10.0, "step_size": 5.0}

        def end(wait: jax.Array) -> jax.Array:
            return self._end(self.CUBIC, 2.0, 2.0 + direction * wait, **options)

        # x = ∫ t³ dt from t_r = 2 + d w to 2 + 10 d, so dx/dw = −d t_r³
        expected = -direction * (2.0 + direction * release) ** 3
        for mode in (jax.grad, jax.jacfwd):
            assert_matches_oracle(
                mode(end)(jnp.asarray(release)), expected, rtol=1e-12, atol=1e-12
            )

    def test_in_the_start_time(self) -> None:
        # released before the run: dx/dt₀ = (t₀ + 10)³ − t₀³
        for release, expected in ((-1.0, 1000.0), (0.0, 1000.0)):
            gradient = jax.grad(
                lambda t0, r=release: self._end(self.CUBIC, t0, r + t0, **self.OPTIONS)
            )(jnp.asarray(0.0))
            assert_matches_oracle(gradient, expected, rtol=1e-12)

    def test_of_a_sum(self) -> None:
        """The time a particle has moved for is T − r."""

        def moved(release: jax.Array) -> jax.Array:
            state = _released_at(at_point(jnp.zeros(2), 1), release)
            result = integrate(
                state, self.CUBIC, t0=0.0, accumulate=Moving(1), **self.OPTIONS
            )
            return result.accumulated["elapsed"][0]

        for release in (2.0, 5.0):
            gradient = jax.grad(moved)(jnp.asarray(release))
            assert_matches_oracle(gradient, -1.0, rtol=1e-14)


class TestDeterministicUnderTheLoop:
    @pytest.mark.parametrize("duration", [42.0, -42.0])
    @pytest.mark.parametrize("dim", [2, 3])
    def test_it_is_the_solve(self, duration: float, dim: int) -> None:
        start, stop = jnp.arange(dim) + 1.0, jnp.arange(dim) + 9.0
        state = along_line(start, stop, 5, release_time=7.0)
        options = {"t0": 7.0, "duration": duration}
        result = integrate(state, Deterministic(ROTATION), step_size=3.0, **options)
        final, _ = deterministic(state, ROTATION, n_steps=14, **options)
        assert_matches_oracle(
            result.state.position, final.position, rtol=1e-12, atol=1e-12
        )

    def test_a_short_last_step_reaches_the_horizon(self) -> None:
        state = along_line(jnp.array([1.0, 0.0]), jnp.array([9.0, 0.0]), 5)
        step = Deterministic(ROTATION)
        result = integrate(state, step, t0=0.0, duration=41.0, step_size=0.5)
        # fourth order: ωT (ωΔt)⁴/120 of 9 m
        expected = ROTATION.exact(state.position, 41.0)
        assert_matches_oracle(result.state.position, expected, rtol=0.0, atol=2e-7)

    def test_forward_and_back(self) -> None:
        """The backward run undoes the short interval first."""
        state = along_line(jnp.array([1.0, 0.0]), jnp.array([9.0, 0.0]), 5)
        step = Deterministic(ROTATION)
        there = integrate(state, step, t0=0.0, duration=41.0, step_size=0.5)
        back = integrate(
            _released_at(there.state, 41.0),
            step,
            t0=41.0,
            duration=-41.0,
            step_size=0.5,
        )
        assert_matches_oracle(back.state.position, state.position, rtol=0.0, atol=1e-7)


class TestDomain:
    def test_a_particle_that_leaves_stops_where_it_is(self) -> None:
        state = along_line(jnp.array([0.0, 0.0]), jnp.array([10.0, 0.0]), 5)
        step = Deterministic(uniform(jnp.array([1.0, 0.0])))
        result = integrate(
            state,
            step,
            t0=0.0,
            duration=6.0,
            step_size=0.5,
            domain=Box([0.0, -1.0], [10.0, 1.0]),
            accumulate=Moving(5),
        )
        # starts at 1, 3, 5, 7, 9; a step that ends at 10 is outside
        assert_matches_oracle(
            result.state.position[:, 0], [7.0, 9.0, 10.0, 10.0, 10.0], rtol=1e-14
        )
        assert_matches_oracle(
            result.state.active, [True, True, False, False, False], rtol=0.0
        )
        assert_matches_oracle(
            result.accumulated["elapsed"], [6.0, 6.0, 5.0, 3.0, 1.0], rtol=1e-14
        )

    @pytest.mark.parametrize("duration", [3.0, 0.0])
    def test_a_particle_that_starts_outside_never_moves(self, duration: float) -> None:
        """It would move back in, in a wind of 2 m/s."""
        state = along_line(jnp.array([-2.0, 0.0]), jnp.array([2.0, 0.0]), 2)
        step, box = Deterministic(uniform(jnp.array([2.0, 0.0]))), Box([0, -1], [9, 1])
        result = integrate(
            state, step, t0=0.0, duration=duration, step_size=1.0, domain=box
        )
        assert_matches_oracle(result.state.active, [False, True], rtol=0.0)
        assert_matches_oracle(
            result.state.position[:, 0], [-1.0, 1.0 + 2.0 * duration], rtol=1e-14
        )
        if duration:
            final, _ = deterministic(
                state, step.velocity, t0=0.0, duration=3.0, n_steps=3, domain=box
            )
            assert_matches_oracle(final.position, result.state.position, rtol=1e-12)
            assert_matches_oracle(final.active, result.state.active, rtol=0.0)

    def test_a_particle_that_reads_outside_the_data_stays_finite(self) -> None:
        nodes = jnp.linspace(0.0, 10.0, 5)
        axes = (Axis(jnp.array([0.0, 9.0])), Axis(nodes), Axis(nodes))
        grid, field = RectilinearGrid(axes), jnp.ones((2, 5, 5))
        velocity = GriddedVelocity.fit(
            (grid, grid), (field, 0.0 * field), policies=Policies(outside="nan")
        )
        state = along_line(jnp.array([2.0, 5.0]), jnp.array([11.0, 5.0]), 3)
        result = integrate(
            state,
            Deterministic(velocity),
            t0=0.0,
            duration=4.0,
            step_size=1.0,
            domain=Box([0.0, 0.0], [10.0, 10.0]),
            accumulate=Moving(3),
        )
        assert bool(jnp.isfinite(result.state.position).all())
        # from 3.5, 6.5 and 9.5: the last would read past 10 in its first step
        assert_matches_oracle(result.state.position[:, 0], [7.5, 9.5, 9.5], rtol=1e-12)
        assert_matches_oracle(result.state.active, [True, False, False], rtol=0.0)
        assert_matches_oracle(
            result.accumulated["elapsed"], [4.0, 3.0, 0.0], rtol=1e-14
        )

    def test_an_inactive_particle_does_not_change_the_others(self) -> None:
        state = _cloud()
        active = jnp.array([True, False, True, True, False, True])
        masked = eqx.tree_at(lambda s: s.active, state, active)
        options = {"t0": 0.0, "duration": 60.0, "step_size": 5.0, "key": KEY}
        every = integrate(state, STEP, accumulate=Moving(6), **options)
        some = integrate(masked, STEP, accumulate=Moving(6), **options)
        expected = jnp.where(active[:, None], every.state.position, state.position)
        assert_matches_oracle(some.state.position, expected, rtol=0.0)
        assert_matches_oracle(
            some.accumulated["elapsed"], jnp.where(active, 60.0, 0.0), rtol=1e-14
        )


class TestNoise:
    OPTIONS = {"t0": 0.0, "duration": 62.0, "step_size": 5.0}  # noqa: RUF012

    def test_noise_passed_is_noise_drawn(self) -> None:
        state = _cloud()
        drawn = integrate(state, STEP, key=KEY, **self.OPTIONS)
        passed = integrate(state, STEP, noise=_noise(13, 6), **self.OPTIONS)
        assert_matches_oracle(passed.state.position, drawn.state.position, rtol=0.0)
        assert_matches_oracle(passed.state.velocity, drawn.state.velocity, rtol=0.0)

    def test_antithetic_pairs_have_no_sampling_error_in_the_mean(self) -> None:
        state = at_point(jnp.array([0.0, 0.0, 0.0]), 6)
        step = Langevin(
            uniform(jnp.array([2.0, 1.0, 0.0])), Homogeneous(), Walls(-1e6, 1e6)
        )
        noise = _noise(13, 6)
        plus = integrate(state, step, noise=noise, **self.OPTIONS).state.position
        minus = integrate(state, step, noise=-noise, **self.OPTIONS).state.position
        assert float(jnp.abs(plus - minus).max()) > 1.0
        expected = np.broadcast_to([124.0, 62.0, 0.0], (6, 3))
        # rounding, with the walls 1e6 m away
        assert_matches_oracle(0.5 * (plus + minus), expected, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize(
        ("step", "options", "match"),
        [
            (STEP, {}, "exactly one"),
            (STEP, {"key": KEY, "noise": jnp.zeros((13, 6, 3))}, "exactly one"),
            (STEP, {"noise": jnp.zeros((12, 6, 3))}, "'noise' must have one array"),
            (STEP, {"noise": jnp.zeros((13, 6, 2))}, "'noise' must have one array"),
            (Deterministic(WIND), {"key": KEY}, "must both be None"),
            (
                Deterministic(WIND),
                {"noise": jnp.zeros((13, 6, 3))},
                "must both be None",
            ),
        ],
    )
    def test_invalid_noise(self, step: Any, options: Any, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            integrate(_cloud(), step, **self.OPTIONS, **options)

    def test_an_ensemble_is_a_vmap_over_keys(self) -> None:
        state = _cloud()
        keys = jax.random.split(KEY, 3)

        def run(key: jax.Array) -> jax.Array:
            return integrate(state, STEP, key=key, **self.OPTIONS).state.position

        members = jax.vmap(run)(keys)
        assert_matches_oracle(members[1], run(keys[1]), rtol=1e-12)
        assert float(jnp.abs(members[0] - members[1]).max()) > 0.1


class TestSaves:
    @pytest.mark.parametrize("duration", [60.0, -60.0])
    def test_each_saved_position_is_a_run_stopped_there(self, duration: float) -> None:
        state = _cloud()
        noise = _noise(12, 6)
        options = {"t0": 0.0, "step_size": 5.0}
        result = integrate(
            state, STEP, duration=duration, noise=noise, save_every=3, **options
        )
        sign = np.sign(duration)
        assert_matches_oracle(
            result.offsets, sign * np.array([0.0, 15.0, 30.0, 45.0, 60.0]), rtol=1e-15
        )
        assert_matches_oracle(result.trajectory[0], state.position, rtol=0.0)
        assert_matches_oracle(result.trajectory[-1], result.state.position, rtol=0.0)
        for index in (1, 3):
            stopped = integrate(
                state,
                STEP,
                duration=sign * 15.0 * index,
                noise=noise[: 3 * index],
                **options,
            )
            assert_matches_oracle(
                result.trajectory[index], stopped.state.position, rtol=0.0
            )

    def test_the_saves_do_not_change_the_run(self) -> None:
        options = {"t0": 0.0, "duration": 60.0, "step_size": 5.0, "key": KEY}
        plain = integrate(_cloud(), STEP, accumulate=Moving(6), **options)
        saved = integrate(_cloud(), STEP, accumulate=Moving(6), save_every=4, **options)
        assert_matches_oracle(saved.state.position, plain.state.position, rtol=0.0)
        assert_matches_oracle(
            saved.accumulated["path"], plain.accumulated["path"], rtol=0.0
        )

    def test_the_last_offset_is_the_horizon(self) -> None:
        step = Deterministic(WIND)
        result = integrate(
            _cloud(), step, t0=0.0, duration=-57.0, step_size=5.0, save_every=6
        )
        # backward, the short step of 2 s is the first
        assert_matches_oracle(result.offsets, [0.0, -27.0, -57.0], rtol=0.0)

    def test_a_run_of_no_duration_is_the_input_state(self) -> None:
        state = _cloud()
        result = integrate(
            state,
            STEP,
            t0=0.0,
            duration=0.0,
            step_size=5.0,
            key=KEY,
            save_every=3,
            accumulate=Moving(6),
        )
        assert_matches_oracle(result.state.position, state.position, rtol=0.0)
        assert result.trajectory.shape == (1, 6, 3)
        assert result.offsets.tolist() == [0.0]
        assert_matches_oracle(result.accumulated["elapsed"], np.zeros(6), rtol=0.0)

    @pytest.mark.parametrize("save_every", [5, 0, -1, 1.5])
    def test_it_divides_the_steps(self, save_every: Any) -> None:
        with pytest.raises((ValidationError, TypeError), match="save_every"):
            integrate(
                _cloud(),
                STEP,
                t0=0.0,
                duration=60.0,
                step_size=5.0,
                key=KEY,
                save_every=save_every,
            )


class TestAdjoints:
    OPTIONS = {"t0": 0.0, "duration": 60.0, "step_size": 5.0, "key": KEY}  # noqa: RUF012

    def _loss(self, sigma: jax.Array, **options: Any) -> jax.Array:
        step = Langevin(WIND, LinearSigma(sigma_ground=sigma), Walls(0.0, 50.0))
        result = integrate(
            _cloud(), step, accumulate=Moving(6), **self.OPTIONS, **options
        )
        return jnp.var(result.state.position[:, 2]) + 1e-3 * result.accumulated["path"]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "options",
        [
            {"adjoint": Checkpointed(every=5)},
            {"save_every": 4},
            {"save_every": 4, "adjoint": Checkpointed(every=2)},
            {"adjoint": Truncated(k=12)},
        ],
    )
    def test_values_and_gradients_are_the_plain_ones(self, options: Any) -> None:
        sigma = jnp.asarray(0.2)
        value, gradient = jax.value_and_grad(self._loss)(sigma)
        other, other_gradient = jax.value_and_grad(lambda s: self._loss(s, **options))(
            sigma
        )
        assert_matches_oracle(other, value, rtol=1e-14)
        assert_matches_oracle(other_gradient, gradient, rtol=1e-11)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "options", [{}, {"adjoint": Checkpointed(every=5)}, {"save_every": 4}]
    )
    def test_forward_mode_is_reverse_mode(self, options: Any) -> None:
        sigma = jnp.asarray(0.2)
        gradient = jax.grad(lambda s: self._loss(s, **options))(sigma)
        _, tangent = jax.jvp(
            lambda s: self._loss(s, **options), (sigma,), (jnp.ones(()),)
        )
        assert_matches_oracle(tangent, gradient, rtol=1e-11)

    @pytest.mark.slow
    def test_gradient_against_differences(self) -> None:
        assert_grad_matches_fd(self._loss, jnp.asarray(0.2), rtol=1e-5)

    def test_truncation_cuts_the_gradient(self) -> None:
        state = along_line(jnp.array([1.0, 0.0]), jnp.array([9.0, 0.0]), 3)

        def end(speed: jax.Array, adjoint: Any) -> jax.Array:
            step = Deterministic(lambda t, x: jnp.full_like(x, speed))
            result = integrate(
                state, step, t0=0.0, duration=10.0, step_size=1.0, adjoint=adjoint
            )
            return result.state.position[0, 0]

        # d x(T)/d speed is T, and 4 of the 10 steps
        for adjoint, expected in ((None, 10.0), (Truncated(k=4), 4.0)):
            assert_matches_oracle(
                jax.grad(end)(jnp.asarray(1.0), adjoint), expected, rtol=1e-13
            )

    def test_gradient_with_particles_that_wait(self) -> None:
        state = _cloud(6, release_time=jnp.array([0.0, 13.0, 30.0, 59.0, 60.0, 99.0]))

        def loss(sigma: jax.Array) -> jax.Array:
            step = Langevin(WIND, LinearSigma(sigma_ground=sigma), Walls(0.0, 50.0))
            return jnp.sum(integrate(state, step, **self.OPTIONS).state.position ** 2)

        gradient = jax.grad(loss)(jnp.asarray(0.2))
        assert bool(jnp.isfinite(gradient))
        assert_grad_matches_fd(loss, jnp.asarray(0.2), rtol=1e-5)


class TestTracing:
    def test_fifty_start_times_trace_once(self) -> None:
        """G1-trace."""
        traces = 0
        state = _cloud()

        @jax.jit
        def run(t0: jax.Array) -> jax.Array:
            nonlocal traces
            traces += 1
            result = integrate(
                _released_at(state, t0),
                STEP,
                t0=t0,
                duration=-62.0,
                step_size=5.0,
                key=KEY,
            )
            return result.state.position

        ends = [np.asarray(run(jnp.asarray(10.0 * k))) for k in range(50)]
        assert traces == 1
        # the wind is read on the clock of the start time
        assert np.abs(ends[0] - ends[30]).max() > 0.1

    def test_transforms(self) -> None:
        state = _cloud()

        def run(position: jax.Array, t0: jax.Array) -> Any:
            moved = eqx.tree_at(lambda s: s.position, state, position)
            moved = _released_at(moved, t0)
            result = integrate(
                moved,
                STEP,
                t0=t0,
                duration=13.0,
                step_size=5.0,
                key=KEY,
                accumulate=Moving(6),
                save_every=1,
            )
            return result.state.position, result.trajectory, result.accumulated

        positions = 50.0 * jax.random.uniform(KEY, (3, 6, 3))
        assert_traces(run, positions, jnp.array([0.0, 50.0, 99.0]), rtol=1e-10)

    def test_float32(self) -> None:
        state = in_box(
            jnp.zeros(3, jnp.float32), jnp.full(3, 50.0, jnp.float32), 4, key=KEY
        )
        result = integrate(
            state, STEP, t0=0.0, duration=20.0, step_size=5.0, key=KEY, save_every=2
        )
        for leaf in (result.state.position, result.trajectory, result.offsets):
            assert leaf.dtype == jnp.float32

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"duration": float("inf")}, "'duration'"),
            ({"step_size": 0.0}, "'step_size'"),
            ({"step_size": -1.0}, "'step_size'"),
            ({"t0": jnp.zeros(2)}, "'t0'"),
        ],
    )
    def test_invalid_arguments(self, options: Any, match: str) -> None:
        arguments = {"t0": 0.0, "duration": 10.0, "step_size": 5.0} | options
        with pytest.raises((ValidationError, TypeError), match=match):
            integrate(_cloud(), Deterministic(WIND), **arguments)

    def test_the_protocol_of_an_accumulator(self) -> None:
        assert isinstance(Moving(3), Accumulator)
        assert not isinstance(object(), Accumulator)

    def test_the_result_without_options(self) -> None:
        result = integrate(
            _cloud(), Deterministic(WIND), t0=0.0, duration=10.0, step_size=5.0
        )
        assert isinstance(result, Result)
        assert (result.accumulated, result.trajectory, result.offsets) == (None,) * 3


@pytest.mark.slow
class TestStatistics:
    def test_the_displacement_variance_is_taylors(self) -> None:
        """G1-taylor, at Δt = τ/12."""
        sigma, timescale, duration = 0.5, 60.0, 600.0
        closure = Homogeneous(sigma=sigma, timescale=timescale)
        key_start, key_run = jax.random.split(KEY)
        state = with_turbulent_velocity(
            at_point(jnp.zeros(3), 50_000), closure, key=key_start
        )
        step = Langevin(uniform(jnp.zeros(3)), closure, Walls(-1e6, 1e6))
        result = integrate(
            state, step, t0=0.0, duration=duration, step_size=5.0, key=key_run
        )
        expected = float(taylor_variance(sigma, timescale, duration))
        for column in range(3):
            assert_variance(result.state.position[:, column], expected)
            assert_mean(result.state.position[:, column], 0.0)

    @pytest.mark.parametrize("duration", [1800.0, -1800.0])
    def test_a_uniform_column_stays_uniform(self, duration: float) -> None:
        """G1-mixed, under the loop and with releases spread over the first minute."""
        key_start, key_run = jax.random.split(KEY)
        times = np.sign(duration) * np.asarray(release_times(0.0, 60.0, 20_000))
        state = _cloud(20_000, release_time=jnp.asarray(times))
        state = with_turbulent_velocity(state, LinearSigma(), key=key_start)
        result = integrate(
            state, STEP, t0=0.0, duration=duration, step_size=5.0, key=key_run
        )
        assert_uniform(result.state.position[:, 2], 0.0, 50.0)
