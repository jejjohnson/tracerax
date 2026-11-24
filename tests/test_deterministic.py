"""Tests for deterministic advection: the solve, the step and the domain."""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import equinox as eqx
import integrix.march
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import DoubleGyre, SolidBodyRotation, uniform_wind_position

from tracerax import ValidationError
from tracerax.advect import Box, Deterministic, deterministic
from tracerax.particles import ParticleState, ParticleStep, along_line, at_point
from tracerax.velocity import uniform, uniform_in_space


KEY = jax.random.key(0)
RATE = 2.0 * np.pi / 100.0
ROTATION = SolidBodyRotation(RATE)
SOLVERS = {"rk2": integrix.march.RK2Heun(), "rk4": integrix.march.RK4Classic()}


def _line(dim: int = 2, **options: Any) -> ParticleState:
    start, stop = jnp.arange(dim) + 0.5, jnp.arange(dim) - 2.0
    return along_line(start, stop, 5, **options)


def _released_at(state: ParticleState, time: Any) -> ParticleState:
    times = jnp.full(state.n_particles, time)
    return eqx.tree_at(lambda particles: particles.release_time, state, times)


def _shear(t: Any, position: jax.Array) -> jax.Array:
    """A velocity that is smooth in time and space, and not linear in either."""
    phase = 0.3 * position + 0.1 * jnp.reshape(t, (-1, 1))
    return jnp.sin(phase) + 0.5 * jnp.roll(position, 1, axis=-1) * jnp.cos(phase)


def _by_hand(
    step: Any, state: ParticleState, t0: float, duration: float, n_steps: int
) -> ParticleState:
    direction = float(np.sign(duration))
    length = abs(duration) / n_steps
    durations = jnp.full(state.n_particles, length)
    for k in range(n_steps):
        start = jnp.full(state.n_particles, t0 + direction * k * length)
        state = step(state, start, durations, direction, None)
    return state


class TestBox:
    def test_the_lower_corner_is_inside_and_the_upper_is_not(self) -> None:
        box = Box([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        position = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.0, 3.0]])
        assert_matches_oracle(box.contains(position), [True, False, False], rtol=0.0)

    def test_transforms(self) -> None:
        box = Box([0.0, 0.0], [1.0, 2.0])
        assert_traces(lambda x: box.contains(x), jax.random.normal(KEY, (4, 5, 2)))

    def test_invalid_corners(self) -> None:
        with pytest.raises(ValidationError, match="'lower' must be below"):
            Box([0.0, 2.0], [1.0, 1.0])
        with pytest.raises(ValidationError, match="same dimension"):
            Box([0.0, 2.0], [1.0, 3.0, 4.0])
        with pytest.raises(ValidationError, match="2 or 3 columns"):
            Box([0.0], [1.0])


class TestDeterministic:
    """The solve."""

    @pytest.mark.parametrize("n_steps", [50, 100, 200])
    def test_solid_body_rotation_returns_with_rk4s_error(self, n_steps: int) -> None:
        state = along_line(jnp.array([1.0, 0.0]), jnp.array([3.0, 0.0]), 4)
        final, saved = deterministic(
            state, ROTATION, t0=0.0, duration=100.0, n_steps=n_steps
        )
        assert saved is None
        error = np.linalg.norm(np.asarray(final.position - state.position), axis=1)
        radius = np.linalg.norm(np.asarray(state.position), axis=1)
        # RK4 on ẋ = iωx errs by (ωΔt)⁵/120 a step, so 2π (2π/n)⁴/120 a turn:
        # 1.3e-5, 8.1e-7 and 5.1e-8, to O((2π/n)²)
        expected = 2.0 * np.pi * (2.0 * np.pi / n_steps) ** 4 / 120.0
        assert_matches_oracle(error / radius, np.full(4, expected), rtol=0.02)

    def test_it_is_diffeqsolve(self) -> None:
        state = _line()
        final, _ = deterministic(state, ROTATION, t0=7.0, duration=30.0, n_steps=20)
        reference = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: ROTATION(t, y)),
            SOLVERS["rk4"],
            0.0,
            30.0,
            1.5,
            state.position,
        ).ys[0]
        # rounding of 20 steps
        assert_matches_oracle(final.position, reference, rtol=1e-13)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_forward_and_back(self, dim: int) -> None:
        state = _line(dim)
        final, _ = deterministic(state, _shear, t0=5.0, duration=3.0, n_steps=100)
        final = _released_at(final, 8.0)
        back, _ = deterministic(final, _shear, t0=8.0, duration=-3.0, n_steps=100)
        assert float(jnp.abs(final.position - state.position).max()) > 0.1
        assert_matches_oracle(back.position, state.position, rtol=0.0, atol=1e-6)

    def test_an_order_four_solve_converges(self) -> None:
        state = _line()

        def end(n_steps: int) -> np.ndarray:
            final, _ = deterministic(
                state, _shear, t0=1.0, duration=4.0, n_steps=n_steps
            )
            return np.asarray(final.position)

        fine = end(1280)
        errors = [np.abs(end(n) - fine).max() for n in (40, 80, 160)]
        # a factor of 16 per halving, to O(Δt)
        assert_matches_oracle(errors[0] / errors[1], 16.0, rtol=0.2)
        assert_matches_oracle(errors[1] / errors[2], 16.0, rtol=0.2)

    @pytest.mark.parametrize("duration", [4.0, -4.0])
    def test_a_saved_position_is_a_run_stopped_there(self, duration: float) -> None:
        state = _released_at(_line(), 1.0)
        save_at = duration * jnp.array([0.0, 0.25, 0.5, 1.0])
        final, saved = deterministic(
            state, _shear, t0=1.0, duration=duration, n_steps=8, save_at=save_at
        )
        assert saved.shape == (4, *state.position.shape)
        assert_matches_oracle(saved[0], state.position, rtol=1e-15)
        assert_matches_oracle(saved[-1], final.position, rtol=1e-14)
        for index, n_steps in ((1, 2), (2, 4)):
            stopped, _ = deterministic(
                state,
                _shear,
                t0=1.0,
                duration=float(save_at[index]),
                n_steps=n_steps,
            )
            assert_matches_oracle(saved[index], stopped.position, rtol=1e-13)

    @pytest.mark.slow
    @pytest.mark.parametrize("domain", [None, Box([-9.0, -9.0], [9.0, 9.0])])
    def test_forward_mode_is_reverse_mode(self, domain: Box | None) -> None:
        state = _line()

        def flow(position: jax.Array, adjoint: Any) -> jax.Array:
            moved = eqx.tree_at(lambda s: s.position, state, position)
            final, _ = deterministic(
                moved,
                _shear,
                t0=1.0,
                duration=2.0,
                n_steps=20,
                domain=domain,
                adjoint=adjoint,
            )
            return final.position

        forward = jax.jacfwd(flow)(state.position, dfx.ForwardMode())
        reverse = jax.jacrev(flow)(state.position, dfx.RecursiveCheckpointAdjoint())
        assert_matches_oracle(forward, reverse, rtol=0.0, atol=1e-10)

    @pytest.mark.slow
    def test_gradient_in_the_field_and_the_start(self) -> None:
        state = _line()

        def end(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
            position, speed = inputs
            moved = eqx.tree_at(lambda s: s.position, state, position)
            final, _ = deterministic(
                moved,
                lambda t, x: speed * _shear(t, x),
                t0=1.0,
                duration=2.0,
                n_steps=10,
            )
            return final.position

        assert_grad_matches_fd(end, (state.position, jnp.asarray(0.7)))

    @pytest.mark.slow
    def test_the_double_gyre_keeps_area(self) -> None:
        points = jnp.array([[0.5, 0.3], [1.1, 0.6], [1.7, 0.2], [0.2, 0.9]])

        def flow(point: jax.Array) -> jax.Array:
            state = at_point(jnp.zeros(2), 1)
            state = eqx.tree_at(lambda s: s.position, state, point[None])
            final, _ = deterministic(
                state,
                DoubleGyre(),
                t0=0.0,
                duration=15.0,
                n_steps=150,
                adjoint=dfx.ForwardMode(),
            )
            return final.position[0]

        determinant = jnp.linalg.det(jax.vmap(jax.jacfwd(flow))(points))
        # 3.7e-6 was measured on the design's points
        assert_matches_oracle(determinant, np.ones(4), rtol=1e-5)

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_a_release_on_a_step_waits_for_it(self, direction: float) -> None:
        wind = np.array([2.0, -1.0])
        release = 100.0 + direction * np.array([0.0, 4.0, 10.0, 20.0])
        state = at_point(jnp.zeros(2), 4, release_time=release)
        final, _ = deterministic(
            state, uniform(wind), t0=100.0, duration=direction * 10.0, n_steps=5
        )
        expected = uniform_wind_position(
            np.zeros((4, 2)), direction * wind, np.array([0.0, 4.0, 10.0, 20.0]), 10.0
        )
        assert_matches_oracle(final.position, expected, rtol=1e-13, atol=1e-13)

    def test_a_backward_run_waits_for_a_release_in_its_past(self) -> None:
        """The clock of a backward run has not come down to 0 yet."""
        state = at_point(jnp.zeros(2), 1, release_time=0.0)
        final, _ = deterministic(
            state, uniform(jnp.ones(2)), t0=10.0, duration=-5.0, n_steps=5
        )
        assert_matches_oracle(final.position, np.zeros((1, 2)), rtol=0.0)

    def test_a_release_at_a_steps_end_does_not_reach_its_last_stage(self) -> None:
        """RK4's last stage is evaluated at the end of the step."""
        state = at_point(jnp.zeros(2), 1, release_time=2.0)
        final, _ = deterministic(
            state, uniform(jnp.array([1.0, 0.0])), t0=0.0, duration=2.0, n_steps=1
        )
        assert_matches_oracle(final.position, np.zeros((1, 2)), rtol=0.0)

    def test_a_release_inside_a_step_is_refused(self) -> None:
        state = at_point(jnp.zeros(2), 1, release_time=3.0)
        with pytest.raises(ValidationError, match="'release_time' must lie"):
            deterministic(state, ROTATION, t0=0.0, duration=10.0, n_steps=5)

    def test_a_particle_that_leaves_stops_and_is_inactive(self) -> None:
        box = Box([-1.0, -1.0], [5.0, 5.0])
        state = at_point(jnp.zeros(2), 2)
        state = eqx.tree_at(
            lambda s: s.position, state, jnp.array([[0.0, 0.0], [4.0, 0.0]])
        )
        final, _ = deterministic(
            state,
            uniform(jnp.array([0.3, 0.0])),
            t0=0.0,
            duration=10.0,
            n_steps=100,
            domain=box,
        )
        assert_matches_oracle(final.active, [True, False], rtol=0.0)
        assert_matches_oracle(final.position[0], [3.0, 0.0], rtol=1e-12)
        # crossed at first order: within one step of 0.03 m
        assert 5.0 <= float(final.position[1, 0]) <= 5.03

    def test_an_inactive_particle_stays_and_is_finite(self) -> None:
        state = eqx.tree_at(
            lambda s: s.active, _line(), jnp.array([True, False, True, True, False])
        )
        moved, _ = deterministic(_line(), _shear, t0=0.0, duration=2.0, n_steps=4)
        final, _ = deterministic(state, _shear, t0=0.0, duration=2.0, n_steps=4)
        expected = jnp.where(state.active[:, None], moved.position, state.position)
        assert_matches_oracle(final.position, expected, rtol=0.0)

    def test_fifty_start_times_trace_once(self) -> None:
        traces = 0
        state = _line()

        @jax.jit
        def run(t0: jax.Array) -> jax.Array:
            nonlocal traces
            traces += 1
            moved = _released_at(state, t0)
            final, _ = deterministic(moved, _shear, t0=t0, duration=2.0, n_steps=4)
            return final.position

        ends = [np.asarray(run(jnp.asarray(float(k)))) for k in range(50)]
        assert traces == 1
        # the start time is the time the velocity is read at
        assert np.abs(ends[0] - ends[30]).max() > 1e-3

    def test_transforms(self) -> None:
        state = _line()

        def run(t0: jax.Array, position: jax.Array) -> jax.Array:
            moved = eqx.tree_at(lambda s: s.position, state, position)
            final, _ = deterministic(moved, _shear, t0=t0, duration=1.0, n_steps=3)
            return final.position

        times, positions = jnp.arange(3.0), jax.random.normal(KEY, (3, 5, 2))
        # a traced t0 skips the release check, and nothing else
        state = _released_at(state, -1.0)
        assert_traces(run, times, positions)

    def test_float32(self) -> None:
        state = along_line(jnp.zeros(2, jnp.float32), jnp.ones(2, jnp.float32), 3)
        final, saved = deterministic(
            state,
            uniform(jnp.ones(2)),
            t0=0.0,
            duration=10.0,
            n_steps=4,
            save_at=jnp.array([10.0]),
        )
        assert final.position.dtype == saved.dtype == jnp.float32

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"duration": 0.0}, "'duration'"),
            ({"duration": float("inf")}, "'duration'"),
            ({"n_steps": 0}, "'n_steps'"),
            ({"t0": jnp.zeros(2)}, "'t0'"),
            ({"save_at": jnp.array([-1.0])}, "'save_at'"),
            ({"save_at": jnp.array([5.0, 1.0])}, "'save_at'"),
            ({"save_at": jnp.array([11.0])}, "'save_at'"),
        ],
    )
    def test_invalid_arguments(self, options: dict[str, Any], match: str) -> None:
        arguments = {"t0": 0.0, "duration": 10.0, "n_steps": 4} | options
        with pytest.raises((ValidationError, TypeError), match=match):
            deterministic(_line(), ROTATION, **arguments)


class TestDeterministicStep:
    def test_it_is_a_particle_step(self) -> None:
        step = Deterministic(ROTATION)
        assert isinstance(step, ParticleStep)
        assert step.noise_shape(_line()) is None

    @pytest.mark.parametrize("solver", SOLVERS.values(), ids=SOLVERS)
    @pytest.mark.parametrize("duration", [3.0, -3.0])
    @pytest.mark.parametrize("dim", [2, 3])
    def test_by_hand_it_is_the_solve(
        self, solver: Any, duration: float, dim: int
    ) -> None:
        state = _released_at(_line(dim), 5.0)
        run = _by_hand(Deterministic(_shear, solver), state, 5.0, duration, 6)
        final, _ = deterministic(
            state, _shear, t0=5.0, duration=duration, n_steps=6, solver=solver
        )
        assert_matches_oracle(run.position, final.position, rtol=1e-12, atol=1e-13)

    def test_durations_per_particle_are_separate_runs(self) -> None:
        state = _line()
        step = Deterministic(_shear)
        start = jnp.arange(5.0)
        durations = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])
        moved = step(state, start, durations, -1.0, None)
        for p in range(5):
            alone = jax.tree.map(lambda leaf, p=p: leaf[p : p + 1], state)
            single = step(alone, start[p : p + 1], durations[p : p + 1], -1.0, None)
            assert_matches_oracle(
                moved.position[p], single.position[0], rtol=1e-14, atol=1e-15
            )

    def test_the_wind_is_read_at_the_stage_times(self) -> None:
        """RK2 integrates a wind linear in time exactly, RK4 a cubic one."""
        step = Deterministic(uniform_in_space(lambda t: jnp.array([t**3, 1.0])))
        state = at_point(jnp.zeros(2), 2)
        moved = step(state, jnp.array([1.0, 4.0]), jnp.array([2.0, 3.0]), -1.0, None)
        # ∫ t³ dt from 1 to −1 is 0, and from 4 to 1 is −255/4
        expected = np.array([[0.0, -2.0], [-63.75, -3.0]])
        assert_matches_oracle(moved.position, expected, rtol=1e-14, atol=1e-14)

    def test_a_zero_duration_is_bit_for_bit(self) -> None:
        state = _line()
        moved = Deterministic(_shear)(state, jnp.zeros(5), jnp.zeros(5), 1.0, None)
        assert_matches_oracle(moved.position, state.position, rtol=0.0)

    def test_an_inactive_particle_stays(self) -> None:
        state = eqx.tree_at(lambda s: s.active, _line(), jnp.zeros(5, dtype=bool))
        moved = Deterministic(_shear)(state, jnp.zeros(5), jnp.ones(5), 1.0, None)
        assert_matches_oracle(moved.position, state.position, rtol=0.0)

    def test_transforms_and_gradient(self) -> None:
        state = _line()

        def move(position: jax.Array, duration: jax.Array) -> jax.Array:
            moved = eqx.tree_at(lambda s: s.position, state, position)
            step = Deterministic(_shear)
            return step(moved, jnp.zeros(5), duration, 1.0, None).position

        positions = jax.random.normal(KEY, (3, 5, 2))
        durations = 0.5 + jax.random.uniform(KEY, (3, 5))
        assert_traces(move, positions, durations)
        assert_grad_matches_fd(
            lambda inputs: move(*inputs), (positions[0], durations[0])
        )
        forward = jax.jacfwd(move)(positions[0], durations[0])
        reverse = jax.jacrev(move)(positions[0], durations[0])
        assert_matches_oracle(forward, reverse, rtol=0.0, atol=1e-13)

    def test_float32(self) -> None:
        state = along_line(jnp.zeros(2, jnp.float32), jnp.ones(2, jnp.float32), 3)
        zeros = jnp.zeros(3, jnp.float32)
        moved = Deterministic(uniform(jnp.ones(2)))(state, zeros, zeros + 1, 1.0, None)
        assert moved.position.dtype == jnp.float32

    def test_an_implicit_solver_is_refused(self) -> None:
        with pytest.raises((ValidationError, TypeError), match="solver"):
            Deterministic(ROTATION, dfx.ImplicitEuler())
