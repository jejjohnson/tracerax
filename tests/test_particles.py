"""Tests for `tracerax.particles`: the state, the step protocol and the seeders."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous, LinearSigma
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _reference import reference_values
from _sampling import assert_mean, assert_uniform
from hypothesis import given, settings, strategies as st

from tracerax import ValidationError
from tracerax._src.particles.seed import _corner
from tracerax.particles import (
    ParticleState,
    ParticleStep,
    along_line,
    at_point,
    in_box,
    release_times,
    with_turbulent_velocity,
)


SOURCE = jnp.array([10.0, -5.0, 2.0])
KEY = jax.random.key(0)


class Drift(eqx.Module):
    """A toy step: x ← x + d c δ."""

    speed: float

    def noise_shape(self, state: ParticleState) -> None:
        return None

    def __call__(self, state, start, duration, direction, noise):
        moved = state.position.at[:, 0].add(direction * self.speed * duration)
        return eqx.tree_at(lambda particles: particles.position, state, moved)


class TestParticleState:
    def test_it_is_a_pytree_of_five_arrays(self) -> None:
        state = at_point(SOURCE, 4)
        assert len(jax.tree.leaves(state)) == 5
        assert state.n_particles == 4

    def test_it_passes_through_jit_and_scan(self) -> None:
        step = Drift(2.0)

        @jax.jit
        def run(state: ParticleState) -> ParticleState:
            def body(carry: ParticleState, duration: jax.Array):
                start = jnp.zeros(carry.n_particles)
                durations = jnp.full(carry.n_particles, duration)
                return step(carry, start, durations, 1.0, None), None

            final, _ = jax.lax.scan(body, state, jnp.array([1.0, 2.0, 3.0]))
            return final

        final = run(at_point(SOURCE, 4))
        # 10 + 2 (1 + 2 + 3)
        assert_matches_oracle(final.position[:, 0], np.full(4, 22.0), rtol=1e-15)

    def test_it_passes_through_vmap(self) -> None:
        sources = jnp.stack([SOURCE, 2.0 * SOURCE])
        states = jax.vmap(lambda source: at_point(source, 4))(sources)
        assert states.position.shape == (2, 4, 3)
        moved = jax.vmap(lambda state: state.position + 1.0)(states)
        assert moved.shape == (2, 4, 3)

    def test_tree_at_replaces_one_field(self) -> None:
        state = at_point(SOURCE, 4)
        stopped = eqx.tree_at(lambda s: s.active, state, jnp.zeros(4, dtype=bool))
        assert not stopped.active.any()
        assert_matches_oracle(stopped.position, state.position, rtol=0.0)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("position", jnp.zeros((4, 1)), "2 or 3 columns"),
            ("position", jnp.zeros((2, 4, 3)), r"\(N, 2\) or \(N, 3\)"),
            ("velocity", jnp.zeros((4, 2)), "'velocity'"),
            ("mass", jnp.zeros(3), "'mass'"),
            ("release_time", jnp.zeros(3), "'release_time'"),
            ("active", jnp.ones(3, dtype=bool), "'active'"),
        ],
    )
    def test_a_field_of_the_wrong_shape(
        self, field: str, value: jax.Array, match: str
    ) -> None:
        fields = {
            "position": jnp.zeros((4, 3)),
            "velocity": jnp.zeros((4, 3)),
            "mass": jnp.zeros(4),
            "release_time": jnp.zeros(4),
            "active": jnp.ones(4, dtype=bool),
        }
        # jaxtyping's hook, on in the tests only, may refuse it first
        with pytest.raises((ValidationError, TypeError), match=match):
            _unchecked_state(**(fields | {field: value}))


def _unchecked_state(**fields: jax.Array) -> None:
    """Run the state's own check, without jaxtyping's in front of it."""
    state = object.__new__(ParticleState)
    for name, value in fields.items():
        object.__setattr__(state, name, value)
    state.__check_init__()


class TestParticleStep:
    def test_a_toy_step_satisfies_the_protocol(self) -> None:
        assert isinstance(Drift(1.0), ParticleStep)

    @pytest.mark.parametrize("other", [lambda state: state, object(), Homogeneous()])
    def test_something_else_does_not(self, other: object) -> None:
        assert not isinstance(other, ParticleStep)


class TestAtPoint:
    @pytest.mark.parametrize("source", [SOURCE, SOURCE[:2]])
    def test_every_particle_is_at_the_point(self, source: jax.Array) -> None:
        state = at_point(source, 5)
        expected = np.broadcast_to(np.asarray(source), (5, source.size))
        assert_matches_oracle(state.position, expected, rtol=0.0)
        assert_matches_oracle(state.velocity, np.zeros_like(expected), rtol=0.0)
        assert bool(state.active.all())

    def test_the_release_time_is_one_for_all_or_one_each(self) -> None:
        assert_matches_oracle(
            at_point(SOURCE, 3, release_time=7.0).release_time, np.full(3, 7.0), rtol=0
        )
        times = release_times(0.0, 60.0, 3)
        state = at_point(SOURCE, 3, release_time=times)
        assert_matches_oracle(state.release_time, times, rtol=0.0)

    def test_transforms(self) -> None:
        points = jax.random.normal(KEY, (3, 3))
        masses = jnp.array([1.0, 2.0, 3.0])
        assert_traces(lambda point, mass: at_point(point, 4, mass=mass), points, masses)

    def test_gradient(self) -> None:
        assert_grad_matches_fd(
            lambda x: at_point(x[:3], 4, mass=x[3]), jnp.array([1.0, 2.0, 3.0, 4.0])
        )

    def test_an_integer_point_becomes_the_default_float(self) -> None:
        # jaxtyping's hook, on in the tests only, refuses an integer point first
        assert _corner(np.array([1, 2]), "point").dtype == jnp.float64


class TestSeeders:
    """What the three seeders share."""

    SEEDERS = (
        lambda n, **options: at_point(SOURCE, n, **options),
        lambda n, **options: along_line(SOURCE, 2.0 * SOURCE, n, **options),
        lambda n, **options: in_box(SOURCE, SOURCE + 1.0, n, key=KEY, **options),
    )

    @pytest.mark.parametrize("seeder", SEEDERS)
    @settings(max_examples=10)
    # few sizes, because each compiles; XLA flushes subnormal masses to zero
    @given(
        n=st.sampled_from([1, 3, 50]),
        mass=st.just(0.0) | st.floats(1e-6, 1e6),
    )
    def test_the_masses_sum_to_the_release(self, seeder, n: int, mass: float) -> None:
        assert_matches_oracle(seeder(n, mass=mass).mass.sum(), mass, rtol=1e-12)

    @pytest.mark.parametrize("seeder", SEEDERS)
    @pytest.mark.parametrize(
        ("n", "options", "match"),
        [
            (0, {}, "'n'"),
            (True, {}, "'n'"),
            (3, {"mass": -1.0}, "'mass'"),
            (3, {"mass": float("nan")}, "'mass'"),
            (3, {"mass": jnp.ones(3)}, "'mass'"),
            (3, {"release_time": jnp.zeros(2)}, "'release_time'"),
        ],
    )
    def test_invalid_arguments(self, seeder, n, options, match: str) -> None:
        with pytest.raises((ValidationError, TypeError), match=match):
            seeder(n, **options)

    @pytest.mark.parametrize("seeder", SEEDERS)
    def test_float32_in_float32_out(self, seeder) -> None:
        small = SOURCE.astype(jnp.float32)
        state = {
            0: lambda: at_point(small, 4),
            1: lambda: along_line(small, 2.0 * small, 4),
            2: lambda: in_box(small, small + 1.0, 4, key=KEY),
        }[self.SEEDERS.index(seeder)]()
        for leaf in (state.position, state.velocity, state.mass, state.release_time):
            assert leaf.dtype == jnp.float32

    def test_a_traced_mass_is_not_compared(self) -> None:
        assert jax.jit(lambda mass: at_point(SOURCE, 2, mass=mass).mass)(-2.0)[0] == -1

    @pytest.mark.parametrize(
        "corner", [np.zeros(1), np.zeros(4), np.zeros((2, 3)), np.zeros(())]
    )
    def test_a_point_of_the_wrong_shape(self, corner: np.ndarray) -> None:
        with pytest.raises((ValidationError, TypeError), match="point"):
            at_point(corner, 2)


class TestAlongLine:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_the_middles_of_equal_pieces(self, dim: int) -> None:
        start, stop = np.arange(dim) + 1.0, np.arange(dim) * 3.0 - 2.0
        state = along_line(start, stop, 6)
        expected = [start + (p + 0.5) / 6 * (stop - start) for p in range(6)]
        assert_matches_oracle(state.position, np.array(expected), rtol=1e-14)

    def test_transforms_and_gradient(self) -> None:
        ends = jax.random.normal(KEY, (3, 2, 2))
        assert_traces(lambda pair: along_line(pair[0], pair[1], 4), ends)
        assert_grad_matches_fd(lambda pair: along_line(pair[0], pair[1], 4), ends[0])

    def test_ends_of_different_dimensions(self) -> None:
        with pytest.raises((ValidationError, TypeError), match="start"):
            along_line(SOURCE, SOURCE[:2], 4)


class TestInBox:
    LOWER, UPPER = jnp.array([0.0, -5.0, 2.0]), jnp.array([1.0, 5.0, 100.0])

    @pytest.mark.parametrize("column", [0, 1, 2])
    def test_uniform_on_each_axis(self, column: int) -> None:
        state = in_box(self.LOWER, self.UPPER, 20_000, key=KEY)
        low, high = float(self.LOWER[column]), float(self.UPPER[column])
        assert_uniform(state.position[:, column], low, high)

    def test_transforms(self) -> None:
        keys = jax.random.split(KEY, 3)
        assert_traces(lambda key: in_box(self.LOWER, self.UPPER, 4, key=key), keys)

    def test_gradient_in_the_corners(self) -> None:
        corners = jnp.stack([self.LOWER, self.UPPER])
        assert_grad_matches_fd(lambda c: in_box(c[0], c[1], 4, key=KEY), corners)

    def test_float32_rounding_does_not_reach_the_upper_corner(self) -> None:
        lower = jnp.array([1000.0, 0.0], jnp.float32)
        upper = jnp.array([1001.0, 1.0], jnp.float32)
        state = in_box(lower, upper, 200_000, key=KEY)
        assert bool((state.position < upper).all())
        assert bool((state.position >= lower).all())

    def test_lower_below_upper(self) -> None:
        with pytest.raises(ValidationError, match="'lower' must be below"):
            in_box(self.UPPER, self.LOWER, 4, key=KEY)


class TestReleaseTimes:
    @settings(max_examples=30)
    @given(
        start=st.floats(-1e6, 1e6),
        length=st.floats(1e-3, 1e6),
        n=st.sampled_from([1, 7, 100]),
    )
    def test_they_are_inside_with_the_mean_of_the_interval(
        self, start: float, length: float, n: int
    ) -> None:
        stop = start + length
        times = np.asarray(release_times(start, stop, n))
        assert ((times >= start) & (times <= stop)).all()
        # rounding of times as large as the larger end
        scale = max(abs(start), abs(stop))
        assert abs(times.mean() - 0.5 * (start + stop)) <= 1e-13 * scale

    def test_transforms_and_gradient(self) -> None:
        ends = jax.random.normal(KEY, (3, 2))
        assert_traces(lambda pair: release_times(pair[0], pair[1], 5), ends)
        assert_grad_matches_fd(lambda pair: release_times(pair[0], pair[1], 5), ends[0])

    def test_a_count_below_one(self) -> None:
        with pytest.raises(ValidationError, match="'n'"):
            release_times(0.0, 1.0, 0)


class TestWithTurbulentVelocity:
    def test_it_is_the_draw_in_numpy(self) -> None:
        """σ(z₀) times a standard normal."""
        stored = reference_values("residence")
        size = int(stored["size"])
        _, key = jax.random.split(jax.random.PRNGKey(int(stored["seed"])))
        state = at_point(stored["source"], size)
        closure = LinearSigma()
        moved = with_turbulent_velocity(state, closure, key=key)
        sigma, _ = closure.at(jnp.asarray(stored["source"][2]))
        expected = np.asarray(jax.random.normal(key, (size, 3))) * np.asarray(sigma)
        assert_matches_oracle(moved.velocity, expected, rtol=1e-14)
        assert_matches_oracle(moved.position, state.position, rtol=0.0)

    def test_the_spread_is_sigma(self) -> None:
        state = in_box(jnp.zeros(3), jnp.full(3, 50.0), 20_000, key=KEY)
        moved = with_turbulent_velocity(state, LinearSigma(), key=jax.random.key(1))
        sigma, _ = LinearSigma().at(state.position[:, 2])
        # (u'/σ)² has mean 1
        assert_mean((moved.velocity / sigma) ** 2, 1.0)

    def test_transforms_and_gradient(self) -> None:
        points = jnp.abs(jax.random.normal(KEY, (3, 3)))

        def drawn(point: jax.Array, sigma: jax.Array) -> jax.Array:
            closure = Homogeneous(sigma=sigma)
            state = at_point(point, 4)
            return with_turbulent_velocity(state, closure, key=KEY).velocity

        sigmas = jnp.array([0.1, 0.2, 0.3])
        assert_traces(drawn, points, sigmas)
        assert_grad_matches_fd(lambda sigma: drawn(points[0], sigma), sigmas[0])

    def test_float32(self) -> None:
        state = at_point(SOURCE.astype(jnp.float32), 4)
        moved = with_turbulent_velocity(state, Homogeneous(), key=KEY)
        assert moved.velocity.dtype == jnp.float32

    def test_two_dimensions_have_no_turbulence(self) -> None:
        with pytest.raises(ValidationError, match="3 position columns"):
            with_turbulent_velocity(at_point(SOURCE[:2], 4), Homogeneous(), key=KEY)
