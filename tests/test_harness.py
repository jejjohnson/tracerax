"""Each check passes on a right function and fails on a wrong one, and each
reference is what its name says."""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import integrix.march
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import (
    CosineDiffusivity,
    DoubleGyre,
    SolidBodyRotation,
    taylor_variance,
    uniform_wind_position,
)
from _reference import (
    CLOSURES,
    FAMILIES,
    closure_statistics,
    fold,
    reference_values,
)
from _sampling import (
    assert_mean,
    assert_uniform,
    assert_variance,
    ks_critical_value,
    ks_statistic,
)
from hypothesis import given, strategies as st
from scipy import stats


SIGMA, TIMESCALE = 0.5, 60.0


def toy_step(
    position: jax.Array, velocity: jax.Array, noise: jax.Array, step_size: Any
) -> tuple[jax.Array, jax.Array]:
    """One exact Ornstein–Uhlenbeck step: what a tracerax step looks like."""
    # e^{−Δt/τ}
    decay = jnp.exp(-step_size / TIMESCALE)
    # v ← e^{−Δt/τ} v + σ √(1 − e^{−2Δt/τ}) ξ
    moved = decay * velocity + SIGMA * jnp.sqrt(1.0 - decay**2) * noise
    # x ← x + v Δt
    return position + moved * step_size, moved


def toy_run(size: int, step_size: float, num_steps: int) -> jax.Array:
    """The displacements of ``size`` particles after ``num_steps`` toy steps."""
    key_start, key_steps = jax.random.split(jax.random.key(0))
    velocity = SIGMA * jax.random.normal(key_start, (size,))
    noise = jax.random.normal(key_steps, (num_steps, size))

    def body(carry: Any, draw: jax.Array) -> tuple[Any, None]:
        return toy_step(*carry, draw, step_size), None

    (position, _), _ = jax.lax.scan(body, (jnp.zeros(size), velocity), noise)
    return position


BATCH = 4
BATCHED = tuple(
    jax.random.normal(key, (BATCH, 3)) for key in jax.random.split(jax.random.key(1), 3)
)


class TestTheChecksOnAToyStep:
    """One toy function passes every check."""

    def test_oracle(self) -> None:
        position, velocity, noise = (np.asarray(values) for values in BATCHED)
        decay = np.exp(-5.0 / TIMESCALE)
        expected = decay * velocity + SIGMA * np.sqrt(1.0 - decay**2) * noise
        moved = toy_step(*BATCHED, 5.0)
        # rounding of two ways to write the same sum
        assert_matches_oracle(moved[1], expected, rtol=1e-14)
        assert_matches_oracle(moved[0], position + 5.0 * expected, rtol=1e-13)

    def test_traces(self) -> None:
        assert_traces(lambda *state: toy_step(*state, 5.0), *BATCHED)

    def test_gradient(self) -> None:
        start = tuple(values[0] for values in BATCHED)
        assert_grad_matches_fd(
            lambda step_size: toy_step(*start, step_size), jnp.asarray(5.0)
        )

    def test_adjoint(self) -> None:
        """A deposit and its transpose: the read at the same points."""
        weights = jnp.asarray([[0.25, 0.75, 0.0], [0.0, 0.5, 0.5]])
        assert_adjoint(
            lambda values: weights @ values,
            lambda mass: weights.T @ mass,
            3,
            2,
            key=jax.random.key(0),
        )

    @pytest.mark.slow
    def test_variance(self) -> None:
        # Δt = τ/12: the first-order bias is under the bound 4 (2/N)^½ = 0.025
        size, step_size, num_steps = 50_000, TIMESCALE / 12.0, 120
        expected = taylor_variance(SIGMA, TIMESCALE, step_size * num_steps)
        assert_variance(toy_run(size, step_size, num_steps), float(expected))


class TestTheChecksRefuseAWrongFunction:
    def test_oracle_names_the_worst_index(self) -> None:
        with pytest.raises(AssertionError, match=r"worst at index \(1,\)"):
            assert_matches_oracle([1.0, 9.0], [1.0, 2.0], rtol=1e-12)

    def test_oracle_refuses_a_nan_on_one_side(self) -> None:
        with pytest.raises(AssertionError, match="out of tolerance"):
            assert_matches_oracle([1.0, np.nan], [1.0, 2.0], rtol=1.0)

    def test_oracle_refuses_another_shape(self) -> None:
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_matches_oracle([1.0], [[1.0]], rtol=1.0)

    def test_oracle_compares_masks_exactly(self) -> None:
        with pytest.raises(AssertionError, match="out of tolerance"):
            assert_matches_oracle([True, False], [True, True], rtol=10.0)

    def test_traces_refuses_a_function_that_sees_the_trace(self) -> None:
        def wrong(values: jax.Array) -> jax.Array:
            return values if isinstance(values, jax.core.Tracer) else values + 1.0

        with pytest.raises(AssertionError, match=r"jit\(fn\)"):
            assert_traces(wrong, BATCHED[0])

    def test_gradient_refuses_a_stopped_one(self) -> None:
        with pytest.raises(AssertionError, match="out of tolerance"):
            assert_grad_matches_fd(
                lambda x: jnp.sum(jax.lax.stop_gradient(x) * x), jnp.array([0.3, 0.7])
            )

    def test_adjoint_refuses_a_scaled_one(self) -> None:
        weights = jnp.asarray([[0.25, 0.75, 0.0], [0.0, 0.5, 0.5]])
        with pytest.raises(AssertionError, match="dot-product"):
            assert_adjoint(
                lambda values: weights @ values,
                lambda mass: 2.0 * weights.T @ mass,
                3,
                2,
                key=jax.random.key(0),
            )

    def test_variance_refuses_a_step_that_is_too_long(self) -> None:
        # Δt = τ: the measured bias of the scheme is 11 %
        expected = taylor_variance(SIGMA, TIMESCALE, 600.0)
        with pytest.raises(AssertionError, match="standard errors"):
            assert_variance(toy_run(50_000, TIMESCALE, 10), float(expected))


class TestStatistics:
    def test_the_ks_statistic_is_scipys(self) -> None:
        sample = np.random.default_rng(0).uniform(2.0, 5.0, 500)
        expected = stats.kstest(sample, stats.uniform(2.0, 3.0).cdf).statistic
        assert_matches_oracle(ks_statistic(sample, 2.0, 5.0), expected, rtol=1e-12)

    def test_the_critical_value_is_the_limiting_law(self) -> None:
        # the Kolmogorov law's upper tail, to its leading term: 1 % at 1e-3
        expected = stats.kstwobign.isf(1e-3) / np.sqrt(10_000)
        assert_matches_oracle(ks_critical_value(10_000, 1e-3), expected, rtol=1e-2)
        assert round(ks_critical_value(1, 1e-3), 2) == 1.95

    def test_uniform_refuses_another_law(self) -> None:
        sample = np.random.default_rng(0).uniform(0.0, 1.0, 5_000) ** 1.2
        with pytest.raises(AssertionError, match="not uniform"):
            assert_uniform(sample, 0.0, 1.0)

    def test_uniform_refuses_a_draw_outside(self) -> None:
        sample = np.append(np.random.default_rng(0).uniform(0.0, 1.0, 99), 1.1)
        with pytest.raises(AssertionError, match="1 of 100 draws outside"):
            assert_uniform(sample, 0.0, 1.0)

    def test_mean_refuses_a_shifted_one(self) -> None:
        sample = np.random.default_rng(0).normal(0.2, 1.0, 1_000)
        with pytest.raises(AssertionError, match="standard errors"):
            assert_mean(sample, 0.0)

    def test_mean_refuses_a_constant(self) -> None:
        with pytest.raises(AssertionError, match="no spread"):
            assert_mean(np.ones(100), 1.0)

    @pytest.mark.parametrize("check", [assert_mean, assert_variance])
    def test_a_nan_is_refused(self, check: Any) -> None:
        with pytest.raises(AssertionError, match="not finite"):
            check(np.append(np.zeros(99), np.nan), 1.0)

    @pytest.mark.parametrize("check", [assert_mean, assert_variance])
    def test_few_draws_are_refused(self, check: Any) -> None:
        with pytest.raises(AssertionError, match="too few"):
            check(np.zeros(5), 1.0)


class TestClosedForms:
    @given(
        angle=st.floats(0.0, 2.0 * np.pi),
        radius=st.floats(0.1, 1e4),
        duration=st.floats(-1e4, 1e4),
    )
    def test_rotation_keeps_the_radius(
        self, angle: float, radius: float, duration: float
    ) -> None:
        start = radius * jnp.array([[np.cos(angle), np.sin(angle)]])
        moved = SolidBodyRotation(1e-3).exact(start, duration)
        assert_matches_oracle(jnp.linalg.norm(moved), radius, rtol=1e-12)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_rotation_velocity_is_the_derivative_of_its_path(self, dim: int) -> None:
        rotation = SolidBodyRotation(1e-3)
        start = jax.random.normal(jax.random.key(0), (5, dim))
        _, rate = jax.jvp(lambda time: rotation.exact(start, time), (0.0,), (1.0,))
        assert_matches_oracle(rate, rotation(0.0, start), rtol=1e-12, atol=1e-18)

    def test_rotation_returns_after_a_period(self) -> None:
        rotation = SolidBodyRotation(1e-3)
        start = jnp.array([[1e3, -2e3, 5.0]])
        # rounding of 2π, 2e-16 of the radius, with room for its sine
        moved = rotation.exact(start, rotation.period)
        assert_matches_oracle(moved, start, rtol=1e-12)

    @pytest.mark.parametrize("duration", [800.0, -800.0])
    def test_rotation_is_what_integrix_steps_to(self, duration: float) -> None:
        rotation = SolidBodyRotation(1e-3)
        start = jnp.array([[1e3, -2e3]])
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, position, args: rotation(t, position)),
            integrix.march.RK4Classic(),
            0.0,
            duration,
            duration / 200,
            start,
        )
        # fourth order: (ωΔt)⁴ = (4e-3)⁴ per radian
        expected = rotation.exact(start, duration)
        assert_matches_oracle(solution.ys[0], expected, rtol=1e-10)

    def test_uniform_wind_waits_for_the_release(self) -> None:
        moved = uniform_wind_position(
            np.zeros((3, 2)), np.array([2.0, -1.0]), np.array([0.0, 4.0, 20.0]), 10.0
        )
        expected = np.array([[20.0, -10.0], [12.0, -6.0], [0.0, 0.0]])
        assert_matches_oracle(moved, expected, rtol=0.0)

    def test_taylor_variance_at_both_ends(self) -> None:
        # ballistic: σ² t², to O(t/τ); diffusive: 2 σ² τ (t − τ), to e^{−t/τ}
        short, long = 1e-4 * TIMESCALE, 100.0 * TIMESCALE
        assert_matches_oracle(
            taylor_variance(SIGMA, TIMESCALE, short), (SIGMA * short) ** 2, rtol=1e-4
        )
        diffusive = 2.0 * SIGMA**2 * TIMESCALE * (long - TIMESCALE)
        assert_matches_oracle(
            taylor_variance(SIGMA, TIMESCALE, long), diffusive, rtol=1e-12
        )

    def test_taylor_variance_integrates_the_correlation(self) -> None:
        """d²Var/dt² = 2 σ² e^{−t/τ}, by a central difference."""
        time, step = 40.0, 0.5
        ahead, here, behind = taylor_variance(
            SIGMA, TIMESCALE, np.array([time + step, time, time - step])
        )
        second = (ahead - 2.0 * here + behind) / step**2
        expected = 2.0 * SIGMA**2 * np.exp(-time / TIMESCALE)
        # truncation: step² / (12 τ²) = 6e-6
        assert_matches_oracle(second, expected, rtol=1e-5)

    def test_the_double_gyre_is_incompressible(self) -> None:
        gyre = DoubleGyre()
        points = jax.random.uniform(jax.random.key(0), (50, 2)) * jnp.array([2.0, 1.0])
        jacobian = jax.vmap(jax.jacfwd(lambda point: gyre(3.0, point)))(points)
        divergence = jacobian[:, 0, 0] + jacobian[:, 1, 1]
        assert_matches_oracle(divergence, np.zeros(50), rtol=0.0, atol=1e-14)

    def test_the_double_gyre_does_not_cross_its_walls(self) -> None:
        gyre = DoubleGyre()
        along = jnp.linspace(0.0, 1.0, 11)
        for x in (0.0, 2.0):
            side = jnp.stack([jnp.full(11, x), along], axis=-1)
            assert_matches_oracle(
                gyre(3.0, side)[:, 0], np.zeros(11), rtol=0.0, atol=1e-15
            )
        for y in (0.0, 1.0):
            end = jnp.stack([2.0 * along, jnp.full(11, y)], axis=-1)
            assert_matches_oracle(
                gyre(3.0, end)[:, 1], np.zeros(11), rtol=0.0, atol=1e-15
            )

    @pytest.mark.slow
    def test_the_double_gyres_flow_map_keeps_area(self) -> None:
        gyre = DoubleGyre()

        def flow(point: jax.Array) -> jax.Array:
            solution = dfx.diffeqsolve(
                dfx.ODETerm(lambda t, position, args: gyre(t, position)),
                integrix.march.RK4Classic(),
                0.0,
                10.0,
                0.05,
                point,
                adjoint=dfx.ForwardMode(),
            )
            return solution.ys[0]

        points = jnp.array([[0.5, 0.3], [1.1, 0.6], [1.7, 0.2]])
        determinant = jnp.linalg.det(jax.vmap(jax.jacfwd(flow))(points))
        # fourth order in Δt = 0.05
        assert_matches_oracle(determinant, np.ones(3), rtol=1e-6)

    def test_the_diffusivity_has_its_own_gradient_and_is_flat_at_the_walls(
        self,
    ) -> None:
        diffusivity = CosineDiffusivity(mean=0.03, contrast=0.4, length=50.0)
        points = jnp.linspace(0.0, 50.0, 9)
        gradient = jax.vmap(jax.grad(diffusivity))(points)
        assert_matches_oracle(
            diffusivity.gradient(points), gradient, rtol=1e-12, atol=1e-18
        )
        assert_matches_oracle(
            gradient[jnp.array([0, -1])], np.zeros(2), rtol=0, atol=1e-18
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("with_drift", [True, False])
    def test_a_uniform_cloud_needs_the_drift_to_stay_uniform(
        self, with_drift: bool
    ) -> None:
        diffusivity = CosineDiffusivity(mean=0.03, contrast=0.5, length=1.0)
        size, step_size, num_steps = 20_000, 2e-3, 2_000
        key_start, key_steps = jax.random.split(jax.random.key(0))
        start = jax.random.uniform(key_start, (size,))

        def body(position: jax.Array, key: jax.Array) -> tuple[jax.Array, None]:
            noise = jax.random.normal(key, (size,))
            spread = jnp.sqrt(2.0 * diffusivity(position) * step_size)
            drift = diffusivity.gradient(position) if with_drift else 0.0
            # dx = ∂K/∂x dt + √(2K) dW, folded at both walls
            moved = jnp.abs(position + drift * step_size + spread * noise)
            return 1.0 - jnp.abs(1.0 - moved), None

        final, _ = jax.lax.scan(body, start, jax.random.split(key_steps, num_steps))
        if with_drift:
            assert_uniform(final, 0.0, 1.0)
            return
        with pytest.raises(AssertionError, match="not uniform"):
            assert_uniform(final, 0.0, 1.0)


# --- the runs written out in NumPy -------------------------------------------


class TestTheReferenceRuns:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_array_is_double_and_finite(self, family: str) -> None:
        for name, values in reference_values(family).items():
            if name == "key":
                continue
            assert values.dtype in (np.float64, np.int64), name
            assert np.isfinite(values).all(), name

    def test_a_caller_gets_its_own_copy(self) -> None:
        reference_values("wind")["u"][:] = np.nan
        assert np.isfinite(reference_values("wind")["u"]).all()

    @pytest.mark.parametrize("name", CLOSURES)
    def test_the_closures_are_the_formulas_there(self, name: str) -> None:
        height = np.linspace(0.0, 50.0, 7)
        sigma, variance_slope = closure_statistics(name, height)
        closure = CLOSURES[name]
        got_sigma, got_timescale = closure.at(jnp.asarray(height))
        assert_matches_oracle(got_sigma, sigma, rtol=1e-14)
        assert_matches_oracle(got_timescale, np.full((7, 3), 60.0), rtol=0.0)
        slope = jax.vmap(jax.grad(lambda z: closure.at(z)[0][..., 2] ** 2))(
            jnp.asarray(height)
        )
        assert_matches_oracle(slope, variance_slope, rtol=1e-14)

    def test_the_fold_reflects_one_wall_at_a_time(self) -> None:
        height = np.linspace(-180.0, 230.0, 83)
        folded, sign = height.copy(), np.ones_like(height)
        # five reflections bring 4.6 layers inside
        for _ in range(5):
            below, above = folded < 0.0, folded > 50.0
            folded = np.where(below, -folded, np.where(above, 100.0 - folded, folded))
            sign = np.where(below | above, -sign, sign)
        got, got_sign = fold(height, 50.0)
        assert_matches_oracle(got, folded, rtol=0.0, atol=1e-12)
        # at a wall, the sign is a convention
        between = (folded > 1e-9) & (folded < 50.0 - 1e-9)
        assert np.array_equal(got_sign[between], sign[between])

    @pytest.mark.parametrize("name", CLOSURES)
    def test_the_step_meets_both_walls(self, name: str) -> None:
        """A case that never reflects would not test a wall."""
        run = reference_values("langevin")
        start = run[f"{name}_start_position"][:, 2]
        unfolded_up = run[f"{name}_velocity"][:, 2] + run["wind"][2]
        moved = run[f"{name}_position"][:, 2] - start
        reflected = np.abs(moved - unfolded_up * run["step_size"]) > 1e-9
        assert np.sum(reflected & (start < 25.0)) >= 1
        assert np.sum(reflected & (start > 25.0)) >= 1

    @pytest.mark.parametrize("prefix", ["", "stretched_"])
    def test_the_counts_are_scipys_away_from_the_edges(self, prefix: str) -> None:
        run = reference_values("binning")
        edges = [run[f"{prefix}{axis}_edges"] for axis in "xyz"]
        counts, _ = np.histogramdd(run["position"], bins=edges)
        # histogramdd closes the last cell, and the second particle is on its edge
        on_edge, _ = np.histogramdd(run["position"][1:2], bins=edges)
        assert on_edge.sum() == 1.0
        assert_matches_oracle(run[f"{prefix}counts"], counts - on_edge, rtol=0.0)
        # some particles are outside, and some inside
        assert 0 < run[f"{prefix}counts"].sum() < len(run["position"]) - 1

    @pytest.mark.parametrize("name", CLOSURES)
    def test_the_runs_fill_their_cells(self, name: str) -> None:
        assert reference_values("residence")[f"{name}_concentration"].sum() > 0.0
        assert reference_values("footprint")[f"{name}_footprint"].sum() > 0.0

    def test_the_wind_is_read_outside_its_grid(self) -> None:
        assert (np.abs(reference_values("wind")["position"][:, 0]) > 25.0).any()
