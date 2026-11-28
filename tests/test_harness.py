"""Each closed form and each reference run is what its name says."""

from __future__ import annotations

import diffrax as dfx
import integrix.march
import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
from hypothesis import given, strategies as st
from pipekit_array.testing import (
    assert_matches_oracle,
    assert_uniform,
)


SIGMA, TIMESCALE = 0.5, 60.0


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
