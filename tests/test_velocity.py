"""Tests for `tracerax.velocity`: uniform, gridded and summed velocities."""

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
from _oracles import plumax_values
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import Cubic, Linear, Policies

from tracerax import ValidationError
from tracerax.velocity import (
    GriddedVelocity,
    WeightedSum,
    cgrid_grids,
    uniform,
    uniform_in_space,
)


KEY = jax.random.key(0)

# plumax computed the values in XLA; another order of the same sums rounds
# differently, and the fields are of order 1
RTOL, ATOL = 1e-12, 1e-12


def _stored_wind() -> tuple[GriddedVelocity, dict[str, np.ndarray]]:
    stored = plumax_values("wind")
    grids = cgrid_grids(stored["times"], stored["x"], stored["y"], stored["z"])
    fields = (stored["u"], stored["v"], stored["w"])
    return GriddedVelocity.fit(grids, fields), stored


def _gridded_2d(scheme: Any = None) -> GriddedVelocity:
    """A random 2-D velocity on [0, 100] × [0, 60] and [0, 120] s."""
    times, x, y = (
        jnp.linspace(0, 120, 4),
        jnp.linspace(0, 100, 9),
        jnp.linspace(0, 60, 7),
    )
    grids = cgrid_grids(times, x, y)
    fields = [jax.random.normal(key, (4, 7, 9)) for key in jax.random.split(KEY)]
    return GriddedVelocity.fit(grids, fields, scheme=scheme or Linear())


class TestUniform:
    @pytest.mark.parametrize("dim", [2, 3])
    def test_the_same_everywhere_and_always(self, dim: int) -> None:
        value = jnp.arange(dim) + 1.0
        velocity = uniform(value)
        position = jax.random.normal(KEY, (5, dim))
        expected = np.broadcast_to(np.asarray(value), (5, dim))
        for t in (0.0, jnp.arange(5.0)):
            assert_matches_oracle(velocity(t, position), expected, rtol=0.0)

    def test_gradient_reaches_the_value(self) -> None:
        position = jnp.zeros((4, 2))
        assert_grad_matches_fd(
            lambda value: uniform(value)(0.0, position), jnp.array([1.0, -2.0])
        )

    def test_float32_positions_give_float32(self) -> None:
        velocity = uniform(jnp.array([1.0, 2.0]))
        assert velocity(0.0, jnp.zeros((3, 2), jnp.float32)).dtype == jnp.float32

    @pytest.mark.parametrize("value", [np.zeros(1), np.zeros(4), np.zeros((2, 2))])
    def test_a_value_of_the_wrong_shape(self, value: np.ndarray) -> None:
        with pytest.raises((ValidationError, TypeError), match="value"):
            uniform(value)


class TestUniformInSpace:
    VELOCITY = staticmethod(
        uniform_in_space(lambda t: jnp.array([1.0 + 0.1 * t, jnp.sin(t), 0.0]))
    )

    def test_a_scalar_time_is_every_particles(self) -> None:
        position = jnp.zeros((4, 3))
        for t in (3.0, jnp.asarray(3.0), jnp.array([3.0])):
            expected = np.broadcast_to([1.3, np.sin(3.0), 0.0], (4, 3))
            assert_matches_oracle(self.VELOCITY(t, position), expected, rtol=1e-15)

    def test_a_time_per_particle(self) -> None:
        times = np.array([0.0, 5.0, -2.0])
        expected = np.column_stack([1.0 + 0.1 * times, np.sin(times), 0.0 * times])
        moved = self.VELOCITY(jnp.asarray(times), jnp.zeros((3, 3)))
        assert_matches_oracle(moved, expected, rtol=1e-15)

    def test_transforms(self) -> None:
        position = jax.random.normal(KEY, (3, 4, 3))
        assert_traces(self.VELOCITY, jnp.array([0.0, 1.0, 2.0]), position)

    def test_float32_positions_give_float32(self) -> None:
        velocity = self.VELOCITY(jnp.arange(3.0), jnp.zeros((3, 3), jnp.float32))
        assert velocity.dtype == jnp.float32


class TestGriddedVelocity:
    def test_it_is_plumaxs_gridded_wind(self) -> None:
        velocity, stored = _stored_wind()
        position = jnp.asarray(stored["position"])
        for time, expected in zip(stored["t"], stored["velocity"], strict=True):
            assert_matches_oracle(
                velocity(time, position), expected, rtol=RTOL, atol=ATOL
            )

    def test_one_time_per_particle(self) -> None:
        velocity, stored = _stored_wind()
        position = jnp.asarray(stored["position"][: stored["t"].size])
        expected = [stored["velocity"][k, k] for k in range(stored["t"].size)]
        moved = velocity(jnp.asarray(stored["t"]), position)
        assert_matches_oracle(moved, np.array(expected), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_a_linear_field_is_read_exactly(self, dim: int) -> None:
        """u = x, v = y, w = z at their own staggered points."""
        x, y, z = jnp.arange(5.0), 2.0 * jnp.arange(4.0), jnp.array([0.0, 1.0, 5.0])
        levels = None if dim == 2 else z
        grids = cgrid_grids(jnp.array([0.0, 10.0]), x, y, levels)
        fields = []
        for column, grid in enumerate(grids):
            axes = [axis.values for axis in grid.axes]
            mesh = jnp.meshgrid(*axes, indexing="ij")
            fields.append(mesh[-1 - column])
        velocity = GriddedVelocity.fit(grids, fields)
        low, high = np.array([0.5, 1.0, 0.0])[:dim], np.array([4.0, 6.0, 5.0])[:dim]
        position = low + jax.random.uniform(KEY, (50, dim)) * (high - low)
        assert_matches_oracle(velocity(3.0, position), position, rtol=1e-13)

    def test_linear_in_time_and_held_outside(self) -> None:
        grids = cgrid_grids(jnp.array([0.0, 10.0]), jnp.arange(3.0), jnp.arange(3.0))
        ramp = jnp.stack([jnp.zeros((3, 3)), jnp.ones((3, 3))])
        velocity = GriddedVelocity.fit(grids, (ramp, 2.0 * ramp))
        position = jnp.array([[1.0, 1.0], [-50.0, 90.0]])
        for t, fraction in ((-5.0, 0.0), (2.5, 0.25), (10.0, 1.0), (99.0, 1.0)):
            expected = np.broadcast_to([fraction, 2.0 * fraction], (2, 2))
            assert_matches_oracle(velocity(t, position), expected, rtol=1e-15)

    def test_one_sample_in_time_is_every_times(self) -> None:
        grids = cgrid_grids(jnp.array([5.0]), jnp.arange(3.0), jnp.arange(3.0))
        fields = [jax.random.normal(key, (1, 3, 3)) for key in jax.random.split(KEY)]
        velocity = GriddedVelocity.fit(grids, fields)
        position = jax.random.uniform(KEY, (6, 2)) * 2.0
        reference = velocity(5.0, position)
        assert bool(jnp.isfinite(reference).all())
        for t in (-100.0, 0.0, 99.0):
            assert_matches_oracle(velocity(t, position), reference, rtol=0.0)

    def test_transforms(self) -> None:
        velocity = _gridded_2d()
        position = jax.random.uniform(KEY, (3, 4, 2)) * 50.0
        # a traced axis and a concrete one round their spacing differently
        times = jnp.array([0.0, 50.0, 130.0])
        assert_traces(lambda t, x: velocity(t, x), times, position, rtol=1e-11)

    def test_a_traced_velocity_reads_the_same(self) -> None:
        velocity = _gridded_2d()
        position = jax.random.uniform(KEY, (20, 2)) * 50.0
        traced = eqx.filter_jit(lambda v, x: v(30.0, x))(velocity, position)
        assert_matches_oracle(traced, velocity(30.0, position), rtol=1e-11)

    @pytest.mark.parametrize("scheme", [Linear(), Cubic()])
    def test_jacobian_in_the_position(self, scheme: Any) -> None:
        """Inside a cell: the linear read has a kink at every node."""
        velocity = _gridded_2d(scheme)
        position = jnp.array([[40.3, 21.7], [77.1, 48.9]])
        forward = jax.jacfwd(lambda x: velocity(33.0, x))(position)
        reverse = jax.jacrev(lambda x: velocity(33.0, x))(position)
        assert_matches_oracle(forward, reverse, rtol=0.0, atol=1e-13)
        assert_grad_matches_fd(lambda x: velocity(33.0, x), position, eps=1e-4)

    def test_outside_the_jacobian_is_zero(self) -> None:
        velocity = _gridded_2d()
        outside = jnp.array([[-500.0, -500.0]])
        jacobian = jax.jacfwd(lambda x: velocity(33.0, x))(outside)
        assert_matches_oracle(jacobian, np.zeros((1, 2, 1, 2)), rtol=0.0)

    def test_gradient_in_the_samples_is_the_plans_adjoint(self) -> None:
        stored = plumax_values("wind")
        grids = cgrid_grids(stored["times"], stored["x"], stored["y"], stored["z"])
        fields = tuple(jnp.asarray(stored[name]) for name in "uvw")
        position = jnp.asarray(stored["position"])
        residual = jax.random.normal(KEY, position.shape)
        time = 30.0

        def projected(samples: tuple[jax.Array, ...]) -> jax.Array:
            return jnp.vdot(
                GriddedVelocity.fit(grids, samples)(time, position), residual
            )

        gradients = jax.grad(projected)(fields)
        targets = Points(
            jnp.column_stack([jnp.full(len(position), time), position[:, ::-1]])
        )
        for column, (grid, gradient) in enumerate(zip(grids, gradients, strict=True)):
            plan = Linear().plan(grid, targets, policies=Policies(outside="clamp"))
            expected = plan.adjoint(residual[:, column]).reshape(grid.shape)
            assert_matches_oracle(gradient, expected, rtol=1e-12, atol=1e-13)

    def test_a_periodic_axis_reads_the_same_a_period_away(self) -> None:
        period = 360.0
        longitude = Axis(jnp.arange(0.0, period, 30.0), period=period)
        grid = RectilinearGrid(
            (Axis(jnp.array([0.0])), Axis(jnp.arange(4.0)), longitude)
        )
        fields = [jax.random.normal(key, grid.shape) for key in jax.random.split(KEY)]
        velocity = GriddedVelocity.fit((grid, grid), fields)
        # 345 is in the cell across the seam
        position = jnp.array([[10.0, 1.2], [345.0, 2.7], [359.9, 0.1]])
        shifted = position.at[:, 0].add(period)
        expected = velocity(0.0, position)
        # rounding of x + P, 6e-14 of a 30° cell
        assert_matches_oracle(velocity(0.0, shifted), expected, rtol=0.0, atol=1e-13)

    def test_land_is_dropped_under_renormalise(self) -> None:
        grids = cgrid_grids(jnp.array([0.0]), jnp.arange(3.0), jnp.arange(3.0))
        field = jnp.ones((1, 3, 3)).at[0, 0, 0].set(jnp.nan)
        policies = Policies(outside="clamp", missing="renormalise")
        velocity = GriddedVelocity.fit(grids, (field, field), policies=policies)
        moved = velocity(0.0, jnp.array([[0.9, 0.9]]))
        assert_matches_oracle(moved, np.ones((1, 2)), rtol=1e-14)

    def test_it_runs_in_integrix_departure_points(self) -> None:
        velocity = _gridded_2d()
        nodes = jax.random.uniform(KEY, (10, 2)) * 50.0
        # Δt |∂u/∂x| < 1, so the fixed point is reached
        departed = integrix.march.departure_points(
            nodes, velocity, 0.0, 1.0, iterations=60
        )
        # x_d = x − Δt u(t + Δt/2, (x + x_d)/2)
        expected = nodes - velocity(0.5, 0.5 * (nodes + departed))
        assert_matches_oracle(departed, expected, rtol=1e-10)
        assert departed.shape == nodes.shape

    def test_float32_positions_give_float32(self) -> None:
        position = jnp.ones((3, 2), jnp.float32)
        assert _gridded_2d()(0.0, position).dtype == jnp.float32

    def test_invalid_arguments(self) -> None:
        grids = cgrid_grids(jnp.array([0.0]), jnp.arange(3.0), jnp.arange(4.0))
        field = jnp.zeros((1, 4, 3))
        fit = GriddedVelocity.fit
        with pytest.raises(ValidationError, match="2 or 3 components"):
            fit(grids[:1], (field,))
        with pytest.raises(ValidationError, match="one grid per component"):
            fit(grids[:1], (field, field))
        with pytest.raises(ValidationError, match="a time axis and 3 axes"):
            fit((*grids, grids[0]), (field, field, field))
        with pytest.raises(ValidationError, match=r"'fields\[1\]'"):
            fit(grids, (field, jnp.zeros((1, 3, 4))))
        with pytest.raises(ValidationError, match="'position' must have 2 columns"):
            fit(grids, (field, field))(0.0, jnp.zeros((2, 3)))


class TestCgridGrids:
    @pytest.mark.parametrize("bad", [np.array([0.0, 1.0, 3.0]), np.array([1.0])])
    @pytest.mark.parametrize("name", ["x", "y"])
    def test_x_and_y_are_uniform(self, name: str, bad: np.ndarray) -> None:
        axes = {"x": np.arange(3.0), "y": np.arange(3.0)} | {name: bad}
        with pytest.raises(ValidationError, match=f"'{name}'"):
            cgrid_grids(np.array([0.0]), **axes)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_a_long_axis_far_from_zero_is_uniform(self, dtype: Any) -> None:
        axis = np.linspace(5e5, 6e5, 2001, dtype=dtype)
        grids = cgrid_grids(np.array([0.0]), axis, axis)
        # Δ = 50 m
        assert_matches_oracle(grids[0].axes[2].values[0], 500_025.0, rtol=1e-7)

    def test_a_wind_is_built_inside_a_jit_on_axes_from_outside_it(self) -> None:
        # concrete axes, though the arithmetic on them is traced
        times, axis = jnp.array([0.0, 10.0]), jnp.arange(4.0)
        position = jax.random.uniform(KEY, (5, 2)) * 3.0

        def moved(speed: jax.Array) -> jax.Array:
            grids = cgrid_grids(times, axis, axis)
            fields = [speed * jnp.ones(grid.shape) for grid in grids]
            return GriddedVelocity.fit(grids, fields)(2.0, position)

        assert_matches_oracle(
            jax.jit(moved)(jnp.array(3.0)), np.full((5, 2), 3.0), rtol=1e-14
        )

        nodes = jnp.array([0.0, 1.0, 3.0])

        def stretched(speed: jax.Array) -> jax.Array:
            return speed * cgrid_grids(times, nodes, axis)[0].axes[2].values

        with pytest.raises(ValidationError, match="uniformly spaced"):
            jax.jit(stretched)(jnp.array(1.0))


class TestWeightedSum:
    @staticmethod
    def _terms() -> tuple[GriddedVelocity, Any]:
        coarse = cgrid_grids(
            jnp.array([0.0, 200.0]), jnp.array([0.0, 200.0]), jnp.array([0.0, 90.0])
        )
        fields = [jax.random.normal(key, (2, 2, 2)) for key in jax.random.split(KEY)]
        return _gridded_2d(), GriddedVelocity.fit(coarse, fields)

    def test_it_is_the_sum_of_its_terms_on_different_grids(self) -> None:
        current, wind = self._terms()
        position = jax.random.uniform(KEY, (20, 2)) * 50.0
        times = jax.random.uniform(KEY, (20,)) * 100.0
        drift = WeightedSum((current, wind), [1.0, 0.03])
        expected = current(times, position) + 0.03 * wind(times, position)
        assert_matches_oracle(drift(times, position), expected, rtol=1e-13)

    def test_gradient_in_the_weights(self) -> None:
        terms = self._terms()
        position = jax.random.uniform(KEY, (5, 2)) * 50.0
        assert_grad_matches_fd(
            lambda weights: WeightedSum(terms, weights)(30.0, position),
            jnp.array([1.0, 0.03]),
            rtol=1e-8,
        )

    def test_transforms(self) -> None:
        drift = WeightedSum(self._terms(), jnp.array([1.0, 0.03]))
        position = jax.random.uniform(KEY, (3, 4, 2)) * 50.0
        times = jnp.array([0.0, 50.0, 130.0])
        assert_traces(lambda t, x: drift(t, x), times, position, rtol=1e-11)

    def test_a_float64_term_keeps_float32_positions(self) -> None:
        drift = WeightedSum((lambda t, x: jnp.ones(x.shape, jnp.float64),), [2.0])
        assert drift(0.0, jnp.ones((3, 2), jnp.float32)).dtype == jnp.float32

    def test_float32_positions_give_float32(self) -> None:
        drift = WeightedSum(self._terms(), jnp.array([1.0, 0.03]))
        assert drift(0.0, jnp.ones((3, 2), jnp.float32)).dtype == jnp.float32

    def test_a_weight_per_term(self) -> None:
        with pytest.raises(ValidationError, match="'weights'"):
            WeightedSum((uniform(jnp.ones(2)),), jnp.ones(2))
        with pytest.raises(ValidationError, match="'velocities'"):
            WeightedSum((), jnp.ones(0))

    @pytest.mark.slow
    def test_windage_is_recovered_from_paths(self) -> None:
        """Gauss–Newton through RK4 paths, for a problem with a zero residual."""
        day = 86_400.0

        def current(t: Any, position: jax.Array) -> jax.Array:
            rate = 2.0 * jnp.pi / (5.0 * day)
            return rate * jnp.column_stack([-position[:, 1], position[:, 0]])

        def wind(t: Any, position: jax.Array) -> jax.Array:
            phase = 2.0 * jnp.pi * t / (3.0 * day) + position[:, 1] / 1e5
            return 8.0 * jnp.column_stack([jnp.cos(phase), jnp.sin(phase)])

        start = jnp.array([[0.0, 2e4], [3e4, 0.0]])

        def paths(windage: jax.Array) -> jax.Array:
            weights = jnp.stack([jnp.ones_like(windage), windage])
            drift = WeightedSum((current, wind), weights)
            solution = dfx.diffeqsolve(
                dfx.ODETerm(lambda t, position, args: drift(t, position)),
                integrix.march.RK4Classic(),
                0.0,
                2.0 * day,
                2.0 * day / 96,
                start,
                saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 2.0 * day, 9)),
                adjoint=dfx.ForwardMode(),
            )
            return solution.ys.ravel()

        truth = 0.03
        paths, jacobian = jax.jit(paths), jax.jit(jax.jacfwd(paths))
        observed = paths(jnp.asarray(truth))
        windage = jnp.asarray(0.0)
        for _ in range(5):
            residual = observed - paths(windage)
            slope = jacobian(windage)
            # β ← β + (Jᵀ J)⁻¹ Jᵀ r
            windage = windage + jnp.vdot(slope, residual) / jnp.vdot(slope, slope)
        assert_matches_oracle(windage, truth, rtol=0.0, atol=1e-10)
