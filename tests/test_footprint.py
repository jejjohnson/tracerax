"""Tests for the surface footprint."""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous, LinearSigma
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import plumax_values
from integrix.adjoint import Checkpointed, Truncated

from tracerax import ValidationError
from tracerax.advect import Langevin, Walls, integrate
from tracerax.bin import Cells, Residence
from tracerax.footprint import Footprint, surface
from tracerax.particles import in_box, with_turbulent_velocity
from tracerax.velocity import uniform, uniform_in_space


KEY = jax.random.key(0)
CLOSURES = {"homogeneous": Homogeneous(), "linear": LinearSigma()}
WIND = jnp.array([3.0, -1.0, 0.0])
CELLS = Cells(
    (
        np.array([-400.0, -120.0, -50.0, -20.0, 100.0]),
        np.array([-100.0, -10.0, 5.0, 30.0, 150.0]),
    )
)
# every particle stays over it
EVERYWHERE = Cells(([-5e4, 0.0, 5e4], [-5e4, 5e4]))
RECEPTOR = jnp.array([0.0, 0.0, 12.0])
OPTIONS: dict[str, Any] = {
    "n_particles": 32,
    "receptor_time": 300.0,
    "duration": 62.0,
    "step_size": 5.0,
    "surface_depth": 25.0,
    "air_density": 1.2,
    "walls": Walls(0.0, 50.0),
    "key": KEY,
}


def _surface(receptor: Any = RECEPTOR, **changes: Any) -> Footprint:
    options = {"velocity": uniform(WIND), "turbulence": LinearSigma(), "cells": CELLS}
    options |= OPTIONS | changes
    return surface(receptor, **options)


class TestAgainstPlumax:
    """G2-plumax, for the footprint."""

    @pytest.mark.parametrize("prefix", ["", "steady_"])
    @pytest.mark.parametrize("name", CLOSURES)
    def test_compute_footprint(self, name: str, prefix: str) -> None:
        stored = plumax_values("footprint")
        constant, slope = jnp.asarray(stored["wind"])
        varying = uniform_in_space(lambda t: constant + t * slope)
        domains = [stored[f"domain_{axis}"] for axis in "xy"]
        depth = float(stored["pbl_fraction"] * stored["pbl_height"])
        size, density = int(stored["size"]), float(stored["air_density"])
        footprint = surface(
            jnp.asarray(stored["receptor"]),
            uniform(constant) if prefix else varying,
            CLOSURES[name],
            Cells([np.linspace(low, high, int(n) + 1) for low, high, n in domains]),
            n_particles=size,
            receptor_time=float(stored["receptor_time"]),
            duration=float(stored["t_back"]),
            step_size=float(stored["dt"]),
            surface_depth=depth,
            air_density=density,
            walls=Walls(0.0, float(stored["pbl_height"])),
            key=jax.random.PRNGKey(int(stored["seed"])),
        )
        expected = stored[f"{prefix}{name}_footprint"]
        assert_matches_oracle(footprint.sensitivity, expected, rtol=1e-12)
        # R = F N ρ h_s
        assert_matches_oracle(
            footprint.residence, expected * size * density * depth, rtol=1e-12
        )

    def test_the_two_winds_differ(self) -> None:
        stored = plumax_values("footprint")
        for name in CLOSURES:
            change = stored[f"{name}_footprint"] - stored[f"steady_{name}_footprint"]
            assert np.abs(change).max() > 0.1 * stored[f"{name}_footprint"].max()


class TestSurface:
    @pytest.mark.parametrize("kind", ["nearest", "linear"])
    def test_a_layer_that_is_all_surface_sums_to_its_time(self, kind: str) -> None:
        """∑_j F_j = T / (ρ h_s) when every particle is over a cell, below h_s."""
        footprint = _surface(cells=EVERYWHERE, surface_depth=60.0, kind=kind)
        total = footprint.sensitivity.sum()
        assert_matches_oracle(total, 62.0 / (1.2 * 60.0), rtol=1e-12)
        assert_matches_oracle(footprint.residence.sum(), 32 * 62.0, rtol=1e-12)

    def test_it_is_upwind(self) -> None:
        """With no turbulence, the time over a cell is its crossing time."""
        footprint = _surface(
            turbulence=Homogeneous(sigma=0.0),
            cells=Cells(([-121.0, -61.0, -16.0, 0.0, 50.0], [-100.0, 0.0, 100.0])),
            duration=40.0,
            step_size=0.5,
        )
        # back along (−3, 1) m s⁻¹ at 12 m, below the 25 m: x = −3 k / 2 at step k,
        # which is in [−16, 0) for k = 1…10, then 11…40, then 41…80
        assert footprint.residence[:, 0].tolist() == [0.0] * 4
        expected = 32 * 0.5 * np.array([40.0, 30.0, 10.0, 0.0])
        assert_matches_oracle(footprint.residence[:, 1], expected, rtol=1e-12)

    def test_only_the_surface_layer_counts(self) -> None:
        """Deeper layers see more, until all the column is seen."""
        totals = [
            float(_surface(cells=EVERYWHERE, surface_depth=depth).residence.sum())
            for depth in (5.0, 20.0, 50.1)
        ]
        assert 0.0 < totals[0] < totals[1] < totals[2]
        assert_matches_oracle(totals[2], 32 * 62.0, rtol=1e-12)

    def test_the_layer_is_above_the_ground(self) -> None:
        """Raising the ground, the lid and the receptor together changes nothing."""
        low = _surface(turbulence=Homogeneous())
        high = _surface(
            RECEPTOR.at[2].add(1000.0),
            turbulence=Homogeneous(),
            walls=Walls(1000.0, 1050.0),
        )
        assert low.residence.sum() > 0.0
        assert_matches_oracle(high.residence, low.residence, rtol=1e-12)

    def test_it_is_a_backward_run(self) -> None:
        """The same as the loop, spelled out: the key order is the issue's."""
        key_run, key_velocity = jax.random.split(KEY)
        state = in_box(RECEPTOR, RECEPTOR + 1.0, 32, release_time=300.0, key=KEY)
        position = jnp.broadcast_to(RECEPTOR, (32, 3))
        state = type(state)(
            position, state.velocity, state.mass, state.release_time, state.active
        )
        state = with_turbulent_velocity(state, LinearSigma(), key=key_velocity)
        result = integrate(
            state,
            Langevin(uniform(WIND), LinearSigma(), Walls(0.0, 50.0)),
            t0=300.0,
            duration=-62.0,
            step_size=5.0,
            key=key_run,
            accumulate=Residence(CELLS, where=lambda s: s.position[:, 2] < 25.0),
        )
        assert_matches_oracle(
            _surface().residence, result.accumulated, rtol=1e-12, atol=1e-12
        )

    @pytest.mark.slow
    def test_reciprocity(self) -> None:
        """G2-reciprocity, through `surface`: forward and backward residence agree.

        Two boxes of 50 m, 500 m apart along a 3 m s⁻¹ wind, in homogeneous
        turbulence over a reflecting ground: the mean time that particles released
        in one spend in the other, 400 s at 1 s.
        """
        runs, size, receptors = 8, 10_000, 500
        closure = Homogeneous(sigma=0.5, timescale=60.0)
        wind = uniform(jnp.array([3.0, 0.0, 0.0]))
        low, high = jnp.array([-25.0, -25.0, 0.0]), jnp.array([25.0, 25.0, 50.0])
        offset = jnp.array([500.0, 0.0, 0.0])
        walls = Walls(0.0)
        source = Cells(([-25.0, 25.0], [-25.0, 25.0]))
        target = Cells(([475.0, 525.0], [-25.0, 25.0], [0.0, 50.0]))
        times = {"duration": 400.0, "step_size": 1.0}

        @jax.jit
        def forward(key: jax.Array) -> jax.Array:
            key_start, key_velocity, key_run = jax.random.split(key, 3)
            state = in_box(low, high, size, key=key_start)
            state = with_turbulent_velocity(state, closure, key=key_velocity)
            result = integrate(
                state,
                Langevin(wind, closure, walls),
                t0=0.0,
                key=key_run,
                accumulate=Residence(target),
                **times,
            )
            return result.accumulated.sum() / size

        @jax.jit
        def backward(key: jax.Array) -> jax.Array:
            key_start, key_run = jax.random.split(key)
            points = in_box(low + offset, high + offset, receptors, key=key_start)

            def seen(receptor: jax.Array, key: jax.Array) -> jax.Array:
                footprint = surface(
                    receptor,
                    wind,
                    closure,
                    source,
                    n_particles=size // receptors,
                    receptor_time=400.0,
                    surface_depth=50.0,
                    air_density=1.2,
                    walls=walls,
                    key=key,
                    **times,
                )
                # F ρ h_s, the mean time per particle
                return footprint.sensitivity[0, 0] * 1.2 * 50.0

            keys = jax.random.split(key_run, receptors)
            return jax.vmap(seen)(points.position, keys).mean()

        def estimate(run: Any, seed: int) -> tuple[float, float]:
            keys = jax.random.split(jax.random.key(seed), runs)
            means = np.array([float(run(key)) for key in keys])
            return means.mean(), means.std(ddof=1) / np.sqrt(runs)

        ahead, ahead_error = estimate(forward, 1)
        back, back_error = estimate(backward, 2)
        # a puff that is all still in flight would agree too
        assert ahead > 1.0
        # four combined standard errors, of the replicates
        assert abs(ahead - back) < 4.0 * np.hypot(ahead_error, back_error)
        assert np.hypot(ahead_error, back_error) < 0.05 * ahead

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["nearest", "linear"])
    def test_traces(self, kind: str) -> None:
        """vmap over receptors, receptor times and keys is a Python loop."""
        varying = uniform_in_space(lambda t: WIND + t * jnp.array([0.01, 0.0, 0.0]))
        receptors = RECEPTOR + jnp.array([[0.0, 0, 0], [5.0, 3.0, 2.0], [1.0, 1, 30]])
        times = jnp.array([0.0, 300.0, 1000.0])
        seeds = jnp.arange(3)
        assert_traces(
            lambda receptor, time, seed: _surface(
                receptor,
                velocity=varying,
                receptor_time=time,
                key=jax.random.key(seed),
                kind=kind,
            ),
            receptors,
            times,
            seeds,
        )

    def test_receptor_times_trace_once(self) -> None:
        traces = []
        varying = uniform_in_space(lambda t: WIND + t * jnp.array([0.01, 0.0, 0.0]))

        @jax.jit
        def run(time: jax.Array) -> jax.Array:
            traces.append(None)
            return _surface(velocity=varying, receptor_time=time).sensitivity

        values = [run(jnp.asarray(time)) for time in np.linspace(0.0, 1e4, 50)]
        assert len(traces) == 1
        assert float(jnp.abs(values[0] - values[-1]).max()) > 0.0
        # with no host callback
        traced = str(jax.make_jaxpr(run)(0.0))
        assert "callback" not in traced

    @pytest.mark.slow
    def test_gradients(self) -> None:
        def seen(params: jax.Array) -> jax.Array:
            footprint = _surface(
                jnp.append(params[3:5], 12.0),
                velocity=uniform(params[:3]),
                n_particles=6,
                kind="linear",
                surface_depth=params[5],
                air_density=params[6],
                # a layer whose top no particle crosses
                walls=Walls(0.0, 20.0),
                adjoint=Checkpointed(every=4),
            )
            return footprint.sensitivity

        params = jnp.array([3.0, -1.0, 0.0, 2.0, 1.0, 25.0, 1.2])
        gradient = jax.jacrev(seen)(params)
        assert jnp.isfinite(gradient).all() and (jnp.abs(gradient[..., 0]) > 0).any()
        assert_grad_matches_fd(seen, params)

    def test_the_adjoint_is_the_runs(self) -> None:
        def total(wind: jax.Array, adjoint: Any) -> jax.Array:
            footprint = _surface(
                velocity=uniform(wind), kind="linear", n_particles=8, adjoint=adjoint
            )
            return jnp.sum(footprint.sensitivity**2)

        whole, blocks, cut = (
            jax.grad(total)(WIND, adjoint)
            for adjoint in (None, Checkpointed(every=3), Truncated(2))
        )
        assert_matches_oracle(blocks, whole, rtol=1e-12, atol=1e-12)
        assert float(jnp.abs(cut - whole).max()) > 0.01 * float(jnp.abs(whole).max())

    @pytest.mark.parametrize(
        "dtype", [int, float, np.uint8, np.uint32, np.int64, np.float32, jnp.asarray]
    )
    def test_any_number_is_a_duration(self, dtype: Any) -> None:
        """An unsigned integer has no negative: −62 is 194 as a uint8."""
        # jaxtyping's hook, on in the tests only, refuses all but a float
        footprint = inspect.unwrap(surface)(
            RECEPTOR,
            uniform(WIND),
            LinearSigma(),
            CELLS,
            **(OPTIONS | {"duration": dtype(62)}),
        )
        assert_matches_oracle(footprint.residence, _surface().residence, rtol=1e-12)

    def test_nearest_has_no_gradient_in_the_wind(self) -> None:
        gradient = jax.jacrev(lambda u: _surface(velocity=uniform(u)).sensitivity)(WIND)
        assert (gradient == 0.0).all()

    def test_float32(self) -> None:
        single = _surface(RECEPTOR.astype(jnp.float32), cells=EVERYWHERE)
        assert single.sensitivity.dtype == single.residence.dtype == jnp.float32
        # with edges in float64; and in a trace as well
        traced = jax.jit(lambda point: _surface(point, cells=EVERYWHERE))(
            RECEPTOR.astype(jnp.float32)
        )
        assert traced.sensitivity.dtype == jnp.float32
        assert_matches_oracle(traced.residence, single.residence, rtol=1e-4)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"duration": -60.0}, "'duration' must be a positive, finite number"),
            ({"duration": 0.0}, "'duration' must be a positive, finite number"),
            ({"duration": np.inf}, "'duration' must be a positive, finite number"),
            ({"step_size": 0.0}, "'step_size' must be a positive, finite number"),
            (
                {"receptor": RECEPTOR.at[0].set(jnp.nan)},
                "'receptor' must be finite and between the walls",
            ),
            (
                {"receptor": RECEPTOR.at[2].set(-1.0)},
                "'receptor' must be finite and between the walls",
            ),
            (
                {"receptor": RECEPTOR.at[2].set(50.1)},
                "'receptor' must be finite and between the walls",
            ),
            (
                {"receptor": RECEPTOR.astype(jnp.float32), "receptor_time": 1.7e9},
                "'receptor_time' is too far from its epoch",
            ),
            ({"surface_depth": 0.0}, "'surface_depth' must be positive"),
            ({"air_density": -1.2}, "'air_density' must be positive"),
            ({"air_density": np.nan}, "'air_density' must be positive"),
            ({"n_particles": 0}, "'n_particles' must be an integer of at least 1"),
            ({"cells": Cells(CELLS.edges[:1])}, r"'cells' must have two axes"),
            (
                {"cells": Cells((*CELLS.edges, np.array([0.0, 1.0])))},
                r"'cells' must have two axes",
            ),
        ],
    )
    def test_bad_arguments(self, changes: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            _surface(**changes)

    def test_a_bad_receptor(self) -> None:
        # jaxtyping's hook, on in the tests only, may refuse these first
        for receptor in (jnp.zeros(2), jnp.zeros((4, 3))):
            with pytest.raises((ValidationError, TypeError), match="receptor"):
                _surface(receptor)
        for name in ("receptor_time", "surface_depth", "air_density"):
            with pytest.raises((ValidationError, TypeError), match=name):
                _surface(**{name: jnp.ones(32)})
