"""Particles on the sphere: closed forms, the poles, the date line and masks."""

from __future__ import annotations

from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import SphereRotation
from functax.geometry import Axis, RectilinearGrid
from geophysix.earth import SPHERE, WGS84, great_circle_distance, sphere

from tracerax._src._sphere import to_lonlat
from tracerax.advect import (
    Box,
    Deterministic,
    DeterministicSphere,
    deterministic,
    deterministic_sphere,
    integrate,
)
from tracerax.exceptions import ValidationError
from tracerax.particles import ParticleState, ParticleStep, at_point
from tracerax.velocity import GriddedVelocity, uniform


DAY = 86_400.0
RADIUS = SPHERE.semi_major


def _state(lonlat: list[list[float]], dtype: type = jnp.float64) -> ParticleState:
    position = jnp.asarray(lonlat, dtype=dtype)
    n = position.shape[0]
    return ParticleState(
        position,
        jnp.zeros_like(position),
        jnp.ones(n, dtype),
        jnp.zeros(n, dtype),
        jnp.ones(n, bool),
    )


def _distance(left: jax.Array, right: jax.Array) -> np.ndarray:
    """The great-circle distance between positions [m], in float64."""
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return np.asarray(great_circle_distance(a[:, 0], a[:, 1], b[:, 0], b[:, 1]))


def _solve(state: ParticleState, velocity: Callable, **run: float) -> ParticleState:
    final, _ = deterministic_sphere(state, velocity, **run)
    return final


def _loop(
    state: ParticleState, velocity: Callable, *, n_steps: int, **run: float
) -> ParticleState:
    step = DeterministicSphere(velocity)
    size = abs(run["duration"]) / n_steps
    return integrate(state, step, step_size=size, **run).state


BOTH = pytest.mark.parametrize("move", [_solve, _loop], ids=["solve", "loop"])


# --- solid-body rotation ----------------------------------------------------


@BOTH
@pytest.mark.parametrize(
    ("latitude", "n_steps", "bound"),
    [
        # the starts of the epic's table, by the highest latitude on their path
        pytest.param(10.0, 100, 10.0, id="55.8 degrees"),
        pytest.param(60.0, 100, 10.0, id="74.2 degrees"),
        pytest.param(66.5, 100, 10.0, id="85.8 degrees"),
        pytest.param(67.05, 100, 10.0, id="88.8 degrees"),
        pytest.param(67.08, 400, 0.0065, id="89.8 degrees"),
    ],
)
def test_one_turn_returns_to_the_start(
    move: Callable, latitude: float, n_steps: int, bound: float
) -> None:
    """Gate G3-pole: rotation about an axis tilted by 0.4 rad."""
    state = _state([[0.0, latitude]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    final = move(state, velocity, t0=0.0, duration=DAY, n_steps=n_steps)
    assert _distance(final.position, state.position)[0] < bound


@BOTH
def test_a_path_through_both_poles_returns_to_the_start(move: Callable) -> None:
    """Gate G3-pole: the axis is on the equator, so the path is a meridian."""
    state = _state([[0.0, 60.0], [35.0, -20.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.5 * np.pi)
    final = move(state, velocity, t0=0.0, duration=DAY, n_steps=100)
    assert (_distance(final.position, state.position) < 10.0).all()


@pytest.mark.parametrize("fraction", [0.1, 0.5, -0.3])
def test_rotation_matches_the_closed_form_part_of_the_way(fraction: float) -> None:
    state = _state([[0.0, 60.0], [170.0, 10.0], [-120.0, -70.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    duration = fraction * DAY
    final = _solve(state, velocity, t0=0.0, duration=duration, n_steps=200)
    exact = velocity.exact(state.position, duration)
    assert (_distance(final.position, exact) < 0.1).all()


def test_the_saved_positions_are_on_the_path() -> None:
    state = _state([[170.0, 10.0], [-120.0, -70.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    save_at = jnp.array([0.0, 0.123, 0.5, 1.0]) * DAY
    final, saved = deterministic_sphere(
        state, velocity, t0=0.0, duration=DAY, n_steps=400, save_at=save_at
    )
    assert saved.shape == (4, 2, 2)
    assert jnp.array_equal(saved[0], state.position)
    for offset, position in zip(save_at, saved, strict=True):
        exact = velocity.exact(state.position, offset)
        # between steps the interpolation is third order
        assert (_distance(position, exact) < 1.0).all()
    assert_matches_oracle(saved[-1], final.position, rtol=1e-14)


# --- against the plane, and across the date line ----------------------------


@BOTH
def test_a_degree_near_the_equator_matches_the_plane(move: Callable) -> None:
    state = _state([[10.0, 0.0]])
    current = jnp.array([0.8, 0.6])
    duration = 111_000.0
    final = move(state, uniform(current), t0=0.0, duration=duration, n_steps=20)
    # metres in a frame tangent at the start
    moved = jnp.deg2rad(final.position - state.position) * RADIUS
    assert_matches_oracle(moved[0], current * duration, rtol=1e-4)


@BOTH
def test_a_path_across_the_date_line_is_continuous(move: Callable) -> None:
    state = _state([[179.5, 0.0], [-179.5, 0.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.0)
    east = move(state, velocity, t0=0.0, duration=600.0, n_steps=4)
    west = move(state, velocity, t0=0.0, duration=-600.0, n_steps=4)
    # 600 s of a day is 2.5 degrees
    expected = np.array([[-178.0, -177.0], [177.0, 178.0]])
    found = np.stack([east.position[:, 0], west.position[:, 0]])
    assert_matches_oracle(found, expected, rtol=1e-9)


def test_a_global_grid_is_read_on_both_sides_of_the_date_line() -> None:
    """A periodic axis from 0 to 360 reads longitudes in (−180, 180]."""
    longitude = jnp.arange(0.0, 360.0, 2.0)
    latitude = jnp.arange(-90.0, 91.0, 2.0)
    grid = RectilinearGrid(
        (Axis(jnp.array([0.0])), Axis(latitude), Axis(longitude, period=360.0))
    )
    nodes_lat, nodes_lon = jnp.meshgrid(latitude, longitude, indexing="ij")
    rotation = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    nodes = jnp.column_stack([nodes_lon.ravel(), nodes_lat.ravel()])
    samples = rotation(0.0, nodes).reshape(*nodes_lat.shape, 2)[None]
    gridded = GriddedVelocity.fit((grid, grid), (samples[..., 0], samples[..., 1]))
    state = _state([[178.0, 10.0], [-178.0, 10.0]])
    run = {"t0": 0.0, "duration": 0.25 * DAY, "n_steps": 100}
    read, exact = _solve(state, gridded, **run), _solve(state, rotation, **run)
    # a 2° linear read is within 0.03 % of the velocity
    travelled = _distance(exact.position, state.position)
    assert (_distance(read.position, exact.position) < 1e-3 * travelled).all()


# --- the run ----------------------------------------------------------------


def test_the_loop_and_the_solve_agree() -> None:
    state = _state([[0.0, 60.0], [170.0, 10.0], [-120.0, -70.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    run = {"t0": 5.0, "duration": 0.7 * DAY, "n_steps": 50}
    state = eqx.tree_at(lambda s: s.release_time, state, jnp.full(3, 5.0))
    left, right = _solve(state, velocity, **run), _loop(state, velocity, **run)
    assert (_distance(left.position, right.position) < 1e-6).all()


@BOTH
def test_a_backward_run_undoes_a_forward_one(move: Callable) -> None:
    def unsteady(t: jax.Array, position: jax.Array) -> jax.Array:
        rate = 1.0 + 0.3 * jnp.sin(2.0 * np.pi * jnp.asarray(t) / DAY)
        return rate[..., None] * SphereRotation(2.0 * np.pi / DAY, 0.4)(t, position)

    state = _state([[0.0, 60.0], [170.0, 10.0]])
    run = {"n_steps": 200}
    there = move(state, unsteady, t0=0.0, duration=0.4 * DAY, **run)
    there = eqx.tree_at(lambda s: s.release_time, there, jnp.full(2, 0.4 * DAY))
    back = move(there, unsteady, t0=0.4 * DAY, duration=-0.4 * DAY, **run)
    assert (_distance(back.position, state.position) < 1e-2).all()


def test_a_release_inside_a_step_is_exact_in_the_loop() -> None:
    state = _state([[0.0, 60.0]])
    late = eqx.tree_at(lambda s: s.release_time, state, jnp.array([1_234.0]))
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    run = {"t0": 0.0, "duration": 0.2 * DAY, "n_steps": 40}
    final = _loop(late, velocity, **run)
    exact = velocity.exact(state.position, 0.2 * DAY - 1_234.0)
    assert _distance(final.position, exact)[0] < 0.1


def test_the_solve_refuses_a_release_inside_a_step() -> None:
    state = _state([[0.0, 60.0]])
    late = eqx.tree_at(lambda s: s.release_time, state, jnp.array([1_234.0]))
    with pytest.raises(ValidationError, match="release_time"):
        _solve(late, uniform(jnp.ones(2)), t0=0.0, duration=DAY, n_steps=4)


@BOTH
def test_a_particle_that_does_not_move_is_unchanged_bit_for_bit(
    move: Callable,
) -> None:
    """One is inactive, with no position at all, and one is released too late."""
    state = _state([[0.0, 60.0], [np.nan, np.nan], [190.1, 33.3]])
    state = eqx.tree_at(
        lambda s: (s.active, s.release_time),
        state,
        (jnp.array([True, False, True]), jnp.array([0.0, 0.0, 2.0 * DAY])),
    )
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    run = {"t0": 0.0, "duration": 0.1 * DAY, "n_steps": 10}
    final = move(state, velocity, **run)
    assert np.array_equal(final.position[1:], state.position[1:], equal_nan=True)
    alone = move(_state([[0.0, 60.0]]), velocity, **run)
    assert_matches_oracle(final.position[:1], alone.position, rtol=1e-14)

    def total(rate: jax.Array) -> jax.Array:
        moved = move(state, SphereRotation(rate, tilt=0.4), **run)
        return jnp.nansum(moved.position)

    assert bool(jnp.isfinite(jax.grad(total)(2.0 * np.pi / DAY)))


def test_a_saved_position_before_a_release_is_the_input_one() -> None:
    state = _state([[190.1, 33.3]])
    state = eqx.tree_at(lambda s: s.release_time, state, jnp.array([0.5 * DAY]))
    _, saved = deterministic_sphere(
        state,
        uniform(jnp.ones(2)),
        t0=0.0,
        duration=DAY,
        n_steps=4,
        save_at=jnp.array([0.0, 0.25, 0.5, 1.0]) * DAY,
    )
    assert jnp.array_equal(saved[:3, 0], jnp.broadcast_to(state.position[0], (3, 2)))
    assert bool(saved[3, 0, 0] < 180.0)


def test_a_domain_in_longitude_and_latitude_stops_a_particle() -> None:
    state = _state([[0.0, 0.0]])
    step = DeterministicSphere(uniform(jnp.array([10.0, 0.0])))
    box = Box([-5.0, -5.0], [5.0, 5.0])
    result = integrate(state, step, t0=0.0, duration=DAY, step_size=600.0, domain=box)
    assert not bool(result.state.active[0])
    assert 5.0 <= float(result.state.position[0, 0]) < 5.1


# --- transforms, gradients and precision ------------------------------------


@BOTH
def test_transforms_agree(move: Callable) -> None:
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)

    def final(position: jax.Array, t0: jax.Array) -> jax.Array:
        state = eqx.tree_at(lambda s: s.position, _state([[0.0, 0.0]] * 2), position)
        state = eqx.tree_at(lambda s: s.release_time, state, jnp.full(2, t0))
        return move(state, velocity, t0=t0, duration=3_600.0, n_steps=3).position

    positions = jnp.array([[[0.0, 60.0], [179.9, 5.0]], [[12.0, -80.0], [5.0, 1.0]]])
    assert_traces(final, positions, jnp.array([0.0, 50.0]), rtol=1e-11)


@BOTH
def test_gradients_match_finite_differences(move: Callable) -> None:
    def final(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
        position, rate = inputs
        state = eqx.tree_at(lambda s: s.position, _state([[0.0, 0.0]] * 2), position)
        velocity = SphereRotation(rate * 2.0 * np.pi / DAY, tilt=0.4)
        return move(state, velocity, t0=0.0, duration=0.2 * DAY, n_steps=5).position

    start = jnp.array([[0.0, 60.0], [100.0, -40.0]])
    assert_grad_matches_fd(final, (start, jnp.array(1.0)), eps=1e-5, rtol=1e-6)


def test_forward_mode_matches_reverse_mode() -> None:
    def longitude(latitude: jax.Array, adjoint: dfx.AbstractAdjoint) -> jax.Array:
        state = _state([[0.0, 0.0]])
        state = eqx.tree_at(lambda s: s.position, state, state.position + latitude)
        velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
        run = {"t0": 0.0, "duration": 0.2 * DAY, "n_steps": 5}
        final, _ = deterministic_sphere(state, velocity, adjoint=adjoint, **run)
        return final.position[0, 0]

    forward = jax.jacfwd(longitude)(20.0, dfx.ForwardMode())
    reverse = jax.grad(longitude)(20.0, dfx.RecursiveCheckpointAdjoint())
    assert_matches_oracle(forward, reverse, rtol=1e-12)


@BOTH
def test_float32_stays_float32_and_within_its_resolution(move: Callable) -> None:
    state = _state([[0.0, 66.5], [179.0, 10.0]], jnp.float32)
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    final = move(state, velocity, t0=0.0, duration=DAY, n_steps=100)
    assert final.position.dtype == jnp.float32
    # 100 steps of rounding at 1.7 m
    assert (_distance(final.position, state.position) < 50.0).all()


def test_another_radius_moves_a_particle_by_another_angle() -> None:
    state = _state([[0.0, 0.0]])
    run = {"t0": 0.0, "duration": 1_000.0, "n_steps": 200}
    east = uniform(jnp.array([1.0, 0.0]))
    final, _ = deterministic_sphere(state, east, earth=sphere(1_000.0), **run)
    # one radian
    assert_matches_oracle(final.position[0, 0], np.rad2deg(1.0), rtol=1e-9)


# --- what is refused --------------------------------------------------------


def test_the_step_is_a_particle_step() -> None:
    step = DeterministicSphere(uniform(jnp.ones(2)))
    assert isinstance(step, ParticleStep)
    assert step.noise_shape(at_point(jnp.zeros(2), 1)) is None


@BOTH
@pytest.mark.parametrize(
    ("lonlat", "match"),
    [
        ([[0.0, 1.0, 2.0]], "longitude and latitude"),
        ([[45.0, 120.0]], "latitudes between"),
        ([[45.0, -90.5], [0.0, np.nan]], "latitudes between"),
    ],
)
def test_positions_that_are_not_longitude_and_latitude_are_refused(
    move: Callable, lonlat: list[list[float]], match: str
) -> None:
    width = len(lonlat[0])
    with pytest.raises(ValidationError, match=match):
        move(
            _state(lonlat),
            uniform(jnp.ones(width)),
            t0=0.0,
            duration=1.0,
            n_steps=1,
        )


def test_an_ellipsoid_is_refused() -> None:
    velocity = uniform(jnp.ones(2))
    with pytest.raises(ValidationError, match="must be a sphere"):
        DeterministicSphere(velocity, earth=WGS84)
    with pytest.raises(ValidationError, match="must be a sphere"):
        deterministic_sphere(
            _state([[0.0, 0.0]]), velocity, t0=0.0, duration=1.0, n_steps=1, earth=WGS84
        )


def test_an_implicit_solver_is_refused_by_the_step() -> None:
    with pytest.raises((ValidationError, TypeError), match="solver"):
        DeterministicSphere(uniform(jnp.ones(2)), solver=dfx.ImplicitEuler())


@pytest.mark.parametrize("duration", [0.0, np.inf])
def test_a_duration_that_is_not_a_length_is_refused(duration: float) -> None:
    with pytest.raises(ValidationError, match="duration"):
        _solve(
            _state([[0.0, 0.0]]),
            uniform(jnp.ones(2)),
            t0=0.0,
            duration=duration,
            n_steps=1,
        )


@pytest.mark.parametrize("solver", [dfx.Heun(), dfx.Bosh3(), dfx.Tsit5(), dfx.Dopri8()])
def test_any_explicit_solver_runs_and_agrees_with_the_loop(
    solver: dfx.AbstractERK,
) -> None:
    """Solvers that reuse their last stage are told of the normalised vector."""
    state = _state([[0.0, 60.0], [170.0, 10.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    final, _ = deterministic_sphere(
        state, velocity, t0=0.0, duration=0.3 * DAY, n_steps=20, solver=solver
    )
    step = DeterministicSphere(velocity, solver)
    result = integrate(
        state, step, t0=0.0, duration=0.3 * DAY, step_size=0.3 * DAY / 20
    )
    assert (_distance(final.position, result.state.position) < 1e-6).all()


# --- at the poles, and what the review found -----------------------------------


@BOTH
@pytest.mark.parametrize("latitude", [90.0, -90.0])
def test_a_path_from_a_pole_is_right_and_so_is_its_gradient(
    move: Callable, latitude: float
) -> None:
    state = _state([[33.0, latitude]])

    def after(rate: jax.Array) -> jax.Array:
        velocity = SphereRotation(rate * 2.0 * np.pi / DAY, tilt=0.4)
        return move(state, velocity, t0=0.0, duration=0.1 * DAY, n_steps=20).position[0]

    exact = SphereRotation(2.0 * np.pi / DAY, 0.4).exact(state.position, 0.1 * DAY)
    assert _distance(after(jnp.array(1.0))[None], exact)[0] < 1.0
    assert_grad_matches_fd(after, jnp.array(1.0), eps=1e-5, rtol=1e-5)


def test_longitudes_are_half_open_as_cells_are() -> None:
    vector = jnp.array([[-1.0, 0.0, 0.0], [-1.0, -0.0, 0.0], [-1.0, 1e-9, 0.0]])
    longitude = np.asarray(to_lonlat(vector)[:, 0])
    assert longitude[:2].tolist() == [-180.0, -180.0]
    assert 179.9 < longitude[2] < 180.0


@BOTH
def test_float32_steps_of_hundreds_of_metres_are_good_to_a_per_cent(
    move: Callable,
) -> None:
    """Ten days at 0.5 m/s in steps of 450 m, against the same in float64."""
    run = {"t0": 0.0, "duration": 10 * DAY, "n_steps": 960}
    east = uniform(jnp.array([0.5, 0.0]))
    start = [[100.1, 30.2], [-150.3, -10.7], [179.0, 60.1]]
    single = move(_state(start, jnp.float32), east, **run)
    double = move(_state(start), east, **run)
    assert (_distance(single.position, double.position) < 0.01 * 0.5 * 10 * DAY).all()


def test_an_implicit_solver_solves_on_the_sphere() -> None:
    state = _state([[0.0, 60.0]])
    velocity = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    solver = dfx.Kvaerno3(root_finder=dfx.VeryChord(rtol=1e-10, atol=1e-10))
    final, _ = deterministic_sphere(
        state, velocity, t0=0.0, duration=0.1 * DAY, n_steps=40, solver=solver
    )
    exact = velocity.exact(state.position, 0.1 * DAY)
    assert _distance(final.position, exact)[0] < 1.0


@pytest.mark.parametrize("radius", [6.4e6, np.float64(6.4e6), jnp.array(6.4e6)])
def test_the_radius_may_be_any_number(radius: float) -> None:
    """Also one that jit traces, in float32 positions."""
    earth = sphere(radius)
    east = uniform(jnp.array([1e3, 0.0]))
    state = _state([[0.0, 0.0]], jnp.float32)
    step = DeterministicSphere(east, earth=earth)
    run = eqx.filter_jit(integrate)
    final = run(state, step, t0=0.0, duration=DAY, step_size=600.0).state
    assert final.position.dtype == jnp.float32
    expected = np.rad2deg(1e3 * DAY / 6.4e6)
    found = float(final.position[0, 0]) % 360.0
    assert found == pytest.approx(expected % 360.0, rel=1e-5)


def test_a_start_that_float32_cannot_resolve_is_refused() -> None:
    state = _state([[0.0, 0.0]], jnp.float32)
    east = uniform(jnp.array([1.0, 0.0]))
    run = {"t0": 1.7e9, "duration": 900.0, "n_steps": 9}
    for move in (deterministic_sphere, deterministic):
        with pytest.raises(ValidationError, match="'t0' is too far"):
            move(state, east, **run)


@BOTH
def test_what_an_inactive_particle_holds_is_not_looked_at(move: Callable) -> None:
    state = _state([[0.0, 60.0], [1e30, 1e30]])
    state = eqx.tree_at(lambda s: s.active, state, jnp.array([True, False]))
    final = move(state, uniform(jnp.ones(2)), t0=0.0, duration=DAY, n_steps=24)
    assert jnp.array_equal(final.position[1], state.position[1])


def test_the_plane_step_is_not_the_sphere_step() -> None:
    """Read as degrees, a plane step moves by 1° for each metre."""
    state = _state([[0.0, 0.0]])
    east = uniform(jnp.array([1.0, 0.0]))
    run = {"t0": 0.0, "duration": 1.0, "step_size": 1.0}
    plane = integrate(state, Deterministic(east), **run).state.position
    globe = integrate(state, DeterministicSphere(east), **run).state.position
    assert float(plane[0, 0] / globe[0, 0]) == pytest.approx(111_195.0, rel=1e-4)
