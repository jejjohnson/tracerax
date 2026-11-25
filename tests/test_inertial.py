"""Inertial particles: closed forms, a reference solution, the slow manifold."""

from __future__ import annotations

import functools
import itertools
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import SphereRotation
from geophysix.earth import SPHERE, WGS84, sphere

from tracerax import ValidationError
from tracerax._src.advect.inertial import _phi
from tracerax._src.velocity.derivatives import velocity_derivatives
from tracerax.advect import (
    Deterministic,
    DeterministicSphere,
    Inertial,
    SlowManifold,
    integrate,
)
from tracerax.particles import ParticleState, ParticleStep, at_point
from tracerax.velocity import uniform


DAY = 86_400.0
CORIOLIS = 1e-4
WIND = uniform(jnp.array([8.0, 3.0]))
STILL = uniform(jnp.zeros(2))
START = jnp.array([1.2e4, 0.7e4])


def _cells(t: Any, position: jax.Array) -> jax.Array:
    """Unsteady cells of 50 km in a flow of 0.5 m/s, with no divergence."""
    wavenumber = 2.0 * np.pi / 5e4
    x = wavenumber * position[..., 0] + 1e-5 * t
    y = wavenumber * position[..., 1]
    u, v = -jnp.sin(x) * jnp.cos(y), jnp.cos(x) * jnp.sin(y)
    return 0.5 * jnp.stack([u, v], axis=-1)


def _inertial(response_time: float, **changes: Any) -> Inertial:
    options = {"buoyancy": 0.6, "air_weight": 0.05, "coriolis": CORIOLIS, **changes}
    return Inertial(_cells, WIND, response_time=response_time, **options)


def _state(position: Any = START, velocity: Any = 0.0) -> ParticleState:
    state = at_point(position, 1)
    moving = state.velocity + jnp.asarray(velocity)
    return eqx.tree_at(lambda s: s.velocity, state, moving)


@functools.cache
def _reference(response_time: float) -> np.ndarray:
    """x after a day, by an adaptive eighth-order solve of the equation itself.

    The right-hand side is written out here, term by term, and shares no code
    with the step.
    """
    buoyancy, share = 0.6, 0.05

    def water(t: Any, x: jax.Array) -> jax.Array:
        return _cells(t, x)

    def rhs(t: Any, y: jax.Array, args: Any) -> jax.Array:
        x, p = y[:2], y[2:]
        v = water(t, x)
        jac = jax.jacfwd(lambda x: water(t, x))(x)
        material = jax.jacfwd(lambda t: water(t, x))(t) + jac @ v
        vorticity = jac[1, 0] - jac[0, 1]
        u = (1.0 - share) * v + share * jnp.array([8.0, 3.0])
        rotate = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        change = (
            buoyancy * material
            + (u - p) / response_time
            + buoyancy * (CORIOLIS + vorticity / 3.0) * rotate @ v
            - (CORIOLIS + buoyancy * vorticity / 3.0) * rotate @ p
        )
        return jnp.concatenate([p, change])

    start = _inertial(response_time).slow_manifold(0.0, START[None])[0]
    solution = dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        dfx.Dopri8(),
        0.0,
        DAY,
        10.0,
        jnp.concatenate([START, start]),
        stepsize_controller=dfx.PIDController(1e-10, 1e-10),
        max_steps=2_000_000,
    )
    return np.asarray(solution.ys[0][:2])


def _after_a_day(step: Any, state: ParticleState, size: float) -> jax.Array:
    run = eqx.filter_jit(integrate)
    return run(state, step, t0=0.0, duration=DAY, step_size=size).state.position[0]


# --- closed forms -----------------------------------------------------------


def test_without_drag_a_particle_returns_after_an_inertial_period() -> None:
    step = Inertial(
        STILL,
        STILL,
        buoyancy=0.6,
        response_time=jnp.inf,
        air_weight=0.0,
        coriolis=CORIOLIS,
    )
    state = _state(jnp.zeros(2), [0.3, 0.1])
    period = 2.0 * np.pi / CORIOLIS
    result = integrate(
        state, step, t0=0.0, duration=period, step_size=period / 100, save_every=25
    )
    path = result.trajectory[:, 0]
    assert_matches_oracle(path[-1], np.zeros(2), rtol=0.0, atol=2e-7)
    assert_matches_oracle(result.state.velocity, state.velocity, rtol=1e-12)
    # a circle of radius |v| / f: opposite points are a diameter apart
    diameter = 2.0 * np.hypot(0.3, 0.1) / CORIOLIS
    assert_matches_oracle(jnp.linalg.norm(path[2] - path[0]), diameter, rtol=1e-12)
    assert_matches_oracle(jnp.linalg.norm(path[3] - path[1]), diameter, rtol=1e-12)


@pytest.mark.parametrize("size", [60.0, 7_200.0])
def test_a_uniform_flow_gives_the_terminal_velocity_at_any_step(size: float) -> None:
    step = Inertial(
        uniform(jnp.array([0.5, 0.1])),
        WIND,
        buoyancy=0.6,
        response_time=3_600.0,
        air_weight=0.05,
        coriolis=CORIOLIS,
    )
    state = _state()
    run = eqx.filter_jit(integrate)
    final = run(state, step, t0=0.0, duration=3 * DAY, step_size=size).state
    # v_p = −b/λ, in complex form
    water, wind = 0.5 + 0.1j, 8.0 + 3.0j
    rate = -1.0 / 3_600.0 - 1j * CORIOLIS
    forcing = (0.95 * water + 0.05 * wind) / 3_600.0 + 0.6j * CORIOLIS * water
    terminal = -forcing / rate
    expected = np.array([[terminal.real, terminal.imag]])
    assert_matches_oracle(final.velocity, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("size", "bound"), [(900.0, 1.5), (450.0, 0.45), (225.0, 0.13)]
)
def test_a_neutral_tracer_stays_with_the_water(size: float, bound: float) -> None:
    """Gate G3-inertial: R = 1, α = 0 and v_p = v at the start, in metres."""
    solution = dfx.diffeqsolve(
        dfx.ODETerm(lambda t, y, args: _cells(t, y)),
        dfx.Dopri8(),
        0.0,
        DAY,
        60.0,
        START,
        stepsize_controller=dfx.PIDController(1e-11, 1e-11),
        max_steps=100_000,
    )
    step = _inertial(3_600.0, buoyancy=1.0, air_weight=0.0)
    state = _state(velocity=_cells(0.0, START[None]))
    distance = jnp.linalg.norm(_after_a_day(step, state, size) - solution.ys[0])
    assert float(distance) < bound


def test_the_phi_functions_match_their_series_and_their_quotients() -> None:
    """φ₁ and φ₂ at small, middling and large z, against a sum in extended precision."""
    sizes = np.array([1e-14, 1e-6, 1e-2, 0.3, 0.999, 1.001, 3.0, 8.0])
    angles = np.array([np.pi, 2.5, -1.7, -0.5 * np.pi])
    z = np.ravel(sizes[:, None] * np.exp(1j * angles))
    first, second = _phi(jnp.asarray(z))
    # ∑ₖ zᵏ / (k + 1)! and ∑ₖ zᵏ / (k + 2)!, largest term 8⁸/8! = 416
    extended = z.astype(np.clongdouble)
    term, one, two = np.ones_like(extended), 0.0, 0.0
    for k in range(80):
        one = one + term / (k + 1)
        two = two + term / ((k + 1) * (k + 2))
        term = term * extended / (k + 1)
    assert np.max(np.abs(np.asarray(first) - one)) < 1e-15
    assert np.max(np.abs(np.asarray(second) - two)) < 1e-15


# --- the reference ----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("response_time", "bounds"),
    [
        # the epic's table: 1.52, 0.35, 0.085; 0.59, 0.20, 0.059; 1.41, 0.26, 0.056
        (3_600.0, (1.6, 0.4, 0.1)),
        (600.0, (0.65, 0.22, 0.065)),
        (60.0, (1.5, 0.28, 0.06)),
    ],
)
def test_the_midpoint_step_converges_to_the_reference(
    response_time: float, bounds: tuple[float, ...]
) -> None:
    """Gate G3-inertial: the errors fall at least 2.5 times per halving of Δt."""
    step = _inertial(response_time)
    state = step.release(_state(), t0=0.0)
    reference = _reference(response_time)
    errors = [
        float(jnp.linalg.norm(_after_a_day(step, state, size) - reference))
        for size in (900.0, 450.0, 225.0)
    ]
    assert all(error < bound for error, bound in zip(errors, bounds, strict=True))
    assert errors[0] / errors[1] > 2.5
    assert errors[1] / errors[2] > 2.5


@pytest.mark.slow
def test_the_slow_manifold_agrees_to_second_order_in_the_response_time() -> None:
    """The velocities along one orbit differ by O(τ²); the paths by O(τ²) too."""

    def distance(response_time: float) -> float:
        step = _inertial(response_time)
        state = step.release(_state(), t0=0.0)
        inertial = _after_a_day(step, state, 56.25)
        slow = _after_a_day(Deterministic(step.slow_manifold), state, 225.0)
        return float(jnp.linalg.norm(inertial - slow))

    distances = [distance(tau) for tau in (480.0, 240.0, 120.0)]
    ratios = [a / b for a, b in itertools.pairwise(distances)]
    # 4 per halving; the epic's 116 m at 600 s and 1.4 m at 60 s is 10^1.92
    assert all(3.5 < ratio < 4.5 for ratio in ratios)


# --- the slow manifold ------------------------------------------------------


def test_the_slow_manifold_of_a_neutral_tracer_is_the_water() -> None:
    position = jax.random.uniform(jax.random.key(0), (7, 2)) * 5e4
    step = _inertial(3_600.0, buoyancy=1.0, air_weight=0.0)
    velocity = step.slow_manifold(123.0, position)
    assert_matches_oracle(velocity, _cells(123.0, position), rtol=1e-12, atol=1e-13)


def test_the_slow_manifold_runs_backward_under_a_deterministic_step() -> None:
    step = Deterministic(_inertial(600.0).slow_manifold)
    run = {"t0": 0.0, "duration": 6 * 3_600.0, "step_size": 225.0}
    there = integrate(_state(), step, **run).state
    there = eqx.tree_at(lambda s: s.release_time, there, jnp.array([6 * 3_600.0]))
    run = {"t0": 6 * 3_600.0, "duration": -6 * 3_600.0, "step_size": 225.0}
    back = integrate(there, step, **run).state
    assert_matches_oracle(back.position[0], START, rtol=0.0, atol=1e-2)


def test_one_time_per_particle_equals_one_run_each() -> None:
    slow = _inertial(600.0).slow_manifold
    position = jax.random.uniform(jax.random.key(1), (3, 2)) * 5e4
    times = jnp.array([0.0, 4e4, 9e4])
    alone = [slow(t, position[i : i + 1])[0] for i, t in enumerate(times)]
    assert_matches_oracle(slow(times, position), np.stack(alone), rtol=1e-13)


def test_the_covariant_derivatives_of_a_zonal_rotation() -> None:
    """u = ω R cos φ: Dv/Dt = (0, u² tan φ / R) and ω_z = 2 ω sin φ."""
    rate = 1e-5
    lonlat = jnp.array([[0.0, 0.0], [170.0, 45.0], [-20.0, -89.0], [12.0, 89.99]])
    flow = SphereRotation(rate, tilt=0.0)
    found = velocity_derivatives(flow, jnp.zeros(4), lonlat, earth=SPHERE)
    latitude = np.deg2rad(np.asarray(lonlat)[:, 1])
    zonal = rate * SPHERE.semi_major * np.cos(latitude)
    expected = np.stack([zonal, np.zeros(4)], axis=-1)
    assert_matches_oracle(found.value, expected, rtol=1e-12, atol=1e-12)
    material = np.stack(
        [np.zeros(4), zonal**2 * np.tan(latitude) / SPHERE.semi_major], axis=-1
    )
    scale = rate**2 * SPHERE.semi_major
    found_material = found.following(found.value)
    assert_matches_oracle(found_material, material, rtol=0.0, atol=1e-11 * scale)
    vorticity = 2.0 * rate * np.sin(latitude)
    assert_matches_oracle(found.vorticity, vorticity, rtol=0.0, atol=1e-11 * rate)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("latitude", [90.0, -90.0])
def test_the_derivatives_of_a_velocity_are_right_at_a_pole(
    dtype: type, latitude: float
) -> None:
    """A rotation about n has the vorticity 2 ω n · r, and no strain."""
    lonlat = jnp.array([[10.0, latitude], [-170.0, latitude]], dtype)
    flow = SphereRotation(1e-5, tilt=0.4)
    found = velocity_derivatives(flow, jnp.zeros(2, dtype), lonlat, earth=SPHERE)
    rtol = 10.0 * np.sqrt(np.finfo(dtype).eps)
    expected = 2e-5 * np.cos(0.4) * np.sign(latitude)
    assert_matches_oracle(
        found.vorticity, np.full(2, expected, dtype), rtol=0.0, atol=2e-5 * rtol
    )
    strain = found.gradient + jnp.swapaxes(found.gradient, 1, 2)
    assert_matches_oracle(strain, np.zeros((2, 2, 2)), rtol=0.0, atol=2e-5 * rtol)


@pytest.mark.parametrize("radius", [6.4e6, np.float64(6.4e6), jnp.array(6.4e6)])
def test_the_radius_may_be_any_number(radius: Any) -> None:
    lonlat = jnp.array([[10.0, 40.0]], jnp.float32)
    flow = SphereRotation(1e-5, tilt=0.0, radius=6.4e6)
    found = velocity_derivatives(
        flow, jnp.zeros(1, jnp.float32), lonlat, earth=sphere(radius)
    )
    assert found.gradient.dtype == jnp.float32
    expected = 2e-5 * np.sin(np.deg2rad(40.0))
    assert float(found.vorticity[0]) == pytest.approx(expected, rel=1e-4)


def test_the_slow_manifold_on_the_sphere() -> None:
    """Of a neutral tracer it is the water, and f is 2Ω sin φ."""
    lonlat = jnp.array([[0.0, 10.0], [170.0, 45.0], [-20.0, -80.0]])
    flow = SphereRotation(1e-6, tilt=0.4)
    options = {"response_time": 3_600.0, "earth": SPHERE}
    neutral = SlowManifold(flow, STILL, buoyancy=1.0, air_weight=0.0, **options)
    assert_matches_oracle(neutral(0.0, lonlat), flow(0.0, lonlat), rtol=1e-10)
    # in a uniform wind over still water, to first order in τ:
    # v_p = α v_a − τ (D(α v_a)/Dt + f J α v_a)
    slow = SlowManifold(STILL, flow, buoyancy=0.5, air_weight=0.5, **options)
    air = velocity_derivatives(flow, jnp.zeros(3), lonlat, earth=SPHERE)
    coriolis = 2.0 * SPHERE.rotation_rate * np.sin(np.deg2rad(lonlat[:, 1]))
    carried = np.asarray(0.5 * air.value)
    material = np.asarray(0.25 * air.following(air.value))
    rotated = np.stack([-carried[:, 1], carried[:, 0]], axis=-1)
    expected = carried - 3_600.0 * (material + np.asarray(coriolis)[:, None] * rotated)
    assert_matches_oracle(slow(0.0, lonlat), expected, rtol=1e-12)


def test_the_slow_manifold_runs_on_the_sphere() -> None:
    """Under the sphere's step, where a neutral tracer goes with the water."""
    flow = SphereRotation(2.0 * np.pi / DAY, tilt=0.4)
    slow = SlowManifold(
        flow,
        STILL,
        buoyancy=1.0,
        air_weight=0.0,
        response_time=600.0,
        earth=SPHERE,
    )
    state = at_point(jnp.array([170.0, 60.0]), 1)
    run = {"t0": 0.0, "duration": 0.2 * DAY, "step_size": 432.0}
    found = integrate(state, DeterministicSphere(slow), **run).state
    water = integrate(state, DeterministicSphere(flow), **run).state
    assert_matches_oracle(found.position, water.position, rtol=1e-10)


def test_the_sphere_is_locally_the_plane() -> None:
    """At the equator a degree is R π/180 metres, and f is given."""
    metres = float(np.deg2rad(1.0)) * SPHERE.semi_major

    def in_degrees(t: Any, lonlat: jax.Array) -> jax.Array:
        return _cells(t, lonlat * metres)

    parameters = {
        "buoyancy": 0.6,
        "response_time": 600.0,
        "air_weight": 0.05,
        "coriolis": CORIOLIS,
    }
    plane = SlowManifold(_cells, WIND, **parameters)
    globe = SlowManifold(in_degrees, WIND, earth=SPHERE, **parameters)
    position = jnp.array([[1.2e4, 100.0], [-3e4, -50.0]])
    # the metric terms are 1e-5 at 100 m from the equator
    assert_matches_oracle(
        globe(500.0, position / metres), plane(500.0, position), rtol=1e-4
    )


# --- releases and masks -----------------------------------------------------


def test_a_release_puts_each_memory_on_the_slow_manifold_at_its_time() -> None:
    step = _inertial(600.0)
    state = at_point(START, 3)
    times = jnp.array([-500.0, 0.0, 4e4])
    state = eqx.tree_at(lambda s: s.release_time, state, times)
    released = step.release(state, t0=10.0)
    expected = step.slow_manifold(jnp.array([10.0, 10.0, 4e4]), state.position)
    assert_matches_oracle(released.velocity, expected, rtol=1e-14)


def test_a_release_inside_a_step_is_its_own_shorter_step() -> None:
    step = _inertial(600.0)
    state = eqx.tree_at(lambda s: s.release_time, _state(), jnp.array([130.0]))
    state = step.release(state, t0=0.0)
    result = integrate(state, step, t0=0.0, duration=900.0, step_size=450.0).state
    one = step(state, jnp.array([130.0]), jnp.array([320.0]), 1.0, None)
    two = step(one, jnp.array([450.0]), jnp.array([450.0]), 1.0, None)
    assert_matches_oracle(result.position, two.position, rtol=1e-14)
    assert_matches_oracle(result.velocity, two.velocity, rtol=1e-13)


def test_a_masked_particle_is_unchanged_and_spoils_nothing() -> None:
    step = _inertial(600.0)
    state = at_point(START, 3)
    position = state.position.at[1].set(jnp.nan)
    state = eqx.tree_at(
        lambda s: (s.position, s.active, s.release_time),
        state,
        (position, jnp.array([True, False, True]), jnp.array([0.0, 0.0, 1e9])),
    )
    released = step.release(state, t0=0.0)
    assert bool((released.velocity[1] == 0.0).all())
    run = {"t0": 0.0, "duration": 1_800.0, "step_size": 450.0}
    final = integrate(released, step, **run).state
    assert np.array_equal(final.position[1:], position[1:], equal_nan=True)
    assert np.array_equal(final.velocity[1:], released.velocity[1:])
    alone = integrate(step.release(_state(), t0=0.0), step, **run).state
    assert_matches_oracle(final.position[:1], alone.position, rtol=1e-14)

    def total(share: jax.Array) -> jax.Array:
        moving = _inertial(600.0, air_weight=share)
        start = moving.release(state, t0=0.0)
        return jnp.nansum(integrate(start, moving, **run).state.position)

    assert bool(jnp.isfinite(jax.grad(total)(0.05)))


# --- transforms, gradients and precision ------------------------------------


def test_transforms_agree() -> None:
    def final(position: jax.Array, share: jax.Array) -> tuple[jax.Array, jax.Array]:
        step = _inertial(600.0, air_weight=share)
        state = eqx.tree_at(lambda s: s.position, at_point(START, 2), position)
        state = step.release(state, t0=0.0)
        moved = integrate(state, step, t0=0.0, duration=1_350.0, step_size=450.0)
        return moved.state.position, moved.state.velocity

    key = jax.random.key(2)
    positions = 5e4 * jax.random.uniform(key, (3, 2, 2))
    assert_traces(final, positions, jnp.array([0.0, 0.03, 0.1]), rtol=1e-11)


def test_gradients_in_the_parameters_match_finite_differences() -> None:
    def final(parameters: dict[str, jax.Array]) -> jax.Array:
        # scaled, so that one finite-difference step fits the three
        step = _inertial(
            3_600.0 * parameters["response_time"],
            buoyancy=parameters["buoyancy"],
            air_weight=parameters["air_weight"],
        )
        state = step.release(_state(), t0=0.0)
        result = integrate(state, step, t0=0.0, duration=3_600.0, step_size=900.0)
        return result.state.position / 1e3

    values = {"buoyancy": 0.6, "response_time": 1.0, "air_weight": 0.05}
    point = {name: jnp.array(value) for name, value in values.items()}
    assert_grad_matches_fd(final, point, eps=1e-5, rtol=1e-6)


def test_gradients_in_the_start_match_finite_differences() -> None:
    step = _inertial(600.0)

    def final(position: jax.Array) -> jax.Array:
        state = step.release(_state(position), t0=0.0)
        result = integrate(state, step, t0=0.0, duration=3_600.0, step_size=900.0)
        return result.state.position

    assert_grad_matches_fd(final, START, eps=1e-2, rtol=1e-6)


def test_the_gradient_is_finite_where_the_rate_vanishes() -> None:
    """No drag and no rotation: λ = 0, where the closed form divides by it."""

    def final(coriolis: jax.Array) -> jax.Array:
        step = Inertial(
            STILL,
            STILL,
            buoyancy=0.6,
            response_time=jnp.inf,
            air_weight=0.0,
            coriolis=coriolis,
        )
        state = _state(jnp.zeros(2), [0.3, 0.1])
        result = integrate(state, step, t0=0.0, duration=900.0, step_size=450.0)
        return result.state.position[0]

    assert_matches_oracle(final(0.0), np.array([270.0, 90.0]), rtol=1e-13)
    assert_grad_matches_fd(final, jnp.array(0.0), eps=1e-6, rtol=1e-6)


def test_float32_stays_float32() -> None:
    state = jax.tree.map(
        lambda leaf: leaf if leaf.dtype == bool else leaf.astype(jnp.float32), _state()
    )
    step = _inertial(600.0)
    state = step.release(state, t0=0.0)
    final = integrate(state, step, t0=0.0, duration=900.0, step_size=450.0).state
    assert final.position.dtype == final.velocity.dtype == jnp.float32
    exact = integrate(
        step.release(_state(), t0=0.0), step, t0=0.0, duration=900.0, step_size=450.0
    ).state
    assert_matches_oracle(final.position, np.float32(exact.position), rtol=1e-6)


# --- what is refused --------------------------------------------------------


def test_the_step_is_a_particle_step() -> None:
    step = _inertial(600.0)
    assert isinstance(step, ParticleStep)
    assert step.noise_shape(_state()) is None


def test_a_backward_run_is_refused() -> None:
    with pytest.raises(ValidationError, match="slow_manifold"):
        integrate(_state(), _inertial(600.0), t0=0.0, duration=-900.0, step_size=450.0)


def test_three_columns_are_refused() -> None:
    state, step = at_point(jnp.zeros(3), 1), _inertial(600.0)
    with pytest.raises(ValidationError, match="2 columns"):
        step.release(state, t0=0.0)
    with pytest.raises(ValidationError, match="2 columns"):
        integrate(state, step, t0=0.0, duration=900.0, step_size=450.0)
    with pytest.raises(ValidationError, match="2 columns"):
        step.slow_manifold(0.0, state.position)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"buoyancy": 0.0}, "buoyancy"),
        ({"buoyancy": 1.1}, "buoyancy"),
        ({"buoyancy": np.nan}, "buoyancy"),
        ({"air_weight": -0.1}, "air_weight"),
        ({"air_weight": jnp.zeros(3)}, "air_weight"),
        ({"response_time": 0.0}, "response_time"),
        ({"coriolis": jnp.zeros(2)}, "coriolis"),
    ],
)
def test_parameters_out_of_range_are_refused(changes: dict, match: str) -> None:
    options = {"response_time": 600.0, **changes}
    with pytest.raises(ValidationError, match=match):
        _inertial(**options)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("buoyancy", 1.1),
        ("buoyancy", 0.0),
        ("response_time", 0.0),
        ("response_time", -1.0),
        ("air_weight", 1.5),
        ("coriolis", np.nan),
    ],
)
def test_traced_parameters_out_of_range_are_refused(name: str, value: float) -> None:
    @eqx.filter_jit
    def final(parameter: jax.Array) -> jax.Array:
        step = _inertial(**{"response_time": 600.0, name: parameter})
        return integrate(
            _state(), step, t0=0.0, duration=450.0, step_size=450.0
        ).state.position

    with pytest.raises(RuntimeError, match=f"'{name}' must be"):
        final(value)
    # in range, the same program runs
    inside = {"buoyancy": 0.5, "response_time": 60.0, "air_weight": 0.1}
    assert np.isfinite(final(inside.get(name, 1e-4))).all()


def test_a_slow_manifold_needs_a_response_time_a_sphere_and_a_rotation() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _inertial(np.inf).release(_state(), t0=0.0)
    for value in (np.nan, np.inf):
        with pytest.raises(ValidationError, match="coriolis"):
            _inertial(600.0, coriolis=value)
    parameters = {"buoyancy": 0.6, "response_time": 600.0, "air_weight": 0.05}
    with pytest.raises(ValidationError, match="coriolis"):
        SlowManifold(STILL, STILL, **parameters)
    with pytest.raises(ValidationError, match="sphere"):
        SlowManifold(STILL, STILL, earth=WGS84, **parameters)
