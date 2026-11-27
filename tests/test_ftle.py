"""Finite-time Lyapunov exponents: closed forms, the double gyre, the sphere."""

from __future__ import annotations

import inspect
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import DoubleGyre, SphereRotation
from functax.geometry import Axis, RectilinearGrid
from functax.interp import Cubic, Linear
from geophysix.earth import SPHERE, WGS84

from tracerax import ValidationError
from tracerax.advect import Deterministic, integrate
from tracerax.analysis import ftle
from tracerax.particles import ParticleState
from tracerax.velocity import GriddedVelocity, uniform


GYRE = DoubleGyre()
DAY = 86_400.0


def _gyre_seeds() -> jax.Array:
    x, y = jnp.linspace(0.05, 1.95, 24), jnp.linspace(0.05, 0.95, 12)
    grid = jnp.meshgrid(x, y, indexing="ij")
    return jnp.column_stack([axis.ravel() for axis in grid])


def _end(
    state: ParticleState, velocity: Any, *, duration: float, n_steps: int, **run: Any
) -> jax.Array:
    """The positions after ``n_steps`` equal steps of the deterministic step."""
    size = abs(duration) / n_steps
    result = integrate(
        state, Deterministic(velocity), duration=duration, step_size=size, **run
    )
    return result.state.position


def _gyre_exponents(velocity: Any = GYRE, **options: Any) -> jax.Array:
    run = {"t0": 0.0, "duration": 15.0, "n_steps": 150, **options}
    return jax.jit(lambda seeds: ftle(seeds, velocity, **run))(_gyre_seeds())


def _central_differences(velocity: Any, seeds: jax.Array, **run: Any) -> np.ndarray:
    """Λ from the flow map at four neighbours of each seed."""

    def flow(position: jax.Array) -> np.ndarray:
        n = position.shape[0]
        state = ParticleState(
            position,
            jnp.zeros_like(position),
            jnp.ones(n),
            jnp.zeros(n),
            jnp.ones(n, bool),
        )
        return np.asarray(_end(state, velocity, **run))

    step = 1e-5
    columns = []
    for axis in np.eye(seeds.shape[1]):
        columns.append(flow(seeds + step * axis) - flow(seeds - step * axis))
    jacobian = np.stack(columns, axis=-1) / (2.0 * step)
    stretch = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
    return np.log(stretch) / abs(run["duration"])


# --- closed forms on the plane ----------------------------------------------


@pytest.mark.parametrize("n_windows", [1, 5])
@pytest.mark.parametrize("duration", [40.0, -40.0])
def test_a_shear_has_a_closed_form(n_windows: int, duration: float) -> None:
    """Gate G3-ftle: u = (γ y, 0), whose flow map is a shear by s = γT."""
    rate = 0.05

    def shear(t: Any, position: jax.Array) -> jax.Array:
        return rate * position[:, ::-1] * jnp.array([1.0, 0.0])

    seeds = jnp.array([[0.0, 0.0], [10.0, -3.0]])
    found = ftle(
        seeds, shear, t0=0.0, duration=duration, n_steps=10, n_windows=n_windows
    )
    s = rate * abs(duration)
    # λ_max = 1 + s²/2 + s (1 + s²/4)^½
    largest = 1.0 + 0.5 * s**2 + s * np.sqrt(1.0 + 0.25 * s**2)
    expected = np.full(2, 0.5 * np.log(largest) / abs(duration))
    assert_matches_oracle(found, expected, rtol=1e-10)


def test_a_rotation_and_a_translation_do_not_stretch() -> None:
    def rotation(t: Any, position: jax.Array) -> jax.Array:
        return 0.1 * position[:, ::-1] * jnp.array([-1.0, 1.0])

    seeds = jnp.array([[1.0, 2.0]])
    for velocity in (rotation, uniform(jnp.array([2.0, 1.0]))):
        found = ftle(seeds, velocity, t0=0.0, duration=30.0, n_steps=300)
        assert abs(float(found[0])) < 1e-10


def test_three_dimensions_take_the_largest_rate() -> None:
    rates = jnp.array([0.02, -0.05, 0.03])

    def strain(t: Any, position: jax.Array) -> jax.Array:
        return rates * position

    seeds = jnp.array([[1.0, 2.0, 3.0]])
    found = ftle(seeds, strain, t0=0.0, duration=50.0, n_steps=100)
    assert_matches_oracle(found, np.array([0.03]), rtol=1e-8)
    # backward, what contracted stretches
    found = ftle(seeds, strain, t0=0.0, duration=-50.0, n_steps=100)
    assert_matches_oracle(found, np.array([0.05]), rtol=1e-8)


# --- the double gyre --------------------------------------------------------


def test_the_double_gyre_matches_central_differences() -> None:
    """Gate G3-ftle: the range of the exponents, and the flow map's differences."""
    found = np.asarray(_gyre_exponents())
    assert f"{found.min():.3f} {found.max():.3f}" == "0.009 0.463"
    run = {"t0": 0.0, "duration": 15.0, "n_steps": 150}
    expected = _central_differences(GYRE, _gyre_seeds(), **run)
    # the differences themselves are good to 2e-4 of the largest Jacobian
    assert_matches_oracle(found, expected, rtol=0.0, atol=2e-4)


def test_windows_do_not_change_the_exponent() -> None:
    """Gate G3-ftle: 1 and 5 windows agree to 1e-10 in float64."""
    one, five = _gyre_exponents(), _gyre_exponents(n_windows=5)
    assert_matches_oracle(five, one, rtol=0.0, atol=1e-10)


def test_a_flow_without_divergence_stretches_the_same_backward() -> None:
    """det ∇Φ = 1, so the two singular values are inverse: Λ₋ at Φ(x₀) is Λ₊."""
    seeds = _gyre_seeds()
    n = seeds.shape[0]
    state = ParticleState(
        seeds, jnp.zeros_like(seeds), jnp.ones(n), jnp.zeros(n), jnp.ones(n, bool)
    )
    run = {"duration": 15.0, "n_steps": 300}
    end = _end(state, GYRE, t0=0.0, **run)
    forward = ftle(seeds, GYRE, t0=0.0, **run)
    backward = ftle(end, GYRE, t0=15.0, duration=-15.0, n_steps=300)
    assert_matches_oracle(backward, forward, rtol=0.0, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scheme", "median", "largest"),
    [
        pytest.param(Linear(), 9e-4, 0.065, id="linear"),
        pytest.param(Cubic(), 2e-4, 0.11, id="cubic"),
    ],
)
def test_the_double_gyre_read_from_a_grid(
    scheme: Any, median: float, largest: float
) -> None:
    """The errors of an 81 × 41 grid at 61 times, the largest on the ridges."""
    times = jnp.linspace(0.0, 15.0, 61)
    x, y = jnp.linspace(0.0, 2.0, 81), jnp.linspace(0.0, 1.0, 41)
    nodes = jnp.stack(jnp.meshgrid(x, y), axis=-1)
    samples = jnp.stack([GYRE(t, nodes) for t in times])
    grid = RectilinearGrid((Axis(times), Axis(y), Axis(x)))
    velocity = GriddedVelocity.fit(
        (grid, grid), (samples[..., 0], samples[..., 1]), scheme=scheme
    )
    error = np.abs(np.asarray(_gyre_exponents(velocity) - _gyre_exponents()))
    assert np.median(error) < median
    assert error.max() < largest


# --- the sphere -------------------------------------------------------------


def test_a_rotation_of_the_sphere_does_not_stretch() -> None:
    """Through both poles and the date line."""
    seeds = jnp.array([[0.0, 60.0], [179.9, 5.0], [35.0, -89.99], [0.0, 89.0]])
    for tilt in (0.4, 0.5 * np.pi):
        flow = SphereRotation(2.0 * np.pi / DAY, tilt)
        found = ftle(
            seeds, flow, t0=0.0, duration=DAY, n_steps=200, n_windows=4, earth=SPHERE
        )
        # the step's own error, 1e-9 of a turn
        assert (np.abs(np.asarray(found)) * DAY < 1e-6).all()


def test_a_zonal_shear_on_the_sphere_has_a_closed_form() -> None:
    """dλ/dt = Ω₀ sin² φ: the flow map is a shear by s = cos φ T Ω₀ sin 2φ."""
    rate = 2.0 * np.pi / (10.0 * DAY)

    def zonal(t: Any, lonlat: jax.Array) -> jax.Array:
        latitude = jnp.deg2rad(lonlat[:, 1])
        angular = rate * jnp.sin(latitude) ** 2
        east = SPHERE.semi_major * jnp.cos(latitude) * angular
        return jnp.column_stack([east, jnp.zeros_like(east)])

    latitudes = np.array([-60.0, 0.5, 30.0, 45.0, 80.0, 89.9])
    seeds = jnp.column_stack([jnp.linspace(-170.0, 170.0, 6), jnp.asarray(latitudes)])
    duration = 20.0 * DAY
    found = ftle(
        seeds, zonal, t0=0.0, duration=duration, n_steps=500, n_windows=2, earth=SPHERE
    )
    phi = np.deg2rad(latitudes)
    s = np.abs(np.cos(phi) * duration * rate * np.sin(2.0 * phi))
    largest = 1.0 + 0.5 * s**2 + s * np.sqrt(1.0 + 0.25 * s**2)
    expected = 0.5 * np.log(largest) / duration
    assert_matches_oracle(found * duration, expected * duration, rtol=0.0, atol=1e-6)


def test_the_sphere_is_locally_the_plane() -> None:
    """A strain field 1 km wide at 40° N, in metres about the first seed."""
    radius = SPHERE.semi_major
    centre = jnp.array([12.0, 40.0])
    shrink = jnp.cos(jnp.deg2rad(centre[1]))

    def metres(lonlat: jax.Array) -> jax.Array:
        offset = jnp.deg2rad(lonlat - centre) * radius
        return offset * jnp.array([shrink, 1.0])

    def strain(t: Any, position: jax.Array) -> jax.Array:
        x, y = position[:, 0], position[:, 1]
        return 1e-4 * jnp.column_stack([x + 0.5 * y, 0.3 * x - y])

    run = {"t0": 0.0, "duration": 10_000.0, "n_steps": 50}
    seeds = centre + jnp.array([[0.0, 0.0], [1e-3, -1e-3]])
    globe = ftle(
        seeds, lambda t, lonlat: strain(t, metres(lonlat)), earth=SPHERE, **run
    )
    plane = ftle(metres(seeds), strain, **run)
    assert_matches_oracle(globe, plane, rtol=1e-3)


# --- transforms, gradients and precision ------------------------------------


def test_transforms_agree() -> None:
    def exponents(seeds: jax.Array, t0: jax.Array) -> jax.Array:
        return ftle(seeds, GYRE, t0=t0, duration=2.0, n_steps=10, n_windows=2)

    seeds = jnp.array([[[0.3, 0.4], [1.2, 0.7]], [[1.8, 0.1], [0.9, 0.5]]])
    assert_traces(exponents, seeds, jnp.array([0.0, 3.0]), rtol=1e-10)


def test_one_trace_serves_every_start_time() -> None:
    count = 0

    @jax.jit
    def exponents(t0: jax.Array) -> jax.Array:
        nonlocal count
        count += 1
        return ftle(_gyre_seeds()[:2], GYRE, t0=t0, duration=2.0, n_steps=4)

    values = [exponents(t0) for t0 in (0.0, 1.0, 2.5)]
    assert count == 1
    assert len({float(value[0]) for value in values}) == 3


def test_a_reverse_mode_gradient() -> None:
    def exponent(amplitude: jax.Array) -> jax.Array:
        return ftle(
            jnp.array([[0.3, 0.4], [1.2, 0.7]]),
            DoubleGyre(amplitude=amplitude),
            t0=0.0,
            duration=2.0,
            n_steps=10,
            n_windows=2,
        )

    assert_grad_matches_fd(exponent, jnp.array(0.1), eps=1e-6, rtol=1e-6)


def test_a_seed_that_is_nan_spoils_no_other() -> None:
    seeds = jnp.array([[0.3, 0.4], [jnp.nan, 0.5], [1.2, 0.7]])
    run = {"t0": 0.0, "duration": 2.0, "n_steps": 10}
    found = ftle(seeds, GYRE, **run)
    assert np.isnan(found[1])
    # also where the velocity does not depend on the position
    still = ftle(seeds, uniform(jnp.zeros(2)), **run)
    assert_matches_oracle(still, np.array([0.0, np.nan, 0.0]), rtol=0.0)
    alone = ftle(seeds[jnp.array([0, 2])], GYRE, **run)
    assert_matches_oracle(found[jnp.array([0, 2])], alone, rtol=1e-13)


def test_a_seed_that_is_nan_spoils_no_gradient() -> None:
    seeds = jnp.array([[0.2, 0.3], [jnp.nan, 0.3]])

    def exponents(rate: jax.Array, seeds: jax.Array) -> jax.Array:
        def velocity(t: Any, position: jax.Array) -> jax.Array:
            return rate * jnp.sin(position) * jnp.array([1.0, -1.0])

        run = {"t0": 0.0, "duration": 2.0, "n_steps": 10}
        return ftle(seeds, velocity, **run)

    beside = jax.grad(lambda rate: exponents(rate, seeds)[0])(0.5)
    alone = jax.grad(lambda rate: exponents(rate, seeds[:1])[0])(0.5)
    assert_matches_oracle(beside, float(alone), rtol=1e-12)
    # and with no valid seed at all
    none = exponents(jnp.array(0.5), jnp.full((2, 2), jnp.nan))
    assert np.isnan(none).all()


def test_float32_needs_windows_where_the_stretch_overflows() -> None:
    """e^60 is safe, though its square overflows float32, and e^120 is not."""

    def strain(t: Any, position: jax.Array) -> jax.Array:
        return 0.6 * position * jnp.array([1.0, -1.0], position.dtype)

    seeds = jnp.zeros((1, 2), jnp.float32)
    exponent = np.array([0.6], np.float32)
    one = ftle(seeds, strain, t0=0.0, duration=100.0, n_steps=1000)
    assert one.dtype == jnp.float32
    assert_matches_oracle(one, exponent, rtol=1e-4)
    run = {"t0": 0.0, "duration": 200.0, "n_steps": 2000}
    assert not np.isfinite(ftle(seeds, strain, **run)[0])
    assert_matches_oracle(ftle(seeds, strain, n_windows=10, **run), exponent, rtol=1e-4)


def test_float32_keeps_a_flow_that_contracts_everywhere() -> None:
    def sink(t: Any, position: jax.Array) -> jax.Array:
        return -0.6 * position

    seeds = jnp.ones((1, 2), jnp.float32)
    found = ftle(seeds, sink, t0=0.0, duration=100.0, n_steps=1000)
    assert_matches_oracle(found, np.array([-0.6], np.float32), rtol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_a_seed_at_a_pole(dtype: type) -> None:
    """A strain of a = 1e-5 s⁻¹ about a pole where the velocity is zero."""
    rate = 1e-5

    def strain(t: Any, lonlat: jax.Array) -> jax.Array:
        # A r with A = diag(a, −a, 0), along the sphere
        longitude, latitude = jnp.deg2rad(lonlat).T
        x = jnp.cos(latitude) * jnp.cos(longitude)
        y = jnp.cos(latitude) * jnp.sin(longitude)
        east = -x * jnp.sin(longitude) - y * jnp.cos(longitude)
        north = jnp.sin(latitude) * (-x * jnp.cos(longitude) + y * jnp.sin(longitude))
        scale = rate * SPHERE.semi_major
        return (scale * jnp.column_stack([east, north])).astype(lonlat.dtype)

    seeds = jnp.array([[0.0, 90.0], [123.0, 90.0], [-7.0, -90.0]], dtype)
    found = ftle(seeds, strain, t0=0.0, duration=140_000.0, n_steps=100, earth=SPHERE)
    rtol = 10.0 * np.sqrt(np.finfo(dtype).eps)
    assert_matches_oracle(found, np.full(3, rate, dtype), rtol=rtol)


def test_integer_seeds_and_seeds_of_another_shape() -> None:
    bare = inspect.unwrap(ftle)
    run = {"t0": 0.0, "duration": 2.0, "n_steps": 10}
    found = bare(jnp.array([[1, 0]]), GYRE, **run)
    assert_matches_oracle(found, ftle(jnp.array([[1.0, 0.0]]), GYRE, **run), rtol=0.0)
    with pytest.raises(ValidationError, match="'seeds' must be"):
        bare(jnp.zeros(2), GYRE, **run)


def test_traced_latitudes_beyond_a_pole_are_refused() -> None:
    flow = SphereRotation(1e-5, tilt=0.4)

    @eqx.filter_jit
    def exponents(seeds: jax.Array) -> jax.Array:
        return ftle(seeds, flow, t0=0.0, duration=60.0, n_steps=2, earth=SPHERE)

    with pytest.raises(RuntimeError, match="'seeds' must hold latitudes"):
        exponents(jnp.array([[45.0, 120.0]]))


def test_a_start_that_float32_cannot_resolve_is_refused() -> None:
    seeds = jnp.array([[0.3, 0.4]], jnp.float32)
    with pytest.raises(ValidationError, match="'t0' is too far"):
        ftle(seeds, GYRE, t0=1.7e9, duration=15.0, n_steps=150)


def test_another_solver() -> None:
    exact = _gyre_exponents(solver=dfx.Dopri8())
    found = _gyre_exponents(solver=dfx.Tsit5())
    assert_matches_oracle(found, exact, rtol=0.0, atol=1e-6)
    assert_matches_oracle(_gyre_exponents(), exact, rtol=0.0, atol=1e-4)


# --- what is refused --------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"duration": 0.0}, "duration"),
        ({"n_steps": 0}, "n_steps"),
        ({"n_windows": 0}, "n_windows"),
        ({"n_windows": 3}, "divide"),
        ({"seeds": jnp.zeros((2, 4))}, "2 or 3 columns"),
        ({"seeds": jnp.zeros((2, 3)), "earth": SPHERE}, "longitude and latitude"),
        ({"seeds": jnp.array([[0.0, 95.0]]), "earth": SPHERE}, "latitudes"),
        ({"earth": WGS84}, "sphere"),
    ],
)
def test_invalid(changes: dict, match: str) -> None:
    options = {"t0": 0.0, "duration": 1.0, "n_steps": 10, **changes}
    seeds = options.pop("seeds", jnp.zeros((1, 2)))
    with pytest.raises(ValidationError, match=match):
        ftle(seeds, uniform(jnp.zeros(seeds.shape[-1])[:3]), **options)
