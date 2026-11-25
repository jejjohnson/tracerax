"""Absolute and pair dispersion, in the plane and on the sphere."""

from __future__ import annotations

from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from _oracles import SphereRotation, taylor_variance
from geophysix.earth import SPHERE, WGS84, great_circle_distance, sphere

from tracerax import ValidationError
from tracerax.advect import (
    Deterministic,
    DeterministicSphere,
    Langevin,
    Walls,
    integrate,
)
from tracerax.analysis import absolute_dispersion, pair_dispersion
from tracerax.particles import at_point, in_box, with_turbulent_velocity
from tracerax.velocity import uniform


KEY = jax.random.key(0)
DAY = 86_400.0


def _walk(n_times: int = 5, n_particles: int = 6, dim: int = 2) -> jax.Array:
    steps = jax.random.normal(KEY, (n_times, n_particles, dim))
    return jnp.cumsum(steps, axis=0)


def _globe(n_times: int = 5, n_particles: int = 6) -> jax.Array:
    """Positions on both sides of the date line and near the north pole."""
    draws = jax.random.uniform(KEY, (n_times, n_particles, 2))
    lonlat = jnp.array([170.0, 60.0]) + draws * jnp.array([20.0, 29.9])
    longitude = jnp.mod(lonlat[..., 0] + 180.0, 360.0) - 180.0
    return jnp.stack([longitude, lonlat[..., 1]], axis=-1)


# --- closed forms -------------------------------------------------------------


@pytest.mark.slow
def test_homogeneous_turbulence_has_taylors_variance() -> None:
    sigma, timescale, n = 0.5, 60.0, 4_000
    closure = Homogeneous(sigma, timescale)
    key_start, key_run = jax.random.split(KEY)
    state = at_point(jnp.array([0.0, 0.0, 500.0]), n)
    state = with_turbulent_velocity(state, closure, key=key_start)
    step = Langevin(uniform(jnp.zeros(3)), closure, Walls(0.0, 1e3))
    result = eqx.filter_jit(integrate)(
        state,
        step,
        t0=0.0,
        duration=600.0,
        step_size=2.5,
        key=key_run,
        save_every=60,
    )
    # two horizontal axes, which no wall touches
    found = np.asarray(absolute_dispersion(result.trajectory[..., :2]))
    expected = 2.0 * taylor_variance(sigma, timescale, result.offsets)
    assert found[0] == 0.0
    # the mean of N squared distances in two dimensions has a relative standard
    # error of N^−½
    ratio = found[1:] / expected[1:]
    assert (np.abs(ratio - 1.0) < 4.0 / np.sqrt(n)).all()


def test_a_strain_separates_a_pair_exponentially() -> None:
    rate = 0.1

    def strain(t: Any, position: jax.Array) -> jax.Array:
        return rate * position * jnp.array([1.0, -1.0])

    state = in_box(jnp.zeros(2), jnp.ones(2), 8, key=KEY)
    result = integrate(
        state,
        Deterministic(strain),
        t0=0.0,
        duration=10.0,
        step_size=0.05,
        save_every=50,
    )
    pairs = jnp.array([[0, 1], [2, 3], [4, 7]])
    gap = np.asarray(state.position)[np.asarray(pairs)]
    gap = gap[:, 0] - gap[:, 1]
    growth = np.exp(rate * np.asarray(result.offsets))
    expected = [(gap[:, 0] * g) ** 2 + (gap[:, 1] / g) ** 2 for g in growth]
    found = pair_dispersion(result.trajectory, pairs)
    assert_matches_oracle(found, np.mean(expected, axis=1), rtol=1e-9)


def test_a_pair_across_the_date_line_has_the_short_distance() -> None:
    trajectory = jnp.array([[[179.5, 0.0], [-179.5, 0.0]]])
    found = pair_dispersion(trajectory, jnp.array([[0, 1]]), distance="sphere")
    expected = np.deg2rad(1.0) * SPHERE.semi_major
    assert_matches_oracle(jnp.sqrt(found), np.array([expected]), rtol=1e-9)
    # of another sphere, in proportion
    small = pair_dispersion(
        trajectory, jnp.array([[0, 1]]), distance="sphere", earth=sphere(1.0)
    )
    assert_matches_oracle(jnp.sqrt(small), np.array([np.deg2rad(1.0)]), rtol=1e-9)


def test_a_rotation_of_the_sphere_keeps_every_pair_apart() -> None:
    state = at_point(jnp.zeros(2), 4)
    start = jnp.array([[0.0, 60.0], [1.0, 61.0], [179.9, 5.0], [-179.9, 5.0]])
    state = eqx.tree_at(lambda s: s.position, state, start)
    step = DeterministicSphere(SphereRotation(2.0 * np.pi / DAY, 0.5 * np.pi))
    result = integrate(
        state, step, t0=0.0, duration=DAY, step_size=DAY / 400, save_every=40
    )
    pairs = jnp.array([[0, 1], [2, 3]])
    found = pair_dispersion(result.trajectory, pairs, distance="sphere")
    assert_matches_oracle(found, np.full(11, float(found[0])), rtol=1e-7)
    # and at the end every particle is back
    back = absolute_dispersion(result.trajectory, distance="sphere")
    assert float(back[5]) > 1e12
    assert float(back[-1]) < 1.0


# --- against loops ------------------------------------------------------------


@pytest.mark.parametrize("distance", ["euclidean", "sphere"])
def test_both_match_a_loop(distance: str) -> None:
    trajectory = _globe() if distance == "sphere" else _walk(dim=3)
    # two missing positions, and a particle whose start is missing
    trajectory = trajectory.at[2, 1].set(jnp.nan).at[3, 4, 0].set(jnp.nan)
    trajectory = trajectory.at[0, 5].set(jnp.nan)
    pairs = np.array([[0, 1], [4, 2], [3, 3]])
    values = np.asarray(trajectory)

    def squared(left: np.ndarray, right: np.ndarray) -> float:
        if distance == "sphere":
            return float(great_circle_distance(*left, *right)) ** 2
        return float(np.sum((left - right) ** 2))

    absolute, relative = [], []
    for k in range(5):
        kept = [p for p in range(6) if np.isfinite(values[[0, k], p]).all()]
        absolute.append(np.mean([squared(values[k, p], values[0, p]) for p in kept]))
        kept = [(p, q) for p, q in pairs if np.isfinite(values[k, [p, q]]).all()]
        relative.append(np.mean([squared(values[k, p], values[k, q]) for p, q in kept]))
    found = absolute_dispersion(trajectory, distance=distance)
    assert_matches_oracle(found, np.array(absolute), rtol=1e-12)
    found = pair_dispersion(trajectory, jnp.asarray(pairs), distance=distance)
    assert_matches_oracle(found, np.array(relative), rtol=1e-12)


def test_a_time_with_nothing_known_is_nan() -> None:
    trajectory = _walk().at[2].set(jnp.nan)
    found = absolute_dispersion(trajectory)
    assert np.isnan(found).tolist() == [False, False, True, False, False]
    found = pair_dispersion(trajectory, jnp.array([[0, 1]]))
    assert np.isnan(found).tolist() == [False, False, True, False, False]


def test_a_missing_particle_changes_nothing() -> None:
    trajectory = _walk()
    missing = trajectory.at[:, 5].set(jnp.nan)
    found = absolute_dispersion(missing)
    assert_matches_oracle(found, absolute_dispersion(trajectory[:, :5]), rtol=1e-14)


# --- transforms, gradients and precision --------------------------------------


@pytest.mark.parametrize("distance", ["euclidean", "sphere"])
def test_transforms_agree(distance: str) -> None:
    trajectory = _globe(4, 12) if distance == "sphere" else _walk(4, 12)
    batch = einx.id(
        "time (batch member) axis -> batch time member axis", trajectory, batch=2
    )
    pairs = jnp.array([[0, 1], [2, 5]])
    # at the start the distance is rounding, which differs under jit
    assert_traces(lambda x: absolute_dispersion(x, distance=distance)[1:], batch)
    assert_traces(lambda x: pair_dispersion(x, pairs, distance=distance), batch)


@pytest.mark.parametrize("distance", ["euclidean", "sphere"])
def test_gradients_match_finite_differences(distance: str) -> None:
    trajectory = _globe(3, 4) if distance == "sphere" else _walk(3, 4)
    pairs = jnp.array([[0, 1], [2, 3]])
    scale = 1e-10 if distance == "sphere" else 1.0

    def both(positions: jax.Array) -> jax.Array:
        # one missing position, and the start, where the distance is zero
        positions = positions.at[2, 1].set(jnp.nan)
        absolute = absolute_dispersion(positions, distance=distance)
        relative = pair_dispersion(positions, pairs, distance=distance)
        return scale * (absolute + relative)

    assert_grad_matches_fd(both, trajectory, eps=1e-5, rtol=1e-6)


def test_float32_stays_float32() -> None:
    trajectory = _walk().astype(jnp.float32)
    assert absolute_dispersion(trajectory).dtype == jnp.float32
    assert pair_dispersion(trajectory, jnp.array([[0, 1]])).dtype == jnp.float32


# --- what is refused ----------------------------------------------------------


def test_invalid() -> None:
    trajectory = _walk()
    with pytest.raises((ValidationError, TypeError), match="distance"):
        absolute_dispersion(trajectory, distance="great-circle")
    with pytest.raises(ValidationError, match="longitude and latitude"):
        absolute_dispersion(_walk(dim=3), distance="sphere")
    with pytest.raises(ValidationError, match="latitudes"):
        absolute_dispersion(100.0 + trajectory, distance="sphere")
    with pytest.raises(ValidationError, match="sphere"):
        absolute_dispersion(_globe(), distance="sphere", earth=WGS84)
    for pairs in ([[0, 6]], [[-1, 0]]):
        with pytest.raises(ValidationError, match="'pairs' must name"):
            pair_dispersion(trajectory, jnp.array(pairs))
        with pytest.raises(RuntimeError, match="'pairs' must name"):
            eqx.filter_jit(lambda pairs: pair_dispersion(trajectory, pairs))(
                jnp.array(pairs)
            )
    with pytest.raises((ValidationError, TypeError), match="pairs"):
        pair_dispersion(trajectory, jnp.array([[0, 1, 2]]))
    with pytest.raises((ValidationError, TypeError), match="trajectory"):
        absolute_dispersion(trajectory[0])
