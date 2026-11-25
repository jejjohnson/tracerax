"""Separation, its normalised cumulative sum, and the skill of a trajectory."""

from __future__ import annotations

import inspect
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize
from _harness import assert_grad_matches_fd, assert_matches_oracle, assert_traces
from geophysix.earth import SPHERE, WGS84, great_circle_distance, sphere

from tracerax import ValidationError
from tracerax.advect import deterministic
from tracerax.analysis import cumulative_separation, separation, skill_score
from tracerax.particles import at_point
from tracerax.velocity import WeightedSum, uniform


KEY = jax.random.key(0)
DISTANCES = pytest.mark.parametrize("distance", ["euclidean", "sphere"])


def _observed(distance: str) -> jax.Array:
    """A day of hourly positions, across the date line on the sphere."""
    t = jnp.linspace(0.0, 1.0, 25)
    if distance == "euclidean":
        return 1e3 * jnp.column_stack([t, jnp.sin(3.0 * t)])
    longitude = jnp.mod(179.5 + t + 180.0, 360.0) - 180.0
    return jnp.column_stack([longitude, 40.0 + 0.5 * t])


def _simulated(distance: str) -> jax.Array:
    """It drifts away from the same start."""
    t = jnp.linspace(0.0, 1.0, 25)
    drift = {"euclidean": 100.0, "sphere": 0.01}[distance]
    return _observed(distance).at[:, 1].add(drift * t)


def _by_hand(left: Any, right: Any, distance: str) -> float:
    if distance == "sphere":
        return float(great_circle_distance(*left, *right))
    return float(np.linalg.norm(left - right))


def _score_by_hand(simulated: Any, observed: Any, distance: str) -> float:
    """∑ d_k / ∑ l_k, one time at a time."""
    simulated, observed = np.asarray(simulated), np.asarray(observed)
    total = scale = length = 0.0
    last = None
    for x, y in zip(simulated, observed, strict=True):
        if not np.isfinite(y).all():
            continue
        if last is not None:
            length += _by_hand(last, y, distance)
        last = y
        total += _by_hand(x, y, distance)
        scale += length
    return total / scale


# --- values -------------------------------------------------------------------


@DISTANCES
def test_the_values_match_a_loop(distance: str) -> None:
    """Gate G3-skill: s equals a NumPy loop."""
    simulated, observed = _simulated(distance), _observed(distance)
    expected = [
        _by_hand(x, y, distance)
        for x, y in zip(np.asarray(simulated), np.asarray(observed), strict=True)
    ]
    found = separation(simulated, observed, distance=distance)
    assert_matches_oracle(found, np.array(expected), rtol=1e-12)
    s = _score_by_hand(simulated, observed, distance)
    found = cumulative_separation(simulated, observed, distance=distance)
    assert_matches_oracle(found, s, rtol=1e-12)
    found = skill_score(simulated, observed, distance=distance)
    assert_matches_oracle(found, 1.0 - s, rtol=1e-12)
    found = skill_score(simulated, observed, distance=distance, tolerance=2.0 * s)
    assert_matches_oracle(found, 0.5, rtol=1e-12)


@DISTANCES
@pytest.mark.parametrize(
    "missing", [[0], [0, 1, 2], [7], [3, 4, 5, 11], [24], [0, 12, 23, 24]]
)
def test_a_missing_observation_is_left_out_and_bridged(
    distance: str, missing: list[int]
) -> None:
    simulated, observed = _simulated(distance), _observed(distance)
    gaps = observed.at[jnp.array(missing)].set(jnp.nan)
    # one column is enough to be missing
    gaps = gaps.at[15, 0].set(jnp.nan)
    expected = _score_by_hand(simulated, gaps, distance)
    found = cumulative_separation(simulated, gaps, distance=distance)
    assert_matches_oracle(found, expected, rtol=1e-12)
    apart = separation(simulated, gaps, distance=distance)
    assert np.flatnonzero(np.isnan(apart)).tolist() == sorted([*missing, 15])
    # what is simulated there does not matter, whatever it is
    dirty = simulated.at[jnp.array(missing)].set(jnp.nan)
    again = cumulative_separation(dirty, gaps, distance=distance)
    assert_matches_oracle(again, expected, rtol=1e-12)


@DISTANCES
def test_a_simulated_nan_is_not_hidden(distance: str) -> None:
    simulated = _simulated(distance).at[7].set(jnp.nan)
    observed = _observed(distance)
    assert np.isnan(separation(simulated, observed, distance=distance)[7])
    assert np.isnan(cumulative_separation(simulated, observed, distance=distance))
    assert np.isnan(skill_score(simulated, observed, distance=distance))


def test_the_skill_is_zero_beyond_the_tolerance() -> None:
    observed = _observed("euclidean")
    far = observed + jnp.array([0.0, 1e6])
    assert float(cumulative_separation(far, observed)) > 1.0
    assert float(skill_score(far, observed)) == 0.0
    assert float(skill_score(observed, observed)) == 1.0


def test_an_observation_that_does_not_move_needs_a_length() -> None:
    observed = jnp.ones((5, 2))
    simulated = observed.at[1:, 0].add(2.0)
    assert np.isinf(cumulative_separation(simulated, observed))
    assert float(skill_score(simulated, observed)) == 0.0
    # 8 m over 5 lengths of 4 m
    found = cumulative_separation(simulated, observed, min_length=4.0)
    assert_matches_oracle(found, 0.4, rtol=1e-14)
    found = skill_score(simulated, observed, min_length=4.0)
    assert_matches_oracle(found, 0.6, rtol=1e-14)


def test_a_drifter_that_does_not_move_spoils_no_gradient() -> None:
    """One good drifter and one stationary one, with a shared parameter."""
    moving, still = _observed("euclidean"), jnp.ones((25, 2))
    observed = jnp.stack([moving, still])

    def total(drift: jax.Array, fn: Any) -> jax.Array:
        simulated = observed + drift * jnp.array([1.0, 2.0])
        scores = fn(simulated, observed)
        return jnp.where(jnp.isfinite(scores), scores, 0.0).sum()

    for fn in (cumulative_separation, skill_score):
        gradient = jax.grad(total)(3.0, fn)
        alone = jax.grad(
            lambda drift, fn=fn: fn(moving + drift * jnp.array([1.0, 2.0]), moving)
        )(3.0)
        assert_matches_oracle(gradient, float(alone), rtol=1e-12)
        assert float(gradient) != 0.0


def test_integers_do_not_truncate_the_observations() -> None:
    simulated = jnp.array([[0, 0], [10, 0], [20, 0]])
    observed = jnp.array([[0.0, 0.0], [10.5, 0.0], [jnp.nan, 0.0]])
    bare = inspect.unwrap(separation)
    assert_matches_oracle(bare(simulated, observed), [0.0, 0.5, np.nan], rtol=1e-14)


def test_the_bounds_are_one_number_each_and_keep_the_dtype() -> None:
    simulated = _simulated("euclidean").astype(jnp.float32)
    observed = _observed("euclidean")
    score = skill_score(
        simulated, observed, tolerance=np.float64(2.0), min_length=np.float64(1.0)
    )
    assert score.dtype == jnp.float32
    with pytest.raises((ValidationError, TypeError), match="min_length"):
        cumulative_separation(simulated, observed, min_length=jnp.ones(25))


def test_no_observation_is_nan() -> None:
    simulated = _simulated("euclidean")
    nothing = jnp.full_like(simulated, jnp.nan)
    assert np.isnan(cumulative_separation(simulated, nothing))
    assert np.isnan(skill_score(simulated, nothing))


def test_three_columns() -> None:
    observed = jnp.array([[0.0, 0.0, 0.0], [2.0, 3.0, 6.0]])
    simulated = observed.at[1].add(jnp.array([1.0, 2.0, 2.0]))
    assert_matches_oracle(cumulative_separation(simulated, observed), 3 / 7, rtol=1e-14)


def test_another_sphere_changes_the_separation_and_not_the_score() -> None:
    simulated, observed = _simulated("sphere"), _observed("sphere")
    options: dict[str, Any] = {"distance": "sphere"}
    large = separation(simulated, observed, **options)
    small = separation(simulated, observed, earth=sphere(1.0), **options)
    assert_matches_oracle(small * SPHERE.semi_major, large, rtol=1e-12)
    large = cumulative_separation(simulated, observed, **options)
    small = cumulative_separation(simulated, observed, earth=sphere(1.0), **options)
    assert_matches_oracle(small, large, rtol=1e-12)


# --- batches and transforms ---------------------------------------------------


@DISTANCES
def test_one_reference_serves_a_batch(distance: str) -> None:
    observed = _observed(distance).at[4].set(jnp.nan)
    scales = jnp.array([0.0, 1.0, 3.0])
    drift = _simulated(distance) - _observed(distance)
    batch = jnp.stack([_observed(distance) + scale * drift for scale in scales])
    found = cumulative_separation(batch, observed[None], distance=distance)
    expected = [
        cumulative_separation(member, observed, distance=distance) for member in batch
    ]
    assert_matches_oracle(found, np.stack(expected), rtol=1e-13, atol=1e-15)


@DISTANCES
def test_transforms_agree(distance: str) -> None:
    observed = jnp.stack([_observed(distance), _observed(distance)])
    # apart at the start too: a distance of 0 is rounding, which differs under jit
    apart = _simulated(distance) + jnp.array([0.0, 0.02])
    simulated = jnp.stack([apart, 2.0 * apart - observed[0]])
    for fn in (separation, cumulative_separation, skill_score):
        assert_traces(
            lambda x, y, fn=fn: fn(x, y, distance=distance), simulated, observed
        )


def test_float32_stays_float32() -> None:
    simulated = _simulated("euclidean").astype(jnp.float32)
    observed = _observed("euclidean")
    for fn in (separation, cumulative_separation, skill_score):
        assert fn(simulated, observed).dtype == jnp.float32


# --- gradients ----------------------------------------------------------------


@DISTANCES
def test_gradients_match_finite_differences(distance: str) -> None:
    """With a shared starting point and a missing observation."""
    observed = _observed(distance).at[4].set(jnp.nan)
    eps = {"euclidean": 1e-4, "sphere": 1e-7}[distance]

    def of_simulated(later: jax.Array) -> jax.Array:
        simulated = jnp.concatenate([observed[:1], later])
        return cumulative_separation(simulated, observed, distance=distance)

    later = _simulated(distance)[1:]
    assert_grad_matches_fd(of_simulated, later, eps=eps, rtol=1e-6)

    def of_observed(later: jax.Array) -> jax.Array:
        path = jnp.concatenate([observed[:1], later]).at[4].set(jnp.nan)
        return cumulative_separation(_simulated(distance), path, distance=distance)

    later = _observed(distance)[1:]
    assert_grad_matches_fd(of_observed, later, eps=eps, rtol=1e-6)


@DISTANCES
def test_the_gradient_is_finite_at_a_perfect_forecast(distance: str) -> None:
    """Gate G3-skill."""
    observed = _observed(distance).at[4].set(jnp.nan)
    perfect = _observed(distance)
    for fn in (cumulative_separation, skill_score):
        for argument in (0, 1):
            gradient = jax.grad(fn, argnums=argument)(
                perfect, observed, distance=distance
            )
            assert np.isfinite(np.delete(np.asarray(gradient), 4, axis=0)).all()
    gradient = jax.grad(cumulative_separation)(perfect, observed, distance=distance)
    assert_matches_oracle(gradient, np.zeros((25, 2)), rtol=0.0)


def test_a_windage_fit_recovers_three_per_cent() -> None:
    """The loss is the score of a drifter under a current and β of the wind."""
    save_at = 3_600.0 * jnp.arange(25.0)

    def current(t: Any, position: jax.Array) -> jax.Array:
        x, y = position[:, 0] / 1e4, position[:, 1] / 1e4
        return 0.3 * jnp.column_stack([jnp.sin(y), jnp.cos(x)])

    def path(windage: Any) -> jax.Array:
        weights = jnp.stack([jnp.ones(()), jnp.asarray(windage)])
        drift = WeightedSum((current, uniform(jnp.array([8.0, 3.0]))), weights)
        _, saved = deterministic(
            at_point(jnp.zeros(2), 1),
            drift,
            t0=0.0,
            duration=86_400.0,
            n_steps=48,
            save_at=save_at,
            adjoint=dfx.ForwardMode(),
        )
        return saved[:, 0]

    observed = path(0.03).at[5].set(jnp.nan)

    @jax.jit
    def loss(windage: jax.Array) -> jax.Array:
        return cumulative_separation(path(windage), observed)

    fit = scipy.optimize.minimize_scalar(
        lambda value: float(loss(value)),
        bounds=(0.0, 0.08),
        method="bounded",
        options={"xatol": 1e-9},
    )
    assert fit.x == pytest.approx(0.03, abs=1e-6)
    # the gradient points to it from both sides
    slope = jax.jit(jax.jacfwd(loss))
    assert float(slope(0.02)) < 0.0 < float(slope(0.04))


# --- what is refused ----------------------------------------------------------


@pytest.mark.parametrize("fn", [separation, cumulative_separation, skill_score])
def test_invalid(fn: Any) -> None:
    simulated, observed = _simulated("euclidean"), _observed("euclidean")
    with pytest.raises((ValidationError, TypeError), match="distance"):
        fn(simulated, observed, distance="manhattan")
    with pytest.raises(ValidationError, match="latitudes"):
        fn(simulated, observed, distance="sphere")
    with pytest.raises((ValidationError, TypeError), match="reference"):
        fn(simulated, observed[:5])
    with pytest.raises((ValidationError, TypeError), match="reference"):
        fn(simulated, jnp.zeros((25, 3)))
    globe = _observed("sphere")
    with pytest.raises(ValidationError, match="must be a sphere"):
        fn(globe, globe, distance="sphere", earth=WGS84)


def test_traced_latitudes_beyond_a_pole_are_refused() -> None:
    swapped = _observed("sphere")[:, ::-1]
    score = jax.jit(lambda x, y: cumulative_separation(x, y, distance="sphere"))
    with pytest.raises(RuntimeError, match="must hold latitudes"):
        score(swapped, swapped)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_bounds(value: float) -> None:
    observed = _observed("euclidean")
    with pytest.raises(ValidationError, match="min_length"):
        cumulative_separation(observed, observed, min_length=value)
    with pytest.raises(ValidationError, match="tolerance"):
        skill_score(observed, observed, tolerance=value)
    with pytest.raises(ValidationError, match="tolerance"):
        skill_score(observed, observed, tolerance=0.0)
