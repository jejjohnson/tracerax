"""A masked particle spoils no other particle's value or gradient.

Every field here is undefined, with an infinite slope or NaN, where the masked
particles are: at x < 0, or below the ground.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from _closures import Homogeneous
from _harness import assert_matches_oracle

from tracerax._src.advect.masks import choose, stand_in
from tracerax.advect import (
    Box,
    ConstantDiffusivity,
    Deterministic,
    Langevin,
    RandomDisplacement,
    deterministic,
    integrate,
)
from tracerax.particles import ParticleState, at_point


KEY = jax.random.key(0)
OPTIONS = {"t0": 0.0, "duration": 6.0, "step_size": 1.0}


class Root(eqx.Module):
    """u = a √x along x, NaN at x < 0."""

    scale: Any

    def __call__(self, t: Any, position: jax.Array) -> jax.Array:
        along = self.scale * jnp.sqrt(position[:, 0])
        return jnp.zeros_like(position).at[:, 0].set(along)


class RootDiffusivity(eqx.Module):
    """K = a √x on every axis, NaN at x < 0."""

    scale: Any

    def at(self, t: Any, position: jax.Array) -> jax.Array:
        along = self.scale * jnp.sqrt(position[:, :1])
        return jnp.broadcast_to(along, position.shape)


class RootTurbulence(eqx.Module):
    """σ = a √z, NaN below the ground."""

    scale: Any

    def at(self, height: jax.Array) -> Any:
        sigma = self.scale * jnp.sqrt(height)
        ones = jnp.ones((*height.shape, 3))
        return sigma[..., None] * ones, 60.0 * ones


def _state(first: Any, second: Any, active: Any = (True, False)) -> ParticleState:
    state = at_point(jnp.zeros(3), 2)
    return eqx.tree_at(
        lambda s: (s.position, s.active),
        state,
        (jnp.array([first, second]), jnp.array(active)),
    )


STEPS = {
    "deterministic": lambda a: Deterministic(Root(a)),
    "velocity": lambda a: RandomDisplacement(Root(a), ConstantDiffusivity(jnp.ones(3))),
    "diffusivity": lambda a: RandomDisplacement(
        lambda t, x: jnp.ones_like(x), RootDiffusivity(a)
    ),
    "wind": lambda a: Langevin(Root(a), Homogeneous()),
    "turbulence": lambda a: Langevin(lambda t, x: jnp.ones_like(x), RootTurbulence(a)),
}


def _key(name: str) -> dict[str, Any]:
    return {} if name == "deterministic" else {"key": KEY}


@pytest.mark.parametrize("name", STEPS)
class TestAMaskedParticle:
    """The first particle is at (90, 90, 90) and the second where nothing is."""

    GOOD, BAD = (90.0, 90.0, 90.0), (-1.0, -1.0, -1.0)

    def _gradient(self, name: str, state: ParticleState, **options: Any) -> jax.Array:
        def loss(scale: jax.Array, start: jax.Array) -> jax.Array:
            moved = eqx.tree_at(lambda s: s.position, state, start)
            result = integrate(
                moved, STEPS[name](scale), **OPTIONS, **_key(name), **options
            )
            return jnp.sum(result.state.position[0] ** 2)

        return jax.grad(loss, argnums=(0, 1))(jnp.asarray(0.3), state.position)

    def _alone(self, name: str) -> jax.Array:
        """The gradient when the second particle is valid, and masked all the same."""
        return self._gradient(name, _state(self.GOOD, (80.0, 70.0, 60.0)))

    def test_an_inactive_one(self, name: str) -> None:
        gradients = self._gradient(name, _state(self.GOOD, self.BAD))
        for gradient, expected in zip(gradients, self._alone(name), strict=True):
            assert_matches_oracle(gradient, expected, rtol=1e-10, atol=1e-12)

    def test_one_outside_the_domain(self, name: str) -> None:
        box = Box([0.0, 0.0, 0.0], [1e3, 1e3, 1e3])
        state = _state(self.GOOD, self.BAD, (True, True))
        gradients = self._gradient(name, state, domain=box)
        for gradient, expected in zip(gradients, self._alone(name), strict=True):
            assert_matches_oracle(gradient, expected, rtol=1e-10, atol=1e-12)

    def test_one_that_is_not_released(self, name: str) -> None:
        state = _state(self.GOOD, self.BAD, (True, True))
        state = eqx.tree_at(lambda s: s.release_time, state, jnp.array([0.0, 99.0]))
        gradients = self._gradient(name, state)
        for gradient, expected in zip(gradients, self._alone(name), strict=True):
            assert_matches_oracle(gradient, expected, rtol=1e-10, atol=1e-12)

    def test_the_step_by_hand(self, name: str) -> None:
        state = _state(self.GOOD, self.BAD)
        noise = None if name == "deterministic" else jnp.ones((2, 3))

        def loss(scale: jax.Array) -> jax.Array:
            moved = STEPS[name](scale)(state, jnp.zeros(2), jnp.ones(2), 1.0, noise)
            return jnp.sum(moved.position[0] ** 2)

        assert bool(jnp.isfinite(jax.grad(loss)(jnp.asarray(0.3))))


class TestWhenEveryParticleLeaves:
    """No particle moves in the last steps, so there is none to read at."""

    BOX = Box([0.0, -1e3, -1e3], [5.0, 1e3, 1e3])

    def _state(self) -> ParticleState:
        return _state((3.0, 1.0, 1.0), (4.0, 1.0, 1.0), (True, True))

    def test_the_loop(self) -> None:
        def loss(scale: jax.Array) -> jax.Array:
            # it ends past x = 6, where the velocity is undefined
            step = Deterministic(lambda t, x: scale * jnp.sqrt(6.0 - x))
            result = integrate(
                self._state(),
                step,
                t0=0.0,
                duration=60.0,
                step_size=1.0,
                domain=self.BOX,
            )
            return jnp.sum(result.state.position)

        value, gradient = jax.value_and_grad(loss)(jnp.asarray(1.7))
        assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(gradient))

    def test_the_solve(self) -> None:
        def loss(scale: jax.Array) -> jax.Array:
            final, _ = deterministic(
                self._state(),
                lambda t, x: scale * jnp.sqrt(jnp.abs(x)) + jnp.log(5.5 - x[:, :1]),
                t0=0.0,
                duration=20.0,
                n_steps=20,
                domain=self.BOX,
            )
            return jnp.sum(final.position)

        value, gradient = jax.value_and_grad(loss)(jnp.asarray(0.2))
        assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(gradient))


class TestHelpers:
    def test_choose_keeps_the_dtype_of_the_other(self) -> None:
        mask = jnp.array([True, False])
        chosen = choose(mask, jnp.ones((2, 2)), jnp.zeros((2, 2), jnp.float32))
        assert chosen.dtype == jnp.float32
        assert chosen.tolist() == [[1.0, 1.0], [0.0, 0.0]]

    def test_stand_in_reads_the_first_in_the_mask(self) -> None:
        position = jnp.arange(6.0).reshape(3, 2)
        mask = jnp.array([False, True, False])
        assert stand_in(position, mask).tolist() == [[2.0, 3.0]] * 3
        given = stand_in(position, mask, jnp.array([9.0, 9.0]))
        assert given.tolist() == [[9.0, 9.0], [2.0, 3.0], [9.0, 9.0]]

    def test_no_gradient_reaches_the_reference(self) -> None:
        position = jnp.arange(6.0).reshape(3, 2)
        mask = jnp.array([False, True, False])
        gradient = jax.grad(lambda x: jnp.sum(stand_in(x, mask)))(position)
        assert gradient.tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
