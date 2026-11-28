"""Regions: labels, residence times per particle, and connectivity."""

from __future__ import annotations

import inspect
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pipekit_array.testing import assert_matches_oracle, assert_traces

from tracerax import ValidationError
from tracerax.advect import (
    Accumulator,
    Box,
    ConstantDiffusivity,
    Deterministic,
    RandomDisplacement,
    Walls,
    integrate,
)
from tracerax.analysis import Regions, RegionTime, connectivity, visited_connectivity
from tracerax.bin import Cells
from tracerax.particles import ParticleState, at_point, in_box
from tracerax.velocity import uniform


KEY = jax.random.key(0)
CELLS = Cells(([0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 3.0]))
# one region on two separate cells, and one cell in none
LABELS = jnp.array([[0, 1], [2, -1], [1, 0]])
REGIONS = Regions(CELLS, LABELS, n_regions=3)


def _label_by_hand(position: np.ndarray) -> np.ndarray:
    """L(x), one particle at a time."""
    edges_x, edges_y = (np.asarray(edge) for edge in CELLS.edges)
    found = []
    for x, y in np.asarray(position)[:, :2]:
        label = -1
        for i in range(3):
            for j in range(2):
                in_x = edges_x[i] <= x < edges_x[i + 1]
                in_y = edges_y[j] <= y < edges_y[j + 1]
                if in_x and in_y:
                    label = int(LABELS[i, j])
        found.append(label)
    return np.array(found)


def _scattered(n: int, key: Any = KEY, dim: int = 2) -> jax.Array:
    """Positions inside and outside the cells."""
    low, high = jnp.array([-1.0, -1.0, 0.0]), jnp.array([10.0, 4.0, 9.0])
    draws = jax.random.uniform(key, (n, dim))
    return low[:dim] + (high - low)[:dim] * draws


class TestRegions:
    def test_labels_match_a_loop(self) -> None:
        position = _scattered(300)
        assert_matches_oracle(REGIONS.at(position), _label_by_hand(position), rtol=0.0)

    def test_every_height_is_labelled_by_two_axes(self) -> None:
        position = _scattered(50, dim=3)
        assert_matches_oracle(REGIONS.at(position), _label_by_hand(position), rtol=0.0)

    def test_three_axes_label_three_columns(self) -> None:
        cells = Cells(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 5.0]))
        regions = Regions(cells, jnp.array([[[-1, 0]]]), n_regions=1)
        position = jnp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.0], [0.5, 1.0, 1.0]])
        assert regions.at(position).tolist() == [-1, 0, -1]

    def test_a_position_that_is_nan_or_on_the_last_edge_is_in_no_region(self) -> None:
        position = jnp.array([[jnp.nan, 1.0], [0.5, jnp.nan], [9.0, 2.5], [0.0, 0.0]])
        assert REGIONS.at(position).tolist() == [-1, -1, -1, 0]

    def test_transforms(self) -> None:
        positions = jnp.stack([_scattered(20), _scattered(20, jax.random.key(1))])
        assert_traces(lambda position: REGIONS.at(position), positions)

    def test_labels_may_be_traced(self) -> None:
        position = _scattered(40)

        @jax.jit
        def at(labels: jax.Array) -> jax.Array:
            return Regions(CELLS, labels, n_regions=3).at(position)

        assert jnp.array_equal(at(LABELS), REGIONS.at(position))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint32])
    def test_labels_of_any_integer_type_are_minus_one_outside(
        self, dtype: type
    ) -> None:
        cells = Cells(([0.0, 1.0, 2.0], [0.0, 1.0]))
        regions = Regions(cells, np.array([[0], [1]], dtype), n_regions=256)
        assert regions.at(jnp.array([[1.5, 0.5], [50.0, 0.5]])).tolist() == [1, -1]
        start, end = jnp.array([[0.5, 0.5]]), jnp.array([[50.0, 0.5]])
        assert float(connectivity(start, end, regions)[0].sum()) == 0.0

    def test_traced_labels_are_checked_when_the_program_runs(self) -> None:
        position = _scattered(5)

        @eqx.filter_jit
        def at(labels: jax.Array) -> jax.Array:
            return Regions(CELLS, labels, n_regions=3).at(position)

        for wrong in (7, -5):
            with pytest.raises(RuntimeError, match="from −1 to 2"):
                at(LABELS.at[0, 0].set(wrong))

    def test_concrete_labels_add_no_callback_under_jit(self) -> None:
        # a jnp comparison of a concrete array is traced under jit, and its check
        # would be a host callback; the labels are compared on the host instead
        def labelled(scale: jax.Array) -> jax.Array:
            return scale * Regions(CELLS, LABELS, n_regions=3).labels.sum()

        assert "callback" not in str(jax.make_jaxpr(labelled)(1.0))

    def test_float32_positions(self) -> None:
        position = _scattered(100).astype(jnp.float32)
        assert_matches_oracle(REGIONS.at(position), _label_by_hand(position), rtol=0.0)

    def test_a_label_float32_cannot_hold_is_read_exactly(self) -> None:
        """2²⁴ + 1 has no float32: a label is read as an integer, not a position."""
        large = 2**24 + 1
        regions = Regions(CELLS, LABELS.at[0, 0].set(large), n_regions=large + 1)
        position = jnp.array([[0.5, 1.0], [5.0, 1.0]], dtype=jnp.float32)
        assert regions.at(position).tolist() == [large, 1]

    @pytest.mark.parametrize(
        ("labels", "n_regions", "match"),
        [
            (LABELS[:2], 3, "shape of the cells"),
            (LABELS.astype(float), 3, "labels"),
            (LABELS, 2, "from −1 to 1"),
            (LABELS - 1, 3, "from −1 to 2"),
            (LABELS, 0, "n_regions"),
            (np.array([[0, 255], [0, 0], [0, 0]], np.uint8), 3, "from −1 to 2"),
        ],
    )
    def test_invalid(self, labels: Any, n_regions: int, match: str) -> None:
        with pytest.raises((ValidationError, TypeError), match=match):
            Regions(CELLS, labels, n_regions=n_regions)

    def test_one_axis_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="two axes"):
            Regions(Cells(([0.0, 1.0],)), jnp.array([0]), n_regions=1)


class TestRegionTime:
    @pytest.mark.parametrize("step_size", [0.7, 0.1])
    def test_a_box_holds_a_particle_for_its_length_over_the_speed(
        self, step_size: float
    ) -> None:
        """τ = L/u to one step, for particles that start at different places."""
        cells = Cells(([0.0, 10.0, 13.0], [0.0, 1.0]))
        regions = Regions(cells, jnp.array([[-1], [0]]), n_regions=1)
        state = in_box(jnp.zeros(2), jnp.array([10.0, 1.0]), 50, key=KEY)
        result = integrate(
            state,
            Deterministic(uniform(jnp.array([0.5, 0.0]))),
            t0=0.0,
            duration=40.0 // step_size * step_size,
            step_size=step_size,
            accumulate=RegionTime(regions, n_particles=50),
        )
        time = np.asarray(result.accumulated)[:, 0]
        assert (np.abs(time - 3.0 / 0.5) <= step_size * (1.0 + 1e-9)).all()
        # over uniform starts the errors of the rule average out
        assert abs(time.mean() - 6.0) < 0.3 * step_size

    def test_the_total_matches_a_loop_over_the_trajectory(self) -> None:
        """With a random step, a release inside a step and a shorter last step."""
        step = RandomDisplacement(
            uniform(jnp.array([0.4, 0.1])), ConstantDiffusivity([0.3, 0.1])
        )
        state = in_box(jnp.zeros(2), jnp.array([9.0, 3.0]), 20, key=KEY)
        times = jnp.linspace(0.0, 4.0, 20)
        state = eqx.tree_at(lambda s: s.release_time, state, times)
        result = integrate(
            state,
            step,
            t0=0.0,
            duration=5.25,
            step_size=0.5,
            key=jax.random.key(1),
            accumulate=RegionTime(REGIONS, n_particles=20),
            save_every=1,
        )
        ends = np.minimum(0.5 * np.arange(1, 12), 5.25)
        begins = 0.5 * np.arange(11)
        expected = np.zeros((20, 3))
        for k in range(11):
            label = _label_by_hand(result.trajectory[k + 1])
            for p in range(20):
                moved = max(ends[k] - max(begins[k], float(times[p])), 0.0)
                if label[p] >= 0:
                    expected[p, label[p]] += moved
        assert_matches_oracle(result.accumulated, expected, rtol=1e-12, atol=1e-12)

    def test_a_particle_that_left_or_never_moved_adds_nothing(self) -> None:
        state = at_point(jnp.array([0.5, 0.5]), 4)
        state = eqx.tree_at(
            lambda s: (s.position, s.active, s.release_time),
            state,
            (
                state.position.at[1].set(jnp.nan),
                jnp.array([True, False, True, True]),
                jnp.array([0.0, 0.0, 100.0, 0.0]),
            ),
        )
        result = integrate(
            state,
            Deterministic(uniform(jnp.array([1.0, 0.0]))),
            t0=0.0,
            duration=6.0,
            step_size=0.25,
            domain=Box([0.0, 0.0], [2.0, 2.0]),
            accumulate=RegionTime(REGIONS, n_particles=4),
        )
        # it leaves region 0 at x = 1, after two steps, and the domain at x = 2
        row = [0.25, 0.0, 1.0]
        expected = np.array([row, [0.0] * 3, [0.0] * 3, row])
        assert_matches_oracle(result.accumulated, expected, rtol=1e-14)

    def test_it_is_an_accumulator(self) -> None:
        accumulator = RegionTime(REGIONS, n_particles=2)
        assert isinstance(accumulator, Accumulator)
        assert accumulator.init().shape == (2, 3)

    def test_another_number_of_particles_is_refused(self) -> None:
        accumulator = RegionTime(REGIONS, n_particles=2)
        state = at_point(jnp.zeros(2), 3)
        with pytest.raises(ValidationError, match="n_particles"):
            # past the annotations, which refuse it first in this suite
            update = inspect.unwrap(RegionTime.update)
            update(accumulator, accumulator.init(), state, jnp.ones(3))
        with pytest.raises(ValidationError, match="n_particles"):
            RegionTime(REGIONS, n_particles=0)

    def test_transforms_of_a_run(self) -> None:
        def total(position: jax.Array) -> jax.Array:
            state: ParticleState = at_point(jnp.zeros(2), 5)
            state = eqx.tree_at(lambda s: s.position, state, position)
            return integrate(
                state,
                Deterministic(uniform(jnp.array([1.0, 0.2]))),
                t0=0.0,
                duration=3.0,
                step_size=0.5,
                accumulate=RegionTime(REGIONS, n_particles=5),
            ).accumulated

        positions = jnp.stack([_scattered(5), _scattered(5, jax.random.key(1))])
        assert_traces(total, positions)


class TestConnectivity:
    def _ends(self) -> tuple[jax.Array, jax.Array]:
        return _scattered(400), _scattered(400, jax.random.key(1))

    def test_the_matrix_matches_a_loop(self) -> None:
        start, end = self._ends()
        first, last = _label_by_hand(start), _label_by_hand(end)
        expected = np.zeros((3, 3))
        for i in range(3):
            members = [p for p in range(400) if first[p] == i]
            for j in range(3):
                reached = [p for p in members if last[p] == j]
                expected[i, j] = len(reached) / len(members)
        found = connectivity(start, end, REGIONS)
        assert_matches_oracle(found, expected, rtol=1e-14)
        # some end in no region
        assert (found.sum(axis=1) < 1.0).all()

    def test_rows_sum_to_one_in_a_closed_domain(self) -> None:
        """Walls and a periodic flow keep every particle in some region."""
        cells = Cells(([-1e6, 0.0, 1e6], [-1e6, 1e6], [0.0, 10.0, 20.0]))
        regions = Regions(cells, jnp.array([[[0, 1]], [[2, 3]]]), n_regions=4)
        step = RandomDisplacement(
            uniform(jnp.zeros(3)),
            ConstantDiffusivity([5.0, 5.0, 5.0]),
            Walls(0.0, 20.0),
        )
        low, high = jnp.array([-20.0, 0.0, 0.0]), jnp.array([20.0, 1.0, 20.0])
        state = in_box(low, high, 200, key=KEY)
        result = integrate(
            state, step, t0=0.0, duration=20.0, step_size=1.0, key=jax.random.key(1)
        )
        found = connectivity(state.position, result.state.position, regions)
        assert_matches_oracle(found.sum(axis=1), np.ones(4), rtol=1e-14)
        assert (np.diag(found) < 1.0).all()

    def test_a_region_without_particles_is_a_row_of_nan(self) -> None:
        start = jnp.array([[0.5, 0.5], [0.5, 2.5]])
        found = connectivity(start, start, REGIONS)
        assert_matches_oracle(found[:2], np.eye(3)[:2], rtol=0.0)
        assert np.isnan(found[2]).all()

    def test_a_particle_that_is_not_counted_changes_nothing(self) -> None:
        start, end = self._ends()
        counted = jnp.arange(400) < 300
        # what is not counted may be anything
        dirty = start.at[300:350].set(jnp.nan).at[350:].set(0.5)
        found = connectivity(dirty, end, REGIONS, counted=counted)
        alone = connectivity(start[:300], end[:300], REGIONS)
        assert_matches_oracle(found, alone, rtol=0.0)
        time = jax.random.uniform(KEY, (400, 3))
        found = visited_connectivity(dirty, time, REGIONS, counted=counted)
        alone = visited_connectivity(start[:300], time[:300], REGIONS)
        assert_matches_oracle(found, alone, rtol=0.0)

    def test_the_visited_matrix_matches_a_loop(self) -> None:
        start = _scattered(400)
        visits = jax.random.uniform(KEY, (400, 3))
        time = jnp.where(visits < 0.6, 0.0, visits)
        first = _label_by_hand(start)
        expected = np.zeros((3, 3))
        for i in range(3):
            members = [p for p in range(400) if first[p] == i]
            for j in range(3):
                reached = [p for p in members if float(time[p, j]) > 0.0]
                expected[i, j] = len(reached) / len(members)
        found = visited_connectivity(start, time, REGIONS)
        assert_matches_oracle(found, expected, rtol=1e-14)

    def test_transforms_and_dtype(self) -> None:
        start, end = self._ends()
        starts, ends = start.reshape(4, 100, 2), end.reshape(4, 100, 2)
        assert_traces(lambda s, e: connectivity(s, e, REGIONS), starts, ends)
        single = connectivity(
            start.astype(jnp.float32), end.astype(jnp.float32), REGIONS
        )
        assert single.dtype == jnp.float32

    def test_the_gradient_is_zero_beside_an_empty_region(self) -> None:
        def weighted(start: jax.Array) -> jax.Array:
            return connectivity(start, 1.1 * start, REGIONS)[0].sum()

        gradient = jax.grad(weighted)(jnp.array([[0.5, 0.5], [0.2, 0.1]]))
        assert_matches_oracle(gradient, np.zeros((2, 2)), rtol=0.0)

    def test_invalid(self) -> None:
        start, end = self._ends()
        with pytest.raises(ValidationError, match="'end'"):
            inspect.unwrap(connectivity)(start, end[:10], REGIONS)
        with pytest.raises((ValidationError, TypeError), match="counted"):
            inspect.unwrap(connectivity)(start, end, REGIONS, counted=jnp.ones(3, bool))
        with pytest.raises(ValidationError, match="'time'"):
            inspect.unwrap(visited_connectivity)(start, jnp.ones((400, 2)), REGIONS)
