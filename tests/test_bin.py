"""Tests for the cells, the deposits, the residence and the concentrations."""

from __future__ import annotations

import inspect
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _closures import Homogeneous, LinearSigma
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _reference import CLOSURES, reference_values
from hypothesis import given, strategies as st
from integrix.adjoint import Checkpointed
from scipy.interpolate import RegularGridInterpolator

from tracerax import ValidationError
from tracerax.advect import (
    Accumulator,
    Box,
    Deterministic,
    Langevin,
    Result,
    Walls,
    integrate,
)
from tracerax.bin import (
    Cells,
    Residence,
    column,
    concentration,
    deposit,
    steady_concentration,
)
from tracerax.particles import (
    ParticleState,
    at_point,
    in_box,
    release_times,
    with_turbulent_velocity,
)
from tracerax.velocity import uniform, uniform_in_space


KEY = jax.random.key(0)
KINDS = ("nearest", "linear")

# one axis of each: uniform, stretched, and stretched with a single wide end cell
UNIFORM = (
    np.linspace(-4.0, 4.0, 9),
    np.linspace(0.0, 3.0, 4),
    np.linspace(0.0, 40.0, 6),
)
STRETCHED = (
    np.array([-4.0, -3.5, -1.0, 0.0, 0.25, 4.0]),
    np.array([0.0, 2.9, 3.0]),
    np.array([0.0, 1.0, 3.0, 10.0, 11.0, 40.0]),
)
GRIDS = {"uniform": UNIFORM, "stretched": STRETCHED}


def _particles(
    edges: tuple[np.ndarray, ...], size: int = 400, seed: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Particles over the cells and a margin, and on every edge; their weights."""
    rng = np.random.default_rng(seed)
    low = np.array([edge[0] for edge in edges])
    high = np.array([edge[-1] for edge in edges])
    margin = 0.1 * (high - low)
    position = rng.uniform(low - margin, high + margin, (size, len(edges)))
    for axis, edge in enumerate(edges):
        # on the edges of this axis, inside the others
        rows = slice(axis * 20, axis * 20 + edge.size)
        position[rows] = 0.5 * (low + high)
        position[rows, axis] = edge
    return jnp.asarray(position), jnp.asarray(rng.uniform(0.5, 2.0, size))


def _inside(edges: tuple[np.ndarray, ...], position: Any) -> np.ndarray:
    low = np.array([edge[0] for edge in edges])
    high = np.array([edge[-1] for edge in edges])
    values = np.asarray(position)
    return ((values >= low) & (values < high)).all(axis=1)


def _read_nearest(
    edges: tuple[np.ndarray, ...], field: np.ndarray, position: Any
) -> np.ndarray:
    """H c for nearest, one particle at a time."""
    read = np.zeros(len(position))
    for row in np.flatnonzero(_inside(edges, position)):
        point = np.asarray(position[row])
        cell = [
            int(np.flatnonzero(edge <= value)[-1])
            for edge, value in zip(edges, point, strict=True)
        ]
        read[row] = field[tuple(cell)]
    return read


def _read_linear(
    edges: tuple[np.ndarray, ...], field: np.ndarray, position: Any
) -> np.ndarray:
    """H c for linear: SciPy on the centres, with the end values held to the edges."""
    nodes = [
        np.concatenate([edge[:1], 0.5 * (edge[:-1] + edge[1:]), edge[-1:]])
        for edge in edges
    ]
    inside = _inside(edges, position)
    read = np.zeros(len(position))
    interpolate = RegularGridInterpolator(nodes, np.pad(field, 1, mode="edge"))
    read[inside] = interpolate(np.asarray(position)[inside])
    return read


READS = {"nearest": _read_nearest, "linear": _read_linear}


class TestCells:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("grid", GRIDS)
    def test_volumes_sum_to_the_box(self, grid: str, dim: int) -> None:
        edges = GRIDS[grid][:dim]
        cells = Cells(edges)
        volumes = cells.volumes()
        assert volumes.shape == cells.shape == tuple(edge.size - 1 for edge in edges)
        box = np.prod([edge[-1] - edge[0] for edge in edges])
        assert_matches_oracle(volumes.sum(), box, rtol=1e-12)
        # and each is the product of its widths
        index = (1,) * dim
        widths = [edge[2] - edge[1] for edge in edges]
        assert_matches_oracle(volumes[index], np.prod(widths), rtol=1e-14)

    def test_centres(self) -> None:
        cells = Cells(STRETCHED)
        for centres, edge in zip(cells.centres, STRETCHED, strict=True):
            assert_matches_oracle(centres, 0.5 * (edge[1:] + edge[:-1]), rtol=1e-15)

    def test_contains(self) -> None:
        position, _ = _particles(STRETCHED)
        inside = Cells(STRETCHED).contains(position)
        assert_matches_oracle(inside, _inside(STRETCHED, position), rtol=0.0)
        assert 0 < inside.sum() < inside.size
        nan = jnp.array([[jnp.nan, 1.0, 1.0], [0.0, jnp.inf, 1.0]])
        assert not Cells(STRETCHED).contains(nan).any()

    def test_integer_edges_become_floating(self) -> None:
        cells = Cells(([0, 1, 5],))
        assert cells.edges[0].dtype == jnp.float64
        assert cells.centres[0].tolist() == [0.5, 3.0]

    @pytest.mark.parametrize(
        ("edges", "message"),
        [
            ((), "1, 2 or 3 axes, got 0"),
            (([0.0, 1.0],) * 4, "1, 2 or 3 axes, got 4"),
            (([0.0, 1.0], [0.0]), r"'edges\[1\]' must be 1-D with at least two"),
            ([0.0, 1.0], r"'edges\[0\]' must be 1-D with at least two"),
            ((np.zeros((2, 2)),), r"'edges\[0\]' must be 1-D"),
            (([0.0, 0.0, 1.0],), "strictly increasing"),
            (([1.0, 0.0],), "strictly increasing"),
            (([0.0, np.nan],), "strictly increasing"),
            (([0.0, np.inf],), "strictly increasing"),
            # a cell with no room for a centre
            (([1.0, np.nextafter(1.0, 2.0)],), "strictly increasing"),
        ],
    )
    def test_bad_edges(self, edges: Any, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            Cells(edges)

    @pytest.mark.parametrize(
        "edge", [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0], [0.0, np.nan], [0.0, np.inf]]
    )
    def test_bad_traced_edges(self, edge: list[float]) -> None:
        with pytest.raises(RuntimeError, match="strictly increasing"):
            jax.jit(lambda edge: Cells((edge,)).volumes())(jnp.asarray(edge))

    def test_traces(self) -> None:
        position, _ = _particles(UNIFORM, size=96)
        batch = position.reshape(4, 24, 3)
        cells = Cells(UNIFORM)
        assert_traces(lambda position: cells.contains(position), batch)
        # the edges may be traced too
        edges = tuple(jnp.stack([edge, 2.0 * edge]) for edge in UNIFORM)
        assert_traces(lambda *edges: Cells(edges).volumes(), *edges)


class TestAgainstNumPy:
    @pytest.mark.parametrize("prefix", ["", "stretched_"])
    def test_counts(self, prefix: str) -> None:
        stored = reference_values("binning")
        cells = Cells([stored[f"{prefix}{axis}_edges"] for axis in "xyz"])
        position = jnp.asarray(stored["position"])
        counts = deposit(cells, position, jnp.ones(len(position)))
        assert_matches_oracle(counts, stored[f"{prefix}counts"], rtol=0.0)


class TestDeposit:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("grid", GRIDS)
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_mass_is_kept(self, kind: str, grid: str, dim: int) -> None:
        """G2-mass: each particle inside deposits its whole weight, walls included."""
        edges = GRIDS[grid][:dim]
        position, weight = _particles(edges)
        inside = _inside(edges, position)
        total = deposit(Cells(edges), position, weight, kind=kind).sum()
        assert_matches_oracle(total, np.asarray(weight)[inside].sum(), rtol=1e-12)
        # and one at a time
        alone = jax.vmap(
            lambda point, mass: deposit(
                Cells(edges), point[None], mass[None], kind=kind
            ).sum()
        )(position, weight)
        assert_matches_oracle(alone, np.where(inside, weight, 0.0), rtol=1e-12)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    @pytest.mark.parametrize("grid", GRIDS)
    @pytest.mark.parametrize("kind", KINDS)
    def test_it_is_the_transpose_of_a_read(
        self, kind: str, grid: str, dim: int
    ) -> None:
        """⟨H c, m⟩ = ⟨c, Hᵀ m⟩, with H from SciPy and NumPy alone."""
        edges = GRIDS[grid][:dim]
        cells = Cells(edges)
        position, _ = _particles(edges)
        n_cells = int(np.prod(cells.shape))
        rows = [
            READS[kind](edges, unit.reshape(cells.shape), position)
            for unit in np.eye(n_cells)
        ]
        read = jnp.asarray(np.stack(rows, axis=1))
        assert_adjoint(
            lambda field: read @ field,
            lambda weight: deposit(cells, position, weight, kind=kind).ravel(),
            n_cells,
            len(position),
            key=jax.random.key(1),
            rtol=1e-12,
        )

    @given(
        # one size, so that one compilation serves every example
        widths=st.lists(st.floats(1e-3, 1.0), min_size=5, max_size=5),
        kind=st.sampled_from(KINDS),
    )
    def test_the_mass_is_kept_on_any_cells(
        self, widths: list[float], kind: str
    ) -> None:
        edge = np.cumsum([-1.0, *widths])
        # on every edge and on every centre
        position = np.concatenate([edge, 0.5 * (edge[1:] + edge[:-1])])
        sums = deposit(
            Cells((edge,)),
            jnp.asarray(position)[:, None],
            jnp.ones(position.size),
            kind=kind,
        )
        # all but the last edge
        assert_matches_oracle(sums.sum(), position.size - 1.0, rtol=1e-12)
        assert (sums >= 0.0).all()

    def test_an_edge_belongs_to_the_upper_cell(self) -> None:
        cells = Cells((STRETCHED[0],))
        position = jnp.asarray(STRETCHED[0])[:, None]
        counts = deposit(cells, position, jnp.ones(6))
        # one per cell, and the last edge is outside
        assert counts.tolist() == [1.0] * 5

    def test_an_outer_half_cell_falls_in_the_end_cell(self) -> None:
        cells = Cells(([0.0, 10.0, 20.0, 40.0],))
        position = jnp.array([[0.0], [4.9], [30.1], [39.9]])
        counts = deposit(cells, position, jnp.ones(4), kind="linear")
        assert_matches_oracle(counts, [2.0, 0.0, 2.0], rtol=1e-12, atol=1e-12)

    def test_one_cell_per_axis(self) -> None:
        cells = Cells(([0.0, 1.0], [0.0, 2.0]))
        position = jnp.array([[0.1, 0.1], [0.9, 1.9], [1.1, 0.0]])
        for kind in KINDS:
            assert deposit(cells, position, jnp.ones(3), kind=kind).tolist() == [[2.0]]

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_particle_outside_changes_nothing(self, kind: str) -> None:
        cells = Cells(STRETCHED)
        position, weight = _particles(STRETCHED)
        inside = jnp.asarray(_inside(STRETCHED, position))
        for far in (jnp.nan, jnp.inf, 1e30):
            moved = jnp.where(inside[:, None], position, far)
            # nor does its weight
            spoiled = jnp.where(inside, weight, jnp.nan)

            def loss(position: jax.Array, weight: jax.Array) -> jax.Array:
                return jnp.sum(deposit(cells, position, weight, kind=kind) ** 2)

            assert_matches_oracle(
                deposit(cells, moved, spoiled, kind=kind),
                deposit(cells, position, weight, kind=kind),
                rtol=0.0,
            )
            at_far = jax.grad(loss, argnums=(0, 1))(moved, spoiled)
            expected = jax.grad(loss, argnums=(0, 1))(position, weight)
            for got, want in zip(at_far, expected, strict=True):
                assert_matches_oracle(got, want, rtol=0.0)
                assert (got[~inside] == 0.0).all()

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", KINDS)
    def test_traces(self, kind: str) -> None:
        position, weight = _particles(STRETCHED, size=96)
        cells = Cells(STRETCHED)
        assert_traces(
            lambda position, weight: deposit(cells, position, weight, kind=kind),
            position.reshape(4, 24, 3),
            weight.reshape(4, 24),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", KINDS)
    def test_traces_in_the_edges(self, kind: str) -> None:
        position, weight = _particles(STRETCHED, size=96)
        edges = tuple(jnp.stack([edge, 2.0 * edge]) for edge in STRETCHED[:2])
        assert_traces(
            lambda *edges: deposit(Cells(edges), position[:, :2], weight, kind=kind),
            *edges,
        )

    @pytest.mark.parametrize("kind", KINDS)
    def test_concrete_cells_need_no_host_callback(self, kind: str) -> None:
        position, weight = _particles(STRETCHED)
        cells = Cells(STRETCHED)
        traced = jax.make_jaxpr(
            lambda position: deposit(cells, position, weight, kind=kind)
        )(position)
        assert "callback" not in str(traced)

    @pytest.mark.parametrize("grid", GRIDS)
    def test_gradients(self, grid: str) -> None:
        edges = GRIDS[grid]
        cells = Cells(edges)
        rng = np.random.default_rng(2)
        # away from the nodes: between the first two centres of an axis, at a
        # fraction inside (0.2, 0.8), or in the outer half cells
        position = []
        for edge in edges:
            centres = 0.5 * (edge[1:] + edge[:-1])
            between = centres[0] + rng.uniform(0.2, 0.8, 8) * (centres[1] - centres[0])
            outer = edge[0] + rng.uniform(0.2, 0.8, 4) * (centres[0] - edge[0])
            position.append(rng.permutation(np.concatenate([between, outer])))
        position = jnp.asarray(np.stack(position, axis=1))
        weight = jnp.asarray(rng.uniform(0.5, 2.0, 12))
        assert_grad_matches_fd(
            lambda position: deposit(cells, position, weight, kind="linear"), position
        )
        for kind in KINDS:
            assert_grad_matches_fd(
                lambda weight, kind=kind: deposit(cells, position, weight, kind=kind),
                weight,
            )
        # nearest is constant between the edges
        slope = jax.jacrev(lambda position: deposit(cells, position, weight))(position)
        assert (slope == 0.0).all()

    def test_the_gradient_is_zero_in_an_outer_half_cell(self) -> None:
        cells = Cells((STRETCHED[0],))
        field = jnp.arange(5.0) ** 2

        def loss(position: jax.Array) -> jax.Array:
            sums = deposit(cells, position, jnp.ones(3), kind="linear")
            return jnp.vdot(field, sums)

        slope = jax.grad(loss)(jnp.array([[-3.9], [-3.0], [3.9]]))
        assert slope[0, 0] == slope[2, 0] == 0.0
        # (c_1 − c_0) / (x_1 − x_0)
        assert_matches_oracle(slope[1, 0], 1.0 / (-2.25 + 3.75), rtol=1e-12)

    @pytest.mark.parametrize("kind", KINDS)
    def test_float32(self, kind: str) -> None:
        position, weight = _particles(STRETCHED)
        cells = Cells(STRETCHED)
        exact = deposit(cells, position, weight, kind=kind)
        single = deposit(
            cells, position.astype(jnp.float32), weight.astype(jnp.float32), kind=kind
        )
        assert single.dtype == jnp.float32
        assert_matches_oracle(single, exact, rtol=1e-4, atol=1e-4)

    def test_a_float32_particle_on_an_edge_is_on_it(self) -> None:
        """0.7 is below 0.7 in float32, and still belongs to the upper cell."""
        edge = np.array([0.0, 0.3, 0.7, 0.9])
        position = jnp.asarray(edge, dtype=jnp.float32)[:, None]
        counts = deposit(Cells((edge,)), position, jnp.ones(4, jnp.float32))
        assert counts.tolist() == [1.0, 1.0, 1.0]

    @pytest.mark.parametrize("kind", KINDS)
    def test_cells_too_narrow_for_float32(self, kind: str) -> None:
        cells = Cells((5e6 + np.array([0.0, 0.1, 0.2]),))
        position = jnp.full((1, 1), 5e6)
        assert deposit(cells, position, jnp.ones(1), kind=kind).tolist() == [1.0, 0.0]
        with pytest.raises(ValidationError, match="too narrow for float32 positions"):
            deposit(
                cells, position.astype(jnp.float32), jnp.ones(1, jnp.float32), kind=kind
            )

    def test_a_width_that_float32_changes(self) -> None:
        """Increasing in float32 is not enough: the widths must survive."""
        edge = 1e5 + np.array([0.0, 0.02, 0.04, 0.06])
        assert (np.diff(edge.astype(np.float32)) > 0.0).all()
        position = jnp.full((1, 1), 1e5, jnp.float32)
        with pytest.raises(ValidationError, match="too narrow for float32 positions"):
            deposit(Cells((edge,)), position, jnp.ones(1, jnp.float32))

    @pytest.mark.parametrize("kind", KINDS)
    def test_numpy_positions_are_those_of_jax(self, kind: str) -> None:
        """Without x64 they are float32 from the start: 2 − 1e-10 is the edge."""
        cells = Cells(([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
        position, weight = np.array([[0.5, 1.9999999999]]), np.ones(1)
        # jaxtyping's hook, on in the tests only, refuses NumPy arrays
        bare = inspect.unwrap(deposit)
        assert bare(cells, position, weight, kind=kind).sum() == 1.0
        with jax.enable_x64(False):
            narrow = Cells(([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
            assert (bare(narrow, position, weight, kind=kind) == 0.0).all()
            assert not inspect.unwrap(Cells.contains)(narrow, position).any()

    def test_bad_arguments(self) -> None:
        cells = Cells(UNIFORM[:2])
        position, weight = _particles(UNIFORM)
        with pytest.raises(ValidationError, match="one column per axis of the cells"):
            deposit(cells, position, weight)
        with pytest.raises(ValidationError, match="one column per axis of the cells"):
            cells.contains(position)
        # jaxtyping's hook, on in the tests only, may refuse these first
        with pytest.raises((ValidationError, TypeError), match="weight"):
            deposit(cells, position[:, :2], weight[:-1])
        with pytest.raises((ValidationError, TypeError), match="kind"):
            deposit(cells, position[:, :2], weight, kind="cubic")  # ty: ignore


def _reference_cells(stored: dict[str, np.ndarray], axes: str) -> Cells:
    domains = [stored[f"domain_{axis}"] for axis in axes]
    return Cells([np.linspace(low, high, int(n) + 1) for low, high, n in domains])


def _layer(size: int = 64) -> ParticleState:
    """A turbulent column, 50 m deep, whose particles are released over 5 minutes."""
    state = in_box(
        jnp.zeros(3),
        jnp.array([10.0, 10.0, 50.0]),
        size,
        mass=3.0,
        release_time=release_times(0.0, 300.0, size),
        key=jax.random.key(3),
    )
    return with_turbulent_velocity(state, LinearSigma(), key=jax.random.key(4))


# wider than any particle goes in 10 minutes, and stretched
WIDE = Cells(
    (
        np.array([-4e3, -100.0, 0.0, 5.0, 20.0, 4e3]),
        np.array([-4e3, 3.0, 4e3]),
        np.array([0.0, 2.0, 5.0, 30.0, 50.0]),
    )
)
LAYER = Langevin(uniform(jnp.array([2.0, -1.0, 0.0])), LinearSigma(), Walls(0.0, 50.0))
RUN = {"t0": 0.0, "duration": 602.0, "step_size": 5.0}


class TestResidence:
    @pytest.mark.parametrize("name", CLOSURES)
    def test_a_run(self, name: str) -> None:
        """G2-reference: the residence, the concentration and the column."""
        stored = reference_values("residence")
        size = int(stored["size"])
        # the second key draws the velocities
        key, key_velocity = jax.random.split(jax.random.PRNGKey(int(stored["seed"])))
        state = at_point(stored["source"], size)
        state = with_turbulent_velocity(state, CLOSURES[name], key=key_velocity)
        constant, slope = jnp.asarray(stored["wind"])
        step = Langevin(
            uniform_in_space(lambda t: constant + t * slope),
            CLOSURES[name],
            Walls(0.0, float(stored["lid"])),
        )
        cells = _reference_cells(stored, "xyz")
        result = integrate(
            state,
            step,
            t0=0.0,
            duration=float(stored["duration"]),
            step_size=float(stored["step_size"]),
            key=key,
            accumulate=Residence(cells),
        )
        rate = float(stored["emission_rate"])
        steady = steady_concentration(result.accumulated, rate, size, cells)
        expected = stored[f"{name}_concentration"]
        assert_matches_oracle(steady, expected, rtol=1e-12)
        # H = C N V / Q
        volume = np.prod([float(edge[1] - edge[0]) for edge in cells.edges])
        residence = expected * size * volume / rate
        assert_matches_oracle(result.accumulated, residence, rtol=1e-12)
        depth = float(cells.edges[2][1] - cells.edges[2][0])
        assert_matches_oracle(
            column(steady, cells), expected.sum(axis=2) * depth, rtol=1e-12
        )

    def test_it_is_an_accumulator(self) -> None:
        assert isinstance(Residence(WIDE), Accumulator)

    @pytest.mark.parametrize("kind", KINDS)
    def test_between_walls_it_is_the_whole_time(self, kind: str) -> None:
        """∑ R_j = ∑ₚ (T − t_{r,p}), with every particle inside all along."""
        state = _layer()
        result = integrate(
            state, LAYER, key=KEY, accumulate=Residence(WIDE, kind=kind), **RUN
        )
        expected = float((602.0 - state.release_time).sum())
        assert_matches_oracle(result.accumulated.sum(), expected, rtol=1e-12)
        assert (result.accumulated >= 0.0).all()
        # released at once, ∑ R_j = N T
        at_once = eqx.tree_at(
            lambda state: state.release_time, state, jnp.zeros(state.n_particles)
        )
        result = integrate(
            at_once, LAYER, key=KEY, accumulate=Residence(WIDE, kind=kind), **RUN
        )
        assert_matches_oracle(result.accumulated.sum(), 64 * 602.0, rtol=1e-12)

    def test_with_an_open_side_it_is_the_active_time(self) -> None:
        """A particle counts until the step that takes it out, which it leaves."""
        times = jnp.array([0.0, 3.5, 1e4])
        state = at_point(jnp.array([0.5, 0.5]), 3, release_time=times)
        cells = Cells(([0.0, 10.0, 40.0], [0.0, 1.0]))
        result = integrate(
            state,
            Deterministic(uniform(jnp.array([2.0, 0.0]))),
            t0=0.0,
            duration=60.0,
            step_size=1.0,
            domain=Box([0.0, 0.0], [20.0, 1.0]),
            accumulate=Residence(cells),
        )
        # x = ½ + 2 k < 10 for k ≤ 4 and < 20 for k ≤ 9
        # and x = ½ + 2 (½ + k), from 3.5 s on: the first step lasts ½ s
        expected = np.array([4.0 + 4.5, 5.0 + 5.0])
        assert_matches_oracle(result.accumulated[:, 0], expected, rtol=1e-14)
        assert result.state.active.tolist() == [False, False, True]

    def test_two_axes_bin_every_height(self) -> None:
        options = {"key": KEY, **RUN}
        volume = integrate(_layer(), LAYER, accumulate=Residence(WIDE), **options)
        surface = integrate(
            _layer(), LAYER, accumulate=Residence(Cells(WIDE.edges[:2])), **options
        )
        assert_matches_oracle(
            surface.accumulated, volume.accumulated.sum(axis=2), rtol=1e-12
        )

    def test_where(self) -> None:
        """Only below 5 m, which is the sum of the two lowest layers."""

        class Below(eqx.Module):
            height: jax.Array

            def __call__(self, state: ParticleState) -> jax.Array:
                return state.position[:, 2] < self.height

        horizontal = Cells(WIDE.edges[:2])
        options = {"key": KEY, **RUN}
        volume = integrate(_layer(), LAYER, accumulate=Residence(WIDE), **options)

        # the condition may hold a traced height
        @jax.jit
        def run(height: jax.Array) -> jax.Array:
            near = Residence(horizontal, where=Below(height))
            return integrate(_layer(), LAYER, accumulate=near, **options).accumulated

        expected = volume.accumulated[..., :2].sum(axis=2)
        assert 0.0 < expected.sum() < 0.5 * volume.accumulated.sum()
        assert_matches_oracle(run(jnp.asarray(5.0)), expected, rtol=1e-12)

    @pytest.mark.parametrize("kind", KINDS)
    def test_an_inactive_particle_changes_nothing(self, kind: str) -> None:
        state = _layer()
        gone = state.active.at[::3].set(False)
        spoiled = eqx.tree_at(
            lambda state: (state.position, state.active),
            state,
            (jnp.where(gone[:, None], state.position, jnp.nan), gone),
        )

        def total(state: ParticleState, wind: jax.Array) -> jax.Array:
            step = Langevin(uniform(wind), LinearSigma(), Walls(0.0, 50.0))
            near = Residence(WIDE, kind=kind)
            options = {"key": KEY, **RUN, "duration": 60.0}
            result = integrate(state, step, accumulate=near, **options)
            return jnp.sum(result.accumulated**2)

        wind = jnp.array([2.0, -1.0, 0.0])
        kept = eqx.tree_at(lambda state: state.active, state, gone)
        got = jax.value_and_grad(total, argnums=1)(spoiled, wind)
        expected = jax.value_and_grad(total, argnums=1)(kept, wind)
        assert jnp.isfinite(expected[0]) and expected[0] > 0.0
        for left, right in zip(got, expected, strict=True):
            assert_matches_oracle(left, right, rtol=1e-12)

    def test_a_release_after_the_end_adds_nothing(self) -> None:
        state = at_point(jnp.array([1.0, 1.0, 10.0]), 4, release_time=1e4)
        result = integrate(state, LAYER, key=KEY, accumulate=Residence(WIDE), **RUN)
        assert (result.accumulated == 0.0).all()

    def test_a_backward_run(self) -> None:
        state = at_point(jnp.array([1.0, 1.0, 10.0]), 8, release_time=1000.0)
        options = RUN | {"t0": 1000.0, "duration": -602.0}
        result = integrate(state, LAYER, key=KEY, accumulate=Residence(WIDE), **options)
        assert_matches_oracle(result.accumulated.sum(), 8 * 602.0, rtol=1e-12)
        # upwind of the release: the wind is (2, −1)
        assert result.accumulated[:2].sum() > 0.8 * 8 * 602.0

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", KINDS)
    def test_traces(self, kind: str) -> None:
        near = Residence(WIDE, kind=kind)

        def run(noise: jax.Array, wind: jax.Array) -> Result:
            step = Langevin(uniform(wind), LinearSigma(), Walls(0.0, 50.0))
            options = RUN | {"duration": 30.0}
            return integrate(_layer(16), step, noise=noise, accumulate=near, **options)

        noise = jax.random.normal(KEY, (3, 6, 16, 3))
        winds = jnp.array([[1.0, 0.0, 0.0], [2.0, -1.0, 0.0], [0.0, 3.0, 0.0]])
        assert_traces(lambda noise, wind: run(noise, wind).accumulated, noise, winds)

    def test_no_host_callback_in_a_run(self) -> None:
        near = Residence(WIDE, kind="linear")
        traced = jax.make_jaxpr(
            lambda t0: (
                integrate(
                    _layer(8), LAYER, key=KEY, accumulate=near, **(RUN | {"t0": t0})
                ).accumulated
            )
        )(0.0)
        without = jax.make_jaxpr(
            lambda t0: (
                integrate(
                    _layer(8), LAYER, key=KEY, **(RUN | {"t0": t0})
                ).state.position
            )
        )(0.0)
        assert str(traced).count("callback") == str(without).count("callback")

    def test_float32(self) -> None:
        state = _layer()
        single = jax.tree.map(
            lambda leaf: (
                leaf.astype(jnp.float32) if eqx.is_inexact_array(leaf) else leaf
            ),
            state,
        )
        cells = Cells([edge.astype(jnp.float32) for edge in WIDE.edges])
        step = Langevin(
            uniform(jnp.array([2.0, -1.0, 0.0], jnp.float32)),
            LinearSigma(),
            Walls(0.0, 50.0),
        )
        result = integrate(single, step, key=KEY, accumulate=Residence(cells), **RUN)
        assert result.accumulated.dtype == jnp.float32
        expected = float((602.0 - state.release_time).sum())
        assert_matches_oracle(result.accumulated.sum(), expected, rtol=1e-5)
        # the total keeps the dtype of the edges
        total = Residence(WIDE).init()
        updated = Residence(WIDE).update(total, single, jnp.ones(64, jnp.float32))
        assert updated.dtype == total.dtype == jnp.float64

    def test_bad_arguments(self) -> None:
        with pytest.raises(ValidationError, match="two axes"):
            Residence(Cells(WIDE.edges[:1]))
        with pytest.raises((ValidationError, TypeError), match="kind"):
            Residence(WIDE, kind="cubic")  # ty: ignore
        drifters = at_point(jnp.zeros(2), 2)
        with pytest.raises(ValidationError, match="one column per axis of the cells"):
            Residence(WIDE).update(Residence(WIDE).init(), drifters, jnp.ones(2))


class TestConcentration:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("grid", GRIDS)
    def test_it_holds_the_mass_inside(self, grid: str, kind: str) -> None:
        position, mass = _particles(GRIDS[grid])
        state = at_point(jnp.zeros(3), len(position))
        state = eqx.tree_at(
            lambda state: (state.position, state.mass), state, (position, mass)
        )
        cells = Cells(GRIDS[grid])
        values = concentration(state, cells, t=0.0, kind=kind)
        inside = _inside(GRIDS[grid], position)
        held = jnp.vdot(values, cells.volumes())
        assert_matches_oracle(held, np.asarray(mass)[inside].sum(), rtol=1e-12)

    def test_only_what_is_active_and_released_by_then(self) -> None:
        times = jnp.array([0.0, 10.0, 20.0, 30.0])
        state = at_point(jnp.ones(2), 4, mass=4.0, release_time=times)
        state = eqx.tree_at(lambda state: state.active, state, times != 10.0)
        cells = Cells(([0.0, 2.0], [0.0, 4.0]))
        held = [
            float(concentration(state, cells, t=t)[0, 0]) for t in (-1.0, 0, 20, 99)
        ]
        assert held == [0.0, 1.0 / 8.0, 2.0 / 8.0, 3.0 / 8.0]
        # what does not count may be anywhere, and may weigh anything
        spoiled = eqx.tree_at(
            lambda state: (state.position, state.mass),
            state,
            (
                jnp.where(times[:, None] == 10.0, jnp.nan, state.position),
                jnp.where(times == 10.0, jnp.nan, state.mass),
            ),
        )
        assert float(concentration(spoiled, cells, t=99.0)[0, 0]) == 3.0 / 8.0

    def test_a_continuous_source(self) -> None:
        """In a uniform wind with no turbulence, c = Q / (u A) behind the front."""
        rate, speed, duration, size = 0.5, 2.0, 100.0, 1000
        state = at_point(
            jnp.array([0.0, 0.5, 0.5]),
            size,
            mass=rate * duration,
            release_time=release_times(0.0, duration, size),
        )
        step = Deterministic(uniform(jnp.array([speed, 0.0, 0.0])))
        result = integrate(state, step, t0=0.0, duration=duration, step_size=10.0)
        cells = Cells(([0.0, 20.0, 60.0, 200.0, 220.0], [0.0, 1.0], [0.0, 1.0, 3.0]))
        values = concentration(result.state, cells, t=duration)
        expected = rate / speed
        assert_matches_oracle(
            values[:, 0, 0], [expected, expected, expected, 0.0], rtol=1e-12
        )

    def test_the_steady_field_of_a_uniform_wind(self) -> None:
        """c = Q / (u A) everywhere downwind, on cells of unequal volume."""
        rate, speed = 0.5, 2.0
        # half a step of 1 m before the first edge, so that the midpoint of each
        # step's path is what is summed, and no particle ends on an edge
        state = at_point(jnp.array([-0.5, 0.5, 0.5]), 3)
        cells = Cells(([0.0, 20.0, 60.0, 200.0], [0.0, 1.0], [0.0, 1.0, 3.0]))
        result = integrate(
            state,
            Deterministic(uniform(jnp.array([speed, 0.0, 0.0]))),
            t0=0.0,
            # the last steps are after the particles have left
            duration=120.0,
            step_size=0.5,
            accumulate=Residence(cells, kind="nearest"),
        )
        steady = steady_concentration(result.accumulated, rate, 3, cells)
        assert_matches_oracle(steady[:, 0, 0], np.full(3, rate / speed), rtol=1e-12)
        # on unequal layers a column is the sum of c Δz
        assert_matches_oracle(
            column(steady, cells), steady @ np.array([1.0, 2.0]), rtol=1e-14
        )

    def test_gradients(self) -> None:
        state = with_turbulent_velocity(
            at_point(jnp.array([0.0, 0.0, 12.0]), 6), Homogeneous(), key=KEY
        )
        cells = Cells(
            (np.array([-50.0, 0.0, 20.0, 90.0]), [-60.0, 5.0, 60.0], [0.0, 20.0, 50.0])
        )

        def steady(params: jax.Array) -> jax.Array:
            step = Langevin(uniform(params[:3]), Homogeneous(), Walls(0.0, 50.0))
            result = integrate(
                state,
                step,
                t0=0.0,
                duration=30.0,
                step_size=5.0,
                key=KEY,
                accumulate=Residence(cells, kind="linear"),
                adjoint=Checkpointed(every=2),
            )
            return steady_concentration(result.accumulated, params[3], 6, cells)

        assert_grad_matches_fd(steady, jnp.array([2.0, -1.0, 0.01, 0.5]))

    def test_gradients_of_a_snapshot(self) -> None:
        cells = Cells(STRETCHED)
        state = at_point(jnp.array([0.5, 1.0, 4.0]), 3, mass=2.0)

        def snapshot(masses: jax.Array) -> jax.Array:
            carried = eqx.tree_at(lambda state: state.mass, state, masses)
            return concentration(carried, cells, t=0.0, kind="linear")

        assert_grad_matches_fd(snapshot, state.mass)

    def test_traces(self) -> None:
        position, mass = _particles(STRETCHED, size=96)
        state = at_point(jnp.zeros(3), 24, release_time=release_times(0.0, 10.0, 24))
        cells = Cells(STRETCHED)

        def all_three(position: jax.Array, mass: jax.Array, t: jax.Array) -> Any:
            moved = eqx.tree_at(
                lambda state: (state.position, state.mass), state, (position, mass)
            )
            values = concentration(moved, cells, t=t)
            return (
                values,
                steady_concentration(values, t, 24, cells),
                column(values, cells),
            )

        times = jnp.array([1.0, 5.0, 9.0, 20.0])
        assert_traces(all_three, position.reshape(4, 24, 3), mass.reshape(4, 24), times)

    def test_the_dtype_is_the_positions(self) -> None:
        state = at_point(jnp.ones(3, jnp.float32), 4)
        cells = Cells(UNIFORM)
        values = concentration(state, cells, t=0.0)
        assert values.dtype == jnp.float32
        assert steady_concentration(values, 0.5, 4, cells).dtype == jnp.float32
        assert column(values, cells).dtype == jnp.float32

    def test_bad_arguments(self) -> None:
        cells = Cells(UNIFORM)
        values = jnp.zeros(cells.shape)
        with pytest.raises(ValidationError, match="'residence' must have one value"):
            steady_concentration(values[:-1], 1.0, 4, cells)
        with pytest.raises(ValidationError, match="'n_particles' must be an integer"):
            steady_concentration(values, 1.0, 0, cells)
        with pytest.raises((ValidationError, TypeError), match="rate"):
            steady_concentration(values, jnp.ones(2), 4, cells)
        with pytest.raises(ValidationError, match="three axes"):
            column(values, Cells(UNIFORM[:2]))
        with pytest.raises(ValidationError, match="'values' must have one value"):
            column(values[:-1], cells)
        drifters = at_point(jnp.zeros(2), 2)
        for times in (jnp.zeros(2), jnp.zeros((2, 1))):
            with pytest.raises((ValidationError, TypeError), match="'t'"):
                concentration(drifters, Cells(UNIFORM[:2]), t=times)
        with pytest.raises(ValidationError, match="one column per axis of the cells"):
            concentration(drifters, cells, t=0.0)
        with pytest.raises(ValidationError, match="two axes"):
            concentration(drifters, Cells(UNIFORM[:1]), t=0.0)
