"""Tests for the cells and the deposits."""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_adjoint,
    assert_grad_matches_fd,
    assert_matches_oracle,
    assert_traces,
)
from _oracles import plumax_values
from hypothesis import given, strategies as st
from scipy.interpolate import RegularGridInterpolator

from tracerax import ValidationError
from tracerax.bin import (
    Cells,
    deposit,
)


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


class TestAgainstPlumax:
    @pytest.mark.parametrize("prefix", ["", "stretched_"])
    def test_bin_positions(self, prefix: str) -> None:
        stored = plumax_values("binning")
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
