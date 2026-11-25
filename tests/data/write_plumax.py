"""Write, or check, the plumax values that tracerax's tests compare against.

plumax will import tracerax, so it cannot be a test dependency; its values are
stored once, float64, with the commit in each file's name. Every case is one
where a single reflection per wall is enough (jejjohnson/plumax#392), and the
script checks that. The files are deleted with Wave 4's switch, when plumax's
own tests take over.

Run it in a plumax checkout at that commit, from that checkout's root::

    git checkout --detach 96449d2
    uv run --no-default-groups --with cyclopts --with loguru \
        python <tracerax>/tests/data/write_plumax.py --check
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import cyclopts
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _closures import Homogeneous, LinearSigma


#: The plumax commit these values come from.
COMMIT = "96449d2"

#: The closures, by the name their arrays carry.
CLOSURES = {"homogeneous": Homogeneous(), "linear": LinearSigma()}

#: The reflecting lid [m], and the number of particles.
LID = 50.0
SIZE = 200

type Arrays = dict[str, np.ndarray]


def _state(closure: Any, seed: int) -> Any:
    """Particles uniform in the layer, with velocities drawn at their height."""
    from plumax.lagrangian import ParticleState

    key_position, key_velocity = jax.random.split(jax.random.PRNGKey(seed))
    low = jnp.array([-50.0, -50.0, 0.0])
    high = jnp.array([50.0, 50.0, LID])
    position = jax.random.uniform(key_position, (SIZE, 3), minval=low, maxval=high)
    sigma, _ = closure.at(position[:, 2])
    # v ~ 𝒩(0, σ²(z))
    velocity = sigma * jax.random.normal(key_velocity, (SIZE, 3))
    return ParticleState(position=position, velocity=velocity)


def _wind(t: Any) -> Any:
    """u(t) = u₀ + t a, the uniform wind of the runs; `_WIND` holds u₀ and a."""
    return jnp.asarray(_WIND[0]) + t * jnp.asarray(_WIND[1])


_WIND = np.array([[3.0, -1.0, 0.0], [0.01, 0.005, 0.0]])


def _check_one_reflection(height: Any) -> None:
    """Every height is inside the layer, so one reflection per wall was enough."""
    height = np.asarray(height)
    if not ((height >= 0.0) & (height <= LID)).all():
        raise RuntimeError("a particle left the layer: one reflection is not enough")


def langevin() -> Arrays:
    """One `langevin_step` per closure, with the noise it draws."""
    from plumax.lagrangian import langevin_step

    key = jax.random.PRNGKey(1)
    step_size = 10.0
    wind = jnp.array([3.0, -1.0, 0.05])
    out: Arrays = {
        "key": np.asarray(key),
        "noise": np.asarray(jax.random.normal(key, (SIZE, 3))),
        "wind": np.asarray(wind),
        "step_size": np.asarray(step_size),
        "lid": np.asarray(LID),
    }
    for name, closure in CLOSURES.items():
        state = _state(closure, seed=0)
        moved = langevin_step(state, wind, closure, step_size, key, pbl_height=LID)
        _check_one_reflection(moved.position[:, 2])
        out |= {
            f"{name}_start_position": np.asarray(state.position),
            f"{name}_start_velocity": np.asarray(state.velocity),
            f"{name}_position": np.asarray(moved.position),
            f"{name}_velocity": np.asarray(moved.velocity),
        }
    return out


def trajectory() -> Arrays:
    """`integrate_particles` in a wind that varies in time, with a short last step."""
    from plumax.lagrangian import integrate_particles

    key = jax.random.PRNGKey(2)
    t0, duration, step_size = 100.0, 42.0, 5.0
    num_steps = 9
    noise = [
        jax.random.normal(step_key, (SIZE, 3))
        for step_key in jax.random.split(key, num_steps)
    ]
    out: Arrays = {
        "key": np.asarray(key),
        "noise": np.stack([np.asarray(draw) for draw in noise]),
        "wind": _WIND,
        "t0": np.asarray(t0),
        "duration": np.asarray(duration),
        "step_size": np.asarray(step_size),
        "lid": np.asarray(LID),
    }
    for name, closure in CLOSURES.items():
        state = _state(closure, seed=3)
        final, positions = integrate_particles(
            state,
            _wind,
            closure,
            t0=t0,
            t1=t0 + duration,
            dt=step_size,
            key=key,
            pbl_height=LID,
            save_trajectory=True,
        )
        assert positions is not None and positions.shape == (num_steps + 1, SIZE, 3)
        _check_one_reflection(positions[..., 2])
        out |= {
            f"{name}_start_velocity": np.asarray(state.velocity),
            f"{name}_trajectory": np.asarray(positions),
            f"{name}_velocity": np.asarray(final.velocity),
        }
    return out


def binning() -> Arrays:
    """`bin_positions` on uniform and stretched edges, with particles on edges."""
    from plumax.lagrangian import bin_positions

    position = np.asarray(_state(CLOSURES["homogeneous"], seed=4).position).copy()
    x_edges = np.linspace(-40.0, 40.0, 9)
    y_edges = np.linspace(-60.0, 60.0, 7)
    z_edges = np.linspace(0.0, 40.0, 6)
    # one on an inner edge of each axis, one on the last edge of x
    position[0] = (x_edges[3], y_edges[2], z_edges[1])
    position[1] = (x_edges[-1], 0.0, 1.0)
    edges = (x_edges, y_edges, z_edges)
    counts = bin_positions(jnp.asarray(position), *map(jnp.asarray, edges))
    # the same particles, the two on edges included, on cells that grow
    stretched = (
        np.array([-40.0, -35.0, -24.0, x_edges[3], 0.0, 2.0, 40.0]),
        np.array([-60.0, -50.0, y_edges[2], 10.0, 60.0]),
        np.array([0.0, 1.0, 3.0, z_edges[1], 20.0, 40.0]),
    )
    stretched_counts = bin_positions(
        jnp.asarray(position), *map(jnp.asarray, stretched)
    )
    return {
        "position": position,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "counts": np.asarray(counts),
        "stretched_x_edges": stretched[0],
        "stretched_y_edges": stretched[1],
        "stretched_z_edges": stretched[2],
        "stretched_counts": np.asarray(stretched_counts),
    }


def _cells(domain: tuple[float, float, int]) -> np.ndarray:
    """A plumax domain (start, stop, cells) as an array."""
    return np.asarray(domain, dtype=np.float64)


def residence() -> Arrays:
    """`simulate_lagrangian`: the steady concentration from the residence times.

    C = Q H / (N V), so the residence time is H = C N V / Q.
    """
    from plumax.lagrangian import simulate_lagrangian

    domain_x, domain_y, domain_z = (-20.0, 580.0, 20), (-160.0, 80.0, 12), (0.0, LID, 5)
    source = (0.0, 0.0, 12.0)
    rate, horizon, step_size, seed = 0.5, 122.0, 5.0, 7
    out: Arrays = {
        "domain_x": _cells(domain_x),
        "domain_y": _cells(domain_y),
        "domain_z": _cells(domain_z),
        "source": np.asarray(source),
        "wind": _WIND,
        "emission_rate": np.asarray(rate),
        "duration": np.asarray(horizon),
        "step_size": np.asarray(step_size),
        "seed": np.asarray(seed),
        "lid": np.asarray(LID),
        "size": np.asarray(SIZE),
    }
    for name, closure in CLOSURES.items():
        dataset = simulate_lagrangian(
            rate,
            source,
            closure,
            domain_x,
            domain_y,
            domain_z,
            n_particles=SIZE,
            t_end=horizon,
            dt=step_size,
            wind=_wind,
            pbl_height=LID,
            seed=seed,
        )
        out[f"{name}_concentration"] = dataset["concentration"].values
    return out


def footprint() -> Arrays:
    """`compute_footprint` in a steady wind and in one that varies in time.

    Both have a short first step.
    """
    from plumax.lagrangian import compute_footprint

    domain_x, domain_y = (-580.0, 20.0, 20), (-80.0, 160.0, 12)
    receptor = (0.0, 0.0, 12.0)
    options = {
        "t_back": 122.0,
        "dt": 5.0,
        "pbl_height": LID,
        "pbl_fraction": 0.5,
        "air_density": 1.2,
        "receptor_time": 300.0,
        "seed": 8,
    }
    out: Arrays = {
        "domain_x": _cells(domain_x),
        "domain_y": _cells(domain_y),
        "receptor": np.asarray(receptor),
        "wind": _WIND,
        "size": np.asarray(SIZE),
        **{name: np.asarray(value) for name, value in options.items()},
    }
    winds = {"": _wind, "steady_": lambda t: jnp.asarray(_WIND[0])}
    for name, closure in CLOSURES.items():
        for prefix, wind in winds.items():
            values, _, _ = compute_footprint(
                receptor,
                closure,
                domain_x,
                domain_y,
                wind=wind,
                n_particles=SIZE,
                **options,
            )
            out[f"{prefix}{name}_footprint"] = values
    return out


def wind() -> Arrays:
    """`GriddedWind` inside and outside its grid, in space and in time."""
    from plumax.lagrangian import GriddedWind

    rng = np.random.default_rng(5)
    x = 10.0 * np.arange(6) - 25.0
    y = 12.0 * np.arange(5) - 24.0
    z = 8.0 * np.arange(4) + 4.0
    times = np.array([0.0, 60.0, 90.0])
    fields = rng.normal(size=(3, times.size, z.size, y.size, x.size))
    gridded = GriddedWind(*map(jnp.asarray, (x, y, z, times, *fields)))
    position = np.asarray(_state(CLOSURES["homogeneous"], seed=6).position)
    # before the first time, between two, at a node, and after the last
    clock = np.array([-10.0, 15.0, 60.0, 100.0])
    velocity = [gridded(jnp.asarray(position), time) for time in clock]
    return {
        "x": x,
        "y": y,
        "z": z,
        "times": times,
        "u": fields[0],
        "v": fields[1],
        "w": fields[2],
        "position": position,
        "t": clock,
        "velocity": np.stack([np.asarray(values) for values in velocity]),
    }


FAMILIES = (binning, footprint, langevin, residence, trajectory, wind)


def _require_commit() -> None:
    """Stop unless the plumax that is imported is a clean checkout at `COMMIT`."""
    import plumax

    root = Path(plumax.__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        command = ["git", "-C", str(root), *arguments]
        run = subprocess.run(command, check=True, capture_output=True, text=True)
        return run.stdout.strip()

    head = git("rev-parse", "HEAD")
    if not head.startswith(COMMIT):
        raise SystemExit(f"plumax in {root} is at {head[:7]}, not at {COMMIT}")
    if git("status", "--porcelain", "--", "src"):
        raise SystemExit(f"plumax in {root} has changes in src")


app = cyclopts.App(help=__doc__)


@app.default
def main(*, check: bool = False) -> None:
    """Write the files, or with ``--check`` compare them with a fresh run."""
    _require_commit()
    jax.config.update("jax_enable_x64", True)
    logger.info("plumax {} on JAX {}", COMMIT, jax.__version__)
    total = 0
    for family in FAMILIES:
        path = HERE / f"plumax_{family.__name__}_{COMMIT}.npz"
        arrays = family()
        if check:
            with np.load(path) as stored:
                assert set(stored) == set(arrays), f"{path.name}: other names"
                for name, values in arrays.items():
                    # 1e-12: rounding of another CPU; the counts are exact
                    np.testing.assert_allclose(
                        values, stored[name], rtol=1e-12, atol=1e-12, err_msg=name
                    )
        else:
            np.savez_compressed(path, **arrays)
        total += path.stat().st_size
        logger.success("{}: {} kB", path.name, path.stat().st_size // 1000)
    if total >= 500_000:
        raise SystemExit(f"{total} bytes stored, more than 500 kB")
    logger.success("{} kB in all", total // 1000)


if __name__ == "__main__":
    app()
