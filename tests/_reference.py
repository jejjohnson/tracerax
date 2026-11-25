"""Runs of the particle model written out in NumPy, for the tests to compare with.

Each family is one small run: its inputs, and what tracerax must return from
them, computed here with NumPy and SciPy and none of tracerax's code. The runs
come from plumax's particle model at ``96449d2``, whose stored values they
reproduced to 1e-12 until plumax took its particles from tracerax
(jejjohnson/plumax#410). JAX only draws the random numbers.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import numpy as np
from _closures import Homogeneous, LinearSigma
from scipy.interpolate import RegularGridInterpolator


__all__ = ["CLOSURES", "FAMILIES", "closure_statistics", "fold", "reference_values"]

#: The closures, by the name their arrays carry.
CLOSURES = {"homogeneous": Homogeneous(), "linear": LinearSigma()}

#: The families, in `reference_values`.
FAMILIES = ("binning", "footprint", "langevin", "residence", "trajectory", "wind")

#: The reflecting lid [m], the timescale of both closures [s], and the number of
#: particles.
LID, TIMESCALE, SIZE = 50.0, 60.0, 200

#: u(t) = u₀ + t a [m s⁻¹], the uniform wind of the runs.
WIND = np.array([[3.0, -1.0, 0.0], [0.01, 0.005, 0.0]])

type Arrays = dict[str, np.ndarray]


# --- the model --------------------------------------------------------------


def closure_statistics(name: str, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """σ(z), shape (n, 3), and ∂σ_w²/∂z, shape (n,), written out."""
    if name == "homogeneous":
        return np.full((*height.shape, 3), 0.5), np.zeros_like(height)
    sigma_vertical = 0.2 + 0.001 * height
    horizontal = np.full_like(height, 0.4)
    sigma = np.stack([horizontal, horizontal, sigma_vertical], axis=-1)
    return sigma, 2.0 * 0.001 * sigma_vertical


def fold(height: np.ndarray, lid: float) -> tuple[np.ndarray, np.ndarray]:
    """The triangle-wave fold onto [0, h], and the sign it gives the velocity."""
    wrapped = np.mod(height, 2.0 * lid)
    inside = np.where(wrapped > lid, 2.0 * lid - wrapped, wrapped)
    crossings = np.floor(height / lid).astype(int)
    return inside, np.where(crossings % 2 == 0, 1.0, -1.0)


def _langevin(
    name: str,
    position: np.ndarray,
    velocity: np.ndarray,
    wind: np.ndarray,
    noise: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One well-mixed Langevin step with folded walls."""
    sigma, variance_slope = closure_statistics(name, position[:, 2])
    decay = np.exp(-step_size / TIMESCALE)
    # v ← e^{−Δt/τ} v + σ √(1 − e^{−2Δt/τ}) ξ
    velocity = decay * velocity + sigma * np.sqrt(1.0 - decay**2) * noise
    vertical = velocity[:, 2]
    # ½ (1 + w²/σ_w²) ∂σ_w²/∂z Δt
    drift = 0.5 * (1.0 + vertical**2 / sigma[:, 2] ** 2) * variance_slope * step_size
    velocity[:, 2] += drift
    position = position + (wind + velocity) * step_size
    position[:, 2], sign = fold(position[:, 2], LID)
    velocity[:, 2] *= sign
    return position, velocity


def _steps(duration: float, step_size: float) -> np.ndarray:
    """Step lengths that sum to the duration, the last one shorter."""
    num_steps = int(np.ceil(duration / step_size))
    durations = np.full(num_steps, step_size)
    durations[-1] = duration - step_size * (num_steps - 1)
    return durations


def _wind_at(time: float) -> np.ndarray:
    return WIND[0] + time * WIND[1]


def _edges(domain: tuple[float, float, int]) -> np.ndarray:
    start, stop, cells = domain
    return np.linspace(start, stop, cells + 1)


def _normal(key: Any) -> np.ndarray:
    return np.array(jax.random.normal(key, (SIZE, 3)))


def _layer(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions uniform in the layer, and velocities drawn at their height."""
    key_position, key_velocity = jax.random.split(jax.random.PRNGKey(seed))
    low, high = np.array([-50.0, -50.0, 0.0]), np.array([50.0, 50.0, LID])
    position = low + (high - low) * np.asarray(
        jax.random.uniform(key_position, (SIZE, 3))
    )
    sigma, _ = closure_statistics(name, position[:, 2])
    return position, sigma * _normal(key_velocity)


def _release(
    name: str, source: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Positions and drawn velocities at a point, and the key of the steps."""
    key_steps, key_velocity = jax.random.split(jax.random.PRNGKey(seed))
    sigma, _ = closure_statistics(name, source[2:])
    return np.tile(source, (SIZE, 1)), _normal(key_velocity) * sigma, key_steps


# --- the families -----------------------------------------------------------


def _langevin_family() -> Arrays:
    """One step of 10 s per closure, in a wind that rises."""
    key = jax.random.PRNGKey(1)
    step_size, wind, noise = 10.0, np.array([3.0, -1.0, 0.05]), _normal(key)
    out = {
        "key": np.asarray(key),
        "noise": noise,
        "wind": wind,
        "step_size": np.asarray(step_size),
        "lid": np.asarray(LID),
    }
    for name in CLOSURES:
        start, start_velocity = _layer(name, seed=0)
        moved, velocity = _langevin(name, start, start_velocity, wind, noise, step_size)
        out |= {
            f"{name}_start_position": start,
            f"{name}_start_velocity": start_velocity,
            f"{name}_position": moved,
            f"{name}_velocity": velocity,
        }
    return out


def _trajectory_family() -> Arrays:
    """Nine steps in a wind that varies in time, the last one short."""
    key = jax.random.PRNGKey(2)
    t0, duration, step_size = 100.0, 42.0, 5.0
    durations = _steps(duration, step_size)
    noise = np.stack([_normal(k) for k in jax.random.split(key, durations.size)])
    out = {
        "key": np.asarray(key),
        "noise": noise,
        "wind": WIND,
        "t0": np.asarray(t0),
        "duration": np.asarray(duration),
        "step_size": np.asarray(step_size),
        "lid": np.asarray(LID),
    }
    for name in CLOSURES:
        position, velocity = _layer(name, seed=3)
        path, start_velocity = [position], velocity
        for step, length in enumerate(durations):
            wind = _wind_at(t0 + step * step_size)
            position, velocity = _langevin(
                name, position, velocity, wind, noise[step], length
            )
            path.append(position)
        out |= {
            f"{name}_start_velocity": start_velocity,
            f"{name}_trajectory": np.stack(path),
            f"{name}_velocity": velocity,
        }
    return out


def _counts(position: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """Right-open cells, the last one included: not histogramdd, which closes it."""
    counts = np.zeros([len(axis) - 1 for axis in edges])
    for point in position:
        index = [
            int(np.searchsorted(axis, value, side="right")) - 1
            for axis, value in zip(edges, point, strict=True)
        ]
        if all(0 <= i < n for i, n in zip(index, counts.shape, strict=True)):
            counts[tuple(index)] += 1.0
    return counts


def _binning_family() -> Arrays:
    """Counts on uniform and stretched edges, with particles on edges."""
    position, _ = _layer("homogeneous", seed=4)
    x_edges = np.linspace(-40.0, 40.0, 9)
    y_edges = np.linspace(-60.0, 60.0, 7)
    z_edges = np.linspace(0.0, 40.0, 6)
    # one on an inner edge of each axis, one on the last edge of x
    position[0] = (x_edges[3], y_edges[2], z_edges[1])
    position[1] = (x_edges[-1], 0.0, 1.0)
    grids = {
        "": [x_edges, y_edges, z_edges],
        # the same particles, the two on edges included, on cells that grow
        "stretched_": [
            np.array([-40.0, -35.0, -24.0, x_edges[3], 0.0, 2.0, 40.0]),
            np.array([-60.0, -50.0, y_edges[2], 10.0, 60.0]),
            np.array([0.0, 1.0, 3.0, z_edges[1], 20.0, 40.0]),
        ],
    }
    out = {"position": position}
    for prefix, edges in grids.items():
        out |= {
            f"{prefix}{axis}_edges": edge
            for axis, edge in zip("xyz", edges, strict=True)
        }
        out[f"{prefix}counts"] = _counts(position, edges)
    return out


def _residence_family() -> Arrays:
    """The steady concentration of a point source, C = Q R / (N V)."""
    domains = {"x": (-20.0, 580.0, 20), "y": (-160.0, 80.0, 12), "z": (0.0, LID, 5)}
    edges = [_edges(domain) for domain in domains.values()]
    source, rate = np.array([0.0, 0.0, 12.0]), 0.5
    duration, step_size, seed = 122.0, 5.0, 7
    durations = _steps(duration, step_size)
    out = {
        **{f"domain_{axis}": np.asarray(d, float) for axis, d in domains.items()},
        "source": source,
        "wind": WIND,
        "emission_rate": np.asarray(rate),
        "duration": np.asarray(duration),
        "step_size": np.asarray(step_size),
        "seed": np.asarray(seed),
        "lid": np.asarray(LID),
        "size": np.asarray(SIZE),
    }
    volume = np.prod([axis[1] - axis[0] for axis in edges])
    for name in CLOSURES:
        position, velocity, key = _release(name, source, seed)
        keys = jax.random.split(key, durations.size)
        residence = np.zeros([len(axis) - 1 for axis in edges])
        for step, length in enumerate(durations):
            wind = _wind_at(step * step_size)
            position, velocity = _langevin(
                name, position, velocity, wind, _normal(keys[step]), length
            )
            residence += _counts(position, edges) * length
        out[f"{name}_concentration"] = rate * residence / (SIZE * volume)
    return out


def _footprint_family() -> Arrays:
    """F = R / (N ρ h_s), in a steady wind and in one that varies in time."""
    domains = {"x": (-580.0, 20.0, 20), "y": (-80.0, 160.0, 12)}
    edges = [_edges(domain) for domain in domains.values()]
    receptor, seed = np.array([0.0, 0.0, 12.0]), 8
    t_back, dt, fraction, density, time = 122.0, 5.0, 0.5, 1.2, 300.0
    depth = fraction * LID
    # backward: the short step comes first, on the receptor's clock
    durations = _steps(t_back, dt)[::-1]
    starts = time - np.cumsum(durations)
    out = {
        **{f"domain_{axis}": np.asarray(d, float) for axis, d in domains.items()},
        "receptor": receptor,
        "wind": WIND,
        "size": np.asarray(SIZE),
        "t_back": np.asarray(t_back),
        "dt": np.asarray(dt),
        "pbl_height": np.asarray(LID),
        "pbl_fraction": np.asarray(fraction),
        "air_density": np.asarray(density),
        "receptor_time": np.asarray(time),
        "seed": np.asarray(seed),
    }
    for name in CLOSURES:
        for prefix in ("", "steady_"):
            position, velocity, key = _release(name, receptor, seed)
            keys = jax.random.split(key, durations.size)
            residence = np.zeros([len(axis) - 1 for axis in edges])
            for step, length in enumerate(durations):
                wind = -_wind_at(0.0 if prefix else starts[step])
                position, velocity = _langevin(
                    name, position, velocity, wind, _normal(keys[step]), length
                )
                seen = position[position[:, 2] < depth]
                residence += _counts(seen[:, :2], edges) * length
            out[f"{prefix}{name}_footprint"] = residence / (SIZE * density * depth)
    return out


def _wind_family() -> Arrays:
    """A gridded wind inside and outside its grid, in space and in time.

    u is half a cell east of the T points and v half a cell north, and the edge
    values are held outside.
    """
    rng = np.random.default_rng(5)
    x = 10.0 * np.arange(6) - 25.0
    y = 12.0 * np.arange(5) - 24.0
    z = 8.0 * np.arange(4) + 4.0
    times = np.array([0.0, 60.0, 90.0])
    fields = rng.normal(size=(3, times.size, z.size, y.size, x.size))
    position, _ = _layer("homogeneous", seed=6)
    # before the first time, between two, at a node, and after the last
    clock = np.array([-10.0, 15.0, 60.0, 100.0])
    half_x, half_y = 0.5 * (x[1] - x[0]), 0.5 * (y[1] - y[0])
    grids = ((times, z, y, x + half_x), (times, z, y + half_y, x), (times, z, y, x))
    velocity = np.empty((clock.size, SIZE, 3))
    for step, time in enumerate(clock):
        targets = np.column_stack([np.full(SIZE, time), position[:, ::-1]])
        for column, (grid, field) in enumerate(zip(grids, fields, strict=True)):
            low, high = [axis[0] for axis in grid], [axis[-1] for axis in grid]
            held = np.clip(targets, low, high)
            velocity[step, :, column] = RegularGridInterpolator(grid, field)(held)
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
        "velocity": velocity,
    }


_BUILDERS = {
    "binning": _binning_family,
    "footprint": _footprint_family,
    "langevin": _langevin_family,
    "residence": _residence_family,
    "trajectory": _trajectory_family,
    "wind": _wind_family,
}


@functools.cache
def _built(family: str) -> Arrays:
    return _BUILDERS[family]()


def reference_values(family: str) -> Arrays:
    """The inputs of a family's run and what must come out of it, by name."""
    # copies, so that no test changes another's
    return {name: values.copy() for name, values in _built(family).items()}
