# tracerax

[![Tests](https://github.com/jejjohnson/tracerax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/tracerax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/tracerax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/tracerax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/tracerax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/tracerax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/tracerax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/tracerax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/tracerax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/tracerax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)


Lagrangian particles in JAX: advection, dispersion, deposits, footprints and trajectory analysis.

tracerax is the GeoStack's library of Lagrangian particles: it carries a set of positions forward or backward in time, and analyses such positions. It holds particle states and seeding, velocities read from a grid, and deterministic, Langevin and random-displacement steps under one loop, and deposits in cells, residence times and concentrations. Surface footprints from backward runs and trajectory analysis come next.

It is written in JAX: every function is pure, its shapes are static, and the caller calls `jax.jit` once, at the top. Particles are masked, never dropped, so a run can be differentiated.

tracerax writes no interpolation, binning or time stepper of its own. [functax](https://github.com/jejjohnson/functax) reads the velocity and bins the particles, and [integrix](https://github.com/jejjohnson/integrix) gives the solvers and the adjoints. Turbulence closures, plume physics and source inversions are [plumax](https://github.com/jejjohnson/plumax)'s.

## Install

tracerax is not on PyPI yet, and neither are functax and integrix, so it installs from a clone:

```bash
git clone https://github.com/jejjohnson/tracerax.git
cd tracerax
uv run poe install
```

## Status

The modules marked *available* are built. The order is:

| Module | Holds | Wave |
|---|---|---|
| `particles` | the particle state, the step protocol, seeding at points, along lines and over boxes | *available* |
| `velocity` | velocities at particles: uniform, gridded through functax, weighted sums | *available* |
| `advect` | deterministic, Langevin and random-displacement steps, walls, and the loop that runs any step | *available* |
| `bin` | cells from edges, deposits that conserve mass at walls, residence, concentrations | *available* |
| `footprint` | surface footprints from backward runs | [Wave 2](https://github.com/jejjohnson/tracerax/issues/10) |
| `analysis` | FTLE, residence times, connectivity, dispersion, trajectory skill; the sphere and inertial particles join `advect` | [Wave 3](https://github.com/jejjohnson/tracerax/issues/11) |

## Design

The backlog is [geostack#304](https://github.com/jejjohnson/geostack/issues/304), the decisions and the measurements behind them are in [#14](https://github.com/jejjohnson/tracerax/issues/14), and the design page is in [geostack](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/tracerax.md).
