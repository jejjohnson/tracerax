# tracerax

> Lagrangian particles in JAX: advection, dispersion, deposits, footprints and trajectory analysis.

tracerax is the GeoStack's library of Lagrangian particles: it carries a set of positions forward or backward in time, and analyses such positions. It will hold particle states and seeding, velocities read from a grid, deterministic, Langevin and random-displacement steps, deposits in cells, surface footprints from backward runs, and trajectory analysis.

It is written in JAX: every function is pure, its shapes are static, and the caller calls `jax.jit` once, at the top. Particles are masked, never dropped, so a run can be differentiated.

tracerax writes no interpolation, binning or time stepper of its own. [functax](https://github.com/jejjohnson/functax) reads the velocity and bins the particles, and [integrix](https://github.com/jejjohnson/integrix) gives the solvers and the adjoints. Turbulence closures, plume physics and source inversions are [plumax](https://github.com/jejjohnson/plumax)'s.

## Installation

tracerax is not on PyPI yet, and neither are [functax](https://github.com/jejjohnson/functax) and [integrix](https://github.com/jejjohnson/integrix), which it builds on, so it installs from a clone.

```bash
git clone https://github.com/jejjohnson/tracerax.git
cd tracerax
uv run poe install
```

## Status

The package has its scaffold, and no public module is built yet. The [API overview](https://jejjohnson.github.io/tracerax/reference/) lists the modules and the wave that builds each.

## Design rules

1. **One loop, several steps.** What moves a state is a step; the loop owns releases, durations, masks, noise, accumulators, saving and the adjoint.
2. **Masked, never dropped.** Shapes are static, and a masked particle keeps finite values.
3. **Times are offsets.** A run has a start, which may be traced, and a signed duration, negative for a backward run.
4. **One key per run**, or the noise itself.
5. **The output dtype follows the positions' dtype**, and nothing enables x64.
6. **Fail with a typed exception.** Everything raised derives from [`TraceraxError`](xref:api#tracerax.TraceraxError) and from the closest standard-library exception.
