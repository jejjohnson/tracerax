# API Reference

The public API is the set of public modules below, plus the exceptions that `tracerax.__all__` exports. Private code lives in `tracerax._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| `tracerax.particles` | The particle state, the step protocol, seeding at points, along lines and over boxes | Wave 1 |
| `tracerax.velocity` | Velocities at particles: uniform, gridded through functax, weighted sums | Wave 1 |
| `tracerax.advect` | Deterministic, Langevin and random-displacement steps, walls, and the loop that runs any step | Wave 1 |
| `tracerax.bin` | Cells from edges, deposits that conserve mass at walls, residence, concentrations | Wave 2 |
| `tracerax.footprint` | Surface footprints from backward runs | Wave 2 |
| `tracerax.analysis` | FTLE, residence times, connectivity, dispersion, trajectory skill; the sphere and inertial particles join `advect` | Wave 3 |
| [Type aliases](typing.md) | `ScalarLike` | available |
| [Exceptions](exceptions.md) | `TraceraxError`, `ValidationError` | available |

## Package overview

::: tracerax
    options:
      members: false
      show_root_heading: false
      show_source: false
