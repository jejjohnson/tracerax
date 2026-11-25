# API Reference

The public API is the set of public modules below, plus the exceptions that `tracerax.__all__` exports. Private code lives in `tracerax._src`. The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each.

## Modules

| Module | Contents | Status |
|---|---|---|
| [`tracerax.particles`](particles.md) | The particle state, the step protocol, seeding at points, along lines and over boxes | available |
| [`tracerax.velocity`](velocity.md) | Velocities at particles: uniform, gridded through functax, weighted sums | available |
| [`tracerax.advect`](advect.md) | Deterministic steps on the plane and on the sphere, Langevin, random-displacement and inertial steps, walls, the domain, and the loop that runs any step | available |
| [`tracerax.bin`](bin.md) | Cells from edges, deposits that conserve mass at walls, residence, concentrations | available |
| [`tracerax.footprint`](footprint.md) | Surface footprints from backward runs | available |
| [`tracerax.analysis`](analysis.md) | Residence times and connectivity of labelled regions; FTLE, dispersion and trajectory skill are next | available |
| [Type aliases](typing.md) | `ScalarLike` | available |
| [Exceptions](exceptions.md) | `TraceraxError`, `ValidationError` | available |

## Package overview

::: tracerax
    options:
      members: false
      show_root_heading: false
      show_source: false
