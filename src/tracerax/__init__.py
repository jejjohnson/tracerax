"""Lagrangian particles in JAX.

``tracerax`` owns a calculation whose state is a set of positions carried
forward in time, or that analyses such positions: where parcels go, how a cloud
disperses, what it leaves in a cell, and which surface a receptor saw.

## Modules

Each is created by its first feature issue, so the later ones may not be here yet.

| Module | What it gives you |
|---|---|
| `tracerax.particles` | the particle state, the step protocol, seeding |
| `tracerax.velocity` | velocities at particles, uniform, gridded or summed |
| `tracerax.advect` | deterministic, Langevin and random-displacement steps, the loop |
| `tracerax.bin` | cells, deposits, residence, concentrations |
| `tracerax.footprint` | surface footprints from backward runs |
| `tracerax.analysis` | trajectory diagnostics, on the plane and on the sphere |
"""

from __future__ import annotations

from tracerax.exceptions import TraceraxError, ValidationError


__version__ = "0.0.1"

__all__ = [
    "TraceraxError",
    "ValidationError",
    "__version__",
]
