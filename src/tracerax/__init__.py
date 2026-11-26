"""Lagrangian particles in JAX.

``tracerax`` owns a calculation whose state is a set of positions carried
forward in time, or an analysis that needs such a run: where parcels go, how a
cloud disperses, what it leaves in a cell, and how fast neighbours separate.

## Modules

Each is created by its first feature issue, so the later ones may not be here yet.

| Module | What it gives you |
|---|---|
| `tracerax.particles` | the particle state, the step protocol, seeding |
| `tracerax.velocity` | velocities at particles, uniform, gridded or summed |
| `tracerax.advect` | deterministic, turbulent, diffusing and inertial steps, the loop |
| `tracerax.bin` | cells, deposits, residence, concentrations |
| `tracerax.analysis` | regions, connectivity and FTLE, on the plane and on the sphere |
"""

from __future__ import annotations

from tracerax.exceptions import TraceraxError, ValidationError


__version__ = "0.0.6"

__all__ = [
    "TraceraxError",
    "ValidationError",
    "__version__",
]
