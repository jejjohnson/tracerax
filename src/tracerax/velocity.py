"""Velocities at particles: uniform, gridded through functax, and sums of them."""

from __future__ import annotations

from tracerax._src.velocity.combine import WeightedSum
from tracerax._src.velocity.gridded import GriddedVelocity, cgrid_grids
from tracerax._src.velocity.uniform import Velocity, uniform, uniform_in_space


__all__ = [
    "GriddedVelocity",
    "Velocity",
    "WeightedSum",
    "cgrid_grids",
    "uniform",
    "uniform_in_space",
]
