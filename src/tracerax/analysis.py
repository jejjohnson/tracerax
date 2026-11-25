"""Analyses of trajectories, on the plane and on the sphere."""

from __future__ import annotations

from tracerax._src.analysis.dispersion import (
    absolute_dispersion,
    pair_dispersion,
)
from tracerax._src.analysis.ftle import ftle
from tracerax._src.analysis.regions import (
    Regions,
    RegionTime,
    connectivity,
    visited_connectivity,
)
from tracerax._src.analysis.skill import (
    cumulative_separation,
    separation,
    skill_score,
)


__all__ = [
    "RegionTime",
    "Regions",
    "absolute_dispersion",
    "connectivity",
    "cumulative_separation",
    "ftle",
    "pair_dispersion",
    "separation",
    "skill_score",
    "visited_connectivity",
]
