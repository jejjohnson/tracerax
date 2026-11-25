# Cells and concentrations

What particles leave in cells. `Cells` are the boxes between edges, `deposit` adds one weight per particle to them, `Residence` sums a run, and the concentrations divide by the volumes.

## Deposits

```python
import jax.numpy as jnp
from tracerax.bin import Cells, deposit

# cells that grow with height
cells = Cells(
    (
        jnp.linspace(-100.0, 900.0, 21),
        jnp.linspace(-200.0, 200.0, 9),
        jnp.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0]),
    )
)
position = jnp.array([[5.0, 0.0, 3.0], [420.0, -60.0, 180.0]])
mass = jnp.array([0.5, 0.5])
deposited = deposit(cells, position, mass, kind="linear")
concentration = deposited / cells.volumes()
```

- **Kinds.** `"nearest"` is a histogram, whose gradient in the positions is zero. `"linear"` is cloud-in-cell, which splits a particle between the centres around it and is differentiable in the positions.
- **Mass.** Both keep the whole weight of every particle inside the edges, on stretched cells and at the walls: a particle within half a cell of the ground counts in full in the lowest cell. What is outside the edges is dropped.
- **Edges.** A cell holds its lower edge and not its upper one, so a particle on an inner edge belongs to the upper cell and one on the last edge is outside. This differs from `numpy.histogram` at the last edge.
- **Axes.** The cells have one axis per position column, in their order. For a map of a plume, pass two axes and `position[:, :2]`.
- **Precision.** Edges are compared in the positions' dtype, and a cell too narrow for it raises, so count both from a local origin in float32. A float32 sum also loses digits with the number of equal weights in one cell: of the order of 1e-3 of it at 10⁵ and 1e-2 at 10⁶. Cells built from concrete edges cost no check at run time.

## Residence and the steady field

A continuous source is a superposition of puffs, so its steady field comes from one puff: release N particles at once, sum the time they spend in each cell, and scale. `Residence` is an `Accumulator` of `integrate`, so the run stores one array and no trajectory.

```python
import equinox as eqx
import jax
from tracerax.advect import Langevin, Walls, integrate
from tracerax.bin import Residence, column, steady_concentration
from tracerax.particles import at_point, with_turbulent_velocity
from tracerax.velocity import uniform


class Isotropic(eqx.Module):
    sigma: float
    timescale: float

    def at(self, height):
        ones = jnp.ones((*height.shape, 3))
        return self.sigma * ones, self.timescale * ones


closure = Isotropic(sigma=0.5, timescale=60.0)
key_start, key_run = jax.random.split(jax.random.key(0))
state = at_point(jnp.array([0.0, 0.0, 10.0]), 1000)
state = with_turbulent_velocity(state, closure, key=key_start)
step = Langevin(
    uniform(jnp.array([3.0, 0.0, 0.0])), closure, Walls(bottom=0.0, top=200.0)
)
result = integrate(
    state,
    step,
    t0=0.0,
    duration=600.0,
    step_size=5.0,
    key=key_run,
    accumulate=Residence(cells),
)
# a source of 0.5 kg s⁻¹
steady = steady_concentration(result.accumulated, 0.5, 1000, cells)
total_column = column(steady, cells)
```

- **When it holds.** The release is instantaneous, the velocity steady, and the run long enough for the particles to leave the cells.
- **What counts.** A particle adds the time it moved for, at its position after the move, if it is released and active then. The step that takes it out of the domain is not counted.
- **Maps.** Cells with two axes are (x, y) and bin every height together; `where` adds a condition, such as a height below which a particle counts.
- **The lid.** The last edge is outside, so a particle folded exactly onto a lid at the last edge is not counted. Put that edge just above the lid.
- **Gradients.** With `kind="linear"` a residence is differentiable in the positions. The condition `where` is a step, and nothing flows through it: the gradient is that of the particles that are counted, with the count held fixed.
- **Precision.** The total has the dtype of the edges.

## A snapshot

`concentration` deposits the masses of the active particles released by the time `t`. A source that has been on for a time T is a release spread over it, with the total mass Q T.

```python
from tracerax.bin import concentration
from tracerax.particles import release_times

plume = at_point(
    jnp.array([0.0, 0.0, 10.0]),
    1000,
    mass=0.5 * 600.0,
    release_time=release_times(0.0, 600.0, 1000),
)
plume = integrate(plume, step, t0=0.0, duration=600.0, step_size=5.0, key=key_run)
snapshot = concentration(plume.state, cells, t=600.0)
```

::: tracerax.bin
