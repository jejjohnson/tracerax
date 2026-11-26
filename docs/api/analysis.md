# Analysis

What runs of particles say: where particles stay, where they go, and how fast neighbours separate.

## Regions

A `Regions` is an integer label per cell of a `Cells`, with −1 for none. `RegionTime` sums the time each particle spends in each region while the loop runs, so no trajectory is stored, and `connectivity` is the fraction of each region's particles that end in each region.

```python
import jax
import jax.numpy as jnp
from tracerax.advect import ConstantDiffusivity, RandomDisplacement, integrate
from tracerax.analysis import Regions, RegionTime, connectivity
from tracerax.bin import Cells
from tracerax.particles import in_box
from tracerax.velocity import uniform

# a lagoon and the sea
cells = Cells(([0.0, 2_000.0, 10_000.0], [0.0, 5_000.0]))
regions = Regions(cells, jnp.array([[0], [1]]), n_regions=2)
key_start, key_run = jax.random.split(jax.random.key(0))
state = in_box(jnp.zeros(2), jnp.array([2_000.0, 5_000.0]), 500, key=key_start)
step = RandomDisplacement(
    uniform(jnp.array([0.02, 0.0])), ConstantDiffusivity([5.0, 5.0])
)
result = integrate(
    state,
    step,
    t0=0.0,
    duration=86_400.0,
    step_size=300.0,
    key=key_run,
    accumulate=RegionTime(regions, n_particles=500),
)
residence_time = result.accumulated[:, 0].mean()
exchange = connectivity(state.position, result.state.position, regions)
```

- **The time is right to one step** per crossing: a step's whole duration goes to the region the particle is in after it.
- **`visited_connectivity`** counts every region a particle has spent time in, from the total of a `RegionTime`, and not only the one it ends in. The start is not a visit: a particle must be in its region after the first step.
- **Rows.** A row sums to less than one when particles end outside every region. A region without particles has a row of NaN.
- **Gradients.** All of these are piecewise constant in the positions, so their gradient is zero.
- **On the sphere** the cells are in degrees and the longitudes of a run are in [−180°, 180°), so give the edges in that range. Cells are half open, so a particle released at 180° is in none (release it at −180°), and one at the north pole is in the last cell only when its edge is above 90°.

## Lyapunov exponents

`ftle` is the rate at which two neighbours separate, from the Jacobian of the flow map. The Jacobian is exact: each run carries one tangent per position column, so there are no neighbours to place and no spacing to tune.

```python
from tracerax.analysis import ftle


def strain(t, position):
    return 1e-5 * position * jnp.array([1.0, -1.0])


x, y = jnp.meshgrid(jnp.linspace(0.0, 1e4, 20), jnp.linspace(0.0, 1e4, 20))
seeds = jnp.column_stack([x.ravel(), y.ravel()])
exponent = ftle(seeds, strain, t0=0.0, duration=86_400.0, n_steps=96)
```

- **A gridded velocity** has the gradient of its scheme. In a double gyre on an 81 × 41 grid at 61 times the median error is 9e-4 s⁻¹ with a linear read and 2e-4 s⁻¹ with a cubic one, against exponents of 0.009 to 0.463 s⁻¹; the largest is 0.06 and 0.11 s⁻¹, on the ridges.
- **Forward** in time the ridges repel, and **backward** they attract.
- **On the sphere** (`earth=geophysix.earth.SPHERE`) seeds are longitudes and latitudes in degrees, and the Jacobian is taken in the local east–north frames at both ends. A seed may be at a pole: the velocity is then read 0.1 m from it in float64, and 2 km in float32.
- **Long runs.** The stretch $e^{\Lambda T}$ overflows when ΛT is large, in float32 from 88 on; `n_windows` splits the run and rescales.
- **Gradients.** `jax.grad` of the exponent works as it is, through the steps of the run. A seed at NaN spoils no other seed's gradient, but a path that leaves the data and reads a NaN does: clamp the field at its edge, which is `GriddedVelocity`'s default.

## Scores of trajectories

The separation of a trajectory from an observed one, its normalised cumulative form (Liu and Weisberg, 2011), the skill score, and the absolute and pair dispersion need no run: they take arrays of positions. They are in `diagnostix.trajectories`, beside the stack's other scores, and take the `trajectory` that `integrate` saves.

::: tracerax.analysis
