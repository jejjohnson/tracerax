# Analysis

What trajectories say: where particles stay, where they go, and how far apart they get.

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

::: tracerax.analysis
