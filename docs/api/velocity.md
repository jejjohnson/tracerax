# Velocities

A velocity is any callable `velocity(t, position)` that returns one velocity per particle, at one time or at one time per particle. It is the argument order of diffrax and of `integrix.march`, so a velocity also runs there.

A C-grid keeps u, v and w at different points. `cgrid_grids` gives each component its own grid, so a particle reads the wind where the Eulerian solver puts it:

```python
import jax.numpy as jnp
import tracerax.velocity as velocity

times = jnp.array([0.0, 600.0])
x, y, z = 50.0 * jnp.arange(8.0), 50.0 * jnp.arange(6.0), jnp.array([5.0, 20.0, 60.0])
# each (time, z, y, x)
u, v, w = 3.0 * jnp.ones((3, 2, 3, 6, 8))
grids = velocity.cgrid_grids(times, x, y, z)
wind = velocity.GriddedVelocity.fit(grids, (u, v, w))
```

Four things to know about the read:

- Outside the grid the nearest sample is held, in space and in time, and the velocity's gradient in the position is zero there.
- A linear read is continuous and its gradient is not. With an adaptive step, fit with `scheme=functax.interp.Cubic()`.
- NaN on land needs `policies=functax.interp.Policies(outside="clamp", missing="renormalise")`.
- A periodic axis that covers only part of its period is read across the gap ([functax#170](https://github.com/jejjohnson/functax/issues/170)), so keep the longitudes of a regional grid and of its particles in one convention.

::: tracerax.velocity
