# Footprints

What a receptor saw. `surface` releases particles at a receptor, runs them backward, and turns the time they spend near the ground into the sensitivity of the receptor to the surface flux of each cell.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from tracerax.advect import Walls
from tracerax.bin import Cells
from tracerax.footprint import surface
from tracerax.velocity import uniform


class Isotropic(eqx.Module):
    sigma: float
    timescale: float

    def at(self, height):
        ones = jnp.ones((*height.shape, 3))
        return self.sigma * ones, self.timescale * ones


cells = Cells((jnp.linspace(-2000.0, 100.0, 22), jnp.linspace(-500.0, 500.0, 11)))


def footprint(receptor, time, key):
    return surface(
        receptor,
        uniform(jnp.array([3.0, 0.0, 0.0])),
        Isotropic(sigma=0.5, timescale=60.0),
        cells,
        n_particles=200,
        receptor_time=time,
        duration=600.0,
        step_size=5.0,
        surface_depth=100.0,
        air_density=1.2,
        walls=Walls(bottom=0.0, top=800.0),
        key=key,
    ).sensitivity


# one map per receptor
receptors = jnp.array([[0.0, 0.0, 10.0], [0.0, 200.0, 10.0], [0.0, 0.0, 60.0]])
times = jnp.array([3600.0, 3600.0, 7200.0])
keys = jax.random.split(jax.random.key(0), 3)
maps = jax.jit(jax.vmap(footprint))(receptors, times, keys)

# a receptor value is ∑ⱼ Fⱼ qⱼ, for a flux in kg m⁻² s⁻¹
flux = jnp.full(cells.shape, 1e-9)
enhancement = jnp.sum(maps * flux, axis=(1, 2))
```

- **Units.** The sensitivity is in s m² kg⁻¹ and multiplies a flux. There is no division by the cell area, which would give the sensitivity to an emission rate in kg s⁻¹.
- **Time.** `duration` is positive, in seconds back from the receptor, and static. `receptor_time` may be traced, so one compiled run serves every time.
- **Velocity.** Pass the wind forward in time; the run reverses it. A backward step reads it at the start of the forward interval it undoes, so the mean wind is reversed exactly.
- **Turbulence.** Homogeneous turbulence is reversed in law. In inhomogeneous turbulence the backward drift is not the forward one with the wind reversed (Flesch et al., 1995), and this run is approximate.
- **The surface layer** is `surface_depth` deep above `walls.bottom`. The air density is an argument: `geophysix.thermo.dry_air_density` gives it.
- **Gradients.** With `kind="linear"` a footprint is differentiable in the horizontal wind and in the receptor's x and y; with `"nearest"` that gradient is zero. Whether a particle is in the surface layer is a step that no gradient flows through, so the gradient in what moves particles across its top (the receptor's height, the vertical wind, the closure, the depth) is not that of the footprint.

::: tracerax.footprint
