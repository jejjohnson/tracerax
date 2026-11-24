# Particles

The state a run moves, the protocol of a step that moves it, and the seeders of a release.

A `ParticleState` holds positions in metres in a local frame, the step's memory, the mass each particle carries, when it starts, and whether it is still active. Particles are masked and never dropped, so shapes are static. A continuous release is a point with one release time per particle:

```python
import jax.numpy as jnp
import tracerax.particles as particles

source = jnp.array([0.0, 0.0, 10.0])
# 0.5 kg s⁻¹ for 10 minutes, over 1000 particles
state = particles.at_point(
    source,
    1000,
    mass=0.5 * 600.0,
    release_time=particles.release_times(0.0, 600.0, 1000),
)
```

::: tracerax.particles
