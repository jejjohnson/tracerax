# Advection

What moves the particles. A step moves every particle over its own duration, and the loop that runs steps owns releases, masks and noise.

## Deterministic

`deterministic` is a solve: it carries particles along a velocity with a diffrax or integrix solver, and saves chosen times. `Deterministic` is the same scheme as a step, one explicit Runge–Kutta step per particle.

```python
import jax.numpy as jnp
from tracerax.advect import deterministic
from tracerax.particles import along_line


def rotation(t, position):
    return 1e-3 * jnp.stack([-position[:, 1], position[:, 0]], axis=-1)


state = along_line(jnp.array([0.0, 100.0]), jnp.array([0.0, 900.0]), 5)
final, saved = deterministic(
    state,
    rotation,
    t0=0.0,
    duration=3600.0,
    n_steps=120,
    save_at=jnp.array([0.0, 1800.0, 3600.0]),
)
```

Four things to know:

- **Backward.** A backward run has a negative `duration`, and its saved offsets are negative too. A particle is released once the clock has passed its release time in the run's direction, so a backward run usually takes `release_time` equal to its own start.
- **Releases.** In `deterministic` they must lie on the step times. A release inside a step needs `Deterministic` and the loop.
- **Forward mode** (`jax.jacfwd`) needs `adjoint=diffrax.ForwardMode()` in `deterministic`. `Deterministic` supports both modes as it is.
- **A domain's wall** is crossed at first order: a particle stops within one step of it.

::: tracerax.advect
