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

## Turbulence

`Langevin` adds a turbulent velocity with a memory: a Markov-1 process whose standard deviation σ and Lagrangian time scale τ_L come from a closure you pass in. The closures are not here: any object whose `at(height)` returns σ and τ_L is a `TurbulenceModel`, and plumax's are.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from tracerax.advect import Langevin, Walls
from tracerax.particles import at_point, with_turbulent_velocity
from tracerax.velocity import uniform


class Isotropic(eqx.Module):
    sigma: float
    timescale: float

    def at(self, height):
        ones = jnp.ones((*height.shape, 3))
        return self.sigma * ones, self.timescale * ones


closure = Isotropic(sigma=0.5, timescale=60.0)
key_start, key_step = jax.random.split(jax.random.key(0))
state = at_point(jnp.array([0.0, 0.0, 10.0]), 1000)
state = with_turbulent_velocity(state, closure, key=key_start)
step = Langevin(
    uniform(jnp.array([3.0, 0.0, 0.0])), closure, Walls(bottom=0.0, top=800.0)
)
noise = jax.random.normal(key_step, step.noise_shape(state))
start, duration = jnp.zeros(1000), jnp.full(1000, 5.0)
moved = step(state, start, duration, 1.0, noise)
```

The linear part is integrated exactly, so homogeneous turbulence has no step-size bias in the velocity. The drift that keeps an uneven column well mixed is first order, so keep the step short against τ_L, and the position is first order too: use a step of τ_L / 12 or less. The walls fold a particle back any number of times, so none is ever outside.

::: tracerax.advect
