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

## The sphere

`deterministic_sphere` and `DeterministicSphere` move particles in longitude and latitude, in degrees, along an eastward and northward velocity in m s⁻¹. The step works on unit vectors, so a path may cross the date line and the poles.

```python
from tracerax.advect import DeterministicSphere, deterministic_sphere
from tracerax.particles import at_point
from tracerax.velocity import uniform

drifter = at_point(jnp.array([179.9, 0.0]), 1)
current = uniform(jnp.array([1.0, 0.0]))
final, _ = deterministic_sphere(drifter, current, t0=0.0, duration=86400.0, n_steps=24)
```

- **Longitudes come back in [−180°, 180°)**, so a path across the date line jumps by 360°. Distances along it are great-circle distances (`geophysix.earth.great_circle_distance`), never differences of longitude.
- **A global grid** needs a periodic longitude axis, `functax.geometry.Axis(..., period=360.0)`. A regional grid must have the longitudes of the particles, in [−180°, 180°), until [functax#170](https://github.com/jejjohnson/functax/issues/170): with a period it is read across its gap.
- **Near a pole** the two components turn quickly with longitude, so the read of a grid is the limit and not the step. Within 0.1 m of a pole in float64, and 2 km in float32, the velocity is read at that distance from it, so that its derivatives are right. The derivative of a longitude is not defined at a pole itself.
- **Float32** resolves 1.7 m of longitude, and a step's rounding can have the same sign at every step. `DeterministicSphere` returns to longitude and latitude at every step: at 6 m per step a month's path was 4 % off, and at 1.5 m per step a particle did not move at all. `deterministic_sphere` steps unit vectors, which resolve 0.4 m. Take steps that move particles by hundreds of metres, or float64.
- **A `Box`** under the loop is in degrees, cannot cross the date line, and is half open: release a particle at −180° and not at 180°.
- **Depth** is not here: the two columns are all there is.

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

## Diffusion

A surface drifter has no turbulence closure; what a model gives is a diffusivity K, a constant or a field. `RandomDisplacement` follows the mean velocity and diffuses with it, in two or three dimensions.

```python
from tracerax.advect import ConstantDiffusivity, RandomDisplacement

drifter = RandomDisplacement(
    uniform(jnp.array([0.3, 0.1])), ConstantDiffusivity([10.0, 10.0])
)
```

Where K varies, the step adds the drift ∂K/∂x. Without it particles gather where K is small, at every step size, and a cloud that starts uniform does not stay so. The scheme takes K to be nearly linear over one step, so keep the step under 0.1 / max |∂²K/∂x²|. A `GriddedDiffusivity` reads K from fields as `GriddedVelocity` reads a velocity; a closure such as Smagorinsky's is computed on the model's grid and passed that way.

## Inertia

A buoyant object of finite size at the sea surface does not follow the water. `Inertial` integrates its own velocity by the Maxey–Riley equation for the surface ocean (Beron-Vera, Olascoaga and Miron, 2019), and `SlowManifold` is the velocity that tends to when the response time τ is short: a `Velocity`, which runs under any step.

```python
from tracerax.advect import Inertial, integrate

object_step = Inertial(
    uniform(jnp.array([0.3, 0.1])),
    uniform(jnp.array([8.0, 3.0])),
    buoyancy=0.6,
    response_time=600.0,
    air_weight=0.05,
    coriolis=1e-4,
)
state = at_point(jnp.zeros(2), 1)
# each memory on the slow manifold at its release, so that there is no transient
state = object_step.release(state, t0=0.0)
result = integrate(state, object_step, t0=0.0, duration=86400.0, step_size=900.0)
```

- **The step is exact in the drag and the rotation**, with the flow frozen at the middle of the step, so it is stable for any τ. In a cellular flow of 50 km and 0.5 m s⁻¹ the error after a day and 32 to 35 km is below 1.6 m at a step of 900 s, and falls about four times per halving.
- **`SlowManifold` is off by** 1.4 m at τ = 1 min, 116 m at 10 min and 1.9 km at 1 h in that flow.
- **The derivatives of the velocities enter**, so a gridded one needs a scheme with a continuous gradient, such as `functax.interp.Cubic`; under a linear read the forcing jumps at cell faces.
- **Forward only.** Backward, the drag grows an error like $e^{t/\tau}$, so `Inertial` refuses; run its `slow_manifold` under `Deterministic`.
- **On the sphere** run `SlowManifold(..., earth=SPHERE)` under `DeterministicSphere`. It takes its derivatives in the local eastward and northward directions, and f = 2Ω sin φ when `coriolis` is not given. `Inertial` is in the plane only.

## The loop

`integrate` runs any `ParticleStep`, forward or backward: a drifter, a diffusing tracer and a plume share one set of releases, masks, sums and saves.

```python
from tracerax.advect import integrate
from tracerax.particles import release_times

# a continuous release of 10 minutes, followed for half an hour
key_start, key_run = jax.random.split(jax.random.key(0))
state = at_point(
    jnp.array([0.0, 0.0, 10.0]),
    1000,
    mass=0.5 * 600.0,
    release_time=release_times(0.0, 600.0, 1000),
)
state = with_turbulent_velocity(state, closure, key=key_start)
result = integrate(
    state, step, t0=0.0, duration=1800.0, step_size=5.0, key=key_run, save_every=60
)
```

- **Time.** `duration` and `step_size` are static and `t0` may be traced, so one compiled run serves every start time. Inside a run times are offsets from `t0`. In float32, count times from an epoch near the run: float32 resolves 128 s at Unix-epoch seconds, and the loop raises when a concrete time is too coarse for the step.
- **Releases** are exact, also inside a step, because every particle has its own duration.
- **Noise** is drawn from one key, or passed in, one array per step: negate it for antithetic pairs, keep it for common random numbers.
- **Sums.** An `Accumulator` is called after every step with the state and how long each particle moved, so a run of thousands of steps stores sums and no trajectory.
- **Memory.** Reverse mode stores every carry; `integrix.adjoint.Checkpointed(every=k)` stores O(K/k + k) of them, least near k = √K. A trajectory adds one array of positions per save. Forward mode works with both.
- **Ensembles** are a `jax.vmap` over keys.

::: tracerax.advect
