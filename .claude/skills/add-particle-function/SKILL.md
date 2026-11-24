---
name: add-particle-function
description: Add a function or an Equinox module to a public module of the package — whether functax, integrix or plumax already owns it, where it goes in _src/<module>/, its units and axis order, masks and keys, a Google-style docstring, the oracle, transform, gradient and mask tests, the API page and the capability index. Use when asked to add, port or implement a step, a velocity, a seeder, an accumulator, a deposit, a footprint, a trajectory diagnostic or a new module in src/tracerax.
---

# Add a particle function

Read "1. The conventions" and "2. Public functions and modules" under "The contracts" in `AGENTS.md` first; this is the step-by-step.

## 1. Make sure it belongs here, and does not exist yet

- Search `docs/api/capabilities.md` for it and its synonyms, and the "Reuse before you write" table in `AGENTS.md`.
- Is its state a set of positions carried forward in time, or does it analyse such positions? If not, it is not tracerax's: a scheme that discards its departure points is integrix's, interpolation and binning are functax's, closures and plume physics are plumax's.
- A gap in functax or integrix is an issue there, not a function here.
- A new way to move particles is a `ParticleStep` ([#16](https://github.com/jejjohnson/tracerax/issues/16)), run by the one loop ([#20](https://github.com/jejjohnson/tracerax/issues/20)), never a second loop.

## 2. Pick its home

Private code goes in `src/tracerax/_src/<module>/<topic>.py`; the public module (`particles`, `velocity`, `advect`, `bin`, `footprint`, `analysis`) only re-exports it. A new public module is created by its first feature issue. A helper that two modules need goes in `src/tracerax/_src/`. A new exception goes in `exceptions.py`, derives from `TraceraxError` and the closest standard-library exception, and joins `tests/test_exceptions.py`.

## 3. Write it

- Pure functions and `eqx.Module`s; arrays are leaves, configuration is `eqx.field(static=True)`. Never call `jax.jit`.
- Every array is annotated with jaxtyping and named dims (`Float[Array, "particle dim"]`); einx instead of `axis=`, `.T`, `reshape` and `[:, None]`.
- **Units**: SI, stated in `Args:` and `Returns:`. Positions are metres in a local frame, with columns (x, y) or (x, y, z); a gridded velocity keeps (time, z, y, x).
- **Masks**: a particle that is inactive or not yet released is unchanged and finite. Replace its value through `where` before the arithmetic, so that its gradient is finite too.
- **Time**: offsets from a traced `t0` over a static signed `duration`. Write down what a negative duration does.
- **Randomness**: `key` keyword-only and last, or the noise as an argument; split once, up front.
- **Precision**: constants in the positions' dtype; float32 in gives float32 out.
- Comments above the line with the Unicode equation above each mathematical line; descriptive names; one operation per line.
- Google-style docstring in the `python-style` layout: a one-line summary (it becomes the capability-index entry), the defining equation in Unicode (never `$…$`), `Args:` with meaning, symbol and unit, `Returns:`, `Raises:`, an `Examples:` block that prints shapes or `allclose` booleans, and `References:` when a paper defines the method.

## 4. Export and document

- Add the name to the public module's `__all__` (sorted); a new module joins `SUBMODULES` in `tests/test_public_api.py`.
- API page `docs/api/<module>.md` with a `::: tracerax.<module>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/test_<module>.py`, with the helpers of `tests/`:

- **Oracle.** A closed form (`tests/_oracles.py`) or a stored plumax value (`plumax_values`), in float64, with the tolerance and its source in a comment (`assert_matches_oracle`; `assert_adjoint` for a linear map). A sample goes through `assert_uniform`, `assert_mean` or `assert_variance` (`tests/_sampling.py`), with a constant key.
- **Transforms.** Eager, `jit` and `vmap` over a batch of one agree (`assert_traces`).
- **Gradients.** Autodiff against central differences at a fixed key, where the function is smooth (`assert_grad_matches_fd`).
- **Masks.** An inactive or unreleased particle changes nothing and stays finite, in value and in gradient.
- The edges: no active particle, a release inside a step, a release after the end, a particle on a wall, a backward run, float32 in and out, each invalid argument (`match=`).

## 6. Verify

```bash
uv run pytest --no-cov tests/test_<module>.py src/tracerax/_src/<module> tests/test_public_api.py tests/test_capabilities.py -v
uv run poe capabilities
uv run poe test-cov
```

then the `pre-pr-check` skill.
