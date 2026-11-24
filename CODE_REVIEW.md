# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a backward wind with the wrong sign, a key used twice, a particle dropped instead of masked, mass lost at a wall, a constant that promotes float32, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented interpolation, binning or cell area functax has, or a solver, checkpointed scan or truncated scan integrix has, is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the library imports functax and integrix, and never geotoolz, pipekit, gaussx, kernellib or any model at runtime.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **JAX only.** Pure functions and Equinox modules; nothing calls `jax.jit`.
- **Static shapes.** Particles are masked, never dropped; a masked particle keeps finite values, and a NaN never reaches another particle's gradient.
- **Units.** SI; positions are metres in a local frame.
- **Axes.** Position columns are (x, y) or (x, y, z); a gridded velocity keeps (time, z, y, x) or (time, y, x).
- **Velocity.** `velocity(t, position)` returns one velocity per particle; `t` is a scalar or one time per particle.
- **Steps.** A step satisfies `ParticleStep`, and the one loop runs it.
- **Time.** Offsets from a traced `t0` over a static signed `duration`; a backward run has a negative one, and no float32 time is far from its epoch.
- **Randomness.** One key per run, split once, or the noise passed in; no key is used twice.
- **Precision.** The output dtype follows the positions' dtype; float32 in gives float32 out.
- **Exceptions** derive from `TraceraxError` and the closest standard-library exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` (the top-level `__all__` holds `__version__` and the exceptions only); a new module in `SUBMODULES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (rounded values, not 17-digit floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the public name (`xref:api#tracerax.<module>.<Name>`, or `xref:api#tracerax.<Name>` for an exception); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each function has its four tests ("What enforces them" in `AGENTS.md`): an oracle, transforms, gradients and masks. The edges (no active particle, a release inside a step, a release after the end, a particle on a wall, a backward run) have their own.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

### A particle is masked, never dropped, and stays finite

```python
import jax
import jax.numpy as jnp

position = jnp.array([1.0, 2.0])
velocity = jnp.array([0.5, jnp.nan])
active = jnp.array([True, False])

# ❌ A boolean index has a shape that depends on values: it fails under jit
#     position[active] + velocity[active]


# ❌ A single where: 0 × NaN is NaN in the backward pass, so one inactive
#    particle's velocity poisons the gradient
def moved_bad(scale):
    return jnp.sum(jnp.where(active, position + scale * velocity, position))


# ✅ The unused value is replaced before the arithmetic
def moved(scale):
    safe_velocity = jnp.where(active, velocity, 0.0)
    return jnp.sum(position + scale * safe_velocity)


assert jnp.isnan(jax.grad(moved_bad)(1.0))
assert jnp.isfinite(jax.grad(moved)(1.0))
```

### A key is used once

```python
import jax
import jax.numpy as jnp

key = jax.random.key(0)

# ❌ The same key twice gives the same draw: u and w are perfectly correlated
u_bad = jax.random.normal(key, (3,))
w_bad = jax.random.normal(key, (3,))
assert jnp.array_equal(u_bad, w_bad)

# ✅ Split once, up front, one key per use
key_u, key_w = jax.random.split(key)
u = jax.random.normal(key_u, (3,))
w = jax.random.normal(key_w, (3,))
assert not jnp.array_equal(u, w)
```

### A backward run has a negative duration, not a reversed wind

```python
import jax.numpy as jnp


def velocity(t, position):
    return jnp.ones_like(position) * t


# ✅ dx = u(t0 + s) ds with ds < 0: the sign is the step's, and the wind is
#    sampled on the clock it was given on
t0, duration, n_steps = 10.0, -10.0, 100
step_size = duration / n_steps
position = jnp.zeros(1)
for k in range(n_steps):
    midpoint = t0 + (k + 0.5) * step_size
    position = position + step_size * velocity(midpoint, position)

# ∫₁₀⁰ t dt = −50
assert jnp.allclose(position, -50.0, atol=1e-3)
```

### Times are offsets from an epoch near the run

```python
import numpy as np

# ❌ float32 resolves 128 s at Unix-epoch seconds, so a 10 s step is lost
unix_time = np.float32(1.7e9)
assert unix_time + np.float32(10.0) == unix_time

# ✅ The same step, as an offset from the start of the run
assert np.float32(0.0) + np.float32(10.0) == np.float32(10.0)
```

### Constants take the input's dtype

```python
import jax.numpy as jnp

position = jnp.zeros(3, dtype=jnp.float32)

# ✅ A Python scalar is weakly typed, so float32 stays float32 with x64 on
assert (position + 0.5).dtype == jnp.float32


# ✅ An array constant is created in the input's dtype
def shifted(position):
    return position + jnp.asarray(0.5, dtype=position.dtype)


assert shifted(position).dtype == jnp.float32
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
