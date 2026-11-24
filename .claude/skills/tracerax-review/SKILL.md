---
name: tracerax-review
description: Review a change or pull request in this repository against its own rules, CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package, functax and integrix already provide, and Lagrangian defects (backward sign, clock, releases, keys, masks, walls, axis order, dtype, gradients). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note which modules, tests and docs it touches.
2. **Reuse**: run the `reuse-reviewer` subagent on the diff: a second loop, a private interpolation or a private tableau is the main way this package drifts.
3. **Correctness**: run the `correctness-reviewer` subagent in parallel with step 2: backward sign, clock, releases, keys and noise, masks, walls, stochastic terms, axis order, dtype, gradients, doctest accuracy, test tolerances.
4. **Contracts**: for each module touched, the rules under "The contracts" in `AGENTS.md`: JAX only and nothing calls `jax.jit`; particles are masked and finite; a step satisfies `ParticleStep`; times are offsets from a traced `t0`; one key per run; constants enter in the positions' dtype; private code sits in `_src/` and the public module only re-exports; the layer rule holds (functax and integrix, nothing else from the stack); exceptions derive from `TraceraxError`; the function has its four tests.
5. **Checklist**: the rest of `CODE_REVIEW.md` (public API, API page and `docs/api/index.md` row, capability index, MyST prose, tests and tiers), skipping what ruff, ty or the tests already enforce.
6. **House style**: the `python-style` review checklist if the skill is installed, otherwise "House style" in `AGENTS.md`: docstring layout with a Unicode equation, comments above the line, descriptive names, maths in the `eman-style-math` notation, `cyclopts` and `loguru` in scripts.
7. **Verify claims**: for anything you flag as a bug, run it (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with priority, file, lines and a concrete change, summary).
