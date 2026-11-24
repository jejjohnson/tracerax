# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the single source of truth for every coding agent working here (the module map, "reuse before you write", the contracts, the tests that enforce them, commands, the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/tracerax/`: the GeoStack's library of Lagrangian particles, in JAX. It imports functax and integrix, and nothing else from the GeoStack at runtime (never geotoolz, pipekit, gaussx, kernellib or a model). Public modules, each created by its first feature issue: `particles`, `velocity`, `advect`, `bin`, `footprint`, `analysis`; private code in `_src/<module>/`. Search [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a helper.
- Ownership test: is the state a set of positions carried forward in time, or does it analyse such positions? A scheme that discards its departure points is integrix's; closures, plume physics and source inversions are plumax's; interpolation and binning are functax's.
- Keep the contracts in `AGENTS.md`:
  - **JAX only**: pure functions and Equinox modules; nothing calls `jax.jit`;
  - **shapes are static**: particles are masked, never dropped, and a masked particle keeps finite values;
  - **units** are SI; position columns are (x, y) or (x, y, z); a gridded velocity keeps (time, z, y, x);
  - **velocity** is `velocity(t, position)`; a **step** is a `ParticleStep`, run by the one loop;
  - **time** is an offset from a traced `t0` over a static signed `duration`, negative for a backward run;
  - **randomness** is one key per run, or the noise itself; nothing reads a global seed;
  - **precision** follows the positions' dtype; every exception derives from `TraceraxError` and a standard-library exception.

The rules for every project:

- Every docstring `Examples:` block runs (`--doctest-modules`): the expected output must be what the code prints.
- Before committing, from the repo root: `uv run poe test`, `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`, `uv run poe typecheck`; `uv run poe capabilities` after a public API change. Lint the whole repo, never a subdirectory.
- When you suggest a change, propose the code; when you fix a bug, add the test that reproduces it. Don't nitpick what ruff and the formatter handle.
- House style ("House style" in `AGENTS.md`): the maintainer's `python-style` for Python (comments above the line, a Unicode equation above each mathematical line, descriptive names), `eman-style-math` for maths (Unicode in code and docstrings, macros in MyST and MkDocs pages), `cyclopts` and `loguru` for new CLIs and scripts, Markdown never hard-wrapped, Mermaid rather than ASCII diagrams.
- Partial test runs need `--no-cov` (the 90 % coverage gate fails a subset).
- Never push or merge to `main` unless told to; never replace an existing PR description, only append to it.
- No AI attribution: no `Co-Authored-By:` trailer for an AI and no "Generated with …" line, unless asked. PR descriptions open with why, then a small before/after table; one `Closes #N` line per issue.
- Step-by-step recipes (add a particle function, guide page or notebook; pre-PR check; review) are plain Markdown in `.claude/skills/<name>/SKILL.md`, and the two review checklists in `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows [`CODE_REVIEW.md`](../CODE_REVIEW.md).
