---
description: Guidelines for developing, testing, documenting, and contributing to City2Graph.
keywords: contributing, pull request, development environment, pre-commit, pytest, coverage, code style, ruff, mypy, numpydoc, documentation
hide:
  - navigation
---

# Contributing

We welcome contributions to City2Graph. This guide is the canonical reference
for setting up the repository, making and testing changes, and submitting a
pull request.

## Set up the development environment

1. Fork the repository on GitHub.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-name>/city2graph.git
   cd city2graph
   git remote add upstream https://github.com/c2g-dev/city2graph.git
   ```

3. Install [uv](https://docs.astral.sh/uv/) if it is not already available on
   your `PATH`.
4. Install the development dependencies with the CPU-only PyTorch extra:

   ```bash
   uv sync --extra cpu --group dev
   ```

The CPU build keeps development installations fast and deterministic. The
project supports Python 3.11 through 3.14, and CI tests every supported version.

There is no separate virtual-environment activation step. Run project commands
with `uv run`; uv automatically uses the managed environment.

## Make a change

Create a branch from an up-to-date `main` branch. Direct commits to `main` are
blocked by the `no-commit-to-branch` pre-commit hook.

```bash
git checkout -b your-change-name
```

Keep each pull request focused. Update tests and documentation with the code
they describe, and use a concise, descriptive commit message.

Before committing, run both required checks:

```bash
uv run pytest -q
uv run pre-commit run --all-files
```

These are the same commands used by the repository's local workflow and CI.

## Code and API quality

All code, comments, docstrings, and documentation must be written in English.

City2Graph uses:

- **Ruff** for linting, import ordering, and formatting. The formatter line
  length is 100 characters; the lint configuration allows lines up to 140
  characters where Ruff cannot rewrite them cleanly.
- **mypy** with strict type checking. Public functions must have type hints.
- **numpydoc** validation. Public modules, functions, classes, and methods must
  use NumPy-style docstrings.

Use four spaces for indentation. Prefer the existing public API and established
module patterns when extending functionality. If a change affects public
behaviour, update the corresponding docstring and documentation.

## Testing

Tests live under `tests/` and mirror the package layout. Add a new test class
only when the test does not belong to an existing class.

Test private helper functions indirectly through the public function that calls
them. Do not create tests that call private helpers directly. This keeps tests
focused on supported behaviour and allows internal implementations to evolve.

New behaviour and bug fixes must include tests that cover the changed paths.
Behaviour-preserving refactors may rely on existing public-API tests when those
tests already exercise the refactored paths.

Coverage is enabled automatically by the pytest configuration:

```bash
uv run pytest -q
```

The command prints a `term-missing` report. Check its missing-line output before
opening a pull request and keep new code covered; no additional coverage flags
are needed. `@overload` and `if TYPE_CHECKING:` blocks are excluded from
coverage, so do not add tests solely to execute those blocks.

## Pre-commit checks

Install the hooks once per clone:

```bash
uv run pre-commit install
```

They then run automatically on `git commit`. Always run the complete suite
before committing:

```bash
uv run pre-commit run --all-files
```

For a focused local iteration, run one hook or a selected set of files:

```bash
uv run pre-commit run ruff-check --all-files
uv run pre-commit run --files city2graph/graph.py tests/test_graph.py
```

The configured checks cover:

- Python AST, TOML, and YAML syntax;
- case conflicts, merge-conflict markers, and private keys;
- executable and shebang consistency;
- byte-order marks, mixed line endings, trailing whitespace, and final
  newlines;
- strict YAML linting;
- NumPy-style docstrings in `city2graph/`;
- Ruff linting with automatic fixes and Ruff formatting;
- strict mypy type checking; and
- protection against direct commits to protected branches.

Some hooks modify files automatically. Review those changes, stage them, and
rerun the full suite until every hook passes. Do not bypass failed checks.

## Documentation

Update the relevant files under `docs/` for new features or significant
behaviour changes. Add or update examples when they materially help users
understand a feature.

Install the documentation dependencies and build the site before submitting
documentation changes:

```bash
uv sync --group docs --extra cpu
uv run mkdocs build
```

For a local preview, run:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

## Pull requests

Push your branch to your fork and open a pull request against
`c2g-dev/city2graph:main`:

```bash
git push -u origin your-change-name
```

Follow the repository's pull request template:

- **Summary**: explain what changed and why.
- **Related issues**: link issues with closing keywords when appropriate, or
  state that none apply.
- **Testing**: list the exact automated and manual checks run; explain any
  check that was not run.
- **Documentation**: identify documentation changes, or write "Not applicable".
- **Reviewer notes**: call out compatibility concerns, trade-offs, or areas
  deserving particular attention.

Both `uv run pytest -q` and `uv run pre-commit run --all-files` must pass before
the contribution is committed. Documentation changes must also pass
`uv run mkdocs build`.
