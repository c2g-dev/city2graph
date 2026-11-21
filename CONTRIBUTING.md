# Contributing to City2Graph

Thank you for your interest in contributing to City2Graph.

## Human-in-the-loop requirement

All contributions **must** involve active human review and verification.
Autonomous or unreviewed AI-agent pull requests are prohibited and may be closed
without review.

Before opening a pull request, you must:

1. Personally read, understand, and verify every changed line of code, tests,
   and documentation.
2. Run both `uv run pytest -q` and `uv run pre-commit run --all-files` locally
   in the full development environment (`uv sync --extra cpu --group dev`) and
   confirm that both pass.
3. Submit one focused pull request at a time rather than an automated or
   uncoordinated batch of pull requests.
4. Complete the human verification checklist and disclose any AI assistance in
   the pull request template.

## Full contributing guide

See [docs/contributing.md](docs/contributing.md) (also published at
[city2graph.net](https://city2graph.net/latest/contributing.html)) for the complete
development setup, code quality rules, testing conventions, documentation build
steps, and pull request requirements.
