# AGENTS.md

## Purpose
This repository follows a disciplined, maintainable, and CPU-first workflow. Agents must keep changes small, readable, and well tested, while aligning with Dribia's coding standards.

## Environment & Tooling
- **CPU-only**: all training and scripts must default to CPU execution. Do not assume GPU availability.
- **Package management**: use **uv** for environments and dependencies (e.g., `uv venv`, `uv run`).
- **Testing**: use **pytest** for unit tests (`make test`, `make test-unit`).
- **Linting/formatting**: use **ruff** (lint + format) and **tombi** for TOML formatting (`make lint`, `make format`).
- **Type checking**: use **ty** (`make lint`).
- **Docs**: MkDocs with Material theme (`mkdocs.yml`, `docs/`).

## Coding Standards (Dribia-inspired)
- Write **Pythonic**, maintainable code emphasizing **readability**, **composability**, and **clarity**.
- Follow **PEP 8** naming conventions and use explicit, descriptive names.
- Add **module header docstrings** and **Google-style** docstrings for functions and classes.
- Prefer **vectorized pandas operations**; avoid `apply` when possible.
- Avoid `inplace=True` mutations; return new objects instead.
- Keep code in **English** for identifiers, comments, and documentation.
- Keep tests organized: one test file per module, named `tests/test_<module>.py`.

## Required Agent Workflow (TDD & Quality Gates)
1. **TDD red/green only**:
   - Write unit tests **first**, mocking external dependencies as needed.
   - Run tests to confirm they **fail** (red).
   - Implement the code to make tests **pass** (green).
2. **After each change**:
   - Run the standard quality gates: `make format`, `make lint`, `make test`.
   - Run `uv run --frozen python scripts/check_forecast_mae.py` **only when substantive model changes** are performed, and display the MAE comparison table.
3. **When everything passes**:
   - Update **README** and relevant documentation using the **`write-docs` skill**.
   - Ensure the README notes that changes are **agent-made** when relevant.
4. **Commit**:
   - Use the **`commit` skill** and follow Conventional Commits.

## Integrity & Evaluation Rules
- **No cheating or shortcuts** when training or evaluating models.
- Do not peek at future data, leak labels, or use the target horizon to initialize model states.
- Do not seed models with ground-truth values beyond the input window.
- All baselines and models must use the same train/validation/test splits.

## CLI Standards
When creating or modifying CLIs, use the **`create-cli` skill** and ensure the interface follows consistent CLI UX standards (clear help text, sensible defaults, subcommands where appropriate).
- **CLI rendering**: all CLI output must use **Rich**.
- **Scripts/entrypoints**: any script must be implemented with **Typer**.

## Documentation Expectations
- Keep README and docs accurate and concise.
- Ensure any agent-driven change is reflected in documentation where it matters.
- After each run, refresh affected docs with MkDocs-friendly formatting (headings, lists, code blocks) and follow the repo's MkDocs conventions.

## Execution Logs
- Always generate logs for script and command executions.
- Maintain a rolling log with a maximum size of **5 MB** (rotate or truncate as needed).
- Follow Dribia logging practices: use the standard `logging` module (no bare `print` for operational logs),
  apply consistent log levels, and prefer structured, readable messages that satisfy Ruff logging rules.
- If a script must emit output for human consumption, keep it concise and mirror it to the log file.
