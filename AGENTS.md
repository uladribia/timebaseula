# AGENTS.md

## Purpose
This repository follows a disciplined, maintainable, and CPU-first workflow. Agents must keep changes small, readable, and well tested, while aligning with Dribia's coding standards.

## Environment & Tooling
- **CPU-only**: all training and scripts must default to CPU execution. Do not assume GPU availability.
- **Package management**: use **uv** / **uvx** for environments, dependencies, and tool execution (e.g., `uv venv`, `uv run`, `uvx ...`). Do **not** use `uv pip`.
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
- When testing main library code, prefer **property-based testing with Hypothesis** for invariant-driven behavior such as tensor shapes, default-resolution rules, configuration contracts, and deterministic transformations. Keep example-based tests for fixed API facts, imports/exports, object identity, and heavy integration behavior where property-based testing would add noise more than value.
- Before implementing custom logic, inspect the upstream library source and prefer built-in APIs (for example `cross_validation` or native baseline models in NeuralForecast / StatsForecast) over handwritten benchmark orchestration whenever they provide the needed behavior.

## Required Agent Workflow (TDD & Quality Gates)
1. **TDD red/green only**:
   - Write unit tests **first**, mocking external dependencies as needed.
   - Run tests to confirm they **fail** (red).
   - Implement the code to make tests **pass** (green).
2. **Keep unit and integration tests separate**:
   - Treat tests marked `integration` as **heavy**, optional validation.
   - The default quality gate is **unit-only**: use `make test` or `make test-unit`.
   - Run integration tests with `make test-integration` **only when they are justified** by the change.
   - In every substantial update, explicitly decide whether integration tests are worth running and state that decision in your summary.
3. **After each change**:
   - Run the standard fast quality gates: `make format`, `make lint`, `make test`.
   - Run `uv run --frozen python scripts/check_forecast_mae.py` **only when substantive model changes** are performed, and display the MAE comparison table.
   - Run `make test-integration` only for changes that affect end-to-end training, NeuralForecast integration, CLI workflows, or other cross-module behavior that unit tests cannot cover well.
4. **When everything passes**:
   - Update **README** and relevant documentation using the **`write-docs` skill**.
   - Ensure the README notes that changes are **agent-made** when relevant.
5. **Use feature branches for new work**:
   - Whenever a **new feature** is requested, create a dedicated branch before making changes.
   - Use a clear, descriptive branch name tied to the feature.
   - Keep feature work isolated on that branch until the feature is considered complete.
6. **Commit after each relevant completed change**:
   - Use the **`commit` skill** and follow Conventional Commits.
   - Do not leave substantial finished code or documentation changes uncommitted.
   - After a relevant change clears its intended quality gates, create a commit before moving on to the next substantial task.
7. **Close feature branches deliberately**:
   - When a feature is considered complete, explicitly prompt the user for validation before closing the work.
   - After validation, squash the branch commits into a clean final commit.
   - Provide a concise summary of what is being merged.
   - Merge the squashed result and delete the feature branch once closure is complete.

## Integrity & Evaluation Rules
- **No cheating or shortcuts** when training or evaluating models.
- Do not peek at future data, leak labels, or use the target horizon to initialize model states.
- Do not seed models with ground-truth values beyond the input window.
- All baselines and models must use the same train/validation/test splits.

## CLI Standards
When creating or modifying CLIs, use the **`create-cli` skill** and ensure the interface follows consistent CLI UX standards (clear help text, sensible defaults, subcommands where appropriate).
- **CLI rendering**: all CLI output must use **Rich**.
- **Scripts/entrypoints**: any script must be implemented with **Typer**.

## Visualization Standards
- Prefer **Matplotlib** for charts, diagnostics, and generated reports across the repository.
- For HTML outputs, embed Matplotlib figures as static images rather than relying on browser-side chart runtimes.
- Keep figures readable, reproducible, scriptable, and suitable for docs, reports, and offline review.
- Avoid introducing **Altair** for new visualization work in this repository.
- When updating existing plots, prefer migrating them toward Matplotlib if the change is substantial or the plot is user-facing.

## Documentation Expectations
- Keep README and docs accurate and concise.
- Ensure any agent-driven change is reflected in documentation where it matters.
- After each run, refresh affected docs with MkDocs-friendly formatting (headings, lists, code blocks) and follow the repo's MkDocs conventions.

## Repository Branch Strategy
- The repository maintains two long-lived branches with different purposes:
  - `benchmark`: full benchmarking, tuning, experiment scripts, benchmark-oriented tests, and workflow docs.
  - `main`: release-oriented library branch with publishable package code, curated docs, and published benchmark result pages, but without benchmark-generation scripts or related scaffolding.
- Agents must treat `benchmark` as the source branch for benchmark workflow development.
- When preparing `main`, agents should curate from `benchmark` rather than reimplementing content independently.
- On `main`, keep benchmark result reports and images when they support library documentation, but remove benchmark orchestration scripts, tuning scripts, and benchmark-only test scaffolding.
- On `main`, README and docs must clearly state that full benchmarking and tuning workflows live on the `benchmark` branch.
- When curating `main`, preserve the library package, core library tests, and user-facing docs first; remove only workflow machinery that is not needed for the release-oriented branch.

### Release Curation Workflow
- Treat `benchmark` as the canonical source branch for any release preparation. `main` must not be ahead of `benchmark` in shared library functionality, shared workflows, or shared metadata.
- Do not implement release changes independently on `main` when the same change belongs on `benchmark` first.
- Classify branch differences into three buckets before curating `main`:
  - **shared files**: files that should stay aligned across both branches, such as package code, shared tests, shared workflow files, and common metadata;
  - **benchmark-only files**: scripts, tuning artifacts, benchmark-only tests, and workflow docs that should remain only on `benchmark`;
  - **main-curated files**: README and user-facing docs that may be rewritten on `main` to explain that full workflows live on `benchmark`.
- When generating a release from `benchmark` to `main`, follow this order:
  1. update and validate the intended changes on `benchmark` first;
  2. diff `benchmark` against `main` and confirm that every shared-file difference is intentional;
  3. bring shared files on `main` back in sync with `benchmark` before curating branch-specific removals;
  4. remove benchmark-only files from the `main` candidate rather than reimplementing them differently;
  5. keep curated benchmark result pages and images on `main` only when they are copied from `benchmark` intentionally, not regenerated independently on `main`;
  6. update `main` README and docs so they explicitly point users to `benchmark` for reproducible workflows.
- Before finalizing release work, verify that:
  - `main` is a curated subset of `benchmark` plus intentional main-only documentation wording;
  - `main` is not ahead of `benchmark` in shared code or shared repository configuration;
  - the remaining branch diff is explained entirely by benchmark-only removals or approved main-doc curation.

## Execution Logs
- Always generate logs for script and command executions.
- Maintain a rolling log with a maximum size of **5 MB** (rotate or truncate as needed).
- Follow Dribia logging practices: use the standard `logging` module (no bare `print` for operational logs),
  apply consistent log levels, and prefer structured, readable messages that satisfy Ruff logging rules.
- If a script must emit output for human consumption, keep it concise and mirror it to the log file.
