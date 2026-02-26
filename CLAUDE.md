# Combined Agent Instructions
This document combines the following instruction modules: base, python
================================================================================

# Base Agent Instructions

Universal coding guidelines that apply to all projects, regardless of language or framework.

## Core Principles

### Design Philosophy
- **Single Responsibility**: Each function/class/module should do one thing well
- **Composition over Inheritance**: Prefer composing small, focused pieces over deep inheritance hierarchies
- **Fail Fast**: Validate inputs early and throw errors immediately rather than propagating invalid state
- **Make Illegal States Unrepresentable**: Use types and structure to prevent bugs at compile/parse time
- **Principle of Least Surprise**: Code should behave as a reader would expect

### When to Abstract
- **Rule of Three**: Don't abstract until you have 3+ similar cases
- **Avoid Premature Abstraction**: Duplication is cheaper than the wrong abstraction
- **Keep It Simple**: The simplest solution that works is usually best
- **YAGNI** (You Aren't Gonna Need It): Don't build features for hypothetical future needs

## Code Quality

### Naming
- Use descriptive names that reveal intent: `getUserById()` not `get()`
- Avoid abbreviations unless universally known: `HttpRequest` OK, `usrMgr` not OK
- Boolean variables/functions should sound like yes/no questions: `isActive`, `hasPermission`, `canDelete`
- Use verbs for functions: `calculateTotal()`, `sendEmail()`, `validateInput()`
- Use nouns for variables/classes: `userAccount`, `shoppingCart`, `EmailService`

### Functions
- Keep functions short (ideally < 20 lines, definitely < 50)
- Functions should have no more than 3-4 parameters (use objects/config if more needed)
- Avoid boolean parameters - they usually indicate the function does two things
- Return early to avoid deep nesting:
  ```
  // Good
  if (!isValid) return null;
  if (!hasPermission) return null;
  return processData();

  // Avoid
  if (isValid) {
    if (hasPermission) {
      return processData();
    }
  }
  return null;
  ```

### Error Handling
- Never silently swallow errors
- Fail loudly with clear error messages
- Include context in error messages: what failed, why, and what the input was
- Use exceptions/errors for exceptional cases, not control flow
- Validate inputs at boundaries (API endpoints, function entry points)
- Don't validate internal functions that are only called by trusted code

### Code Organization
- Group related code together
- Organize by feature/domain, not by type (e.g., `users/` not `controllers/`, `models/`, `views/`)
- Keep files focused and reasonably sized (< 500 lines as a guideline)
- Put public/exported items at the top of files, private/internal at the bottom
- Order functions so they read top-to-bottom (high-level calls low-level)

## Version Control

### Git Commits
- **Commit and push** after every logical chunk of work — don't wait until the end
- **Atomic Commits**: Each commit should be a complete, working change
- **Commit Messages**:
  - First line: imperative mood, < 72 chars: "Add user authentication" not "Added" or "Adds"
  - Add context in body if needed: why this change, what problem it solves
  - Reference issues/tickets when applicable

### What to Commit
- **Never commit**: secrets, credentials, API keys, generated files, dependencies, IDE config
- **Always commit**: source code, tests, documentation, schema/migration files, lock files
- **Use .gitignore**: Set it up early and maintain it

### Branching
- Keep main/master stable and deployable
- Use feature branches for development
- Delete branches after merging
- Keep branches short-lived (days, not weeks)

## Testing

### Testing Philosophy
- Write tests for business logic and complex functions
- Don't test framework code or trivial getters/setters
- Test behavior, not implementation details
- Each test should test one thing
- Tests should be independent and runnable in any order

### Test Organization
- Use descriptive test names that explain what's being tested
- Follow Arrange-Act-Assert pattern
- Keep test data/fixtures minimal - only what's needed for that test
- Mock external dependencies (APIs, databases, file systems)

### What to Test
- **Critical paths**: User authentication, payment processing, data persistence
- **Edge cases**: Empty inputs, null values, boundary conditions
- **Error cases**: Invalid inputs, network failures, permission denied
- **Business logic**: Calculations, transformations, validation rules

## Documentation

### When to Comment
- **Do comment**:
  - Why something is done a certain way (especially if non-obvious)
  - Known limitations or gotchas
  - Complex algorithms or business rules
  - Public APIs and interfaces

- **Don't comment**:
  - What the code does (code should be self-explanatory)
  - Obvious things: `i++; // increment i`
  - Commented-out code (use git history instead)

### README Files
Every project should have a README with:
- What the project does
- How to install/setup
- How to run it
- How to run tests
- Key dependencies or requirements

### Code Documentation
- Document public APIs, exported functions, and interfaces
- Include parameter types, return values, and examples
- Keep docs in sync with code (outdated docs are worse than no docs)

## Security

### Input Validation
- **Validate all external input**: User input, API requests, file uploads, environment variables
- **Sanitize for context**: HTML encode for web, parameterize for SQL, validate for commands
- **Whitelist over blacklist**: Define what's allowed, not what's forbidden

### Secrets Management
- Never hardcode secrets in source code
- Use environment variables or secret management systems
- Rotate credentials regularly
- Use different credentials for dev/staging/production

### Common Vulnerabilities
- **SQL Injection**: Use parameterized queries, never string concatenation
- **XSS**: Escape/sanitize user input before rendering
- **CSRF**: Use tokens for state-changing operations
- **Path Traversal**: Validate file paths, don't trust user input
- **Command Injection**: Avoid executing shell commands with user input

## Performance

### Premature Optimization
- "Premature optimization is the root of all evil" - Don't optimize until you measure
- Write correct, readable code first
- Profile before optimizing
- Optimize the bottlenecks, not everything

### General Performance Guidelines
- Use appropriate data structures (hash maps for lookups, not arrays)
- Avoid N+1 queries (database or API calls in loops)
- Cache expensive computations when appropriate
- Be mindful of memory allocation in hot paths
- Use pagination for large datasets

## Dependencies

### Adding Dependencies
- Evaluate before adding: Is it maintained? Well-tested? Right size for the problem?
- Prefer standard library when possible
- Keep dependencies up to date (security patches)
- Review dependency licenses

### Dependency Management
- Lock dependency versions for reproducible builds
- Document why non-obvious dependencies are needed
- Regularly audit for security vulnerabilities
- Remove unused dependencies

## Code Review Mindset

When writing code that will be reviewed (or reviewing your own code):
- Is this code clear and easy to understand?
- Have I handled error cases?
- Are there security implications?
- Is this tested adequately?
- Does this follow project conventions?
- Would I want to debug this code at 2am?

================================================================================

# Python-Specific Agent Instructions

These instructions apply when working with Python code.

## Package Management with uv

Use **uv** exclusively. Never pip, pip-tools, poetry, or conda.

```bash
uv add <package>              # Install dependency
uv remove <package>           # Remove dependency
uv sync                       # Sync from lockfile (includes dev group by default)
uv sync --locked              # Sync, fail if lockfile is stale (use in CI)
uv sync --no-dev              # Sync without dev dependencies (production)
uv run script.py              # Run script
uv run pytest                 # Run tools (pytest, ruff, mypy)
uv run python                 # REPL
uvx <tool>                    # Run CLI tools without installing (e.g., uvx ruff check .)
```

### Dependency Groups

Dev deps added via `uv add --dev` go into `[dependency-groups]` in pyproject.toml and sync by default. Use named groups for organization:

```bash
uv add --dev pytest pytest-cov        # Goes into the "dev" group (syncs by default)
uv add --group lint ruff               # Named group for linters
uv add --group test pytest pytest-cov  # Named group for test deps
uv sync --all-groups                   # Sync all groups
```

**Important:** `--extra` is for optional user-facing features (`[project.optional-dependencies]`), not dev deps. Never `uv sync --extra dev` — just `uv sync`.

### Lockfile Management

```bash
uv lock                                 # Resolve and write lockfile
uv lock --upgrade-package requests      # Bump a single dep
uv lock --check                         # Check if lockfile is fresh (CI)
uv tree                                 # Visualize dependency tree
uv export --format requirements-txt     # For Docker/legacy systems
```

### PEP 723 Inline Script Metadata

For standalone scripts with dependencies:

```python
# /// script
# dependencies = [
#   "requests",
#   "pandas>=2.0",
# ]
# ///
```

```bash
uv run script.py                        # Run (auto-installs deps)
uv add requests --script script.py      # Add dep to script
uv lock --script script.py              # Create script.py.lock for reproducibility
```

### Python Version Management

uv can replace pyenv entirely:

```bash
uv python install 3.12                  # Install a version
uv python pin 3.12                      # Pin for current project (.python-version)
uv python list                          # Show available/installed
uv run --python 3.13 script.py          # Auto-downloads if missing
```

## Style

- PEP 8, 4-space indent, 88-char line length (ruff/Black standard)
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants

```bash
uv run ruff format .          # Format
uv run ruff check .           # Lint
uv run ruff check --fix .     # Auto-fix
uv run mypy .                 # Type check
```

## Key Preferences

- **Type hints** on all function signatures
- **f-strings** for formatting (never `.format()` or `%`)
- **`pathlib.Path`** for file operations (never `os.path`)
- **`set`** for membership testing (O(1) vs O(n) for lists)
- **Generators** for large datasets to save memory
- **`logging`** module, not `print` statements
- **Specific exceptions** — never bare `except:` or `except Exception`
- **Context managers** (`with`) for resource management
- Include context in error messages: what failed, why, what the input was

## Testing with pytest

```bash
uv add --dev pytest pytest-cov
uv run pytest                                     # Run all
uv run pytest --cov=src tests/                    # With coverage
uv run pytest tests/test_module.py::test_function # Specific test
```

- Descriptive test names, Arrange-Act-Assert pattern
- Use `tmp_path` fixture for file tests, `@pytest.mark.parametrize` for multiple inputs
- `pytest.raises(ValueError, match="...")` for exception testing
- Mock external dependencies; keep test data minimal

## Virtual Environments

- uv handles venvs automatically
- Keep `pyproject.toml` and lock file in version control
- Profile before optimizing: `uv run python -m cProfile script.py`

================================================================================

