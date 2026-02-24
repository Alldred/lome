<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Testing and Development

[Docs Home](README.md)

## Run Tests

```bash
uv run pytest
```

## Run Tests with Coverage

```bash
uv run pytest --cov=lome --cov-report=term-missing
```

## Lint and Formatting

This repository uses pre-commit hooks (including Ruff and formatting hooks).
Typical local workflow:

```bash
uv run pre-commit run --all-files
```

## Contributor Workflow

1. Add or update tests alongside code changes.
2. Run full tests before committing.
3. Keep docs aligned with externally visible behavior.
4. Prefer small, focused commits by change theme.

## Debugging Tips

- Use raw `peek_*` / `poke_*` APIs for deterministic test setup.
- Inspect `ChangeRecord.to_detailed_dict()` for precise effect tracking.
- For decode/dispatch issues, isolate via `Lome.execute()` and known instruction words.
