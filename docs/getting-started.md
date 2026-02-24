<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Getting Started

[Docs Home](README.md) | [API Reference](API.md)

## Prerequisites

- Python `3.13+`
- `uv` for dependency and environment management
- Access to `eumos` (installed from GitHub via project dependency)

## Install

```bash
git clone <repository-url>
cd lome
uv sync --extra dev
```

## Minimal Example

```python
from eumos import Eumos
from lome import Lome

isa = Eumos()
model = Lome(isa)

# addi x1, x0, 42
addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
changes = model.execute(addi)

print(model.get_gpr(1))
print(model.get_pc())
print(changes.to_simple_dict())
```

## Core Concepts

- Architectural access: `get_*` / `set_*`
- Raw access: `peek_*` / `poke_*`
- Speculation: `speculate()` executes without mutating state
- Change tracking: every execution returns a `ChangeRecord`

See [Model Semantics](model-semantics.md) for behavior details.

## Next Steps

- Explore [API Reference](API.md)
- Review [Instruction Support](instruction-support.md)
- Run the full test suite via [Testing and Development](testing-and-development.md)
