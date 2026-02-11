<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# RISC-V Model API Reference

The public API is the `RISCVModel` class and the change-tracking types it uses.

## Package

From `riscv_model`:

- **`RISCVModel`** — main entry point.

```python
from riscv_model import RISCVModel
```

---

## RISCVModel

Main interface for instruction execution, speculation, and state access.

| Method | Signature | Description |
|--------|-----------|-------------|
| **execute** | `execute(instruction_bytes: Union[int, bytes], speculate: bool = False) -> Optional[ChangeRecord]` | Execute one instruction. Use 32-bit instruction as `int` or little-endian `bytes`. If `speculate=True`, runs without modifying state. Returns `ChangeRecord` or `None` if instruction is unknown. |
| **speculate** | `speculate(instruction_bytes: Union[int, bytes]) -> Optional[ChangeRecord]` | Same as `execute(..., speculate=True)`: returns what would change without applying it. |
| **get_gpr** | `get_gpr(reg: int) -> int` | Get GPR value. x0 always returns 0. |
| **get_csr** | `get_csr(csr: Union[int, str]) -> Optional[int]` | Get CSR value by 12-bit address (int) or name (str). |
| **get_pc** | `get_pc() -> int` | Get program counter. |
| **set_pc** | `set_pc(value: int) -> None` | Set program counter. |
| **reset** | `reset() -> None` | Reset all state (GPRs, CSRs, PC) to initial values; clears last changes. |
| **get_changes** | `get_changes() -> Optional[ChangeRecord]` | Get the change record from the last execution (or `None`). |
| **get_branch_info** | `get_branch_info()` | Get branch information from last execution, or `None`. |

---

## ChangeRecord

Returned by `execute()`, `speculate()`, and `get_changes()`. Defined in `riscv_model.changes`.

**Fields:**

- `gpr_writes` — list of `GPRWrite`
- `csr_writes` — list of `CSRWrite`
- `pc_change` — `(new_pc, old_pc)` or `None`
- `memory_accesses` — list of `MemoryAccess`
- `branch_info` — `BranchInfo` or `None`
- `exception` — exception type string or `None`

**Methods:**

- `has_changes() -> bool`
- `get_gpr_changes() -> Dict[int, Tuple[int, int]]` — register → (new_value, old_value)
- `get_csr_changes() -> Dict[int, Tuple[int, int]]` — address → (new_value, old_value)
- `get_pc_change() -> Optional[Tuple[int, int]]`
- `get_branch_info() -> Optional[BranchInfo]`
- `get_memory_accesses() -> List[MemoryAccess]`
- `get_all_changes() -> Dict` — full change structure
- **`to_simple_dict() -> Dict`** — compact summary (new values only, no old values).
- **`to_detailed_dict() -> Dict`** — full change record as a dict (same as `get_all_changes()`).

Use the return value of `execute()` or `get_changes()` then call `.to_simple_dict()` or `.to_detailed_dict()`.

---

## Dataclasses (changes module)

- **GPRWrite** — `register`, `value`, `old_value`
- **CSRWrite** — `address`, `name`, `value`, `old_value`
- **MemoryAccess** — `address`, `value` (None for reads), `size`, `is_write`
- **BranchInfo** — `taken`, `target`, `condition` (optional)

These structure the contents of a `ChangeRecord`; you typically get them via `get_changes()` or the `ChangeRecord` methods above.
