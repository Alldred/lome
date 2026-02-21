<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# RISC-V Model API Reference

The public API is the `Lome` class and the change-tracking types it uses.

## Package

```python
from eumos import Eumos
from lome import Lome

# Load Eumos once, share with the model, ISG, or any other component
isa   = Eumos()
model = Lome(isa)
```

You can also import individual types:

```python
from lome import (
    Lome,
    State,
    ChangeRecord,
    GPRWrite,
    CSRWrite,
    MemoryAccess,
    BranchInfo,
)
```

---

## Lome

Main interface for instruction execution, speculation, and state access.

### Instruction Execution

| Method | Signature | Description |
|--------|-----------|-------------|
| **execute** | `execute(instruction_bytes: int \| bytes, speculate: bool = False) -> ChangeRecord \| None` | Execute one instruction. 32-bit word as `int` or little-endian `bytes`. Returns `ChangeRecord` or `None`. |
| **speculate** | `speculate(instruction_bytes: int \| bytes) -> ChangeRecord \| None` | Dry-run — same as `execute(..., speculate=True)`. |

### Architectural Register Access (get / set)

These methods enforce architectural constraints (x0 = 0, CSR read-only) and
trigger CSR side-effect hooks (e.g. mstatus → sstatus mirroring).

| Method | Signature | Description |
|--------|-----------|-------------|
| **get_gpr** | `get_gpr(reg: int) -> int` | Read GPR. x0 always returns 0. |
| **set_gpr** | `set_gpr(reg: int, value: int) -> int` | Write GPR. x0 writes ignored. Returns previous value. |
| **get_csr** | `get_csr(csr: int \| str) -> int \| None` | Read CSR by address or name. |
| **set_csr** | `set_csr(csr: int \| str, value: int) -> int \| None` | Write CSR. Respects read-only, triggers hooks. Returns previous value. |
| **get_fpr** | `get_fpr(reg: int) -> int` | Read FPR (64-bit bits). |
| **set_fpr** | `set_fpr(reg: int, value: int) -> int` | Write FPR. Returns previous value. |
| **get_pc** | `get_pc() -> int` | Read program counter. |
| **set_pc** | `set_pc(value: int) -> int` | Write program counter. Returns previous value. |

### Raw Register Access (peek / poke)

These methods bypass all architectural checks and side effects.  Use for test
setup, debugging, and checkpoint restore.

| Method | Signature | Description |
|--------|-----------|-------------|
| **peek_gpr** | `peek_gpr(reg: int) -> int` | Raw read (no x0 enforcement). |
| **poke_gpr** | `poke_gpr(reg: int, value: int) -> int` | Raw write (even x0). Returns previous raw value. |
| **peek_csr** | `peek_csr(csr: int \| str) -> int \| None` | Raw read (no side effects). |
| **poke_csr** | `poke_csr(csr: int \| str, value: int) -> int \| None` | Raw write (bypasses read-only, no hooks). |
| **peek_fpr** | `peek_fpr(reg: int) -> int` | Raw read FPR. |
| **poke_fpr** | `poke_fpr(reg: int, value: int) -> int` | Raw write FPR. Returns previous value. |
| **peek_pc** | `peek_pc() -> int` | Raw read PC. |
| **poke_pc** | `poke_pc(value: int) -> int` | Raw write PC. Returns previous value. |

### State Management

| Method | Signature | Description |
|--------|-----------|-------------|
| **reset** | `reset() -> None` | Reset GPRs, FPRs, CSRs, PC to initial values; clears last changes. |
| **get_changes** | `get_changes() -> ChangeRecord \| None` | Last execution's change record. |
| **get_branch_info** | `get_branch_info() -> BranchInfo \| None` | Branch info from last execution. |

### JSON Serialisation

| Method | Signature | Description |
|--------|-----------|-------------|
| **export_state** | `export_state() -> dict` | Export complete state as JSON-serialisable dict. |
| **restore_state** | `restore_state(data: dict) -> None` | Restore from dict (raw — no hooks). |
| **export_state_json** | `export_state_json(indent=2) -> str` | Export as formatted JSON string. |
| **from_json** *(classmethod)* | `from_json(json_str, eumos) -> Lome` | Create model from JSON string. |

#### Export Format

```json
{
  "pc": 4096,
  "gprs": { "0": 0, "1": 42, ..., "31": 0 },
  "fprs": { "0": 0, "1": 1078530011, ..., "31": 0 },
  "csrs": { "768": 6144, ... },
  "metadata": {
    "gpr_names": { "0": "zero", "1": "ra", ... },
    "fpr_names": { "0": "ft0", "1": "ft1", ... },
    "csr_names": { "768": "mstatus", ... }
  }
}
```

---

## ChangeRecord

Returned by `execute()`, `speculate()`, and `get_changes()`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `gpr_writes` | `list[GPRWrite]` | GPR writes. |
| `fpr_writes` | `list[FPRWrite]` | FPR writes. |
| `csr_writes` | `list[CSRWrite]` | CSR writes. |
| `pc_change` | `(new_pc, old_pc) \| None` | PC change (branches/jumps). |
| `memory_accesses` | `list[MemoryAccess]` | Load/store accesses. |
| `branch_info` | `BranchInfo \| None` | Branch outcome. |
| `exception` | `str \| None` | Exception type. |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `has_changes()` | `bool` | Any field has a change? |
| `get_gpr_changes()` | `dict[int, (int, int)]` | `{register: (new, old)}` |
| `get_csr_changes()` | `dict[int, (int, int)]` | `{address: (new, old)}` |
| `get_pc_change()` | `(int, int) \| None` | `(new_pc, old_pc)` |
| `get_branch_info()` | `BranchInfo \| None` | Branch outcome. |
| `get_memory_accesses()` | `list[MemoryAccess]` | Copy of accesses. |
| `get_all_changes()` | `dict` | Full detailed dict. |
| **`to_simple_dict()`** | `dict` | Compact summary (new values only). |
| **`to_detailed_dict()`** | `dict` | Full record (same as `get_all_changes()`). |

---

## Dataclasses

| Class | Fields | Description |
|-------|--------|-------------|
| **GPRWrite** | `register`, `value`, `old_value` | Single GPR write. |
| **CSRWrite** | `address`, `name`, `value`, `old_value` | Single CSR write. |
| **MemoryAccess** | `address`, `value` (None for reads), `size`, `is_write` | Load or store. |
| **BranchInfo** | `taken`, `target`, `condition` (optional) | Branch outcome. |

---

## CSR Side Effects

`set_csr()` (and `set_csr_by_name()`) trigger registered write hooks after
storing the new value.  Built-in hooks:

* **mstatus / sstatus mirroring** — writing `mstatus` (0x300) copies S-mode
  bits to `sstatus` (0x100), and vice-versa.  Only active if both CSRs exist
  in the Eumos definitions.

Custom hooks can be registered on the model:

```python
model.register_csr_write_hook(
    0x300,
    lambda state, addr, old, new: print(f"mstatus: {old:#x} → {new:#x}")
)
```

`poke_csr()` **never** triggers hooks.

---

## Read-Only Machine ID CSRs

The following read-only CSRs are defined by Eumos and available in the model:

| CSR | Address | Description |
|-----|---------|-------------|
| **mvendorid** | `0xF11` | Vendor ID — manufacturer identifier |
| **marchid** | `0xF12` | Architecture ID — base architecture identifier |
| **mimpid** | `0xF13` | Implementation ID — processor implementation version |
| **mhartid** | `0xF14` | Hart ID — hardware thread identifier |

These CSRs are **read-only** architecturally:

- `set_csr("mhartid", x)` is a no-op (returns the current value without modifying it)
- `poke_csr("mhartid", x)` bypasses read-only — useful for test setup (e.g. setting hart IDs in multi-hart scenarios)
- All four reset to 0 and are included in JSON export/restore

---

## Examples

### Basic Execution

```python
from lome import Lome

model = Lome(isa)

# ADDI x1, x0, 42
addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
changes = model.execute(addi)

print(model.get_gpr(1))                    # 42
print(changes.to_simple_dict())            # {'gpr_changes': {1: 42}}
```

### Peek / Poke for Test Setup

```python
model = Lome(isa)
model.poke_gpr(1, 0x1000)    # raw setup — no side effects
model.poke_pc(0x8000_0000)   # jump to a custom start address
model.poke_csr(0x300, 0xFF)  # force mstatus — bypasses read-only & hooks
```

### Speculation

```python
model = Lome(isa)
model.poke_gpr(1, 10)
addi = 0x13 | (2 << 7) | (0 << 12) | (1 << 15) | (5 << 20)
spec = model.speculate(addi)
print(spec.gpr_writes[0].value)  # 15  (would-be result)
print(model.get_gpr(2))          # 0   (state unchanged)
```

### JSON Round-Trip

```python
import json

model = Lome(isa)
model.poke_gpr(1, 42)
model.poke_pc(0x1000)

# Export
data = model.export_state()
json_str = model.export_state_json(indent=2)
print(json_str)

# Restore
model2 = Lome(isa)
model2.restore_state(data)
# or
model3 = Lome.from_json(json_str, isa)

assert model3.get_gpr(1) == 42
assert model3.get_pc() == 0x1000
```

### CSR Side Effects

```python
model = Lome(isa)

# Architectural write — triggers mstatus→sstatus mirroring
model.set_csr("mstatus", 0x0000_0002)   # SIE bit
print(hex(model.get_csr("sstatus")))     # 0x2 (mirrored)

# Raw write — no mirroring
model.poke_csr("mstatus", 0x0000_0002)
print(hex(model.peek_csr("sstatus")))    # unchanged
```
