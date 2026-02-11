<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Borrowdale

Python-based RISC-V functional model for the core instruction set (extension I).

## Features

- **Instruction Execution**: Execute all RISC-V I extension instructions currently defined in Eumos
- **Speculation**: Execute instructions without modifying state to see what would change
- **Change Tracking**: Track all state modifications (GPRs, CSRs, PC, memory accesses, branches)
- **Two-Level Access**: Architectural `get`/`set` (with side effects) and raw `peek`/`poke` (for testing & debug)
- **CSR Side Effects**: Architectural writes trigger registered hooks (e.g. mstatus → sstatus mirroring)
- **JSON Serialisation**: Export and restore complete model state via JSON
- **CSR Support**: Full CSR state management with all necessary CSRs
- **Instruction Generator Support**: Designed for use with instruction generators

## Requirements

- Python 3.13+
- Eumos: RISC-V ISA specification (installed from GitHub as a dependency)

## Installation

```bash
# Clone repository
git clone <repository-url>
cd borrowdale

# Install dependencies (uv will fetch eumos from GitHub)
uv sync --extra dev
```

## Quick Start

```python
from eumos import load_all_gprs, load_all_csrs
from eumos.decoder import Decoder
from riscv_model import RISCVModel

# Load Eumos once -- share with the model, ISG, or any other component
gprs = load_all_gprs()
csrs = load_all_csrs()
dec  = Decoder()

model = RISCVModel(decoder=dec, gpr_defs=gprs, csr_defs=csrs)

# Execute instruction (32-bit integer or bytes)
# addi x1, x0, 42
addi = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
changes = model.execute(addi)

print(f"x1 = {model.get_gpr(1)}")  # 42
print(changes.to_simple_dict())     # {'gpr_changes': {1: 42}}
```

## Access Levels: get/set vs peek/poke

The model provides two access levels for every register kind:

| Level | Methods | Side Effects | x0 / Read-Only | Use Case |
|-------|---------|-------------|-----------------|----------|
| **Architectural** | `get_*` / `set_*` | Yes — CSR hooks fire | Enforced | Normal operation, instruction execution |
| **Raw** | `peek_*` / `poke_*` | None | Bypassed | Test setup, debugging, checkpoint restore |

```python
# Architectural access
model.set_gpr(1, 42)         # x0 writes silently ignored
model.set_csr("mstatus", x)  # triggers mstatus→sstatus mirror

# Raw access
model.poke_gpr(1, 42)        # identical storage write, no hooks
model.poke_csr(0x300, x)     # bypasses read-only, no hooks
model.poke_gpr(0, 0xDEAD)    # even x0 can be written (peek sees it)
```

## JSON Serialisation

```python
# Export
data = model.export_state()            # dict
json_str = model.export_state_json()   # formatted JSON string

# Restore
model.restore_state(data)
# or
model2 = RISCVModel.from_json(json_str, decoder=dec, gpr_defs=gprs, csr_defs=csrs)
```

## Speculation

```python
spec = model.speculate(instruction)
print(spec.gpr_writes[0].value)  # what would happen
print(model.get_gpr(1))          # state unchanged
```

## API

**[Full API reference →](docs/API.md)** — signatures, return types, change-tracking types, CSR side effects, and examples.

Summary:

- **Architectural access**: `get_gpr()`, `set_gpr()`, `get_csr()`, `set_csr()`, `get_pc()`, `set_pc()`
- **Raw access**: `peek_gpr()`, `poke_gpr()`, `peek_csr()`, `poke_csr()`, `peek_pc()`, `poke_pc()`
- **Execution**: `execute()`, `speculate()`
- **State management**: `reset()`, `export_state()`, `restore_state()`, `export_state_json()`, `from_json()`
- **Change tracking**: `get_changes()`, `get_branch_info()` → `ChangeRecord.to_simple_dict()` / `.to_detailed_dict()`

## Supported Instructions

All instructions currently defined in Eumos:

- **Arithmetic**: ADD, ADDI, SUB, ADDW, ADDIW, SUBW
- **Logical**: AND, ANDI, OR, ORI, XOR, XORI
- **Shift**: SLL, SLLI, SRL, SRLI, SRA, SRAI, SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW
- **Compare**: SLT, SLTI, SLTU, SLTIU
- **Branch**: BEQ, BNE, BLT, BGE, BLTU, BGEU
- **Jump**: JAL, JALR
- **Load/Store**: LB, LH, LW, LBU, LHU, LD, LWU, SB, SH, SW, SD
- **System**: LUI, AUIPC, CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI, ECALL, EBREAK, MRET, FENCE, FENCE.TSO

## Testing

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=riscv_model --cov-report=term-missing
```

## Examples

See `examples/basic_usage.py` for comprehensive usage examples including peek/poke, JSON round-trips, and CSR side effects.

## Future Extensions

- Memory model for load/store instructions (external model coming soon)
- Additional CSR support as needed
- Exception handling model
- Performance counters
- Support for other RISC-V extensions (M, A, etc.) as they are added to Eumos

## License

MIT License - see LICENSE file for details.
