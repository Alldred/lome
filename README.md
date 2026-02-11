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
- **State Queries**: Easy access to register values and execution results
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

## Usage

```python
from riscv_model import RISCVModel

# Create model
model = RISCVModel()

# Execute instruction (32-bit integer or bytes)
# addi x1, x0, 42
addi_instr = 0x13 | (1 << 7) | (0 << 12) | (0 << 15) | (42 << 20)
changes = model.execute(addi_instr)

# Check results
print(f"x1 = {model.get_gpr(1)}")  # 42
print(f"Changes: {model.get_changes().to_simple_dict()}")

# Speculate (execute without modifying state)
spec_changes = model.speculate(addi_instr)
print(f"Would write x1 = {spec_changes.gpr_writes[0].value}")
print(f"Current x1 = {model.get_gpr(1)}")  # Still 0

# Query branch information
branch_info = model.get_branch_info()
if branch_info:
    print(f"Branch taken: {branch_info.taken}")
    print(f"Target: 0x{branch_info.target:x}")
```

## API

**[Full API reference →](docs/API.md)** — signatures, return types, and change-tracking types.

Summary:

- **RISCVModel**: `execute()`, `speculate()`, `get_gpr()`, `get_csr()`, `get_pc()`, `set_pc()`, `reset()`, `get_changes()`, `get_branch_info()`
- **Change tracking**: GPR/CSR writes, PC updates, memory accesses, branch info, exceptions. `ChangeRecord.to_simple_dict()` / `to_detailed_dict()`; see [API docs](docs/API.md).

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

See `examples/basic_usage.py` for comprehensive usage examples.

## Future Extensions

- Memory model for load/store instructions (external model coming soon)
- Additional CSR support as needed
- Exception handling model
- Performance counters
- Support for other RISC-V extensions (M, A, etc.) as they are added to Eumos

## License

MIT License - see LICENSE file for details.
