<!--
  ~ SPDX-License-Identifier: MIT
  ~ Copyright (c) 2026 Stuart Alldred.
  -->

# Instruction Support

[Docs Home](README.md) | [API Reference](API.md)

Lome supports instructions currently defined in Eumos for this project.

## Supported Groups

- Arithmetic: `ADD`, `ADDI`, `SUB`, `ADDW`, `ADDIW`, `SUBW`
- Logical: `AND`, `ANDI`, `OR`, `ORI`, `XOR`, `XORI`
- Shift: `SLL`, `SLLI`, `SRL`, `SRLI`, `SRA`, `SRAI`, `SLLW`, `SLLIW`, `SRLW`, `SRLIW`, `SRAW`, `SRAIW`
- Compare: `SLT`, `SLTI`, `SLTU`, `SLTIU`
- Branch: `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU`
- Jump: `JAL`, `JALR`
- Load/Store: `LB`, `LH`, `LW`, `LBU`, `LHU`, `LD`, `LWU`, `SB`, `SH`, `SW`, `SD`, `FLW`, `FSW`, `FLD`, `FSD`
- Float (F/D): `FADD`, `FSUB`, `FMUL`, `FDIV`, `FSQRT`, fused multiply-add variants, sign-injection ops, min/max, comparisons, moves, conversions, class
- System: `LUI`, `AUIPC`, `CSRRW`, `CSRRS`, `CSRRC`, `CSRRWI`, `CSRRSI`, `CSRRCI`, `ECALL`, `EBREAK`, `MRET`, `FENCE`, `FENCE.TSO`

## Notes

- Final supported set is coupled to Eumos instruction definitions.
- Unsupported decoded mnemonics return `illegal_instruction` change records.
- Load/store handlers can operate without memory backend (loads return `0`).

## Adding a New Instruction

1. Implement handler in the relevant `lome/instructions/*.py` module.
2. Register mnemonic in executor dispatch map.
3. Add coverage in tests (`tests/test_instructions.py` and/or dedicated files).
4. Update docs in this page and `README.md` if user-visible support changed.
