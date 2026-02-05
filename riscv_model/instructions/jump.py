# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Jump instruction implementations: JAL, JALR."""

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_jal(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute JAL: rd = pc + 4; pc = pc + imm"""
    rd = operand_values.get("rd")
    imm = operand_values.get("imm")

    return_addr = pc + 4
    target = pc + imm

    changes = ChangeRecord()
    # Write return address to rd
    old_value = state.set_gpr(rd, return_addr)
    changes.gpr_writes.append(GPRWrite(register=rd, value=return_addr, old_value=old_value))
    # Update PC
    changes.pc_change = (target, pc)
    return changes


def execute_jalr(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute JALR: rd = pc + 4; pc = (rs1 + imm) & ~1"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    return_addr = pc + 4
    target = (rs1_val + imm) & ~1  # Clear LSB for alignment

    changes = ChangeRecord()
    # Write return address to rd
    old_value = state.set_gpr(rd, return_addr)
    changes.gpr_writes.append(GPRWrite(register=rd, value=return_addr, old_value=old_value))
    # Update PC
    changes.pc_change = (target, pc)
    return changes
