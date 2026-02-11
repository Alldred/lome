# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Arithmetic instruction implementations: ADD, ADDI, SUB, ADDW, ADDIW, SUBW."""

from __future__ import annotations

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_add(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADD: rd = rs1 + rs2

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, rs2).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=10, x2=20 → ADD x3,x1,x2 → x3=30
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    result = (rs1_val + rs2_val) & 0xFFFFFFFFFFFFFFFF
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_addi(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADDI: rd = rs1 + imm

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, imm).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=10, imm=5 → ADDI x2,x1,5 → x2=15
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")
    rs1_val = state.get_gpr(rs1_idx)
    result = (rs1_val + imm) & 0xFFFFFFFFFFFFFFFF
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sub(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SUB: rd = rs1 - rs2

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, rs2).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=20, x2=10 → SUB x3,x1,x2 → x3=10
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    result = (rs1_val - rs2_val) & 0xFFFFFFFFFFFFFFFF
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_addw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADDW: rd = sign_extend((rs1 + rs2)[31:0])

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, rs2).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=0x0000000012345678, x2=0x0000000000000001 → ADDW x3,x1,x2 → x3=0x0000000012345679
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx) & 0xFFFFFFFF
    result_32 = (rs1_val + rs2_val) & 0xFFFFFFFF
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_addiw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADDIW: rd = sign_extend((rs1 + imm)[31:0])

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, imm).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=0x0000000012345678, imm=1 → ADDIW x2,x1,1 → x2=0x0000000012345679
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")
    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    result_32 = (rs1_val + imm) & 0xFFFFFFFF
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_subw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SUBW: rd = sign_extend((rs1 - rs2)[31:0])

    Parameters:
        operand_values: Dictionary containing instruction operands (rd, rs1, rs2).
        state: Current processor state.
        pc: Program counter value (unused for this instruction).

    Returns:
        ChangeRecord containing the GPR write operation.

    Example:
        # x1=0x0000000012345679, x2=0x0000000000000001 → SUBW x3,x1,x2 → x3=0x0000000012345678
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")
    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx) & 0xFFFFFFFF
    result_32 = (rs1_val - rs2_val) & 0xFFFFFFFF
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32
    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
