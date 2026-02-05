# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Arithmetic instruction implementations: ADD, ADDI, SUB, ADDW, ADDIW, SUBW."""

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_add(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADD: rd = rs1 + rs2"""
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
    """Execute ADDI: rd = rs1 + imm"""
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
    """Execute SUB: rd = rs1 - rs2"""
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
    """Execute ADDW: rd = sign_extend((rs1 + rs2)[31:0])"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx) & 0xFFFFFFFF
    result_32 = (rs1_val + rs2_val) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_addiw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ADDIW: rd = sign_extend((rs1 + imm)[31:0])"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    result_32 = (rs1_val + imm) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_subw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SUBW: rd = sign_extend((rs1 - rs2)[31:0])"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx) & 0xFFFFFFFF
    result_32 = (rs1_val - rs2_val) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
