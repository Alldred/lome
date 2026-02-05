# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Logical instruction implementations: AND, ANDI, OR, ORI, XOR, XORI."""

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_and(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute AND: rd = rs1 & rs2"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    result = rs1_val & rs2_val

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_andi(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ANDI: rd = rs1 & imm"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    result = rs1_val & imm

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_or(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute OR: rd = rs1 | rs2"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    result = rs1_val | rs2_val

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_ori(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute ORI: rd = rs1 | imm"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    result = rs1_val | imm

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_xor(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute XOR: rd = rs1 ^ rs2"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    result = rs1_val ^ rs2_val

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_xori(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute XORI: rd = rs1 ^ imm"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    result = rs1_val ^ imm

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
