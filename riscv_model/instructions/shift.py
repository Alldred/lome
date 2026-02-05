# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred. All Rights Reserved

"""Shift instruction implementations: SLL, SLLI, SRL, SRLI, SRA, SRAI, SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW."""

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_sll(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLL: rd = rs1 << (rs2 & 0x3F)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x3F  # Lower 6 bits
    result = (rs1_val << shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_slli(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLI: rd = rs1 << (imm & 0x3F)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    shamt = imm & 0x3F  # Lower 6 bits
    result = (rs1_val << shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srl(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRL: rd = rs1 >> (rs2 & 0x3F) (logical right shift)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x3F  # Lower 6 bits
    result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srli(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLI: rd = rs1 >> (imm & 0x3F) (logical right shift)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    shamt = imm & 0x3F  # Lower 6 bits
    result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sra(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRA: rd = rs1 >> (rs2 & 0x3F) (arithmetic right shift)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx)
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x3F  # Lower 6 bits
    # Arithmetic right shift: sign extend
    if rs1_val & 0x8000000000000000:
        # Negative number, preserve sign
        mask = (1 << (64 - shamt)) - 1
        mask = ~mask
        result = ((rs1_val >> shamt) | mask) & 0xFFFFFFFFFFFFFFFF
    else:
        result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srai(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAI: rd = rs1 >> (imm & 0x3F) (arithmetic right shift)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    shamt = imm & 0x3F  # Lower 6 bits
    # Arithmetic right shift: sign extend
    if rs1_val & 0x8000000000000000:
        # Negative number, preserve sign
        mask = (1 << (64 - shamt)) - 1
        mask = ~mask
        result = ((rs1_val >> shamt) | mask) & 0xFFFFFFFFFFFFFFFF
    else:
        result = (rs1_val >> shamt) & 0xFFFFFFFFFFFFFFFF

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sllw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLW: rd = sign_extend((rs1 << (rs2 & 0x1F))[31:0])"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val << shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_slliw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SLLIW: rd = sign_extend((rs1 << (imm & 0x1F))[31:0])"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val << shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srlw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLW: rd = sign_extend((rs1 >> (rs2 & 0x1F))[31:0]) (logical)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_srliw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRLIW: rd = sign_extend((rs1 >> (imm & 0x1F))[31:0]) (logical)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sraw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAW: rd = sign_extend((rs1 >> (rs2 & 0x1F))[31:0]) (arithmetic)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    rs2_val = state.get_gpr(rs2_idx)
    shamt = rs2_val & 0x1F  # Lower 5 bits for 32-bit shift
    # Arithmetic right shift on 32-bit value
    if rs1_val & 0x80000000:
        # Negative number, preserve sign
        mask = (1 << (32 - shamt)) - 1
        mask = ~mask
        result_32 = ((rs1_val >> shamt) | mask) & 0xFFFFFFFF
    else:
        result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes


def execute_sraiw(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute SRAIW: rd = sign_extend((rs1 >> (imm & 0x1F))[31:0]) (arithmetic)"""
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx) & 0xFFFFFFFF
    shamt = imm & 0x1F  # Lower 5 bits for 32-bit shift
    # Arithmetic right shift on 32-bit value
    if rs1_val & 0x80000000:
        # Negative number, preserve sign
        mask = (1 << (32 - shamt)) - 1
        mask = ~mask
        result_32 = ((rs1_val >> shamt) | mask) & 0xFFFFFFFF
    else:
        result_32 = (rs1_val >> shamt) & 0xFFFFFFFF
    # Sign extend from 32 bits
    if result_32 & 0x80000000:
        result = result_32 | 0xFFFFFFFF00000000
    else:
        result = result_32

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
