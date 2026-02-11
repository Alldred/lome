# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Logical instruction implementations: AND, ANDI, OR, ORI, XOR, XORI."""

from __future__ import annotations

from riscv_model.changes import ChangeRecord, GPRWrite
from riscv_model.state import State


def execute_and(operand_values: dict, state: State, pc: int) -> ChangeRecord:
    """Execute AND: rd = rs1 & rs2

    Perform bitwise AND of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # and x1, x2, x3  — x1 = x2 & x3
    """
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
    """Execute ANDI: rd = rs1 & imm

    Perform bitwise AND of rs1 and the sign-extended immediate, storing
    the result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # andi x1, x2, 0xFF  — x1 = x2 & 0xFF (mask lower byte)
    """
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
    """Execute OR: rd = rs1 | rs2

    Perform bitwise OR of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # or x1, x2, x3  — x1 = x2 | x3
    """
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
    """Execute ORI: rd = rs1 | imm

    Perform bitwise OR of rs1 and the sign-extended immediate, storing
    the result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # ori x1, x2, 0x10  — x1 = x2 | 0x10 (set bit 4)
    """
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
    """Execute XOR: rd = rs1 ^ rs2

    Perform bitwise exclusive-OR of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # xor x1, x2, x3  — x1 = x2 ^ x3
    """
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
    """Execute XORI: rd = rs1 ^ imm

    Perform bitwise exclusive-OR of rs1 and the sign-extended immediate,
    storing the result in rd.  ``xori rd, rs1, -1`` is equivalent to
    bitwise NOT.

    Parameters:
        operand_values: dict with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # xori x1, x2, -1  — x1 = ~x2 (bitwise NOT)
    """
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = state.get_gpr(rs1_idx)
    result = rs1_val ^ imm

    changes = ChangeRecord()
    old_value = state.set_gpr(rd, result)
    changes.gpr_writes.append(GPRWrite(register=rd, value=result, old_value=old_value))
    return changes
