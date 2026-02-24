# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Stuart Alldred.

"""Logical instruction implementations: AND, ANDI, OR, ORI, XOR, XORI."""

from __future__ import annotations

from lome.changes import ChangeRecord
from lome.instructions.common import read_gpr, write_gpr
from lome.state import State
from lome.types import OperandValues


def execute_and(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute AND: rd = rs1 & rs2

    Perform bitwise AND of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # and x1, x2, x3  — x1 = x2 & x3
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    result = rs1_val & rs2_val

    write_gpr(changes, state, rd, result)
    return changes


def execute_andi(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute ANDI: rd = rs1 & imm

    Perform bitwise AND of rs1 and the sign-extended immediate, storing
    the result in rd.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # andi x1, x2, 0xFF  — x1 = x2 & 0xFF (mask lower byte)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = read_gpr(changes, state, rs1_idx)
    result = rs1_val & imm

    write_gpr(changes, state, rd, result)
    return changes


def execute_or(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute OR: rd = rs1 | rs2

    Perform bitwise OR of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # or x1, x2, x3  — x1 = x2 | x3
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    result = rs1_val | rs2_val

    write_gpr(changes, state, rd, result)
    return changes


def execute_ori(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute ORI: rd = rs1 | imm

    Perform bitwise OR of rs1 and the sign-extended immediate, storing
    the result in rd.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # ori x1, x2, 0x10  — x1 = x2 | 0x10 (set bit 4)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = read_gpr(changes, state, rs1_idx)
    result = rs1_val | imm

    write_gpr(changes, state, rd, result)
    return changes


def execute_xor(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute XOR: rd = rs1 ^ rs2

    Perform bitwise exclusive-OR of rs1 and rs2, storing the result in rd.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``rs2``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # xor x1, x2, x3  — x1 = x2 ^ x3
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    rs2_idx = operand_values.get("rs2")

    rs1_val = read_gpr(changes, state, rs1_idx)
    rs2_val = read_gpr(changes, state, rs2_idx)
    result = rs1_val ^ rs2_val

    write_gpr(changes, state, rd, result)
    return changes


def execute_xori(operand_values: OperandValues, state: State, pc: int) -> ChangeRecord:
    """Execute XORI: rd = rs1 ^ imm

    Perform bitwise exclusive-OR of rs1 and the sign-extended immediate,
    storing the result in rd.  ``xori rd, rs1, -1`` is equivalent to
    bitwise NOT.

    Parameters:
        operand_values: OperandValues with keys ``rd``, ``rs1``, and ``imm``.
        state: Current architectural state.
        pc: Program counter of this instruction.

    Returns:
        ChangeRecord containing the GPR write to rd.

    Example::

        # xori x1, x2, -1  — x1 = ~x2 (bitwise NOT)
    """
    changes = ChangeRecord()
    rd = operand_values.get("rd")
    rs1_idx = operand_values.get("rs1")
    imm = operand_values.get("imm")

    rs1_val = read_gpr(changes, state, rs1_idx)
    result = rs1_val ^ imm
    write_gpr(changes, state, rd, result)
    return changes
